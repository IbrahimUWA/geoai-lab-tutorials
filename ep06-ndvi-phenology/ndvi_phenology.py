"""
GeoAI Lab - Episode 6
NDVI time series and crop phenology from Sentinel-2, using only open data and free software.

  1. STAC search: every Sentinel-2 L2A scene over our window for one year (AWS Earth Search, no account)
  2. Read red, near-infrared and the scene classification for the window only, straight from the
     cloud-optimised GeoTIFFs, several scenes at a time in threads
  3. Mask clouds and shadows, compute NDVI, stack everything into a (dates, rows, cols) cube
  4. Fill the gaps by linear interpolation in time, resample to a regular 5-day grid and smooth with a
     Savitzky-Golay filter plus one upper-envelope pass (cloud residuals only ever pull NDVI down)
  5. Phenology per pixel: start of season, peak date, end of season, length, amplitude, seasonal integral
  6. Compare with the USDA Cropland Data Layer: curves per crop, peak-date histograms, and a one-number
     corn / soybean rule
  7. Export the metrics as a multi-band GeoTIFF and the per-class curves as CSV

Usage:
  python ndvi_phenology.py output_folder [--bbox lon_min lat_min lon_max lat_max] [--year 2021] [--step 5]

Dependencies: pystac-client planetary-computer rasterio shapely numpy scipy matplotlib
"""
import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
# fewer round trips when reading remote cloud-optimised GeoTIFFs
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from scipy import ndimage
from scipy.signal import savgol_filter
from shapely.geometry import box, shape
from pystac_client import Client
import planetary_computer as pc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

BG = "#0b1220"
FG = "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#334155", "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "font.family": "Segoe UI", "font.size": 14, "axes.titlesize": 20, "axes.titleweight": "bold",
})
FIGSIZE = (19.2, 10.8)

STAC = "https://earth-search.aws.element84.com/v1"
CLOUD_SCL = [0, 1, 3, 8, 9, 10, 11]   # SCL: no data, saturated, cloud shadow, cloud medium / high, cirrus, snow

# CDL codes -> classes (same grouping as episode 4)
CLASS_MAP = {1: "Corn", 5: "Soybeans", 36: "Alfalfa / hay", 37: "Alfalfa / hay", 176: "Grass / pasture",
             141: "Forest", 142: "Forest", 143: "Forest", 111: "Water",
             121: "Developed", 122: "Developed", 123: "Developed", 124: "Developed", 190: "Wetlands", 195: "Wetlands"}
CLASSES = ["Corn", "Soybeans", "Alfalfa / hay", "Grass / pasture", "Forest", "Water", "Developed", "Wetlands"]
COLORS = ["#fbbf24", "#22c55e", "#a3e635", "#d9f99d", "#166534", "#3b82f6", "#ef4444", "#67e8f9"]
METRICS = ["sos", "peak_doy", "eos", "los", "peak_ndvi", "amplitude", "integral", "n_obs"]


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=100)
    plt.close(fig)
    print("  figure ->", name)


# =====================================================================
# 1. EVERY SCENE OF THE YEAR
# =====================================================================
def find_scenes(bbox, year):
    """All Sentinel-2 L2A scenes over the window in one year, one per date, sorted by time."""
    cat = Client.open(STAC)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31",
                            query={"eo:cloud_cover": {"lt": 85}}, max_items=1000).items())
    window = box(*bbox)
    best = {}                                      # date -> (coverage of our window, item)
    for it in items:
        cover = shape(it.geometry).intersection(window).area / window.area
        day = it.datetime.date()
        if cover > 0.98 and (day not in best or cover > best[day][0]):
            best[day] = (cover, it)
    scenes = [best[d][1] for d in sorted(best)]
    print(f"  {len(items)} scenes found, {len(scenes)} dates cover the whole window")
    return scenes


def read_window(href, bbox, shape=None, transform=None, resampling=Resampling.bilinear):
    """Read only the bbox from a remote COG. Optionally resample onto a given grid."""
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*b, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win).astype(np.float32)
        tr = src.window_transform(win)
        crs = src.crs
    if shape is not None and (arr.shape != shape or tr != transform):
        dst = np.zeros(shape, np.float32)
        reproject(arr, dst, src_transform=tr, src_crs=crs, dst_transform=transform, dst_crs=crs, resampling=resampling)
        arr, tr = dst, transform
    return arr, tr, crs


def reflectance(item, band, bbox, grid):
    """Surface reflectance 0..1. Scenes reprocessed with baseline 04.00 or later advertise an offset of -0.1
    in their metadata, but the Earth Search COGs are already harmonised (the offset is removed), so applying
    it would push dark pixels negative and NDVI above 1. check_offset() prints the evidence."""
    arr, tr, crs = read_window(item.assets[band].href, bbox, *grid)
    return arr * 1e-4, tr, crs


def check_offset(scenes, bbox):
    """Median red DN over vegetation for the newest and the oldest processing baseline in the list.
    If the new one were offset-encoded it would be about 1000 higher."""
    by_base = {it.properties.get("s2:processing_baseline", "?"): it for it in scenes}
    for base in (max(by_base), min(by_base)):
        it = by_base[base]
        red, tr, crs = read_window(it.assets["red"].href, bbox)
        scl, _, _ = read_window(it.assets["scl"].href, bbox, red.shape, tr, resampling=Resampling.nearest)
        meta = it.assets["red"].extra_fields.get("raster:bands", [{}])[0].get("offset", 0)
        veg = scl == 4
        med = np.median(red[veg]) if veg.any() else float("nan")
        print(f"  baseline {base}: metadata offset {meta}, median red DN over vegetation {med:.0f}  ({it.id})")


def read_ndvi(item, bbox, grid):
    """One date: NDVI for the window with clouds, shadows and snow set to NaN. Returns (ndvi, clear fraction)."""
    red, tr, crs = reflectance(item, "red", bbox, grid)
    nir, _, _ = reflectance(item, "nir", bbox, grid)
    scl, _, _ = read_window(item.assets["scl"].href, bbox, *grid, resampling=Resampling.nearest)
    ndvi = (nir - red) / (nir + red + 1e-6)
    bad = np.isin(scl, CLOUD_SCL) | (nir + red <= 0.01)
    bad |= ndimage.binary_dilation(bad, iterations=3)          # a 30 m buffer around every cloud edge
    ndvi[bad] = np.nan
    return ndvi.astype(np.float32), float(1 - bad.mean())


def build_cube(scenes, bbox, workers=8):
    """(dates, rows, cols) NDVI cube with NaN gaps, the day-of-year of every date, clear fraction per date."""
    first, tr, crs = read_window(scenes[0].assets["red"].href, bbox)
    grid = (first.shape, tr)
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda it: read_ndvi(it, bbox, grid), scenes))
    cube = np.stack([r[0] for r in results])
    clear = np.array([r[1] for r in results])
    doy = np.array([it.datetime.timetuple().tm_yday for it in scenes], dtype=np.float32)
    print(f"  cube {cube.shape}, {cube.nbytes / 1e6:.0f} MB, mean clear fraction {clear.mean() * 100:.0f}%")
    return cube, doy, clear, tr, crs


# =====================================================================
# 2. LABELS FROM THE CROPLAND DATA LAYER
# =====================================================================
def read_cdl(bbox, year, shape, transform, crs):
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = [i for i in cat.search(collections=["usda-cdl"], bbox=bbox, max_items=50).items()
             if i.id.startswith(f"cropland_{year}")]
    if not items:
        raise SystemExit(f"no CDL item for {year}")
    href = items[0].assets["cropland"].href
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*b, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        tr = src.window_transform(win)
        cdl = np.zeros(shape, np.uint8)
        reproject(arr, cdl, src_transform=tr, src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    labels = np.full(shape, -1, np.int16)
    for code, name in CLASS_MAP.items():
        labels[cdl == code] = CLASSES.index(name)
    print(f"  CDL {year} read: {(labels >= 0).mean() * 100:.0f}% of pixels fall in our {len(CLASSES)} classes")
    return labels


# =====================================================================
# 3. GAP FILLING AND SMOOTHING
# =====================================================================
def despike(Y, drop=0.15):
    """Drop observations that sit far below both of their temporal neighbours: a cloud the mask missed.
    Clouds and shadows only ever lower NDVI, so a one-date dip is never real."""
    T = Y.shape[0]
    valid = ~np.isnan(Y)
    idx = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(valid, idx, -1), axis=0)            # last valid index at or before t
    nxt = np.minimum.accumulate(np.where(valid, idx, T)[::-1], axis=0)[::-1]  # first valid index at or after t
    before, after = np.full_like(prev, -1), np.full_like(nxt, T)
    before[1:], after[:-1] = prev[:-1], nxt[1:]                             # strictly before / strictly after
    has = (before >= 0) & (after < T)
    yb = np.take_along_axis(Y, np.clip(before, 0, T - 1), 0)
    ya = np.take_along_axis(Y, np.clip(after, 0, T - 1), 0)
    spike = valid & has & (Y < np.minimum(yb, ya) - drop)
    out = Y.copy()
    out[spike] = np.nan
    return out


def interp_time(t_obs, Y, t_grid):
    """Linear interpolation in time for every pixel at once.
    Y is (dates, pixels) with NaN gaps; returns (grid dates, pixels). Edges are held at the nearest value."""
    T = len(t_obs)
    valid = ~np.isnan(Y)
    idx = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(valid, idx, -1), axis=0)            # last valid index at or before t
    nxt = np.minimum.accumulate(np.where(valid, idx, T)[::-1], axis=0)[::-1]  # first valid index at or after t
    k = np.clip(np.searchsorted(t_obs, t_grid, side="right") - 1, 0, T - 1)   # observation slot of each grid date
    p, n = prev[k], nxt[np.clip(k + 1, 0, T - 1)]
    has_p, has_n = p >= 0, n < T
    p = np.where(has_p, p, np.where(has_n, n, 0))
    n = np.where(has_n, n, p)
    yp, yn = np.take_along_axis(Y, p, 0), np.take_along_axis(Y, n, 0)
    tp, tn = t_obs[p], t_obs[n]
    w = np.clip(np.where(tn > tp, (t_grid[:, None] - tp) / (tn - tp + 1e-9), 0.0), 0, 1)
    out = yp + w * (yn - yp)
    out[~(has_p | has_n)] = np.nan
    return out


def smooth(Y, window=7, order=2, passes=3):
    """Savitzky-Golay along time, then upper-envelope passes: where the data sit below the curve
    (left-over cloud, haze) lift them to the curve and smooth again. Clouds only ever lower NDVI."""
    s = savgol_filter(Y, window, order, axis=0)
    for _ in range(passes):
        s = savgol_filter(np.maximum(Y, s), window, order, axis=0)
    return s


# =====================================================================
# 4. PHENOLOGY METRICS
# =====================================================================
def phenology(S, t_grid, frac=0.5, min_amp=0.2):
    """S is (grid dates, pixels) smoothed NDVI. Threshold method: the season starts when NDVI first rises above
    half way between the spring minimum and the peak, and ends when it drops below half way between the peak
    and the autumn minimum. Returns a dict of (pixels,) arrays, NaN where there is no clear season."""
    G, N = S.shape
    gi = np.arange(G)[:, None]
    ipk = np.nanargmax(S, axis=0)
    peak = S[ipk, np.arange(N)]
    before, after = gi <= ipk, gi >= ipk
    base_l = np.min(np.where(before, S, np.inf), axis=0)
    base_r = np.min(np.where(after, S, np.inf), axis=0)
    thr_l = base_l + frac * (peak - base_l)
    thr_r = base_r + frac * (peak - base_r)
    # SOS: last grid step before the peak that is still below the left threshold, plus one
    below_l = before & (S < thr_l)
    isos = G - 1 - np.argmax(below_l[::-1], axis=0) + 1
    # EOS: first grid step after the peak below the right threshold
    below_r = after & (S < thr_r)
    ieos = np.argmax(below_r, axis=0)
    ok = below_l.any(0) & below_r.any(0) & ((peak - np.maximum(base_l, base_r)) >= min_amp)
    isos, ieos = np.clip(isos, 0, G - 1), np.clip(ieos, 0, G - 1)
    sos, eos = t_grid[isos], t_grid[ieos]
    in_season = (gi >= isos) & (gi <= ieos)
    step = float(t_grid[1] - t_grid[0])
    integral = np.sum(np.where(in_season, S - np.minimum(base_l, base_r), 0), axis=0) * step
    out = {"sos": sos, "peak_doy": t_grid[ipk], "eos": eos, "los": eos - sos, "peak_ndvi": peak,
           "amplitude": peak - np.maximum(base_l, base_r), "integral": integral}
    for k in out:
        out[k] = np.where(ok, out[k], np.nan).astype(np.float32)
    return out


def process(cube, doy, t_grid, rows_per_chunk=120):
    """Run interpolation, smoothing and phenology over the whole image in row chunks.
    Returns the smoothed cube as float16 (grid dates, rows, cols) and a dict of metric rasters."""
    T, H, W = cube.shape
    G = len(t_grid)
    S_all = np.empty((G, H, W), np.float16)
    metrics = {k: np.full((H, W), np.nan, np.float32) for k in METRICS}
    metrics["n_obs"] = (~np.isnan(cube)).sum(0).astype(np.float32)
    for r0 in range(0, H, rows_per_chunk):
        r1 = min(H, r0 + rows_per_chunk)
        Y = despike(cube[:, r0:r1].reshape(T, -1))
        Z = interp_time(doy, Y, t_grid)
        nodata = np.isnan(Z).any(0)
        Z[:, nodata] = 0
        S = smooth(Z)
        S[:, nodata] = np.nan
        S_all[:, r0:r1] = S.reshape(G, r1 - r0, W).astype(np.float16)
        m = phenology(np.where(np.isnan(S), 0, S), t_grid)
        for k, v in m.items():
            v[nodata] = np.nan
            metrics[k][r0:r1] = v.reshape(r1 - r0, W)
        print(f"  rows {r0:5d}-{r1:5d} done", end="\r")
    print()
    return S_all, metrics


# =====================================================================
# FIGURES
# =====================================================================
DATE_FMT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def doy_label(d):
    import datetime
    dt = datetime.date(2021, 1, 1) + datetime.timedelta(days=int(d) - 1)
    return f"{dt.day} {DATE_FMT[dt.month - 1]}"


def fig_observations(cube, scenes, clear, out):
    """Timeline of every scene and twelve monthly thumbnails, cloud shown in grey."""
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(3, 6, figure=fig, height_ratios=[0.9, 1, 1], hspace=0.25, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    doy = [it.datetime.timetuple().tm_yday for it in scenes]
    ax.bar(doy, clear * 100, width=3.5, color=np.where(clear > 0.8, "#38bdf8", "#64748b"))
    ax.set_xlim(0, 366)
    ax.set_ylim(0, 100)
    ax.set_ylabel("clear pixels, %")
    ax.set_xticks(np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) + 15)
    ax.set_xticklabels(DATE_FMT)
    ax.set_title(f"{len(scenes)} Sentinel-2 dates in the year; blue = more than 80% of our window is clear")
    months = {}
    for i, it in enumerate(scenes):
        m = it.datetime.month
        if m not in months or clear[i] > clear[months[m]]:
            months[m] = i
    for j, m in enumerate(range(1, 13)):
        ax = fig.add_subplot(gs[1 + j // 6, j % 6])
        ax.set_xticks([])
        ax.set_yticks([])
        if m not in months:
            ax.set_title(f"{DATE_FMT[m - 1]}: no scene", fontsize=13)
            continue
        i = months[m]
        img = cube[i][::4, ::4]
        ax.imshow(np.full(img.shape, 0.5), cmap="Greys", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_invalid(img), cmap="RdYlGn", vmin=0.0, vmax=0.9)
        ax.set_title(f"{doy_label(doy[i])}: {clear[i] * 100:.0f}% clear", fontsize=13)
    fig.suptitle("A year of Sentinel-2 over Ames, Iowa: the clearest date of each month", fontsize=24,
                 fontweight="bold", y=0.98)
    save(fig, out, "fig01_year.png")


def field_centre(labels, cls):
    """Pixel in the middle of the largest field of a class (max distance to anything else)."""
    dist = ndimage.distance_transform_edt(np.pad(labels == cls, 1))[1:-1, 1:-1]   # pad: edges count as "else"
    return np.unravel_index(np.argmax(dist), dist.shape)


def fig_pixel(cube, doy, S_all, t_grid, metrics, labels, out):
    """Also returns the dates of the two example pixels (for the narration)."""
    px = {}
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, cls in zip(axes, ("Corn", "Soybeans")):
        c = CLASSES.index(cls)
        r, col = field_centre(labels, c)
        raw = cube[:, r, col]
        sm = S_all[:, r, col].astype(np.float32)
        Z = interp_time(doy, raw[:, None], t_grid)[:, 0]
        ok = ~np.isnan(raw)
        ax.plot(t_grid, Z, color="#64748b", lw=1.5, label="linear interpolation")
        ax.plot(t_grid, sm, color=COLORS[c], lw=4, label="Savitzky-Golay + envelope")
        ax.scatter(doy[ok], raw[ok], s=70, color="white", zorder=5, label=f"clear observations ({ok.sum()})")
        ax.scatter(doy[~ok], np.full((~ok).sum(), -0.05), s=70, marker="x", color="#ef4444", label=f"cloudy ({(~ok).sum()})")
        for k, lab, mk in (("sos", "start", "^"), ("peak_doy", "peak", "*"), ("eos", "end", "v")):
            d = metrics[k][r, col]
            px[f"px_{cls[:4].lower()}_{k.split('_')[0]}"] = doy_label(d)
            y = np.interp(d, t_grid, sm)
            ax.scatter([d], [y], s=320, marker=mk, color="#38bdf8", zorder=6, edgecolor="black")
            ax.annotate(f"{lab}: {doy_label(d)}", (d, y), textcoords="offset points", xytext=(12, -4 if k != "peak_doy" else 14),
                        fontsize=14, color="#38bdf8")
        ax.set_xlim(0, 366)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xticks(np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) + 15)
        ax.set_xticklabels(DATE_FMT)
        ax.grid(alpha=0.15)
        ax.set_title(f"One {cls.lower()} pixel  (row {r}, col {col})")
        ax.legend(loc="upper left", fontsize=12)
    axes[0].set_ylabel("NDVI")
    fig.suptitle("From scattered observations to a smooth seasonal curve", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig02_pixel_curves.png")
    return px


def class_curves(S_all, labels, t_grid, per_class=4000, seed=0):
    """Median and quartiles of the smoothed curve per class, from a random sample of pixels."""
    rng = np.random.default_rng(seed)
    flat = labels.reshape(-1)
    Sf = S_all.reshape(len(t_grid), -1)
    curves = {}
    for c, name in enumerate(CLASSES):
        idx = np.flatnonzero(flat == c)
        if len(idx) < 50:
            continue
        idx = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        s = Sf[:, idx].astype(np.float32)
        curves[name] = np.nanpercentile(s, [25, 50, 75], axis=1)
    return curves


def fig_curves(curves, t_grid, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name in ("Corn", "Soybeans", "Alfalfa / hay", "Grass / pasture", "Forest", "Developed"):
        if name not in curves:
            continue
        q25, q50, q75 = curves[name]
        col = COLORS[CLASSES.index(name)]
        ax.fill_between(t_grid, q25, q75, color=col, alpha=0.15)
        ax.plot(t_grid, q50, color=col, lw=4, label=name)
        i = int(np.argmax(q50))
        ax.scatter([t_grid[i]], [q50[i]], s=140, color=col, edgecolor="black", zorder=6)
    ax.set_xlim(0, 366)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) + 15)
    ax.set_xticklabels(DATE_FMT)
    ax.set_ylabel("NDVI (median, with the 25-75% band)")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=15)
    ax.set_title("Every land cover has its own seasonal signature (4,000 random pixels per class, labelled by the Cropland Data Layer)")
    fig.tight_layout()
    save(fig, out, "fig03_class_curves.png")


def fig_maps(metrics, out):
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1], wspace=0.25, hspace=0.12,
                  left=0.13, right=0.87, top=0.90, bottom=0.03)
    panels = [("sos", "Start of season", "viridis", (110, 190)),
              ("peak_doy", "Date of peak NDVI", "plasma", (170, 250)),
              ("eos", "End of season", "viridis", (230, 300)),
              ("los", "Length of season, days", "cividis", (60, 160))]
    for j, (k, title, cmap, (lo, hi)) in enumerate(panels):
        ax = fig.add_subplot(gs[j // 2, 2 * (j % 2):2 * (j % 2) + 2])
        im = ax.imshow(np.ma.masked_invalid(metrics[k]), cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if k != "los":
            ticks = cb.get_ticks()
            cb.set_ticks(ticks)
            cb.set_ticklabels([doy_label(t) for t in ticks])
        cb.ax.tick_params(labelsize=12)
        ax.set_title(title, fontsize=17)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Phenology at 10 m for every pixel (dark = no clear growing season: water, roads, buildings)",
                 fontsize=24, fontweight="bold", y=0.975)
    save(fig, out, "fig04_phenology_maps.png")


def corn_soy_rule(metrics, labels):
    """One threshold on the peak date separates corn from soybeans. Returns threshold, accuracy, rule map."""
    pk = metrics["peak_doy"]
    corn, soy = labels == CLASSES.index("Corn"), labels == CLASSES.index("Soybeans")
    known = (corn | soy) & ~np.isnan(pk)
    best = (0, None)
    for thr in range(180, 250):
        pred_corn = pk < thr
        acc = (pred_corn[known] == corn[known]).mean()
        if acc > best[0]:
            best = (acc, thr)
    acc, thr = best
    rule = np.where(np.isnan(pk), -1, np.where(pk < thr, 0, 1)).astype(np.int16)
    return thr, acc, rule, known


def fig_corn_soy(metrics, labels, thr, acc, rule, out):
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.95, 1, 1], wspace=0.06,
                  left=0.05, right=0.99, top=0.86, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    pk = metrics["peak_doy"]
    bins = np.arange(160, 262, 5)
    for name in ("Corn", "Soybeans"):
        c = CLASSES.index(name)
        v = pk[(labels == c) & ~np.isnan(pk)]
        ax.hist(v, bins=bins, color=COLORS[c], alpha=0.75, label=f"{name} ({len(v):,} px)")
    ax.axvline(thr, color="white", lw=3, ls="--")
    ax.text(thr + 2, ax.get_ylim()[1] * 0.92, f"split at {doy_label(thr)}", fontsize=15, color="white")
    ax.set_xlabel("date of peak NDVI (day of year)")
    ax.set_ylabel("pixels")
    ax.legend(fontsize=14)
    ax.set_title("Peak date by crop (CDL labels)")
    cmap = matplotlib.colors.ListedColormap([COLORS[0], COLORS[1]])
    ax = fig.add_subplot(gs[1])
    shown = np.where((labels == 0) | (labels == 1), rule, -1).astype(np.float32)
    shown[shown < 0] = np.nan
    ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Our rule: peak before the split = corn")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(gs[2])
    ref = np.where((labels == 0) | (labels == 1), labels, -1).astype(np.float32)
    ref[ref < 0] = np.nan
    ax.imshow(np.ma.masked_invalid(ref), cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Cropland Data Layer")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(f"One number per pixel separates corn from soybeans: {acc * 100:.1f}% agreement with the reference",
                 fontsize=24, fontweight="bold", y=0.97)
    save(fig, out, "fig05_corn_soy.png")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-93.75, 41.95, -93.55, 42.10])
    ap.add_argument("--year", type=int, default=2021, help="a year the Cropland Data Layer is available for")
    ap.add_argument("--step", type=int, default=5, help="days between grid dates after interpolation")
    ap.add_argument("--workers", type=int, default=8, help="parallel scene reads")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    print("\n[1] Scenes")
    scenes = find_scenes(a.bbox, a.year)

    print("\n[2] NDVI cube")
    check_offset(scenes, a.bbox)
    t1 = time.time()
    cube, doy, clear, transform, crs = build_cube(scenes, a.bbox, a.workers)
    read_s = time.time() - t1
    print(f"  {len(scenes)} dates read in {read_s / 60:.1f} min")
    T, H, W = cube.shape
    fig_observations(cube, scenes, clear, a.out)

    print("\n[3] Labels")
    labels = read_cdl(a.bbox, a.year, (H, W), transform, crs)

    print("\n[4] Interpolate, smooth, phenology")
    t_grid = np.arange(1, 366, a.step, dtype=np.float32)
    t1 = time.time()
    S_all, metrics = process(cube, doy, t_grid)
    print(f"  {H * W:,} pixels x {len(t_grid)} grid dates in {time.time() - t1:.0f} s")
    px = fig_pixel(cube, doy, S_all, t_grid, metrics, labels, a.out)

    print("\n[5] Curves per class and maps")
    curves = class_curves(S_all, labels, t_grid)
    fig_curves(curves, t_grid, a.out)
    fig_maps(metrics, a.out)
    with open(os.path.join(a.out, "class_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doy"] + [f"{n} {q}" for n in curves for q in ("q25", "median", "q75")])
        for i, d in enumerate(t_grid):
            w.writerow([int(d)] + [f"{curves[n][q][i]:.3f}" for n in curves for q in range(3)])
    print("  table -> class_curves.csv")

    print("\n[6] Corn versus soybeans with one number")
    thr, acc, rule, known = corn_soy_rule(metrics, labels)
    print(f"  best split: peak before day {thr} ({doy_label(thr)}) = corn; agreement {acc * 100:.1f}% on {known.sum():,} pixels")
    fig_corn_soy(metrics, labels, thr, acc, rule, a.out)

    print("\n[7] Export")
    path = os.path.join(a.out, "phenology.tif")
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=len(METRICS), dtype="float32",
                       nodata=np.nan, crs=crs, transform=transform, compress="deflate") as dst:
        for i, k in enumerate(METRICS):
            dst.write(metrics[k], i + 1)
            dst.set_band_description(i + 1, k)
    print("  raster -> phenology.tif  (bands: " + ", ".join(METRICS) + ")")

    def med(name, k):
        v = metrics[k][labels == CLASSES.index(name)]
        return float(np.nanmedian(v))

    season = ~np.isnan(metrics["sos"])
    stats = {
        "scenes": len(scenes), "clear_dates": int((clear > 0.8).sum()), "grid": f"{H}x{W}",
        "pixels": H * W, "cube_mb": int(cube.nbytes / 1e6), "read_min": f"{read_s / 60:.1f}",
        "mean_obs": f"{np.nanmean(metrics['n_obs']):.0f}", "grid_dates": len(t_grid),
        "season_pct": f"{season.mean() * 100:.0f}",
        "corn_sos": doy_label(med("Corn", "sos")), "corn_peak": doy_label(med("Corn", "peak_doy")),
        "corn_eos": doy_label(med("Corn", "eos")), "corn_los": f"{med('Corn', 'los'):.0f}",
        "soy_sos": doy_label(med("Soybeans", "sos")), "soy_peak": doy_label(med("Soybeans", "peak_doy")),
        "soy_eos": doy_label(med("Soybeans", "eos")), "soy_los": f"{med('Soybeans', 'los'):.0f}",
        "corn_peak_ndvi": f"{med('Corn', 'peak_ndvi'):.2f}", "soy_peak_ndvi": f"{med('Soybeans', 'peak_ndvi'):.2f}",
        "split_doy": thr, "split_date": doy_label(thr), "rule_acc": f"{acc * 100:.1f}", "rule_pixels": int(known.sum()),
        "minutes": f"{(time.time() - t0) / 60:.1f}",
    }
    # narration-friendly extras
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    lag = (med("Soybeans", "peak_doy") - med("Corn", "peak_doy")) / 7
    stats["soy_lag"] = words[min(8, int(round(lag)))]
    stats["read_min_text"] = words[min(8, int(round(read_s / 60)))]
    stats.update(px)
    stats["rule_pixels_text"] = f"{known.sum() / 1e6:.1f} million" if known.sum() >= 1e6 else f"{known.sum() // 1000:,} thousand"
    with open(os.path.join(a.out, "stats.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}={v}\n")
    print(f"\nDone in {stats['minutes']} min")
    for k, v in stats.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
