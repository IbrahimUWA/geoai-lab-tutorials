"""GeoAI Lab - Metashape lecture, step 2: what you can measure from the outputs.

DSM vs DTM, canopy height model (CHM = DSM - DTM), individual tree tops, an RGB vegetation index (VARI),
slope and contours from the DTM, a cross-section, and a side-by-side check against the OpenDroneMap run of the
same 77 photos (episode 1).

    python 02_analysis.py

Writes results/analysis_stats.json and figures results/fig_*.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import laspy
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, ListedColormap
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from skimage.feature import peak_local_max

ROOT = os.environ.get("GEOAI_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # folder that holds data/
MS = os.path.join(ROOT, "data", "drone", "aukerman_metashape")
ODM = os.path.join(ROOT, "data", "drone", "aukerman")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

BG, PANEL, FG, MUTED = "#0b1220", "#162034", "#f1f5f9", "#94a3b8"
ACCENT, GREEN, AMBER, PINK, PURPLE, RED = "#38bdf8", "#4ade80", "#fbbf24", "#f472b6", "#c084fc", "#f87171"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#334155",
                     "font.size": 15, "axes.titlesize": 20, "axes.titleweight": "bold", "font.family": "Segoe UI"})
GRID = 0.10      # common analysis grid, metres


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=100, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def load(path, like=None, res=GRID, band=1, resampling=Resampling.bilinear):
    """Read a raster band onto a common grid (the Metashape DSM extent at `res` metres)."""
    with rasterio.open(path) as src:
        if like is None:
            b = src.bounds
            w, h = int((b.right - b.left) / res), int((b.top - b.bottom) / res)
            tr = rasterio.transform.from_origin(b.left, b.top, res, res)
            like = (tr, w, h, src.crs)
        tr, w, h, crs = like
        dst = np.full((h, w), np.nan, np.float32)
        reproject(rasterio.band(src, band), dst, dst_transform=tr, dst_crs=crs, dst_nodata=np.nan,
                  resampling=resampling, src_nodata=src.nodata)
    return dst, like


def ext(like):
    tr, w, h, _ = like
    return (tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f)


def hill(z):
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(np.nan_to_num(z, nan=np.nanmedian(z)), vert_exag=2, dx=GRID, dy=GRID)


if __name__ == "__main__":
    st = {}
    dsm, like = load(os.path.join(MS, "dsm.tif"))
    dtm, _ = load(os.path.join(MS, "dtm.tif"), like)
    rgb = np.stack([load(os.path.join(MS, "orthomosaic.tif"), like, band=b)[0] for b in (1, 2, 3)], -1)
    alpha, _ = load(os.path.join(MS, "orthomosaic.tif"), like, band=4, resampling=Resampling.nearest)
    valid = np.isfinite(dsm) & np.isfinite(dtm) & np.isfinite(rgb).all(-1) & (np.nan_to_num(alpha) > 0)
    # near the edge of the photo coverage the surfaces are extrapolated: keep measurements 10 m inside it
    core = valid & (ndimage.distance_transform_edt(valid) * GRID > 10)
    E = ext(like)
    rgb8 = np.clip(np.nan_to_num(rgb) / 255, 0, 1)
    rgb8[~valid] = 0.04

    # ---- DSM vs DTM
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    lo, hi = np.nanpercentile(dtm[valid], 1), np.nanpercentile(dsm[valid], 99.5)
    for a, z, t in ((axs[0], dsm, "DSM: top of everything"), (axs[1], dtm, "DTM: bare ground")):
        a.imshow(np.where(valid, hill(z), np.nan), cmap="gray", extent=E)
        im = a.imshow(np.where(valid, z, np.nan), cmap="terrain", vmin=lo, vmax=hi, alpha=0.55, extent=E)
        a.set_title(t); a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=axs, shrink=0.8); cb.set_label("elevation (m)")
    save(fig, "fig_dsm_dtm.png")
    st["elevation_range_m"] = [round(float(lo), 1), round(float(hi), 1)]

    # ---- CHM and trees
    chm = np.where(valid, np.clip(dsm - dtm, 0, None), np.nan)
    chm_s = ndimage.gaussian_filter(np.nan_to_num(chm), 3)            # smooth 0.3 m so one crown gives one top
    tops = peak_local_max(np.where(core, chm_s, 0), min_distance=int(2.5 / GRID), threshold_abs=4.0, exclude_border=False)
    tr = like[0]
    heights = chm_s[tops[:, 0], tops[:, 1]]
    st["trees_detected"] = int(len(tops))
    st["tree_height_m"] = {"median": round(float(np.median(heights)), 1), "p90": round(float(np.percentile(heights, 90)), 1),
                           "max": round(float(heights.max()), 1)}
    area_m2 = float(valid.sum()) * GRID ** 2
    st["core_area_ha"] = round(float(core.sum()) * GRID ** 2 / 1e4, 2)
    st["area_ha"] = round(area_m2 / 1e4, 2)
    st["cover_over_2m_pct"] = round(100 * float((chm[core] > 2).mean()), 1)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    im = axs[0].imshow(chm, cmap="viridis", vmin=0, vmax=np.nanpercentile(chm, 99.5), extent=E)
    axs[0].set_title("Canopy height model: DSM minus DTM"); fig.colorbar(im, ax=axs[0], shrink=0.8).set_label("height above ground (m)")
    axs[1].imshow(rgb8, extent=E)
    xs = tr.c + (tops[:, 1] + 0.5) * tr.a
    ys = tr.f + (tops[:, 0] + 0.5) * tr.e
    axs[1].scatter(xs, ys, s=22, facecolor="none", edgecolor=AMBER, lw=1.6)
    axs[1].set_title(f"{len(tops)} tree tops found (over 4 m)")
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    save(fig, "fig_chm_trees.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(heights, bins=np.arange(4, heights.max() + 1, 1), color=GREEN, edgecolor=BG)
    ax.set(xlabel="tree height (m)", ylabel="trees", title="How tall are the trees?")
    ax.grid(alpha=0.15)
    save(fig, "fig_tree_heights.png")

    # ---- VARI vegetation index from plain RGB
    r_, g_, b_ = [rgb[..., i].astype(float) for i in range(3)]
    vari = np.where(valid, (g_ - r_) / (g_ + r_ - b_ + 1e-6), np.nan)
    vari = np.clip(vari, -0.5, 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].imshow(rgb8, extent=E); axs[0].set_title("Orthomosaic")
    im = axs[1].imshow(vari, cmap="RdYlGn", vmin=-0.3, vmax=0.4, extent=E); axs[1].set_title("VARI: greenness from an RGB camera")
    fig.colorbar(im, ax=axs[1], shrink=0.8)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    save(fig, "fig_vari.png")
    st["green_pct"] = round(100 * float((vari[core] > 0.05).mean()), 1)

    # ---- slope + contours from the DTM
    gy, gx = np.gradient(ndimage.gaussian_filter(np.nan_to_num(dtm, nan=np.nanmedian(dtm)), 10), GRID)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~valid] = np.nan
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(np.where(valid, hill(dtm), np.nan), cmap="gray", extent=E)
    im = ax.imshow(slope, cmap="magma_r", vmin=0, vmax=25, alpha=0.6, extent=E)
    zc = ndimage.gaussian_filter(np.nan_to_num(dtm, nan=np.nanmedian(dtm)), 15)
    zc[~valid] = np.nan
    X = np.linspace(E[0], E[1], zc.shape[1]); Y = np.linspace(E[3], E[2], zc.shape[0])
    lv = np.arange(np.floor(np.nanmin(zc)), np.ceil(np.nanmax(zc)) + 1, 1.0)
    cs = ax.contour(X, Y, zc, levels=lv, colors="white", linewidths=0.8)
    ax.clabel(cs, cs.levels[::2], fmt="%d m", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("slope (degrees)")
    ax.set_title("Slope and 1 m contours from the DTM"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig_slope_contours.png")
    st["slope_median_deg"] = round(float(np.nanmedian(slope[core])), 1)

    # ---- cross-section through the tallest tree, west to east
    i0 = tops[np.argmax(heights), 0]
    row = slice(None)
    xline = np.linspace(E[0], E[1], dsm.shape[1])
    fig, ax = plt.subplots(figsize=(17, 6))
    ax.fill_between(xline - xline[0], dtm[i0], np.nanmin(dtm[i0]) - 2, color="#78350f", alpha=0.8, label="ground (DTM)")
    ax.plot(xline - xline[0], dsm[i0], color=GREEN, lw=1.5, label="surface (DSM)")
    ax.set(xlabel="distance west to east (m)", ylabel="elevation (m)", title="Cross-section: surface over ground")
    ax.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=FG); ax.grid(alpha=0.15)
    save(fig, "fig_cross_section.png")
    sec = dtm[i0][np.isfinite(dtm[i0])]
    st["section_dtm_drop_m"] = round(float(np.median(sec) - sec.min()), 1)

    # ---- classified point cloud, oblique view
    las = laspy.read(os.path.join(MS, "point_cloud.laz"))
    n = len(las.points)
    k = np.random.default_rng(0).choice(n, min(n, 600000), replace=False)
    x, y, z, c = np.asarray(las.x)[k], np.asarray(las.y)[k], np.asarray(las.z)[k], np.asarray(las.classification)[k]
    st["point_classes"] = {int(v): int(m) for v, m in zip(*np.unique(np.asarray(las.classification), return_counts=True))}
    st["ground_pct"] = round(100 * st["point_classes"].get(2, 0) / n, 1)
    fig = plt.figure(figsize=(20, 9))
    fig.subplots_adjust(left=0, right=1, wspace=-0.05)
    for p, (title, col) in enumerate((("Point cloud in true colour", None), ("Classified: ground vs everything else", "cls"))):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d")
        if col is None:
            cc = np.stack([las.red[k], las.green[k], las.blue[k]], -1).astype(float)
            cc /= 65535 if cc.max() > 255 else 255
        else:
            cc = np.where((c == 2)[:, None], np.array([[0.47, 0.35, 0.2]]), np.array([[0.29, 0.87, 0.5]]))
        ax.scatter(x, y, z, c=cc, s=0.3, linewidths=0)
        ax.view_init(elev=32, azim=-62); ax.set_axis_off(); ax.set_title(title)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z) * 3), zoom=1.55)
        ax.set_facecolor(BG)
    save(fig, "fig_point_cloud.png")

    # ---- Metashape vs OpenDroneMap (same 77 photos)
    o_dsm, _ = load(os.path.join(ODM, "odm_dem", "dsm.tif"), like)
    both = core & np.isfinite(o_dsm)
    d = (dsm - o_dsm)[both]
    off = float(np.median(d))
    st["vs_odm"] = {"dsm_median_offset_m": round(off, 2), "dsm_abs_diff_after_offset_m_median": round(float(np.median(np.abs(d - off))), 2),
                    "dsm_abs_diff_after_offset_m_p90": round(float(np.percentile(np.abs(d - off), 90)), 2)}
    bench = {}
    for line in open(os.path.join(ODM, "benchmark.txt")):
        if "runtime" in line:
            k_, v_ = line.split(" runtime:")
            bench[k_.strip()] = float(v_.split()[0])
    ms = json.load(open(os.path.join(OUT, "metashape_stats.json")))
    st["vs_odm"]["odm_minutes"] = round(sum(bench.values()) / 60, 1)
    st["vs_odm"]["metashape_minutes"] = ms["total_minutes"]
    with rasterio.open(os.path.join(MS, "orthomosaic.tif")) as r:
        st["metashape_ortho_cm"] = round(100 * r.res[0], 2)
    with rasterio.open(os.path.join(ODM, "odm_orthophoto", "odm_orthophoto.tif")) as r:
        st["odm_ortho_cm"] = round(100 * abs(r.res[0]), 2)
    # zoomed crops of both orthomosaics around the tallest tree
    def crop(cy_, cx_, half_m=22):
        h2 = int(half_m / GRID)
        r0, c0 = max(0, cy_ - h2), max(0, cx_ - h2)
        fine = (rasterio.transform.from_origin(tr.c + c0 * GRID, tr.f - r0 * GRID, 0.02, 0.02), 2 * h2 * 5, 2 * h2 * 5, like[3])
        out = []
        for path, has_alpha in ((os.path.join(MS, "orthomosaic.tif"), True), (os.path.join(ODM, "odm_orthophoto", "odm_orthophoto.tif"), True)):
            im = np.stack([load(path, fine, band=b)[0] for b in (1, 2, 3)], -1)
            al = load(path, fine, band=4, resampling=Resampling.nearest)[0]
            im = np.clip(np.nan_to_num(im) / 255, 0, 1)
            im[~(np.nan_to_num(al) > 0)] = (0.04, 0.07, 0.12)
            out.append(im)
        return out
    # the building and car park: a fair, typical patch
    rows, cols = np.nonzero(core)
    by, bx = int(rows.min() + 0.42 * (rows.max() - rows.min())), int(cols.min() + 0.2 * (cols.max() - cols.min()))
    a_rgb, b_rgb = crop(by, bx)
    p_rgb, q_rgb = crop(*tops[np.argmax(heights)])                 # the pond edge: where matching struggles
    fig, axs = plt.subplots(2, 3, figsize=(20, 13), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.42})
    for r_, (m_, o_, lab) in enumerate(((a_rgb, b_rgb, "building and car park"), (p_rgb, q_rgb, "pond edge"))):
        axs[r_, 0].imshow(m_); axs[r_, 0].set_title(f"Metashape: {lab}", fontsize=17)
        axs[r_, 1].imshow(o_); axs[r_, 1].set_title(f"OpenDroneMap: {lab}", fontsize=17)
        for a in axs[r_, :2]:
            a.set_xticks([]); a.set_yticks([])
    axs[1, 2].axis("off")
    axs = [axs[0, 0], axs[0, 1], axs[0, 2]]
    axs[2].barh(["OpenDroneMap\n(CPU, episode 1)", "Metashape\n(GPU)"], [st["vs_odm"]["odm_minutes"], ms["total_minutes"]],
                color=[ACCENT, GREEN])
    for i, v in enumerate([st["vs_odm"]["odm_minutes"], ms["total_minutes"]]):
        axs[2].text(v, i, f"  {v:.0f} min" if v >= 10 else f"  {v:.1f} min", va="center", fontsize=18, color=FG)
    axs[2].set_xlim(0, st["vs_odm"]["odm_minutes"] * 1.35); axs[2].set_title("Processing time, same 77 photos")
    axs[2].tick_params(axis="y", labelsize=15, labelcolor=FG)
    axs[2].set_xlabel("minutes"); axs[2].grid(alpha=0.15, axis="x")
    save(fig, "fig_vs_odm.png")

    json.dump(st, open(os.path.join(OUT, "analysis_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))
