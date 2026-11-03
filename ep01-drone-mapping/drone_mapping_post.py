"""
GeoAI Lab - Episode 1
Drone mapping from scratch with OpenDroneMap: from a folder of JPEGs to an orthomosaic and elevation models.

Part A (one Docker command, see run_odm.sh / run_odm.ps1):
    docker run --rm -v <data folder>:/datasets opendronemap/odm --project-path /datasets aukerman \
        --dsm --dtm --orthophoto-resolution 5 --dem-resolution 10 --pc-quality medium

Part B (this script): open the ODM outputs with rasterio and make the figures:
  1. the input images and their GPS positions
  2. the orthomosaic
  3. DSM and DTM with hillshade
  4. height above ground and contour lines
  5. a cross-section, and the numbers that matter (GSD, area, point count)

Usage:
  python drone_mapping_post.py <project folder> <output folder>

Dependencies: rasterio numpy scipy matplotlib pillow laspy lazrs
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter
from PIL import Image, ExifTags

BG = "#0b1220"
FG = "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#334155", "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "font.family": "Segoe UI", "font.size": 14, "axes.titlesize": 20, "axes.titleweight": "bold",
})
FIGSIZE = (19.2, 10.8)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=100)
    plt.close(fig)
    print("  figure ->", name)


# ---------------------------------------------------------------- 1. images and GPS
def exif_gps(path):
    try:
        exif = Image.open(path)._getexif() or {}
    except Exception:
        return None
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    gps = tags.get("GPSInfo")
    if not gps:
        return None
    g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}

    def dms(v, ref):
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W") else d

    try:
        return (dms(g["GPSLongitude"], g["GPSLongitudeRef"]), dms(g["GPSLatitude"], g["GPSLatitudeRef"]),
                float(g.get("GPSAltitude", 0)))
    except KeyError:
        return None


def fig_images(project, out):
    paths = sorted(glob.glob(os.path.join(project, "images", "*.[jJ][pP]*[gG]")))
    pts = [p for p in (exif_gps(x) for x in paths) if p]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.2, 1]})
    # contact sheet of the first 12 images
    sheet = axes[0]
    sheet.set_title(f"{len(paths)} photos straight from the drone")
    sheet.set_xticks([])
    sheet.set_yticks([])
    cols, rows = 4, 3
    thumbs = []
    for p in paths[:cols * rows]:
        im = Image.open(p)
        im.thumbnail((400, 300))
        thumbs.append(np.asarray(im))
    if thumbs:
        h, w = thumbs[0].shape[:2]
        canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            canvas[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t[..., :3]
        sheet.imshow(canvas)
    if pts:
        lon, lat, alt = map(np.array, zip(*pts))
        sc = axes[1].scatter(lon, lat, c=alt, cmap="viridis", s=40)
        axes[1].plot(lon, lat, color="#64748b", lw=0.8, alpha=0.7)
        plt.colorbar(sc, ax=axes[1], fraction=0.04, pad=0.02, label="GPS altitude (m)")
        axes[1].set_title("Where each photo was taken (EXIF GPS)")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        axes[1].set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
        axes[1].ticklabel_format(useOffset=False, style="plain")
        axes[1].tick_params(axis="x", labelrotation=30)
    fig.suptitle("Input: overlapping photos with GPS tags. That is all photogrammetry needs", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig01_images.png")
    return len(paths)


# ---------------------------------------------------------------- 2-4. rasters
def read(path, bands=None, max_px=2200, shape=None):
    with rasterio.open(path) as src:
        scale = max(1, int(max(src.width, src.height) / max_px))
        if shape is None:
            shape = (src.height // scale, src.width // scale)
        if bands:
            arr = src.read(bands, out_shape=(len(bands),) + shape)
        else:
            arr = src.read(1, out_shape=shape).astype(np.float32)
            nod = src.nodata
            if nod is not None:
                arr[arr == nod] = np.nan
            arr[arr < -1000] = np.nan
        res = (src.res[0] * scale, src.res[1] * scale)
        return arr, res, src.crs, src.bounds, src.res


def hillshade(z, res):
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=1.5, dx=res[0], dy=res[1])


def fig_ortho(project, out):
    path = os.path.join(project, "odm_orthophoto", "odm_orthophoto.tif")
    rgb, res, crs, bounds, full_res = read(path, bands=[1, 2, 3])
    rgb = np.moveaxis(rgb, 0, -1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(rgb)
    w_m = (bounds.right - bounds.left)
    h_m = (bounds.top - bounds.bottom)
    ax.set_title(f"Orthomosaic: {full_res[0] * 100:.1f} cm per pixel, {w_m:.0f} m x {h_m:.0f} m, {w_m * h_m / 10_000:.1f} ha   ({crs})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save(fig, out, "fig02_orthomosaic.png")
    return full_res[0] * 100, w_m * h_m / 10_000


def fig_dems(project, out):
    dsm, res, crs, bounds, _ = read(os.path.join(project, "odm_dem", "dsm.tif"))
    # ODM's DSM and DTM can differ by a pixel in size; resample the DTM onto the DSM grid
    dtm, _, _, _, _ = read(os.path.join(project, "odm_dem", "dtm.tif"), shape=dsm.shape)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    vmin, vmax = np.nanpercentile(dtm, 1), np.nanpercentile(dsm, 99)
    for ax, z, name in zip(axes, (dsm, dtm), ("Digital surface model", "Digital terrain model")):
        im = ax.imshow(z, cmap="terrain", vmin=vmin, vmax=vmax)
        ax.imshow(hillshade(z, res), cmap="gray", alpha=0.45)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Elevation (m)")
    fig.suptitle("Elevation from photos alone: structure from motion plus dense matching", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig03_dsm_dtm.png")
    return dsm, dtm, res


def fig_height_contours(dsm, dtm, res, out):
    hag = np.clip(dsm - dtm, 0, None)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    im = axes[0].imshow(hag, cmap="YlGn", vmin=0, vmax=np.nanpercentile(hag, 99))
    axes[0].set_title("Height above ground = DSM - DTM")
    plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, label="metres")
    axes[1].imshow(hillshade(dtm, res), cmap="gray")
    # smooth the terrain a little before contouring (dense-matching DTMs are noisy at the pixel level)
    # and mask everything outside the footprint so no contour is drawn along the edge
    valid = ~np.isnan(dtm)
    smooth = gaussian_filter(np.where(valid, dtm, np.nanmedian(dtm)), sigma=1.5 / res[0])   # ~1.5 m radius
    smooth = np.ma.masked_where(~valid, smooth)
    lo, hi = np.nanpercentile(dtm, 0.5), np.nanpercentile(dtm, 99.5)
    step = max(0.5, round((hi - lo) / 25 * 2) / 2)
    cs = axes[1].contour(smooth, levels=np.arange(np.floor(lo), hi, step), colors="#fde047", linewidths=0.7)
    axes[1].clabel(cs, fmt="%.0f", fontsize=8, colors="#fde047")
    axes[1].set_title(f"Contours every {step:g} m on the terrain model")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Two products people actually ask for", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig04_height_contours.png")
    return float(np.nanpercentile(hag, 99))


def fig_cross_section(dsm, dtm, res, out):
    r = dsm.shape[0] // 2
    x = np.arange(dsm.shape[1]) * res[0]
    top, bot = dsm[r], dtm[r]
    ok = ~np.isnan(top) & ~np.isnan(bot)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(x, bot, np.maximum(top, bot), where=ok, color="#4ade80", alpha=0.5,
                    label="above ground (trees, buildings)")
    ax.plot(x, np.where(ok, top, np.nan), color="#f97316", lw=1.5, label="DSM")
    ax.plot(x, np.where(ok, bot, np.nan), color="#fde047", lw=2.5, label="DTM")
    ax.set_xlabel("Distance along the section (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Cross-section through the middle of the site")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig05_cross_section.png")


def point_count(project):
    try:
        import laspy
        p = glob.glob(os.path.join(project, "odm_georeferencing", "odm_georeferenced_model.laz"))
        if p:
            with laspy.open(p[0]) as f:
                return f.header.point_count
    except Exception:
        pass
    return 0


def odm_minutes(project):
    """Total ODM processing time from benchmark.txt (one '<stage> runtime: N seconds' line per stage)."""
    path = os.path.join(project, "benchmark.txt")
    if not os.path.exists(path):
        return 0
    secs = 0.0
    for line in open(path):
        if "runtime:" in line:
            secs += float(line.split("runtime:")[1].split()[0])
    return secs / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("\n[1] Images")
    n = fig_images(a.project, a.out)
    print("\n[2] Orthomosaic")
    gsd, ha = fig_ortho(a.project, a.out)
    print("\n[3] DSM / DTM")
    dsm, dtm, res = fig_dems(a.project, a.out)
    print("\n[4] Height and contours")
    hmax = fig_height_contours(dsm, dtm, res, a.out)
    print("\n[5] Cross-section")
    fig_cross_section(dsm, dtm, res, a.out)
    pts = point_count(a.project)
    with open(os.path.join(a.out, "stats.txt"), "w") as f:
        f.write(f"images={n}\ngsd_cm={gsd:.1f}\narea_ha={ha:.1f}\npoints={pts}\npoints_m={pts / 1e6:.1f}\nhmax={hmax:.0f}\n"
                f"odm_min={odm_minutes(a.project):.0f}\n")
    print(f"\n{n} images, GSD {gsd:.1f} cm, {ha:.1f} ha, {pts:,} points")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
