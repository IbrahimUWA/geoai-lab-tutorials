"""
GeoAI Lab - Episode 2
LiDAR point clouds in Python: ground classification, terrain and canopy height.

Runs on any LAS/LAZ file. Steps:
  1. Load and explore the point cloud
  2. Classify ground points with a progressive morphological filter (written from scratch)
  3. Build a digital terrain model (DTM) and a hillshade
  4. Build a digital surface model (DSM) and a canopy height model (CHM)
  5. Map the tree canopy and draw a cross-section
  6. Export GeoTIFFs and a classified LAS for QGIS

Usage:
  python lidar_pipeline.py input.laz output_folder [--cell 3.0]

Dependencies: laspy lazrs numpy scipy matplotlib rasterio
"""
import argparse
import os
import sys
import time

import laspy
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# ---------- styling for the figures that go into the video ----------
BG = "#0b1220"
FG = "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#334155", "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "font.family": "Segoe UI", "font.size": 14, "axes.titlesize": 20, "axes.titleweight": "bold",
})
FIGSIZE = (19.2, 10.8)  # 1920 x 1080 at dpi 100


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("  figure ->", name)


# =====================================================================
# 1. LOAD AND EXPLORE
# =====================================================================
def load(path):
    t = time.time()
    las = laspy.read(path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    print(f"Loaded {len(xyz):,} points in {time.time() - t:.1f} s")
    print("  extent x:", xyz[:, 0].min().round(1), xyz[:, 0].max().round(1))
    print("  extent y:", xyz[:, 1].min().round(1), xyz[:, 1].max().round(1))
    print("  extent z:", xyz[:, 2].min().round(1), xyz[:, 2].max().round(1))
    vendor_class = np.asarray(las.classification)
    rgb = None
    if "red" in las.point_format.dimension_names:
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float32)
        rgb /= rgb.max() if rgb.max() > 0 else 1
    return las, xyz, vendor_class, rgb


def fig_overview(xyz, rgb, out, n=400_000):
    idx = np.random.default_rng(0).choice(len(xyz), min(n, len(xyz)), replace=False)
    p = xyz[idx]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    sc = axes[0].scatter(p[:, 0], p[:, 1], c=p[:, 2], s=0.4, cmap="viridis", rasterized=True)
    axes[0].set_title("Top view coloured by elevation")
    axes[0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0], fraction=0.03, pad=0.02, label="Elevation (survey units)")
    if rgb is not None:
        axes[1].scatter(p[:, 0], p[:, 1], c=np.clip(rgb[idx] * 1.6, 0, 1), s=0.4, rasterized=True)
        axes[1].set_title("Top view with true colour")
    else:
        axes[1].hist(xyz[:, 2], bins=200, color="#38bdf8")
        axes[1].set_title("Elevation histogram")
    axes[1].set_aspect("equal")
    for a in axes:
        a.set_xlabel("Easting")
        a.set_ylabel("Northing")
    fig.suptitle(f"{len(xyz):,} points - what the raw cloud looks like", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig01_overview.png")


# =====================================================================
# 2. GROUND CLASSIFICATION  (progressive morphological filter)
# =====================================================================
def grid_index(xyz, cell):
    """Map every point to a raster cell. Returns (rows, cols, shape, transform origin)."""
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    x1, y1 = xyz[:, 0].max(), xyz[:, 1].max()
    ncol = int(np.ceil((x1 - x0) / cell)) + 1
    nrow = int(np.ceil((y1 - y0) / cell)) + 1
    col = ((xyz[:, 0] - x0) / cell).astype(np.int64)
    row = ((y1 - xyz[:, 1]) / cell).astype(np.int64)  # row 0 = north edge
    return row, col, (nrow, ncol), (x0, y1)


def bin_reduce(values, row, col, shape, how="min"):
    """Per-cell min / max / mean of a point attribute, NaN where empty."""
    flat = row * shape[1] + col
    n = shape[0] * shape[1]
    if how == "min":
        g = np.full(n, np.inf)
        np.minimum.at(g, flat, values)
        g[np.isinf(g)] = np.nan
    elif how == "max":
        g = np.full(n, -np.inf)
        np.maximum.at(g, flat, values)
        g[np.isinf(g)] = np.nan
    else:
        s = np.zeros(n)
        c = np.zeros(n)
        np.add.at(s, flat, values)
        np.add.at(c, flat, 1)
        with np.errstate(invalid="ignore"):
            g = s / c
        g[c == 0] = np.nan
    return g.reshape(shape)


def fill_nan_nearest(a):
    """Fill NaN cells with the value of the nearest valid cell."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def classify_ground(xyz, cell=3.0, max_window=65, slope=0.3, init_thresh=0.5, max_thresh=10.0):
    """
    Progressive morphological filter (Zhang et al. 2003) on a min-elevation raster.
    Everything in survey units (this tile is in feet).
    Returns a boolean ground mask per point and the final ground surface raster.
    """
    row, col, shape, _ = grid_index(xyz, cell)
    zmin = fill_nan_nearest(bin_reduce(xyz[:, 2], row, col, shape, "min"))
    surface = zmin.copy()
    window, prev = 1, 0
    while window <= max_window:
        opened = ndimage.grey_opening(surface, size=(window, window))
        # allowed height step grows with how much the window grew (so real slopes survive)
        thresh = min(slope * (window - prev) * cell + init_thresh, max_thresh)
        bump = surface - opened            # how far each cell sticks up above the opened surface
        surface = np.where(bump > thresh, opened, surface)
        print(f"  window {window:>3} cells  threshold {thresh:4.2f}")
        prev, window = window, window * 2 + 1
    # a point is ground if it lies close to the filtered surface under it
    ground = (xyz[:, 2] - surface[row, col]) < init_thresh + 0.5
    return ground, surface


def fig_ground(xyz, ground, vendor_class, out, n=400_000):
    idx = np.random.default_rng(1).choice(len(xyz), min(n, len(xyz)), replace=False)
    p, g = xyz[idx], ground[idx]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].scatter(p[~g, 0], p[~g, 1], c="#f97316", s=0.4, label="non-ground", rasterized=True)
    axes[0].scatter(p[g, 0], p[g, 1], c="#38bdf8", s=0.4, label="ground", rasterized=True)
    axes[0].set_title("Our ground classification")
    axes[0].legend(markerscale=20, loc="lower right")
    v = vendor_class[idx] == 2
    axes[1].scatter(p[~v, 0], p[~v, 1], c="#f97316", s=0.4, rasterized=True)
    axes[1].scatter(p[v, 0], p[v, 1], c="#38bdf8", s=0.4, rasterized=True)
    axes[1].set_title("Vendor ground labels (class 2)")
    for a in axes:
        a.set_aspect("equal")
        a.set_xlabel("Easting")
        a.set_ylabel("Northing")
    ref = vendor_class == 2
    agree = (ground == ref).mean() * 100
    fig_ground.recall = (ground & ref).sum() / ref.sum() * 100
    fig_ground.precision = (ground & ref).sum() / max(ground.sum(), 1) * 100
    fig.suptitle(f"Ground vs non-ground - {agree:.1f}% agreement with the vendor labels",
                 fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig02_ground.png")
    return agree


# =====================================================================
# 3-4. DTM, DSM, CHM
# =====================================================================
def hillshade(z, cell, az=315, alt=45):
    ls = LightSource(azdeg=az, altdeg=alt)
    return ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), vert_exag=1, dx=cell, dy=cell)


def build_rasters(xyz, ground, cell):
    row, col, shape, origin = grid_index(xyz, cell)
    dtm = bin_reduce(xyz[ground, 2], row[ground], col[ground], shape, "mean")
    dtm = fill_nan_nearest(dtm)
    dtm = ndimage.median_filter(dtm, size=3)          # knock out single-cell spikes
    dsm = fill_nan_nearest(bin_reduce(xyz[:, 2], row, col, shape, "max"))
    chm = np.clip(dsm - dtm, 0, None)
    return dtm, dsm, chm, shape, origin


def fig_dtm_dsm(dtm, dsm, cell, out):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, z, name in zip(axes, (dtm, dsm), ("Digital terrain model (bare earth)", "Digital surface model (everything)")):
        hs = hillshade(z, cell)
        im = ax.imshow(z, cmap="terrain")
        ax.imshow(hs, cmap="gray", alpha=0.45)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Elevation (survey units)")
    fig.suptitle("Two rasters from one point cloud", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig03_dtm_dsm.png")


def fig_chm(chm, cell, out, tree_min=6.0):
    trees = chm >= tree_min
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    im = axes[0].imshow(chm, cmap="YlGn", vmin=0, vmax=100)
    axes[0].set_title("Canopy height model = DSM - DTM")
    plt.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, label="Height above ground")
    axes[1].imshow(np.where(trees, 1, np.nan), cmap="Greens", vmin=0, vmax=1.3)
    axes[1].imshow(hillshade(chm, cell), cmap="gray", alpha=0.25)
    pct = trees.mean() * 100
    axes[1].set_title(f"Objects above {tree_min:g} units: trees and buildings   ({pct:.1f}% of the area)")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Where the trees are, and how tall", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig04_chm.png")
    return pct


def fig_cross_section(xyz, ground, dtm, shape, origin, cell, out, width=4.0):
    """Points within a strip through the middle of the tile, plus the DTM line."""
    x0, y1 = origin
    ymid = y1 - (shape[0] // 2) * cell
    strip = np.abs(xyz[:, 1] - ymid) < width
    p, g = xyz[strip], ground[strip]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(p[~g, 0], p[~g, 2], s=1.5, c="#f97316", label="non-ground points", rasterized=True)
    ax.scatter(p[g, 0], p[g, 2], s=1.5, c="#38bdf8", label="ground points", rasterized=True)
    xs = x0 + np.arange(shape[1]) * cell + cell / 2
    ax.plot(xs, dtm[shape[0] // 2], c="#fde047", lw=2.5, label="DTM")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Elevation")
    ax.set_title("Cross-section through the tile: trees and buildings stand on the terrain line")
    ax.legend(markerscale=6, loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig05_cross_section.png")


# =====================================================================
# 6. EXPORT
# =====================================================================
def export(las, ground, dtm, dsm, chm, origin, cell, out):
    import rasterio
    from rasterio.transform import from_origin
    crs = las.header.parse_crs()
    transform = from_origin(origin[0], origin[1], cell, cell)
    for name, arr in (("dtm", dtm), ("dsm", dsm), ("chm", chm)):
        path = os.path.join(out, f"{name}.tif")
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                           dtype="float32", crs=crs, transform=transform, compress="deflate") as dst:
            dst.write(arr.astype(np.float32), 1)
        print("  raster ->", os.path.basename(path))
    new = laspy.LasData(las.header)
    new.points = las.points.copy()
    cls = np.where(ground, 2, 1).astype(np.uint8)
    new.classification = cls
    path = os.path.join(out, "classified.laz")
    new.write(path)
    print("  cloud  ->", os.path.basename(path))


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--cell", type=float, default=3.0, help="raster cell size in the file's units")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    print("\n[1] Load and explore")
    las, xyz, vendor_class, rgb = load(a.input)
    fig_overview(xyz, rgb, a.out)

    print("\n[2] Ground classification")
    ground, _ = classify_ground(xyz, cell=a.cell)
    print(f"  ground points: {ground.sum():,} of {len(xyz):,} ({ground.mean() * 100:.1f}%)")
    agree = fig_ground(xyz, ground, vendor_class, a.out)

    print("\n[3-4] DTM, DSM, CHM")
    dtm, dsm, chm, shape, origin = build_rasters(xyz, ground, a.cell)
    print(f"  raster size {shape[0]} x {shape[1]} cells of {a.cell:g} units")
    fig_dtm_dsm(dtm, dsm, a.cell, a.out)

    print("\n[5] Canopy and cross-section")
    pct = fig_chm(chm, a.cell, a.out)
    fig_cross_section(xyz, ground, dtm, shape, origin, a.cell, a.out)

    print("\n[6] Export")
    export(las, ground, dtm, dsm, chm, origin, a.cell, a.out)

    print(f"\nDone in {time.time() - t0:.1f} s. Agreement with vendor ground labels {agree:.1f}%, canopy cover {pct:.1f}%")
    with open(os.path.join(a.out, "stats.txt"), "w") as f:
        f.write(f"points={len(xyz)}\nground_pct={ground.mean() * 100:.1f}\nagreement_pct={agree:.1f}\n"
                f"recall_pct={fig_ground.recall:.1f}\nprecision_pct={fig_ground.precision:.1f}\n"
                f"canopy_pct={pct:.1f}\nraster={shape[0]}x{shape[1]}\ncell={a.cell}\nseconds={time.time() - t0:.0f}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
