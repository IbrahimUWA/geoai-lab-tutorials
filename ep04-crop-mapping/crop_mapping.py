"""
GeoAI Lab - Episode 4
Crop type mapping from Sentinel-2 with a random forest, using only open data and free software.

  1. Find cloud-free Sentinel-2 scenes for the area and season with a STAC search (AWS Earth Search, no account)
  2. Read just the window we need from the cloud-optimised GeoTIFFs: 10 m bands on three dates + NDVI
  3. Get reference labels: USDA Cropland Data Layer (via Microsoft Planetary Computer, free, no account)
  4. Split train / test by spatial blocks, not random pixels, and train a random forest
  5. Score it, look at the confusion matrix and feature importances, export the crop map as GeoTIFF

Usage:
  python crop_mapping.py output_folder [--bbox lon_min lat_min lon_max lat_max] [--year 2023]

Dependencies: pystac-client planetary-computer rasterio numpy scikit-learn matplotlib
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin
from pystac_client import Client
import planetary_computer as pc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
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

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]   # 10 m bands + two 20 m SWIR bands (resampled)
DATES = [("{y}-05-20", "{y}-06-25"), ("{y}-07-10", "{y}-08-05"), ("{y}-08-10", "{y}-09-10")]
DATE_NAMES = ["June", "late July", "August"]

# CDL codes -> our classes
CLASS_MAP = {1: "Corn", 5: "Soybeans", 36: "Alfalfa", 37: "Other hay", 176: "Grass / pasture",
             141: "Deciduous forest", 142: "Evergreen forest", 143: "Mixed forest",
             111: "Water", 121: "Developed", 122: "Developed", 123: "Developed", 124: "Developed",
             190: "Wetlands", 195: "Wetlands"}
CLASSES = ["Corn", "Soybeans", "Alfalfa / hay", "Grass / pasture", "Forest", "Water", "Developed", "Wetlands"]
MERGE = {"Alfalfa": "Alfalfa / hay", "Other hay": "Alfalfa / hay",
         "Deciduous forest": "Forest", "Evergreen forest": "Forest", "Mixed forest": "Forest"}
COLORS = ["#fbbf24", "#22c55e", "#a3e635", "#d9f99d", "#166534", "#3b82f6", "#ef4444", "#67e8f9"]


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=100)
    plt.close(fig)
    print("  figure ->", name)


# =====================================================================
# 1-2. SENTINEL-2 VIA STAC
# =====================================================================
CLOUD_SCL = [3, 8, 9, 10, 11]      # SCL classes: cloud shadow, cloud medium / high probability, cirrus, snow


def cloud_fraction(item, bbox):
    """Fraction of OUR window that the scene classification (SCL) flags as cloud or shadow."""
    scl, _, _ = read_window(item.assets["scl"].href, bbox, resampling=Resampling.nearest)
    return float(np.isin(scl, CLOUD_SCL).mean())


def find_scenes(bbox, year):
    """The tile-level cloud percentage lies about a small window, so check the window itself."""
    cat = Client.open("https://earth-search.aws.element84.com/v1")
    scenes = []
    for (d0, d1), name in zip(DATES, DATE_NAMES):
        d0, d1 = d0.format(y=year), d1.format(y=year)
        items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{d0}/{d1}",
                                query={"eo:cloud_cover": {"lt": 30}}, max_items=40).items())
        if not items:
            raise SystemExit(f"no scene for {name}; widen the date range")
        items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:8]
        scored = [(cloud_fraction(i, bbox), i) for i in items]
        frac, best = min(scored, key=lambda t: t[0])
        print(f"  {name:10} {best.id}  tile cloud {best.properties['eo:cloud_cover']:.1f}%  our window {frac * 100:.1f}%")
        scenes.append(best)
    return scenes


def read_window(href, bbox, shape=None, transform=None, resampling=Resampling.bilinear):
    """Read only the bbox from a remote COG. Optionally resample onto a given grid."""
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*b, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win).astype(np.float32)
        tr = src.window_transform(win)
        crs = src.crs
    if shape is not None and arr.shape != shape:
        dst = np.zeros(shape, np.float32)
        reproject(arr, dst, src_transform=tr, src_crs=crs, dst_transform=transform, dst_crs=crs, resampling=resampling)
        arr, tr = dst, transform
    return arr, tr, crs


def build_stack(scenes, bbox):
    """Returns features (H, W, F), feature names, transform, crs, and a mask of cloudy pixels on any date."""
    layers, names = [], []
    grid = None
    cloudy = None
    for scene, dname in zip(scenes, DATE_NAMES):
        bands = {}
        for b in BANDS:
            arr, tr, crs = read_window(scene.assets[b].href, bbox, *(grid or (None, None)))
            if grid is None:
                grid = (arr.shape, tr)
            bands[b] = arr / 10000.0
            layers.append(bands[b])
            names.append(f"{b} {dname}")
        ndvi = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"] + 1e-6)
        layers.append(ndvi)
        names.append(f"NDVI {dname}")
        scl, _, _ = read_window(scene.assets["scl"].href, bbox, *grid, resampling=Resampling.nearest)
        c = np.isin(scl, CLOUD_SCL)
        cloudy = c if cloudy is None else (cloudy | c)
        print(f"  {dname:10} {len(BANDS)} bands + NDVI read, grid {arr.shape}, cloud/shadow {c.mean() * 100:.1f}%")
    return np.stack(layers, -1), names, grid[1], crs, cloudy


# =====================================================================
# 3. LABELS FROM THE CROPLAND DATA LAYER
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
        labels[cdl == code] = CLASSES.index(MERGE.get(name, name))
    cov = (labels >= 0).mean() * 100
    print(f"  CDL {year} read: {cov:.0f}% of pixels fall in our {len(CLASSES)} classes")
    return labels


# =====================================================================
# FIGURES
# =====================================================================
def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, lo), np.nanpercentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)


def fig_imagery(X, names, out):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    i = names.index("red late July")
    rgb = np.clip((np.dstack([X[..., i], X[..., i - 1], X[..., i - 2]]) - 0.02) / 0.23, 0, 1) ** 0.8
    axes[0].imshow(rgb)
    axes[0].set_title("True colour, late July")
    for ax, dn in zip(axes[1:], ("June", "August")):
        im = ax.imshow(X[..., names.index(f"NDVI {dn}")], cmap="RdYlGn", vmin=0, vmax=0.9)
        ax.set_title(f"NDVI, {dn}")
    plt.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02, label="NDVI")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Sentinel-2 at 10 m: the same fields green up at different times", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig01_imagery.png")


def class_cmap():
    return ListedColormap(COLORS), BoundaryNorm(np.arange(-0.5, len(CLASSES) + 0.5), len(CLASSES))


def fig_labels(labels, blocks_train, out):
    cmap, norm = class_cmap()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].imshow(np.where(labels >= 0, labels, np.nan), cmap=cmap, norm=norm)
    axes[0].set_title("USDA Cropland Data Layer (reference labels)")
    axes[0].legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASSES)], loc="lower left", fontsize=11)
    axes[1].imshow(np.where(labels >= 0, labels, np.nan), cmap=cmap, norm=norm, alpha=0.35)
    axes[1].imshow(np.where(blocks_train, np.nan, 1), cmap="Greys", vmin=0, vmax=1.6, alpha=0.75)
    axes[1].set_title("Spatial split: grey blocks are held out for testing")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Reference labels, and a split that keeps test fields away from training fields",
                 fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig02_labels_split.png")


def fig_metrics(cm, names, importances, oa, out):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.1, 1]})
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    axes[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(CLASSES)))
    axes[0].set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=11)
    axes[0].set_yticks(range(len(CLASSES)))
    axes[0].set_yticklabels(CLASSES, fontsize=11)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i].sum():
                axes[0].text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center",
                             color="white" if cmn[i, j] > 0.5 else "#cbd5e1", fontsize=11)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Reference (CDL)")
    axes[0].set_title(f"Confusion matrix on held-out blocks, row %   (overall accuracy {oa * 100:.1f}%)")
    order = np.argsort(importances)[-15:]
    axes[1].barh([names[i] for i in order], importances[order], color="#38bdf8")
    axes[1].set_title("Top 15 features by importance")
    axes[1].set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    save(fig, out, "fig03_metrics.png")


def fig_map(pred, labels, out):
    cmap, norm = class_cmap()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].imshow(np.where(pred == 255, np.nan, pred), cmap=cmap, norm=norm)
    axes[0].set_title("Random forest crop map, every clear pixel")
    axes[1].imshow(np.where(labels >= 0, labels, np.nan), cmap=cmap, norm=norm)
    axes[1].set_title("Cropland Data Layer")
    axes[0].legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASSES)], loc="lower left", fontsize=11)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    agree = (pred[labels >= 0] == labels[labels >= 0]).mean() * 100
    fig.suptitle(f"Our map next to the reference: {agree:.1f}% of labelled pixels agree", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig04_map.png")
    return agree


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-93.75, 41.95, -93.55, 42.10])
    ap.add_argument("--year", type=int, default=2021, help="CDL on Planetary Computer currently ends at 2021")
    ap.add_argument("--block", type=int, default=100, help="block size in pixels for the spatial split (100 = 1 km)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(0)

    print("\n[1] Scenes")
    scenes = find_scenes(a.bbox, a.year)

    print("\n[2] Features")
    X, names, transform, crs, cloudy = build_stack(scenes, a.bbox)
    H, W, F = X.shape
    fig_imagery(X, names, a.out)

    print("\n[3] Labels")
    labels = read_cdl(a.bbox, a.year, (H, W), transform, crs)
    labels[cloudy] = -1                      # never train or test on a cloudy pixel

    print("\n[4] Spatial split and training")
    rows, cols = np.indices((H, W))
    block_id = (rows // a.block) * 1000 + (cols // a.block)
    uniq = np.unique(block_id)
    test_blocks = rng.choice(uniq, size=len(uniq) // 3, replace=False)
    is_test = np.isin(block_id, test_blocks)
    train_mask = (labels >= 0) & ~is_test
    test_mask = (labels >= 0) & is_test
    fig_labels(labels, ~is_test, a.out)

    # training sample: up to 6000 pixels per class (rare classes contribute everything they have)
    tr_idx = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(train_mask & (labels == c))
        if len(idx):
            tr_idx.append(rng.choice(idx, size=min(6000, len(idx)), replace=False))
    tr_idx = np.concatenate(tr_idx)
    Xf = X.reshape(-1, F)
    yf = labels.reshape(-1)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0)
    t1 = time.time()
    rf.fit(Xf[tr_idx], yf[tr_idx])
    print(f"  trained on {len(tr_idx):,} pixels, {F} features, in {time.time() - t1:.0f} s")

    print("\n[5] Evaluation and map")
    te_idx = np.flatnonzero(test_mask)
    yp = rf.predict(Xf[te_idx])
    oa = accuracy_score(yf[te_idx], yp)
    f1 = f1_score(yf[te_idx], yp, average=None, labels=range(len(CLASSES)))
    cm = confusion_matrix(yf[te_idx], yp, labels=range(len(CLASSES)))
    print(f"  overall accuracy on held-out blocks: {oa * 100:.1f}%")
    for c, v in zip(CLASSES, f1):
        print(f"    F1 {c:16} {v:.2f}")
    fig_metrics(cm, names, rf.feature_importances_, oa, a.out)
    pred = rf.predict(Xf).reshape(H, W).astype(np.uint8)
    pred[cloudy] = 255                       # nodata where any date was cloudy
    agree = fig_map(pred, labels, a.out)

    path = os.path.join(a.out, "crop_map.tif")
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8", nodata=255,
                       crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(pred, 1)
        dst.update_tags(classes=";".join(f"{i}={c}" for i, c in enumerate(CLASSES)))
    print("  raster -> crop_map.tif")

    area_ha = {c: float((pred == i).sum() * abs(transform.a * transform.e) / 10_000) for i, c in enumerate(CLASSES)}
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    with open(os.path.join(a.out, "stats.txt"), "w") as f:
        f.write(f"scenes={';'.join(s.id for s in scenes)}\ngrid={H}x{W}\nfeatures={F}\ntrain_pixels={len(tr_idx)}\n"
                f"test_pixels={len(te_idx)}\noa={oa * 100:.1f}\nagree={agree:.1f}\n"
                f"f1_corn={f1[0]:.2f}\nf1_soy={f1[1]:.2f}\ncorn_ha={area_ha['Corn']:.0f}\nsoy_ha={area_ha['Soybeans']:.0f}\n"
                f"top_feature={names[int(np.argmax(rf.feature_importances_))]}\nseconds={time.time() - t0:.0f}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
