"""
GeoAI Lab - Episode 3
Detect individual trees in drone / airborne RGB imagery with deep learning (DeepForest, RetinaNet).

Steps:
  1. Look at the data: RGB tiles with hand-drawn crown boxes (NEON sites, bundled with DeepForest)
  2. Run the pre-trained tree model on a test tile and score it against the hand labels
  3. Fine-tune the model on one annotated tile from the new site (CPU is enough)
  4. Score again: before vs after
  5. Predict on a georeferenced tile, count the trees, export a GeoPackage and a crown-size histogram

Usage:
  python tree_detection.py output_folder [--epochs 6] [--patch 400]

Dependencies: deepforest (pulls torch, torchvision, pytorch-lightning), geopandas, matplotlib
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from deepforest import main as df_main
from deepforest import get_data, utilities, preprocess

BG = "#0b1220"
FG = "#e5e7eb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#334155", "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "font.family": "Segoe UI", "font.size": 14, "axes.titlesize": 20, "axes.titleweight": "bold",
})
FIGSIZE = (19.2, 10.8)

TRAIN_TILE = "2019_YELL_2_528000_4978000_image_crop2"   # 574 hand-labelled crowns, Yellowstone
TEST_TILE = "2019_YELL_2_541000_4977000_image_crop"     # 2nd Yellowstone tile, held out
GEO_TILE = "OSBS_029.tif"                               # georeferenced tile, Florida


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=100)
    plt.close(fig)
    print("  figure ->", name)


def draw_boxes(ax, df, color, lw=1.2, label=None):
    first = True
    for _, r in df.iterrows():
        ax.add_patch(Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                               fill=False, edgecolor=color, linewidth=lw, label=label if first else None))
        first = False


def load_labels(stem):
    df = utilities.read_file(get_data(stem + ".xml"))
    df["image_path"] = stem + ".png"
    return df


# =====================================================================
def fig_data(out):
    train = load_labels(TRAIN_TILE)
    test = load_labels(TEST_TILE)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, stem, df, name in zip(axes, (TRAIN_TILE, TEST_TILE), (train, test), ("Training tile", "Test tile (held out)")):
        im = Image.open(get_data(stem + ".png"))
        ax.imshow(im)
        draw_boxes(ax, df, "#fde047", 0.9)
        ax.set_title(f"{name}: {len(df)} hand-labelled crowns, {im.width} x {im.height} px", fontsize=17)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("The data: 10 cm airborne RGB from NEON with crown boxes drawn by hand", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, "fig01_data.png")
    return train, test


def score(model, stem, labels, out, patch, name=None):
    """Predict on a tile and evaluate against hand labels. Returns (predictions, recall, precision)."""
    path = get_data(stem + ".png")
    pred = model.predict_tile(path=path, patch_size=patch, patch_overlap=0.25, iou_threshold=0.15)
    pred = pred if pred is not None else pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score"])
    csv = os.path.join(out, f"labels_{stem}.csv")
    labels[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]].to_csv(csv, index=False)
    root = os.path.dirname(path)
    res = model.evaluate(csv_file=csv, root_dir=root, predictions=pred.copy(), iou_threshold=0.4)
    rec = float(res.get("box_recall", np.nan))
    prec = float(res.get("box_precision", np.nan))
    print(f"  {name or stem}: {len(pred)} predictions, recall {rec:.3f}, precision {prec:.3f}")
    return pred, rec, prec


def fig_predictions(stem, labels, pred, rec, prec, out, name, title):
    im = Image.open(get_data(stem + ".png"))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].imshow(im)
    draw_boxes(axes[0], labels, "#fde047", 1.0, "hand labels")
    axes[0].set_title(f"Hand labels: {len(labels)} crowns")
    axes[1].imshow(im)
    draw_boxes(axes[1], pred, "#4ade80", 1.0, "model")
    axes[1].set_title(f"Model: {len(pred)} crowns   recall {rec * 100:.0f}%   precision {prec * 100:.0f}%")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, name)


def fig_before_after(stem, labels, pred_before, pred_after, scores, out):
    im = Image.open(get_data(stem + ".png"))
    # zoom into a 500 px window with many trees
    cx, cy = labels.xmin.median(), labels.ymin.median()
    x0, y0 = int(max(cx - 250, 0)), int(max(cy - 250, 0))
    box = (x0, y0, x0 + 500, y0 + 500)
    crop = im.crop(box)

    def sub(df):
        d = df[(df.xmin >= x0) & (df.ymin >= y0) & (df.xmax <= box[2]) & (df.ymax <= box[3])].copy()
        d[["xmin", "xmax"]] -= x0
        d[["ymin", "ymax"]] -= y0
        return d

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for ax, df, col, name in zip(axes, (sub(labels), sub(pred_before), sub(pred_after)),
                                 ("#fde047", "#f97316", "#4ade80"),
                                 ("Hand labels", f"Pre-trained model\nrecall {scores[0][0] * 100:.0f}%  precision {scores[0][1] * 100:.0f}%",
                                  f"After fine-tuning\nrecall {scores[1][0] * 100:.0f}%  precision {scores[1][1] * 100:.0f}%")):
        ax.imshow(crop)
        draw_boxes(ax, df, col, 1.6)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Same 50 m window on the test tile: before and after fine-tuning", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig04_before_after.png")


def fig_training(history, out):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(range(1, len(history) + 1), history, marker="o", color="#38bdf8", lw=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Fine-tuning on one annotated tile: the loss falls fast, then flattens")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, out, "fig03_training.png")


def fig_export(model, out, patch):
    import geopandas as gpd
    import rasterio
    path = get_data(GEO_TILE)
    pred = model.predict_tile(path=path, patch_size=patch, patch_overlap=0.25, iou_threshold=0.15)
    with rasterio.open(path) as src:
        crs, transform, res = src.crs, src.transform, src.res
        img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    # pixel boxes -> map coordinates
    from shapely.geometry import box
    geoms = [box(*(transform * (r.xmin, r.ymax)), *(transform * (r.xmax, r.ymin))) for _, r in pred.iterrows()]
    gdf = gpd.GeoDataFrame({"score": pred.score.values}, geometry=geoms, crs=crs)
    gdf["crown_area_m2"] = gdf.geometry.area
    gdf["crown_diam_m"] = 2 * np.sqrt(gdf.crown_area_m2 / np.pi)
    gpkg = os.path.join(out, "tree_crowns.gpkg")
    gdf.to_file(gpkg, driver="GPKG")
    print(f"  {len(gdf)} crowns -> tree_crowns.gpkg (CRS {crs})")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].imshow(img)
    draw_boxes(axes[0], pred, "#4ade80", 1.4)
    ha = img.shape[0] * res[0] * img.shape[1] * res[1] / 10_000
    axes[0].set_title(f"Georeferenced tile: {len(gdf)} trees on {ha:.1f} ha  ({len(gdf) / ha:.0f} trees per ha)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].hist(gdf.crown_diam_m, bins=25, color="#4ade80", edgecolor=BG)
    axes[1].set_xlabel("Crown diameter (m)")
    axes[1].set_ylabel("Trees")
    axes[1].set_title(f"Crown sizes: median {gdf.crown_diam_m.median():.1f} m")
    fig.suptitle("From boxes to a map layer: counts, density and crown size per tree", fontsize=24, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "fig05_export.png")
    return len(gdf), ha, float(gdf.crown_diam_m.median())


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patch", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    print("\n[1] Data")
    train, test = fig_data(a.out)

    print("\n[2] Pre-trained model on the test tile")
    model = df_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree")
    pred0, rec0, prec0 = score(model, TEST_TILE, test, a.out, a.patch, "pre-trained")
    fig_predictions(TEST_TILE, test, pred0, rec0, prec0, a.out, "fig02_pretrained.png",
                    "Pre-trained model, no training on this site yet")

    print("\n[3] Fine-tune on the training tile")
    crops_dir = os.path.join(a.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    train_csv = os.path.join(a.out, f"labels_{TRAIN_TILE}.csv")
    train[["image_path", "xmin", "ymin", "xmax", "ymax", "label"]].to_csv(train_csv, index=False)
    crop_df = preprocess.split_raster(path_to_raster=get_data(TRAIN_TILE + ".png"), annotations_file=train_csv,
                                      patch_size=a.patch, patch_overlap=0.1, save_dir=crops_dir)
    crop_csv = os.path.join(crops_dir, "crops.csv")
    crop_df.to_csv(crop_csv, index=False)
    print(f"  {crop_df.image_path.nunique()} training crops of {a.patch} px, {len(crop_df)} boxes")

    model.config["train"]["csv_file"] = crop_csv
    model.config["train"]["root_dir"] = crops_dir
    model.config["train"]["epochs"] = a.epochs
    model.config["train"]["lr"] = 0.0005
    model.config["batch_size"] = 2
    model.config["workers"] = 0

    history = []

    class Hist:
        """Collect the epoch training loss for the plot."""
        def __init__(self):
            import pytorch_lightning as pl
            self.cb = type("Cb", (pl.Callback,), {"on_train_epoch_end": lambda s, trainer, m: history.append(
                float(trainer.callback_metrics.get("train_loss", trainer.callback_metrics.get("loss", float("nan")))))})()

    h = Hist()
    model.create_trainer(callbacks=[h.cb], enable_checkpointing=False, logger=False, enable_progress_bar=False)
    t1 = time.time()
    model.trainer.fit(model)
    print(f"  trained {a.epochs} epochs in {(time.time() - t1) / 60:.1f} min; loss per epoch: {[round(x, 3) for x in history]}")
    fig_training(history, a.out)

    print("\n[4] Fine-tuned model on the test tile")
    pred1, rec1, prec1 = score(model, TEST_TILE, test, a.out, a.patch, "fine-tuned")
    fig_before_after(TEST_TILE, test, pred0, pred1, ((rec0, prec0), (rec1, prec1)), a.out)

    print("\n[5] Predict on a georeferenced tile and export")
    n, ha, med = fig_export(model, a.out, a.patch)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    with open(os.path.join(a.out, "stats.txt"), "w") as f:
        f.write(f"train_boxes={len(train)}\ntest_boxes={len(test)}\n"
                f"recall0={rec0 * 100:.0f}\nprecision0={prec0 * 100:.0f}\nrecall1={rec1 * 100:.0f}\nprecision1={prec1 * 100:.0f}\n"
                f"epochs={a.epochs}\ncrops={crop_df.image_path.nunique()}\ntrain_minutes={(time.time() - t1) / 60:.0f}\n"
                f"geo_trees={n}\ngeo_ha={ha:.1f}\ngeo_density={n / ha:.0f}\nmedian_crown_m={med:.1f}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
