# Episode 3: Detect individual trees in drone / airborne RGB imagery with deep learning

Uses [DeepForest](https://deepforest.readthedocs.io) (a RetinaNet pre-trained on NEON airborne imagery).
The script scores the pre-trained model on a held-out tile, fine-tunes it on one hand-labelled tile
(CPU is enough, about an hour), scores again, and exports crowns from a georeferenced tile as a
GeoPackage with crown diameters.

## Run

```
pip install deepforest geopandas matplotlib
python tree_detection.py results --epochs 6 --patch 400
```

The sample tiles (NEON Yellowstone and Ordway-Swisher sites, 10 cm RGB, hand-labelled crowns) ship
inside the `deepforest` package, so there is nothing to download.

## Outputs (`results/`)

| File | What it is |
|---|---|
| `tree_crowns.gpkg` | detected crowns on the georeferenced tile, with `crown_diam_m` and `score` |
| `fig01..05_*.png` | data, pre-trained predictions, training loss, before/after, export |
| `stats.txt` | recall and precision before and after fine-tuning, counts, density |

## Using your own imagery

1. Label 300 to 600 crowns on one tile with a box tool (CVAT, QGIS, Label Studio) and export Pascal VOC XML or a CSV with `image_path,xmin,ymin,xmax,ymax,label`.
2. Point `TRAIN_TILE` / `TEST_TILE` at your files (or load the CSV directly).
3. Keep `--patch` near 400 px for 10 cm imagery; use a larger patch for coarser imagery so crowns stay 20 to 60 px wide.
