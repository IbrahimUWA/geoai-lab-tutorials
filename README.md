# GeoAI Lab tutorials

Code and data links for the [GeoAI Lab](https://www.youtube.com/channel/UC_HGuLPYUpp3s9LMVaDIX0A) YouTube channel:
drones, LiDAR and machine learning for the land. Every episode uses open data and free software, so
you can run the whole thing on your own machine.

| Episode | Topic | Folder |
|---|---|---|
| 2 | LiDAR point clouds in Python: ground classification, terrain model and canopy height | [`ep02-lidar-python`](ep02-lidar-python) |
| 3 | Detect individual trees in drone / airborne imagery with deep learning (DeepForest) | [`ep03-tree-detection`](ep03-tree-detection) |
| 4 | Crop type mapping from Sentinel-2 with a random forest | [`ep04-crop-mapping`](ep04-crop-mapping) |
| 5 | A geospatial web app in an hour with AI-assisted coding (Leaflet, GitHub Pages) | [`ep05-web-app`](ep05-web-app) |

More episodes are added as they are published.

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Each episode folder has its own README with the data download link and the exact command to run.

## Licence

Code is released under the MIT licence (see `LICENSE`). Datasets belong to their publishers and are linked, not redistributed.
