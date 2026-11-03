# GeoAI Lab tutorials

Code and data links for the [GeoAI Lab](https://www.youtube.com/channel/UC_HGuLPYUpp3s9LMVaDIX0A) YouTube channel:
drones, LiDAR and machine learning for the land. Every episode uses open data and free software, so
you can run the whole thing on your own machine.

| Episode | Topic | Video | Folder |
|---|---|---|---|
| 1 | Drone mapping from scratch with OpenDroneMap: orthomosaic, DSM, DTM and contours from a folder of photos | [Watch](https://youtu.be/dRM6VbwCwlw) | [`ep01-drone-mapping`](ep01-drone-mapping) |
| 2 | LiDAR point clouds in Python: ground classification, terrain model and canopy height | [Watch](https://youtu.be/HzDmpLFB380) | [`ep02-lidar-python`](ep02-lidar-python) |
| 3 | Detect individual trees in drone / airborne imagery with deep learning (DeepForest) | [Watch](https://youtu.be/sM0gVm-4oBo) | [`ep03-tree-detection`](ep03-tree-detection) |
| 4 | Crop type mapping from Sentinel-2 with a random forest | [Watch](https://youtu.be/69nm5NyzPng) | [`ep04-crop-mapping`](ep04-crop-mapping) |
| 5 | A geospatial web app in an hour with AI-assisted coding (Leaflet, GitHub Pages) | [Watch](https://youtu.be/SDgUc4aq4P8) | [`ep05-web-app`](ep05-web-app) |

Playlist: https://www.youtube.com/playlist?list=PLJ4-bgxsdmvg

Live demo of the episode 5 web app (served from the `docs` folder by GitHub Pages): https://ibrahimuwa.github.io/geoai-lab-tutorials/

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
