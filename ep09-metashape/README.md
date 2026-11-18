# Drone mapping, part 2: processing in Agisoft Metashape, then analysing the results

The same 77 Aukerman photos as episode 1, processed with the Metashape Professional Python API:
aligned cameras, a dense point cloud, ground classification, DSM, orthomosaic and DTM. A second
script measures things from the outputs: canopy height, tree tops, an RGB vegetation index (VARI),
slope and contours, a cross-section, and a comparison against the OpenDroneMap run from episode 1.

Video: https://youtu.be/3jzh7yXuYOI (part 1: https://youtu.be/Smw86pLmYBI)

## What you need

- **Agisoft Metashape Professional** 2.x with an activated licence (a 30-day trial works). The Standard
  edition has no Python API and no ground classification. Install the Python module with the wheel
  from https://www.agisoft.com/downloads/installer/ (`pip install Metashape-2.x.x-cp37.cp38...-win_amd64.whl`)
  into Python 3.11, or run the script inside Metashape (Tools > Run Script).
- An NVIDIA GPU is optional but makes depth maps much faster.
- For the analysis: `pip install rasterio laspy[lazrs] scikit-image scipy matplotlib numpy pyproj pillow`

## Data

Aukerman sample flight published by the OpenDroneMap project (77 JPEGs, Sony DSC-WX220):

```
git clone https://github.com/OpenDroneMap/odm_data_aukerman data/drone/aukerman
```

Paths are relative to the repository root (`data/drone/...`); set `GEOAI_ROOT` to use another folder.

## Run

```
python code/01_metashape_process.py            # about 6 minutes on a workstation GPU
python code/02_analysis.py                     # figures + results/analysis_stats.json
```

`01` writes `data/drone/aukerman_metashape/`: `aukerman.psx` (open it in Metashape), `orthomosaic.tif`,
`dsm.tif`, `dtm.tif`, `point_cloud.laz` (classified), `report.pdf`, plus `results/metashape_stats.json`
(aligned cameras, tie points, reprojection error, camera-vs-GPS error, timings).

`02_analysis.py` compares against `data/drone/aukerman/odm_dem/dsm.tif` and the ODM orthophoto from
episode 1; run episode 1 first or remove that last section.

## Results on this dataset

| | |
|---|---|
| Cameras aligned | 77 of 77, reprojection error 1.2 px |
| Point cloud | 24.5 million points, 52% ground |
| Orthomosaic / DEM | 2.8 cm / 5.7 cm per pixel |
| Processing time | 5.9 min (Metashape, GPU) vs 178 min (OpenDroneMap, CPU in Docker, episode 1) |
| Tree tops over 4 m | 300, median height 18.5 m |

Known problem, kept on purpose: water reflections at the pond pulled ground points about 14 m below the
water surface, so the DTM dips there and the "tallest tree" next to the pond is an artefact. Draw a
boundary around water in Metashape and exclude it before building the DTM.

No ground control was used, so absolute positions are only as good as the drone's GPS (metres). Tree
heights are not validated against field measurements.
