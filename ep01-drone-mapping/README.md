# Episode 1: Drone mapping from scratch with OpenDroneMap

A folder of overlapping drone photos in, a georeferenced orthomosaic, digital surface model (DSM),
digital terrain model (DTM), height-above-ground map and contours out. OpenDroneMap does the
photogrammetry with one command; a short Python script opens the outputs and makes the figures.

Video: https://youtu.be/dRM6VbwCwlw

## Data

Aukerman sample flight published by the OpenDroneMap project: 77 JPEGs (520 MB) from a Sony
DSC-WX220 compact camera, every photo with an EXIF GPS position.

```
git clone https://github.com/OpenDroneMap/odm_data_aukerman drone/aukerman
```

Expected layout: `drone/aukerman/images/*.JPG`. Any folder of geotagged photos with 70-80 %
overlap works the same way.

## Run OpenDroneMap

Docker (Windows, macOS, Linux):

```
docker run --rm -v <full path to drone>:/datasets opendronemap/odm \
    --project-path /datasets aukerman \
    --dsm --dtm --orthophoto-resolution 5 --dem-resolution 10 \
    --pc-quality medium --feature-quality high --max-concurrency 10
```

`run_odm.sh` and `run_odm.ps1` wrap this command. The native Windows installer from
https://www.opendronemap.org takes the same flags (`run_odm_native.bat` is the version used in the
video; adjust the install and data paths at the top).

Outputs land next to `images/`: `odm_orthophoto/odm_orthophoto.tif`, `odm_dem/dsm.tif`,
`odm_dem/dtm.tif`, `odm_georeferencing/odm_georeferenced_model.laz`, `odm_texturing/`,
`odm_report/report.pdf`.

## Make the figures

```
pip install rasterio numpy scipy matplotlib pillow laspy lazrs
python drone_mapping_post.py drone/aukerman results
```

## Outputs (in `results/`)

| File | What it is |
|---|---|
| `fig01_images.png` | contact sheet of the photos and their GPS positions |
| `fig02_orthomosaic.png` | the orthomosaic with GSD, extent and area |
| `fig03_dsm_dtm.png` | DSM and DTM with hillshade |
| `fig04_height_contours.png` | height above ground (DSM - DTM) and contours on the DTM |
| `fig05_cross_section.png` | DSM / DTM profile through the middle of the site |
| `stats.txt` | image count, GSD, area, point count, max height, ODM runtime |

## Parameters worth adjusting

| ODM flag | Meaning | Used here |
|---|---|---|
| `--orthophoto-resolution` | mosaic pixel size in cm; no finer than the photo GSD | 5 |
| `--dem-resolution` | DSM/DTM pixel size in cm; 2-4x the orthophoto | 10 |
| `--pc-quality` | dense matching detail (`low` to `ultra`); main driver of time and RAM | medium |
| `--feature-quality` | features per photo; raise when matching fails on low-texture ground | high |
| `--fast-orthophoto` | skip the dense cloud when only the mosaic is needed | off |
| `--split` | process a large flight in chunks when memory runs out | off |

## Reference

OpenDroneMap documentation: https://docs.opendronemap.org
