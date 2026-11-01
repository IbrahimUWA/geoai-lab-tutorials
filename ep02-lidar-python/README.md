# Episode 2: LiDAR point clouds in Python

Ground classification, digital terrain model, digital surface model and canopy height model from a raw
LAS/LAZ file, in plain Python. The ground classifier is a progressive morphological filter
(Zhang et al., 2003) written from scratch with NumPy and SciPy.

Video: https://youtu.be/Ia4OCT15oeQ

## Data

Autzen Stadium tile, an open sample published by the PDAL project (74 MB, 10.6 million points,
NAD83 / Oregon GIC Lambert, units in feet):

```
curl -L -o autzen-classified.laz https://github.com/PDAL/data/raw/main/autzen/autzen-classified.laz
```

The file carries a vendor classification. The script ignores it while classifying and uses it only
to report agreement at the end.

## Run

```
pip install laspy lazrs numpy scipy matplotlib rasterio
python lidar_pipeline.py autzen-classified.laz results --cell 3.0
```

`--cell` is the raster cell size in the file's units. Use roughly the point spacing (about 1 m for
typical airborne data in metres, 3 ft for this tile).

## Outputs (in `results/`)

| File | What it is |
|---|---|
| `dtm.tif` | bare-earth terrain model |
| `dsm.tif` | surface model (highest return per cell) |
| `chm.tif` | canopy height model, `dsm - dtm` |
| `classified.laz` | the input cloud with class 2 = ground, 1 = unclassified |
| `fig01..05_*.png` | the figures shown in the video |
| `stats.txt` | point count, ground fraction, agreement with vendor labels |

## Parameters worth adjusting

| Argument of `classify_ground` | Meaning | Default |
|---|---|---|
| `cell` | raster cell size | 3.0 |
| `max_window` | largest opening window in cells; must exceed the largest building | 65 |
| `slope` | allowed height change per unit of window growth | 0.3 |
| `init_thresh` | base height threshold | 0.5 |
| `max_thresh` | cap on the threshold | 10.0 |

Hilly terrain: raise `slope` and `max_thresh`. Flat urban sites: lower them. Always check a cross-section.

## Reference

Zhang, K., Chen, S., Whitman, D., Shyu, M., Yan, J. and Zhang, C. (2003). A progressive morphological
filter for removing nonground measurements from airborne LIDAR data. *IEEE Transactions on Geoscience
and Remote Sensing*, 41(4), 872-882.
