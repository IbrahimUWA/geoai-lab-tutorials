# Episode 4: Crop type mapping from Sentinel-2 with a random forest

Open imagery (Sentinel-2 L2A on AWS, via the Earth Search STAC API), open labels (USDA Cropland Data
Layer via Microsoft Planetary Computer), windowed reads from cloud-optimised GeoTIFFs, a spatial block
split and a random forest. No accounts, no downloads of whole scenes.

## Run

```
pip install pystac-client planetary-computer rasterio numpy scikit-learn matplotlib
python crop_mapping.py results --bbox -93.75 41.95 -93.55 42.10 --year 2021
```

`--bbox` is lon_min lat_min lon_max lat_max in WGS84. `--year` must be a year the Cropland Data Layer
is available for on Planetary Computer (2008 to 2021 at the time of recording). Any US area works;
outside the US, replace `read_cdl` with your own labels on the same grid.

## Outputs (`results/`)

| File | What it is |
|---|---|
| `crop_map.tif` | uint8 class map on the Sentinel-2 10 m grid, nodata 255 where any date was cloudy; class names in the tags |
| `fig01..04_*.png` | imagery, labels and split, metrics, final map |
| `stats.txt` | scenes used, accuracy, per-class F1, hectares |

## Things worth changing

- `DATES`: add more windows across the season for better separation.
- `BANDS`: `swir16` and `swir22` matter most; `rededge1..3` are also available at 20 m.
- `--block`: block size in pixels for the spatial split (100 = 1 km).
