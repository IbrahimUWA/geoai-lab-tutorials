# Episode 6: NDVI time series and crop phenology from Sentinel-2

A full year of Sentinel-2 over a 16 km window, read straight from the cloud-optimised GeoTIFFs on AWS
(Earth Search STAC, no account), cloud-masked with the scene classification band, gap-filled and smoothed
into a growth curve for every 10 m pixel, and reduced to phenology metrics: start of season, peak date,
end of season, length, amplitude, seasonal integral. Checked against the USDA Cropland Data Layer
(Microsoft Planetary Computer, no account): per-crop curves and a one-number corn / soybean rule.

## Run

```
pip install pystac-client planetary-computer rasterio shapely numpy scipy matplotlib
python ndvi_phenology.py results --bbox -93.75 41.95 -93.55 42.10 --year 2021
```

`--bbox` is lon_min lat_min lon_max lat_max in WGS84. `--year` must be a year the Cropland Data Layer
covers on Planetary Computer; outside the US, replace `read_cdl` with your own labels on the same grid,
or skip the comparison. `--step` is the grid spacing in days (default 5), `--workers` the number of
parallel scene reads (default 8). About 70 dates and 2.8 million pixels take a few minutes on a laptop;
the NDVI cube is kept in memory (about 800 MB at 10 m for a 16 km window).

Southern hemisphere: the season crosses New Year, so search July to June instead of a calendar year
(edit the `datetime` range in `find_scenes`).

## Method

1. `find_scenes`: every L2A scene of the year, one per date, keeping the scene that best covers the window.
2. `read_ndvi`: red, NIR and SCL for the window only; cloud, shadow, cirrus, snow and saturated pixels
   masked and the mask grown by 3 pixels. Reads run in a thread pool. Reflectance is DN / 10000: the
   reprocessed scenes advertise an offset of -0.1 in `raster:bands`, but the Earth Search `sentinel-2-l2a`
   COGs are already harmonised, so applying it drives NDVI above 1. `check_offset` prints the median red
   DN over vegetation for the newest and oldest baseline so you can verify this for your own area.
3. `despike`: an observation more than 0.15 below both temporal neighbours is a missed cloud.
4. `interp_time`: vectorised linear interpolation in time onto a regular grid (cumulative-max trick for
   the previous / next valid index, no loop over pixels).
5. `smooth`: Savitzky-Golay (7 steps x 5 days, order 2) plus three upper-envelope passes.
6. `phenology`: threshold method (50% between base and peak on each side), amplitude < 0.2 = no season.
7. Figures, per-class curves (median and quartiles of 4,000 random pixels per class) and export.

## Outputs (`results/`)

| File | What it is |
|---|---|
| `phenology.tif` | float32, 8 bands on the Sentinel-2 10 m grid: `sos`, `peak_doy`, `eos`, `los`, `peak_ndvi`, `amplitude`, `integral`, `n_obs`; NaN = no clear season; band names in the file |
| `class_curves.csv` | median and quartiles of the smoothed NDVI curve per land-cover class, every grid date |
| `fig01..05_*.png` | year of scenes, pixel curves, class curves, phenology maps, corn vs soybean rule |
| `stats.txt` | scene count, clear dates, median phenology per crop, the split date and its agreement |

## Things worth changing

- `smooth(window=7, passes=3)`: widen the window for cloudy regions, fewer passes if curves look inflated.
- `phenology(frac=0.5, min_amp=0.2)`: 0.2 for an earlier start of season; raise `min_amp` to drop weak seasons.
- `CLOUD_SCL`: add class 2 (dark area pixels) if shadows are leaking through; remove 11 (snow) for alpine sites.
- `read_ndvi`: swap NDVI for EVI, NDRE or kNDVI; the rest of the script does not care.
