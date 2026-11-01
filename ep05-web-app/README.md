# Episode 5: A geospatial web app in an hour, with AI-assisted coding

A single static HTML page (Leaflet) that shows the outputs of episodes 2, 3 and 4 on an OpenStreetMap
basemap, with layer toggles, an opacity slider, legends and click-to-read values. No server-side
code, so it hosts for free on GitHub Pages.

## Build

```
pip install rasterio numpy pillow geopandas matplotlib
python build_app.py --lidar ../ep02-lidar-python/results --trees ../ep03-tree-detection/results \
                    --crops ../ep04-crop-mapping/results --out ../app
```

`build_app.py` reprojects each raster to WGS84, writes a PNG overlay plus a small JSON value grid
(so the page can read a value on click), converts the crowns GeoPackage to GeoJSON, and fills
`template.html` with a layer manifest.

## Run locally

Browsers block `fetch()` from `file://`, so serve the folder:

```
cd app && python -m http.server 8765
```

then open http://127.0.0.1:8765/. `?site=1` opens a given site directly.

## Deploy

Push the `app` folder to a GitHub repository and enable Pages (Settings, Pages, deploy from branch).
Total size is about 7 MB; keep overlays under roughly 1500 px a side or the page gets slow on phones.

## Known limits

Image overlays are placed in WGS84 bounds on a Web Mercator map, which is fine for areas of a few
kilometres and increasingly wrong for larger ones. For big rasters, serve tiles instead (`gdal2tiles`).
