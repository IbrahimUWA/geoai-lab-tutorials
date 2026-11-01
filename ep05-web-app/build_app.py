"""
GeoAI Lab - Episode 5
Build a geospatial web app in an hour with AI-assisted coding: a single static HTML page (Leaflet)
that shows the results of episodes 2, 3 and 4 and can be hosted for free on GitHub Pages.

This script turns the GeoTIFFs / GeoPackage from the earlier episodes into web-ready layers:
  - rasters -> PNG overlays in WGS84 plus a small numeric grid (JSON) so the page can read values on click
  - tree crowns -> GeoJSON
Then it writes app/index.html from the template with the layer manifest embedded.

Usage:
  python build_app.py --lidar ../../ep02-lidar-python/results --trees ../../ep03-tree-detection/results \
                      --crops ../../ep04-crop-mapping/results --out ../app

Dependencies: rasterio numpy pillow geopandas matplotlib
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


def to_wgs84(path, max_px=1400):
    """Reproject a raster to EPSG:4326 and return (array, bounds [S, W, N, E])."""
    with rasterio.open(path) as src:
        dst_crs = "EPSG:4326"
        tr, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        scale = max(1, max(w, h) / max_px)
        w, h = int(w / scale), int(h / scale)
        tr = rasterio.Affine(tr.a * scale, tr.b, tr.c, tr.d, tr.e * scale, tr.f)
        dst = np.full((h, w), np.nan, np.float32)
        reproject(src.read(1).astype(np.float32), dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tr, dst_crs=dst_crs, resampling=Resampling.nearest, dst_nodata=np.nan)
        w_, s, e, n = transform_bounds(src.crs, dst_crs, *src.bounds)   # left, bottom, right, top
    return dst, [[s, w_], [n, e]]


def raster_layer(path, out_dir, name, cmap, vmin=None, vmax=None, classes=None, colors=None, grid_px=300):
    arr, bounds = to_wgs84(path)
    valid = ~np.isnan(arr)
    if classes:
        lut = np.array([tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for c in colors], np.uint8)
        idx = np.clip(np.nan_to_num(arr, nan=0).astype(int), 0, len(lut) - 1)
        rgb = lut[idx]
    else:
        vmin = np.nanpercentile(arr, 1) if vmin is None else vmin
        vmax = np.nanpercentile(arr, 99) if vmax is None else vmax
        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
        rgb = (cm.get_cmap(cmap)(np.nan_to_num(norm))[..., :3] * 255).astype(np.uint8)
    alpha = (valid * 255).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha])).save(os.path.join(out_dir, f"{name}.png"))
    # coarse numeric grid for click-to-read
    h, w = arr.shape
    step = max(1, int(max(h, w) / grid_px))
    small = arr[::step, ::step]
    grid = np.where(np.isnan(small), None, np.round(small.astype(np.float64), 1)).tolist()
    json.dump({"bounds": bounds, "rows": small.shape[0], "cols": small.shape[1], "values": grid},
              open(os.path.join(out_dir, f"{name}.json"), "w"))
    print(f"  {name}: {w}x{h} overlay, {small.shape[1]}x{small.shape[0]} value grid")
    return {"id": name, "png": f"{name}.png", "grid": f"{name}.json", "bounds": bounds,
            "vmin": None if classes else float(vmin), "vmax": None if classes else float(vmax),
            "classes": classes, "colors": colors, "cmap": cmap}


def crowns_layer(gpkg, out_dir, name="crowns"):
    import geopandas as gpd
    g = gpd.read_file(gpkg).to_crs(4326)
    g["crown_diam_m"] = g["crown_diam_m"].round(1)
    g[["crown_diam_m", "score", "geometry"]].to_file(os.path.join(out_dir, f"{name}.geojson"), driver="GeoJSON")
    b = g.total_bounds
    print(f"  {name}: {len(g)} crowns")
    return {"id": name, "geojson": f"{name}.geojson", "bounds": [[b[1], b[0]], [b[3], b[2]]], "count": int(len(g))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("Layers:")
    manifest = {
        "sites": [
            {"name": "Autzen, Oregon: LiDAR terrain and canopy (Episode 2)",
             "layers": [raster_layer(os.path.join(a.lidar, "chm.tif"), a.out, "chm", "YlGn", 0, 100),
                        raster_layer(os.path.join(a.lidar, "dtm.tif"), a.out, "dtm", "terrain")],
             "units": "ft"},
            {"name": "Ordway-Swisher, Florida: tree crowns (Episode 3)",
             "layers": [crowns_layer(os.path.join(a.trees, "tree_crowns.gpkg"), a.out)], "units": "m"},
            {"name": "Story County, Iowa: crop types (Episode 4)",
             "layers": [raster_layer(os.path.join(a.crops, "crop_map.tif"), a.out, "crops", None,
                                     classes=["Corn", "Soybeans", "Alfalfa / hay", "Grass / pasture", "Forest", "Water", "Developed", "Wetlands"],
                                     colors=["#fbbf24", "#22c55e", "#a3e635", "#d9f99d", "#166534", "#3b82f6", "#ef4444", "#67e8f9"])],
             "units": ""},
        ]
    }
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"), encoding="utf-8").read()
    html = tpl.replace("/*MANIFEST*/", "const MANIFEST = " + json.dumps(manifest) + ";")
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(html)
    print("app ->", os.path.join(a.out, "index.html"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
