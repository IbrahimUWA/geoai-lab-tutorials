"""GeoAI Lab - Metashape lecture, step 1: drone photos -> aligned cameras -> point cloud -> DSM, DTM, orthomosaic.

Runs the whole Agisoft Metashape Professional workflow from Python (the same steps as the Workflow menu) and
records how long each step took and how well it worked.

    python 01_metashape_process.py <images folder> <output folder>

Needs Metashape Professional (the Python module ships with it, or pip install the Metashape wheel and activate
a licence). Standard edition has no Python API and no DTM/ground classification.
"""
import glob
import json
import math
import os
import sys
import time

import Metashape

_DATA = os.path.join(os.environ.get("GEOAI_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "data", "drone")
IMAGES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_DATA, "aukerman", "images")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DATA, "aukerman_metashape")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
UTM = Metashape.CoordinateSystem("EPSG::32617")          # WGS 84 / UTM zone 17N: metres, for this site in Ohio

os.makedirs(OUT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
Metashape.app.gpu_mask = 2 ** len(Metashape.app.enumGPUDevices()) - 1     # use every GPU
stats, timing = {"metashape_version": Metashape.app.version}, {}


def step(name, fn):
    t = time.time()
    print(f"== {name}", flush=True)
    fn()
    timing[name] = round(time.time() - t, 1)
    print(f"   {timing[name]} s", flush=True)
    doc.save()


doc = Metashape.Document()
doc.save(os.path.join(OUT, "aukerman.psx"))
chunk = doc.addChunk()
photos = sorted(glob.glob(os.path.join(IMAGES, "*.[jJ][pP][gG]")))
chunk.addPhotos(photos)
stats["photos"] = len(photos)
proj = Metashape.OrthoProjection()
proj.crs = UTM

# 1. Find the same features in overlapping photos (High accuracy = full-resolution images).
step("match_photos", lambda: chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=True,
                                               keypoint_limit=40000, tiepoint_limit=4000))
# 2. Solve where each photo was taken and how the lens bends light (bundle adjustment).
step("align_cameras", lambda: chunk.alignCameras())
step("optimize_cameras", lambda: chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_k1=True, fit_k2=True,
                                                       fit_k3=True, fit_p1=True, fit_p2=True))
aligned = [c for c in chunk.cameras if c.transform]
stats["cameras_aligned"] = len(aligned)
stats["tie_points"] = len(chunk.tie_points.points)

# How far each solved camera position is from the drone's own GPS tag (metres, in UTM).
T, crs = chunk.transform.matrix, chunk.crs
errs = []
for c in aligned:
    if c.reference.location:
        est = UTM.project(T.mulp(c.center))                 # solved position, geocentric -> UTM
        ref = UTM.project(crs.unproject(c.reference.location))
        errs.append([est[i] - ref[i] for i in range(3)])
if errs:
    rms = lambda k: math.sqrt(sum(e[k] ** 2 for e in errs) / len(errs))
    stats["camera_vs_gps_rms_m"] = {"x": round(rms(0), 2), "y": round(rms(1), 2), "z": round(rms(2), 2)}

# Reprojection error: how many pixels the 3D tie points miss their photo positions by.
pts, tracks, sq, n = chunk.tie_points.points, chunk.tie_points.tracks, 0.0, 0
proj_by_cam = chunk.tie_points.projections
track_to_pt = {p.track_id: i for i, p in enumerate(pts) if p.valid}
for c in aligned:
    for pr in proj_by_cam[c]:
        i = track_to_pt.get(pr.track_id)
        if i is None:
            continue
        xy = c.project(pts[i].coord)
        if xy is None:
            continue
        d = xy - pr.coord
        sq += d.x ** 2 + d.y ** 2
        n += 1
stats["reprojection_rms_px"] = round(math.sqrt(sq / n), 3) if n else None

# 3. Depth maps (High = half resolution) and the dense point cloud.
step("depth_maps", lambda: chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering))
step("point_cloud", lambda: chunk.buildPointCloud(point_colors=True, point_confidence=True))
stats["point_cloud_points"] = chunk.point_cloud.point_count

# 4. Ground classification: which points are bare earth, which are trees, buildings, cars.
step("classify_ground", lambda: chunk.point_cloud.classifyGroundPoints(max_angle=15, max_distance=0.5,
                                                                         max_terrain_slope=15, cell_size=25))

# 5. DSM: the top surface, from every point.
step("dsm", lambda: chunk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.EnabledInterpolation,
                                   projection=proj))
dsm_key = chunk.elevation.key
# 6. Orthomosaic: every photo projected straight down onto the DSM.
step("orthomosaic", lambda: chunk.buildOrthomosaic(surface_data=Metashape.ElevationData,
                                                    blending_mode=Metashape.MosaicBlending, fill_holes=True,
                                                    projection=proj))
stats["orthomosaic_resolution_m"] = round(chunk.orthomosaic.resolution, 4)
stats["dsm_resolution_m"] = round(chunk.elevation.resolution, 4)
# 7. DTM: only the ground points, holes under trees and buildings interpolated.
step("dtm", lambda: chunk.buildDem(source_data=Metashape.PointCloudData, interpolation=Metashape.EnabledInterpolation,
                                   projection=proj, classes=[Metashape.PointClass.Ground]))
dtm_key = chunk.elevation.key

# 8. Exports: GeoTIFFs, classified LAZ, PDF report.
def exports():
    chunk.exportRaster(os.path.join(OUT, "orthomosaic.tif"), source_data=Metashape.OrthomosaicData, projection=proj,
                       image_compression=Metashape.ImageCompression())
    for name, key in (("dsm", dsm_key), ("dtm", dtm_key)):
        chunk.elevation = [e for e in chunk.elevations if e.key == key][0]
        chunk.exportRaster(os.path.join(OUT, f"{name}.tif"), source_data=Metashape.ElevationData, projection=proj)
    chunk.exportPointCloud(os.path.join(OUT, "point_cloud.laz"), source_data=Metashape.PointCloudData, crs=UTM,
                           save_point_color=True, save_point_classification=True)
    chunk.exportReport(os.path.join(OUT, "report.pdf"), title="Aukerman park, 77 photos")


step("exports", exports)
stats["timing_s"] = timing
stats["total_minutes"] = round(sum(timing.values()) / 60, 1)
json.dump(stats, open(os.path.join(RESULTS, "metashape_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))

os._exit(0)   # Metashape can crash while Python shuts down; all outputs are already written
