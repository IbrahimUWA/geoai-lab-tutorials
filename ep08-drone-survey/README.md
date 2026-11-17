# Drone mapping, part 1: planning the survey

Reverse-engineers a real mapping flight from its photos: flight path, height above ground, ground
sample distance (GSD), photo footprint, speed, photo spacing, front and side overlap, motion blur, and a
measured overlap map (every ground point projected into every aligned camera). Plus planning charts
for GSD and photo count against flight height for three cameras.

Video: https://youtu.be/Smw86pLmYBI (part 2: https://youtu.be/Y_2GcpozGiE)

## Data

Aukerman sample flight published by the OpenDroneMap project (77 JPEGs, Sony DSC-WX220):

```
git clone https://github.com/OpenDroneMap/odm_data_aukerman data/drone/aukerman
```

The height above ground uses a terrain model: the OpenDroneMap DTM from episode 1
(`data/drone/aukerman/odm_dem/dtm.tif`) or the Metashape DTM from part 2. The overlap map needs the
aligned Metashape project from part 2 (`data/drone/aukerman_metashape/aukerman.psx`), so run part 2's
`01_metashape_process.py` first. Paths are relative to the repository root; set `GEOAI_ROOT` to change.

```
pip install rasterio pyproj matplotlib numpy pillow      # plus the Metashape module for the overlap map
python code/01_flight_analysis.py
```

## What the flight turned out to be

| | |
|---|---|
| Flights | 2 (33 + 44 photos, about 4 minutes each, 9-minute pause) |
| Pattern | loops spiralling inwards, not a lawnmower grid |
| Height above ground | about 101 m, GSD 2.9 cm |
| Photo spacing / interval / speed | 43 m / 4 s / 11 m/s |
| Front / side overlap (estimated) | 59% / 82% |
| Measured overlap | 10+ photos in the middle, a quarter of the area under 5 |
| Motion blur | 1.5 px at 1/250 s (target: under 0.5 px) |

## Formulas

- GSD = sensor width x height / (focal length x image width in pixels)
- Blur (px) = ground speed x exposure time / GSD
- Overlap = 1 - spacing / footprint length in that direction

Australian rules (CASA) change: check https://www.casa.gov.au before flying.
