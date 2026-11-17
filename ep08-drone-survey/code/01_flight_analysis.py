"""GeoAI Lab - drone survey lecture: reverse-engineer a real mapping flight from its photos.

Reads the GPS tags and camera settings of the 77 Aukerman photos and works out what the pilot planned:
height above ground, ground sample distance (GSD), photo footprint, flight lines, speed, overlap, motion blur.
Overlap is measured, not assumed: every ground point is projected into every aligned camera of the Metashape
project from the processing lecture, and we count how many photos see it.

    python 01_flight_analysis.py

Writes results/flight_stats.json and figures results/fig_*.png.
"""
import glob
import json
import math
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import ExifTags, Image
from pyproj import Transformer

ROOT = os.environ.get("GEOAI_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # folder that holds data/
DATA = os.path.join(ROOT, "data", "drone", "aukerman")
MS = os.path.join(ROOT, "data", "drone", "aukerman_metashape")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

SENSOR_W_MM, SENSOR_H_MM = 6.17, 4.55          # Sony DSC-WX220: 1/2.3-inch sensor
TAGS = {v: k for k, v in ExifTags.TAGS.items()}
to_utm = Transformer.from_crs(4326, 32617, always_xy=True)

BG, PANEL, FG, MUTED = "#0b1220", "#162034", "#f1f5f9", "#94a3b8"
ACCENT, GREEN, AMBER, PINK, PURPLE = "#38bdf8", "#4ade80", "#fbbf24", "#f472b6", "#c084fc"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#334155",
                     "font.size": 15, "axes.titlesize": 20, "axes.titleweight": "bold", "font.family": "Segoe UI"})


def exif(path):
    im = Image.open(path)
    ex = im._getexif()
    g = ex[TAGS["GPSInfo"]]
    dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    lat = dms(g[2]) * (1 if g[1] == "N" else -1)
    lon = dms(g[4]) * (1 if g[3] == "E" else -1)
    x, y = to_utm.transform(lon, lat)
    return dict(file=os.path.basename(path), x=x, y=y, alt=float(g[6]), w=im.size[0], h=im.size[1],
                f_mm=float(ex[TAGS["FocalLength"]]), exposure_s=float(ex[TAGS["ExposureTime"]]),
                fnumber=float(ex[TAGS["FNumber"]]), iso=int(ex[TAGS["ISOSpeedRatings"]]),
                t=datetime.strptime(ex[TAGS["DateTimeOriginal"]], "%Y:%m:%d %H:%M:%S"))


def sample(raster, xs, ys):
    with rasterio.open(raster) as r:
        return np.array([v[0] for v in r.sample(zip(xs, ys))], float)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=100, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def flight_lines(P, turn_deg=35):
    """Split the track where the heading turns by more than turn_deg."""
    hd = [math.degrees(math.atan2(P[i + 1]["y"] - P[i]["y"], P[i + 1]["x"] - P[i]["x"])) for i in range(len(P) - 1)]
    lines, cur = [], [0]
    for i in range(1, len(P) - 1):
        d = abs((hd[i] - hd[i - 1] + 180) % 360 - 180)
        if d > turn_deg:
            lines.append(cur)
            cur = [i]
        else:
            cur.append(i)
    cur.append(len(P) - 1)
    lines.append(cur)
    return [l for l in lines if len(l) >= 3], hd


def overlap_map(step=4.0):
    """Count, on a ground grid, how many aligned photos see each point (needs the Metashape project)."""
    import Metashape
    doc = Metashape.Document()
    doc.open(os.path.join(MS, "aukerman.psx"), read_only=True)
    ch = doc.chunk
    T, crs = ch.transform.matrix, ch.crs
    utm = Metashape.CoordinateSystem("EPSG::32617")
    with rasterio.open(os.path.join(MS, "dtm.tif")) as r:
        b = r.bounds
        xs = np.arange(b.left + step / 2, b.right, step)
        ys = np.arange(b.bottom + step / 2, b.top, step)
        X, Y = np.meshgrid(xs, ys)
        Z = np.array([v[0] for v in r.sample(zip(X.ravel(), Y.ravel()))], float).reshape(X.shape)
        nod = r.nodata
    cams = [c for c in ch.cameras if c.transform]
    Tinv = T.inv()
    cnt = np.zeros(X.shape, int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = Z[i, j]
            if nod is not None and z == nod or not np.isfinite(z) or z < -1000:
                cnt[i, j] = -1
                continue
            p = Tinv.mulp(utm.unproject(Metashape.Vector([X[i, j], Y[i, j], z])))
            n = 0
            for c in cams:
                xy = c.project(p)
                if xy is not None and 0 <= xy.x < c.sensor.width and 0 <= xy.y < c.sensor.height:
                    if (c.transform.inv().mulp(p)).z > 0:
                        n += 1
            cnt[i, j] = n
    return xs, ys, cnt


if __name__ == "__main__":
    P = sorted((exif(p) for p in glob.glob(os.path.join(DATA, "images", "*.JPG"))), key=lambda d: d["t"])
    xs, ys = np.array([p["x"] for p in P]), np.array([p["y"] for p in P])
    dtm = os.path.join(DATA, "odm_dem", "dtm.tif")                  # episode 1 (OpenDroneMap) terrain model,
    if not os.path.exists(dtm):                                      # or the Metashape one from part 2
        dtm = os.path.join(MS, "dtm.tif")
    ground = sample(dtm, xs, ys)
    # ODM's DTM may be in a different datum than the camera GPS altitude: use the median offset at take-off as a check
    agl = np.array([p["alt"] for p in P]) - ground
    f_mm, w_px, h_px = P[0]["f_mm"], P[0]["w"], P[0]["h"]
    gsd = SENSOR_W_MM * agl / (f_mm * w_px)                         # metres per pixel
    foot_w, foot_h = gsd * w_px, gsd * h_px
    t = np.array([(p["t"] - P[0]["t"]).total_seconds() for p in P])
    dt, dd = np.diff(t), np.hypot(np.diff(xs), np.diff(ys))
    brk = np.where(dt > 120)[0]                                   # a long pause = landing / battery swap
    flights = np.split(np.arange(len(P)), brk + 1)
    # a "pass" step: the next photo comes after the normal interval and roughly the normal distance
    interval = float(np.median(dt))
    step = (dt <= interval + 1.5) & (dt > 0)
    spacing = float(np.median(dd[step]))
    speed = float(np.median(dd[step] / dt[step]))
    # distance from each photo to the nearest photo taken on a different pass of the same flight
    near = []
    for f_ in flights:
        for i in f_:
            others = [j for j in f_ if abs(j - i) > 2]
            if others:
                near.append(min(math.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in others))
    pass_gap = float(np.median(near))
    blur_px = speed * P[0]["exposure_s"] / float(np.median(gsd))
    duration = t[-1]
    st = {
        "camera": "Sony DSC-WX220", "sensor_mm": [SENSOR_W_MM, SENSOR_H_MM], "image_px": [w_px, h_px], "focal_mm": f_mm,
        "photos": len(P), "date": P[0]["t"].strftime("%d %B %Y"), "span_minutes": round(duration / 60, 1),
        "flights": len(flights), "photos_per_flight": [len(f_) for f_ in flights],
        "flight_minutes_each": [round((t[f_[-1]] - t[f_[0]]) / 60, 1) for f_ in flights],
        "pause_minutes": [round(float(dt[i]) / 60, 1) for i in brk],
        "height_agl_m": {"median": round(float(np.median(agl)), 1), "min": round(float(agl.min()), 1), "max": round(float(agl.max()), 1)},
        "gsd_cm": round(100 * float(np.median(gsd)), 2),
        "footprint_m": [round(float(np.median(foot_w)), 1), round(float(np.median(foot_h)), 1)],
        "photo_spacing_m": round(spacing, 1), "photo_interval_s": round(interval, 1), "speed_m_s": round(speed, 1),
        "nearest_other_pass_m": round(pass_gap, 1),
        "exposure_s": P[0]["exposure_s"], "fnumber": P[0]["fnumber"], "iso": P[0]["iso"], "motion_blur_px": round(blur_px, 2),
    }
    st["flight_minutes"] = round(sum(st["flight_minutes_each"]), 1)
    # overlap along and across track from footprint and spacing (long side of the image across the track)
    st["front_overlap_pct_est"] = round(100 * (1 - spacing / st["footprint_m"][1]))
    st["side_overlap_pct_est"] = round(100 * (1 - pass_gap / st["footprint_m"][0]))

    # ---- figure: the flight on top of the orthomosaic
    ortho = os.path.join(MS, "orthomosaic.tif") if os.path.exists(os.path.join(MS, "orthomosaic.tif")) else \
        os.path.join(DATA, "odm_orthophoto", "odm_orthophoto.tif")
    with rasterio.open(ortho) as r:
        f = max(1, int(max(r.width, r.height) / 2000))
        img = r.read([1, 2, 3], out_shape=(3, r.height // f, r.width // f))
        if r.count >= 4:                       # transparent (no photo) areas -> page background
            al = r.read(4, out_shape=(r.height // f, r.width // f))
            img[:, al == 0] = np.array([11, 18, 32])[:, None]
        ext = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
        rcrs = r.crs.to_epsg()
    if rcrs != 32617:
        tr = Transformer.from_crs(32617, rcrs, always_xy=True)
        px, py = tr.transform(xs, ys)
    else:
        px, py = xs, ys
    fig, ax = plt.subplots(figsize=(13, 10))
    rgbimg = np.moveaxis(img, 0, -1).copy()
    ax.imshow(rgbimg, extent=ext)
    for k, f_ in enumerate(flights):
        col = [ACCENT, AMBER][k % 2]
        for i in f_[:-1]:
            ax.annotate("", xy=(px[i + 1], py[i + 1]), xytext=(px[i], py[i]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, alpha=0.9, mutation_scale=16))
        ax.plot(px[f_], py[f_], "o", color=col, ms=7, mec="white", mew=0.8,
                label=f"flight {k + 1}: {len(f_)} photos, {st['flight_minutes_each'][k]} min")
        ax.plot(px[f_[0]], py[f_[0]], "*", color="white", ms=24, mec=col, mew=1.5)
    ax.set_title(f"Two flights, {len(P)} photos: a photo every {st['photo_spacing_m']:.0f} m, loops spiralling inwards")
    ax.legend(loc="lower right", fontsize=13, facecolor=PANEL, edgecolor="#334155", labelcolor=FG)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig_flight_path.png")

    # ---- figure: overlap map (from the aligned Metashape cameras)
    if os.path.exists(os.path.join(MS, "aukerman.psx")):
        gx, gy, cnt = overlap_map()
        m = np.ma.masked_less(cnt, 0)
        fig, ax = plt.subplots(figsize=(13, 10))
        ax.imshow(np.moveaxis(img, 0, -1), extent=ext, alpha=0.35)
        cm = plt.get_cmap("viridis", 10)
        im = ax.imshow(np.ma.masked_less(m, 1), extent=(gx[0], gx[-1], gy[0], gy[-1]), origin="lower", cmap=cm,
                       vmin=0.5, vmax=10.5, alpha=0.85)
        cb = fig.colorbar(im, ax=ax, ticks=range(1, 11), shrink=0.8)
        cb.set_label("photos that see this spot")
        cb.ax.set_yticklabels([str(i) if i < 10 else "10+" for i in range(1, 11)])
        ax.set_title("Image overlap: how many photos see each spot")
        ax.set_xticks([]); ax.set_yticks([])
        save(fig, "fig_overlap_map.png")
        v = cnt[cnt >= 0]
        st["overlap_count"] = {"median": int(np.median(v[v > 0])), "pct_area_9_plus": round(100 * float((v >= 9).mean())),
                               "pct_area_under_5": round(100 * float(((v > 0) & (v < 5)).mean()))}

    # ---- figure: GSD and photos needed vs flight height (planning chart)
    H = np.arange(30, 125, 5)
    cams = [("This flight: Sony WX220 (1/2.3-inch)", SENSOR_W_MM, f_mm, w_px, AMBER),
            ("Phantom 4 RTK (1-inch, 20 MP)", 13.2, 8.8, 5472, ACCENT),
            ("Mavic 3 Enterprise (4/3, 20 MP)", 17.3, 12.29, 5280, GREEN)]
    fig, axs = plt.subplots(1, 2, figsize=(17, 7))
    area_ha, fo, so = 10, 0.75, 0.65
    for name, sw, fl, wp, c in cams:
        g = sw * H / (fl * wp) * 100
        axs[0].plot(H, g, color=c, lw=3, label=name)
        hp = wp * 3 / 4 if "WX220" not in name else h_px
        fw, fh = g / 100 * wp, g / 100 * hp
        n = (area_ha * 1e4) / ((fw * (1 - so)) * (fh * (1 - fo)))
        axs[1].plot(H, n, color=c, lw=3, label=name)
    axs[0].axvline(120, color="#f87171", ls="--", lw=2); axs[0].text(118, axs[0].get_ylim()[1] * 0.92, "CASA 120 m limit", color="#f87171", ha="right")
    axs[0].scatter([st["height_agl_m"]["median"]], [st["gsd_cm"]], s=160, color=AMBER, zorder=5, edgecolor="white")
    axs[0].set(xlabel="flight height above ground (m)", ylabel="ground sample distance (cm per pixel)", title="Higher flight, coarser pixels")
    axs[1].set(xlabel="flight height above ground (m)", ylabel="photos for a 10 ha field", title="Lower flight, many more photos")
    axs[1].set_yscale("log")
    for a in axs:
        a.grid(alpha=0.15); a.legend(fontsize=12, facecolor=PANEL, edgecolor="#334155", labelcolor=FG)
    save(fig, "fig_gsd_planning.png")

    # ---- figure: overlap geometry, two strips: along the flight and between passes
    fw, fh = st["footprint_m"]
    fig, axs = plt.subplots(2, 1, figsize=(16, 8.5))
    for ax, (step_m, size_m, pct, what, col) in zip(axs, (
            (st["photo_spacing_m"], fh, st["front_overlap_pct_est"], "Along the flight (front overlap)", ACCENT),
            (st["nearest_other_pass_m"], fw, st["side_overlap_pct_est"], "Between passes (side overlap)", GREEN))):
        for k in range(3):
            x0 = k * step_m
            ax.add_patch(plt.Rectangle((x0, k * 0.28), size_m, 0.22, color=col, alpha=0.25, lw=0))
            ax.add_patch(plt.Rectangle((x0, k * 0.28), size_m, 0.22, fill=False, color=col, lw=2))
            ax.plot(x0 + size_m / 2, k * 0.28 + 0.11, "o", color=col, ms=9)
            ax.text(x0 + 3, k * 0.28 + 0.11, f"photo {k + 1}", va="center", fontsize=13, color=FG)
        ax.annotate("", xy=(step_m, 0.95), xytext=(0, 0.95), arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2))
        ax.text(step_m / 2, 1.0, f"{step_m:.0f} m apart", ha="center", color=AMBER, fontsize=15)
        ax.annotate("", xy=(size_m, 0.95), xytext=(step_m, 0.95), arrowprops=dict(arrowstyle="<->", color=PINK, lw=2))
        ax.text((size_m + step_m) / 2, 1.0, f"shared: {size_m - step_m:.0f} m of {size_m:.0f} m = {pct}%", ha="center", color=PINK, fontsize=15)
        ax.set_xlim(-5, 2 * step_m + size_m + 5); ax.set_ylim(-0.05, 1.15)
        ax.set_title(what, loc="left", fontsize=18); ax.axis("off")
    fig.suptitle(f"One photo covers {fw:.0f} m x {fh:.0f} m of ground", fontsize=22, fontweight="bold", color=FG)
    save(fig, "fig_overlap_geometry.png")

    json.dump(st, open(os.path.join(OUT, "flight_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))
