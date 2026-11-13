"""Generate the bus-cabin camera simulation video with exact ground truth.

A 40-seat BMTC-style cabin (10 rows x 2+2 seats) is rendered as seen by a camera mounted
at the FRONT of the bus, above the driver, looking back down the aisle. Real people cut out
of public sample videos (Intel IoT DevKit sample-videos, Apache-2.0) are composited into seats
and the aisle. Passengers board through the front door, walk down the aisle and sit; when all
40 seats are taken, extra passengers stand in the aisle. At each stop some passengers alight.

Because the simulator places every passenger itself, the true headcount (and which seats are
occupied) is known for every frame -> used to measure detection accuracy.

Usage:
    python scripts/make_simulation.py --cutouts data/sim_cutouts --out frontend/sim
    python -c "from backend import simulation; simulation.evaluate('bus_cabin_sim')"   # re-score
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FPS = 10
VP = (480, 105)          # vanishing point
CAM_H = 2.25             # camera height (m) - mounted high above the windscreen, tilted down
ROWS = 10
SEAT_X = [-1.55, -0.62, 0.62, 1.55]   # lateral seat centres (cabin units), aisle in the middle

# Cutout indices chosen from the contact sheet (see README): seated front-facing upper bodies
SEATED = [0, 1, 2, 3, 4, 6, 7, 8, 12, 15, 16, 20, 21, 22, 26, 27, 31, 41, 42, 46, 61, 62, 69, 71, 76, 80,
          84, 92, 96, 100, 102, 103, 104, 108, 112, 128, 133, 135, 161, 169, 170, 171, 174, 176, 186, 187,
          188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 202, 206, 209, 211, 213, 222, 228, 230, 231]
STANDING = [159, 160, 162, 163, 164, 165, 166, 167, 168, 175, 179, 180, 184, 185, 23, 24, 30, 33, 37, 40, 45, 49]

ROW_D = [1.75 + 0.46 * k for k in range(ROWS)]
DOOR_D = 1.05   # depth of the front door / entry step, closest to the camera          # depth of each row (row 0 nearest camera)


def proj(x, y, d):
    """Project cabin point (x lateral, y height above floor, depth d) to pixels."""
    f = 420
    return int(VP[0] + f * x / d), int(VP[1] + f * (CAM_H - y) / d)


def seat_number(row, col):
    """Seat number on the top-view seat map (front row = 1-4, numbered left to right as seen from
    the top with the driver on the right). The camera faces backwards, so the camera's left
    (col 0) is the bus's right-hand side (seat map column 4)."""
    return row * 4 + (3 - col) + 1


def load_cutouts(folder: Path):
    """Bundled cut-outs are named seated_XX.png / standing_XX.png (data/sim_cutouts). A raw folder
    of numbered cut-outs (as produced by YOLOv8-seg from the sample videos) is also accepted."""
    if list(folder.glob("seated_*.png")):
        seated = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in sorted(folder.glob("seated_*.png"))]
        standing = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in sorted(folder.glob("standing_*.png"))]
        return seated, standing
    files = sorted(folder.glob("*.png"))
    seated, standing = [], []
    for i, f in enumerate(files):
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if i in SEATED:
            seated.append(im)
        if i in STANDING:
            standing.append(im)
    return seated, standing


def jitter(im, rng):
    """Vary clothing colour/brightness so repeated people don't look identical."""
    bgr = im[:, :, :3].astype(np.float32)
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    h, w = im.shape[:2]
    lower = np.zeros((h, w), np.float32)
    lower[int(h * 0.62):] = 1  # clothes only, keep faces natural
    lower = cv2.GaussianBlur(lower, (1, 31), 0)
    hsv[:, :, 0] = (hsv[:, :, 0] + lower * rng.uniform(-35, 35)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + lower * rng.uniform(-0.2, 0.5)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.8, 1.15), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.5:
        out = out[:, ::-1]
        return np.dstack([out, im[:, ::-1, 3]])
    return np.dstack([out, im[:, :, 3]])


def paste(canvas, im, cx, bottom, height, clip_bottom=None):
    """Paste RGBA image scaled to `height`, centred at cx with its bottom at `bottom`."""
    h, w = im.shape[:2]
    s = height / h
    nw, nh = max(1, int(w * s)), max(1, int(height))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    x1, y1 = int(cx - nw / 2), int(bottom - nh)
    x2, y2 = x1 + nw, y1 + nh
    if clip_bottom is not None:
        y2 = min(y2, clip_bottom)
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = im[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    a = crop[:, :, 3:4].astype(np.float32) / 255
    roi = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
    canvas[cy1:cy2, cx1:cx2] = (crop[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
    return [cx1, cy1, cx2, cy2]


def draw_cabin():
    img = np.zeros((H, W, 3), np.uint8)
    far = ROW_D[-1] + 0.6
    # ceiling, walls, floor as perspective polygons
    def P(x, y, d): return proj(x, y, d)
    near = 1.2
    floor = np.array([P(-2.2, 0, near), P(2.2, 0, near), P(2.2, 0, far), P(-2.2, 0, far)])
    ceil = np.array([P(-2.2, 2.4, near), P(2.2, 2.4, near), P(2.2, 2.4, far), P(-2.2, 2.4, far)])
    lw = np.array([P(-2.2, 0, near), P(-2.2, 2.4, near), P(-2.2, 2.4, far), P(-2.2, 0, far)])
    rw = np.array([P(2.2, 0, near), P(2.2, 2.4, near), P(2.2, 2.4, far), P(2.2, 0, far)])
    cv2.fillPoly(img, [ceil], (200, 205, 205))
    cv2.fillPoly(img, [lw], (150, 160, 165))
    cv2.fillPoly(img, [rw], (150, 160, 165))
    cv2.fillPoly(img, [floor], (70, 75, 80))
    back = np.array([P(-2.2, 0, far), P(2.2, 0, far), P(2.2, 2.4, far), P(-2.2, 2.4, far)])
    cv2.fillPoly(img, [back], (120, 130, 135))
    # rear window
    cv2.fillPoly(img, [np.array([P(-1.6, 1.2, far), P(1.6, 1.2, far), P(1.6, 2.1, far), P(-1.6, 2.1, far)])], (225, 210, 180))
    # side windows (bright, like daylight)
    for d0 in np.arange(near + 0.2, far - 0.2, 0.62):
        d1 = d0 + 0.5
        for side in (-2.2, 2.2):
            cv2.fillPoly(img, [np.array([P(side, 1.25, d0), P(side, 2.05, d0), P(side, 2.05, d1), P(side, 1.25, d1)])], (230, 215, 185))
    # ceiling lights + grab rails
    for d0 in np.arange(1.0, far, 0.9):
        cv2.fillPoly(img, [np.array([P(-0.25, 2.39, d0), P(0.25, 2.39, d0), P(0.25, 2.39, d0 + 0.4), P(-0.25, 2.39, d0 + 0.4)])], (250, 250, 250))
    for x in (-0.9, 0.9):
        cv2.line(img, P(x, 2.15, near), P(x, 2.15, far), (180, 180, 175), 2)
    # aisle stripe
    cv2.fillPoly(img, [np.array([P(-0.3, 0.001, near), P(0.3, 0.001, near), P(0.3, 0.001, far), P(-0.3, 0.001, far)])], (60, 62, 66))
    return img


def draw_seat_back(img, row, col, color):
    """Seat back (the part facing the camera is the back of the seat in front, so we draw the
    seat's own back-rest just behind the passenger; legs are hidden by the nearer row)."""
    d = ROW_D[row] + 0.18
    x = SEAT_X[col]
    p1, p2 = proj(x - 0.38, 1.12, d), proj(x + 0.38, 0.45, d)
    cv2.rectangle(img, p1, p2, color, -1)
    cv2.rectangle(img, p1, (p2[0], p1[1] + max(2, (p2[1] - p1[1]) // 6)), (40, 40, 140), -1)


def draw_row_front(img, row):
    """Back-rests of row `row` as seen from the front: they hide the legs of row+1 passengers."""
    d = ROW_D[row] + 0.2
    for col in range(4):
        x = SEAT_X[col]
        p1, p2 = proj(x - 0.4, 0.78, d), proj(x + 0.4, 0.0, d)
        cv2.rectangle(img, p1, p2, (55, 55, 150), -1)  # BMTC red/maroon seats (BGR)
        cv2.rectangle(img, p1, p2, (30, 30, 90), 1)


def build_timeline(rng, stops):
    """Return list of events per stop: (stop_name, boarders, alighters)."""
    return stops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutouts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    seated_src, standing_src = load_cutouts(args.cutouts)
    print(len(seated_src), "seated cutouts,", len(standing_src), "standing cutouts")

    # A trip on route 500D-like corridor: (stop, boarding, alighting)
    stops = [("Hebbala Bridge", 12, 0), ("Nagavara", 10, 1), ("HBR Layout", 12, 3), ("Kalyana Nagara", 10, 2),
             ("Babusab Palya", 8, 2), ("KR Puram", 3, 14), ("Marathahalli", 3, 13), ("Bellandur", 2, 12),
             ("Agara", 0, 10)]

    cabin = draw_cabin()
    seats = {(r, c): None for r in range(ROWS) for c in range(4)}
    # seat preference: window seats first, spread over the bus
    order = sorted(seats, key=lambda rc: (rc[1] in (1, 2), rng.random()))
    passengers = []  # dict(kind, img, seat or stand pos, state)
    frames, gt = [], []
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_frames.avi"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    seat_px = {}
    for (r, c) in seats:
        # head position of a seated passenger in pixels (for mapping detections to seats)
        seat_px[f"{r},{c}"] = proj(SEAT_X[c], 1.25, ROW_D[r] + 0.05)
    stand_slots = [(0.0, ROW_D[k] + 0.23) for k in range(2, ROWS - 1)] + [(0.14 * (-1) ** k, ROW_D[k] + 0.0) for k in range(2, ROWS - 1)]

    def render(stop_name, phase, shake):
        img = cabin.copy()
        # draw from far to near: for each row: seat backs, seated passengers, standers at that depth, then row front
        dets = []
        standers = [p for p in passengers if p["state"] in ("stand", "walk")]
        for r in reversed(range(ROWS)):
            for c in range(4):
                draw_seat_back(img, r, c, (70, 70, 165))
            for c in range(4):
                p = seats[(r, c)]
                if p is not None and p["state"] == "seated":
                    d = ROW_D[r] + 0.05
                    cx, _ = proj(SEAT_X[c], 0, d)
                    _, bottom = proj(0, 0.45, d)
                    _, top = proj(0, 1.62, d)
                    paste(img, p["img"], cx, bottom, bottom - top)
            for p in standers:
                if ROW_D[r] - 0.1 <= p["d"] < (ROW_D[r + 1] - 0.1 if r + 1 < ROWS else 99):
                    cx, bottom = proj(p["x"], 0, p["d"])
                    _, top = proj(0, 1.7, p["d"])
                    paste(img, p["img"], cx, bottom, bottom - top)
            if r > 0:
                draw_row_front(img, r - 1)
        # people very near the camera (entering through the front door)
        for p in standers:
            if p["d"] < ROW_D[0] - 0.1:
                cx, bottom = proj(p["x"], 0, p["d"])
                _, top = proj(0, 1.7, p["d"])
                paste(img, p["img"], cx, bottom, bottom - top)
        # front of row 0 is out of view (behind camera); add slight vignette + camera shake + noise
        if shake:
            M = np.float32([[1, 0, nrng.normal(0, 1.2)], [0, 1, nrng.normal(0, 1.0)]])
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        noise = nrng.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # camera OSD
        cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(img, f"CAM-1 FRONT  KA-01-F-4101  |  {stop_name}  |  {phase}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def snapshot():
        seated = sorted(seat_number(r, c) for (r, c), p in seats.items() if p is not None and p["state"] == "seated")
        standing = sum(1 for p in passengers if p["state"] in ("stand", "walk"))
        return {"count": len(passengers), "seated": len(seated), "standing": standing, "seats": seated}

    def emit(stop_name, phase, shake=False, n=1):
        for _ in range(n):
            vw.write(render(stop_name, phase, shake))
            s = snapshot()
            s.update(t=round(len(gt) / FPS, 2), stop=stop_name, phase=phase)
            gt.append(s)

    seated_pool = seated_src[:]
    for si, (name, n_on, n_off) in enumerate(stops):
        # --- arrival / alighting
        emit(name, "arriving", shake=True, n=6)
        leavers = rng.sample(passengers, min(n_off, len(passengers)))
        # standing passengers prefer to leave first in reality; keep random
        for p in leavers:
            # passenger gets up, walks down the aisle to the front door and leaves
            start = p["d"] if p["state"] != "seated" else ROW_D[p["seat"][0]] + 0.05
            if p["state"] == "seated":
                seats[p["seat"]] = None
            p["state"], p["x"], p["d"] = "walk", 0.0, start
            p["img"] = jitter(rng.choice(standing_src), nrng)
            steps = max(5, int((start - DOOR_D) / 0.35) + 2)
            for k in range(1, steps + 1):
                p["d"] = start + (DOOR_D - start) * k / steps
                emit(name, "alighting")
            passengers[:] = [q for q in passengers if q is not p]
            emit(name, "alighting")
        # --- boarding: walk from the front door down the aisle
        for _ in range(n_on):
            free = [s for s in order if seats[s] is None]
            p = {"state": "walk", "x": 0.0, "d": DOOR_D}
            if free:
                s = free[0]
                p["img"] = jitter(rng.choice(standing_src), nrng)
                p["seat_img"] = jitter(rng.choice(seated_pool), nrng)
                p["seat"] = s
                target = ROW_D[s[0]] + 0.05
            else:
                p["img"] = jitter(rng.choice(standing_src), nrng)
                taken = {(q["x"], q["d"]) for q in passengers if q["state"] == "stand"}
                slot = next((sl for sl in stand_slots if sl not in taken), stand_slots[-1])
                p["seat"] = None
                p["slot"] = slot
                target = slot[1]
            passengers.append(p)
            steps = max(5, int((target - DOOR_D) / 0.35) + 2)
            emit(name, "boarding")
            for k in range(1, steps + 1):
                p["d"] = DOOR_D + (target - DOOR_D) * k / steps
                emit(name, "boarding")
            if p["seat"] is not None:
                seats[p["seat"]] = p
                p["state"] = "seated"
                p["img"] = p["seat_img"]
            else:
                p["state"] = "stand"
                p["x"], p["d"] = p["slot"]
            emit(name, "boarding")
        # --- doors close, bus moves
        emit(name, "departing", shake=True, n=12)
    vw.release()

    mp4 = out / "bus_cabin_sim.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-r", str(FPS), str(mp4)], check=True)
    tmp.unlink()
    meta = {
        "id": "bus_cabin_sim", "order": 1,
        "line": {"pos": 0.55, "orientation": "horizontal", "entry_direction": -1, "anchor": "top"}, "title": "Bus cabin simulation — front camera, 40 seats",
        "kind": "simulated", "fps": FPS, "width": W, "height": H, "capacity": 40,
        "camera": "Mounted at the front above the driver, looking back down the aisle",
        "source": "Rendered by scripts/make_simulation.py using people cut out of Intel IoT DevKit sample videos",
        "seat_px": seat_px, "frames": gt,
    }
    (out / "bus_cabin_sim.json").write_text(json.dumps(meta))
    print("frames", len(gt), "max count", max(g["count"] for g in gt), "->", mp4)


if __name__ == "__main__":
    main()
