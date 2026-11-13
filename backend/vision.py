"""Camera-based passenger detection, tracking and door-line counting.

Detector: YOLOv8n (COCO class 0 = person) via ultralytics. If ultralytics/torch is not
installed, falls back to OpenCV's built-in HOG pedestrian detector so the feature still works.

Counting: each person gets a track ID (centroid + IoU tracker). A virtual counting line is
drawn across the door. When a track's centre crosses the line in the "entry" direction it is
counted IN (boarding); crossing the other way counts OUT (alighting).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"


class Detector:
    def __init__(self):
        self.lock = threading.Lock()
        self.name = "hog"
        self.model = None
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(str(MODEL_FILE) if MODEL_FILE.exists() else "yolov8n.pt")
            self.name = "yolov8n"
        except Exception as e:  # pragma: no cover - fallback path
            print(f"[vision] YOLO unavailable ({e}); using OpenCV HOG detector")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray, conf: float = 0.35) -> list[tuple[float, float, float, float, float]]:
        """Return person boxes as (x1, y1, x2, y2, score) in pixel coordinates."""
        with self.lock:
            if self.model is not None:
                res = self.model.predict(frame, classes=[0], conf=conf, imgsz=640, verbose=False)[0]
                out = []
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    out.append((x1, y1, x2, y2, float(b.conf[0])))
                return out
            h, w = frame.shape[:2]
            scale = 640 / max(w, h) if max(w, h) > 640 else 1.0
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame
            rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            out = []
            for (x, y, bw, bh), s in zip(rects, np.ravel(weights) if len(rects) else []):
                if s < 0.5:
                    continue
                out.append((x / scale, y / scale, (x + bw) / scale, (y + bh) / scale, float(min(s / 2, 1))))
            return out


_detector: Detector | None = None
_det_lock = threading.Lock()


def get_detector() -> Detector:
    global _detector
    with _det_lock:
        if _detector is None:
            _detector = Detector()
        return _detector


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    tid: int
    box: tuple
    cx: float
    cy: float
    side: int | None = None  # which side of the line the track is on (-1 / +1)
    missed: int = 0
    hits: int = 1
    counted: str | None = None
    counted_at: int = 0


@dataclass
class CountingSession:
    """Per-camera state. Coordinates are normalised to 0..1 so frame size can vary."""
    session_id: str
    bus_id: int
    onboard: int = 0
    line_pos: float = 0.5          # position of the line (fraction of width/height)
    orientation: str = "horizontal"  # 'horizontal' line (people move up/down) or 'vertical'
    entry_direction: int = 1       # +1: crossing towards larger coordinate = IN
    anchor: str = "center"         # point of the box tested against the line: 'center' or 'top' (head)
    capacity: int = 40
    tracks: dict = field(default_factory=dict)
    next_id: int = 1
    total_in: int = 0
    total_out: int = 0
    max_in_frame: int = 0
    frames: int = 0
    last_seen: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    MAX_MISSED = 5
    MATCH_DIST = 0.09
    MAX_JUMP = 0.12    # max movement of the box top between processed frames (fraction of height)
    UNDO_WINDOW = 30   # frames: crossing back sooner than this cancels the count (hesitation)
    MIN_HITS = 2       # a track must be seen this many times before it can be counted
    DEAD_ZONE = 0.02  # hysteresis around the line to avoid jitter double-counts

    def _axis(self, cx, cy, top=None):
        if self.orientation == "horizontal":
            return top if (self.anchor == "top" and top is not None) else cy
        return cx

    def _side(self, v):
        if v > self.line_pos + self.DEAD_ZONE:
            return 1
        if v < self.line_pos - self.DEAD_ZONE:
            return -1
        return 0

    def update(self, boxes_norm: list[tuple]) -> list[dict]:
        """Feed normalised detections; returns list of crossing events [{'direction','track'}]."""
        with self.lock:
            self.frames += 1
            self.last_seen = time.time()
            dets = [(b, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes_norm]
            unmatched = set(range(len(dets)))
            pairs = []
            for tid, t in self.tracks.items():
                for i, (b, cx, cy) in enumerate(dets):
                    dist = ((cx - t.cx) ** 2 + (cy - t.cy) ** 2) ** 0.5
                    iou = _iou(t.box, b)
                    # gate by size similarity too: a passing walker must not inherit a seated passenger's track
                    ah = max(t.box[3] - t.box[1], 1e-6)
                    bh = max(b[3] - b[1], 1e-6)
                    if min(ah, bh) / max(ah, bh) < 0.6:
                        continue
                    # a head cannot jump further than this between consecutive frames
                    if abs(b[1] - t.box[1]) > self.MAX_JUMP:
                        continue
                    if iou > 0.3 or dist < self.MATCH_DIST:
                        pairs.append((iou - 2 * dist, tid, i))
            pairs.sort(reverse=True)
            used_t = set()
            events = []
            for _, tid, i in pairs:
                if tid in used_t or i not in unmatched:
                    continue
                used_t.add(tid)
                unmatched.discard(i)
                t = self.tracks[tid]
                b, cx, cy = dets[i]
                if t.missed > 0 and t.counted is not None:
                    # re-acquired after a gap: likely a different person at the same spot (e.g. one
                    # passenger left through the door and the next one stepped in) -> fresh count state
                    t.counted, t.hits, t.side = None, 1, None
                t.box, t.cx, t.cy, t.missed = b, cx, cy, 0
                t.hits += 1
                s = self._side(self._axis(cx, cy, b[1]))
                if s != 0:
                    if t.side is not None and s != t.side and t.hits >= self.MIN_HITS:
                        direction = "in" if s == self.entry_direction else "out"
                        if t.counted is not None and t.counted != direction and \
                                self.frames - t.counted_at > self.UNDO_WINDOW:
                            t.counted = None  # long after the first crossing: a genuine opposite move
                        if t.counted is None:
                            events.append(self._count(direction, tid))
                            t.counted, t.counted_at = direction, self.frames
                        elif t.counted != direction:
                            # crossed back within the same track (hesitation at the door): undo
                            self._undo(t.counted)
                            events.append({"direction": "undo-" + t.counted, "track": tid, "onboard_after": self.onboard})
                            t.counted = None
                    t.side = s
            for tid in list(self.tracks):
                if tid not in used_t:
                    self.tracks[tid].missed += 1
                    if self.tracks[tid].missed > self.MAX_MISSED:
                        del self.tracks[tid]
            for i in unmatched:
                b, cx, cy = dets[i]
                s = self._side(self._axis(cx, cy, b[1]))
                self.tracks[self.next_id] = Track(self.next_id, b, cx, cy, side=s or None)
                self.next_id += 1
            self.max_in_frame = max(self.max_in_frame, len(dets))
            return events

    def _undo(self, direction):
        if direction == "in":
            self.total_in -= 1
            self.onboard = max(0, self.onboard - 1)
        else:
            self.total_out -= 1
            self.onboard += 1

    def _count(self, direction, tid):
        if direction == "in":
            self.total_in += 1
            self.onboard += 1
        else:
            self.total_out += 1
            self.onboard = max(0, self.onboard - 1)
        return {"direction": direction, "track": tid, "onboard_after": self.onboard}

    def snapshot(self, in_frame: int) -> dict:
        return {
            "session_id": self.session_id, "bus_id": self.bus_id, "in_frame": in_frame,
            "total_in": self.total_in, "total_out": self.total_out, "onboard": self.onboard,
            "capacity": self.capacity,
            "occupancy_pct": int(round(100 * self.onboard / self.capacity)) if self.capacity else None,
            "seats_occupied": min(self.onboard, self.capacity), "standing": max(0, self.onboard - self.capacity),
            "max_in_frame": self.max_in_frame, "frames": self.frames,
            "line": {"pos": self.line_pos, "orientation": self.orientation, "entry_direction": self.entry_direction,
                     "anchor": self.anchor},
        }

    def track_list(self):
        return [{"id": t.tid, "box": [round(v, 4) for v in t.box], "counted": t.counted}
                for t in self.tracks.values() if t.missed == 0]


def detect_tiled(frame: np.ndarray, conf: float = 0.3, grid=(2, 2), overlap: float = 0.25):
    """Headcount mode: full frame + overlapping zoomed tiles, merged.

    Small/far passengers (rear rows seen by a front camera) are only a few pixels tall at the
    detector's input size; running it on zoomed tiles raises their resolution. Merging rules:
      * a tile box touching an interior tile edge is dropped (the person is cut by the tile and is
        found whole in a neighbouring tile or the full frame);
      * a tile box lying inside a much taller full-frame box is dropped (it is the head/shoulders of
        a person the full-frame pass already found);
      * remaining boxes go through NMS and a containment check.
    Returns normalised (x1, y1, x2, y2, score)."""
    det = get_detector()
    h, w = frame.shape[:2]
    if det.model is None:  # HOG fallback: no tiling
        return detect_normalised(frame, conf)
    tiles = [(0, 0, frame)]
    gx, gy = grid
    tw, th = w / (gx - (gx - 1) * overlap), h / (gy - (gy - 1) * overlap)
    for j in range(gy):
        for i in range(gx):
            x0, y0 = int(i * tw * (1 - overlap)), int(j * th * (1 - overlap))
            tiles.append((x0, y0, frame[y0:y0 + int(th), x0:x0 + int(tw)]))
    with det.lock:
        results = det.model.predict([t[2] for t in tiles], classes=[0], conf=conf, imgsz=640, verbose=False)
    full, part = [], []
    for k, ((x0, y0, t), r) in enumerate(zip(tiles, results)):
        t_h, t_w = t.shape[:2]
        mx, my = 0.015 * t_w, 0.015 * t_h
        for b, sc in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            if k == 0:
                full.append((b, sc))
                continue
            if ((b[0] < mx and x0 > 0) or (b[2] > t_w - mx and x0 + t_w < w - 2)
                    or (b[1] < my and y0 > 0) or (b[3] > t_h - my and y0 + t_h < h - 2)):
                continue
            part.append(([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], sc))

    def inside(a, B):
        ix = max(0, min(a[2], B[2]) - max(a[0], B[0]))
        iy = max(0, min(a[3], B[3]) - max(a[1], B[1]))
        area = (a[2] - a[0]) * (a[3] - a[1])
        return ix * iy / area if area else 0

    part = [(b, sc) for b, sc in part
            if not any(inside(b, F) > 0.8 and (F[3] - F[1]) > 1.4 * (b[3] - b[1]) for F, _ in full)]
    boxes = [b for b, _ in full] + [b for b, _ in part]
    scores = [sc for _, sc in full] + [sc for _, sc in part]
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], scores, conf, 0.45)
    keep = sorted([int(i) for i in np.array(keep).flatten()], key=lambda i: -scores[i])
    out = []
    for i in keep:  # containment suppression (a head-and-shoulders box inside a full-body box)
        bi = boxes[i]
        ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
        dup = False
        for k in out:
            bk = boxes[k]
            ix = max(0, min(bi[2], bk[2]) - max(bi[0], bk[0]))
            iy = max(0, min(bi[3], bk[3]) - max(bi[1], bk[1]))
            ak = (bk[2] - bk[0]) * (bk[3] - bk[1])
            if ix * iy > 0.8 * min(ai, ak) and min(ai, ak) / max(ai, ak) > 0.35:
                dup = True
                break
        if not dup:
            out.append(i)
    return [(boxes[i][0] / w, boxes[i][1] / h, boxes[i][2] / w, boxes[i][3] / h, float(scores[i])) for i in out]


def warm_up():
    """Load the model and run one inference so the first real request is fast."""
    det = get_detector()
    if det.model is not None:
        det.detect(np.zeros((360, 640, 3), np.uint8))


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_normalised(frame: np.ndarray, conf: float = 0.35):
    h, w = frame.shape[:2]
    boxes = get_detector().detect(frame, conf)
    return [(x1 / w, y1 / h, x2 / w, y2 / h, s) for x1, y1, x2, y2, s in boxes]
