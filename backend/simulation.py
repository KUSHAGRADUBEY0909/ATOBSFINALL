"""Camera simulation: run the passenger-counting pipeline over a test clip and score it
against ground truth.

Each clip lives in frontend/sim/ as <id>.mp4 + <id>.json (metadata + ground truth).
Ground truth is either per-frame ("frames": [{t, count, ...}]) for the rendered bus-cabin
simulation, or sparse ("annotations": [{t, count}]) for real footage annotated by hand.
User-uploaded clips have no ground truth until someone adds manual checks ("manual_gt").

Results are written to <id>.results.json and served to the Simulation page, which plays the
video with the detections overlaid in sync and shows detected vs. actual counts.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import vision

SIM_DIR = Path(__file__).resolve().parent.parent / "frontend" / "sim"
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def list_clips() -> list[dict]:
    clips = []
    for meta_path in sorted(SIM_DIR.glob("*.json")):
        if meta_path.name.endswith(".results.json"):
            continue
        meta = json.loads(meta_path.read_text())
        res_path = SIM_DIR / f"{meta['id']}.results.json"
        summary = None
        if res_path.exists():
            summary = json.loads(res_path.read_text()).get("metrics")
        clips.append({k: meta.get(k) for k in ("id", "title", "kind", "camera", "source", "capacity", "fps",
                                                "width", "height", "line", "order")}
                     | {"video": f"/sim/{meta.get('video_file') or meta['id'] + '.mp4'}", "has_gt": bool(gt_series(meta)),
                        "manual_checks": len(meta.get("manual_gt") or []), "uploaded": bool(meta.get("uploaded")),
                        "metrics": summary, "job": JOBS.get(meta["id"])})
    clips.sort(key=lambda c: (c.get("order") or 99, c["title"]))
    return clips


def load_meta(clip_id: str) -> dict:
    p = SIM_DIR / f"{clip_id}.json"
    if not p.exists() or not clip_id.replace("_", "").replace("-", "").isalnum():
        raise FileNotFoundError(clip_id)
    return json.loads(p.read_text())


def save_meta(meta: dict):
    (SIM_DIR / f"{meta['id']}.json").write_text(json.dumps(meta))


def gt_series(meta: dict) -> list[dict]:
    """Ground truth points [{t, count}] from per-frame, annotated or manual checks."""
    if meta.get("frames"):
        return [{"t": f["t"], "count": f["count"]} for f in meta["frames"]]
    pts = list(meta.get("annotations") or []) + list(meta.get("manual_gt") or [])
    return sorted(pts, key=lambda p: p["t"])


def _gt_at(series, t, tol):
    if not series:
        return None
    ts = [p["t"] for p in series]
    i = int(np.searchsorted(ts, t))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(series) and abs(series[j]["t"] - t) <= tol:
            if best is None or abs(series[j]["t"] - t) < abs(best["t"] - t):
                best = series[j]
    return best["count"] if best else None


def _metrics(pairs):
    """pairs: list of (detected, actual)."""
    if not pairs:
        return None
    d = np.array([p[0] for p in pairs], float)
    a = np.array([p[1] for p in pairs], float)
    e = d - a
    tot = a.sum()
    return {
        "samples": int(len(pairs)),
        "mae": round(float(np.abs(e).mean()), 2),
        "bias": round(float(e.mean()), 2),
        "exact_pct": int(round(100 * float((e == 0).mean()))),
        "within1_pct": int(round(100 * float((np.abs(e) <= 1).mean()))),
        "within2_pct": int(round(100 * float((np.abs(e) <= 2).mean()))),
        "accuracy_pct": int(round(100 * max(0.0, 1 - float(np.abs(e).sum()) / tot))) if tot else None,
    }


def evaluate(clip_id: str, head_every: int | None = None, progress=None) -> dict:
    meta = load_meta(clip_id)
    video = SIM_DIR / (meta.get("video_file") or f"{clip_id}.mp4")
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or meta.get("fps") or 10
    # process at ~10 frames/s for tracking; headcount (tiled, slower) a few times per second
    step = max(1, round(fps / 10))
    head_every = head_every or max(1, round(fps / (3 * step)))
    line = meta.get("line") or {"pos": 0.74, "orientation": "horizontal", "entry_direction": -1}
    sess = vision.CountingSession(f"sim-{clip_id}", 0, line_pos=line["pos"], orientation=line["orientation"],
                                  entry_direction=line["entry_direction"], anchor=line.get("anchor", "center"), capacity=meta.get("capacity") or 40)
    series = gt_series(meta)
    tol = 0.5 / fps * step + 1e-6
    sparse = not meta.get("frames")
    frames_out, head_pairs, onboard_pairs = [], [], []
    idx = 0
    k = 0
    last_head = None
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        t = round(idx / fps, 2)
        boxes = vision.detect_normalised(frame, 0.3)
        events = sess.update(boxes)
        rec = {"t": t, "boxes": [[round(v, 3) for v in b[:4]] for b in boxes], "in": sess.total_in,
               "out": sess.total_out, "onboard": sess.onboard,
               "events": [e["direction"] for e in events]}
        gt = _gt_at(series, t, tol)
        if k % head_every == 0 or (sparse and gt is not None):
            hb = vision.detect_tiled(frame, 0.3)
            last_head = len(hb)
            rec["head"] = last_head
            rec["head_boxes"] = [[round(v, 3) for v in b[:4]] for b in hb]
        if gt is not None:
            rec["gt"] = gt
            if "head" in rec:
                head_pairs.append((rec["head"], gt))
            if meta.get("door_counting", True) and meta.get("frames"):
                onboard_pairs.append((sess.onboard, gt))
        frames_out.append(rec)
        idx += 1
        k += 1
        if progress and k % 10 == 0:
            progress(min(99, int(100 * idx / n)))
    cap.release()
    gt_in = gt_out = None
    if meta.get("frames"):
        counts = [f["count"] for f in meta["frames"]]
        diffs = np.diff([0] + counts)
        gt_in, gt_out = int(diffs[diffs > 0].sum()), int(-diffs[diffs < 0].sum())
    metrics = {
        "detector": vision.get_detector().name,
        "processed_frames": len(frames_out),
        "seconds": round(time.time() - t0, 1),
        "headcount": _metrics(head_pairs),
        "door_onboard": _metrics(onboard_pairs) if meta.get("door_counting", True) else None,
        "boarded": {"detected": sess.total_in, "actual": gt_in},
        "alighted": {"detected": sess.total_out, "actual": gt_out},
        "has_gt": bool(series),
    }
    result = {"id": clip_id, "fps": fps, "step": step, "line": line, "metrics": metrics, "frames": frames_out}
    (SIM_DIR / f"{clip_id}.results.json").write_text(json.dumps(result))
    return result


def start_job(clip_id: str) -> dict:
    load_meta(clip_id)  # validates
    with _lock:
        job = JOBS.get(clip_id)
        if job and job["status"] == "running":
            return job
        job = {"status": "running", "progress": 0, "started": time.time(), "error": None}
        JOBS[clip_id] = job

    def run():
        try:
            evaluate(clip_id, progress=lambda p: job.update(progress=p))
            job.update(status="done", progress=100)
        except Exception as e:  # pragma: no cover
            job.update(status="error", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return job


LINES = {
    # camera at the front of the cabin looking back: people walk away from the camera when boarding
    "front": {"pos": 0.55, "orientation": "horizontal", "entry_direction": -1, "anchor": "top"},
    # camera above the door looking down: people move down the image when boarding
    "door": {"pos": 0.5, "orientation": "horizontal", "entry_direction": 1, "anchor": "center"},
}


def add_upload(data: bytes, filename: str, title: str, camera: str) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in (".mp4", ".webm", ".mov", ".m4v"):
        raise ValueError("Upload an .mp4, .webm or .mov video")
    clip_id = "up_" + secrets.token_hex(4)
    (SIM_DIR / f"{clip_id}{ext}").write_bytes(data)
    cap = cv2.VideoCapture(str(SIM_DIR / f"{clip_id}{ext}"))
    if not cap.isOpened() or not cap.read()[0]:
        (SIM_DIR / f"{clip_id}{ext}").unlink(missing_ok=True)
        raise ValueError("Could not read that video")
    meta = {"id": clip_id, "title": title, "kind": "uploaded", "uploaded": True, "order": 50,
            "video_file": f"{clip_id}{ext}", "fps": cap.get(cv2.CAP_PROP_FPS) or 25,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "camera": "Front of cabin, looking back" if camera == "front" else "Above the door, looking down",
            "source": "Uploaded by staff", "capacity": 40, "line": LINES.get(camera, LINES["front"]),
            "manual_gt": []}
    cap.release()
    save_meta(meta)
    return meta


def add_manual_gt(clip_id: str, t: float, count: int) -> dict:
    """Store a hand-checked headcount at time t together with the detector's count for that frame."""
    meta = load_meta(clip_id)
    cap = cv2.VideoCapture(str(SIM_DIR / (meta.get("video_file") or f"{clip_id}.mp4")))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError("frame")
    detected = len(vision.detect_tiled(frame, 0.3))
    checks = [c for c in (meta.get("manual_gt") or []) if abs(c["t"] - t) > 0.05]
    checks.append({"t": round(t, 2), "count": count, "detected": detected})
    meta["manual_gt"] = sorted(checks, key=lambda c: c["t"])
    save_meta(meta)
    m = _metrics([(c["detected"], c["count"]) for c in meta["manual_gt"] if "detected" in c])
    return {"checks": meta["manual_gt"], "metrics": m, "detected": detected}
