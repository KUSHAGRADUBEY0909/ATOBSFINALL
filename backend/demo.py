"""Live headcount demo: count people in any uploaded photo or video.

Two detection modes:
  * accurate - full frame + 2x2 zoomed tiles (finds small/far passengers), ~3-5x slower
  * fast     - one pass over the full frame

Videos normally play in the browser, which sends frames here one at a time (so you can watch
the count being calculated). Formats the browser can't play (e.g. HEVC .mov from iPhones)
fall back to server-side processing: OpenCV reads the file, samples a frame every N seconds,
and the page polls for each result as it is produced.
"""
from __future__ import annotations

import base64
import secrets
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import vision

JOBS: dict[str, dict] = {}
MAX_JOBS = 20
TMP = Path(tempfile.gettempdir()) / "atobs_demo"
TMP.mkdir(exist_ok=True)


def detect(frame: np.ndarray, mode: str = "accurate", conf: float | None = None):
    """accurate: full image + 3x2 zoomed tiles at confidence 0.25 (tuned on photos with known
    counts: exact on the real sample photos, best result on the packed simulated bus).
    fast: one pass over the full image at confidence 0.3."""
    if mode == "fast":
        return vision.detect_normalised(frame, conf or 0.3)
    return vision.detect_tiled(frame, conf or 0.25, grid=(3, 2))


def _thumb(frame: np.ndarray, width: int = 720) -> str:
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def start_video_job(data: bytes, filename: str, interval: float, mode: str) -> dict:
    # drop finished jobs beyond the cap
    if len(JOBS) >= MAX_JOBS:
        for k in sorted(JOBS, key=lambda k: JOBS[k]["created"])[: len(JOBS) - MAX_JOBS + 1]:
            JOBS.pop(k, None)
    job_id = secrets.token_hex(6)
    path = TMP / f"{job_id}{Path(filename).suffix.lower() or '.mp4'}"
    path.write_bytes(data)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        path.unlink(missing_ok=True)
        raise ValueError("This video format can't be read. Convert it to MP4 (H.264) and try again.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    job = {"id": job_id, "status": "running", "created": time.time(), "duration": round(n / fps, 2) if n else None,
           "interval": interval, "mode": mode, "results": [], "error": None, "stop": False}
    JOBS[job_id] = job

    def run():
        try:
            cap = cv2.VideoCapture(str(path))
            t = 0.0
            while not job["stop"]:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                boxes = detect(frame, mode)
                job["results"].append({
                    "t": round(t, 2), "count": len(boxes), "boxes": [[round(v, 4) for v in b] for b in boxes],
                    "ms": round((time.perf_counter() - t0) * 1000), "image": _thumb(frame)})
                t += interval
                if job["duration"] and t > job["duration"]:
                    break
            cap.release()
            job["status"] = "stopped" if job["stop"] else "done"
        except Exception as e:  # pragma: no cover
            job.update(status="error", error=str(e))
        finally:
            path.unlink(missing_ok=True)

    threading.Thread(target=run, daemon=True).start()
    return {k: v for k, v in job.items() if k not in ("results", "stop")}


def job_state(job_id: str, since: int = 0) -> dict | None:
    job = JOBS.get(job_id)
    if not job:
        return None
    return {k: v for k, v in job.items() if k not in ("results", "stop")} | {
        "results": job["results"][since:], "total": len(job["results"])}
