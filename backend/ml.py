"""Crowd (bus occupancy %) and delay forecasting trained on the ATOBS trip observations."""
import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crowd_model.joblib"
TRAFFIC = {"Low": 0, "Medium": 1, "High": 2}
TRAFFIC_NAMES = {v: k for k, v in TRAFFIC.items()}
PEAK_HOURS = [(8, 11), (17, 20)]
SEATS = 40  # BMTC ordinary bus: 40 seats  # matches the is_peak_hour pattern in the dataset


def is_peak(hour: float) -> int:
    return int(any(a <= hour < b for a, b in PEAK_HOURS))


def features(hour, peak, traffic, seq, total):
    pos = (seq - 1) / max(total - 1, 1)
    ang = 2 * math.pi * (hour / 24)
    return [math.sin(ang), math.cos(ang), peak, traffic, pos, total]


FEATURE_NAMES = ["hour_sin", "hour_cos", "is_peak_hour", "traffic_level", "stop_position", "route_stop_total"]


@lru_cache(maxsize=1)
def load():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def typical_traffic(hour: float) -> int:
    bundle = load()
    if bundle and "traffic_by_hour" in bundle:
        return bundle["traffic_by_hour"].get(int(hour) % 24, 1)
    return 2 if is_peak(hour) else 1


def predict(hour: float, stop_count: int, traffic: int | None = None):
    """Return per-stop [{seq, occupancy, delay}] for a route with stop_count stops at the given hour."""
    bundle = load()
    tr = typical_traffic(hour) if traffic is None else traffic
    peak = is_peak(hour)
    X = np.array([features(hour, peak, tr, s, stop_count) for s in range(1, stop_count + 1)])
    if bundle is None:  # heuristic fallback if model not trained yet
        base = 40 + 20 * peak + 8 * tr
        occ = [base + 10 * math.sin(math.pi * x[4]) for x in X]
        dly = [0.3 + 0.4 * tr for _ in X]
    else:
        occ = bundle["occupancy"].predict(X)
        dly = bundle["delay"].predict(X)
    return {
        "traffic_level": TRAFFIC_NAMES[tr],
        "is_peak_hour": bool(peak),
        "stops": [
            {"seq": i + 1, "occupancy": int(round(float(np.clip(o, 0, 100)))),
             "seats_occupied": int(round(float(np.clip(o, 0, 100)) * SEATS / 100)),
             "delay_min": round(max(float(d), 0), 1)}
            for i, (o, d) in enumerate(zip(occ, dly))
        ],
        "model": bundle["meta"] if bundle else {"name": "heuristic"},
    }


def crowd_label(pct: float) -> str:
    if pct < 40:
        return "Seats available"
    if pct < 65:
        return "Moderate"
    if pct < 85:
        return "Crowded"
    return "Very crowded"


def predict_points(points):
    """Batch occupancy forecast for [(hour, stop_seq, stop_count)] -> list of int % (one model call)."""
    if not points:
        return []
    bundle = load()
    X = np.array([features(h, is_peak(h), typical_traffic(h), seq, n) for h, seq, n in points])
    if bundle is None:
        return [int(round(40 + 20 * x[2] + 8 * x[3])) for x in X]
    return [int(round(float(np.clip(o, 0, 100)))) for o in bundle["occupancy"].predict(X)]
