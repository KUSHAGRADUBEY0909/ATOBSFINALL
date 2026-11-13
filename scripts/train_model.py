"""Train the crowd/delay forecasting models from trip_observations.

Usage: python scripts/train_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from backend import db  # noqa: E402
from backend.ml import FEATURE_NAMES, MODEL_PATH, TRAFFIC, features  # noqa: E402


def main():
    conn = db.connect()
    df = pd.read_sql_query(
        """SELECT route_id, hour, is_peak_hour, traffic_level, stop_sequence, route_stop_total,
                  bus_occupancy, delay_min FROM trip_observations
           WHERE hour IS NOT NULL AND traffic_level IS NOT NULL AND bus_occupancy IS NOT NULL""",
        conn,
    )
    conn.close()
    df["delay_min"] = df["delay_min"].fillna(0)
    df["tr"] = df["traffic_level"].map(TRAFFIC)
    X = np.array([features(h, p, t, s, n) for h, p, t, s, n in
                  df[["hour", "is_peak_hour", "tr", "stop_sequence", "route_stop_total"]].itertuples(index=False)])

    # Hold out whole routes so the score reflects routes the model has not seen
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, groups=df["route_id"]))

    meta = {"name": "HistGradientBoostingRegressor", "features": FEATURE_NAMES,
            "train_rows": int(len(tr_idx)), "test_rows": int(len(te_idx)), "metrics": {}}
    models = {}
    for target in ("bus_occupancy", "delay_min"):
        y = df[target].to_numpy()
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        meta["metrics"][target] = {"mae": round(float(mean_absolute_error(y[te_idx], pred)), 3),
                                   "r2": round(float(r2_score(y[te_idx], pred)), 3)}
        m.fit(X, y)  # refit on all data for serving
        models[target] = m

    df["h"] = df["hour"].astype(int) % 24
    traffic_by_hour = df.groupby("h")["tr"].agg(lambda s: int(s.round().mode().iloc[0])).to_dict()

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"occupancy": models["bus_occupancy"], "delay": models["delay_min"],
                 "traffic_by_hour": traffic_by_hour, "meta": meta}, MODEL_PATH)
    print("Saved", MODEL_PATH)
    print(meta["metrics"])


if __name__ == "__main__":
    main()
