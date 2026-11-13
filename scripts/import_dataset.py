"""Import the ATOBS dataset (zip or extracted folder) into SQLite.

Usage:
    python scripts/import_dataset.py --source path/to/datasetfinal.zip

Nested zips (the batches inside datasetfinal.zip) are unpacked automatically.
Duplicate batch files (e.g. "Batch_3 - Copy") are skipped by file name.
"""
import argparse
import csv
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl  # noqa: E402

from backend import db  # noqa: E402
from backend.auth import hash_password  # noqa: E402


def extract_all(source: Path, dest: Path) -> Path:
    """Copy/extract the source into dest, recursively unzipping nested archives."""
    if source.is_dir():
        shutil.copytree(source, dest / source.name, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as zf:
            zf.extractall(dest / source.stem)
    changed = True
    while changed:
        changed = False
        for z in list(dest.rglob("*.zip")):
            target = z.with_suffix("")
            if not target.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(target)
                changed = True
            z.unlink()
    return dest


def parse_km(s):
    m = re.search(r"[\d.]+", s or "")
    return float(m.group()) if m else None


def parse_minutes(s):
    m = re.match(r"\s*(\d+):(\d+)", s or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def norm_times(s):
    if not s:
        return None
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    return ";".join(parts) or None


def hour_of(t):
    if not t:
        return None
    m = re.match(r"(\d{1,2}):(\d{2})", str(t))
    return int(m.group(1)) + int(m.group(2)) / 60 if m else None


def short_name(name: str) -> str:
    return name.split(",")[0].strip()


def run(source: Path):
    db.init_db()
    tmp = Path(tempfile.mkdtemp(prefix="atobs_"))
    try:
        print(f"Extracting {source} ...")
        extract_all(source, tmp)

        stop_level = next(tmp.rglob("ALL_ROUTES_STOP_LEVEL.csv"), None)
        if not stop_level:
            sys.exit("ALL_ROUTES_STOP_LEVEL.csv not found in the dataset")

        # Registry of generated stop IDs (prefer the most recent/updated one)
        registry = {}
        for reg in sorted(tmp.rglob("Stop_ID_Registry*.csv"), key=lambda p: "Updated" not in p.name):
            with open(reg, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    registry.setdefault(r["stop_name_reference"].strip(), r["stop_id"])
        next_id = max([int(v.split("_")[1]) for v in registry.values()] or [0]) + 1

        conn = db.connect()
        cur = conn.cursor()
        for t in ("route_stops", "trip_observations", "routes", "stops"):
            cur.execute(f"DELETE FROM {t}")

        stops: dict[str, str] = {}
        routes_seen = {}
        route_stop_rows = []
        print("Importing routes and stops ...")
        with open(stop_level, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                name = r["stop_name_raw"].strip()
                if name not in stops:
                    sid = registry.get(name)
                    if not sid:
                        sid = f"STOP_{next_id:05d}"
                        next_id += 1
                    stops[name] = sid
                    cur.execute(
                        "INSERT INTO stops(stop_id,name,short_name,lat,lon) VALUES (?,?,?,?,?)",
                        (sid, name, short_name(name), float(r["stop_lat_raw"] or 0) or None,
                         float(r["stop_lon_raw"] or 0) or None),
                    )
                rid = r["route_id"].strip()
                if rid not in routes_seen:
                    routes_seen[rid] = dict(
                        route_id=rid, origin=r["origin"], destination=r["destination"],
                        distance_km=parse_km(r["distance"]), journey_minutes=parse_minutes(r["journey_time"]),
                        stop_count=0, dep_o=None, dep_d=None,
                    )
                rt = routes_seen[rid]
                rt["stop_count"] += 1
                rt["dep_o"] = rt["dep_o"] or norm_times(r["departure_from_origin_source"])
                rt["dep_d"] = rt["dep_d"] or norm_times(r["departure_from_destination_source"])
                route_stop_rows.append((rid, int(r["stop_sequence"]), stops[name]))

        cur.executemany(
            """INSERT INTO routes(route_id,origin,destination,distance_km,journey_minutes,stop_count,
               departures_from_origin,departures_from_destination) VALUES (?,?,?,?,?,?,?,?)""",
            [(v["route_id"], v["origin"], v["destination"], v["distance_km"], v["journey_minutes"],
              v["stop_count"], v["dep_o"], v["dep_d"]) for v in routes_seen.values()],
        )
        cur.executemany("INSERT OR IGNORE INTO route_stops VALUES (?,?,?)", route_stop_rows)
        print(f"  {len(routes_seen)} routes, {len(stops)} stops, {len(route_stop_rows)} route-stop links")

        # ---- Stop-level observations from processed batch workbooks ----
        print("Importing trip observations from batch workbooks ...")
        seen_files = set()
        obs = []
        for x in sorted(tmp.rglob("*.xlsx")):
            if x.name in seen_files:
                continue  # duplicate copy of a batch
            seen_files.add(x.name)
            wb = openpyxl.load_workbook(x, read_only=True)
            if "Dataset" not in wb.sheetnames:
                continue
            ws = wb["Dataset"]
            it = ws.iter_rows(values_only=True)
            header = [str(h) for h in next(it)]
            idx = {h: i for i, h in enumerate(header)}
            trip_rows = []
            for v in it:
                if v[idx["trip_id"]] is None:
                    continue
                trip_rows.append(v)
            max_seq = {}
            for v in trip_rows:
                t = v[idx["trip_id"]]
                max_seq[t] = max(max_seq.get(t, 0), int(v[idx["stop_sequence"]] or 0))
            for v in trip_rows:
                g = lambda k: v[idx[k]] if k in idx else None  # noqa: E731
                name = (g("stop_name") or "").strip()
                sid = g("stop_id") or stops.get(name) or registry.get(name)
                dep = g("departure_time") or g("arrival_time")
                obs.append((
                    str(g("trip_id")), str(g("route_id")), int(g("stop_sequence") or 0), sid, name,
                    str(dep) if dep else None, hour_of(dep), g("travel_time_between_stops"),
                    int(g("is_peak_hour") or 0), g("traffic_level"), g("bus_occupancy"), g("delay_min"),
                    max_seq[g("trip_id")],
                ))
            wb.close()
        cur.executemany(
            """INSERT INTO trip_observations(trip_id,route_id,stop_sequence,stop_id,stop_name,departure_time,
               hour,travel_time_between_stops,is_peak_hour,traffic_level,bus_occupancy,delay_min,route_stop_total)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            obs,
        )
        print(f"  {len(obs)} observations from {len(seen_files)} workbooks")
        conn.commit()
        import fix_terminals  # same folder
        print("Verifying route start/end stops ...")
        fix_terminals.apply(conn)
        seed(conn)
        conn.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("Done ->", db.DB_PATH)


def seed(conn):
    """Demo accounts and a small fleet on routes that have observation data."""
    cur = conn.cursor()
    users = [
        ("ATOBS Admin", "admin@atobs.local", "admin123", "admin"),
        ("Ravi Conductor", "operator@atobs.local", "operator123", "operator"),
        ("Demo Passenger", "passenger@atobs.local", "passenger123", "passenger"),
    ]
    for name, email, pw, role in users:
        cur.execute(
            "INSERT OR IGNORE INTO users(name,email,password_hash,role) VALUES (?,?,?,?)",
            (name, email, hash_password(pw), role),
        )
    fleet_routes = [r[0] for r in cur.execute(
        """SELECT route_id FROM trip_observations GROUP BY route_id
           ORDER BY COUNT(*) DESC LIMIT 8""").fetchall()]
    for i, rid in enumerate(fleet_routes, start=1):
        first = cur.execute(
            """SELECT s.stop_id, s.lat, s.lon FROM route_stops rs JOIN stops s USING(stop_id)
               WHERE rs.route_id=? ORDER BY stop_sequence LIMIT 1""", (rid,)).fetchone()
        cur.execute(
            """INSERT OR IGNORE INTO buses(registration,route_id,capacity,last_stop_id,lat,lon)
               VALUES (?,?,?,?,?,?)""",
            (f"KA-01-F-{4100 + i}", rid, 40, first[0] if first else None,
             first[1] if first else None, first[2] if first else None),
        )
    conn.commit()
    print(f"  seeded {len(users)} users, {len(fleet_routes)} buses")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="datasetfinal.zip or extracted folder")
    args = ap.parse_args()
    run(args.source)
