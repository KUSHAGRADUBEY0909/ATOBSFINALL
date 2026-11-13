"""Bus timings between two places.

For every route that serves both places (in either direction) we build the day's departures
from the route's starting terminal:

* Timetable trips: the departure times in the ATOBS dataset (departures_from_origin /
  departures_from_destination).
* Frequency trips: BMTC runs most city routes at high frequency; we add a bus every
  10 minutes in peak hours (07:00-11:00, 17:00-21:00) and every 15 minutes otherwise
  (service 05:30-23:00). Frequency slots that fall within 5 minutes of a timetable trip are
  dropped so a timetable trip is never duplicated.

Arrival times are "live-adjusted": the scheduled running time between stops is scaled by a
traffic factor for that time of day, learned from the dataset's travel_time_between_stops
(the same observations that train the crowd model). Buses with a live camera counter on the
route are flagged with their real on-board count.
"""
from __future__ import annotations

import zlib
from datetime import datetime, timedelta
from functools import lru_cache

from . import db, ml

SERVICE_START = 5 * 60 + 30
SERVICE_END = 23 * 60
PEAKS = [(7 * 60, 11 * 60), (17 * 60, 21 * 60)]
SEATS = 40


def headway(minute_of_day: int) -> int:
    return 10 if any(a <= minute_of_day < b for a, b in PEAKS) else 15


def _hm(m: float) -> str:
    m = int(round(m)) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def _parse(t: str) -> int | None:
    try:
        h, m = t.strip().split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return None


@lru_cache(maxsize=1)
def traffic_factor() -> dict[int, float]:
    """Hour -> multiplier on scheduled running time (1.0 = average)."""
    with db.get_db() as conn:
        rows = conn.execute("""SELECT CAST(hour AS INT) h, AVG(travel_time_between_stops) v, COUNT(*) n
                               FROM trip_observations WHERE travel_time_between_stops IS NOT NULL AND hour IS NOT NULL
                               GROUP BY 1""").fetchall()
        overall = conn.execute("SELECT AVG(travel_time_between_stops) FROM trip_observations "
                               "WHERE travel_time_between_stops IS NOT NULL").fetchone()[0] or 1
    # hours with little data are noisy (e.g. a handful of late-night trips): require 300 observations
    # and clamp to a plausible range
    f = {int(r[0]): round(min(1.35, max(0.75, r[1] / overall)), 3) for r in rows if r[2] >= 300}
    return {h: f.get(h, 1.0) for h in range(24)}


def day_departures(timetable: str | None, key: str = "") -> list[tuple[int, str]]:
    """[(minute_of_day, kind)] for one direction of a route. `key` staggers the frequency slots so
    different routes from the same terminal don't all leave on the same minute."""
    tt = sorted({m for m in (_parse(x) for x in (timetable or "").split(";")) if m is not None})
    out = [(m, "timetable") for m in tt]
    m = SERVICE_START + zlib.crc32(key.encode()) % 10
    while m <= SERVICE_END:
        if all(abs(m - t) >= 5 for t in tt):
            out.append((m, f"every {headway(m)} min"))
        m += headway(m)
    return sorted(out)


def find_routes(conn, origin_place: str, dest_place: str):
    """Routes serving both places, in the direction of travel."""
    rows = conn.execute(
        """SELECT r.route_id, r.origin, r.destination, r.distance_km, r.journey_minutes, r.stop_count,
                  r.departures_from_origin, r.departures_from_destination,
                  (SELECT MIN(a.stop_sequence) FROM route_stops a JOIN stops sa USING(stop_id)
                     WHERE a.route_id=r.route_id AND sa.place=?) AS seq_o,
                  (SELECT MIN(b.stop_sequence) FROM route_stops b JOIN stops sb USING(stop_id)
                     WHERE b.route_id=r.route_id AND sb.place=?) AS seq_d
           FROM routes r
           WHERE EXISTS (SELECT 1 FROM route_stops a JOIN stops sa USING(stop_id) WHERE a.route_id=r.route_id AND sa.place=?)
             AND EXISTS (SELECT 1 FROM route_stops b JOIN stops sb USING(stop_id) WHERE b.route_id=r.route_id AND sb.place=?)""",
        (origin_place, dest_place, origin_place, dest_place)).fetchall()
    out = []
    for r in rows:
        r = dict(r)
        if r["seq_o"] is None or r["seq_d"] is None or r["seq_o"] == r["seq_d"]:
            continue
        forward = r["seq_o"] < r["seq_d"]
        n = r["stop_count"] or 2
        # stop index from the trip's starting terminal (reverse trips run the stop list backwards)
        board = r["seq_o"] - 1 if forward else n - r["seq_o"]
        alight = r["seq_d"] - 1 if forward else n - r["seq_d"]
        out.append({
            "route_id": r["route_id"], "forward": forward,
            "from_terminal": r["origin"] if forward else r["destination"],
            "towards": r["destination"] if forward else r["origin"],
            "distance_km": r["distance_km"], "journey_minutes": r["journey_minutes"] or round(n * 2.5),
            "stop_count": n, "board_index": board, "alight_index": alight, "stops_between": alight - board,
            "timetable": r["departures_from_origin"] if forward else r["departures_from_destination"],
        })
    return out


def running_minutes(route: dict, upto_index: int, start_minute: float) -> float:
    """Minutes from the terminal to stop index `upto_index`, traffic-adjusted hour by hour."""
    per_stop = route["journey_minutes"] / max(route["stop_count"] - 1, 1)
    tf = traffic_factor()
    t = start_minute
    for _ in range(upto_index):
        t += per_stop * tf[int(t // 60) % 24]
    return t - start_minute


@lru_cache(maxsize=4096)
def _forecast(hour: float, stop_count: int):
    return ml.predict(hour, stop_count)["stops"]


def journey(origin_place: str, dest_place: str, when: datetime | None = None, window: int = 120, limit: int = 40):
    now = when or datetime.now()
    now_m = now.hour * 60 + now.minute + now.second / 60
    with db.get_db() as conn:
        routes = find_routes(conn, origin_place, dest_place)
        live = {}
        for b in conn.execute("""SELECT registration, route_id, capacity, current_onboard, last_update FROM buses
                                 WHERE last_update >= datetime('now','-2 minutes')""").fetchall():
            live.setdefault(b["route_id"], []).append(dict(b))
    departures = []
    summaries = []
    for r in routes:
        deps = day_departures(r["timetable"], r["route_id"] + ("F" if r["forward"] else "R"))
        tt_count = sum(1 for d in deps if d[1] == "timetable")
        first, last = (deps[0][0], deps[-1][0]) if deps else (None, None)
        sched_per_stop = r["journey_minutes"] / max(r["stop_count"] - 1, 1)
        summaries.append({**{k: r[k] for k in ("route_id", "from_terminal", "towards", "distance_km", "stops_between",
                                                 "forward")},
                          "first_bus": _hm(first) if first is not None else None,
                          "last_bus": _hm(last) if last is not None else None,
                          "timetable_trips": tt_count, "trips_per_day": len(deps),
                          "ride_minutes": round(sched_per_stop * r["stops_between"])})
        for dep_m, kind in deps:
            # cheap pre-filter on the scheduled time before the traffic-adjusted calculation
            approx = dep_m + sched_per_stop * r["board_index"]
            if approx < now_m - 60 or approx > now_m + window + 60:
                continue
            to_board = running_minutes(r, r["board_index"], dep_m)
            eta_board = dep_m + to_board
            if eta_board < now_m - 1 or eta_board > now_m + window:
                continue
            ride = running_minutes(r, r["stops_between"], eta_board)
            sched_board = dep_m + sched_per_stop * r["board_index"]
            delay = round(eta_board - sched_board)
            hour = (eta_board / 60) % 24
            occ = None  # filled in below with one batched model call
            departures.append({
                "route_id": r["route_id"], "towards": r["towards"], "from_terminal": r["from_terminal"],
                "kind": kind, "terminal_departure": _hm(dep_m),
                "scheduled_at_stop": _hm(sched_board), "expected_at_stop": _hm(eta_board),
                "minutes_away": max(0, int(round(eta_board - now_m))),
                "delay_min": delay, "status": "on time" if abs(delay) <= 1 else (f"+{delay} min traffic" if delay > 0 else f"{delay} min early"),
                "arrive_destination": _hm(eta_board + ride), "ride_minutes": int(round(ride)),
                "stops_between": r["stops_between"],
                "_fc": (hour, r["board_index"] + 1, r["stop_count"]),
                "live_buses": live.get(r["route_id"], []),
            })
    departures.sort(key=lambda d: (d["minutes_away"], d["route_id"]))
    total = len(departures)
    departures = departures[:limit]
    for d, occ in zip(departures, ml.predict_points([d["_fc"] for d in departures])):
        d.pop("_fc")
        d.update(occupancy_pct=occ, seats_occupied=min(SEATS, round(occ * SEATS / 100)), crowd_label=ml.crowd_label(occ))
    return {
        "origin": origin_place, "destination": dest_place, "now": now.strftime("%H:%M"),
        "window_minutes": window, "routes": sorted(summaries, key=lambda s: (s["stops_between"], s["route_id"])),
        "departures": departures, "total_upcoming": total,
        "traffic_factor_now": traffic_factor()[now.hour],
        "headway_now": headway(int(now_m)) if SERVICE_START <= now_m <= SERVICE_END else None,
    }
