"""Verify and repair route start/end stops (terminals) and clean place names.

Run automatically at the end of scripts/import_dataset.py, or on its own:
    python scripts/fix_terminals.py

What it does (every change is written to the data_audit table and data/terminal_audit.csv):

1. Verified terminals. The busiest terminals (covering ~half of all route ends) are checked
   against published coordinates (Wikipedia / OpenStreetMap / latlong.net, listed in
   data/terminal_reference.csv). A route end that is more than 1 km from the verified point is
   repaired: if the end stop has the terminal's name its coordinates are corrected; if the
   route's stop list actually starts/ends somewhere else, the route label is corrected to the
   stop the bus really starts/ends at.
2. Unverified terminals. A route end whose coordinates are far (>3 km) from the route's own
   next stop while other routes put the same-named terminal much closer is treated as a bad
   coordinate and re-pointed to the consensus location.
3. Name variants of the same place ("BANASHANKARI TTMC" / "Banashankari Bus Station",
   "K R Puram" / "KR Puram", "Yeshwanthpura" / "Yeshwanthpura TTMC", ...) are merged when they
   are within 1.5 km of each other, so the trip planner offers one clean name per place.
"""
import csv
import math
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REF_CSV = ROOT / "data" / "terminal_reference.csv"
AUDIT_CSV = ROOT / "data" / "terminal_audit.csv"

# Published coordinates for the busiest terminals (checked Nov 2026).
REFERENCE = [
    # canonical name, aliases (lower-case), lat, lon, source
    ("Kempegowda Bus Station", ["kempegowda bus station", "majestic", "kbs"], 12.9781534, 77.5724250,
     "latlong.net POI (OSM) / Wikipedia 'Kempegowda Bus Station'"),
    ("Krishnarajendra Market", ["krishnarajendra market", "kr market", "k r market"], 12.961353, 77.574571,
     "Wikipedia 'Krishna Rajendra Market metro station' (bus stand ~0.4 km NE)"),
    ("Shivajinagar Bus Station", ["shivajinagar bus station", "shivajinagara bus station"], 12.9834865, 77.6031560,
     "OpenStreetMap node 596446558"),
    ("Banashankari TTMC", ["banashankari bus station", "banashankari ttmc"], 12.9177649, 77.5729950,
     "latlong.net POI 'Banashankari TTMC'; Wikipedia lists 'Banashankari Bus Station' as the same place"),
    ("Kengeri TTMC", ["kengeri ttmc", "kengeri bus station"], 12.9148975, 77.4875928,
     "Wikipedia 'Kengeri Bus Terminal metro station' (connects to Kengeri TTMC)"),
    ("Central Silk Board", ["central silk board"], 12.9159764, 77.6237596, "Silk Board junction coordinates"),
    ("Hampinagara RPC Layout", ["hampinagara rpc layout", "hampinagara"], 12.9589791, 77.5354787,
     "latlong.net POI 'Hampinagara Bus Station'"),
    ("Mysore Road Satellite Bus Station", ["mctc bus stand", "mysore road satellite bus station", "mctc"],
     12.9530778, 77.5436611, "Wikipedia 'Mysuru Road Bus Station'"),
    ("Hebbala Bridge", ["hebbala bridge", "hebbala"], 13.0425128, 77.5900176, "latlong.net POI 'Hebbal Bus Station'"),
    ("Domlur TTMC Bus Stand", ["domlur ttmc bus stand", "domlur"], 12.9618006, 77.6387125,
     "latlong.net POI 'Domlur Bus Station'"),
    ("Nelamangala", ["nelamangala"], 13.0987, 77.3936, "Nelamangala town centre (bus stand is in the town centre)"),
    ("Yelahanka", ["yelahanka"], 13.1007, 77.5963, "Yelahanka town coordinates"),
]
VERIFY_KM = 1.0


def hav(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = p2 - p1, math.radians(d - b)
    return 2 * 6371 * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def norm(s: str) -> str:
    s = s.lower()
    for w in ("bus station", "bus stand", "busstand", "ttmc", "bus stop"):
        s = s.replace(w, "")
    return re.sub(r"[^a-z0-9]", "", s)


def nice(name: str) -> str:
    """Title-case names that are entirely upper case (e.g. 'YELAHANKA')."""
    if name.isupper():
        return " ".join(w if len(w) <= 3 and w.isalpha() and w in ("KR", "HAL", "BEL", "MG", "HSR", "BTM", "ITPL") else w.title()
                        for w in name.split())
    return name


def apply(conn: sqlite3.Connection, verbose=True):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS data_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, route_id TEXT, field TEXT, old TEXT, new TEXT,
            reason TEXT, source TEXT);
        DELETE FROM data_audit;
    """)
    for table, col in (("stops", "place"), ("routes", "verified_terminals")):
        if col not in [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
    audit = []

    def log(rid, field, old, new, reason, source=""):
        audit.append((rid, field, str(old), str(new), reason, source))

    def route_ends(rid):
        rows = cur.execute("""SELECT rs.stop_sequence, s.stop_id, s.short_name, s.lat, s.lon FROM route_stops rs
                              JOIN stops s USING(stop_id) WHERE route_id=? ORDER BY stop_sequence""", (rid,)).fetchall()
        return rows

    alias = {}
    for name, al, lat, lon, src in REFERENCE:
        for a in al:
            alias[a] = (name, lat, lon, src)

    routes = cur.execute("SELECT route_id, origin, destination FROM routes").fetchall()

    # ---- 1. verified terminals
    verified_count = 0
    for rid, origin, dest in routes:
        stops = route_ends(rid)
        if not stops:
            continue
        flags = []
        for field, label, st in (("origin", origin, stops[0]), ("destination", dest, stops[-1])):
            ref = alias.get(label.lower().strip())
            if not ref:
                continue
            name, lat, lon, src = ref
            d = hav(st[3], st[4], lat, lon) if st[3] else 99
            if d <= VERIFY_KM:
                flags.append(field)
                verified_count += 1
                if label != name:
                    cur.execute(f"UPDATE routes SET {field}=? WHERE route_id=?", (name, rid))
                    log(rid, field, label, name, "name unified with verified terminal", src)
                continue
            nb = stops[1] if field == "origin" else stops[-2]
            same_name = norm(st[2]) == norm(label) or alias.get(st[2].lower()) == ref
            if same_name and len(stops) > 1 and nb[3] and hav(nb[3], nb[4], st[3], st[4]) < hav(nb[3], nb[4], lat, lon) / 3:
                # the route's own neighbouring stop agrees with the current location -> this is a
                # different place that happens to share the name (e.g. Dommaluru village vs Domlur TTMC)
                full = cur.execute("SELECT name FROM stops WHERE stop_id=?", (st[1],)).fetchone()[0]
                parts = [p.strip() for p in full.split(",")]
                loc = nice(parts[1]) if len(parts) > 1 and norm(parts[1]) != norm(label) else "outskirts"
                new = f"{label} ({loc})"
                cur.execute(f"UPDATE routes SET {field}=? WHERE route_id=?", (new, rid))
                log(rid, field, label, new, f"same name as {name} but {d:.0f} km away and consistent with the route's "
                    f"own stops: a different place, label disambiguated", src)
                continue
            if norm(st[2]) == norm(label) or alias.get(st[2].lower()) == ref:
                cur.execute("UPDATE stops SET lat=?, lon=? WHERE stop_id=?", (lat, lon, st[1]))
                log(rid, f"{field}_coords", f"{st[3]:.5f},{st[4]:.5f}", f"{lat:.5f},{lon:.5f}",
                    f"terminal stop was {d:.1f} km from the verified location", src)
                flags.append(field)
            else:
                cur.execute(f"UPDATE routes SET {field}=? WHERE route_id=?", (st[2], rid))
                log(rid, field, label, st[2],
                    f"route is labelled '{label}' but its stop list {'starts' if field == 'origin' else 'ends'} at "
                    f"'{st[2]}', {d:.1f} km from the verified {name}; label corrected to the actual stop", src)
        cur.execute("UPDATE routes SET verified_terminals=? WHERE route_id=?", (",".join(flags) or None, rid))

    # ---- 2. consensus check for the remaining terminals
    routes = cur.execute("SELECT route_id, origin, destination FROM routes").fetchall()
    occ = defaultdict(list)
    ends = {}
    for rid, origin, dest in routes:
        stops = route_ends(rid)
        if len(stops) < 2:
            continue
        ends[rid] = stops
        occ[norm(origin)].append((stops[0][3], stops[0][4], stops[0][1]))
        occ[norm(dest)].append((stops[-1][3], stops[-1][4], stops[-1][1]))
    consensus = {}
    for k, pts in occ.items():
        pts = [p for p in pts if p[0]]
        if len(pts) < 3:
            continue
        la, lo = statistics.median(p[0] for p in pts), statistics.median(p[1] for p in pts)
        close = [p for p in pts if hav(la, lo, p[0], p[1]) < 0.5]
        if len(close) >= 3:
            sid = Counter(p[2] for p in close).most_common(1)[0][0]
            consensus[k] = (la, lo, sid)
    for rid, origin, dest in routes:
        if rid not in ends:
            continue
        stops = ends[rid]
        for field, label, st, nb, seq in (("origin", origin, stops[0], stops[1], stops[0][0]),
                                          ("destination", dest, stops[-1], stops[-2], stops[-1][0])):
            if alias.get(label.lower().strip()) or norm(label) not in consensus or not st[3]:
                continue
            la, lo, sid = consensus[norm(label)]
            d_now = hav(st[3], st[4], nb[3], nb[4])
            d_cons = hav(la, lo, nb[3], nb[4])
            if d_now > 3 and d_cons < d_now / 2 and sid != st[1]:
                cur.execute("UPDATE route_stops SET stop_id=? WHERE route_id=? AND stop_sequence=?", (sid, rid, seq))
                log(rid, f"{field}_stop", st[1], sid,
                    f"terminal was {d_now:.1f} km from the route's next stop; other routes place '{label}' "
                    f"{d_cons:.1f} km from it — re-pointed to the consensus stop", "consistency check across routes")

    # ---- 3. merge name variants of the same place
    routes = cur.execute("SELECT route_id, origin, destination FROM routes").fetchall()
    counts = Counter()
    pos = defaultdict(list)
    for rid, origin, dest in routes:
        counts[origin] += 1
        counts[dest] += 1
        if rid in ends:
            pos[origin].append(ends[rid][0][3:5])
            pos[dest].append(ends[rid][-1][3:5])
    groups = defaultdict(list)
    for n in counts:
        groups[norm(n)].append(n)
    rename = {}
    for k, variants in groups.items():
        if len(variants) < 2:
            if variants[0].isupper():
                rename[variants[0]] = nice(variants[0])
            continue
        best = max(variants, key=lambda v: (counts[v], not v.isupper()))
        bp = pos.get(best) or [(None, None)]
        for v in variants:
            if v == best:
                continue
            vp = pos.get(v) or [(None, None)]
            if bp[0][0] and vp[0][0] and hav(bp[0][0], bp[0][1], vp[0][0], vp[0][1]) > 1.5:
                continue  # same name, different place
            rename[v] = nice(best)
        if best.isupper():
            rename[best] = nice(best)
    for old, new in rename.items():
        if old == new:
            continue
        n1 = cur.execute("UPDATE routes SET origin=? WHERE origin=?", (new, old)).rowcount
        n2 = cur.execute("UPDATE routes SET destination=? WHERE destination=?", (new, old)).rowcount
        log("*", "terminal_name", old, new, f"name variant merged ({n1 + n2} route ends)", "name normalisation")

    # ---- place names for the planner (one clean name per place)
    place_of = {}
    for v in list(counts):
        place_of[norm(v)] = rename.get(v, v)
    for name, al, *_ in REFERENCE:
        for a in al:
            place_of[norm(a)] = name
    rows = cur.execute("SELECT stop_id, short_name FROM stops").fetchall()
    cur.executemany("UPDATE stops SET place=? WHERE stop_id=?",
                    [(place_of.get(norm(sn), nice(sn)), sid) for sid, sn in rows])

    cur.executemany("INSERT INTO data_audit(route_id,field,old,new,reason,source) VALUES (?,?,?,?,?,?)", audit)
    conn.commit()
    REF_CSV.parent.mkdir(exist_ok=True)
    with open(REF_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["terminal", "aliases", "lat", "lon", "source"])
        for name, al, lat, lon, src in REFERENCE:
            w.writerow([name, "; ".join(al), lat, lon, src])
    with open(AUDIT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["route_id", "field", "old", "new", "reason", "source"])
        w.writerows(audit)
    if verbose:
        kinds = Counter(a[1].split("_")[0] if a[1] != "terminal_name" else "name" for a in audit)
        print(f"  terminals: {verified_count} route ends verified against published coordinates; "
              f"{len(audit)} fixes logged {dict(kinds)}")
    return audit


if __name__ == "__main__":
    from backend import db
    apply(db.connect())
