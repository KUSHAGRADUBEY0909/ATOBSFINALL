"""ATOBS - Bengaluru bus tracking, crowd forecasting, camera passenger counting and emergency alerts."""
import secrets
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

from . import auth, db, demo, ml, schedule, simulation, vision

BASE = Path(__file__).resolve().parent.parent
FRONTEND = BASE / "frontend"

app = FastAPI(title="ATOBS API", version="1.0")
db.init_db()


@app.on_event("startup")
def _warm_detector():
    # load YOLO in the background so the first headcount request doesn't wait for it
    import threading
    threading.Thread(target=vision.warm_up, daemon=True).start()

SESSIONS: dict[str, vision.CountingSession] = {}


# ============================================================ auth
class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(pattern=EMAIL, max_length=120)
    phone: str | None = Field(default=None, max_length=20)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    email: str = Field(max_length=120)
    password: str


def _set_cookie(resp: Response, user: dict):
    resp.set_cookie(auth.COOKIE, auth.make_token(user), httponly=True, samesite="lax", max_age=auth.TOKEN_TTL)


@app.post("/api/auth/register")
def register(body: RegisterIn, response: Response):
    with db.get_db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email=?", (body.email.lower(),)).fetchone():
            raise HTTPException(409, "An account with this email already exists")
        cur = conn.execute(
            "INSERT INTO users(name,email,phone,password_hash,role) VALUES (?,?,?,?, 'passenger')",
            (body.name.strip(), body.email.lower(), body.phone, auth.hash_password(body.password)),
        )
        user = db.row(conn.execute("SELECT id,name,email,phone,role FROM users WHERE id=?", (cur.lastrowid,)))
    _set_cookie(response, user)
    return {"user": user}


@app.post("/api/auth/login")
def login(body: LoginIn, response: Response):
    with db.get_db() as conn:
        u = db.row(conn.execute("SELECT * FROM users WHERE email=?", (body.email.lower(),)))
    if not u or not auth.verify_password(body.password, u["password_hash"]):
        raise HTTPException(401, "Wrong email or password")
    u.pop("password_hash")
    _set_cookie(response, u)
    return {"user": u}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user=Depends(auth.current_user_optional)):
    return {"user": user}


# ============================================================ transit data
@app.get("/api/stats")
def stats():
    with db.get_db() as conn:
        q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
        return {
            "routes": q("SELECT COUNT(*) FROM routes"),
            "stops": q("SELECT COUNT(*) FROM stops"),
            "observations": q("SELECT COUNT(*) FROM trip_observations"),
            "buses_live": q("SELECT COUNT(*) FROM buses WHERE last_update >= datetime('now','-2 minutes')"),
            "model": (ml.load() or {}).get("meta", {"name": "heuristic"}),
            "data_fixes": q("SELECT COUNT(*) FROM data_audit"),
            "verified_routes": q("SELECT COUNT(*) FROM routes WHERE verified_terminals IS NOT NULL"),
        }


@app.get("/api/routes")
def list_routes(q: str = "", limit: int = 30, offset: int = 0):
    limit = max(1, min(limit, 100))
    like = f"%{q.strip()}%"
    with db.get_db() as conn:
        items = db.rows(conn.execute(
            """SELECT route_id, origin, destination, distance_km, journey_minutes, stop_count,
                      (SELECT COUNT(*) FROM buses b WHERE b.route_id=r.route_id) AS buses
               FROM routes r
               WHERE route_id LIKE ? OR origin LIKE ? OR destination LIKE ?
               ORDER BY (route_id = ?) DESC, (route_id LIKE ?) DESC, length(route_id), route_id
               LIMIT ? OFFSET ?""",
            (like, like, like, q.strip(), f"{q.strip()}%", limit, offset)))
        total = conn.execute(
            "SELECT COUNT(*) FROM routes WHERE route_id LIKE ? OR origin LIKE ? OR destination LIKE ?",
            (like, like, like)).fetchone()[0]
    return {"total": total, "items": items}


def _route_or_404(conn, route_id):
    r = db.row(conn.execute("SELECT * FROM routes WHERE route_id=?", (route_id,)))
    if not r:
        raise HTTPException(404, "Route not found")
    return r


@app.get("/api/routes/{route_id}")
def route_detail(route_id: str):
    with db.get_db() as conn:
        r = _route_or_404(conn, route_id)
        r["stops"] = db.rows(conn.execute(
            """SELECT rs.stop_sequence AS seq, s.stop_id, s.name, s.short_name, s.lat, s.lon
               FROM route_stops rs JOIN stops s USING(stop_id) WHERE rs.route_id=? ORDER BY rs.stop_sequence""",
            (route_id,)))
        r["buses"] = _buses(conn, "WHERE b.route_id=?", (route_id,))
        obs = db.row(conn.execute(
            """SELECT COUNT(*) n, CAST(ROUND(AVG(bus_occupancy)) AS INT) avg_occ, CAST(ROUND(MAX(bus_occupancy)) AS INT) max_occ,
                      ROUND(AVG(delay_min),2) avg_delay, COUNT(DISTINCT trip_id) trips
               FROM trip_observations WHERE route_id=?""", (route_id,)))
        r["observations"] = obs if obs and obs["n"] else None
    r["departures_from_origin"] = (r["departures_from_origin"] or "").split(";") if r["departures_from_origin"] else []
    r["departures_from_destination"] = (r["departures_from_destination"] or "").split(";") if r["departures_from_destination"] else []
    return r


@app.get("/api/routes/{route_id}/forecast")
def route_forecast(route_id: str, hour: float | None = None, traffic: str | None = None):
    if hour is None:
        now = datetime.now()
        hour = now.hour + now.minute / 60
    if not 0 <= hour < 24:
        raise HTTPException(400, "hour must be between 0 and 24")
    with db.get_db() as conn:
        r = _route_or_404(conn, route_id)
        observed = db.rows(conn.execute(
            """SELECT stop_sequence seq, CAST(ROUND(AVG(bus_occupancy)) AS INT) occupancy, COUNT(*) n
               FROM trip_observations WHERE route_id=? AND hour>=? AND hour<?
               GROUP BY stop_sequence ORDER BY stop_sequence""",
            (route_id, int(hour), int(hour) + 1)))
    tr = ml.TRAFFIC.get(traffic) if traffic else None
    p = ml.predict(hour, r["stop_count"] or 1, tr)
    peak = max((s["occupancy"] for s in p["stops"]), default=0)
    p.update(hour=hour, route_id=route_id, peak_occupancy=peak, label=ml.crowd_label(peak),
             observed=observed)
    return p


@app.get("/api/places/search")
def place_search(q: str, limit: int = 12):
    """Clean place names (terminal/stop names after the data clean-up) for the trip planner."""
    if len(q.strip()) < 2:
        return {"items": []}
    with db.get_db() as conn:
        items = db.rows(conn.execute(
            """SELECT s.place, COUNT(DISTINCT rs.route_id) AS routes, AVG(s.lat) lat, AVG(s.lon) lon,
                      MAX(EXISTS(SELECT 1 FROM routes r WHERE r.origin=s.place OR r.destination=s.place)) AS terminal
               FROM stops s JOIN route_stops rs USING(stop_id)
               WHERE s.place LIKE ? GROUP BY s.place
               ORDER BY (s.place LIKE ?) DESC, routes DESC LIMIT ?""",
            (f"%{q.strip()}%", f"{q.strip()}%", min(limit, 25))))
    return {"items": items}


@app.get("/api/journey")
def journey(origin: str, destination: str, at: str | None = None, window: int = 120):
    """All buses from origin to destination: timetable + 10/15-min frequency, traffic-adjusted ETAs."""
    when = None
    if at:
        try:
            h, m = map(int, at.split(":"))
            when = datetime.now().replace(hour=h, minute=m, second=0)
        except Exception:
            raise HTTPException(400, "at must be HH:MM")
    if origin.strip().lower() == destination.strip().lower():
        raise HTTPException(400, "Choose two different places")
    return schedule.journey(origin.strip(), destination.strip(), when, max(15, min(window, 360)))


@app.get("/api/data/audit")
def data_audit():
    with db.get_db() as conn:
        items = db.rows(conn.execute("SELECT * FROM data_audit ORDER BY (field='terminal_name'), id"))
        verified = conn.execute("SELECT COUNT(*) FROM routes WHERE verified_terminals IS NOT NULL").fetchone()[0]
    return {"items": items, "routes_with_verified_terminal": verified}


@app.get("/api/stops/search")
def stop_search(q: str, limit: int = 10):
    if len(q.strip()) < 2:
        return {"items": []}
    with db.get_db() as conn:
        items = db.rows(conn.execute(
            """SELECT s.short_name, MIN(s.stop_id) AS stop_id, COUNT(DISTINCT rs.route_id) AS routes,
                      AVG(s.lat) lat, AVG(s.lon) lon
               FROM stops s JOIN route_stops rs USING(stop_id)
               WHERE s.short_name LIKE ? GROUP BY s.short_name
               ORDER BY (s.short_name LIKE ?) DESC, routes DESC LIMIT ?""",
            (f"%{q.strip()}%", f"{q.strip()}%", min(limit, 25))))
    return {"items": items}


@app.get("/api/plan")
def plan(origin: str, destination: str):
    """Direct routes that serve both stops (by stop name) in the right order."""
    with db.get_db() as conn:
        items = db.rows(conn.execute(
            """SELECT r.route_id, r.origin, r.destination, r.distance_km, r.journey_minutes, r.stop_count,
                      MIN(a.stop_sequence) AS board_seq, MAX(b.stop_sequence) AS alight_seq
               FROM route_stops a JOIN stops sa ON sa.stop_id=a.stop_id
               JOIN route_stops b ON b.route_id=a.route_id JOIN stops sb ON sb.stop_id=b.stop_id
               JOIN routes r ON r.route_id=a.route_id
               WHERE sa.short_name=? AND sb.short_name=? AND a.stop_sequence < b.stop_sequence
               GROUP BY r.route_id ORDER BY (MAX(b.stop_sequence)-MIN(a.stop_sequence)) LIMIT 25""",
            (origin, destination)))
    for it in items:
        n = it["alight_seq"] - it["board_seq"]
        it["stops_between"] = n
        if it["journey_minutes"] and it["stop_count"]:
            it["est_minutes"] = round(it["journey_minutes"] * n / max(it["stop_count"] - 1, 1))
    return {"items": items}


# ============================================================ buses / live occupancy
def _buses(conn, where="", args=()):
    items = db.rows(conn.execute(
        f"""SELECT b.*, r.origin, r.destination, s.short_name AS last_stop_name,
                   (b.last_update >= datetime('now','-2 minutes')) AS live
            FROM buses b LEFT JOIN routes r USING(route_id) LEFT JOIN stops s ON s.stop_id=b.last_stop_id
            {where} ORDER BY b.id""", args))
    for b in items:
        b["occupancy_pct"] = int(round(100 * b["current_onboard"] / b["capacity"])) if b["capacity"] else None
        b["seats_occupied"] = min(b["current_onboard"], b["capacity"])
        b["seats_free"] = max(0, b["capacity"] - b["current_onboard"])
        b["standing"] = max(0, b["current_onboard"] - b["capacity"])
        b["crowd_label"] = ml.crowd_label(b["occupancy_pct"] or 0)
        b["live"] = bool(b["live"])
    return items


@app.get("/api/buses")
def list_buses():
    with db.get_db() as conn:
        return {"items": _buses(conn)}


class BusIn(BaseModel):
    registration: str = Field(min_length=3, max_length=20)
    route_id: str
    capacity: int = Field(default=40, ge=5, le=120)


@app.post("/api/buses")
def add_bus(body: BusIn, user=Depends(auth.require_role("admin"))):
    with db.get_db() as conn:
        _route_or_404(conn, body.route_id)
        try:
            cur = conn.execute("INSERT INTO buses(registration,route_id,capacity) VALUES (?,?,?)",
                               (body.registration.upper().strip(), body.route_id, body.capacity))
        except Exception:
            raise HTTPException(409, "A bus with this registration already exists")
        return _buses(conn, "WHERE b.id=?", (cur.lastrowid,))[0]


# ============================================================ camera passenger counting
class SessionIn(BaseModel):
    bus_id: int
    source: str = "webcam"
    line_pos: float = Field(default=0.5, ge=0.05, le=0.95)
    orientation: str = Field(default="horizontal", pattern="^(horizontal|vertical)$")
    entry_direction: int = Field(default=1)
    reset_onboard: bool = False


class SessionPatch(BaseModel):
    line_pos: float | None = Field(default=None, ge=0.05, le=0.95)
    orientation: str | None = Field(default=None, pattern="^(horizontal|vertical)$")
    entry_direction: int | None = None
    onboard: int | None = Field(default=None, ge=0, le=500)
    stop_id: str | None = None


def _session(sid: str) -> vision.CountingSession:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Counting session not found or already ended")
    return s


@app.get("/api/count/detector")
def detector_info(user=Depends(auth.require_role("operator", "admin"))):
    return {"detector": vision.get_detector().name}


@app.post("/api/count/sessions")
def start_session(body: SessionIn, user=Depends(auth.require_role("operator", "admin"))):
    with db.get_db() as conn:
        bus = db.row(conn.execute("SELECT * FROM buses WHERE id=?", (body.bus_id,)))
        if not bus:
            raise HTTPException(404, "Bus not found")
        # one active camera per bus
        for k, s in list(SESSIONS.items()):
            if s.bus_id == bus["id"]:
                _end(k)
        onboard = 0 if body.reset_onboard else bus["current_onboard"]
        sid = uuid.uuid4().hex[:12]
        det = vision.get_detector().name
        conn.execute(
            "INSERT INTO count_sessions(id,bus_id,operator_id,route_id,source,detector) VALUES (?,?,?,?,?,?)",
            (sid, bus["id"], user["id"], bus["route_id"], body.source[:20], det))
        if body.reset_onboard:
            conn.execute("UPDATE buses SET current_onboard=0 WHERE id=?", (bus["id"],))
    s = vision.CountingSession(sid, bus["id"], onboard=onboard, line_pos=body.line_pos,
                               orientation=body.orientation, entry_direction=1 if body.entry_direction >= 0 else -1,
                               capacity=bus["capacity"])
    SESSIONS[sid] = s
    return {**s.snapshot(0), "detector": det, "bus": bus}


@app.post("/api/count/sessions/{sid}/frame")
async def session_frame(sid: str, frame: UploadFile = File(...), conf: float = Form(0.35),
                        user=Depends(auth.require_role("operator", "admin"))):
    s = _session(sid)
    data = await frame.read()
    if len(data) > 6_000_000:
        raise HTTPException(413, "Frame too large")
    img = vision.decode_image(data)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    t0 = time.perf_counter()
    boxes = vision.detect_normalised(img, max(0.1, min(conf, 0.9)))
    events = s.update(boxes)
    ms = round((time.perf_counter() - t0) * 1000)
    snap = s.snapshot(len(boxes))
    with db.get_db() as conn:
        bus = db.row(conn.execute("SELECT last_stop_id FROM buses WHERE id=?", (s.bus_id,)))
        for e in events:
            conn.execute("INSERT INTO count_events(session_id,bus_id,direction,onboard_after,stop_id) VALUES (?,?,?,?,?)",
                         (sid, s.bus_id, e["direction"], e["onboard_after"], bus["last_stop_id"] if bus else None))
        conn.execute(
            """UPDATE buses SET current_onboard=?, in_frame=?, last_update=datetime('now'),
               total_in=total_in+?, total_out=total_out+? WHERE id=?""",
            (s.onboard, len(boxes), sum(e["direction"] == "in" for e in events),
             sum(e["direction"] == "out" for e in events), s.bus_id))
        conn.execute(
            "UPDATE count_sessions SET total_in=?, total_out=?, max_in_frame=?, frames_processed=? WHERE id=?",
            (s.total_in, s.total_out, s.max_in_frame, s.frames, sid))
    return {**snap, "events": events, "tracks": s.track_list(),
            "detections": [[round(v, 4) for v in b] for b in boxes], "inference_ms": ms}


@app.patch("/api/count/sessions/{sid}")
def patch_session(sid: str, body: SessionPatch, user=Depends(auth.require_role("operator", "admin"))):
    s = _session(sid)
    with s.lock:
        if body.line_pos is not None:
            s.line_pos = body.line_pos
        if body.orientation:
            s.orientation = body.orientation
        if body.line_pos is not None or body.orientation:
            for t in s.tracks.values():
                t.side = None
        if body.entry_direction is not None:
            s.entry_direction = 1 if body.entry_direction >= 0 else -1
        if body.onboard is not None:
            s.onboard = body.onboard
    with db.get_db() as conn:
        if body.onboard is not None:
            conn.execute("UPDATE buses SET current_onboard=? WHERE id=?", (body.onboard, s.bus_id))
        if body.stop_id:
            st = db.row(conn.execute("SELECT lat,lon FROM stops WHERE stop_id=?", (body.stop_id,)))
            if not st:
                raise HTTPException(404, "Stop not found")
            conn.execute("UPDATE buses SET last_stop_id=?, lat=?, lon=? WHERE id=?",
                         (body.stop_id, st["lat"], st["lon"], s.bus_id))
    return s.snapshot(0)


def _end(sid):
    s = SESSIONS.pop(sid, None)
    if not s:
        return None
    with db.get_db() as conn:
        conn.execute(
            """UPDATE count_sessions SET ended_at=datetime('now'), total_in=?, total_out=?,
               max_in_frame=?, frames_processed=? WHERE id=?""",
            (s.total_in, s.total_out, s.max_in_frame, s.frames, sid))
        conn.execute("UPDATE buses SET in_frame=0 WHERE id=?", (s.bus_id,))
    return s.snapshot(0)


@app.post("/api/count/sessions/{sid}/end")
def end_session(sid: str, user=Depends(auth.require_role("operator", "admin"))):
    snap = _end(sid)
    if not snap:
        raise HTTPException(404, "Counting session not found or already ended")
    return snap


@app.get("/api/count/sessions/{sid}/events")
def session_events(sid: str, user=Depends(auth.require_role("operator", "admin"))):
    with db.get_db() as conn:
        return {"items": db.rows(conn.execute(
            """SELECT e.*, s.short_name AS stop_name FROM count_events e LEFT JOIN stops s USING(stop_id)
               WHERE session_id=? ORDER BY e.id DESC LIMIT 100""", (sid,)))}


@app.post("/api/count/image")
async def count_image(frame: UploadFile = File(...), conf: float = Form(0.35),
                      user=Depends(auth.require_role("operator", "admin"))):
    """One-off headcount on a single photo (e.g. of the bus interior)."""
    img = vision.decode_image(await frame.read())
    if img is None:
        raise HTTPException(400, "Could not decode image")
    boxes = vision.detect_normalised(img, max(0.1, min(conf, 0.9)))
    return {"count": len(boxes), "detections": [[round(v, 4) for v in b] for b in boxes],
            "detector": vision.get_detector().name}


# ============================================================ emergency alerts
CATEGORIES = {
    "medical": ("Medical emergency", "critical"),
    "harassment": ("Harassment / safety threat", "critical"),
    "accident": ("Accident / collision", "critical"),
    "fire": ("Fire / smoke", "critical"),
    "theft": ("Theft / pickpocketing", "high"),
    "breakdown": ("Bus breakdown", "medium"),
    "overcrowding": ("Dangerous overcrowding", "medium"),
    "other": ("Other", "high"),
}
_rate: dict[str, deque] = defaultdict(deque)


class AlertIn(BaseModel):
    category: str
    description: str = Field(min_length=5, max_length=2000)
    severity: str | None = Field(default=None, pattern="^(critical|high|medium|low)$")
    reporter_name: str | None = Field(default=None, max_length=80)
    reporter_phone: str | None = Field(default=None, max_length=20)
    route_id: str | None = Field(default=None, max_length=12)
    bus_registration: str | None = Field(default=None, max_length=20)
    stop_id: str | None = Field(default=None, max_length=12)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    page_url: str | None = Field(default=None, max_length=300)


@app.get("/api/emergency/categories")
def alert_categories():
    return {"items": [{"key": k, "label": v[0], "severity": v[1]} for k, v in CATEGORIES.items()]}


@app.post("/api/emergency")
def create_alert(body: AlertIn, request: Request, user=Depends(auth.current_user_optional)):
    if body.category not in CATEGORIES:
        raise HTTPException(400, "Unknown emergency category")
    ip = request.client.host if request.client else "?"
    q = _rate[ip]
    now = time.time()
    while q and now - q[0] > 600:
        q.popleft()
    if len(q) >= 10:
        raise HTTPException(429, "Too many alerts sent from this device. Call 112 if you are in danger.")
    q.append(now)
    severity = body.severity or CATEGORIES[body.category][1]
    ref = f"EM-{datetime.now():%y%m%d}-{secrets.token_hex(3).upper()}"
    with db.get_db() as conn:
        route_id = body.route_id if body.route_id and conn.execute(
            "SELECT 1 FROM routes WHERE route_id=?", (body.route_id,)).fetchone() else None
        cur = conn.execute(
            """INSERT INTO emergency_alerts(reference,user_id,reporter_name,reporter_phone,category,severity,
               description,route_id,bus_registration,stop_id,lat,lon,page_url) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ref, user["id"] if user else None,
             body.reporter_name or (user["name"] if user else None),
             body.reporter_phone or (user["phone"] if user else None),
             body.category, severity, body.description.strip(), route_id,
             (body.bus_registration or "").upper().strip() or None, body.stop_id, body.lat, body.lon, body.page_url))
        conn.execute("INSERT INTO alert_updates(alert_id,status,note) VALUES (?, 'new', 'Alert received')",
                     (cur.lastrowid,))
    return {"reference": ref, "status": "new", "severity": severity,
            "message": "Your alert has reached the ATOBS control room."}


def _alert_public(a: dict) -> dict:
    a["category_label"] = CATEGORIES.get(a["category"], (a["category"],))[0]
    return a


@app.get("/api/emergency/track/{reference}")
def track_alert(reference: str):
    with db.get_db() as conn:
        a = db.row(conn.execute(
            "SELECT id,reference,category,severity,status,route_id,bus_registration,created_at,updated_at "
            "FROM emergency_alerts WHERE reference=?", (reference.upper(),)))
        if not a:
            raise HTTPException(404, "No alert with that reference")
        a["updates"] = db.rows(conn.execute(
            "SELECT status, note, created_at FROM alert_updates WHERE alert_id=? ORDER BY id", (a.pop("id"),)))
    return _alert_public(a)


@app.get("/api/emergency/mine")
def my_alerts(user=Depends(auth.current_user)):
    with db.get_db() as conn:
        items = db.rows(conn.execute(
            "SELECT id,reference,category,severity,status,description,route_id,bus_registration,created_at,updated_at "
            "FROM emergency_alerts WHERE user_id=? ORDER BY id DESC", (user["id"],)))
    return {"items": [_alert_public(a) for a in items]}


class AlertPatch(BaseModel):
    status: str | None = Field(default=None, pattern="^(new|acknowledged|dispatched|resolved|false_alarm)$")
    note: str | None = Field(default=None, max_length=1000)


@app.get("/api/admin/alerts")
def admin_alerts(status: str | None = None, user=Depends(auth.require_role("admin", "operator"))):
    sql = """SELECT a.*, r.origin, r.destination, s.short_name AS stop_name, u.email AS user_email
             FROM emergency_alerts a LEFT JOIN routes r USING(route_id) LEFT JOIN stops s ON s.stop_id=a.stop_id
             LEFT JOIN users u ON u.id=a.user_id"""
    args = ()
    if status == "open":
        sql += " WHERE a.status IN ('new','acknowledged','dispatched')"
    elif status:
        sql += " WHERE a.status=?"
        args = (status,)
    sql += """ ORDER BY (a.status IN ('resolved','false_alarm')),
               CASE a.severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
               a.id DESC LIMIT 200"""
    with db.get_db() as conn:
        return {"items": [_alert_public(a) for a in db.rows(conn.execute(sql, args))]}


@app.get("/api/admin/alerts/{alert_id}")
def admin_alert(alert_id: int, user=Depends(auth.require_role("admin", "operator"))):
    with db.get_db() as conn:
        a = db.row(conn.execute(
            """SELECT a.*, r.origin, r.destination, s.short_name AS stop_name FROM emergency_alerts a
               LEFT JOIN routes r USING(route_id) LEFT JOIN stops s ON s.stop_id=a.stop_id WHERE a.id=?""",
            (alert_id,)))
        if not a:
            raise HTTPException(404, "Alert not found")
        a["updates"] = db.rows(conn.execute(
            """SELECT au.*, u.name AS author FROM alert_updates au LEFT JOIN users u ON u.id=au.author_id
               WHERE alert_id=? ORDER BY au.id""", (alert_id,)))
        if a["bus_registration"]:
            a["bus"] = db.row(conn.execute("SELECT * FROM buses WHERE registration=?", (a["bus_registration"],)))
    return _alert_public(a)


@app.patch("/api/admin/alerts/{alert_id}")
def update_alert(alert_id: int, body: AlertPatch, user=Depends(auth.require_role("admin", "operator"))):
    if not body.status and not body.note:
        raise HTTPException(400, "Nothing to update")
    with db.get_db() as conn:
        if not conn.execute("SELECT 1 FROM emergency_alerts WHERE id=?", (alert_id,)).fetchone():
            raise HTTPException(404, "Alert not found")
        if body.status:
            conn.execute("UPDATE emergency_alerts SET status=?, updated_at=datetime('now') WHERE id=?",
                         (body.status, alert_id))
        else:
            conn.execute("UPDATE emergency_alerts SET updated_at=datetime('now') WHERE id=?", (alert_id,))
        conn.execute("INSERT INTO alert_updates(alert_id,author_id,status,note) VALUES (?,?,?,?)",
                     (alert_id, user["id"], body.status, body.note))
    return admin_alert(alert_id, user)


@app.get("/api/admin/summary")
def admin_summary(user=Depends(auth.require_role("admin", "operator"))):
    with db.get_db() as conn:
        q = lambda s, a=(): conn.execute(s, a).fetchone()[0]  # noqa: E731
        return {
            "open_alerts": q("SELECT COUNT(*) FROM emergency_alerts WHERE status IN ('new','acknowledged','dispatched')"),
            "new_alerts": q("SELECT COUNT(*) FROM emergency_alerts WHERE status='new'"),
            "critical_open": q("SELECT COUNT(*) FROM emergency_alerts WHERE severity='critical' AND status IN ('new','acknowledged','dispatched')"),
            "latest_alert_id": q("SELECT COALESCE(MAX(id),0) FROM emergency_alerts"),
            "alerts_today": q("SELECT COUNT(*) FROM emergency_alerts WHERE date(created_at,'localtime')=date('now','localtime')"),
            "boardings_today": q("SELECT COUNT(*) FROM count_events WHERE direction='in' AND date(created_at,'localtime')=date('now','localtime')"),
            "alightings_today": q("SELECT COUNT(*) FROM count_events WHERE direction='out' AND date(created_at,'localtime')=date('now','localtime')"),
            "active_cameras": len(SESSIONS),
            "buses": _buses(conn),
            "recent_sessions": db.rows(conn.execute(
                """SELECT cs.*, b.registration, u.name AS operator FROM count_sessions cs JOIN buses b ON b.id=cs.bus_id
                   LEFT JOIN users u ON u.id=cs.operator_id ORDER BY cs.started_at DESC LIMIT 10""")),
        }


# ============================================================ camera simulation
@app.get("/api/sim/clips")
def sim_clips():
    return {"items": simulation.list_clips(), "detector": vision.get_detector().name}


@app.post("/api/sim/{clip_id}/evaluate")
def sim_evaluate(clip_id: str, user=Depends(auth.require_role("operator", "admin"))):
    try:
        return simulation.start_job(clip_id)
    except FileNotFoundError:
        raise HTTPException(404, "Clip not found")


@app.get("/api/sim/{clip_id}/job")
def sim_job(clip_id: str):
    return simulation.JOBS.get(clip_id) or {"status": "idle"}


class ManualGT(BaseModel):
    t: float = Field(ge=0)
    count: int = Field(ge=0, le=200)


@app.post("/api/sim/{clip_id}/verify")
def sim_verify(clip_id: str, body: ManualGT, user=Depends(auth.require_role("operator", "admin"))):
    """Add a hand-checked headcount at time t (for clips without ground truth) and re-score."""
    try:
        return simulation.add_manual_gt(clip_id, body.t, body.count)
    except FileNotFoundError:
        raise HTTPException(404, "Clip not found")


@app.post("/api/sim/upload")
async def sim_upload(file: UploadFile = File(...), title: str = Form(...), camera: str = Form("front"),
                     user=Depends(auth.require_role("operator", "admin"))):
    data = await file.read()
    if len(data) > 200_000_000:
        raise HTTPException(413, "Video too large (max 200 MB)")
    try:
        meta = simulation.add_upload(data, file.filename or "clip.mp4", title.strip()[:80] or "Uploaded clip", camera)
    except ValueError as e:
        raise HTTPException(400, str(e))
    simulation.start_job(meta["id"])
    return meta


@app.post("/api/sim/detect")
async def sim_detect(frame: UploadFile = File(...)):
    """Live headcount on the current video frame (tiled detection, same as the evaluation)."""
    img = vision.decode_image(await frame.read())
    if img is None:
        raise HTTPException(400, "Could not decode image")
    t0 = time.perf_counter()
    boxes = vision.detect_tiled(img, 0.3)
    return {"count": len(boxes), "boxes": [[round(v, 4) for v in b] for b in boxes],
            "ms": round((time.perf_counter() - t0) * 1000), "detector": vision.get_detector().name}


# ============================================================ live headcount demo
@app.post("/api/demo/detect")
async def demo_detect(frame: UploadFile = File(...), mode: str = Form("accurate")):
    """Headcount on one image (an uploaded photo or the current frame of a video)."""
    data = await frame.read()
    if len(data) > 15_000_000:
        raise HTTPException(413, "Image too large (max 15 MB)")
    img = vision.decode_image(data)
    if img is None:
        raise HTTPException(400, "Could not read this image. Use a JPG or PNG.")
    t0 = time.perf_counter()
    boxes = demo.detect(img, "fast" if mode == "fast" else "accurate")
    return {"count": len(boxes), "boxes": [[round(v, 4) for v in b] for b in boxes],
            "ms": round((time.perf_counter() - t0) * 1000), "detector": vision.get_detector().name,
            "mode": mode, "width": img.shape[1], "height": img.shape[0]}


@app.post("/api/demo/video")
async def demo_video(file: UploadFile = File(...), interval: float = Form(1.0), mode: str = Form("accurate")):
    """Server-side processing for videos the browser can't play (e.g. HEVC .mov)."""
    data = await file.read()
    if len(data) > 300_000_000:
        raise HTTPException(413, "Video too large (max 300 MB)")
    try:
        return demo.start_video_job(data, file.filename or "video.mp4", max(0.25, min(interval, 10)),
                                    "fast" if mode == "fast" else "accurate")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/demo/video/{job_id}")
def demo_video_state(job_id: str, since: int = 0):
    st = demo.job_state(job_id, max(0, since))
    if not st:
        raise HTTPException(404, "Job not found")
    return st


@app.post("/api/demo/video/{job_id}/stop")
def demo_video_stop(job_id: str):
    job = demo.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job["stop"] = True
    return {"ok": True}


class DemoRunIn(BaseModel):
    kind: str = Field(pattern="^(photo|video)$")
    filename: str | None = Field(default=None, max_length=200)
    mode: str | None = Field(default=None, max_length=20)
    frames: int = Field(default=1, ge=0, le=100000)
    max_count: int = Field(default=0, ge=0, le=1000)
    avg_count: float = Field(default=0, ge=0, le=1000)
    last_count: int = Field(default=0, ge=0, le=1000)


class DemoCheckIn(BaseModel):
    t: float | None = Field(default=None, ge=0)
    detected: int = Field(ge=0, le=1000)
    actual: int = Field(ge=0, le=1000)


@app.post("/api/demo/runs")
def demo_save_run(body: DemoRunIn, user=Depends(auth.current_user_optional)):
    with db.get_db() as conn:
        cur = conn.execute(
            "INSERT INTO demo_runs(user_id,kind,filename,mode,frames,max_count,avg_count,last_count) VALUES (?,?,?,?,?,?,?,?)",
            (user["id"] if user else None, body.kind, body.filename, body.mode, body.frames, body.max_count,
             round(body.avg_count, 2), body.last_count))
        return {"id": cur.lastrowid}


@app.patch("/api/demo/runs/{run_id}")
def demo_update_run(run_id: int, body: DemoRunIn):
    with db.get_db() as conn:
        n = conn.execute("UPDATE demo_runs SET frames=?, max_count=?, avg_count=?, last_count=? WHERE id=?",
                         (body.frames, body.max_count, round(body.avg_count, 2), body.last_count, run_id)).rowcount
    if not n:
        raise HTTPException(404, "Run not found")
    return {"ok": True}


def _accuracy(checks):
    if not checks:
        return None
    tot = sum(c["actual"] for c in checks)
    err = sum(abs(c["detected"] - c["actual"]) for c in checks)
    return {"checks": len(checks),
            "accuracy_pct": int(round(100 * max(0.0, 1 - err / tot))) if tot else (100 if err == 0 else 0),
            "exact_pct": int(round(100 * sum(c["detected"] == c["actual"] for c in checks) / len(checks))),
            "within1_pct": int(round(100 * sum(abs(c["detected"] - c["actual"]) <= 1 for c in checks) / len(checks))),
            "avg_error": round(err / len(checks), 2)}


@app.post("/api/demo/runs/{run_id}/checks")
def demo_add_check(run_id: int, body: DemoCheckIn):
    with db.get_db() as conn:
        if not conn.execute("SELECT 1 FROM demo_runs WHERE id=?", (run_id,)).fetchone():
            raise HTTPException(404, "Run not found")
        conn.execute("INSERT INTO demo_checks(run_id,t,detected,actual) VALUES (?,?,?,?)",
                     (run_id, body.t, body.detected, body.actual))
        checks = db.rows(conn.execute("SELECT * FROM demo_checks WHERE run_id=? ORDER BY id", (run_id,)))
    return {"checks": checks, "summary": _accuracy(checks)}


@app.get("/api/demo/runs")
def demo_runs(limit: int = 20):
    with db.get_db() as conn:
        runs = db.rows(conn.execute("SELECT * FROM demo_runs ORDER BY id DESC LIMIT ?", (max(1, min(limit, 100)),)))
        allc = db.rows(conn.execute("SELECT * FROM demo_checks"))
    for r in runs:
        r["summary"] = _accuracy([c for c in allc if c["run_id"] == r["id"]])
    return {"items": runs, "overall": _accuracy(allc)}


# ============================================================ static frontend
@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
