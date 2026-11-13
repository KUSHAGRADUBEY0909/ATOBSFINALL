"""SQLite database access and schema for ATOBS."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "atobs.db"

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'passenger' CHECK (role IN ('passenger','operator','admin')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------- Transit data (imported from the ATOBS dataset) ----------
CREATE TABLE IF NOT EXISTS routes (
    route_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    distance_km REAL,
    journey_minutes INTEGER,
    stop_count INTEGER,
    departures_from_origin TEXT,       -- '07:35;08:15;...'
    departures_from_destination TEXT,
    verified_terminals TEXT           -- 'origin,destination' checked against published coordinates
);

CREATE TABLE IF NOT EXISTS data_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, route_id TEXT, field TEXT, old TEXT, new TEXT, reason TEXT, source TEXT
);

CREATE TABLE IF NOT EXISTS stops (
    stop_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    short_name TEXT NOT NULL,
    place TEXT,                       -- cleaned place name used by the planner
    lat REAL,
    lon REAL
);
CREATE INDEX IF NOT EXISTS idx_stops_short ON stops(short_name);

CREATE TABLE IF NOT EXISTS route_stops (
    route_id TEXT NOT NULL REFERENCES routes(route_id),
    stop_sequence INTEGER NOT NULL,
    stop_id TEXT NOT NULL REFERENCES stops(stop_id),
    PRIMARY KEY (route_id, stop_sequence)
);
CREATE INDEX IF NOT EXISTS idx_route_stops_stop ON route_stops(stop_id);

-- Stop-level trip observations (traffic, occupancy, delay) from the processed batches
CREATE TABLE IF NOT EXISTS trip_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id TEXT,
    route_id TEXT,
    stop_sequence INTEGER,
    stop_id TEXT,
    stop_name TEXT,
    departure_time TEXT,
    hour REAL,
    travel_time_between_stops REAL,
    is_peak_hour INTEGER,
    traffic_level TEXT,
    bus_occupancy REAL,
    delay_min REAL,
    route_stop_total INTEGER
);
CREATE INDEX IF NOT EXISTS idx_obs_route ON trip_observations(route_id);

-- ---------- Camera passenger counting ----------
CREATE TABLE IF NOT EXISTS buses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    registration TEXT NOT NULL UNIQUE,
    route_id TEXT REFERENCES routes(route_id),
    capacity INTEGER NOT NULL DEFAULT 40,  -- seats (BMTC ordinary: 40)
    current_onboard INTEGER NOT NULL DEFAULT 0,
    total_in INTEGER NOT NULL DEFAULT 0,
    total_out INTEGER NOT NULL DEFAULT 0,
    in_frame INTEGER NOT NULL DEFAULT 0,
    last_stop_id TEXT,
    lat REAL, lon REAL,
    last_update TEXT
);

CREATE TABLE IF NOT EXISTS count_sessions (
    id TEXT PRIMARY KEY,
    bus_id INTEGER NOT NULL REFERENCES buses(id),
    operator_id INTEGER REFERENCES users(id),
    route_id TEXT,
    source TEXT,                -- webcam / video
    detector TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    total_in INTEGER DEFAULT 0,
    total_out INTEGER DEFAULT 0,
    max_in_frame INTEGER DEFAULT 0,
    frames_processed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS count_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES count_sessions(id),
    bus_id INTEGER NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('in','out')),
    onboard_after INTEGER NOT NULL,
    stop_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_events_bus ON count_events(bus_id, created_at);

-- ---------- Emergency alerts ----------
CREATE TABLE IF NOT EXISTS emergency_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference TEXT NOT NULL UNIQUE,
    user_id INTEGER REFERENCES users(id),
    reporter_name TEXT,
    reporter_phone TEXT,
    category TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'high' CHECK (severity IN ('critical','high','medium','low')),
    description TEXT NOT NULL,
    route_id TEXT,
    bus_registration TEXT,
    stop_id TEXT,
    lat REAL, lon REAL,
    page_url TEXT,
    status TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new','acknowledged','dispatched','resolved','false_alarm')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON emergency_alerts(status, created_at);

-- ---------- Live headcount demo (uploaded photos / videos) ----------
CREATE TABLE IF NOT EXISTS demo_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    kind TEXT NOT NULL CHECK (kind IN ('photo','video')),
    filename TEXT,
    mode TEXT,
    frames INTEGER DEFAULT 0,
    max_count INTEGER DEFAULT 0,
    avg_count REAL DEFAULT 0,
    last_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS demo_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES demo_runs(id),
    t REAL,
    detected INTEGER NOT NULL,
    actual INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS alert_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL REFERENCES emergency_alerts(id),
    author_id INTEGER REFERENCES users(id),
    status TEXT,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


MIGRATIONS = [("stops", "place", "TEXT"), ("routes", "verified_terminals", "TEXT")]


def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)
        for table, col, typ in MIGRATIONS:  # databases created by older versions
            if col not in [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stops_place ON stops(place)")


def rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def row(cur) -> dict | None:
    r = cur.fetchone()
    return dict(r) if r else None
