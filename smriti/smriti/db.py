"""SQLite persistence.

One file, WAL mode, no ORM. The schema is small enough to read in one sitting
and the access patterns are all "everything for one event", which SQLite serves
comfortably into the hundreds of thousands of faces.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_settings

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    share_code       TEXT NOT NULL UNIQUE,
    admin_token_hash TEXT NOT NULL,
    engine           TEXT NOT NULL,
    embed_dim        INTEGER NOT NULL,
    created_at       REAL NOT NULL,
    expires_at       REAL,
    allow_download   INTEGER NOT NULL DEFAULT 1,
    notes            TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS photos (
    id         TEXT PRIMARY KEY,
    event_id   TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    orig_name  TEXT NOT NULL,
    rel_path   TEXT NOT NULL,
    thumb_path TEXT,
    sha256     TEXT NOT NULL,
    width      INTEGER NOT NULL DEFAULT 0,
    height     INTEGER NOT NULL DEFAULT 0,
    bytes      INTEGER NOT NULL DEFAULT 0,
    taken_at   REAL,
    state      TEXT NOT NULL DEFAULT 'pending',
    n_faces    INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    created_at REAL NOT NULL,
    UNIQUE (event_id, sha256)
);

CREATE TABLE IF NOT EXISTS faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    photo_id   TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x          INTEGER NOT NULL,
    y          INTEGER NOT NULL,
    w          INTEGER NOT NULL,
    h          INTEGER NOT NULL,
    det_score  REAL NOT NULL,
    face_px    REAL NOT NULL,
    blur       REAL NOT NULL DEFAULT 0,
    embedding  BLOB NOT NULL,
    cluster_id INTEGER,
    created_at REAL NOT NULL
);

-- Audit trail for searches. Deliberately stores NO biometric data: just the
-- fact that a search happened, so an organiser can see the event is being used.
CREATE TABLE IF NOT EXISTS search_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL,
    ts         REAL NOT NULL,
    n_queries  INTEGER NOT NULL,
    n_matches  INTEGER NOT NULL,
    top_score  REAL NOT NULL,
    ms         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_photos_event  ON photos(event_id, state);
CREATE INDEX IF NOT EXISTS idx_faces_event   ON faces(event_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo   ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(event_id, cluster_id);
"""

_local = threading.local()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def get_conn() -> sqlite3.Connection:
    """A connection private to the calling thread (SQLite objects are not shared)."""
    settings = get_settings()
    path = settings.db_path
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == str(path):
        return conn
    if conn is not None:
        conn.close()
    settings.ensure_dirs()
    conn = _connect(path)
    _local.conn = conn
    _local.path = str(path)
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def close_all() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
        _local.path = None
