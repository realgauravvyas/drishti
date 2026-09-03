"""Data access. Every SQL statement in the project lives here.

Keeping the queries in one module means the API layer never has to think about
transactions or row shapes, and the schema can be tuned without hunting SQL
through request handlers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from . import security, storage
from .db import get_conn, transaction
from .engines.base import DetectedFace


@dataclass(frozen=True)
class NewEvent:
    event: dict
    admin_token: str  # returned once, never stored in plaintext


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else None


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
def create_event(name: str, engine: str, embed_dim: int, *,
                 retention_days: int = 0, notes: str = "",
                 allow_download: bool = True) -> NewEvent:
    event_id = security.new_event_id()
    token = security.new_admin_token()
    now = time.time()
    expires_at = now + retention_days * 86400 if retention_days > 0 else None

    # A collision on an 8-char code is ~1 in 8.5e11, but retrying is free.
    for _ in range(5):
        code = security.new_share_code()
        try:
            with transaction() as conn:
                conn.execute(
                    "INSERT INTO events (id, name, share_code, admin_token_hash, engine,"
                    " embed_dim, created_at, expires_at, allow_download, notes)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (event_id, name.strip() or "Untitled event", code,
                     security.hash_token(token), engine, embed_dim, now, expires_at,
                     int(allow_download), notes),
                )
            break
        except Exception as exc:  # pragma: no cover - collision path
            if "share_code" not in str(exc):
                raise
    else:  # pragma: no cover
        raise RuntimeError("could not allocate a unique share code")

    return NewEvent(event=get_event(event_id), admin_token=token)


def get_event(event_id: str) -> dict | None:
    return _row_to_dict(
        get_conn().execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    )


def get_event_by_code(code: str) -> dict | None:
    return _row_to_dict(
        get_conn().execute(
            "SELECT * FROM events WHERE share_code = ?", (security.normalise_code(code),)
        ).fetchone()
    )


def list_events() -> list[dict]:
    rows = get_conn().execute(
        "SELECT e.*,"
        " (SELECT COUNT(*) FROM photos p WHERE p.event_id = e.id) AS n_photos,"
        " (SELECT COUNT(*) FROM faces f WHERE f.event_id = e.id) AS n_faces"
        " FROM events e ORDER BY e.created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_event(event_id: str) -> bool:
    with transaction() as conn:
        cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.execute("DELETE FROM faces WHERE event_id = ?", (event_id,))
        conn.execute("DELETE FROM photos WHERE event_id = ?", (event_id,))
        conn.execute("DELETE FROM search_log WHERE event_id = ?", (event_id,))
        deleted = cur.rowcount > 0
    storage.delete_event_files(event_id)
    return deleted


def purge_expired(now: float | None = None) -> list[str]:
    """Delete events past their retention date. Returns the ids removed."""
    now = now or time.time()
    rows = get_conn().execute(
        "SELECT id FROM events WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
    ).fetchall()
    ids = [r["id"] for r in rows]
    for event_id in ids:
        delete_event(event_id)
    return ids


def event_stats(event_id: str) -> dict:
    conn = get_conn()
    photos = conn.execute(
        "SELECT state, COUNT(*) AS n FROM photos WHERE event_id = ? GROUP BY state",
        (event_id,),
    ).fetchall()
    by_state = {r["state"]: r["n"] for r in photos}
    n_faces = conn.execute(
        "SELECT COUNT(*) AS n FROM faces WHERE event_id = ?", (event_id,)
    ).fetchone()["n"]
    n_searches = conn.execute(
        "SELECT COUNT(*) AS n FROM search_log WHERE event_id = ?", (event_id,)
    ).fetchone()["n"]
    total = sum(by_state.values())
    return {
        "photos_total": total,
        "photos_indexed": by_state.get("indexed", 0),
        "photos_pending": by_state.get("pending", 0),
        "photos_failed": by_state.get("failed", 0),
        "faces": n_faces,
        "searches": n_searches,
        "bytes": storage.disk_usage(event_id),
    }


# --------------------------------------------------------------------------
# photos
# --------------------------------------------------------------------------
def find_photo_by_sha(event_id: str, sha256: str) -> dict | None:
    return _row_to_dict(
        get_conn().execute(
            "SELECT * FROM photos WHERE event_id = ? AND sha256 = ?", (event_id, sha256)
        ).fetchone()
    )


def add_photo(event_id: str, photo_id: str, orig_name: str, rel_path: str,
              thumb_rel: str | None, sha256: str, width: int, height: int,
              nbytes: int, taken_at: float | None) -> dict:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO photos (id, event_id, orig_name, rel_path, thumb_path, sha256,"
            " width, height, bytes, taken_at, state, n_faces, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,'pending',0,?)",
            (photo_id, event_id, orig_name, rel_path, thumb_rel, sha256,
             width, height, nbytes, taken_at, time.time()),
        )
    return get_photo(photo_id)


def get_photo(photo_id: str) -> dict | None:
    return _row_to_dict(
        get_conn().execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    )


def list_photos(event_id: str, state: str | None = None,
                limit: int = 5000, offset: int = 0) -> list[dict]:
    sql = "SELECT * FROM photos WHERE event_id = ?"
    args: list[Any] = [event_id]
    if state:
        sql += " AND state = ?"
        args.append(state)
    sql += " ORDER BY COALESCE(taken_at, created_at), id LIMIT ? OFFSET ?"
    args += [limit, offset]
    return [dict(r) for r in get_conn().execute(sql, args).fetchall()]


def pending_photo_ids(event_id: str | None = None) -> list[str]:
    sql = "SELECT id FROM photos WHERE state = 'pending'"
    args: list[Any] = []
    if event_id:
        sql += " AND event_id = ?"
        args.append(event_id)
    return [r["id"] for r in get_conn().execute(sql, args).fetchall()]


def set_photo_state(photo_id: str, state: str, *, n_faces: int = 0,
                    error: str | None = None) -> None:
    with transaction() as conn:
        conn.execute(
            "UPDATE photos SET state = ?, n_faces = ?, error = ? WHERE id = ?",
            (state, n_faces, error, photo_id),
        )


def delete_photo(event_id: str, photo_id: str) -> bool:
    with transaction() as conn:
        conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        cur = conn.execute(
            "DELETE FROM photos WHERE id = ? AND event_id = ?", (photo_id, event_id)
        )
        deleted = cur.rowcount > 0
    storage.delete_photo_files(event_id, photo_id)
    return deleted


def reset_index(event_id: str) -> None:
    """Drop every face and mark all photos for re-indexing."""
    with transaction() as conn:
        conn.execute("DELETE FROM faces WHERE event_id = ?", (event_id,))
        conn.execute(
            "UPDATE photos SET state = 'pending', n_faces = 0, error = NULL"
            " WHERE event_id = ?",
            (event_id,),
        )


# --------------------------------------------------------------------------
# faces
# --------------------------------------------------------------------------
def replace_faces(event_id: str, photo_id: str, faces: Sequence[DetectedFace]) -> int:
    """Write the faces for one photo, replacing anything previously stored."""
    now = time.time()
    rows = [
        (event_id, photo_id, f.x, f.y, f.w, f.h, float(f.det_score), f.face_px,
         float(f.blur), np.asarray(f.embedding, dtype=np.float32).tobytes(), now)
        for f in faces
    ]
    with transaction() as conn:
        conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        if rows:
            conn.executemany(
                "INSERT INTO faces (event_id, photo_id, x, y, w, h, det_score, face_px,"
                " blur, embedding, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
    return len(rows)


def load_face_matrix(event_id: str, dim: int) -> tuple[np.ndarray, list[int], list[str], np.ndarray]:
    """Return (matrix (N, dim), face_ids, photo_ids, face_px) for one event."""
    rows = get_conn().execute(
        "SELECT id, photo_id, embedding, face_px FROM faces WHERE event_id = ? ORDER BY id",
        (event_id,),
    ).fetchall()
    if not rows:
        return np.zeros((0, dim), np.float32), [], [], np.zeros((0,), np.float32)
    matrix = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
    matrix = matrix.reshape(len(rows), -1)
    if matrix.shape[1] != dim:  # pragma: no cover - guarded by the engine lock
        raise ValueError(
            f"stored embeddings are {matrix.shape[1]}-d but the engine emits {dim}-d; "
            f"re-index this event"
        )
    return (matrix,
            [r["id"] for r in rows],
            [r["photo_id"] for r in rows],
            np.array([r["face_px"] for r in rows], dtype=np.float32))


def face_fingerprint(event_id: str) -> tuple[int, int]:
    """(count, max_id) — changes whenever the event's face set changes.

    Used as a cache key for the in-memory search index. Deleting a face lowers
    the count; adding one raises the max id; doing both at once still changes
    the pair, because SQLite's AUTOINCREMENT never reuses an id.
    """
    row = get_conn().execute(
        "SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM faces WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    return int(row["n"]), int(row["m"])


def faces_for_photos(photo_ids: Iterable[str]) -> dict[str, list[dict]]:
    ids = list(photo_ids)
    if not ids:
        return {}
    out: dict[str, list[dict]] = {}
    conn = get_conn()
    for chunk_start in range(0, len(ids), 500):  # SQLite parameter-count limit
        chunk = ids[chunk_start:chunk_start + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT id, photo_id, x, y, w, h, det_score, face_px, blur"
            f" FROM faces WHERE photo_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            out.setdefault(row["photo_id"], []).append(dict(row))
    return out


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------
def log_search(event_id: str, n_queries: int, n_matches: int,
               top_score: float, ms: float) -> None:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO search_log (event_id, ts, n_queries, n_matches, top_score, ms)"
            " VALUES (?,?,?,?,?,?)",
            (event_id, time.time(), n_queries, n_matches, float(top_score), float(ms)),
        )


def recent_searches(event_id: str, limit: int = 50) -> list[dict]:
    rows = get_conn().execute(
        "SELECT ts, n_queries, n_matches, top_score, ms FROM search_log"
        " WHERE event_id = ? ORDER BY ts DESC LIMIT ?",
        (event_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]
