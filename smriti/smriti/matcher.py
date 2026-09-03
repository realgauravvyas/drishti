"""Search: selfie in, ranked photos out.

The maths is deliberately boring. Every embedding is L2-normalised, so cosine
similarity is a dot product, and the whole event is one dense matrix multiply::

    scores = faces @ selfies.T        # (n_faces, n_selfies)
    per_face = scores.max(axis=1)     # best-matching selfie for each face
    per_photo = max over the faces in that photo

At 100k faces x 512 dims that is ~200 MB of float32 and about 40 ms per search
on a laptop CPU, which is well inside "feels instant" and far simpler than an
ANN index. The index is a cache in front of SQLite, rebuilt when the event's
face set changes; the point at which you outgrow it is documented in README.md.

Two product decisions live in this file:

1. **Max, not mean, over a person's selfies.** Someone who uploads a bright
   selfie and a dim one should match on whichever one is closer. Averaging the
   two blurs both.
2. **Three tiers, not one threshold.** A single cut-off has to choose between
   handing you a stranger's photo and hiding one of yours. Splitting into
   "sure / likely / maybe" lets the confident tier stay conservative while the
   long tail is still reachable behind one click.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import repo
from .engines.base import FaceEngine, l2_normalise

# Ranking nudge for tiny faces. A 30 px face in the back row of a group shot is
# both less certain and less wanted than a 300 px one at the same score. The
# penalty is small enough that it never reorders across a tier boundary.
_SIZE_REF_PX = 110.0
_SIZE_WEIGHT = 0.03


@dataclass
class PhotoMatch:
    photo_id: str
    score: float
    rank_score: float
    tier: str  # "sure" | "likely" | "maybe"
    face_id: int
    face_box: tuple[int, int, int, int]
    n_faces_in_photo: int


@dataclass
class FaceIndex:
    """An event's faces held as one contiguous matrix."""

    event_id: str
    dim: int
    matrix: np.ndarray
    face_ids: list[int]
    photo_ids: list[str]
    face_px: np.ndarray
    fingerprint: tuple[int, int]
    built_at: float = field(default_factory=time.time)

    @property
    def n_faces(self) -> int:
        return int(self.matrix.shape[0])

    def search(self, queries: np.ndarray, engine: FaceEngine,
               threshold: float | None = None, limit: int = 500) -> list[PhotoMatch]:
        if self.n_faces == 0 or queries.size == 0:
            return []
        floor = engine.threshold_low if threshold is None else threshold
        cut_match = engine.threshold if threshold is None else threshold
        cut_high = engine.threshold_high if threshold is None else max(threshold, engine.threshold_high)

        scores = self.matrix @ queries.T           # (n_faces, n_selfies)
        per_face = scores.max(axis=1)              # best selfie for each face
        hits = np.flatnonzero(per_face >= floor)
        if hits.size == 0:
            return []

        # Keep only the single best-matching face per photo: the answer to
        # "am I in this picture" is one boolean, not one per face.
        best: dict[str, tuple[float, int, int]] = {}
        for idx in hits:
            photo_id = self.photo_ids[idx]
            score = float(per_face[idx])
            current = best.get(photo_id)
            if current is None or score > current[0]:
                best[photo_id] = (score, self.face_ids[idx], int(idx))

        boxes = _face_boxes(list(best.values()))
        counts = _faces_per_photo(self.photo_ids)

        out: list[PhotoMatch] = []
        for photo_id, (score, face_id, idx) in best.items():
            size_bonus = _SIZE_WEIGHT * np.tanh(self.face_px[idx] / _SIZE_REF_PX - 1.0)
            out.append(PhotoMatch(
                photo_id=photo_id,
                score=round(score, 4),
                rank_score=round(score + float(size_bonus), 4),
                tier="sure" if score >= cut_high else "likely" if score >= cut_match else "maybe",
                face_id=face_id,
                face_box=boxes.get(face_id, (0, 0, 0, 0)),
                n_faces_in_photo=counts.get(photo_id, 0),
            ))
        out.sort(key=lambda m: -m.rank_score)
        return out[:limit]


def _face_boxes(entries: list[tuple[float, int, int]]) -> dict[int, tuple[int, int, int, int]]:
    face_ids = [e[1] for e in entries]
    if not face_ids:
        return {}
    conn_rows = []
    from .db import get_conn

    conn = get_conn()
    for start in range(0, len(face_ids), 500):
        chunk = face_ids[start:start + 500]
        placeholders = ",".join("?" * len(chunk))
        conn_rows += conn.execute(
            f"SELECT id, x, y, w, h FROM faces WHERE id IN ({placeholders})", chunk
        ).fetchall()
    return {r["id"]: (r["x"], r["y"], r["w"], r["h"]) for r in conn_rows}


def _faces_per_photo(photo_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pid in photo_ids:
        counts[pid] = counts.get(pid, 0) + 1
    return counts


# --------------------------------------------------------------------------
# index cache
# --------------------------------------------------------------------------
_cache: dict[str, FaceIndex] = {}
_lock = threading.Lock()


def get_index(event_id: str, dim: int) -> FaceIndex:
    """Return a current index for the event, rebuilding it only if faces changed."""
    fingerprint = repo.face_fingerprint(event_id)
    cached = _cache.get(event_id)
    if cached is not None and cached.fingerprint == fingerprint and cached.dim == dim:
        return cached
    with _lock:
        cached = _cache.get(event_id)
        if cached is not None and cached.fingerprint == fingerprint and cached.dim == dim:
            return cached
        matrix, face_ids, photo_ids, face_px = repo.load_face_matrix(event_id, dim)
        index = FaceIndex(
            event_id=event_id, dim=dim, matrix=np.ascontiguousarray(matrix),
            face_ids=face_ids, photo_ids=photo_ids, face_px=face_px,
            fingerprint=fingerprint,
        )
        _cache[event_id] = index
        return index


def invalidate(event_id: str | None = None) -> None:
    with _lock:
        if event_id is None:
            _cache.clear()
        else:
            _cache.pop(event_id, None)


def cache_info() -> list[dict]:
    return [
        {"event_id": k, "faces": v.n_faces, "mb": round(v.matrix.nbytes / 1e6, 1),
         "age_s": round(time.time() - v.built_at, 1)}
        for k, v in _cache.items()
    ]


# --------------------------------------------------------------------------
# query construction
# --------------------------------------------------------------------------
def build_queries(engine: FaceEngine, images: list[np.ndarray],
                  max_faces_per_selfie: int = 1) -> tuple[np.ndarray, list[dict]]:
    """Turn uploaded selfies into query vectors.

    A selfie may legitimately contain several faces (a friend leaning in), so we
    take the largest face in each image and ignore the rest. That is the least
    surprising rule: the person who took the selfie is the subject of it.
    """
    vectors: list[np.ndarray] = []
    report: list[dict] = []
    for i, bgr in enumerate(images):
        faces = engine.detect_and_embed(bgr)
        faces.sort(key=lambda f: -(f.w * f.h))
        chosen = faces[:max_faces_per_selfie]
        report.append({
            "image": i,
            "faces_found": len(faces),
            "used": len(chosen),
            "face_px": round(chosen[0].face_px, 1) if chosen else 0.0,
            "blur": round(chosen[0].blur, 1) if chosen else 0.0,
        })
        vectors.extend(l2_normalise(f.embedding) for f in chosen)
    if not vectors:
        return np.zeros((0, engine.dim), np.float32), report
    return np.ascontiguousarray(np.stack(vectors).astype(np.float32)), report
