"""Engine registry.

An engine is chosen once per event and recorded on the event row, because
embeddings from different models are not comparable: re-pointing a populated
event at a different engine silently makes every stored vector meaningless.
Changing engines therefore means re-indexing, and the API enforces that.
"""

from __future__ import annotations

import threading

from .base import DetectedFace, FaceEngine, align_112, l2_normalise, order_landmarks

ENGINE_NAMES = ("sface", "arcface", "insightface", "mock")

_cache: dict[str, FaceEngine] = {}
_lock = threading.Lock()


def _build(name: str) -> FaceEngine:
    if name == "sface":
        from .sface import SFaceEngine

        return SFaceEngine()
    if name == "arcface":
        from .arcface import ArcFaceEngine

        return ArcFaceEngine()
    if name == "insightface":
        from .insight import InsightFaceEngine

        return InsightFaceEngine()
    if name == "mock":
        from .mock import MockEngine

        return MockEngine()
    raise ValueError(f"unknown engine {name!r}; choose one of {', '.join(ENGINE_NAMES)}")


def get_engine(name: str | None = None) -> FaceEngine:
    """Return the shared engine instance for ``name`` (default: configured one).

    Engines are expensive to construct (model load) and cheap to share: each one
    keeps its per-thread inference state internally, so a single instance backs
    the whole worker pool.
    """
    if name is None:
        from ..config import get_settings

        name = get_settings().engine
    name = name.strip().lower()
    engine = _cache.get(name)
    if engine is not None:
        return engine
    with _lock:
        if name not in _cache:
            _cache[name] = _build(name)
        return _cache[name]


def clear_cache() -> None:
    with _lock:
        _cache.clear()


__all__ = [
    "DetectedFace",
    "FaceEngine",
    "ENGINE_NAMES",
    "get_engine",
    "clear_cache",
    "align_112",
    "l2_normalise",
    "order_landmarks",
]
