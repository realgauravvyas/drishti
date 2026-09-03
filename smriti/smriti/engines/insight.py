"""Optional engine: the full InsightFace ``buffalo_l`` pipeline.

A thin adapter, so the project can ride InsightFace's detector improvements
(and its pose/age heads, if ever wanted) without owning that logic here.
Heavier to install than ``arcface`` for the same recognition weights, so it is
opt-in rather than default.

Requires: ``pip install insightface onnxruntime``.
"""

from __future__ import annotations

import numpy as np

from ..config import get_settings
from ..imaging import blur_score
from ..pool import ModelPool
from .base import DetectedFace, FaceEngine, l2_normalise


class InsightFaceEngine(FaceEngine):
    name = "insightface"
    dim = 512
    threshold = 0.40
    threshold_high = 0.55
    threshold_low = 0.28

    def __init__(self, pack: str = "buffalo_l") -> None:
        try:
            from insightface.app import FaceAnalysis  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "engine 'insightface' needs: pip install insightface onnxruntime"
            ) from exc
        settings = get_settings()
        self.pack = pack
        self.detect_size = settings.detect_size
        self.det_score_min = settings.det_score_min
        self.min_face_px = settings.min_face_px
        self._pool: ModelPool = ModelPool(self._build, settings.workers + 2,
                                          name="insightface")

    def _build(self):
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=self.pack,
            root=str(get_settings().weights_dir),
            allowed_modules=["detection", "recognition"],
        )
        # InsightFace pads to a square and wants a multiple of 32.
        size = max(320, (self.detect_size // 32) * 32)
        app.prepare(ctx_id=0, det_size=(size, size), det_thresh=self.det_score_min)
        return app

    def warmup(self) -> None:
        self._pool.prewarm()
        super().warmup()

    def detect_and_embed(self, bgr: np.ndarray) -> list[DetectedFace]:
        if bgr is None or bgr.size == 0:
            return []
        with self._pool.borrow() as app:
            faces = app.get(bgr)
        out: list[DetectedFace] = []
        height, width = bgr.shape[:2]
        for face in faces:
            x1, y1, x2, y2 = (int(round(float(v))) for v in face.bbox)
            x, y = max(0, x1), max(0, y1)
            w, h = min(x2 - x, width - x), min(y2 - y, height - y)
            if w < self.min_face_px or h < self.min_face_px:
                continue
            normed = getattr(face, "normed_embedding", None)
            embedding = l2_normalise(normed if normed is not None else face.embedding)
            if not np.any(embedding):
                continue
            kps = getattr(face, "kps", None)
            out.append(DetectedFace(
                x=x, y=y, w=w, h=h,
                det_score=float(face.det_score),
                embedding=embedding,
                landmarks=None if kps is None else np.asarray(kps, np.float32),
                blur=blur_score(bgr[y:y + h, x:x + w]),
            ))
        return out
