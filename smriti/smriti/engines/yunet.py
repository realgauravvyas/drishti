"""YuNet detection, shared by every real engine.

Detection and recognition are separable problems and we solve them separately:
YuNet finds boxes plus five landmarks, and whichever recogniser is configured
turns each aligned crop into a vector. This module is the detection half.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import weights
from ..config import get_settings
from ..imaging import fit_within
from ..pool import ModelPool


class YuNetDetector:
    """Pooled wrapper around OpenCV's YuNet.

    The OpenCV object carries mutable input-size state, so it cannot be shared
    between concurrent callers; instances are borrowed from a bounded pool
    rather than created per thread (see ``smriti/pool.py``).
    """

    def __init__(self, detect_size: int | None = None,
                 score_min: float | None = None,
                 min_face_px: int | None = None,
                 pool_size: int | None = None) -> None:
        settings = get_settings()
        self.model_path = str(weights.ensure("yunet"))
        self.detect_size = detect_size or settings.detect_size
        self.score_min = score_min if score_min is not None else settings.det_score_min
        self.min_face_px = min_face_px if min_face_px is not None else settings.min_face_px
        self._pool: ModelPool = ModelPool(
            self._build, pool_size or settings.workers + 2, name="yunet")

    def _build(self):
        return cv2.FaceDetectorYN.create(
            self.model_path, "", (320, 320), self.score_min, 0.3, 5000
        )

    def prewarm(self) -> None:
        self._pool.prewarm()

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Return an (N, 15) array in ORIGINAL image coordinates.

        Columns follow OpenCV's YuNet layout: x, y, w, h, then five (x, y)
        landmarks, then the detection score.
        """
        if bgr is None or bgr.size == 0:
            return np.zeros((0, 15), dtype=np.float32)

        # Detect on a downscaled copy. A 48 MP phone photo costs seconds at full
        # resolution and ~100 ms at 1280 px, with no recall loss for faces large
        # enough to embed meaningfully.
        small, scale = fit_within(bgr, self.detect_size)
        height, width = small.shape[:2]
        with self._pool.borrow() as model:
            model.setInputSize((width, height))
            _, raw = model.detect(small)
        if raw is None or len(raw) == 0:
            return np.zeros((0, 15), dtype=np.float32)

        rows = np.asarray(raw, dtype=np.float32).reshape(-1, 15).copy()
        rows[:, :14] *= scale  # geometry back to original pixels; col 14 is the score

        full_h, full_w = bgr.shape[:2]
        keep = []
        for row in rows:
            if row[14] < self.score_min:
                continue
            x, y = max(0.0, float(row[0])), max(0.0, float(row[1]))
            w = min(float(row[2]), full_w - x)
            h = min(float(row[3]), full_h - y)
            if w < self.min_face_px or h < self.min_face_px:
                continue
            row[0], row[1], row[2], row[3] = x, y, w, h
            keep.append(row)
        if not keep:
            return np.zeros((0, 15), dtype=np.float32)
        return np.vstack(keep).astype(np.float32)
