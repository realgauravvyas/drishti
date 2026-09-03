"""Deterministic stand-in engine for tests and CI.

It does no machine learning at all: it finds solid-ish rectangular "faces" that
the test fixtures paint, and derives an embedding from the patch's colour so
that two crops of the same synthetic person match and two different ones do
not. That keeps the entire pipeline — ingest, indexing, matching, the API,
the download ZIP — testable in under a second with no weights on disk.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import DetectedFace, FaceEngine, l2_normalise

DIM = 32


class MockEngine(FaceEngine):
    name = "mock"
    dim = DIM
    threshold = 0.90
    threshold_high = 0.95
    threshold_low = 0.80

    def detect_and_embed(self, bgr: np.ndarray) -> list[DetectedFace]:
        if bgr is None or bgr.size == 0:
            return []
        # A synthetic "face" is any saturated non-background blob.
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 90, 60), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        faces: list[DetectedFace] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 16 or h < 16:
                continue
            patch = bgr[y:y + h, x:x + w]
            faces.append(DetectedFace(
                x=x, y=y, w=w, h=h,
                det_score=0.99,
                embedding=self._embed(patch),
                landmarks=None,
                blur=100.0,
            ))
        faces.sort(key=lambda f: (-f.w * f.h))
        return faces

    @staticmethod
    def _embed(patch: np.ndarray) -> np.ndarray:
        """Hue histogram — identity here is literally "what colour are you"."""
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [DIM], [0, 180]).ravel().astype(np.float32)
        return l2_normalise(hist + 1e-3)
