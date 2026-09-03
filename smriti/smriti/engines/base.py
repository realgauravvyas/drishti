"""The face-engine contract.

Everything above this layer — ingest, matching, the API — only ever sees
``DetectedFace``. Swapping SFace for ArcFace is a config change, not a rewrite,
and the mock engine lets the whole system be tested without a 40 MB download.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

# The canonical ArcFace 112x112 alignment template, ordered
# [left eye, right eye, nose, left mouth corner, right mouth corner] in *image*
# space (i.e. "left" = appears on the left of the picture).
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class DetectedFace:
    """One face found in one photo."""

    x: int
    y: int
    w: int
    h: int
    det_score: float
    embedding: np.ndarray  # (D,) float32, L2-normalised
    landmarks: np.ndarray | None = None  # (5, 2) float32, original-image coords
    blur: float = 0.0

    @property
    def face_px(self) -> float:
        """Geometric mean of the box sides — a single number for "how big"."""
        return float(np.sqrt(max(self.w, 1) * max(self.h, 1)))


def l2_normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def order_landmarks(pts: np.ndarray) -> np.ndarray:
    """Put 5 landmarks into ARCFACE_DST order regardless of detector convention.

    Detectors disagree on whether landmark 0 is the subject's left eye or the
    viewer's; both orderings are common and mixing them up mirrors every aligned
    crop. Sorting each pair by x makes us agnostic to that choice.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(5, 2)
    eyes = pts[:2][np.argsort(pts[:2, 0])]
    mouth = pts[3:5][np.argsort(pts[3:5, 0])]
    return np.vstack([eyes, pts[2:3], mouth]).astype(np.float32)


def align_112(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Similarity-transform a face onto the ArcFace 112x112 template."""
    src = order_landmarks(landmarks)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
    if matrix is None:  # degenerate landmarks; fall back to a plain resize
        return cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0.0)


class FaceEngine(ABC):
    """Detect + embed. Implementations must return L2-normalised embeddings."""

    name: str = "base"
    dim: int = 0
    #: Cosine similarity above which two faces are called the same person.
    #: Calibrated per model — these are not interchangeable numbers.
    threshold: float = 0.5
    #: Above this, we are confident enough to pre-select the photo for the user.
    threshold_high: float = 0.6
    #: Between this and ``threshold`` we show the photo as "maybe" for review.
    threshold_low: float = 0.4

    @abstractmethod
    def detect_and_embed(self, bgr: np.ndarray) -> list[DetectedFace]:
        """Find every face in a BGR image and embed each one."""

    def warmup(self) -> None:
        """Run one dummy inference so the first real request isn't the slow one."""
        self.detect_and_embed(np.zeros((256, 256, 3), dtype=np.uint8))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "threshold": self.threshold,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
        }
