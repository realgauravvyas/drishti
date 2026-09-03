"""Default engine: YuNet detector + SFace recogniser, both through OpenCV DNN.

The default because it needs nothing beyond ``opencv-python``, the two models
together are under 40 MB, and it runs on a CPU-only laptop. Accuracy is below
ArcFace-R50 -- see ``arcface.py`` for the upgrade path. Alignment and embedding
always use full-resolution pixels even though detection ran on a downscaled
copy: a face 60 px wide in a group shot has few enough pixels already.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import weights
from ..config import get_settings
from ..imaging import blur_score
from ..pool import ModelPool
from .base import DetectedFace, FaceEngine, l2_normalise
from .yunet import YuNetDetector


class SFaceEngine(FaceEngine):
    name = "sface"
    dim = 128
    # OpenCV's reference cosine threshold for SFace is 0.363. We sit just above
    # it: a false positive hands a stranger's photo to the wrong person, which
    # is a privacy incident, while a false negative only demotes a photo into
    # the reviewable "maybe" tier.
    threshold = 0.38
    threshold_high = 0.52
    threshold_low = 0.30

    def __init__(self) -> None:
        self.detector = YuNetDetector()
        self._rec_path = str(weights.ensure("sface"))
        self._pool: ModelPool = ModelPool(
            lambda: cv2.FaceRecognizerSF.create(self._rec_path, ""),
            get_settings().workers + 2, name="sface")

    def warmup(self) -> None:
        # Build every pooled instance now. Otherwise the cost of loading a
        # 37 MB model lands on whichever guest happens to arrive on a cold
        # thread, which is exactly the request you least want to be slow.
        self.detector.prewarm()
        self._pool.prewarm()
        super().warmup()

    def detect_and_embed(self, bgr: np.ndarray) -> list[DetectedFace]:
        rows = self.detector.detect(bgr)
        if len(rows) == 0:
            return []
        out: list[DetectedFace] = []
        with self._pool.borrow() as rec:
            for row in rows:
                aligned = rec.alignCrop(bgr, row.reshape(1, -1))
                embedding = l2_normalise(rec.feature(aligned))
                if not np.any(embedding):
                    continue
                x, y, w, h = (int(round(float(v))) for v in row[:4])
                out.append(DetectedFace(
                    x=x, y=y, w=w, h=h,
                    det_score=float(row[14]),
                    embedding=embedding,
                    landmarks=row[4:14].reshape(5, 2).copy(),
                    blur=blur_score(bgr[y:y + h, x:x + w]),
                ))
        return out
