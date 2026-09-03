"""Accuracy engine: YuNet detection + ArcFace R50 (buffalo_l) embeddings.

Same detector as the default engine, but the 128-d SFace head is replaced with
the 512-d ArcFace R50 recogniser from the buffalo_l pack, run directly on
onnxruntime. That skips the ``insightface`` package and its model-zoo
downloader while getting the embedding quality that actually matters for a
100-person album, where a 1-in-500 confusion is a visible product failure.

Requires: ``pip install onnxruntime`` and a 166 MB weight download.
"""

from __future__ import annotations

import numpy as np

from .. import weights
from ..config import get_settings
from ..imaging import blur_score
from ..pool import ModelPool
from .base import DetectedFace, FaceEngine, align_112
from .yunet import YuNetDetector


class ArcFaceEngine(FaceEngine):
    name = "arcface"
    dim = 512
    # ArcFace R50 on normalised embeddings. Published operating points put the
    # 1e-4 FAR threshold near 0.28; scripts/benchmark.py on an 11-person set
    # shows zero false matches down to 0.25 while recall keeps climbing. 0.32
    # sits above the published point with margin, rather than at the 0.40 that
    # a first guess suggested and the measurement showed was costing recall for
    # no safety gain. Re-run the benchmark on your own photos before moving it.
    threshold = 0.32
    threshold_high = 0.45
    threshold_low = 0.22

    def __init__(self) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "engine 'arcface' needs onnxruntime -- pip install onnxruntime"
            ) from exc
        self.detector = YuNetDetector()
        self._model_path = str(weights.ensure("arcface"))
        self._pool: ModelPool = ModelPool(
            self._build_session, get_settings().workers + 2, name="arcface")

    def _build_session(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # we parallelise over photos, not within one
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The exported graph declares a fixed output shape of (1, 512) but
        # batches correctly anyway, so every multi-face photo would otherwise
        # print a shape-mismatch warning. Errors still surface.
        options.log_severity_level = 3
        session = ort.InferenceSession(self._model_path, options, providers=_providers())
        return session, session.get_inputs()[0].name

    def warmup(self) -> None:
        self.detector.prewarm()
        self._pool.prewarm()
        super().warmup()

    def _embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        # BGR -> RGB, scaled to [-1, 1], NCHW: standard ArcFace preprocessing.
        batch = np.stack(crops).astype(np.float32)[..., ::-1]
        batch = (batch - 127.5) / 127.5
        batch = np.ascontiguousarray(np.transpose(batch, (0, 3, 1, 2)))
        with self._pool.borrow() as (session, input_name):
            out = session.run(None, {input_name: batch})[0]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-8)).astype(np.float32)

    def detect_and_embed(self, bgr: np.ndarray) -> list[DetectedFace]:
        rows = self.detector.detect(bgr)
        if len(rows) == 0:
            return []
        crops = [align_112(bgr, row[4:14].reshape(5, 2)) for row in rows]
        embeddings = self._embed_batch(crops)
        out: list[DetectedFace] = []
        for row, embedding in zip(rows, embeddings):
            x, y, w, h = (int(round(float(v))) for v in row[:4])
            out.append(DetectedFace(
                x=x, y=y, w=w, h=h,
                det_score=float(row[14]),
                embedding=embedding,
                landmarks=row[4:14].reshape(5, 2).copy(),
                blur=blur_score(bgr[y:y + h, x:x + w]),
            ))
        return out


def _providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or ["CPUExecutionProvider"]
