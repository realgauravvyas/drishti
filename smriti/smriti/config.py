"""Runtime configuration.

Every knob is an environment variable prefixed ``SMRITI_`` so the same image
runs on a laptop and on a server without a code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SMRITI_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- storage -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")).resolve())

    # --- face engine -------------------------------------------------------
    # sface | arcface | insightface | mock   (see smriti/engines/)
    engine: str = field(default_factory=lambda: _env("ENGINE", "sface"))
    # Longest side an image is resized to before detection. Bigger = slower but
    # finds smaller faces in the back of a group shot.
    detect_size: int = field(default_factory=lambda: _env_int("DETECT_SIZE", 1280))
    # Reject detections below this confidence outright.
    det_score_min: float = field(default_factory=lambda: _env_float("DET_SCORE_MIN", 0.60))
    # A face smaller than this (in px, on the original image) is too low-res to
    # embed reliably. 24px is roughly the floor for ArcFace-family models.
    min_face_px: int = field(default_factory=lambda: _env_int("MIN_FACE_PX", 28))

    # --- matching ----------------------------------------------------------
    # Cosine thresholds; 0 means "use the engine's calibrated default".
    match_threshold: float = field(default_factory=lambda: _env_float("MATCH_THRESHOLD", 0.0))
    max_results: int = field(default_factory=lambda: _env_int("MAX_RESULTS", 500))

    # --- ingest ------------------------------------------------------------
    workers: int = field(default_factory=lambda: _env_int("WORKERS", max(1, (os.cpu_count() or 4) // 2)))
    thumb_px: int = field(default_factory=lambda: _env_int("THUMB_PX", 480))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 40))

    # --- privacy / lifecycle ----------------------------------------------
    # Events are auto-purged this many days after creation. 0 disables.
    retention_days: int = field(default_factory=lambda: _env_int("RETENTION_DAYS", 30))
    # Never write a guest's selfie to disk. Leave this on unless you have a
    # very good, documented reason.
    store_selfies: bool = field(default_factory=lambda: _env_bool("STORE_SELFIES", False))

    # --- server ------------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    public_url: str = field(default_factory=lambda: _env("PUBLIC_URL", ""))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", ""))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "smriti.db"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"

    @property
    def weights_dir(self) -> Path:
        return Path(_env("WEIGHTS_DIR", str(self.data_dir / "weights"))).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.photos_dir, self.weights_dir):
            p.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> None:
    """Drop the cached Settings — used by tests that patch the environment."""
    global _settings
    _settings = None
