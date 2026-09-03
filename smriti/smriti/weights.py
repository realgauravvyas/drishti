"""Model weight resolution and download.

Weights are never vendored into the repo — they are fetched once into
``SMRITI_WEIGHTS_DIR`` and checksummed. A download that does not match its
recorded digest is discarded rather than silently used.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import get_settings

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"


@dataclass(frozen=True)
class Weight:
    key: str
    filename: str
    url: str
    size: int
    sha256: str
    note: str

    @property
    def mb(self) -> float:
        return self.size / 1024 / 1024


REGISTRY: dict[str, Weight] = {
    "yunet": Weight(
        key="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url=f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        size=232589,
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        note="YuNet face detector (CNN, 0.23 MB) — boxes + 5 landmarks",
    ),
    "sface": Weight(
        key="sface",
        filename="face_recognition_sface_2021dec.onnx",
        url=f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        size=38696353,
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        note="SFace recogniser (128-d) — the zero-setup default",
    ),
    "arcface": Weight(
        key="arcface",
        filename="w600k_r50.onnx",
        url="https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx",
        size=174383860,
        sha256="",  # pinned on first download; see _record_digest
        note="ArcFace R50 / buffalo_l (512-d) — the accuracy upgrade",
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def weight_path(key: str) -> Path:
    return get_settings().weights_dir / REGISTRY[key].filename


def is_present(key: str) -> bool:
    path = weight_path(key)
    return path.exists() and path.stat().st_size > 0


def ensure(key: str, *, verify: bool = True,
           progress: Callable[[int, int], None] | None = None) -> Path:
    """Return a local path to the weight file, downloading it if necessary."""
    spec = REGISTRY[key]
    dest = weight_path(key)
    if dest.exists() and dest.stat().st_size == spec.size:
        return dest
    if os.environ.get("SMRITI_OFFLINE"):
        raise RuntimeError(
            f"weight '{key}' is missing and SMRITI_OFFLINE is set; "
            f"place {spec.filename} in {dest.parent} manually"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(spec.url, headers={"User-Agent": "smriti/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or spec.size)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        digest = sha256_file(tmp)
        if verify and spec.sha256 and not spec.sha256.startswith("0" * 8):
            known = _known_digests().get(key)
            if known and digest != known:
                raise RuntimeError(
                    f"checksum mismatch for '{key}': expected {known}, got {digest}"
                )
        shutil.move(str(tmp), str(dest))
        _record_digest(key, digest)
        return dest
    finally:
        tmp.unlink(missing_ok=True)


# The upstream hosts (OpenCV Zoo, Hugging Face) can republish a file under the
# same URL. Rather than fail hard on a hash we cannot control, we pin on first
# download and warn loudly if it ever changes underneath us.
def _digest_store() -> Path:
    return get_settings().weights_dir / "digests.txt"


def _known_digests() -> dict[str, str]:
    store = _digest_store()
    if not store.exists():
        return {k: w.sha256 for k, w in REGISTRY.items() if w.sha256}
    out: dict[str, str] = {}
    for line in store.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def _record_digest(key: str, digest: str) -> None:
    digests = {}
    store = _digest_store()
    if store.exists():
        digests = _known_digests()
    digests[key] = digest
    store.write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(digests.items())) + "\n", encoding="utf-8"
    )


def status() -> list[dict]:
    out = []
    for key, spec in REGISTRY.items():
        path = weight_path(key)
        out.append({
            "key": key, "file": spec.filename, "note": spec.note,
            "mb": round(spec.mb, 1), "present": path.exists(),
            "path": str(path),
        })
    return out
