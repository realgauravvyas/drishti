"""Where the bytes live.

Layout, one directory per event so that deleting an event is a single
``rmtree`` and cannot leave orphans behind::

    data/photos/<event_id>/orig/<photo_id>.<ext>
    data/photos/<event_id>/thumb/<photo_id>.jpg

Originals are never modified or re-encoded: whatever the organiser uploaded is
what a guest downloads.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from .config import get_settings
from .imaging import make_thumb

_EXT_BY_FORMAT = {
    "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
    "BMP": ".bmp", "TIFF": ".tif", "HEIF": ".heic",
}


def event_dir(event_id: str) -> Path:
    return get_settings().photos_dir / event_id


def orig_path(event_id: str, photo_id: str, ext: str) -> Path:
    return event_dir(event_id) / "orig" / f"{photo_id}{ext}"


def thumb_path(event_id: str, photo_id: str) -> Path:
    return event_dir(event_id) / "thumb" / f"{photo_id}.jpg"


def extension_for(img: Image.Image, original_name: str) -> str:
    suffix = Path(original_name).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return _EXT_BY_FORMAT.get((img.format or "").upper(), ".jpg")


def write_original(event_id: str, photo_id: str, ext: str, data: bytes) -> Path:
    path = orig_path(event_id, photo_id, ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_thumb(event_id: str, photo_id: str, img: Image.Image) -> Path:
    path = thumb_path(event_id, photo_id)
    make_thumb(img, path, max_px=get_settings().thumb_px)
    return path


def relative(path: Path) -> str:
    """Store paths relative to the data dir so the data dir can be moved."""
    try:
        return str(path.relative_to(get_settings().data_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def absolute(rel: str) -> Path:
    """Resolve a stored relative path, refusing anything that escapes the data dir."""
    root = get_settings().data_dir.resolve()
    path = (root / rel).resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"path escapes the data directory: {rel}")
    return path


def delete_event_files(event_id: str) -> None:
    shutil.rmtree(event_dir(event_id), ignore_errors=True)


def delete_photo_files(event_id: str, photo_id: str) -> None:
    for path in (event_dir(event_id) / "orig").glob(f"{photo_id}.*"):
        path.unlink(missing_ok=True)
    thumb_path(event_id, photo_id).unlink(missing_ok=True)


def disk_usage(event_id: str) -> int:
    root = event_dir(event_id)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
