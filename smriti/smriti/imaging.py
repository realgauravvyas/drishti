"""Image loading, EXIF handling and thumbnailing.

Phone photos are the whole input domain here, so EXIF orientation is not
optional: a portrait shot stored as landscape-with-a-rotation-flag will have
every face lying on its side, and the detector will simply miss them.
"""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 200_000_000  # decompression-bomb guard, generous for 100MP phones

# OpenCV 5 prints a WARN on every DNN target selection ("Targets are not
# supported by the new graph engine for now"). It is cosmetic, it fires once per
# model instance, and at pool scale it buries the real logs.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:  # older OpenCV builds expose no logging control
    pass

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff"}

_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME = 306


class ImageError(ValueError):
    """Raised when bytes cannot be decoded as an image we can work with."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_rgb(data: bytes) -> Image.Image:
    """Decode bytes to an upright RGB PIL image, honouring the EXIF rotation flag."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"unreadable image: {exc}") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def to_bgr(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR ndarray."""
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def load_bgr(path: str | Path) -> np.ndarray:
    return to_bgr(open_rgb(Path(path).read_bytes()))


def taken_at(data: bytes) -> float | None:
    """EXIF capture timestamp as a POSIX float, or None if absent/unparseable."""
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:
        return None
    for tag in (_EXIF_DATETIME_ORIGINAL, _EXIF_DATETIME):
        raw = exif.get(tag)
        if not raw:
            continue
        try:
            dt = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def make_thumb(img: Image.Image, dest: Path, max_px: int = 480, quality: int = 82) -> None:
    """Write a web-sized JPEG preview. Gallery views never touch the originals."""
    thumb = img.copy()
    thumb.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(dest, "JPEG", quality=quality, optimize=True)


def fit_within(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is ``max_side``. Returns (image, scale).

    ``scale`` maps *resized* coordinates back to the original: orig = resized * scale.
    """
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr, 1.0
    ratio = max_side / float(longest)
    resized = cv2.resize(bgr, (max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))),
                         interpolation=cv2.INTER_AREA)
    return resized, 1.0 / ratio


def blur_score(bgr_crop: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap sharpness proxy.

    Higher is sharper. Used only to annotate a face, never to reject one: a soft
    face in the only photo of someone is still that photo of them.
    """
    if bgr_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
