"""Tokens, share codes and rate limiting.

The threat model is modest but real: the share link travels through WhatsApp
groups, so anyone who has it can search the event. What must NOT be possible is
(a) guessing a share code for an event you were never invited to, (b) editing or
deleting an event without the organiser token, or (c) grinding selfies against
an event to enumerate who attended.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time

# Unambiguous alphabet: no 0/O, 1/I/L. Share codes get read aloud and retyped
# from a photo of a screen, so the character set matters more than the entropy.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LEN = 8  # 31^8 ~= 8.5e11 combinations


def new_event_id() -> str:
    return secrets.token_hex(8)


def new_photo_id() -> str:
    return secrets.token_hex(8)


def new_share_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))


def new_admin_token() -> str:
    """High-entropy secret shown to the organiser exactly once."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Tokens are random 256-bit secrets, so a plain SHA-256 is sufficient:
    there is no low-entropy password here for a dictionary attack to bite on."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_token(token), stored_hash)


def normalise_code(code: str) -> str:
    return "".join(ch for ch in code.strip().upper() if ch.isalnum())


def hash_ip(ip: str) -> str:
    """Truncated salted hash — enough to rate-limit, not enough to identify."""
    salt = _process_salt()
    return hashlib.sha256((salt + ip).encode("utf-8")).hexdigest()[:16]


_SALT: str | None = None


def _process_salt() -> str:
    global _SALT
    if _SALT is None:
        _SALT = secrets.token_hex(16)
    return _SALT


def server_secret() -> bytes:
    """A per-installation signing key, persisted so restarts don't invalidate
    outstanding links. Created with 0600 permissions where the OS honours them.
    """
    global _SERVER_SECRET
    if _SERVER_SECRET is not None:
        return _SERVER_SECRET
    from .config import get_settings

    path = get_settings().data_dir / "secret.key"
    if path.exists():
        _SERVER_SECRET = path.read_bytes().strip()
    else:
        _SERVER_SECRET = secrets.token_bytes(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_SERVER_SECRET)
        try:
            path.chmod(0o600)
        except OSError:  # pragma: no cover - Windows/FS dependent
            pass
    return _SERVER_SECRET


_SERVER_SECRET: bytes | None = None

# Photo access tokens. A guest who matches 12 photos gets 12 signed URLs and can
# reach exactly those 12 -- knowing an event's share code does not grant a right
# to browse everyone else's pictures.
_PHOTO_TOKEN_TTL = 12 * 3600


def sign_photo(photo_id: str, event_id: str, ttl: int = _PHOTO_TOKEN_TTL) -> str:
    expiry = int(time.time() + ttl)
    payload = f"{event_id}:{photo_id}:{expiry}".encode("utf-8")
    mac = hmac.new(server_secret(), payload, hashlib.sha256).hexdigest()[:32]
    return f"{expiry}.{mac}"


def verify_photo(token: str, photo_id: str, event_id: str) -> bool:
    try:
        expiry_str, mac = token.split(".", 1)
        expiry = int(expiry_str)
    except (ValueError, AttributeError):
        return False
    if expiry < time.time():
        return False
    payload = f"{event_id}:{photo_id}:{expiry}".encode("utf-8")
    expected = hmac.new(server_secret(), payload, hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(expected, mac)


class RateLimiter:
    """Fixed-window counter, per key. In-process and therefore per-worker.

    Deliberately simple: it exists to blunt selfie-grinding and upload floods on
    a single-node deployment. A multi-node deployment should put a real limiter
    (Redis, or the reverse proxy) in front and can leave this one in place.
    """

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, seconds_until_reset)."""
        now = time.monotonic()
        with self._lock:
            start, count = self._hits.get(key, (now, 0))
            if now - start >= self.window:
                start, count = now, 0
            count += 1
            self._hits[key] = (start, count)
            if len(self._hits) > 10_000:  # cheap unbounded-growth guard
                cutoff = now - self.window
                self._hits = {k: v for k, v in self._hits.items() if v[0] > cutoff}
            return count <= self.limit, max(0.0, self.window - (now - start))

    def reset(self) -> None:
        """Forget every counter. For tests, and for an operator unblocking a
        shared-NAT office that tripped the limit as one client."""
        with self._lock:
            self._hits.clear()
