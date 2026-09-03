"""Ingest: bytes on the wire to faces in the index.

Upload and indexing are deliberately decoupled. An organiser dropping 2,000
photos into the browser should get "received" back in seconds and watch a
progress bar, not hold an HTTP connection open for ten minutes. So the upload
path does only cheap work (hash, decode, thumbnail, insert a ``pending`` row)
and a background pool does the detection.

Crash recovery falls out of that for free: ``pending`` is a durable state in
SQLite, so a process that dies mid-index resumes on the next start rather than
losing the queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import repo, security, storage
from .config import get_settings
from .engines import get_engine
from .imaging import ImageError, open_rgb, sha256_bytes, taken_at, to_bgr
from .matcher import invalidate

log = logging.getLogger("smriti.pipeline")


@dataclass(frozen=True)
class IngestResult:
    photo_id: str | None
    filename: str
    status: str  # "queued" | "duplicate" | "rejected"
    detail: str = ""


def ingest_bytes(event: dict, filename: str, data: bytes) -> IngestResult:
    """Store one uploaded photo and queue it for indexing."""
    settings = get_settings()
    if len(data) > settings.max_upload_bytes:
        return IngestResult(None, filename, "rejected",
                            f"larger than {settings.max_upload_mb} MB")
    if not data:
        return IngestResult(None, filename, "rejected", "empty file")

    digest = sha256_bytes(data)
    existing = repo.find_photo_by_sha(event["id"], digest)
    if existing:
        # The same photo arriving twice is the normal case, not an error:
        # people re-drag a folder, or two phones share the same shot.
        return IngestResult(existing["id"], filename, "duplicate")

    try:
        img = open_rgb(data)
    except ImageError as exc:
        return IngestResult(None, filename, "rejected", str(exc))

    photo_id = security.new_photo_id()
    ext = storage.extension_for(img, filename)
    orig = storage.write_original(event["id"], photo_id, ext, data)
    thumb = storage.write_thumb(event["id"], photo_id, img)

    repo.add_photo(
        event_id=event["id"], photo_id=photo_id, orig_name=filename,
        rel_path=storage.relative(orig), thumb_rel=storage.relative(thumb),
        sha256=digest, width=img.width, height=img.height,
        nbytes=len(data), taken_at=taken_at(data),
    )
    return IngestResult(photo_id, filename, "queued")


def index_photo(event: dict, photo_id: str) -> int:
    """Detect and embed every face in one stored photo. Returns the face count."""
    photo = repo.get_photo(photo_id)
    if photo is None:
        return 0
    try:
        path = storage.absolute(photo["rel_path"])
        bgr = to_bgr(open_rgb(path.read_bytes()))
        engine = get_engine(event["engine"])
        faces = engine.detect_and_embed(bgr)
        n = repo.replace_faces(event["id"], photo_id, faces)
        repo.set_photo_state(photo_id, "indexed", n_faces=n)
        invalidate(event["id"])
        return n
    except Exception as exc:  # one bad file must not stop the queue
        log.warning("indexing failed for %s: %s", photo_id, exc)
        repo.set_photo_state(photo_id, "failed", error=str(exc)[:500])
        return 0


class Indexer:
    """A small background pool that drains the ``pending`` photo queue.

    One pool for the whole process rather than one per event: the bottleneck is
    CPU-bound inference, so more threads than cores would only add contention.
    """

    def __init__(self, workers: int | None = None) -> None:
        self.workers = workers or get_settings().workers
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._pool: ThreadPoolExecutor | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._inflight = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for i in range(self.workers):
            t = threading.Thread(target=self._run, name=f"smriti-index-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("indexer started with %d workers", self.workers)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for _ in self._threads:
            self._queue.put(("", ""))  # wake each worker so it can observe the flag
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def resume_pending(self) -> int:
        """Re-queue everything left ``pending`` by a previous run."""
        total = 0
        for event in repo.list_events():
            ids = repo.pending_photo_ids(event["id"])
            for photo_id in ids:
                self.submit(event["id"], photo_id)
            total += len(ids)
        if total:
            log.info("resumed %d pending photos", total)
        return total

    # -- queue -------------------------------------------------------------
    def submit(self, event_id: str, photo_id: str) -> None:
        with self._lock:
            self._inflight += 1
            self._idle.clear()
        self._queue.put((event_id, photo_id))

    @property
    def depth(self) -> int:
        with self._lock:
            return self._inflight

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the queue drains. Used by the CLI and the tests."""
        return self._idle.wait(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            event_id, photo_id = self._queue.get()
            try:
                if not event_id or self._stop.is_set():
                    continue
                event = repo.get_event(event_id)
                if event is not None:
                    index_photo(event, photo_id)
            except Exception:  # pragma: no cover - defensive
                log.exception("indexer worker crashed on %s", photo_id)
            finally:
                self._queue.task_done()
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                    if self._inflight == 0:
                        self._idle.set()


_indexer: Indexer | None = None


def get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer


def shutdown_indexer() -> None:
    global _indexer
    if _indexer is not None:
        _indexer.stop()
        _indexer = None


def index_event_sync(event: dict, *, progress=None) -> dict:
    """Index every pending photo in this thread. The CLI path, and the tests'."""
    started = time.time()
    ids = repo.pending_photo_ids(event["id"])
    faces = 0
    for i, photo_id in enumerate(ids, 1):
        faces += index_photo(event, photo_id)
        if progress:
            progress(i, len(ids))
    return {"photos": len(ids), "faces": faces, "seconds": round(time.time() - started, 2)}
