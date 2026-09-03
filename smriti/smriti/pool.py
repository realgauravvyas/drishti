"""A bounded pool of model instances.

OpenCV's DNN objects and onnxruntime sessions are stateful and not safe to call
from several threads at once. The obvious fix -- one instance per thread via
``threading.local`` -- is quietly expensive: FastAPI hands work to a thread pool
that grows on demand, and every new thread would reload a 37 MB (or 166 MB)
model, so a burst of guests each pay a one-second model load before their search
even starts.

A pool fixes both ends: instances are created at most ``size`` times and reused
forever, and a thread that finds the pool empty waits for one instead of
allocating another. Borrowing is a context manager so an exception cannot leak
an instance out of the pool.
"""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class ModelPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], size: int, name: str = "model") -> None:
        if size < 1:
            raise ValueError("pool size must be at least 1")
        self._factory = factory
        self._size = size
        self._name = name
        self._free: queue.LifoQueue[T] = queue.LifoQueue()
        self._created = 0
        self._lock = threading.Lock()

    @property
    def created(self) -> int:
        return self._created

    @contextmanager
    def borrow(self, timeout: float | None = None) -> Iterator[T]:
        item = self._acquire(timeout)
        try:
            yield item
        finally:
            self._free.put(item)

    def _acquire(self, timeout: float | None) -> T:
        try:
            return self._free.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self._size:
                self._created += 1
                make = True
            else:
                make = False
        if make:
            try:
                return self._factory()
            except Exception:
                with self._lock:  # a failed build must not consume a slot forever
                    self._created -= 1
                raise
        # At capacity: wait for whoever has one to give it back.
        return self._free.get(timeout=timeout)

    def prewarm(self, n: int | None = None, warm: Callable[[T], None] | None = None) -> None:
        """Build (and optionally exercise) instances up front, at startup."""
        target = min(self._size, n if n is not None else self._size)
        built = []
        for _ in range(max(0, target - self._created)):
            item = self._acquire(None)
            if warm:
                warm(item)
            built.append(item)
        for item in built:
            self._free.put(item)
