"""Shared fixtures.

Every test runs against the ``mock`` engine and a throwaway data directory, so
the suite needs no model weights, no network and no cleanup, and finishes in
about a second. The engines themselves are exercised separately in
``test_engines.py``, which skips when the real weights are absent.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smriti import config, matcher, repo  # noqa: E402
from smriti.db import close_all, init_db  # noqa: E402
from smriti.engines import clear_cache, get_engine  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point the whole package at a fresh temp directory for each test."""
    monkeypatch.setenv("SMRITI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SMRITI_ENGINE", "mock")
    monkeypatch.setenv("SMRITI_WORKERS", "2")
    close_all()
    config.reset_settings()
    clear_cache()
    matcher.invalidate()
    # The API's rate limiters are process-global; without a reset, test number
    # eleven gets a 429 from the counters test number one left behind.
    from smriti.api import _create_limit, _search_limit
    _create_limit.reset()
    _search_limit.reset()
    init_db()
    yield
    close_all()
    config.reset_settings()
    clear_cache()
    matcher.invalidate()


@pytest.fixture
def engine():
    return get_engine("mock")


# --------------------------------------------------------------------------
# synthetic album
# --------------------------------------------------------------------------
#: A "person" in the fixtures is a hue. The mock engine embeds a face patch by
#: its hue histogram, so two crops of the same hue match and different hues do
#: not -- enough to exercise every layer above the model.
ALICE, BOB, CARA, DEV = 10, 60, 120, 150


def make_photo(hues: list[int], width: int = 420, height: int = 300,
               quality: int = 92) -> bytes:
    """Render a JPEG containing one coloured rectangle per person."""
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:] = (240, 240, 240)
    for i, hue in enumerate(hues):
        colour = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
        x = 20 + i * 110
        cv2.rectangle(canvas, (x, 60), (x + 80, 190), colour, -1)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def make_selfie(hue: int) -> bytes:
    return make_photo([hue], width=200, height=220)


@pytest.fixture
def album():
    """Filename -> people in it. Alice is in three photos, Dev in one."""
    return {
        "01.jpg": [ALICE, BOB],
        "02.jpg": [ALICE],
        "03.jpg": [BOB, CARA],
        "04.jpg": [CARA],
        "05.jpg": [ALICE, BOB, CARA],
        "06.jpg": [DEV],
        "07.jpg": [],  # a landscape shot with nobody in it
    }


@pytest.fixture
def event(engine):
    return repo.create_event("Test trip", engine.name, engine.dim, retention_days=30)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from smriti.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def organiser(client, album):
    """A created event with the album uploaded and fully indexed."""
    from smriti.pipeline import get_indexer

    created = client.post("/api/events", data={"name": "Test trip", "retention_days": 7}).json()
    headers = {"Authorization": f"Bearer {created['admin_token']}"}
    files = [("files", (name, make_photo(hues), "image/jpeg")) for name, hues in album.items()]
    client.post(f"/api/events/{created['event_id']}/photos", files=files, headers=headers)
    assert get_indexer().wait_idle(timeout=60), "indexer did not drain"
    return {"created": created, "headers": headers,
            "event_id": created["event_id"], "code": created["share_code"]}
