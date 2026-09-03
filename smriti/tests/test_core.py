"""Unit tests for the layers below HTTP: imaging, engines, storage, repo, matcher."""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from PIL import Image

from smriti import cluster, matcher, repo, security, storage
from smriti.engines.base import align_112, l2_normalise, order_landmarks
from smriti.imaging import (ImageError, blur_score, fit_within, open_rgb,
                            sha256_bytes, taken_at, to_bgr)
from smriti.pipeline import index_event_sync, ingest_bytes

from .conftest import ALICE, BOB, CARA, DEV, make_photo, make_selfie


# --------------------------------------------------------------------------
# imaging
# --------------------------------------------------------------------------
def test_open_rgb_rejects_non_images():
    with pytest.raises(ImageError):
        open_rgb(b"this is not a JPEG")


def test_open_rgb_applies_exif_rotation():
    # Orientation 6 means "rotate 90 deg CW to display". A portrait photo stored
    # landscape must come back portrait, or every face in it is sideways.
    landscape = Image.new("RGB", (200, 100), (120, 90, 60))
    buf = io.BytesIO()
    exif = landscape.getexif()
    exif[274] = 6  # Orientation
    landscape.save(buf, "JPEG", exif=exif)

    opened = open_rgb(buf.getvalue())
    assert (opened.width, opened.height) == (100, 200)


def test_fit_within_scale_maps_back_to_original():
    big = np.zeros((900, 1800, 3), np.uint8)
    small, scale = fit_within(big, 600)
    assert max(small.shape[:2]) == 600
    assert round(small.shape[1] * scale) == 1800

    untouched, scale = fit_within(np.zeros((80, 60, 3), np.uint8), 600)
    assert scale == 1.0 and untouched.shape == (80, 60, 3)


def test_blur_score_ranks_sharp_above_smooth():
    flat = np.full((60, 60, 3), 128, np.uint8)
    noisy = np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    assert blur_score(noisy) > blur_score(flat)
    assert blur_score(np.zeros((0, 0, 3), np.uint8)) == 0.0


def test_sha256_is_content_addressed():
    data = make_photo([ALICE])
    assert sha256_bytes(data) == sha256_bytes(bytes(data))
    assert sha256_bytes(data) != sha256_bytes(make_photo([BOB]))


def test_taken_at_absent_is_none():
    assert taken_at(make_photo([ALICE])) is None
    assert taken_at(b"junk") is None


# --------------------------------------------------------------------------
# engine helpers
# --------------------------------------------------------------------------
def test_l2_normalise_handles_zero_vector():
    assert np.allclose(np.linalg.norm(l2_normalise(np.array([3.0, 4.0]))), 1.0)
    assert not np.any(l2_normalise(np.zeros(8)))


def test_order_landmarks_is_detector_convention_agnostic():
    # Same face, two detector conventions for which eye comes first.
    a = np.array([[30, 50], [70, 50], [50, 70], [35, 90], [65, 90]], np.float32)
    b = np.array([[70, 50], [30, 50], [50, 70], [65, 90], [35, 90]], np.float32)
    assert np.array_equal(order_landmarks(a), order_landmarks(b))


def test_align_112_survives_degenerate_landmarks():
    collapsed = np.zeros((5, 2), np.float32)
    assert align_112(np.zeros((80, 80, 3), np.uint8), collapsed).shape == (112, 112, 3)


def test_mock_engine_separates_people(engine):
    same = engine.detect_and_embed(to_bgr(open_rgb(make_photo([ALICE]))))
    other = engine.detect_and_embed(to_bgr(open_rgb(make_photo([CARA]))))
    assert len(same) == 1 and len(other) == 1
    assert float(same[0].embedding @ same[0].embedding) > engine.threshold
    assert float(same[0].embedding @ other[0].embedding) < engine.threshold


def test_engine_finds_every_person_in_a_group_shot(engine):
    faces = engine.detect_and_embed(to_bgr(open_rgb(make_photo([ALICE, BOB, CARA]))))
    assert len(faces) == 3
    assert all(f.face_px > 0 for f in faces)


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def test_absolute_refuses_paths_outside_the_data_dir():
    with pytest.raises(ValueError):
        storage.absolute("../../../etc/passwd")


def test_relative_and_absolute_round_trip(event):
    img = open_rgb(make_photo([ALICE]))
    path = storage.write_original(event.event["id"], "abc123", ".jpg", make_photo([ALICE]))
    assert storage.absolute(storage.relative(path)) == path.resolve()


# --------------------------------------------------------------------------
# repo
# --------------------------------------------------------------------------
def test_event_creation_issues_a_token_that_is_never_stored(event):
    stored = repo.get_event(event.event["id"])
    assert event.admin_token not in str(dict(stored))
    assert security.token_matches(event.admin_token, stored["admin_token_hash"])
    assert not security.token_matches("wrong", stored["admin_token_hash"])


def test_share_codes_are_unambiguous(engine):
    codes = {repo.create_event(f"e{i}", engine.name, engine.dim).event["share_code"]
             for i in range(25)}
    assert len(codes) == 25
    assert not (set("".join(codes)) & set("01ILO"))


def test_duplicate_upload_is_reported_not_duplicated(event):
    data = make_photo([ALICE])
    first = ingest_bytes(event.event, "a.jpg", data)
    second = ingest_bytes(event.event, "a-copy.jpg", data)
    assert first.status == "queued"
    assert second.status == "duplicate" and second.photo_id == first.photo_id
    assert repo.event_stats(event.event["id"])["photos_total"] == 1


def test_rejects_oversized_and_undecodable_uploads(event, monkeypatch):
    assert ingest_bytes(event.event, "x.jpg", b"").status == "rejected"
    assert ingest_bytes(event.event, "x.jpg", b"not an image").status == "rejected"

    from smriti import config
    monkeypatch.setenv("SMRITI_MAX_UPLOAD_MB", "0")
    config.reset_settings()
    assert ingest_bytes(event.event, "big.jpg", make_photo([ALICE])).status == "rejected"


def test_delete_event_removes_rows_and_files(event, album):
    event_id = event.event["id"]
    for name, hues in album.items():
        ingest_bytes(event.event, name, make_photo(hues))
    index_event_sync(event.event)
    assert storage.event_dir(event_id).exists()

    repo.delete_event(event_id)
    assert repo.get_event(event_id) is None
    assert repo.event_stats(event_id)["faces"] == 0
    assert not storage.event_dir(event_id).exists()


def test_purge_expired_only_removes_lapsed_events(engine):
    live = repo.create_event("live", engine.name, engine.dim, retention_days=30)
    dead = repo.create_event("dead", engine.name, engine.dim, retention_days=30)
    from smriti.db import transaction
    with transaction() as conn:
        conn.execute("UPDATE events SET expires_at = ? WHERE id = ?",
                     (time.time() - 10, dead.event["id"]))

    removed = repo.purge_expired()
    assert removed == [dead.event["id"]]
    assert repo.get_event(live.event["id"]) is not None


def test_failed_photo_is_recorded_not_retried_forever(event):
    result = ingest_bytes(event.event, "ok.jpg", make_photo([ALICE]))
    # Corrupt the stored file behind the pipeline's back.
    photo = repo.get_photo(result.photo_id)
    storage.absolute(photo["rel_path"]).write_bytes(b"corrupted")

    index_event_sync(event.event)
    after = repo.get_photo(result.photo_id)
    assert after["state"] == "failed" and after["error"]
    assert repo.pending_photo_ids(event.event["id"]) == []


# --------------------------------------------------------------------------
# matcher
# --------------------------------------------------------------------------
def indexed_album(event, album):
    for name, hues in album.items():
        ingest_bytes(event.event, name, make_photo(hues))
    index_event_sync(event.event)
    return {p["orig_name"]: p["id"] for p in repo.list_photos(event.event["id"])}


def search_names(event, engine, hue, names, **kwargs):
    queries, _ = matcher.build_queries(engine, [to_bgr(open_rgb(make_selfie(hue)))])
    index = matcher.get_index(event.event["id"], engine.dim)
    by_id = {v: k for k, v in names.items()}
    return {by_id[m.photo_id]: m for m in index.search(queries, engine, **kwargs)}


def test_search_returns_exactly_the_photos_containing_the_person(event, engine, album):
    names = indexed_album(event, album)
    assert set(search_names(event, engine, ALICE, names)) == {"01.jpg", "02.jpg", "05.jpg"}
    assert set(search_names(event, engine, DEV, names)) == {"06.jpg"}


def test_search_of_an_absent_person_returns_nothing(event, engine, album):
    names = indexed_album(event, album)
    assert search_names(event, engine, 95, names) == {}


def test_one_result_per_photo_even_with_many_faces(event, engine, album):
    names = indexed_album(event, album)
    hits = search_names(event, engine, ALICE, names)
    assert hits["05.jpg"].n_faces_in_photo == 3
    assert len([h for h in hits if h == "05.jpg"]) == 1
    assert hits["05.jpg"].face_box[2] > 0  # a real box, for highlighting


def test_multiple_selfies_take_the_best_score_not_the_average(event, engine, album):
    indexed_album(event, album)
    good = to_bgr(open_rgb(make_selfie(ALICE)))
    unrelated = to_bgr(open_rgb(make_selfie(DEV)))
    index = matcher.get_index(event.event["id"], engine.dim)

    alone, _ = matcher.build_queries(engine, [good])
    together, _ = matcher.build_queries(engine, [good, unrelated])
    best_alone = index.search(alone, engine)[0].score
    best_together = max(m.score for m in index.search(together, engine))
    assert best_together >= best_alone  # the weak selfie cannot drag the good one down


def test_tiers_split_by_confidence(event, engine, album):
    names = indexed_album(event, album)
    hits = search_names(event, engine, ALICE, names)
    assert {m.tier for m in hits.values()} == {"sure"}

    # A threshold above every real score empties the result rather than lying.
    assert search_names(event, engine, ALICE, names, threshold=1.01) == {}


def test_index_cache_rebuilds_when_faces_change(event, engine, album):
    indexed_album(event, album)
    first = matcher.get_index(event.event["id"], engine.dim)
    assert matcher.get_index(event.event["id"], engine.dim) is first  # cached

    ingest_bytes(event.event, "08.jpg", make_photo([ALICE, DEV]))
    index_event_sync(event.event)
    second = matcher.get_index(event.event["id"], engine.dim)
    assert second is not first and second.n_faces > first.n_faces


def test_deleting_a_photo_removes_it_from_results(event, engine, album):
    names = indexed_album(event, album)
    assert "02.jpg" in search_names(event, engine, ALICE, names)

    repo.delete_photo(event.event["id"], names["02.jpg"])
    matcher.invalidate(event.event["id"])
    assert "02.jpg" not in search_names(event, engine, ALICE, names)


def test_empty_event_search_is_empty_not_an_error(event, engine):
    queries, _ = matcher.build_queries(engine, [to_bgr(open_rgb(make_selfie(ALICE)))])
    assert matcher.get_index(event.event["id"], engine.dim).search(queries, engine) == []


def test_selfie_without_a_face_produces_no_query(event, engine):
    blank = to_bgr(open_rgb(make_photo([])))
    queries, report = matcher.build_queries(engine, [blank])
    assert queries.shape[0] == 0 and report[0]["faces_found"] == 0


def test_build_queries_uses_the_largest_face_in_a_selfie(event, engine):
    # Two people in the "selfie": the bigger rectangle is the subject.
    import cv2
    canvas = np.zeros((300, 400, 3), np.uint8)
    for hue, (x0, y0, x1, y1) in ((ALICE, (10, 40, 160, 260)), (DEV, (250, 90, 310, 150))):
        colour = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, -1)

    queries, report = matcher.build_queries(engine, [canvas])
    assert queries.shape[0] == 1 and report[0]["faces_found"] == 2
    alice = engine.detect_and_embed(to_bgr(open_rgb(make_selfie(ALICE))))[0].embedding
    assert float(queries[0] @ alice) > engine.threshold


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------
def test_clustering_counts_distinct_people(event, engine, album):
    indexed_album(event, album)
    result = cluster.cluster_event(event.event, engine)
    assert result["n_people"] == 4          # Alice, Bob, Cara, Dev
    assert result["people"][0]["size"] == 3  # the three most-photographed tie at 3
    assert cluster.summarise(result)["n_people"] == 4


def test_chinese_whispers_resists_the_chaining_that_breaks_components():
    """The reason cluster.py does not use connected components.

    Three tight groups plus one 'bridge' vector sitting between two of them --
    the shape that a shared-looking face creates in a real album. Components
    must merge two groups through that single edge; Chinese Whispers outvotes
    it, because the bridge has one weak neighbour in each group and each group
    has many strong ones internally.
    """
    rng = np.random.default_rng(0)
    dim = 64
    centres = np.eye(3, dim, dtype=np.float32)
    vectors = []
    for centre in centres:
        for _ in range(12):
            noisy = centre + rng.normal(0, 0.05, dim).astype(np.float32)
            vectors.append(noisy / np.linalg.norm(noisy))
    bridge = (centres[0] + centres[1]) / np.linalg.norm(centres[0] + centres[1])
    vectors.append(bridge.astype(np.float32))
    matrix = np.ascontiguousarray(np.stack(vectors))

    graph = cluster._neighbours(matrix, threshold=0.55)
    merged = len(set(cluster._components(graph, len(matrix)).tolist()))
    voted = len(set(cluster._chinese_whispers(graph, len(matrix)).tolist()))

    assert merged <= 2, "the bridge should chain two groups together"
    assert voted >= 3, "weighted voting should keep the three groups apart"


def test_clustering_is_deterministic(event, engine, album):
    indexed_album(event, album)
    first = cluster.cluster_event(event.event, engine)
    second = cluster.cluster_event(event.event, engine)
    assert [p["size"] for p in first["people"]] == [p["size"] for p in second["people"]]


def test_clustering_skips_oversized_events(event, engine, album, monkeypatch):
    indexed_album(event, album)
    monkeypatch.setattr(cluster, "MAX_FACES", 1)
    assert cluster.cluster_event(event.event, engine)["skipped"] is True


# --------------------------------------------------------------------------
# security
# --------------------------------------------------------------------------
def test_photo_tokens_are_scoped_and_expiring():
    token = security.sign_photo("photo1", "event1")
    assert security.verify_photo(token, "photo1", "event1")
    assert not security.verify_photo(token, "photo2", "event1")   # other photo
    assert not security.verify_photo(token, "photo1", "event2")   # other event
    assert not security.verify_photo("garbage", "photo1", "event1")
    assert not security.verify_photo(security.sign_photo("photo1", "event1", ttl=-1),
                                     "photo1", "event1")          # expired


def test_rate_limiter_opens_again_after_its_window():
    limiter = security.RateLimiter(limit=2, window_seconds=0.25)
    assert limiter.check("a")[0] and limiter.check("a")[0]
    assert not limiter.check("a")[0]
    assert limiter.check("b")[0]  # per-key, not global
    time.sleep(0.3)
    assert limiter.check("a")[0]


def test_normalise_code_is_forgiving_about_how_people_type():
    assert security.normalise_code(" ab-cd 23 ") == "ABCD23"
