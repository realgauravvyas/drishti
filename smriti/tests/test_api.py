"""HTTP-level tests: the contract a browser actually depends on."""

from __future__ import annotations

import io
import zipfile

import pytest

from smriti.pipeline import get_indexer

from .conftest import ALICE, BOB, DEV, make_photo, make_selfie


def search(client, code, hue, **extra):
    return client.post(
        f"/api/events/by-code/{code}/search",
        files=[("selfies", ("me.jpg", make_selfie(hue), "image/jpeg"))],
        data=extra,
    )


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------
def test_health_reports_the_live_engine(client):
    body = client.get("/api/health").json()
    assert body["ok"] and body["engine"] == "mock"
    assert "sface" in body["engines_available"]


def test_static_pages_are_served(client):
    for path in ("/index.html", "/find.html", "/admin.html", "/static/styles.css"):
        assert client.get(path).status_code == 200, path
    assert client.get("/", follow_redirects=False).status_code in (307, 302)


# --------------------------------------------------------------------------
# creation and auth
# --------------------------------------------------------------------------
def test_create_event_returns_a_one_time_token_and_share_links(client):
    body = client.post("/api/events", data={"name": "Trip", "retention_days": 7}).json()
    assert len(body["share_code"]) == 8
    assert body["share_url"].endswith(f"code={body['share_code']}")
    assert body["admin_token"] and len(body["admin_token"]) > 30


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer wrong-token"},
    {"Authorization": "not-even-bearer"},
])
def test_admin_endpoints_reject_bad_credentials(client, organiser, headers):
    event_id = organiser["event_id"]
    assert client.get(f"/api/events/{event_id}", headers=headers).status_code == 401
    assert client.delete(f"/api/events/{event_id}", headers=headers).status_code == 401
    assert client.get(f"/api/events/{event_id}/photos", headers=headers).status_code == 401


def test_one_organiser_cannot_touch_another_event(client, organiser):
    other = client.post("/api/events", data={"name": "Someone else"}).json()
    stolen = {"Authorization": f"Bearer {other['admin_token']}"}
    assert client.get(f"/api/events/{organiser['event_id']}", headers=stolen).status_code == 401


def test_unknown_share_code_is_404(client):
    assert client.get("/api/events/by-code/ZZZZZZZZ").status_code == 404


# --------------------------------------------------------------------------
# upload and indexing
# --------------------------------------------------------------------------
def test_upload_indexes_in_the_background_and_reports_progress(client, organiser):
    progress = client.get(f"/api/events/{organiser['event_id']}/progress",
                          headers=organiser["headers"]).json()
    assert progress["done"] is True
    assert progress["photos_total"] == 7
    assert progress["photos_indexed"] == 7
    assert progress["photos_failed"] == 0
    assert progress["faces"] == 10         # 2+1+2+1+3+1+0 across the fixture album
    assert progress["percent"] == 100.0


def test_upload_separates_duplicates_and_rejects(client, organiser):
    files = [
        ("files", ("dup.jpg", make_photo([ALICE, BOB]), "image/jpeg")),  # already uploaded
        ("files", ("new.jpg", make_photo([DEV, ALICE]), "image/jpeg")),
        ("files", ("bad.txt", b"not an image", "image/jpeg")),
    ]
    body = client.post(f"/api/events/{organiser['event_id']}/photos",
                       files=files, headers=organiser["headers"]).json()
    assert (body["queued"], body["duplicates"], body["rejected"]) == (1, 1, 1)


def test_public_event_view_hides_organiser_secrets(client, organiser):
    body = client.get(f"/api/events/by-code/{organiser['code']}").json()
    assert body["photos_indexed"] == 7 and body["ready"] is True
    assert "admin_token_hash" not in body and "id" not in body


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def test_search_returns_only_the_guest_own_photos(client, organiser):
    body = search(client, organiser["code"], ALICE).json()
    assert body["count"] == 3
    assert {m["name"] for m in body["matches"]} == {"01.jpg", "02.jpg", "05.jpg"}
    assert body["counts_by_tier"]["sure"] == 3
    assert body["searched_photos"] == 7


def test_search_result_carries_everything_the_ui_needs(client, organiser):
    match = search(client, organiser["code"], ALICE).json()["matches"][0]
    assert match["thumb_url"].startswith("/api/photos/")
    assert "t=" in match["thumb_url"] and match["token"]
    assert len(match["box"]) == 4 and match["box"][2] > 0
    assert match["tier"] in {"sure", "likely", "maybe"}
    assert 0.0 <= match["score"] <= 1.0


def test_search_with_a_faceless_selfie_explains_itself(client, organiser):
    body = client.post(
        f"/api/events/by-code/{organiser['code']}/search",
        files=[("selfies", ("blank.jpg", make_photo([]), "image/jpeg"))],
    ).json()
    assert body["no_face_in_selfie"] is True and body["matches"] == []
    assert "clear" in body["message"].lower()


def test_search_rejects_junk_and_too_many_selfies(client, organiser):
    code = organiser["code"]
    assert client.post(f"/api/events/by-code/{code}/search",
                       files=[("selfies", ("x.jpg", b"junk", "image/jpeg"))]).status_code == 400
    too_many = [("selfies", (f"{i}.jpg", make_selfie(ALICE), "image/jpeg")) for i in range(4)]
    assert client.post(f"/api/events/by-code/{code}/search", files=too_many).status_code == 400


def test_search_is_logged_without_storing_the_face(client, organiser):
    search(client, organiser["code"], ALICE)
    rows = client.get(f"/api/events/{organiser['event_id']}/searches",
                      headers=organiser["headers"]).json()["searches"]
    assert len(rows) == 1
    assert rows[0]["n_matches"] == 3 and rows[0]["n_queries"] == 1
    assert set(rows[0]) == {"ts", "n_queries", "n_matches", "top_score", "ms"}


def test_search_is_rate_limited(client, organiser):
    from smriti.api import _search_limit

    _search_limit.limit = 3
    try:
        codes = [search(client, organiser["code"], ALICE).status_code for _ in range(5)]
    finally:
        _search_limit.limit = 30
    assert 429 in codes


# --------------------------------------------------------------------------
# photo access control
# --------------------------------------------------------------------------
def test_photos_need_a_signed_token_or_the_organiser_token(client, organiser):
    match = search(client, organiser["code"], ALICE).json()["matches"][0]
    photo_id = match["photo_id"]

    assert client.get(match["thumb_url"]).status_code == 200
    assert client.get(match["original_url"]).status_code == 200
    assert client.get(f"/api/photos/{photo_id}/thumb").status_code == 403
    assert client.get(f"/api/photos/{photo_id}/thumb?t=1.deadbeef").status_code == 403
    assert client.get(f"/api/photos/{photo_id}/thumb",
                      headers=organiser["headers"]).status_code == 200


def test_a_guest_token_does_not_unlock_other_peoples_photos(client, organiser):
    alice = search(client, organiser["code"], ALICE).json()["matches"][0]
    dev = search(client, organiser["code"], DEV).json()["matches"][0]
    assert alice["photo_id"] != dev["photo_id"]
    # Alice's signed token must not open Dev's photo.
    assert client.get(f"/api/photos/{dev['photo_id']}/thumb"
                      f"?t={alice['token']}").status_code == 403


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def test_download_bundles_the_selected_photos(client, organiser):
    matches = search(client, organiser["code"], ALICE).json()["matches"]
    response = client.post(
        f"/api/events/by-code/{organiser['code']}/download",
        data={"photo_ids": ",".join(m["photo_id"] for m in matches),
              "tokens": ",".join(m["token"] for m in matches)},
    )
    assert response.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert len(names) == 3 and all(n.endswith(".jpg") for n in names)
    assert names == sorted(names)  # numbered, so the order is stable for the user


def test_download_refuses_forged_or_mismatched_tokens(client, organiser):
    matches = search(client, organiser["code"], ALICE).json()["matches"]
    ids = ",".join(m["photo_id"] for m in matches)
    assert client.post(f"/api/events/by-code/{organiser['code']}/download",
                       data={"photo_ids": ids,
                             "tokens": ",".join(["9999999999.forged"] * 3)}).status_code == 404
    assert client.post(f"/api/events/by-code/{organiser['code']}/download",
                       data={"photo_ids": ids, "tokens": "only-one"}).status_code == 400
    # An empty selection is refused, whether the field arrives blank or not at all.
    assert client.post(f"/api/events/by-code/{organiser['code']}/download",
                       data={"photo_ids": " ", "tokens": " "}).status_code == 400
    assert client.post(f"/api/events/by-code/{organiser['code']}/download",
                       data={}).status_code == 422


def test_download_can_be_disabled_by_the_organiser(client):
    created = client.post("/api/events",
                          data={"name": "No downloads", "allow_download": "false"}).json()
    headers = {"Authorization": f"Bearer {created['admin_token']}"}
    client.post(f"/api/events/{created['event_id']}/photos",
                files=[("files", ("a.jpg", make_photo([ALICE]), "image/jpeg"))], headers=headers)
    get_indexer().wait_idle(timeout=30)

    body = search(client, created["share_code"], ALICE).json()
    assert body["allow_download"] is False
    assert client.post(f"/api/events/by-code/{created['share_code']}/download",
                       data={"photo_ids": body["matches"][0]["photo_id"],
                             "tokens": body["matches"][0]["token"]}).status_code == 403


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------
def test_deleting_a_photo_drops_it_from_later_searches(client, organiser):
    before = search(client, organiser["code"], ALICE).json()["matches"]
    target = next(m for m in before if m["name"] == "02.jpg")

    assert client.delete(f"/api/events/{organiser['event_id']}/photos/{target['photo_id']}",
                         headers=organiser["headers"]).status_code == 200
    after = search(client, organiser["code"], ALICE).json()
    assert "02.jpg" not in {m["name"] for m in after["matches"]}
    assert after["count"] == 2


def test_reindex_rebuilds_every_face(client, organiser):
    event_id, headers = organiser["event_id"], organiser["headers"]
    assert client.post(f"/api/events/{event_id}/reindex", headers=headers).json()["requeued"] == 7
    assert get_indexer().wait_idle(timeout=60)
    assert client.get(f"/api/events/{event_id}/progress", headers=headers).json()["faces"] == 10


def test_people_endpoint_estimates_distinct_guests(client, organiser):
    body = client.get(f"/api/events/{organiser['event_id']}/people",
                      headers=organiser["headers"]).json()
    assert body["n_people"] == 4 and body["faces"] == 10
    assert body["top"] and body["top"][0]["thumb_url"].startswith("/api/photos/")


def test_deleting_an_event_makes_the_code_and_photos_unreachable(client, organiser):
    match = search(client, organiser["code"], ALICE).json()["matches"][0]
    assert client.delete(f"/api/events/{organiser['event_id']}",
                         headers=organiser["headers"]).status_code == 200

    assert client.get(f"/api/events/by-code/{organiser['code']}").status_code == 404
    assert client.get(match["thumb_url"]).status_code == 404
    assert search(client, organiser["code"], ALICE).status_code == 404


def test_expired_event_reports_gone_rather_than_not_found(client, organiser):
    import time as _time

    from smriti.db import transaction
    with transaction() as conn:
        conn.execute("UPDATE events SET expires_at = ? WHERE id = ?",
                     (_time.time() - 1, organiser["event_id"]))
    assert client.get(f"/api/events/by-code/{organiser['code']}").status_code == 410
