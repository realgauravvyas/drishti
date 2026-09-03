"""HTTP API.

Two audiences, two auth models:

* **Organisers** hold a bearer token issued once at event creation. They can
  upload, delete, re-index and read the audit log.
* **Guests** hold only the share code. They can look up the event's public
  facts and run a search. Everything a search returns is accompanied by a
  short-lived signed URL, so a guest can fetch exactly the photos they matched
  and nothing else -- the share code is not a key to the whole album.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from fastapi import (APIRouter, Depends, File, Form, Header, HTTPException,
                     Query, Request, UploadFile)
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from . import cluster, matcher, repo, security, storage
from .config import get_settings
from .engines import ENGINE_NAMES, get_engine
from .imaging import ImageError, open_rgb, to_bgr
from .pipeline import get_indexer, ingest_bytes

log = logging.getLogger("smriti.api")
router = APIRouter(prefix="/api")

# Search is the expensive public endpoint and the one worth grinding, so it gets
# the tighter limit. Uploads are authenticated, so they only need flood control.
_search_limit = security.RateLimiter(limit=30, window_seconds=60)
_create_limit = security.RateLimiter(limit=10, window_seconds=3600)

MAX_SELFIES = 3


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown")
    return security.hash_ip(ip)


def require_event(event_id: str) -> dict:
    event = repo.get_event(event_id)
    if event is None:
        raise HTTPException(404, "event not found")
    return event


def require_admin(event_id: str,
                  authorization: str = Header(default="")) -> dict:
    event = require_event(event_id)
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not security.token_matches(token, event["admin_token_hash"]):
        raise HTTPException(401, "organiser token required")
    return event


def require_shared(code: str) -> dict:
    event = repo.get_event_by_code(code)
    if event is None:
        raise HTTPException(404, "no event with that code")
    if event["expires_at"] and event["expires_at"] < time.time():
        raise HTTPException(410, "this event has expired and its photos were deleted")
    return event


def public_event(event: dict) -> dict:
    stats = repo.event_stats(event["id"])
    return {
        "name": event["name"],
        "share_code": event["share_code"],
        "notes": event["notes"],
        "allow_download": bool(event["allow_download"]),
        "created_at": event["created_at"],
        "expires_at": event["expires_at"],
        "photos_total": stats["photos_total"],
        "photos_indexed": stats["photos_indexed"],
        "photos_pending": stats["photos_pending"],
        "ready": stats["photos_pending"] == 0 and stats["photos_total"] > 0,
    }


def photo_payload(event: dict, photo: dict, *, signed: bool = True) -> dict:
    token = security.sign_photo(photo["id"], event["id"]) if signed else None
    suffix = f"?t={token}" if token else ""
    return {
        "photo_id": photo["id"],
        "name": photo["orig_name"],
        "width": photo["width"],
        "height": photo["height"],
        "bytes": photo["bytes"],
        "taken_at": photo["taken_at"],
        "thumb_url": f"/api/photos/{photo['id']}/thumb{suffix}",
        "original_url": f"/api/photos/{photo['id']}/original{suffix}",
        "token": token,
    }


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    from . import __version__

    settings = get_settings()
    indexer = get_indexer()
    return {
        "ok": True,
        "version": __version__,
        "engine": settings.engine,
        "workers": settings.workers,
        "queue_depth": indexer.depth,
        "engines_available": list(ENGINE_NAMES),
    }


@router.get("/engine")
def engine_info() -> dict:
    engine = get_engine()
    return engine.describe() | {"detect_size": get_settings().detect_size}


# --------------------------------------------------------------------------
# events -- organiser side
# --------------------------------------------------------------------------
@router.post("/events")
def create_event(request: Request,
                 name: str = Form(...),
                 notes: str = Form(default=""),
                 allow_download: bool = Form(default=True),
                 retention_days: int = Form(default=-1)) -> dict:
    allowed, retry = _create_limit.check(client_key(request))
    if not allowed:
        raise HTTPException(429, f"too many events created; retry in {int(retry)}s")

    settings = get_settings()
    engine = get_engine()
    days = settings.retention_days if retention_days < 0 else retention_days
    created = repo.create_event(
        name=name, engine=engine.name, embed_dim=engine.dim,
        retention_days=days, notes=notes.strip(),
        allow_download=allow_download,
    )
    event = created.event
    base = settings.public_url.rstrip("/") or str(request.base_url).rstrip("/")
    return {
        "event_id": event["id"],
        # Shown exactly once. We store only its hash, so it cannot be recovered.
        "admin_token": created.admin_token,
        "share_code": event["share_code"],
        "share_url": f"{base}/find.html?code={event['share_code']}",
        "admin_url": f"{base}/admin.html?event={event['id']}&token={created.admin_token}",
        "engine": event["engine"],
        "expires_at": event["expires_at"],
    }


@router.get("/events/{event_id}")
def event_detail(event: dict = Depends(require_admin)) -> dict:
    return {
        "event_id": event["id"],
        "name": event["name"],
        "notes": event["notes"],
        "share_code": event["share_code"],
        "engine": event["engine"],
        "embed_dim": event["embed_dim"],
        "created_at": event["created_at"],
        "expires_at": event["expires_at"],
        "allow_download": bool(event["allow_download"]),
        "stats": repo.event_stats(event["id"]),
        "queue_depth": get_indexer().depth,
    }


@router.delete("/events/{event_id}")
def delete_event(event: dict = Depends(require_admin)) -> dict:
    matcher.invalidate(event["id"])
    repo.delete_event(event["id"])
    return {"deleted": True, "event_id": event["id"]}


@router.post("/events/{event_id}/photos")
async def upload_photos(files: list[UploadFile] = File(...),
                        event: dict = Depends(require_admin)) -> dict:
    settings = get_settings()
    indexer = get_indexer()
    results = []
    for upload in files:
        data = await upload.read()
        name = os.path.basename(upload.filename or "photo.jpg")
        # Decode + thumbnail is CPU work; keep the event loop free.
        result = await asyncio.to_thread(ingest_bytes, event, name, data)
        if result.status == "queued" and result.photo_id:
            indexer.submit(event["id"], result.photo_id)
        results.append({"file": result.filename, "status": result.status,
                        "photo_id": result.photo_id, "detail": result.detail})
    queued = sum(1 for r in results if r["status"] == "queued")
    return {
        "received": len(results),
        "queued": queued,
        "duplicates": sum(1 for r in results if r["status"] == "duplicate"),
        "rejected": sum(1 for r in results if r["status"] == "rejected"),
        "results": results,
        "max_upload_mb": settings.max_upload_mb,
    }


@router.get("/events/{event_id}/photos")
def list_photos(limit: int = Query(500, le=5000), offset: int = 0,
                event: dict = Depends(require_admin)) -> dict:
    photos = repo.list_photos(event["id"], limit=limit, offset=offset)
    faces = repo.faces_for_photos([p["id"] for p in photos])
    return {
        "photos": [
            photo_payload(event, p) | {
                "state": p["state"],
                "n_faces": p["n_faces"],
                "error": p["error"],
                "faces": [
                    {"id": f["id"], "box": [f["x"], f["y"], f["w"], f["h"]],
                     "det_score": round(f["det_score"], 3), "px": round(f["face_px"], 1)}
                    for f in faces.get(p["id"], [])
                ],
            }
            for p in photos
        ],
        "offset": offset,
        "limit": limit,
    }


@router.delete("/events/{event_id}/photos/{photo_id}")
def delete_photo(photo_id: str, event: dict = Depends(require_admin)) -> dict:
    deleted = repo.delete_photo(event["id"], photo_id)
    if not deleted:
        raise HTTPException(404, "photo not found in this event")
    matcher.invalidate(event["id"])
    return {"deleted": True, "photo_id": photo_id}


@router.get("/events/{event_id}/progress")
def progress(event: dict = Depends(require_admin)) -> dict:
    stats = repo.event_stats(event["id"])
    total = max(1, stats["photos_total"])
    done = stats["photos_indexed"] + stats["photos_failed"]
    return {
        **stats,
        "percent": round(100.0 * done / total, 1),
        "done": stats["photos_pending"] == 0,
        "queue_depth": get_indexer().depth,
    }


@router.post("/events/{event_id}/reindex")
def reindex(event: dict = Depends(require_admin)) -> dict:
    repo.reset_index(event["id"])
    matcher.invalidate(event["id"])
    indexer = get_indexer()
    ids = repo.pending_photo_ids(event["id"])
    for photo_id in ids:
        indexer.submit(event["id"], photo_id)
    return {"requeued": len(ids)}


@router.get("/events/{event_id}/people")
def people(event: dict = Depends(require_admin)) -> dict:
    engine = get_engine(event["engine"])
    result = cluster.cluster_event(event, engine)
    summary = cluster.summarise(result)
    for person in summary["top"]:
        photo = repo.get_photo(person["cover_photo_id"])
        if photo:
            person["thumb_url"] = photo_payload(event, photo)["thumb_url"]
    return summary


@router.get("/events/{event_id}/searches")
def searches(event: dict = Depends(require_admin)) -> dict:
    return {"searches": repo.recent_searches(event["id"])}


# --------------------------------------------------------------------------
# guest side
# --------------------------------------------------------------------------
@router.get("/events/by-code/{code}")
def event_by_code(code: str) -> dict:
    return public_event(require_shared(code))


@router.post("/events/by-code/{code}/search")
async def search(request: Request, code: str,
                 selfies: list[UploadFile] = File(...),
                 threshold: float | None = Form(default=None),
                 limit: int = Form(default=0)) -> dict:
    event = require_shared(code)
    allowed, retry = _search_limit.check(client_key(request))
    if not allowed:
        raise HTTPException(429, f"too many searches; retry in {int(retry)}s")

    settings = get_settings()
    if len(selfies) > MAX_SELFIES:
        raise HTTPException(400, f"at most {MAX_SELFIES} selfies per search")

    images = []
    for upload in selfies[:MAX_SELFIES]:
        data = await upload.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"selfie larger than {settings.max_upload_mb} MB")
        try:
            images.append(to_bgr(open_rgb(data)))
        except ImageError as exc:
            raise HTTPException(400, f"could not read that image: {exc}") from exc
        # `data` goes out of scope here and is never written to disk. See
        # settings.store_selfies for the (off-by-default) escape hatch.

    engine = get_engine(event["engine"])
    started = time.perf_counter()
    queries, report = await asyncio.to_thread(matcher.build_queries, engine, images)
    if queries.shape[0] == 0:
        return {
            "matches": [], "count": 0, "no_face_in_selfie": True,
            "selfies": report,
            "message": "No face found in that photo. Try a clear, front-facing, "
                       "well-lit picture of just you.",
        }

    index = await asyncio.to_thread(matcher.get_index, event["id"], engine.dim)
    hits = index.search(queries, engine, threshold=threshold,
                        limit=limit or settings.max_results)
    elapsed_ms = (time.perf_counter() - started) * 1000

    photos = {p["id"]: p for p in repo.list_photos(event["id"])}
    matches = []
    for hit in hits:
        photo = photos.get(hit.photo_id)
        if photo is None:
            continue
        matches.append(photo_payload(event, photo) | {
            "score": hit.score,
            "tier": hit.tier,
            "box": list(hit.face_box),
            "faces_in_photo": hit.n_faces_in_photo,
        })

    repo.log_search(event["id"], len(queries), len(matches),
                    matches[0]["score"] if matches else 0.0, elapsed_ms)

    return {
        "matches": matches,
        "count": len(matches),
        "counts_by_tier": {
            tier: sum(1 for m in matches if m["tier"] == tier)
            for tier in ("sure", "likely", "maybe")
        },
        "searched_faces": index.n_faces,
        "searched_photos": len(photos),
        "ms": round(elapsed_ms, 1),
        "selfies": report,
        "allow_download": bool(event["allow_download"]),
        "thresholds": engine.describe(),
        "no_face_in_selfie": False,
    }


@router.post("/events/by-code/{code}/download")
def download_zip(code: str, photo_ids: str = Form(...), tokens: str = Form(default="")):
    """Bundle a guest's selected photos into one ZIP.

    Every id must arrive with a valid signed token from that guest's own search
    result, so this cannot be used to pull the whole album.
    """
    event = require_shared(code)
    if not event["allow_download"]:
        raise HTTPException(403, "the organiser disabled downloads for this event")

    ids = [i for i in photo_ids.split(",") if i.strip()]
    token_list = tokens.split(",") if tokens else []
    if not ids:
        raise HTTPException(400, "no photos selected")
    if len(token_list) != len(ids):
        raise HTTPException(400, "each photo id needs its access token")

    # mkstemp hands back an *open* descriptor; close it before anything else
    # opens the path, or Windows refuses to unlink the file afterwards.
    handle, tmp_name = tempfile.mkstemp(prefix="smriti-zip-", suffix=".zip")
    os.close(handle)
    tmp = Path(tmp_name)
    written = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as archive:
        # ZIP_STORED, not DEFLATE: JPEGs are already compressed, so deflating
        # them burns CPU for ~1% and makes a 300-photo download noticeably slower.
        for photo_id, token in zip(ids, token_list):
            if not security.verify_photo(token.strip(), photo_id.strip(), event["id"]):
                continue
            photo = repo.get_photo(photo_id.strip())
            if photo is None or photo["event_id"] != event["id"]:
                continue
            try:
                path = storage.absolute(photo["rel_path"])
            except ValueError:
                continue
            if not path.exists():
                continue
            archive.write(path, arcname=_zip_name(photo, written))
            written += 1

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(404, "none of those photos are available to you")

    filename = _safe_filename(event["name"]) + "-photos.zip"
    return FileResponse(
        tmp, media_type="application/zip", filename=filename,
        background=BackgroundTask(lambda: tmp.unlink(missing_ok=True)),
    )


def _zip_name(photo: dict, ordinal: int) -> str:
    stem = Path(photo["orig_name"]).stem[:60] or "photo"
    suffix = Path(photo["rel_path"]).suffix or ".jpg"
    return f"{ordinal + 1:03d}_{_safe_filename(stem)}{suffix}"


def _safe_filename(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_ ") else "-" for c in text.strip()]
    return ("".join(keep).strip().replace(" ", "-") or "smriti")[:60]


# --------------------------------------------------------------------------
# photo bytes
# --------------------------------------------------------------------------
def _authorised_photo(photo_id: str, token: str | None,
                      authorization: str) -> tuple[dict, dict]:
    photo = repo.get_photo(photo_id)
    if photo is None:
        raise HTTPException(404, "photo not found")
    event = repo.get_event(photo["event_id"])
    if event is None:
        raise HTTPException(404, "photo not found")
    if token and security.verify_photo(token, photo_id, event["id"]):
        return photo, event
    bearer = authorization.removeprefix("Bearer ").strip()
    if bearer and security.token_matches(bearer, event["admin_token_hash"]):
        return photo, event
    raise HTTPException(403, "this photo needs a valid access token")


@router.get("/photos/{photo_id}/thumb")
def photo_thumb(photo_id: str, t: str | None = None,
                authorization: str = Header(default="")):
    photo, _ = _authorised_photo(photo_id, t, authorization)
    if not photo["thumb_path"]:
        raise HTTPException(404, "no thumbnail")
    path = storage.absolute(photo["thumb_path"])
    if not path.exists():
        raise HTTPException(404, "thumbnail missing on disk")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=3600"})


@router.get("/photos/{photo_id}/original")
def photo_original(photo_id: str, t: str | None = None,
                   download: bool = False,
                   authorization: str = Header(default="")):
    photo, _ = _authorised_photo(photo_id, t, authorization)
    path = storage.absolute(photo["rel_path"])
    if not path.exists():
        raise HTTPException(404, "photo missing on disk")
    return FileResponse(
        path,
        filename=photo["orig_name"] if download else None,
        headers={"Cache-Control": "private, max-age=3600"},
    )
