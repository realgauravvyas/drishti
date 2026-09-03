"""Application factory and process lifecycle."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, repo
from .api import router
from .config import get_settings
from .db import init_db
from .engines import get_engine
from .pipeline import get_indexer, shutdown_indexer

log = logging.getLogger("smriti")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_PURGE_INTERVAL_SECONDS = 3600


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings.ensure_dirs()
    init_db()

    # Load the model before serving. Failing here, loudly, at boot beats
    # failing on a guest's first search.
    engine = get_engine()
    await asyncio.to_thread(engine.warmup)
    log.info("engine %s ready (%d-d, match>=%.2f)", engine.name, engine.dim, engine.threshold)

    indexer = get_indexer()
    indexer.start()
    resumed = await asyncio.to_thread(indexer.resume_pending)
    if resumed:
        log.info("resumed %d photos left pending by a previous run", resumed)

    purge_task = asyncio.create_task(_purge_loop())
    log.info("smriti %s listening -- data in %s", __version__, settings.data_dir)
    try:
        yield
    finally:
        purge_task.cancel()
        shutdown_indexer()


async def _purge_loop() -> None:
    """Enforce retention. An event that has expired is deleted, not hidden."""
    while True:
        try:
            removed = await asyncio.to_thread(repo.purge_expired)
            if removed:
                log.info("purged %d expired event(s): %s", len(removed), ", ".join(removed))
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:  # pragma: no cover - never let the loop die
            log.exception("retention purge failed")
        await asyncio.sleep(_PURGE_INTERVAL_SECONDS)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Smriti",
        summary="Find every photo you are in.",
        version=__version__,
        lifespan=lifespan,
    )

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_credentials=False,
            allow_methods=["*"], allow_headers=["*"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/index.html")

    @app.exception_handler(413)
    async def too_large(request: Request, exc):  # pragma: no cover - server dependent
        return JSONResponse({"detail": "upload too large"}, status_code=413)

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    else:  # pragma: no cover
        log.warning("web/ not found at %s -- API only", WEB_DIR)

    return app


app = create_app()
