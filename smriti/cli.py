#!/usr/bin/env python
"""Smriti command line.

The browser is fine for a hundred photos and the wrong tool for five thousand.
This talks to the database and the engine directly -- no server needed -- so an
organiser can point it at a folder on the machine that holds the originals::

    python cli.py create "Goa trip 2026" --retention 30
    python cli.py add    <event_id> "D:/DCIM/goa" --recursive
    python cli.py index  <event_id>
    python cli.py search <event_id> me.jpg
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from smriti import cluster, matcher, repo, weights
from smriti.config import get_settings
from smriti.db import init_db
from smriti.engines import ENGINE_NAMES, get_engine
from smriti.imaging import SUPPORTED_SUFFIXES, load_bgr
from smriti.pipeline import index_event_sync, ingest_bytes


# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------
def bar(done: int, total: int, width: int = 32) -> str:
    filled = 0 if not total else round(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + f"] {done}/{total}"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def resolve_event(handle: str) -> dict:
    """Accept either an event id or a share code."""
    event = repo.get_event(handle) or repo.get_event_by_code(handle)
    if event is None:
        sys.exit(f"no event with id or code {handle!r}")
    return event


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_create(args) -> None:
    engine = get_engine()
    created = repo.create_event(
        name=args.name, engine=engine.name, embed_dim=engine.dim,
        retention_days=args.retention, notes=args.notes,
        allow_download=not args.no_download,
    )
    event = created.event
    print(f"\n  Album:          {event['name']}")
    print(f"  Event id:       {event['id']}")
    print(f"  Share code:     {event['share_code']}")
    print(f"  Guest link:     /find.html?code={event['share_code']}")
    print(f"  Engine:         {engine.name} ({engine.dim}-d)")
    print(f"\n  Organiser token (shown once, store it now):\n\n    {created.admin_token}\n")


def cmd_add(args) -> None:
    event = resolve_event(args.event)
    root = Path(args.folder)
    if not root.exists():
        sys.exit(f"no such folder: {root}")

    pattern = "**/*" if args.recursive else "*"
    files = sorted(p for p in root.glob(pattern)
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        sys.exit(f"no images found in {root}")

    print(f"adding {len(files)} files to {event['name']!r}")
    counts = {"queued": 0, "duplicate": 0, "rejected": 0}
    started = time.time()
    for i, path in enumerate(files, 1):
        result = ingest_bytes(event, path.name, path.read_bytes())
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status == "rejected":
            print(f"\n  ! {path.name}: {result.detail}")
        print(f"\r  {bar(i, len(files))}", end="", flush=True)
    elapsed = time.time() - started
    print(f"\n  added {counts['queued']}, duplicates {counts['duplicate']}, "
          f"rejected {counts['rejected']} in {elapsed:.1f}s")
    if not args.no_index:
        cmd_index(argparse.Namespace(event=event["id"]))


def cmd_index(args) -> None:
    event = resolve_event(args.event)
    pending = repo.pending_photo_ids(event["id"])
    if not pending:
        print("nothing pending -- everything is already indexed")
        return
    engine = get_engine(event["engine"])
    print(f"indexing {len(pending)} photos with {engine.name}")
    started = time.time()

    def progress(done: int, total: int) -> None:
        rate = done / max(1e-6, time.time() - started)
        eta = (total - done) / max(rate, 1e-6)
        print(f"\r  {bar(done, total)}  {rate:.1f}/s  eta {eta:5.0f}s", end="", flush=True)

    result = index_event_sync(event, progress=progress)
    matcher.invalidate(event["id"])
    stats = repo.event_stats(event["id"])
    print(f"\n  {result['photos']} photos -> {result['faces']} faces in {result['seconds']}s "
          f"({result['photos'] / max(result['seconds'], 1e-6):.1f} photos/s)")
    if stats["photos_failed"]:
        print(f"  {stats['photos_failed']} failed -- see the admin page for the reasons")


def cmd_search(args) -> None:
    event = resolve_event(args.event)
    engine = get_engine(event["engine"])
    images = [load_bgr(p) for p in args.selfies]
    started = time.perf_counter()
    queries, report = matcher.build_queries(engine, images)
    if queries.shape[0] == 0:
        sys.exit("no face found in the selfie(s)")
    index = matcher.get_index(event["id"], engine.dim)
    hits = index.search(queries, engine, threshold=args.threshold, limit=args.limit)
    elapsed = (time.perf_counter() - started) * 1000

    print(f"\n  {len(hits)} matches from {index.n_faces} faces in {elapsed:.0f} ms "
          f"(threshold {args.threshold or engine.threshold})\n")
    photos = {p["id"]: p for p in repo.list_photos(event["id"])}
    for hit in hits:
        photo = photos.get(hit.photo_id)
        name = photo["orig_name"] if photo else hit.photo_id
        print(f"  {hit.score:.3f}  {hit.tier:6s}  {name}")

    if args.copy_to:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for hit in hits:
            photo = photos.get(hit.photo_id)
            if photo and (tier_ok := hit.tier != "maybe" or args.include_maybe):
                source = Path(get_settings().data_dir) / photo["rel_path"]
                if source.exists():
                    shutil.copy2(source, dest / photo["orig_name"])
                    copied += 1
        print(f"\n  copied {copied} photos to {dest}")


def cmd_events(args) -> None:
    events = repo.list_events()
    if not events:
        print("no events yet -- create one with: python cli.py create \"My album\"")
        return
    print(f"\n  {'ID':18s} {'CODE':10s} {'PHOTOS':>7s} {'FACES':>7s}  NAME")
    for event in events:
        print(f"  {event['id']:18s} {event['share_code']:10s} "
              f"{event['n_photos']:7d} {event['n_faces']:7d}  {event['name']}")
    print()


def cmd_info(args) -> None:
    event = resolve_event(args.event)
    stats = repo.event_stats(event["id"])
    print(f"\n  {event['name']}")
    print(f"    id            {event['id']}")
    print(f"    share code    {event['share_code']}")
    print(f"    engine        {event['engine']} ({event['embed_dim']}-d)")
    print(f"    photos        {stats['photos_total']} "
          f"({stats['photos_indexed']} indexed, {stats['photos_pending']} pending, "
          f"{stats['photos_failed']} failed)")
    print(f"    faces         {stats['faces']}")
    print(f"    searches      {stats['searches']}")
    print(f"    on disk       {human_bytes(stats['bytes'])}")
    print(f"    expires       {time.ctime(event['expires_at']) if event['expires_at'] else 'never'}\n")


def cmd_people(args) -> None:
    event = resolve_event(args.event)
    engine = get_engine(event["engine"])
    started = time.time()
    result = cluster.cluster_event(event, engine, link_threshold=args.threshold,
                                   method=args.method)
    if result["skipped"]:
        sys.exit(f"skipped: {result['faces']} faces exceeds the clustering limit")
    print(f"\n  ~{result['n_people']} distinct people across {result['faces']} faces "
          f"({time.time() - started:.1f}s, link >= {result['threshold']})\n")
    for person in result["people"][:args.top]:
        print(f"    person {person['id']:3d}  {person['size']:4d} faces  "
              f"{len(person['photos']):4d} photos")
    print()


def cmd_delete(args) -> None:
    event = resolve_event(args.event)
    if not args.yes:
        confirm = input(f"permanently delete {event['name']!r} and all its photos? [y/N] ")
        if confirm.strip().lower() != "y":
            sys.exit("cancelled")
    matcher.invalidate(event["id"])
    repo.delete_event(event["id"])
    print(f"deleted {event['id']}")


def cmd_models(args) -> None:
    if args.download:
        for key in args.download:
            spec = weights.REGISTRY[key]
            print(f"downloading {key} ({spec.mb:.1f} MB) -> {spec.filename}")

            def progress(done: int, total: int, key=key) -> None:
                print(f"\r  {bar(done, total)} {100 * done / max(total, 1):5.1f}%", end="", flush=True)

            path = weights.ensure(key, progress=progress)
            print(f"\n  ok: {path}")
        return
    print(f"\n  weights dir: {get_settings().weights_dir}\n")
    for row in weights.status():
        mark = "yes" if row["present"] else "no "
        print(f"  [{mark}] {row['key']:12s} {row['mb']:7.1f} MB  {row['note']}")
    print(f"\n  engines: {', '.join(ENGINE_NAMES)} (current: {get_settings().engine})\n")


def cmd_purge(args) -> None:
    removed = repo.purge_expired()
    print(f"purged {len(removed)} expired event(s)" + (f": {', '.join(removed)}" if removed else ""))


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smriti", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="create an album")
    p.add_argument("name")
    p.add_argument("--retention", type=int, default=30, help="auto-delete after N days (0 = never)")
    p.add_argument("--notes", default="")
    p.add_argument("--no-download", action="store_true", help="do not let guests download ZIPs")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("add", help="add a folder of photos")
    p.add_argument("event"); p.add_argument("folder")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--no-index", action="store_true", help="store only; index later")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("index", help="detect and embed faces for pending photos")
    p.add_argument("event")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search an album with one or more selfies")
    p.add_argument("event"); p.add_argument("selfies", nargs="+")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--copy-to", help="copy matching photos into this folder")
    p.add_argument("--include-maybe", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("events", help="list albums"); p.set_defaults(func=cmd_events)

    p = sub.add_parser("info", help="show one album"); p.add_argument("event")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("people", help="estimate how many distinct people appear")
    p.add_argument("event")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--method", default="chinese-whispers",
                   choices=("chinese-whispers", "components"))
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_people)

    p = sub.add_parser("delete", help="delete an album and its photos")
    p.add_argument("event"); p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("models", help="show or download model weights")
    p.add_argument("--download", nargs="+", choices=sorted(weights.REGISTRY))
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("purge", help="delete events past their retention date")
    p.set_defaults(func=cmd_purge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    get_settings().ensure_dirs()
    init_db()
    args.func(args)


if __name__ == "__main__":
    main()
