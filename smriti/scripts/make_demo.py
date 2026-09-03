#!/usr/bin/env python
"""Build a demo album with known ground truth -- and score the whole product on it.

``scripts/benchmark.py`` measures the *model*: does face A match face B. This
measures the **product**: an organiser uploads an album, a guest uploads one
selfie, and we count how many of that guest's photos actually come back.

It needs a folder-per-person dataset (see benchmark.py). For each person it
holds out one image as their selfie and composites the rest into group photos,
so the answer key is exact::

    python scripts/make_demo.py --dataset ./dataset --photos 60 --evaluate

The composites are synthetic -- real faces, artificial arrangement -- so treat
the numbers as an upper-ish bound on a real album, where faces are also blurred,
turned away and half behind someone's shoulder. Point ``--dataset`` at your own
photos for a number that means something for your use case.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smriti import matcher, repo  # noqa: E402
from smriti.config import get_settings  # noqa: E402
from smriti.db import init_db  # noqa: E402
from smriti.engines import get_engine  # noqa: E402
from smriti.imaging import SUPPORTED_SUFFIXES, load_bgr  # noqa: E402
from smriti.pipeline import index_event_sync, ingest_bytes  # noqa: E402

BACKDROPS = [  # BGR, loosely: beach, forest, evening, indoor, snow
    (168, 196, 214), (96, 128, 86), (140, 108, 74), (86, 92, 104), (206, 208, 210),
]


def load_people(dataset: Path, min_images: int) -> dict[str, list[Path]]:
    people = {}
    for folder in sorted(p for p in dataset.iterdir() if p.is_dir()):
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if len(images) >= min_images:
            people[folder.name] = images
    return people


def face_crop(engine, path: Path, margin: float = 0.55) -> np.ndarray | None:
    """The largest face in an image, with enough context to look like a person."""
    bgr = load_bgr(path)
    faces = engine.detect_and_embed(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.w * f.h)
    pad_x, pad_y = int(face.w * margin), int(face.h * margin)
    x0 = max(0, face.x - pad_x)
    y0 = max(0, face.y - pad_y)
    x1 = min(bgr.shape[1], face.x + face.w + pad_x)
    y1 = min(bgr.shape[0], face.y + face.h + pad_y)
    crop = bgr[y0:y1, x0:x1]
    return crop if crop.size else None


def compose(crops: list[np.ndarray], rng: random.Random,
            width: int = 1400, height: int = 900) -> bytes:
    """Paste crops side by side on a plain backdrop, with per-person jitter."""
    base = np.zeros((height, width, 3), np.uint8)
    base[:] = BACKDROPS[rng.randrange(len(BACKDROPS))]
    # A soft vertical gradient so the detector is not fed a flat colour field.
    gradient = np.linspace(0.82, 1.12, height, dtype=np.float32)[:, None, None]
    base = np.clip(base * gradient, 0, 255).astype(np.uint8)

    slot = width // max(len(crops), 1)
    for i, crop in enumerate(crops):
        target_h = rng.randint(int(height * 0.34), int(height * 0.72))
        scale = target_h / crop.shape[0]
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), target_h),
                             interpolation=cv2.INTER_AREA)
        # Brightness and blur jitter: a real album is not evenly lit or focused.
        resized = np.clip(resized * rng.uniform(0.78, 1.18), 0, 255).astype(np.uint8)
        if rng.random() < 0.25:
            resized = cv2.GaussianBlur(resized, (3, 3), 0)

        x = i * slot + rng.randint(0, max(1, slot - resized.shape[1] // 2)) - resized.shape[1] // 4
        y = rng.randint(int(height * 0.08), max(int(height * 0.09), height - target_h - 10))
        x, y = max(0, min(x, width - resized.shape[1])), max(0, min(y, height - resized.shape[0]))
        base[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

    ok, buf = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("failed to encode a composite")
    return buf.tobytes()


def evaluate(event: dict, engine, selfies: dict[str, Path],
             truth: dict[str, set[str]], names: dict[str, str]) -> None:
    """Search once per person and score the results against the answer key."""
    index = matcher.get_index(event["id"], engine.dim)
    print(f"\n  end-to-end search over {index.n_faces} faces in "
          f"{len(names)} photos\n")
    print(f"  {'person':20s} {'in album':>9s} {'found':>6s} {'missed':>7s} "
          f"{'wrong':>6s} {'recall':>8s}")
    print("  " + "-" * 62)

    totals = {"expected": 0, "found": 0, "wrong": 0}
    latencies = []
    for person, selfie_path in sorted(selfies.items()):
        queries, _ = matcher.build_queries(engine, [load_bgr(selfie_path)])
        if queries.shape[0] == 0:
            print(f"  {person:20s} (no face in the held-out selfie -- skipped)")
            continue
        started = time.perf_counter()
        hits = index.search(queries, engine)
        latencies.append((time.perf_counter() - started) * 1000)

        # "maybe" is a review tier in the UI, not an assertion, so score on the
        # two tiers the product actually presents as matches.
        returned = {h.photo_id for h in hits if h.tier in ("sure", "likely")}
        expected = truth[person]
        found = returned & expected
        wrong = returned - expected
        totals["expected"] += len(expected)
        totals["found"] += len(found)
        totals["wrong"] += len(wrong)
        recall = len(found) / max(len(expected), 1)
        print(f"  {person:20s} {len(expected):9d} {len(found):6d} "
              f"{len(expected) - len(found):7d} {len(wrong):6d} {recall:8.1%}")

    recall = totals["found"] / max(totals["expected"], 1)
    precision = totals["found"] / max(totals["found"] + totals["wrong"], 1)
    print("  " + "-" * 62)
    print(f"  {'TOTAL':20s} {totals['expected']:9d} {totals['found']:6d} "
          f"{totals['expected'] - totals['found']:7d} {totals['wrong']:6d} {recall:8.1%}")
    print(f"\n  recall {recall:.1%} | precision {precision:.1%} | "
          f"median search {np.median(latencies):.1f} ms\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--photos", type=int, default=60, help="group photos to generate")
    parser.add_argument("--max-per-photo", type=int, default=5)
    parser.add_argument("--name", default="Demo album")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate", action="store_true", help="score the result")
    parser.add_argument("--keep", action="store_true", help="keep an existing demo event")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    engine = get_engine()
    rng = random.Random(args.seed)

    people = load_people(args.dataset, min_images=2)
    if len(people) < 2:
        sys.exit("need at least two person folders with two images each")

    print(f"  engine {engine.name} | {len(people)} people | {args.photos} photos to build")

    # Hold out one image per person as their selfie; the rest become album material.
    selfies, pool = {}, {}
    for person, images in people.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        selfies[person] = shuffled[0]
        crops = [c for c in (face_crop(engine, p) for p in shuffled[1:]) if c is not None]
        if crops:
            pool[person] = crops
        else:
            selfies.pop(person)
    print(f"  usable: {len(pool)} people with {sum(len(v) for v in pool.values())} album crops")

    if not args.keep:
        for existing in repo.list_events():
            if existing["name"] == args.name:
                repo.delete_event(existing["id"])
    created = repo.create_event(args.name, engine.name, engine.dim, retention_days=0,
                                notes="Generated by scripts/make_demo.py")
    event = created.event

    truth: dict[str, set[str]] = {person: set() for person in pool}
    names: dict[str, str] = {}
    roster = list(pool)
    for i in range(args.photos):
        count = rng.randint(1, min(args.max_per_photo, len(roster)))
        cast = rng.sample(roster, count)
        crops = [pool[person][rng.randrange(len(pool[person]))] for person in cast]
        filename = f"demo_{i + 1:03d}.jpg"
        result = ingest_bytes(event, filename, compose(crops, rng))
        if result.status != "queued":
            continue
        names[result.photo_id] = filename
        for person in cast:
            truth[person].add(result.photo_id)
        print(f"\r  building {i + 1}/{args.photos}", end="", flush=True)

    print(f"\n  indexing…")
    stats = index_event_sync(event)
    matcher.invalidate(event["id"])
    print(f"  {stats['photos']} photos -> {stats['faces']} faces in {stats['seconds']}s")

    truth_path = settings.data_dir / f"demo_truth_{event['id']}.json"
    truth_path.write_text(json.dumps(
        {"event": event["id"], "photos": names,
         "truth": {k: sorted(v) for k, v in truth.items()},
         "selfies": {k: str(v) for k, v in selfies.items()}}, indent=1), encoding="utf-8")

    print(f"\n  Album:       {event['name']}")
    print(f"  Share code:  {event['share_code']}")
    print(f"  Guest link:  /find.html?code={event['share_code']}")
    print(f"  Answer key:  {truth_path}")
    print(f"\n  Try a selfie: {selfies[roster[0]]}")

    if args.evaluate:
        evaluate(event, engine, selfies, truth, names)


if __name__ == "__main__":
    main()
