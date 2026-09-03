#!/usr/bin/env python
"""Measure an engine instead of trusting its README.

Point this at any folder-per-person dataset -- LFW's layout, or your own photos
sorted into folders -- and it reports the two numbers that decide whether this
product works: how often two photos of the same person score above the matching
threshold (recall), and how often two photos of *different* people do (the
false-match rate, which is the one that hands your photos to a stranger).

    dataset/
      alice/   a1.jpg a2.jpg a3.jpg
      bob/     b1.jpg b2.jpg
      ...

    python scripts/benchmark.py --dataset ./dataset --engine sface

Label hygiene: web-scraped folders contain mistakes, and a single mislabelled
photo poisons the impostor set. By default the script keeps, per person, only
the largest mutually-consistent group of faces and reports what it dropped.
Pass --no-clean to score the labels exactly as given.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smriti.engines import get_engine  # noqa: E402
from smriti.imaging import SUPPORTED_SUFFIXES, load_bgr  # noqa: E402


def collect(dataset: Path, engine, min_images: int, verbose: bool):
    """Embed the largest face in every image, grouped by folder name."""
    people: dict[str, list[dict]] = {}
    timings: list[float] = []
    no_face = 0

    folders = sorted(p for p in dataset.iterdir() if p.is_dir())
    for folder in folders:
        images = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if len(images) < min_images:
            continue
        entries = []
        for path in images:
            try:
                bgr = load_bgr(path)
            except Exception as exc:
                if verbose:
                    print(f"    unreadable {path.name}: {exc}")
                continue
            started = time.perf_counter()
            faces = engine.detect_and_embed(bgr)
            timings.append((time.perf_counter() - started) * 1000)
            if not faces:
                no_face += 1
                continue
            biggest = max(faces, key=lambda f: f.w * f.h)
            entries.append({"path": path, "vec": biggest.embedding,
                            "px": biggest.face_px, "n_faces": len(faces)})
        if len(entries) >= min_images:
            people[folder.name] = entries
        if verbose:
            print(f"  {folder.name:20s} {len(entries)}/{len(images)} usable")
    return people, timings, no_face


def clean(people: dict[str, list[dict]], loose: float, min_images: int):
    """Keep each person's largest mutually-consistent group of faces."""
    kept, dropped = {}, 0
    for name, entries in people.items():
        if len(entries) < 2:
            continue
        matrix = np.stack([e["vec"] for e in entries])
        sims = matrix @ matrix.T
        # Connected components over "these two are plausibly the same person".
        adjacency = sims >= loose
        seen, best = set(), []
        for start in range(len(entries)):
            if start in seen:
                continue
            stack, component = [start], []
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                component.append(node)
                stack.extend(int(j) for j in np.flatnonzero(adjacency[node]) if j not in seen)
            if len(component) > len(best):
                best = component
        dropped += len(entries) - len(best)
        if len(best) >= min_images:
            kept[name] = [entries[i] for i in best]
    return kept, dropped


def pairs(people: dict[str, list[dict]]):
    genuine, impostor = [], []
    for entries in people.values():
        for a, b in itertools.combinations(entries, 2):
            genuine.append(float(a["vec"] @ b["vec"]))
    names = list(people)
    for left, right in itertools.combinations(names, 2):
        for a in people[left]:
            for b in people[right]:
                impostor.append(float(a["vec"] @ b["vec"]))
    return np.array(genuine), np.array(impostor)


def collisions(people: dict[str, list[dict]], threshold: float, top: int = 8):
    """Cross-person pairs that score like the same person.

    On a clean dataset these are the model's false matches. On a scraped one
    they are usually label errors -- two folders holding the same face, or a
    group photo filed under whichever name was searched for. Either way you want
    to see the filenames before believing the false-match rate below.
    """
    found = []
    names = list(people)
    for left, right in itertools.combinations(names, 2):
        best = None
        for a in people[left]:
            for b in people[right]:
                score = float(a["vec"] @ b["vec"])
                if score >= threshold and (best is None or score > best[0]):
                    best = (score, a["path"], b["path"])
        if best:
            found.append((best[0], left, right, best[1].name, best[2].name))
    found.sort(reverse=True)
    return found[:top], len(found)


def sweep(genuine: np.ndarray, impostor: np.ndarray, engine, steps: np.ndarray) -> None:
    print(f"\n  {'threshold':>9s} {'recall':>8s} {'false match':>12s} {'precision':>10s}")
    print("  " + "-" * 43)
    for threshold in steps:
        tp = int((genuine >= threshold).sum())
        fp = int((impostor >= threshold).sum())
        recall = tp / max(len(genuine), 1)
        far = fp / max(len(impostor), 1)
        precision = tp / max(tp + fp, 1)
        marker = "  <- configured" if abs(threshold - engine.threshold) < 1e-9 else ""
        print(f"  {threshold:9.2f} {recall:8.1%} {far:12.2%} {precision:10.1%}{marker}")


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U); ties count as half."""
    scores = np.concatenate([genuine, impostor])
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tie groups so ties do not inflate the score.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    n_pos, n_neg = len(genuine), len(impostor)
    if not n_pos or not n_neg:
        return float("nan")
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--engine", default=None, help="sface | arcface | insightface")
    parser.add_argument("--min-images", type=int, default=2)
    parser.add_argument("--no-clean", action="store_true", help="trust the folder labels exactly")
    parser.add_argument("--clean-threshold", type=float, default=None,
                        help="loose similarity used for label cleaning (default: engine low tier)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        sys.exit(f"no such dataset folder: {args.dataset}")

    engine = get_engine(args.engine)
    print(f"\n  engine     {engine.name} ({engine.dim}-d, match >= {engine.threshold})")
    print(f"  dataset    {args.dataset}")

    started = time.time()
    people, timings, no_face = collect(args.dataset, engine, args.min_images, args.verbose)
    if not people:
        sys.exit("no person folder had enough usable images")

    raw_images = sum(len(v) for v in people.values())
    if not args.no_clean:
        loose = args.clean_threshold if args.clean_threshold is not None else engine.threshold_low
        people, dropped = clean(people, loose, args.min_images)
        print(f"  cleaning   dropped {dropped} inconsistent face(s) at >= {loose}")
    n_images = sum(len(v) for v in people.values())

    print(f"\n  people     {len(people)}")
    print(f"  images     {n_images} embedded ({no_face} with no detectable face)")
    print(f"  detection  {100 * (raw_images / max(raw_images + no_face, 1)):.1f}% of images yielded a face")
    if timings:
        arr = np.array(timings)
        print(f"  speed      {arr.mean():.0f} ms/photo mean, {np.percentile(arr, 95):.0f} ms p95 "
              f"({1000 / arr.mean():.1f} photos/s per core)")

    genuine, impostor = pairs(people)
    print(f"\n  genuine pairs   {len(genuine):,}  mean {genuine.mean():.3f}  "
          f"p5 {np.percentile(genuine, 5):.3f}")
    print(f"  impostor pairs  {len(impostor):,}  mean {impostor.mean():.3f}  "
          f"p95 {np.percentile(impostor, 95):.3f}  max {impostor.max():.3f}")
    print(f"  separation      {genuine.mean() - impostor.mean():.3f}   "
          f"ROC-AUC {roc_auc(genuine, impostor):.4f}")

    suspects, n_colliding = collisions(people, engine.threshold_high)
    if suspects:
        print(f"\n  {n_colliding} folder pair(s) contain faces scoring above "
              f"{engine.threshold_high} — inspect these before trusting the rate below:")
        for score, left, right, file_a, file_b in suspects:
            print(f"    {score:.3f}  {left}/{file_a}  ==  {right}/{file_b}")

    steps = np.unique(np.round(np.concatenate([
        np.arange(0.20, 0.75, 0.05), [engine.threshold_low, engine.threshold, engine.threshold_high],
    ]), 4))
    sweep(genuine, impostor, engine, steps)

    tp = int((genuine >= engine.threshold).sum())
    fp = int((impostor >= engine.threshold).sum())
    print(f"\n  At the configured threshold {engine.threshold}: "
          f"{tp}/{len(genuine)} genuine pairs matched, "
          f"{fp}/{len(impostor)} impostor pairs wrongly matched.")
    print(f"  ({time.time() - started:.1f}s total)\n")


if __name__ == "__main__":
    main()
