"""Grouping an event's faces into distinct people.

This powers one organiser-facing number — "roughly 11 different people appear in
this album" — and the contact sheet beside it. It is *not* used for guest search,
which always matches against the raw face vectors.

**Why not connected components.** The obvious approach — link faces above a
threshold, take connected components — is forced into a bad trade. Set the
threshold low and components *chain*: A resembles B, B resembles C, so A and C
merge even though they are strangers, and one blob swallows the album. Set it
high to prevent that, and each person shatters into fragments instead. Measured
on a 193-face album containing 11 people, components at 0.57 returned 17 groups:
the 11 real people plus 6 splinters.

**Chinese Whispers** (Biemann, 2006) removes the trade. Every face starts with
its own label, then repeatedly adopts the *weighted-most-popular* label among its
neighbours. A single weak edge between two people is outvoted by the dozens of
strong edges inside each one, so the algorithm can use a much lower threshold —
recovering the fragments — without the chaining. It is iterative and randomised,
so a fixed seed keeps the answer reproducible.

The person count is still an estimate and is reported as one.
"""

from __future__ import annotations

import numpy as np

from . import repo
from .engines.base import FaceEngine

#: Beyond this many faces, skip rather than block the organiser's page.
MAX_FACES = 30_000
#: Cap on neighbours kept per face. Bounds memory at O(n*k) and costs nothing:
#: a face with 64 strong neighbours is already unambiguous.
MAX_NEIGHBOURS = 64
_BLOCK = 2048
_ITERATIONS = 20
_SEED = 20260903


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]  # path halving
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _neighbours(matrix: np.ndarray, threshold: float,
                k: int = MAX_NEIGHBOURS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sparse similarity graph: per face, its strongest neighbours above the bar.

    Computed in blocks so peak memory is (block x n) rather than (n x n).
    """
    n = matrix.shape[0]
    graph: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, n, _BLOCK):
        sims = matrix[start:start + _BLOCK] @ matrix.T
        for row_index in range(sims.shape[0]):
            i = start + row_index
            row = sims[row_index]
            row[i] = -1.0  # never a neighbour of itself
            idx = np.flatnonzero(row >= threshold)
            if idx.size > k:
                idx = idx[np.argpartition(row[idx], -k)[-k:]]
            graph.append((idx.astype(np.int32), row[idx].astype(np.float32)))
    return graph


def _chinese_whispers(graph, n: int, iterations: int = _ITERATIONS) -> np.ndarray:
    labels = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(_SEED)
    for _ in range(iterations):
        changed = 0
        for i in rng.permutation(n):
            idx, weights = graph[i]
            if idx.size == 0:
                continue
            votes: dict[int, float] = {}
            for neighbour, weight in zip(idx, weights):
                label = int(labels[neighbour])
                votes[label] = votes.get(label, 0.0) + float(weight)
            # Deterministic tie-break: heaviest vote, then lowest label id.
            best = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if best != labels[i]:
                labels[i] = best
                changed += 1
        if changed == 0:
            break
    return labels


def _components(graph, n: int) -> np.ndarray:
    uf = _UnionFind(n)
    for i, (idx, _) in enumerate(graph):
        for j in idx:
            uf.union(i, int(j))
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def cluster_event(event: dict, engine: FaceEngine,
                  link_threshold: float | None = None,
                  method: str = "chinese-whispers") -> dict:
    """Return {"people": [...], "n_people": int, "skipped": bool}.

    ``method`` is ``"chinese-whispers"`` (default) or ``"components"``, which is
    kept so the two can be compared on real data rather than argued about.
    """
    matrix, face_ids, photo_ids, face_px = repo.load_face_matrix(event["id"], engine.dim)
    n = matrix.shape[0]
    if n == 0:
        return {"people": [], "n_people": 0, "skipped": False, "faces": 0}
    if n > MAX_FACES:
        return {"people": [], "n_people": 0, "skipped": True, "faces": n}

    if link_threshold is not None:
        threshold = link_threshold
    elif method == "components":
        # Components need a high bar or they chain the whole album together.
        threshold = min(0.98, engine.threshold_high + 0.05)
    else:
        # Halfway between the review floor and the match bar. Chinese Whispers
        # survives a permissive graph, so we can go below the search threshold
        # to gather a person's weaker faces. Measured on a 193-face album of 11
        # people (sface, low 0.30 / match 0.38, so 0.34): 13 groups with zero
        # merge errors, against 16-17 for components at any threshold it can
        # safely use. Going lower starts merging people -- 0.30 costs 2.
        threshold = (engine.threshold_low + engine.threshold) / 2

    graph = _neighbours(matrix, threshold)
    labels = (_components(graph, n) if method == "components"
              else _chinese_whispers(graph, n))

    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(i)

    people = []
    for members in groups.values():
        # The cover face is the biggest one: most likely to be a clear portrait.
        cover = max(members, key=lambda i: float(face_px[i]))
        people.append({
            "size": len(members),
            "photos": sorted({photo_ids[i] for i in members}),
            "cover_face_id": face_ids[cover],
            "cover_photo_id": photo_ids[cover],
        })
    people.sort(key=lambda p: -p["size"])
    for rank, person in enumerate(people, 1):
        person["id"] = rank

    return {
        "people": people,
        "n_people": len(people),
        "skipped": False,
        "faces": n,
        "method": method,
        "threshold": round(float(threshold), 3),
    }


def summarise(result: dict, top: int = 24) -> dict:
    """A compact version for the admin API: counts plus the largest groups."""
    people = result.get("people", [])
    singletons = sum(1 for p in people if p["size"] == 1)
    return {
        "n_people": result["n_people"],
        "faces": result.get("faces", 0),
        "skipped": result.get("skipped", False),
        "singletons": singletons,
        "method": result.get("method"),
        "threshold": result.get("threshold"),
        "top": [
            {"id": p["id"], "faces": p["size"], "photos": len(p["photos"]),
             "cover_photo_id": p["cover_photo_id"], "cover_face_id": p["cover_face_id"]}
            for p in people[:top]
        ],
    }
