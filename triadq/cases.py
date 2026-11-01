# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""A finite set of test cases: name what kind of part a piece is, and predict how reducing it will go from that.

Idea from cav3dw3ll3r (https://github.com/cav3dw3ll3r). A few cheap measurements put a piece in one of a small set of
named cases. `cases_table.json` holds what the camera saw on parts of those cases on other models (rebuild it with
`scripts/build_case_table.py`), so the loss at each size can be guessed before looking. The guess only decides what to
look at first and what to skip. The camera still measures every size that gets picked and the whole result. Pure NumPy.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

CASES = ("tiny part", "flat plate", "thin sheet", "box", "tube", "dome", "rough scan", "ornament", "organic smooth")
FLAT_DEG = 1.0  # an edge bent less than this is flat
SHARP_DEG = 30.0  # and more than this is a hard edge
BEND_BINS = (0.0, 1.0, 5.0, 15.0, 30.0, 60.0, 180.0)  # the curvature histogram: edge length per bend angle in degrees
RING_CELLS = 64  # least rows of the (radius, height) grid a turned part's vertices fall on: finer for dense parts, or anything dense fills every cell
MIN_SPIN = 8  # vertices in one cell to count as a ring
ENOUGH_PARTS = 4  # parts of a case measured at a size before its row is believed


def _loops(boundary_edges: np.ndarray) -> int:
    """How many closed boundary loops the open edges form: union-find, as a loop is one connected group of edges."""
    if not len(boundary_edges):
        return 0
    ids, flat = np.unique(boundary_edges, return_inverse=True)
    parent = list(range(len(ids)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in flat.reshape(-1, 2).tolist():
        parent[find(a)] = find(b)
    return len({find(a) for a in range(len(ids))})


def _revolve_fit(q: np.ndarray, axes: tuple[int, ...] = (0, 1, 2)) -> float:
    """Share of the vertices on rings around the best of the piece's main axes. A turned part has nearly all of them."""
    if len(q) < MIN_SPIN * 3:
        return 0.0
    size = float(np.ptp(q, axis=0).max()) or 1.0
    rows = max(RING_CELLS, int(2.0 * math.sqrt(len(q))))
    best = 0.0
    for axis in axes:
        rest = [k for k in range(3) if k != axis]
        middle = (q[:, rest].max(axis=0) + q[:, rest].min(axis=0)) / 2.0
        radius = np.linalg.norm(q[:, rest] - middle, axis=1) / size
        height = (q[:, axis] - q[:, axis].min()) / size
        cell = np.floor(radius * rows * 2).astype(np.int64) * rows * 4 + np.floor(height * rows).astype(np.int64)
        _ids, inverse, count = np.unique(cell, return_inverse=True, return_counts=True)
        best = max(best, float((count[inverse] >= MIN_SPIN).mean()))
    return best


def features(points, polys, extent: float = 1.0) -> dict:
    """The feature vector of a piece: plain numbers, cheap enough to skip one single look at the piece."""
    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray([(p[0], p[k], p[k + 1]) for p in polys for k in range(1, len(p) - 1)], dtype=np.int64).reshape(-1, 3)
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(double_area, 1e-30)[:, None]
    area = double_area / 2.0
    total = max(float(area.sum()), 1e-30)
    middle = (a + b + c) / 3.0
    mean = (middle * area[:, None]).sum(axis=0) / total
    spread = ((middle - mean) * area[:, None]).T @ (middle - mean) / total
    _values, turn = np.linalg.eigh(spread)
    q = (points - mean) @ turn[:, ::-1]
    span = np.sort(np.ptp(q, axis=0))[::-1]
    long = max(float(span[0]), 1e-30)

    count = len(tris)
    edge = np.vstack((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    owner = np.tile(np.arange(count), 3)
    far = np.concatenate((tris[:, 2], tris[:, 0], tris[:, 1]))  # the corner across each edge, to sign the bend
    edge.sort(axis=1)
    key = edge[:, 0] * (len(points) + 1) + edge[:, 1]
    order = np.argsort(key, kind="stable")
    key, owner, far = key[order], owner[order], far[order]
    edge = edge[order]
    same = np.flatnonzero(key[1:] == key[:-1])
    first = np.ones(len(key), dtype=bool)
    first[same + 1] = False
    unique_edges = int(first.sum())
    counts = np.diff(np.concatenate((np.flatnonzero(first), [len(key)])))
    boundary = edge[first][counts == 1]
    pair_a, pair_b = owner[same], owner[same + 1]
    cosine = np.clip((normal[pair_a] * normal[pair_b]).sum(axis=1), -1.0, 1.0)
    bend = np.degrees(np.arccos(cosine))
    length = np.linalg.norm(points[edge[same, 0]] - points[edge[same, 1]], axis=1)
    length_total = max(float(length.sum()), 1e-30)
    histogram, _bins = np.histogram(bend, bins=BEND_BINS, weights=length)
    # Convex or concave: which side of a triangle's plane the corner across the edge lies on.
    side = ((points[far[same + 1]] - points[edge[same, 0]]) * normal[pair_a]).sum(axis=1)
    bent = bend >= FLAT_DEG
    outward = float(length[bent & (side < 0)].sum())
    inward = float(length[bent & (side >= 0)].sum())
    flat_by_face = np.zeros(count, dtype=bool)  # a triangle is flat when it bends off none of its neighbours
    flat_by_face[pair_a[bend >= FLAT_DEG]] = True
    flat_by_face[pair_b[bend >= FLAT_DEG]] = True
    # Largest flat directions: how much of the area faces one of the piece's three main directions, as a box does.
    aligned = np.abs(normal @ turn).max(axis=1) > math.cos(math.radians(8.0))
    smooth = bend[(bend < SHARP_DEG)]
    return {
        "triangles": count,
        "size": float(long / max(extent, 1e-30)),
        "middle_ratio": float(span[1] / long),
        "thin_ratio": float(span[2] / long),
        "bend": [round(float(v) / length_total, 4) for v in histogram],
        "flat_share": float(area[~flat_by_face].sum() / total),
        "sharp_share": float(length[bend >= SHARP_DEG].sum() / length_total),
        "box_share": float(area[aligned].sum() / total),
        "boundary_loops": _loops(boundary),
        "euler": len(np.unique(tris)) - unique_edges + count,
        "roughness": float(min(outward, inward) / max(outward + inward, 1e-30)),  # 0.5: bumps both ways, as on a scan
        "bend_mean": float(smooth.mean()) if len(smooth) else 0.0,
        "revolve_fit": _revolve_fit(q),
    }


def classify(f: dict) -> str:
    """The case of a piece. Tried in order, first match wins, so each case can be said in a sentence."""
    if f["triangles"] < 150 or f["size"] < 0.015:
        return "tiny part"
    if f["flat_share"] > 0.75 and f["thin_ratio"] < 0.12:
        return "flat plate"
    if f["thin_ratio"] < 0.04 or (f["boundary_loops"] > 0 and f["thin_ratio"] < 0.1):
        return "thin sheet"
    if f["box_share"] > 0.75 and f["flat_share"] > 0.5:
        return "box"
    if f["revolve_fit"] > 0.6:
        return "tube" if f["middle_ratio"] < 0.5 else "dome"
    if f["roughness"] > 0.3 and f["bend_mean"] > 5.0:
        return "rough scan"
    if f["sharp_share"] > 0.15 or f["euler"] < -4:
        return "ornament"
    return "organic smooth"


WHY = {
    "tiny part": "it is very small on the object or has few triangles",
    "flat plate": "it is mostly flat and thin",
    "thin sheet": "it is very thin compared to its length",
    "box": "its faces point along three directions",
    "tube": "its vertices sit on rings along a long axis",
    "dome": "its vertices sit on rings around an axis, and it is about as wide as it is long",
    "rough scan": "its surface bumps both in and out everywhere",
    "ornament": "it has many hard edges or holes",
    "organic smooth": "it curves gently with few hard edges",
}


def load_table(path: str | None = None) -> dict:
    with open(path or os.path.join(os.path.dirname(__file__), "cases_table.json"), encoding="utf-8") as stream:
        return json.load(stream)


def curve(case: str, shares: list[float], table: dict) -> list[tuple[float, float, float]]:
    """(low, typical, high) share of the view that changes at each share of the faces, from the table.

    A case with too few parts measured at a size falls back on all parts together.
    """
    rows = table["cases"].get(case, table["cases"]["all"])["shares"]
    every = table["cases"]["all"]["shares"]
    out = []
    for share in shares:
        row = rows.get(str(share))
        if row is None or row["parts"] < ENOUGH_PARTS:
            row = every.get(str(share), {"loss": (0.0, 0.5, 1.0)})
        out.append(tuple(row["loss"]))
    return out


def first_size(case: str, shares: list[float], floor: float, table: dict) -> int:
    """Which size to look at first: the smallest whose loss could be under `floor`, a share of the view. Smaller ones are hopeless.

    The low end of the table's range is used, so a size is only skipped when even the lucky parts of this case failed.
    The search walks back down if the camera shows this size has room, so a wrong guess costs little.
    """
    for index, (low, _typical, _high) in enumerate(curve(case, shares, table)):
        if low <= floor:
            return index
    return max(len(shares) - 1, 0)


def method_for(case: str, table: dict) -> str:
    """The method to try first on this case: 'revolve' when a rebuild usually beat Decimate on parts it fits, else 'decimate'."""
    seen = table["cases"].get(case, table["cases"]["all"])["revolve"]
    return "revolve" if seen["fits"] >= 3 and seen["wins"] * 2 > seen["fits"] else "decimate"


def explain(case: str, faces: int, change: float | None, skipped: int, looked: int, whole: bool = False, guessed: bool = False) -> str:
    """In plain words: what the part was taken for, what was tried, what was kept. `change` is a share of the part's view."""
    article = "an" if case[0] in "aeiou" else "a"
    plural = "" if looked == 1 else "s"
    tried = f"Looked at {looked} size{plural}."
    if skipped:
        tried = f"Skipped the {skipped} smallest size{'' if skipped == 1 else 's'}, which parts like this don't hold, and looked at {looked}."
    if whole:
        kept = "Kept it as it was, because no smaller size held up."
    elif change is None:
        kept = f"Made to measure at {faces:,} faces before pairing into quads, to land on the face count."
    else:
        kept = f"Kept {faces:,} faces before pairing into quads; {'an estimated ' if guessed else ''}{change * 100.0:.1f}% of its view changes."
    return f"Looks like {article} {case}: {WHY.get(case, 'it matched nothing else')}. {tried} {kept}"
