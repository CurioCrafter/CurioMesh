# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Bringing a source's detail onto a smoothed level of a clean cage: where each vertex should go, and the guards.

Each vertex looks along its own normal, both ways, no further than a little more than the edges of its level, and takes
the closest surface that faces the same way. A wall that faces the other way is the far side of a thin part, so it is
never taken. Vertices that find nothing, and quads that the move would fold, take their neighbours' move instead.
"""

from __future__ import annotations

import numpy as np

REACH_EDGES = 2.0  # how far a vertex may look, in the mean edge lengths of its level
FACING = 0.1  # the surface found must face this much the same way as the vertex (a dot product)
FOLD_PASSES = 4


def look(points: np.ndarray, normals: np.ndarray, reach: float, ray, nearest) -> tuple[np.ndarray, np.ndarray]:
    """(targets, found mask). `ray(origin, direction, reach)` and `nearest(origin, reach)` answer like a BVHTree."""

    targets = points.copy()
    found = np.zeros(len(points), dtype=bool)
    # Limitation: a plain Python loop over every vertex, about 1 s per 100k; vectorize it if it hurts
    for i, (p, n) in enumerate(zip(points.tolist(), normals.tolist())):
        best, best_at = reach * 2.0, None
        for sign in (1.0, -1.0):
            hit = ray(p, (n[0] * sign, n[1] * sign, n[2] * sign), reach)
            if hit[0] is None or hit[3] >= best:
                continue
            if hit[1][0] * n[0] + hit[1][1] * n[1] + hit[1][2] * n[2] < FACING:
                continue
            best, best_at = hit[3], hit[0]
        if best_at is None:  # a corner of the source sticks out past this vertex's normal
            hit = nearest(p, reach)
            if hit[0] is not None and hit[1][0] * n[0] + hit[1][1] * n[1] + hit[1][2] * n[2] >= FACING:
                best_at = hit[0]
        if best_at is not None:
            targets[i] = best_at[0], best_at[1], best_at[2]
            found[i] = True
    return targets, found


def _neighbour_table(count: int, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(from, to) over every quad edge, both ways, without repeats."""

    a = np.concatenate([quads[:, k] for k in range(4)])
    b = np.concatenate([quads[:, (k + 1) % 4] for k in range(4)])
    keys = np.unique(np.minimum(a, b) * count + np.maximum(a, b))
    low, high = keys // count, keys % count
    return np.concatenate((low, high)), np.concatenate((high, low))


def _neighbour_mean(values: np.ndarray, weight: np.ndarray, source: np.ndarray, sink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` over each vertex's neighbours with weight, and the total weight."""

    total = np.bincount(sink, weights=weight[source], minlength=len(values))
    mean = np.stack([np.bincount(sink, weights=values[source, k] * weight[source], minlength=len(values)) for k in range(3)], axis=1)
    return mean / np.maximum(total, 1e-30)[:, None], total


def _quad_normals(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    return np.cross(points[quads[:, 2]] - points[quads[:, 0]], points[quads[:, 3]] - points[quads[:, 1]])


def folded(before: np.ndarray, after: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Mask of the quads that turned over between the smoothed level and the moved one."""

    return (_quad_normals(before, quads) * _quad_normals(after, quads)).sum(axis=1) <= 0.0


def settle(points: np.ndarray, targets: np.ndarray, found: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(new points, vertices that found nothing, quads still folded): the guards, on the smoothed level's quads."""

    moves = targets - points
    source, sink = _neighbour_table(len(points), quads)
    known = found.astype(np.float64)
    lost = int((~found).sum())
    for _ in range(64):  # a move spreads inward from the vertices that found one; what stays out of reach stays put
        missing = np.flatnonzero(known == 0.0)
        if not len(missing):
            break
        mean, total = _neighbour_mean(moves, known, source, sink)
        fill = missing[total[missing] > 0.0]
        if not len(fill):
            break
        moves[fill] = mean[fill]
        known[fill] = 1.0
    every = np.ones(len(points))
    for _ in range(FOLD_PASSES):
        bad = folded(points, points + moves, quads)
        if not bad.any():
            break
        at = np.unique(quads[bad])
        moves[at] = _neighbour_mean(moves, every, source, sink)[0][at]
    bad = folded(points, points + moves, quads)
    if bad.any():  # what is still turned over goes back to the smooth surface
        moves[np.unique(quads[bad])] = 0.0
    return points + moves, lost, int(folded(points, points + moves, quads).sum())


def hard_edges(points: np.ndarray, corners: np.ndarray, sizes: np.ndarray, degrees: float) -> list[tuple[int, int]]:
    """Vertex pairs of the edges whose two faces meet at more than `degrees`. Faces are `sizes` long runs of `corners`."""

    points = np.asarray(points, dtype=np.float64)
    ends = np.cumsum(sizes)
    following = np.arange(1, len(corners) + 1)
    following[ends - 1] = ends - sizes
    a, b = corners, corners[following]
    owner = np.repeat(np.arange(len(sizes)), sizes)
    normal = np.zeros((len(sizes), 3))
    np.add.at(normal, owner, np.cross(points[a], points[b]))  # Newell's sum
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    count = len(points)
    keys = np.minimum(a, b) * count + np.maximum(a, b)
    order = np.argsort(keys, kind="stable")
    twins = np.flatnonzero(keys[order[1:]] == keys[order[:-1]])
    first, second = order[twins], order[twins + 1]
    hard = (normal[owner[first]] * normal[owner[second]]).sum(axis=1) < np.cos(np.radians(degrees))
    found = keys[first][hard]
    return [(int(k // count), int(k % count)) for k in np.unique(found)]


def mesh_facts(points: np.ndarray, tris: np.ndarray) -> dict:
    """What the Auto button reads: how the faces meet, how many loose parts, how much is open."""

    points = np.asarray(points, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    normal = np.cross(points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]])
    area = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(area, 1e-30)[:, None]
    a = tris.ravel()
    b = tris[:, [1, 2, 0]].ravel()
    count = len(points)
    keys = np.minimum(a, b) * count + np.maximum(a, b)
    order = np.argsort(keys, kind="stable")
    same = keys[order[1:]] == keys[order[:-1]]
    first, second = order[:-1][same] // 3, order[1:][same] // 3
    angle = np.degrees(np.arccos(np.clip((normal[first] * normal[second]).sum(axis=1), -1.0, 1.0)))
    uses = np.unique(keys, return_counts=True)[1]
    # Loose parts: label spreading over the edges, jumping through parents, which ends in a few rounds.
    source, sink = np.concatenate((a, b)), np.concatenate((b, a))
    label = np.arange(count)
    while True:
        lowest = label.copy()
        np.minimum.at(lowest, sink, label[source])
        lowest = lowest[lowest]
        if np.array_equal(lowest, label):
            break
        label = lowest
    sizes = np.bincount(label[tris[:, 0]], minlength=count)
    return {
        "faces": len(tris),
        "flat": float((angle < 1.0).mean()) if len(angle) else 0.0,
        "sharp": float((angle > 30.0).mean()) if len(angle) else 0.0,
        "median_angle": float(np.median(angle)) if len(angle) else 0.0,
        "open": float((uses == 1).mean()),
        "parts": int((sizes > 0).sum()),
        "biggest_part": float(sizes.max() / max(len(tris), 1)),
    }
