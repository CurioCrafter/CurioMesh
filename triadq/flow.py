# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Quad flow: putting vertices that closure or QuadriFlow made back on the source, so they carry shape."""

from __future__ import annotations

import numpy as np

from .patch import quad_quality
from .types import MeshData

MIN_QUALITY = 0.05  # a snap may not push a quad below this, unless it was already lower


def snap_new_vertices(closed: MeshData, coarse_points: np.ndarray, find_nearest) -> MeshData:
    """Move the vertices closure added onto the source, so they carry shape instead of lying flat on an edge or face.

    A vertex stays put if the source is farther than its longest edge, or if the move would fold a quad next to it.
    """

    points = np.array(closed.vertices, dtype=np.float64)
    known = {tuple(p) for p in np.round(coarse_points, 9).tolist()}
    added = np.asarray([tuple(p) not in known for p in np.round(points, 9).tolist()])
    quads = np.asarray([f for f in closed.faces if len(f) == 4], dtype=np.int64)
    if not len(quads):
        return closed
    reach = np.zeros(len(points))
    sides = np.linalg.norm(points[quads] - points[np.roll(quads, -1, axis=1)], axis=2)
    for k in range(4):
        np.maximum.at(reach, quads[:, k], sides[:, k])
        np.maximum.at(reach, quads[:, (k + 1) % 4], sides[:, k])
    moved = points.copy()
    for i in np.flatnonzero(added).tolist():
        hit = find_nearest(points[i].tolist())
        if hit[0] is not None and hit[3] <= reach[i]:
            moved[i] = hit[0]
    for _ in range(8):  # give back any vertex on a quad the move made bad
        bad = quads[quad_quality(moved[quads]) < np.minimum(quad_quality(points[quads]), MIN_QUALITY)]
        if not len(bad):
            break
        ids = np.unique(bad)
        moved[ids] = points[ids]
    return MeshData(moved, closed.faces, closed.face_materials, name=closed.name)


def snap_rows(points: np.ndarray, faces, find_nearest, rounds: int = 6) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Put a field-aligned mesh back on the source, and ease folded quads open by moving corners between their neighbours.

    The first rounds stay on the source. Later ones don't, and take the ring around the fold too: a fold that still
    remains costs a little shape instead. Stuck quads with a corner no other quad uses are QuadriFlow debris, and go.
    """

    faces = [tuple(f) for f in faces]
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64)
    moved = np.array(points, dtype=np.float64)

    def onto(ids) -> None:
        for i in ids:
            hit = find_nearest(moved[i].tolist())
            if hit[0] is not None:
                moved[i] = hit[0]

    onto(range(len(moved)))
    a, b = quads.ravel(), np.roll(quads, -1, axis=1).ravel()
    count = np.bincount(a, minlength=len(moved)) + np.bincount(b, minlength=len(moved))
    for k in range(3 * rounds):
        folded = quads[quad_quality(moved[quads]) <= 0.0]
        if not len(folded):
            break
        ids = np.unique(folded)
        if k >= rounds:
            hit = np.zeros(len(moved), dtype=bool)
            hit[ids] = True
            hit[quads[hit[quads].any(axis=1)].ravel()] = True
            ids = np.flatnonzero(hit)
        total = np.zeros_like(moved)
        np.add.at(total, a, moved[b])
        np.add.at(total, b, moved[a])
        moved[ids] = total[ids] / np.maximum(count[ids], 1)[:, None]
        if k < rounds:
            onto(ids.tolist())
    uses = np.bincount(quads.ravel(), minlength=len(moved))
    stuck = quad_quality(moved[quads]) <= 0.0
    debris = {tuple(q) for q in quads[stuck & (uses[quads] == 1).any(axis=1)].tolist()}
    if not debris:
        return moved, faces
    kept = [f for f in faces if f not in debris]
    alive = sorted({v for f in kept for v in f})
    remap = {v: i for i, v in enumerate(alive)}
    return moved[alive], [tuple(remap[v] for v in f) for f in kept]


__all__ = ["snap_new_vertices", "snap_rows"]
