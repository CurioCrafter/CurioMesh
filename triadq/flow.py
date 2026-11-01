# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Quad flow: putting vertices that closure or QuadriFlow made back on the source, so they carry shape."""

from __future__ import annotations

import numpy as np

from . import topology
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


def faces_that_gain(mesh: MeshData, find_nearest, tolerance: float) -> set[int]:
    """Faces with a side middle or a center farther than `tolerance` from the source: new vertices there would carry shape."""

    points = np.asarray(mesh.vertices)
    found = set()
    for index, face in enumerate(mesh.faces):
        p = points[list(face)]
        probes = np.vstack((p.mean(axis=0, keepdims=True), (p + np.roll(p, -1, axis=0)) * 0.5))
        for probe in probes.tolist():
            hit = find_nearest(probe)
            if hit[0] is not None and hit[3] > tolerance:
                found.add(index)
                break
    return found


def split_faces(mesh: MeshData, marked: set[int]) -> MeshData:
    """Split the marked faces into four, and just enough others that the result is all quads and has no T-junctions.

    Triangles are split too, whether marked or not. New vertices lie on the old faces, so the shape doesn't change yet.
    """

    faces = mesh.faces
    wanted = frozenset(
        tuple(sorted((faces[f][k], faces[f][(k + 1) % len(faces[f])]))) for f in marked for k in range(len(faces[f]))
    )
    selected = topology._turbo.parity_path_edges(mesh, wanted) if topology._turbo is not None else None
    if selected is None:
        selected = topology._parity_path_edges(mesh, wanted)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    new_points: list[np.ndarray] = []
    middle: dict[tuple[int, int], int] = {}
    for a, b in sorted(selected):
        middle[(a, b)] = len(points) + len(new_points)
        new_points.append((points[a] + points[b]) * 0.5)
    output: list[tuple[int, ...]] = []
    materials: list[str] = []
    for index, face in enumerate(faces):
        ring: list[int] = []
        for k, v in enumerate(face):
            ring.append(v)
            mid = middle.get(tuple(sorted((v, face[(k + 1) % len(face)]))))
            if mid is not None:
                ring.append(mid)
        where = [k for k, v in enumerate(ring) if v >= len(points)]
        if not where:
            cells = [tuple(ring)]
        elif len(ring) == 6 and len(where) == 2 and where[1] - where[0] == 3:  # opposite sides: cut across, with no new vertex
            turned = ring[where[0] :] + ring[: where[0]]
            cells = [tuple(turned[:4]), (turned[3], turned[4], turned[5], turned[0])]
        else:  # a fan from the center, with original corners only: a side's middle would make a straight corner
            center = len(points) + len(new_points)
            new_points.append(points[list(face)].mean(axis=0))
            size = len(ring)
            first = 1 if ring[1] >= len(points) or ring[3] >= len(points) or (size > 5 and ring[5] >= len(points)) else 0
            cells = [(center, ring[(first + 2 * k) % size], ring[(first + 2 * k + 1) % size], ring[(first + 2 * k + 2) % size]) for k in range(size // 2)]
        output += cells
        materials += [mesh.face_materials[index]] * len(cells)
    stacked = np.vstack([points, *[p[None] for p in new_points]]) if new_points else points
    return MeshData(stacked, output, materials, name=mesh.name)


__all__ = ["faces_that_gain", "snap_new_vertices", "snap_rows", "split_faces"]
