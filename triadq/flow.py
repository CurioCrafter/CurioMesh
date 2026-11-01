# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Quad flow: putting vertices that closure or QuadriFlow made back on the source, so they carry shape."""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import topology
from .patch import quad_quality
from .types import MeshData

FEATURE_TRIANGLES = 6  # with fewer source triangles around a vertex, their planes say more about noise than about shape
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


def seek_features(points: np.ndarray, source: np.ndarray, triangles: np.ndarray, nearest, pull: float = 0.05, reach: float = 0.5) -> np.ndarray:
    """Move each vertex to where the planes of the source triangles around it meet, as dual contouring does.

    A vertex on the surface only follows it. This one finds the crease or peak in its neighbourhood, which is what the
    camera sees. `nearest` gives the closest of `points` to a location. `pull` keeps a flat neighbourhood's vertex where
    it is, and `reach` limits a move to this share of the typical distance between vertices.
    """

    a, b, c = (source[triangles[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > 0.0
    normal = cross[keep] / double_area[keep, None]
    weight = double_area[keep] * 0.5
    middle = (a[keep] + b[keep] + c[keep]) / 3.0
    owner = np.fromiter((nearest(m) for m in middle.tolist()), dtype=np.int64, count=len(middle))
    matrix = np.zeros((len(points), 3, 3))
    target = np.zeros((len(points), 3))
    total = np.zeros(len(points))
    np.add.at(matrix, owner, weight[:, None, None] * normal[:, :, None] * normal[:, None, :])
    np.add.at(target, owner, weight[:, None] * normal * (normal * middle).sum(axis=1, keepdims=True))
    np.add.at(total, owner, weight)
    used = np.bincount(owner, minlength=len(points)) >= FEATURE_TRIANGLES
    if not used.any():
        return points
    ridge = pull * total[used]
    matrix = matrix[used] + ridge[:, None, None] * np.eye(3)
    placed = np.linalg.solve(matrix, (target[used] + ridge[:, None] * points[used])[:, :, None])[:, :, 0]
    spacing = np.sqrt(total[used] / np.maximum(np.median(total[used]), 1e-30)) * np.sqrt(np.median(total[used]))
    step = placed - points[used]
    length = np.linalg.norm(step, axis=1)
    scale = np.minimum(1.0, reach * spacing / np.maximum(length, 1e-30))
    out = points.copy()
    out[used] = points[used] + step * scale[:, None]
    return out


def _collapse_folded(points: np.ndarray, faces: list[tuple[int, ...]]) -> list[int | None]:
    """Close each still folded quad along its shorter diagonal, in place. Returns the new face list's source: index per face."""

    alive: list[int | None] = list(range(len(faces)))
    for _round in range(8):
        quads = [k for k, f in enumerate(faces) if f is not None and len(f) == 4]
        array = np.asarray([faces[k] for k in quads], dtype=np.int64)
        bad = np.flatnonzero(quad_quality(points[array]) <= 0.0)
        if not len(bad):
            break
        users: dict[int, list[int]] = {}
        for k, f in enumerate(faces):
            if f is not None:
                for v in f:
                    users.setdefault(v, []).append(k)
        touched: set[int] = set()
        for row in bad.tolist():
            k = quads[row]
            a, b, c, d = faces[k]
            first, second = (a, c) if np.linalg.norm(points[a] - points[c]) <= np.linalg.norm(points[b] - points[d]) else (b, d)
            around = {v for f in set(users[first]) | set(users[second]) if faces[f] is not None for v in faces[f]}
            if touched & around or set(users[first]) & set(users[second]) != {k}:
                continue  # closing it would pinch a neighbour, or the area changed this round
            touched |= around
            points[first] = (points[first] + points[second]) * 0.5
            for f in users[second]:
                faces[f] = tuple(first if v == second else v for v in faces[f])
            faces[k] = None
            alive[k] = None
    return alive


def unfold(points: np.ndarray, faces, find_nearest, rounds: int = 6) -> tuple[np.ndarray, list[tuple[int, ...]], list[int]]:
    """Ease folded quads open by moving corners between their neighbours. Returns (points, faces, source index per face).

    The first rounds stay on the source. Later ones don't, and take the ring around the fold too: a fold that remains
    costs a little shape instead. Quads still stuck go, either as debris with a corner no other quad uses, or by closing
    them along a diagonal.
    """

    faces = [tuple(f) for f in faces]
    quads = np.asarray([f for f in faces if len(f) == 4], dtype=np.int64)
    moved = np.array(points, dtype=np.float64)
    keep = list(range(len(faces)))
    if not len(quads) or not (quad_quality(moved[quads]) <= 0.0).any():
        return moved, faces, keep
    a, b = quads.ravel(), np.roll(quads, -1, axis=1).ravel()
    count = np.bincount(a, minlength=len(moved)) + np.bincount(b, minlength=len(moved))
    for k in range(8 * rounds):
        folded = quads[quad_quality(moved[quads]) <= 0.0]
        if not len(folded):
            break
        ids = np.unique(folded)
        if k >= rounds:
            hit = np.zeros(len(moved), dtype=bool)
            hit[ids] = True
            for _ring in range(1 + (k - rounds) // rounds):  # a stubborn fold takes a wider area with it
                hit[quads[hit[quads].any(axis=1)].ravel()] = True
            ids = np.flatnonzero(hit)
        total = np.zeros_like(moved)
        np.add.at(total, a, moved[b])
        np.add.at(total, b, moved[a])
        moved[ids] = total[ids] / np.maximum(count[ids], 1)[:, None]
        if k < rounds:
            for i in ids.tolist():
                hit_at = find_nearest(moved[i].tolist())
                if hit_at[0] is not None:
                    moved[i] = hit_at[0]
    uses = np.bincount(quads.ravel(), minlength=len(moved))
    stuck = quad_quality(moved[quads]) <= 0.0
    debris = {tuple(q) for q in quads[stuck & (uses[quads] == 1).any(axis=1)].tolist()}
    working: list = [None if f in debris else f for f in faces]
    source = _collapse_folded(moved, working)
    corners = Counter(tuple(sorted(f)) for k, f in enumerate(working) if f is not None and source[k] is not None)
    # Closing can leave two quads back to back on the same corners, touching nothing else.
    keep = [k for k, f in enumerate(working) if f is not None and source[k] is not None and corners[tuple(sorted(f))] == 1]
    alive = sorted({v for k in keep for v in working[k]})
    remap = {v: i for i, v in enumerate(alive)}
    return moved[alive], [tuple(remap[v] for v in working[k]) for k in keep], keep


def snap_rows(points: np.ndarray, faces, find_nearest, sharpen=None) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Put a field-aligned mesh back on the source, and leave no quad folded."""

    moved = np.array(points, dtype=np.float64)
    for i in range(len(moved)):
        hit = find_nearest(moved[i].tolist())
        if hit[0] is not None:
            moved[i] = hit[0]
    if sharpen is not None:
        moved = sharpen(moved)
    out_points, out_faces, _keep = unfold(moved, faces, find_nearest)
    return out_points, out_faces


def faces_that_gain(mesh: MeshData, find_nearest, share: float) -> set[int]:
    """Faces with a side middle or a center farther from the source than `share` of their longest side: new vertices there would carry shape."""

    points = np.asarray(mesh.vertices)
    found = set()
    for index, face in enumerate(mesh.faces):
        p = points[list(face)]
        tolerance = share * float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).max())
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


__all__ = ["faces_that_gain", "seek_features", "snap_new_vertices", "snap_rows", "split_faces", "unfold"]
