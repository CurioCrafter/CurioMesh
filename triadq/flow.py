# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Quad flow: putting vertices that closure or QuadriFlow made back on the source, so they carry shape."""

from __future__ import annotations

import numpy as np

from . import topology
from .patch import quad_quality
from .shading import corner_table
from .types import MeshData

FEATURE_TRIANGLES = 6  # with fewer source triangles around a vertex, their planes say more about noise than about shape
MIN_QUALITY = 0.05  # a snap may not push a quad below this, unless it was already lower


def _closest(find_nearest, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(closest source point, distance) for each of `points`, infinite where there is none. `find_nearest` is a native tree or Blender's one-at-a-time lookup."""

    if hasattr(find_nearest, "nearest"):
        at, _faces, away = find_nearest.nearest(points)
        return at, away
    at, away = np.array(points, dtype=np.float64), np.full(len(points), np.inf)
    for i, point in enumerate(at.tolist()):
        hit = find_nearest(point)
        if hit[0] is not None:
            at[i], away[i] = hit[0], hit[3]
    return at, away


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
    ids = np.flatnonzero(added)
    at, away = _closest(find_nearest, points[ids])
    near = away <= reach[ids]
    moved[ids[near]] = at[near]
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
    camera sees. `nearest` gives the closest of `points` to each of some locations. `pull` keeps a flat neighbourhood's vertex where
    it is, and `reach` limits a move to this share of the typical distance between vertices.
    """

    a, b, c = (source[triangles[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > 0.0
    normal = cross[keep] / double_area[keep, None]
    weight = double_area[keep] * 0.5
    middle = (a[keep] + b[keep] + c[keep]) / 3.0
    owner = nearest(middle)
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


def _ease(moved: np.ndarray, faces: list, find_nearest, rounds: int, wider: int) -> None:
    """Move corners of folded quads between their neighbours, in place: `rounds` staying on the source, then `rounds` not, over wider rings."""

    quads = np.asarray([f for f in faces if f is not None], dtype=np.int64)
    a, b = quads.ravel(), np.roll(quads, -1, axis=1).ravel()
    count = np.bincount(a, minlength=len(moved)) + np.bincount(b, minlength=len(moved))
    for k in range(2 * rounds):
        folded = quads[quad_quality(moved[quads]) <= 0.0]
        if not len(folded):
            return
        hit = np.zeros(len(moved), dtype=bool)
        hit[np.unique(folded)] = True
        for _ring in range(wider if k >= rounds else 0):
            hit[quads[hit[quads].any(axis=1)].ravel()] = True
        ids = np.flatnonzero(hit)
        total = np.zeros_like(moved)
        np.add.at(total, a, moved[b])
        np.add.at(total, b, moved[a])
        moved[ids] = total[ids] / np.maximum(count[ids], 1)[:, None]
        if k < rounds:
            moved[ids] = _closest(find_nearest, moved[ids])[0]


def unfold(points: np.ndarray, faces, find_nearest, rounds: int = 6, cycles: int = 6) -> tuple[np.ndarray, list[tuple[int, ...]], list[int]]:
    """Leave no quad folded. Returns (points, faces, source index per face).

    Easing costs a little shape on the first rounds and a little more later. A quad it can't open becomes two triangles
    along the diagonal that bends least: on badly crumpled spots that's the honest shape, and they show in the face
    counts. A stuck quad with a corner nothing else uses is QuadriFlow debris, and goes.
    """

    moved = np.array(points, dtype=np.float64)
    all_faces = [tuple(f) for f in faces]
    working: list = [f if len(f) == 4 else None for f in all_faces]
    array = np.asarray([f for f in working if f is not None], dtype=np.int64)
    if not len(array) or not (quad_quality(moved[array]) <= 0.0).any():
        return moved, all_faces, list(range(len(all_faces)))
    for cycle in range(cycles):
        _ease(moved, working, find_nearest, rounds, 1 + cycle)
    stuck = quad_quality(moved[array]) <= 0.0
    if not stuck.any():
        return moved, all_faces, list(range(len(all_faces)))
    uses = np.bincount(np.concatenate([np.asarray(f) for f in all_faces]), minlength=len(moved))
    bad = {tuple(q) for q in array[stuck].tolist()}
    final: list[tuple[int, ...]] = []
    keep: list[int] = []
    for k, f in enumerate(all_faces):
        if f not in bad:
            final.append(f)
            keep.append(k)
        elif not (uses[list(f)] == 1).any():
            a, b, c, d = f
            p = moved[list(f)]
            bend_ac = np.dot(np.cross(p[1] - p[0], p[2] - p[0]), np.cross(p[2] - p[0], p[3] - p[0]))
            bend_bd = np.dot(np.cross(p[2] - p[1], p[3] - p[1]), np.cross(p[3] - p[1], p[0] - p[1]))
            final += [(a, b, c), (a, c, d)] if bend_ac >= bend_bd else [(b, c, d), (b, d, a)]
            keep += [k, k]
    alive = sorted({v for f in final for v in f})
    remap = {v: i for i, v in enumerate(alive)}
    return moved[alive], [tuple(remap[v] for v in f) for f in final], keep


def snap_rows(points: np.ndarray, faces, find_nearest, sharpen=None) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Put a field-aligned mesh back on the source, and leave no quad folded."""

    moved = _closest(find_nearest, np.array(points, dtype=np.float64))[0]
    if sharpen is not None:
        moved = sharpen(moved)
    out_points, out_faces, _keep = unfold(moved, faces, find_nearest)
    return out_points, out_faces


def faces_that_gain(mesh: MeshData, find_nearest, share: float) -> set[int]:
    """Faces with a side middle or a center farther from the source than `share` of their longest side: new vertices there would carry shape."""

    if not len(mesh.faces):
        return set()
    points = np.asarray(mesh.vertices)
    flat, starts, sizes = corner_table(mesh.faces)
    following = np.arange(1, len(flat) + 1)
    following[starts + sizes - 1] = starts
    corners, ahead = points[flat], points[flat[following]]
    tolerance = share * np.maximum.reduceat(np.linalg.norm(ahead - corners, axis=1), starts)
    middles = np.add.reduceat(corners, starts) / sizes[:, None]
    away = _closest(find_nearest, np.vstack((middles, (corners + ahead) * 0.5)))[1]
    face = np.concatenate((np.arange(len(sizes)), np.repeat(np.arange(len(sizes)), sizes)))
    return set(np.unique(face[np.isfinite(away) & (away > tolerance[face])]).tolist())


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
