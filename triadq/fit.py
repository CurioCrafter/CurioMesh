# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Fit finished vertices and shading to the source, using its nearest points. Plain NumPy, the caller does the lookups.

Idea from cav3dw3ll3r: nearest neighbours against the source mesh, to win back lost shape.
"""

from __future__ import annotations

import numpy as np

from .checks import FOLD, _edge_faces, _normals
from .patch import _labels, face_adjacency, quad_quality

DOMINANT = 0.2  # a plane direction counts if it is this strong against the strongest: 1 on flat, 2 on a ridge, 3 at a corner
MOST_MOVE = 0.3  # share of a vertex's shortest side it may move
FACING = -0.2  # samples facing away from the vertex are the far side of a thin wall
MIN_QUALITY = 0.02
INSET = 0.15  # a corner asks for shading from this share of the way to its face's middle, so it hears its own side of a hard edge
SPLIT_DEG = 30.0  # faces this far apart meet at a hard edge
HEARS_DEG = 25.0  # samples turned further than this from the corner's own side are a bevel or another surface


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def fit_vertices(points, quads, sample_pos, sample_nrm, sample_area, owner) -> np.ndarray:
    """Move each vertex to best fit the planes of the source samples it owns (`owner` per sample, -1 for none).

    Solving for the planes together holds a ridge or corner where blending normals would round it off, and weak
    directions are left alone, so a vertex on a flat area only moves along its normal. Any vertex on a quad the move
    made bad goes back, and so does any corner of quads the move folded over each other or pushed through each other.
    """

    points = np.asarray(points, dtype=np.float64)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    normal = np.zeros_like(points)
    area_vector = np.cross(points[quads[:, 2]] - points[quads[:, 0]], points[quads[:, 3]] - points[quads[:, 1]])
    reach = np.full(len(points), np.inf)
    sides = np.linalg.norm(points[quads] - points[np.roll(quads, -1, axis=1)], axis=2)
    for k in range(4):
        np.add.at(normal, quads[:, k], area_vector)
        np.minimum.at(reach, quads[:, k], sides[:, k])
        np.minimum.at(reach, quads[:, (k + 1) % 4], sides[:, k])
    reach[~np.isfinite(reach)] = 0.0
    used = owner >= 0
    who, n = owner[used], sample_nrm[used]
    weight = sample_area[used] * ((n * normal[who]).sum(axis=1) > FACING * np.linalg.norm(normal[who], axis=1))
    # Sums by bincount, which adds in the same order np.add.at does, many times faster on millions of samples.
    count = len(points)
    planes = np.empty((count, 3, 3))
    scaled = weight[:, None] * n
    for i in range(3):
        for j in range(3):
            planes[:, i, j] = np.bincount(who, scaled[:, i] * n[:, j], count)
    along_normal = weight * (n * (sample_pos[used] - points[who])).sum(axis=1)
    pull = np.stack([np.bincount(who, along_normal * n[:, k], count) for k in range(3)], axis=1)
    values, vectors = np.linalg.eigh(planes)
    strong = values > DOMINANT * np.maximum(values[:, 2:], 1e-30)
    along = np.where(strong, np.einsum("nij,ni->nj", vectors, pull) / np.maximum(values, 1e-30), 0.0)
    step = np.einsum("nij,nj->ni", vectors, along)
    step *= np.minimum(1.0, MOST_MOVE * reach / np.maximum(np.linalg.norm(step, axis=1), 1e-30))[:, None]
    moved = points + step
    limit = np.minimum(quad_quality(points[quads]), MIN_QUALITY)
    now = quad_quality(moved[quads])
    for _ in range(8):
        bad = quads[now < limit]
        if not len(bad):
            break
        ids = np.unique(bad)
        moved[ids] = points[ids]
        given = np.zeros(len(points), dtype=bool)
        given[ids] = True
        touched = np.flatnonzero(given[quads].any(axis=1))  # only these quads' corners changed
        now[touched] = quad_quality(moved[quads[touched]])
    return untangled(points, moved, quads)


def _crossings(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Each pair of quads that pass through each other, as one number (lower * count + higher)."""

    from .checks import crossing_pairs

    if not len(quads):
        return np.zeros(0, dtype=np.int64)
    pairs = np.asarray(crossing_pairs(points, [tuple(q) for q in quads.tolist()]), dtype=np.int64).reshape(-1, 2)
    return pairs.min(axis=1) * len(quads) + pairs.max(axis=1)


def untangled(before: np.ndarray, after: np.ndarray, quads: np.ndarray, rounds: int = 6) -> np.ndarray:
    """`after` with the corners put back of every quad the move folded over its neighbour, bent across a diagonal, or
    pushed through another quad, where `before` had none of that. Fitting each corner to its own share of the source
    pulls neighbouring corners different ways on thin walls and in creases, which tangles the faces between them."""

    moved = after.copy()
    if not len(quads):
        return moved
    edge = _edge_faces(quads)

    def folded(at: np.ndarray) -> np.ndarray:
        normal = _normals(at, quads)[0]
        return (normal[edge[:, 0]] * normal[edge[:, 1]]).sum(axis=1) < FOLD

    was_folded, was_bent, was_crossing = folded(before), quad_quality(before[quads]) <= 0.0, _crossings(before, quads)
    for _ in range(rounds):
        bad = np.zeros(len(quads), dtype=bool)
        bad[edge[folded(moved) & ~was_folded].ravel()] = True
        bad |= (quad_quality(moved[quads]) <= 0.0) & ~was_bent
        crossing = _crossings(moved, quads)
        new = crossing[~np.isin(crossing, was_crossing)]
        bad[new // len(quads)] = bad[new % len(quads)] = True
        ids = np.unique(quads[bad])
        ids = ids[(moved[ids] != before[ids]).any(axis=1)]
        if not len(ids):
            break
        moved[ids] = before[ids]
    return moved


def vertex_parts(points, faces, sample_part, nearest_samples) -> tuple[np.ndarray, np.ndarray]:
    """(result part of each vertex, source part of each result part): most vertices of a result part sit on the source part it came from.

    `nearest_samples(points)` gives the index of the source sample nearest to each point.
    """

    fa, fb = face_adjacency(faces)
    face_part = np.unique(_labels(len(faces), fa, fb), return_inverse=True)[1]
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(sizes.sum()))
    part = np.zeros(len(points), dtype=np.int64)
    part[flat] = np.repeat(face_part, sizes)
    count = int(face_part.max()) + 1
    source_parts = int(sample_part.max()) + 1
    if source_parts == 1:
        return part, np.zeros(count, dtype=np.int64)
    votes = np.zeros((count, source_parts), dtype=np.int64)
    nearest = nearest_samples(points[flat])
    np.add.at(votes, (part[flat], sample_part[nearest]), 1)
    return part, votes.argmax(axis=1)


def corner_normals(points, faces, sample_nrm, sample_area, sample_part, vertex_source, find_n, count: int = 12) -> np.ndarray:
    """One smooth-shading normal per face corner, in face order, from source samples next to the corner.

    Faces at a vertex split into sides at a hard edge, as smoothing by angle does, and that side's normal then picks
    which nearby samples may speak: not a bevel, not another loose part. `find_n(point, count)` gives
    [(position, sample index, distance)], as mathutils' KDTree does. Idea from cav3dw3ll3r.
    """

    points = np.asarray(points, dtype=np.float64)
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64, count=int(sizes.sum()))
    face = np.repeat(np.arange(len(faces)), sizes)
    first = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    following = np.arange(len(flat)) + 1
    following[first + sizes - 1] = first
    previous = np.arange(len(flat)) - 1
    previous[first] = first + sizes - 1
    face_normal = np.zeros((len(faces), 3))
    np.add.at(face_normal, face, np.cross(points[flat], points[flat[following]]))
    face_normal = _unit(face_normal)
    middle = np.zeros((len(faces), 3))
    np.add.at(middle, face, points[flat])
    middle /= sizes[:, None]
    turn = np.arccos(np.clip((_unit(points[flat[following]] - points[flat]) * _unit(points[flat[previous]] - points[flat])).sum(axis=1), -1.0, 1.0))
    order = np.argsort(flat, kind="stable")
    breaks = np.flatnonzero(flat[order][1:] != flat[order][:-1]) + 1
    side = np.zeros((len(flat), 3))
    split = np.cos(np.radians(SPLIT_DEG))
    groups = np.split(order, breaks)
    near = []
    for corners in groups:
        n = face_normal[face[corners]]
        near.append((n @ n.T) >= split)
        side[corners] = near[-1] @ (n * turn[corners][:, None])
    side = _unit(side)
    asks = points[flat] + INSET * (middle[face] - points[flat])
    index = np.zeros((len(flat), count), dtype=np.int64)
    away = np.full((len(flat), count), np.inf)
    for i, ask in enumerate(asks.tolist()):
        for j, (_position, sample, distance) in enumerate(find_n(ask, count)):
            index[i, j], away[i, j] = sample, distance
    n = sample_nrm[index]
    agree = np.einsum("nkj,nj->nk", n, side)
    found = np.isfinite(away)
    width = np.maximum(np.median(np.where(found, away, 0.0), axis=1, keepdims=True), 1e-30)
    hears = (np.abs(agree) > np.cos(np.radians(HEARS_DEG))) & found & (sample_part[index] == vertex_source[flat][:, None])
    weight = sample_area[index] * np.exp(-np.square(np.where(found, away, 0.0) / width)) * hears
    # A sample's winding says nothing, so read it the way this side faces. The side itself is the answer when none agree.
    heard = _unit(np.einsum("nk,nkj,nk->nj", weight, n, np.sign(agree)) + 1e-3 * np.maximum(weight.sum(axis=1, keepdims=True), 1e-30) * side)
    # Corners on one side of a vertex share their normal, or every soft edge shows as a seam.
    out = np.zeros_like(heard)
    for corners, together in zip(groups, near):
        out[corners] = together @ (heard[corners] * turn[corners][:, None])
    return _unit(out)
