# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""What a surface looks like under smooth shading: normals blended from its vertices, as a renderer would."""

from __future__ import annotations

import numpy as np



SMOOTH_DEG = 60.0  # faces meeting at more than this shade as a hard edge, on the output and in the check


def corner_normals(points: np.ndarray, polys, degrees: float = SMOOTH_DEG) -> np.ndarray:
    """Unit normal per polygon corner, in the order of `polys`. Faces at the corner's vertex count if they're within
    `degrees` of the corner's own face, weighted by their angle at it, as Blender does."""

    flat, starts, sizes = corner_table(polys)
    ends = np.cumsum(sizes)
    following = np.arange(1, len(flat) + 1)
    following[ends - 1] = ends - sizes
    before = np.empty(len(flat), dtype=np.int64)
    before[following] = np.arange(len(flat))
    out = points[flat[following]] - points[flat]
    back = points[flat[before]] - points[flat]
    owner = np.repeat(np.arange(len(polys)), sizes)
    area = np.zeros((len(polys), 3))
    np.add.at(area, owner, np.cross(points[flat], points[flat[following]]))  # Newell's sum: right for quads that bend
    area /= np.maximum(np.linalg.norm(area, axis=1, keepdims=True), 1e-30)
    cosine = (out * back).sum(axis=1) / np.maximum(np.linalg.norm(out, axis=1) * np.linalg.norm(back, axis=1), 1e-30)
    weighted = area[owner] * np.arccos(np.clip(cosine, -1.0, 1.0))[:, None]
    facing = area[owner]
    limit = np.cos(np.radians(degrees))
    order = np.argsort(flat, kind="stable")
    counts = np.unique(flat, return_counts=True)[1]
    groups = np.split(order, np.cumsum(counts)[:-1])
    result = np.zeros_like(facing)
    for size in np.unique(counts).tolist():  # all vertices with this many corners at once
        rows = np.stack([g for g in groups if len(g) == size])
        near = (facing[rows] @ np.swapaxes(facing[rows], 1, 2)) >= limit
        result[rows] = near.astype(np.float64) @ weighted[rows]
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-30)


def sharp_edges(edge_of_corner: np.ndarray, face_of_corner: np.ndarray, normals: np.ndarray, edge_count: int, degrees: float = SMOOTH_DEG) -> np.ndarray:
    """Mask of the edges to mark sharp: faces meeting at more than `degrees`, or more than two faces on one edge."""

    uses = np.bincount(edge_of_corner, minlength=edge_count)
    sharp = uses > 2
    order = np.argsort(edge_of_corner, kind="stable")
    ends = np.cumsum(uses)
    pairs = np.flatnonzero(uses == 2)
    first, second = face_of_corner[order[ends[pairs] - 2]], face_of_corner[order[ends[pairs] - 1]]
    sharp[pairs] = (normals[first] * normals[second]).sum(axis=1) < np.cos(np.radians(degrees))
    return sharp


def corner_table(polys) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(every corner's vertex, where each polygon's corners start, corner counts), so many hits need no Python loop."""

    if hasattr(polys, "indptr"):  # types.FlatFaces
        return polys.flat, polys.indptr[:-1], np.diff(polys.indptr)
    sizes = np.fromiter((len(p) for p in polys), dtype=np.int64, count=len(polys))
    flat = np.fromiter((v for p in polys for v in p), dtype=np.int64, count=int(sizes.sum()))
    return flat, np.concatenate(([0], np.cumsum(sizes)[:-1])), sizes


def blended_normals(points: np.ndarray, table, normals: np.ndarray, faces: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The smooth-shaded normal at each point `at`, on the polygon `faces` says. `normals` is per corner. Polygons count as fans."""

    flat, starts, sizes = table[0], table[1][faces], table[2][faces]
    out = np.zeros((len(faces), 3))
    if not len(faces):
        return out
    best = np.full(len(faces), -np.inf)
    for k in range(1, int(sizes.max()) - 1):
        rows = np.flatnonzero(sizes > k + 1)
        a, b, c = starts[rows], starts[rows] + k, starts[rows] + k + 1
        edge1, edge2, to = points[flat[b]] - points[flat[a]], points[flat[c]] - points[flat[a]], at[rows] - points[flat[a]]
        d11, d12, d22 = (edge1 * edge1).sum(axis=1), (edge1 * edge2).sum(axis=1), (edge2 * edge2).sum(axis=1)
        d1, d2 = (to * edge1).sum(axis=1), (to * edge2).sum(axis=1)
        under = d11 * d22 - d12 * d12
        under[np.abs(under) < 1e-30] = 1e-30
        v, w = (d22 * d1 - d12 * d2) / under, (d11 * d2 - d12 * d1) / under
        u = 1.0 - v - w
        inside = np.minimum(np.minimum(u, v), w)  # the triangle that holds the point is the one least outside
        take = inside > best[rows]
        best[rows[take]] = inside[take]
        chosen = rows[take]
        out[chosen] = normals[a[take]] * u[take, None] + normals[b[take]] * v[take, None] + normals[c[take]] * w[take, None]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)
