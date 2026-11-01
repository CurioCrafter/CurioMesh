# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""What a surface looks like under smooth shading: normals blended from its vertices, as a renderer would."""

from __future__ import annotations

import numpy as np


def vertex_normals(points: np.ndarray, polys) -> np.ndarray:
    """Unit normal per vertex: the faces around it, weighted by their angle at it, as Blender does."""

    sizes = np.fromiter((len(p) for p in polys), dtype=np.int64, count=len(polys))
    flat = np.fromiter((v for p in polys for v in p), dtype=np.int64, count=int(sizes.sum()))
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
    normals = np.zeros_like(points, dtype=np.float64)
    np.add.at(normals, flat, area[owner] * np.arccos(np.clip(cosine, -1.0, 1.0))[:, None])
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)


def corner_table(polys) -> tuple[np.ndarray, np.ndarray]:
    """(corners per polygon padded to the largest, corner counts), so looking up many hits needs no Python loop."""

    sizes = np.fromiter((len(p) for p in polys), dtype=np.int64, count=len(polys))
    corners = np.zeros((len(polys), int(sizes.max()) if len(polys) else 0), dtype=np.int64)
    for row, poly in enumerate(polys):
        corners[row, : len(poly)] = poly
    return corners, sizes


def blended_normals(points: np.ndarray, table, normals: np.ndarray, faces: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The smooth-shaded normal at each point `at`, which lies on the polygon `faces` says. Polygons count as fans."""

    corners, sizes = table[0][faces], table[1][faces]
    out = np.zeros((len(faces), 3))
    if not len(faces):
        return out
    best = np.full(len(faces), -np.inf)
    for k in range(1, corners.shape[1] - 1):
        rows = np.flatnonzero(sizes > k + 1)
        a, b, c = corners[rows, 0], corners[rows, k], corners[rows, k + 1]
        edge1, edge2, to = points[b] - points[a], points[c] - points[a], at[rows] - points[a]
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
