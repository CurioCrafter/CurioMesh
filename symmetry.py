# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Symmetry: solve one half of a mesh and mirror it, so the seam is exact and the two sides are identical.

`find_plane` looks for the mirror plane (it needs a BVH of the mesh, so it runs inside Blender). `cut_half` and
`mirror_result` are plain NumPy: the first clips the faces to one side of the plane, the second snaps the open
border of the solved half onto the plane and welds a reflected copy to it. Mirrored vertices are reflected
coordinates, so the result is exactly symmetric and has no gap along the seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXES = "XYZ"
MATCH_TOLERANCE = 0.004  # a point counts as matching when its mirror lies this near the surface (share of the size)
ENOUGH = 0.90  # share of points that must match for Auto to call a mesh symmetric
ENOUGH_CHOSEN = 0.70  # ... when you picked the axis yourself: a barrier with different ends is still worth mirroring
SAMPLES = 4000
SNAP = 1e-5  # vertices this near the plane (share of the size) are put exactly on it before cutting


@dataclass
class Plane:
    axis: int
    offset: float
    match: float  # share of the sampled points whose mirror lies on the surface
    worst: float  # distance of the 99th percentile point from its mirror, as a share of the size

    def ok(self, chosen: bool = False) -> bool:
        return self.match >= (ENOUGH_CHOSEN if chosen else ENOUGH)

    def text(self) -> str:
        return f"{AXES[self.axis]}: {self.match * 100:.0f}% of the surface matches its mirror (99% within {self.worst * 100:.1f}% of the size)"


def _score(bvh, pts: np.ndarray, axis: int, offset: float, diag: float):
    mirrored = pts.copy()
    mirrored[:, axis] = 2.0 * offset - mirrored[:, axis]
    found = [bvh.find_nearest(p) for p in mirrored.tolist()]
    dist = np.asarray([f[3] if f[0] is not None else diag for f in found]) / diag
    near = np.asarray([f[0][axis] if f[0] is not None else 0.0 for f in found])
    return float((dist < MATCH_TOLERANCE).mean()), float(np.percentile(dist, 99)), dist, near, mirrored[:, axis]


def find_plane(co: np.ndarray, bvh, axes: str = "XYZ") -> list[Plane]:
    """The best mirror plane on each asked axis, best first. Tries the mesh centre and the origin, then nudges the plane."""
    lo, hi = co.min(axis=0), co.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    step = max(len(co) // SAMPLES, 1)
    pts = co[::step]
    planes = []
    for letter in axes.upper():
        axis = AXES.index(letter)
        best = None
        for offset in dict.fromkeys((float((lo[axis] + hi[axis]) / 2.0), 0.0)):
            match, worst, *_rest = _score(bvh, pts, axis, offset, diag)
            if best is None or match > best[0]:
                best = (match, offset)
        offset = best[1]
        for _ in range(2):  # a mirror that lands a little to one side tells where the plane really is
            match, worst, dist, near, mirrored = _score(bvh, pts, axis, offset, diag)
            used = dist < 3.0 * MATCH_TOLERANCE
            if used.sum() < 16:
                break
            offset += float((near[used] - mirrored[used]).mean()) / 2.0
        match, worst, *_rest = _score(bvh, pts, axis, offset, diag)
        planes.append(Plane(axis, offset, match, worst))
    return sorted(planes, key=lambda p: -p.match)


def cut_half(co: np.ndarray, faces, axis: int, offset: float, keep: int = 1):
    """Clip the faces to one side of the plane (keep=1: the side with larger coordinates).

    Returns (co, faces, source) where source[i] is the index of the face that new face i was cut from. Every new
    vertex on the plane is exactly on it, and two faces sharing a cut edge share its new vertex.
    """
    co = np.asarray(co, dtype=np.float64).copy()
    diag = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0))) or 1.0
    side = keep * (co[:, axis] - offset)
    flat = np.abs(side) <= SNAP * diag
    co[flat, axis] = offset  # almost on the plane: put exactly on it, so the cut has no slivers
    side[flat] = 0.0
    new_co = [co]
    cuts: dict[tuple[int, int], int] = {}
    count = len(co)
    out_faces, source = [], []

    def cross(a: int, b: int) -> int:
        nonlocal count
        key = (a, b) if a < b else (b, a)
        if key not in cuts:
            t = side[a] / (side[a] - side[b])
            p = co[a] + (co[b] - co[a]) * t
            p[axis] = offset
            cuts[key] = count
            new_co.append(p[None, :])
            count += 1
        return cuts[key]

    for i, face in enumerate(faces):
        d = side[list(face)]
        if (d < 0).all() or (d == 0).all():
            continue
        if (d >= 0).all():
            out_faces.append(tuple(face))
            source.append(i)
            continue
        poly = []  # Sutherland-Hodgman against one plane: keep the vertices on the kept side, add the crossings
        for k, a in enumerate(face):
            b = face[(k + 1) % len(face)]
            da, db = side[a], side[b]
            if da >= 0:
                poly.append(a)
            if (da > 0 and db < 0) or (da < 0 and db > 0):
                poly.append(cross(a, b))
        if len(poly) >= 3:
            out_faces.append(tuple(poly))
            source.append(i)
    all_co = np.vstack(new_co)
    used = np.zeros(len(all_co), dtype=bool)
    for f in out_faces:
        used[list(f)] = True
    remap = np.cumsum(used) - 1
    return all_co[used], [tuple(int(remap[v]) for v in f) for f in out_faces], np.asarray(source, dtype=np.int64)


def border_vertices(faces, count: int) -> np.ndarray:
    """Mask of the vertices on an open edge (an edge used by one face only)."""
    seen: dict[tuple[int, int], int] = {}
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            key = (a, b) if a < b else (b, a)
            seen[key] = seen.get(key, 0) + 1
    mask = np.zeros(count, dtype=bool)
    for (a, b), n in seen.items():
        if n == 1:
            mask[a] = mask[b] = True
    return mask


def mirror_result(co: np.ndarray, faces, mat, axis: int, offset: float, size: float):
    """The solved half plus its reflection. Returns (co, faces, mat, snapped).

    The open border of the half that lies near the plane is moved exactly onto it (`snapped` counts those
    vertices, moved by at most a few percent of `size`); the reflected copy reuses them, so nothing is left open there.
    """
    co = np.asarray(co, dtype=np.float64).copy()
    near = border_vertices(faces, len(co)) & (np.abs(co[:, axis] - offset) <= 0.02 * size)
    co[near, axis] = offset
    on = np.abs(co[:, axis] - offset) == 0.0
    twin = np.where(on, np.arange(len(co)), 0)
    fresh = np.flatnonzero(~on)
    twin[fresh] = len(co) + np.arange(len(fresh))
    reflected = co[fresh].copy()
    reflected[:, axis] = 2.0 * offset - reflected[:, axis]
    out_co = np.vstack((co, reflected))
    out_faces = list(faces) + [tuple(int(twin[v]) for v in reversed(f)) for f in faces]
    out_mat = None if mat is None else np.concatenate((np.asarray(mat), np.asarray(mat)))
    return out_co, out_faces, out_mat, int(near.sum())
