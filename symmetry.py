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
    size: float = 0.0  # the mesh's diagonal, in its own units

    def ok(self, chosen: bool = False) -> bool:
        return self.match >= (ENOUGH_CHOSEN if chosen else ENOUGH)

    def text(self) -> str:
        return f"{AXES[self.axis]}: {self.match * 100:.0f}% of the surface matches its mirror (99% within {self.worst * 100:.1f}% of the size)"

    def differ(self) -> str:
        """How far apart the two sides are, in words: what Symmetrize evens out."""
        return f"the sides differ by up to {length_words(self.worst * self.size)} (99% of the surface)"


def length_words(metres: float) -> str:
    if metres < 0.001:
        return f"{metres * 1e6:.0f} um" if metres > 0 else "0 mm"
    return f"{metres * 1000:.1f} mm" if metres < 1.0 else f"{metres:.2f} m"


def _score(bvh, pts: np.ndarray, axis: int, offset: float, diag: float):
    mirrored = pts.copy()
    mirrored[:, axis] = 2.0 * offset - mirrored[:, axis]
    found = [bvh.find_nearest(p) for p in mirrored.tolist()]
    dist = np.asarray([f[3] if f[0] is not None else diag for f in found]) / diag
    near = np.asarray([f[0][axis] if f[0] is not None else 0.0 for f in found])
    return float((dist < MATCH_TOLERANCE).mean()), float(np.percentile(dist, 99)), dist, near, mirrored[:, axis]


def find_plane(co: np.ndarray, bvh, axes: str = "XYZ", offset: float | None = None) -> list[Plane]:
    """The best mirror plane on each asked axis, best first. Tries the mesh centre and the origin, then nudges the plane.
    With `offset` the plane stays there and is only measured."""
    lo, hi = co.min(axis=0), co.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    step = max(len(co) // SAMPLES, 1)
    pts = co[::step]
    planes = []
    for letter in axes.upper():
        axis = AXES.index(letter)
        if offset is not None:
            match, worst, *_rest = _score(bvh, pts, axis, offset, diag)
            planes.append(Plane(axis, offset, match, worst, diag))
            continue
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
        planes.append(Plane(axis, offset, match, worst, diag))
    return sorted(planes, key=lambda p: -p.match)


def cut_half(co: np.ndarray, faces, axis: int, offset: float, keep: int = 1):
    """Clip the faces to one side of the plane (keep=1: the side with larger coordinates).

    Returns (co, faces, source, origin): source[i] is the face that new face i was cut from, origin[v] the original
    vertex that vertex v is (or, for a vertex made on the plane, one end of the edge it cuts). Every new vertex on the
    plane is exactly on it, and two faces sharing a cut edge share its new vertex.
    """
    co = np.asarray(co, dtype=np.float64).copy()
    diag = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0))) or 1.0
    side = keep * (co[:, axis] - offset)
    flat = np.abs(side) <= SNAP * diag
    co[flat, axis] = offset  # almost on the plane: put exactly on it, so the cut has no slivers
    side[flat] = 0.0
    new_co = [co]
    ends = list(range(len(co)))
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
            ends.append(key[0])
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
    return all_co[used], [tuple(int(remap[v]) for v in f) for f in out_faces], np.asarray(source, dtype=np.int64), np.asarray(ends, dtype=np.int64)[used]


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


def mirror_result(co: np.ndarray, faces, mat, axis: int, offset: float, size: float, keep: int = 0):
    """The solved half plus its reflection. Returns (co, faces, mat, snapped, kept): `kept` are the faces of the half
    that are in the result (first as they are, then mirrored).

    The open border of the half that lies near the plane is moved exactly onto it (`snapped` counts those
    vertices, moved by at most a few percent of `size`); the reflected copy reuses them, so nothing is left open there.
    The seam is cleaned first: with `keep` (the side solved, +1 or -1) a vertex that strayed over the plane goes back,
    seam vertices closer than a tenth of an edge are merged, and faces left with no area or lying wholly in the plane
    go, so the two halves meet with no duplicate, overlapping or zero-area faces.
    """
    co = np.asarray(co, dtype=np.float64).copy()
    faces = [tuple(f) for f in faces]
    near = border_vertices(faces, len(co)) & (np.abs(co[:, axis] - offset) <= 0.02 * size)
    co[near, axis] = offset
    tiny = 1e-6 * size
    side = co[:, axis] - offset
    if keep:
        strayed = ~near & (keep * side < 0.0)
        co[strayed, axis] = offset - side[strayed]  # back over the plane: its mirror would overlap it
    loose = ~near & (np.abs(co[:, axis] - offset) < tiny)
    co[loose, axis] = offset + (keep or 1) * tiny  # on the plane but not on the seam: it must not weld to its mirror
    faces, mat, kept = _clean_seam(co, faces, mat, near)
    used = np.zeros(len(co), dtype=bool)
    for f in faces:
        used[list(f)] = True
    remap = np.cumsum(used) - 1
    co, near = co[used], near[used]
    faces = [tuple(int(remap[v]) for v in f) for f in faces]
    twin = np.where(near, np.arange(len(co)), 0)
    fresh = np.flatnonzero(~near)
    twin[fresh] = len(co) + np.arange(len(fresh))
    reflected = co[fresh].copy()
    reflected[:, axis] = 2.0 * offset - reflected[:, axis]
    out_co = np.vstack((co, reflected))
    out_faces = faces + [tuple(int(twin[v]) for v in reversed(f)) for f in faces]
    out_mat = None if mat is None else np.concatenate((np.asarray(mat), np.asarray(mat)))
    return out_co, out_faces, out_mat, int(near.sum()), kept


def _clean_seam(co: np.ndarray, faces: list, mat, on: np.ndarray):
    """Merge seam vertices joined by an edge under a tenth of the usual length, and drop the faces that leaves without
    area or that lie wholly on the seam (their mirror would be the same face). Returns (faces, mat, kept)."""
    step = max(len(faces) // 2000, 1)
    lengths = [float(np.linalg.norm(co[f[k]] - co[f[k - 1]])) for f in faces[::step] for k in range(len(f))]
    short = 0.1 * float(np.median(lengths)) if lengths else 0.0
    parent = np.arange(len(co))

    def root(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return int(v)

    for f in faces:
        for k in range(len(f)):
            a, b = f[k - 1], f[k]
            if on[a] and on[b] and np.linalg.norm(co[a] - co[b]) < short:
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    out, out_mat, kept = [], [], []
    for i, f in enumerate(faces):
        g = [root(v) for v in f]
        g = [v for k, v in enumerate(g) if v != g[k - 1]]
        if len(set(g)) < 3 or all(on[v] for v in g):
            continue
        out.append(tuple(g))
        kept.append(i)
        if mat is not None:
            out_mat.append(mat[i])
    return out, (None if mat is None else out_mat), np.asarray(kept, dtype=np.int64)


def seam_check(co: np.ndarray, faces, axis: int, offset: float, size: float) -> dict:
    """Issue counts along a mirror seam: open and non-manifold seam edges, faces crossing the plane, duplicate faces,
    and the Issue Finder's slivers, pinched and concave faces among those touching the seam."""
    try:
        from .triadq import issues
    except ImportError:  # imported on its own, as the tests do
        from triadq import issues
    co = np.asarray(co, dtype=np.float64)
    d = co[:, axis] - offset
    tol = 1e-7 * size
    on = np.abs(d) <= tol
    edges: dict[tuple[int, int], int] = {}
    for f in faces:
        for k in range(len(f)):
            a, b = f[k - 1], f[k]
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    seam_edges = [n for (a, b), n in edges.items() if on[a] and on[b]]
    touching = np.fromiter((bool(on[list(f)].any()) for f in faces), dtype=bool, count=len(faces))
    across = sum(1 for f in faces if (d[list(f)] > tol).any() and (d[list(f)] < -tol).any())
    flags = issues.topology_flags(co, faces)
    out = {"seam_vertices": int(on.sum()), "seam_faces": int(touching.sum()), "open": sum(1 for n in seam_edges if n == 1), "non_manifold": sum(1 for n in seam_edges if n > 2), "across": across, "duplicate": len(faces) - len({tuple(sorted(f)) for f in faces})}
    for kind in ("sliver", "pinched", "concave"):
        out[kind] = int((flags[kind] & touching).sum())
    return out


def seam_words(check: dict) -> str:
    """The seam check in one line for the panel."""
    bad = {k: check[k] for k in ("open", "non_manifold", "across", "duplicate") if check.get(k)}
    rough = sum(check.get(k, 0) for k in ("sliver", "pinched", "concave"))
    head = "Seam clean" if not bad else "Seam: " + ", ".join(f"{n} {k.replace('_', '-')}" for k, n in bad.items())
    return f"{head} ({check['seam_vertices']} vertices on the plane, {rough} thin or pinched faces along it)"
