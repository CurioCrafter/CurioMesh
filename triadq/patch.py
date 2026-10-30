# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Fixed-boundary region retopology: masks, seam loops and all-quad patch fills.

Pure NumPy (no bpy), so every step is unit-testable. A region is a boolean
face mask; each connected, disk-shaped component is replaced by an all-quad
patch that reuses the component's seam loop vertices exactly, so the patch
welds back by index with no T-junctions.

Pipeline per component:

1. Clean the mask (drop specks, fill pin holes), split it into components,
   grow pinches / enclosed holes until each is a disk, and fix seam parity
   (an all-quad disk needs an even number of seam edges).
2. Embed the disk in the plane (Tutte's barycentric mapping with positive
   1/length weights: bijective onto a convex polygon, Floater 1997).
3. Fill the planar domain:
   * Grid with four clean sides: a discrete Coons patch on a square domain.
   * Otherwise: a coarse triangulation over every second seam vertex
     (Flow: size-field refinement + Lawson flips + smoothing; Grid: an
     axis-aligned lattice), paired into quads by TRIAD-Q and closed to all
     quads with ParityPath, the skipped seam vertices serving as the frozen
     midpoints of the coarse seam edges.
4. Lift back to 3D by barycentric lookup in the embedded source triangles
   (regions) or by a bilaplacian membrane anchored on the surrounding ring
   (hole fills).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time

import numpy as np

from .engine import RemeshOptions, remesh_mesh
from .topology import _turbo, parity_path_quad_closure, quad_only_closure, quads_locally_valid
from .types import MeshData


# --------------------------------------------------------------------------- mesh tables


def _half_edges(faces) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(face id, from vertex, to vertex) for every polygon corner."""

    sizes = np.fromiter((len(face) for face in faces), dtype=np.int64, count=len(faces))
    total = int(sizes.sum())
    flat = np.fromiter((vertex for face in faces for vertex in face), dtype=np.int64, count=total)
    face_of = np.repeat(np.arange(len(faces), dtype=np.int64), sizes)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    following = np.arange(total, dtype=np.int64) + 1
    following[starts + sizes - 1] = starts
    return face_of, flat, flat[following] if total else flat


def face_adjacency(faces) -> tuple[np.ndarray, np.ndarray]:
    """Edge-adjacent face pairs (every face on an edge linked to its first face)."""

    face_of, a, b = _half_edges(faces)
    if not len(a):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    span = int(max(a.max(), b.max())) + 1
    key = np.minimum(a, b) * span + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    key, face_of = key[order], face_of[order]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    head = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    linked = ~first & (face_of != face_of[head])
    return face_of[head][linked], face_of[linked]


def _labels(count: int, fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Connected-component labels by hooking and pointer jumping."""

    labels = np.arange(count, dtype=np.int64)
    while True:
        la, lb = labels[fa], labels[fb]
        differ = la != lb
        if not differ.any():
            return labels
        np.minimum.at(labels, np.maximum(la, lb)[differ], np.minimum(la, lb)[differ])
        while True:
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped


def mask_components(faces, mask: np.ndarray, adjacency=None) -> list[np.ndarray]:
    """Edge-connected components of the masked faces, largest first."""

    mask = np.asarray(mask, dtype=bool)
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    fa, fb = adjacency if adjacency is not None else face_adjacency(faces)
    keep = mask[fa] & mask[fb]
    labels = _labels(len(mask), fa[keep], fb[keep])[ids]
    order = np.argsort(labels, kind="stable")
    _unique, starts = np.unique(labels[order], return_index=True)
    groups = np.split(ids[order], starts[1:])
    return sorted(groups, key=len, reverse=True)


def clean_mask(faces, mask: np.ndarray, min_island: int = 4, max_hole: int = 4, adjacency=None) -> np.ndarray:
    """Drop islands smaller than `min_island` faces and fill enclosed holes up to `max_hole` faces."""

    adjacency = adjacency if adjacency is not None else face_adjacency(faces)
    out = np.asarray(mask, dtype=bool).copy()
    for component in mask_components(faces, out, adjacency):
        if len(component) < min_island:
            out[component] = False
    fa, fb = adjacency
    touching = np.zeros(len(out), dtype=bool)
    touching[fa[out[fb]]] = True
    touching[fb[out[fa]]] = True
    holes = mask_components(faces, ~out, adjacency)
    for hole in holes[1:]:  # the largest unmasked area is the outside
        if len(hole) <= max_hole and touching[hole].any():
            out[hole] = True
    return out


def triangle_mates(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """Triangle pairs that share each other's longest edge (a split quad, as in a triangulated quad mesh)."""

    tri_ids = np.asarray([i for i, face in enumerate(faces) if len(face) == 3], dtype=np.int64)
    if not len(tri_ids):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    tris = np.asarray([faces[i] for i in tri_ids], dtype=np.int64)
    points = np.asarray(vertices, dtype=np.float64)[tris]
    longest = np.argmax(np.linalg.norm(np.roll(points, -1, axis=1) - points, axis=2), axis=1)
    a = tris[np.arange(len(tris)), longest]
    b = tris[np.arange(len(tris)), (longest + 1) % 3]
    key = np.minimum(a, b) * (int(tris.max()) + 1) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    return tri_ids[order[:-1][same]], tri_ids[order[1:][same]]


def smooth_mask(faces, mask: np.ndarray, rounds: int = 2, adjacency=None, mates=None) -> np.ndarray:
    """Straighten a jagged region edge: add notch faces, drop spike faces.

    Split-quad triangle `mates` join the region together first, so a
    triangulated quad mesh is cut along its quad edges. Then an n-sided face
    sharing >= n - 1 edges with the region on one side flips sides. Notches
    and spikes make reflex seam corners no quad layout meets without a
    concave quad.
    """

    fa, fb = adjacency if adjacency is not None else face_adjacency(faces)
    mask = np.asarray(mask, dtype=bool).copy()
    if mates is not None and len(mates[0]):
        joined = mask[mates[0]] | mask[mates[1]]
        mask[mates[0]] = joined
        mask[mates[1]] = joined
    sizes = np.fromiter((len(face) for face in faces), dtype=np.int64, count=len(faces))
    linked = np.bincount(np.concatenate((fa, fb)), minlength=len(mask))
    for _ in range(max(0, int(rounds))):
        inside = np.bincount(np.concatenate((fa[mask[fb]], fb[mask[fa]])), minlength=len(mask))
        outside = linked - inside
        notch = ~mask & (inside >= sizes - 1)
        spike = mask & (outside >= sizes - 1)
        if not (notch.any() or spike.any()):
            break
        mask = (mask | notch) & ~spike
    return mask


def combine_masks(count: int, entries) -> np.ndarray:
    """Evaluate (operation, mask) pairs in order: UNION, SUBTRACT or INTERSECT."""

    result = np.zeros(count, dtype=bool)
    for operation, mask in entries:
        mask = np.asarray(mask, dtype=bool)
        if operation == "SUBTRACT":
            result &= ~mask
        elif operation == "INTERSECT":
            result &= mask
        else:
            result |= mask
    return result


# --------------------------------------------------------------------------- volumes


def winding_number(points: np.ndarray, triangles: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Generalized winding number of a triangle soup (Jacobson et al. 2013).

    Each triangle's solid angle comes from Van Oosterom & Strackee:
    tan(omega / 2) = det[a b c] / (|a||b||c| + (a.b)|c| + (b.c)|a| + (c.a)|b|).
    A closed, consistently wound surface gives ~1 inside and ~0 outside;
    `abs(w) > 0.5` is a robust inside test even for holey or flipped cages.
    """

    fast = _turbo.winding_number(points, triangles) if _turbo is not None else None
    if fast is not None:
        return fast
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    total = np.zeros(len(points))
    if not len(points) or not len(triangles):
        return total
    step = max(1, chunk // len(triangles))
    for start in range(0, len(points), step):
        rel = triangles[None, :, :, :] - points[start : start + step, None, None, :]
        a, b, c = rel[:, :, 0], rel[:, :, 1], rel[:, :, 2]
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        det = np.einsum("ptj,ptj->pt", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("ptj,ptj->pt", a, b) * lc
            + np.einsum("ptj,ptj->pt", b, c) * la
            + np.einsum("ptj,ptj->pt", c, a) * lb
        )
        total[start : start + step] = np.arctan2(det, den).sum(axis=1) / (2.0 * math.pi)
    return total


def inside_volume(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Winding-number inside test with a bounding-box early out."""

    fast = _turbo.inside_volume(points, triangles) if _turbo is not None else None
    if fast is not None:
        return fast
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    inside = np.zeros(len(points), dtype=bool)
    if not len(triangles):
        return inside
    low, high = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    pad = 1e-9 * float(np.linalg.norm(high - low))
    near = np.all((points >= low - pad) & (points <= high + pad), axis=1)
    inside[near] = np.abs(winding_number(points[near], triangles)) > 0.5
    return inside


def inside_box(points: np.ndarray, world_to_local: np.ndarray, half_size: float = 1.0) -> np.ndarray:
    """Points whose cage-local coordinates satisfy |x|, |y|, |z| <= half_size."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = np.asarray(world_to_local, dtype=np.float64)
    local = points @ matrix[:3, :3].T + matrix[:3, 3]
    return np.all(np.abs(local) <= half_size, axis=1)


def lattice_hull(grid: np.ndarray) -> np.ndarray:
    """Closed, consistently oriented triangle hull of a (U, V, W, 3) lattice point grid."""

    grid = np.asarray(grid, dtype=np.float64)
    if min(grid.shape[:3]) < 2:
        return np.zeros((0, 3, 3))
    center = grid.reshape(-1, 3).mean(axis=0)
    sides = [grid[0], grid[-1], grid[:, 0], grid[:, -1], grid[:, :, 0], grid[:, :, -1]]
    triangles = []
    for side in sides:
        a, b = side[:-1, :-1].reshape(-1, 3), side[1:, :-1].reshape(-1, 3)
        c, d = side[1:, 1:].reshape(-1, 3), side[:-1, 1:].reshape(-1, 3)
        tris = np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]).sum(axis=0)
        if float(np.dot(normal, side.reshape(-1, 3).mean(axis=0) - center)) < 0.0:
            tris = tris[:, ::-1]
        triangles.append(tris)
    return np.concatenate(triangles)


# --------------------------------------------------------------------------- seam loops


def boundary_loops(faces, face_ids) -> tuple[list[list[int]], list[int]]:
    """Ordered seam loops of a face set, following the faces' winding.

    Returns (loops, pinched vertices). A pinched vertex has more than one
    outgoing seam half-edge, i.e. the face set touches itself there.
    """

    successor: dict[int, list[int]] = {}
    halves: set[tuple[int, int]] = set()
    for face_index in face_ids:
        face = faces[int(face_index)]
        for corner in range(len(face)):
            halves.add((face[corner], face[(corner + 1) % len(face)]))
    for a, b in halves:
        if (b, a) not in halves:
            successor.setdefault(a, []).append(b)
    pinched = sorted(vertex for vertex, nxt in successor.items() if len(nxt) > 1)
    loops: list[list[int]] = []
    if pinched:
        return loops, pinched
    remaining = set(successor)
    while remaining:
        start = min(remaining)
        loop = [start]
        remaining.discard(start)
        vertex = successor[start][0]
        while vertex != start and vertex in remaining:
            loop.append(vertex)
            remaining.discard(vertex)
            vertex = successor[vertex][0]
        loops.append(loop)
    return loops, pinched


def _euler(faces, face_ids) -> int:
    vertices: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for face_index in face_ids:
        face = faces[int(face_index)]
        vertices.update(face)
        for corner in range(len(face)):
            a, b = face[corner], face[(corner + 1) % len(face)]
            edges.add((a, b) if a < b else (b, a))
    return len(vertices) - len(edges) + len(face_ids)


class _Incidence:
    """Vertex -> faces and edge -> faces lookups for one polygon list."""

    def __init__(self, faces):
        self.faces = faces
        self.vertex_faces: dict[int, list[int]] = {}
        self.edge_faces: dict[tuple[int, int], list[int]] = {}
        for face_index, face in enumerate(faces):
            for corner, vertex in enumerate(face):
                self.vertex_faces.setdefault(vertex, []).append(face_index)
                following = face[(corner + 1) % len(face)]
                key = (vertex, following) if vertex < following else (following, vertex)
                self.edge_faces.setdefault(key, []).append(face_index)


def make_disk(faces, face_ids, incidence: _Incidence, claimed=frozenset(), max_steps: int = 32):
    """Grow a component until it is a topological disk. Returns (face ids, message)."""

    current = set(int(f) for f in face_ids)
    for _step in range(max_steps):
        loops, pinched = boundary_loops(faces, current)
        if pinched:
            grow = {f for v in pinched for f in incidence.vertex_faces.get(v, ())} - current
            if not grow or grow & set(claimed):
                return None, "region touches itself at a vertex"
            current |= grow
            continue
        if not loops:
            return None, "region covers a closed shell; shrink it"
        if len(loops) > 1:
            loops.sort(key=len, reverse=True)
            grown = False
            for loop in loops[1:]:
                hole = _enclosed_faces(loop, current, incidence, limit=len(current))
                if hole is None or hole & set(claimed):
                    continue
                current |= hole
                grown = True
            if not grown:
                return None, f"region has {len(loops)} seam loops (a ring or an open hole); make it a disk"
            continue
        if _euler(faces, current) != 1:
            return None, "region has handles; make it a disk"
        return np.asarray(sorted(current), dtype=np.int64), ""
    return None, "region could not be made a disk"


def _enclosed_faces(loop, current: set[int], incidence: _Incidence, limit: int):
    """Faces on the far side of a seam loop if they form a bounded pocket of <= `limit` faces."""

    seeds = set()
    for index, vertex in enumerate(loop):
        following = loop[(index + 1) % len(loop)]
        key = (vertex, following) if vertex < following else (following, vertex)
        seeds.update(f for f in incidence.edge_faces.get(key, ()) if f not in current)
    if not seeds:
        return None  # an open mesh hole: nothing to fill
    pocket = set(seeds)
    stack = list(seeds)
    while stack:
        face = incidence.faces[stack.pop()]
        for corner, vertex in enumerate(face):
            following = face[(corner + 1) % len(face)]
            key = (vertex, following) if vertex < following else (following, vertex)
            for other in incidence.edge_faces.get(key, ()):
                if other not in current and other not in pocket:
                    pocket.add(other)
                    stack.append(other)
                    if len(pocket) > limit:
                        return None
    return pocket


def _face_quality(vertices: np.ndarray, face) -> float:
    """Shape quality in (0, 1]: 4 sqrt(3) area / sum of squared edges for triangles, generalised by fan."""

    points = vertices[list(face)]
    sides = np.roll(points, -1, axis=0) - points
    area = 0.5 * np.linalg.norm(np.cross(points[1:-1] - points[0], points[2:] - points[0]).sum(axis=0))
    return float(4.0 * math.sqrt(3.0) * area / max(np.einsum("ij,ij->", sides, sides), 1e-30))


def fix_parity(vertices, faces, face_ids, incidence: _Incidence, claimed=frozenset(), max_tries: int = 32):
    """Make the seam edge count even by adding or removing one odd-sided face.

    Adding or removing a face with n sides that shares k seam edges changes the
    seam count by n - 2k, so only odd-sided faces flip parity. Candidates are
    tried best-shaped first; the first that keeps a single even seam loop on a
    disk wins. Returns (face ids, message).
    """

    current = set(int(f) for f in face_ids)
    loops, _pinched = boundary_loops(faces, current)
    if len(loops) == 1 and len(loops[0]) % 2 == 0:
        return np.asarray(sorted(current), dtype=np.int64), ""
    loop = loops[0]
    candidates: dict[int, bool] = {}
    for index, vertex in enumerate(loop):
        following = loop[(index + 1) % len(loop)]
        key = (vertex, following) if vertex < following else (following, vertex)
        for face_index in incidence.edge_faces.get(key, ()):
            if len(faces[face_index]) % 2 and face_index not in claimed:
                candidates[face_index] = face_index in current
    ranked = sorted(candidates, key=lambda f: -_face_quality(vertices, faces[f]))
    for face_index in ranked[:max_tries]:
        trial = current - {face_index} if candidates[face_index] else current | {face_index}
        if not trial:
            continue
        trial_loops, trial_pinched = boundary_loops(faces, trial)
        if (
            not trial_pinched
            and len(trial_loops) == 1
            and len(trial_loops[0]) % 2 == 0
            and _euler(faces, trial) == 1
        ):
            return np.asarray(sorted(trial), dtype=np.int64), ""
    return None, f"odd seam ({len(loop)} edges) and no odd-sided face nearby can fix it"


def _is_disk(faces, face_ids) -> bool:
    loops, pinched = boundary_loops(faces, face_ids)
    return not pinched and len(loops) == 1 and _euler(faces, face_ids) == 1


def _cut_in_two(vertices, faces, face_ids: np.ndarray, adjacency) -> list[np.ndarray]:
    """Split a face set with a plane through its middle, on the axis that leaves the fewest non-disk pieces."""

    centers = np.asarray([vertices[list(faces[int(f)])].mean(axis=0) for f in face_ids])
    centers -= centers.mean(axis=0)
    selected = np.zeros(len(faces), dtype=bool)
    best, best_score = [], None
    for axis in np.linalg.svd(centers, full_matrices=False)[2]:
        along = centers @ axis
        side = along < np.median(along)
        if side.all() or not side.any():
            continue
        pieces = []
        for part in (face_ids[side], face_ids[~side]):
            selected[:] = False
            selected[part] = True
            pieces += mask_components(faces, selected, adjacency)
        score = (sum(not _is_disk(faces, piece) for piece in pieces), len(pieces))
        if best_score is None or score < best_score:
            best, best_score = pieces, score
        if score == (0, 2):
            break
    return best


def _disk_pieces(vertices, faces, face_ids, incidence, adjacency, claimed: set[int], min_island: int, depth: int = 6):
    """Face set -> disks with even seams, cutting a ring, an open hole or a handle into pieces when needed.

    The pieces share the cut as seam, so the result stays joined. Returns (pieces, message of the first failure).
    """

    disk, message = make_disk(faces, face_ids, incidence, frozenset(claimed))
    if disk is not None:
        disk, message = fix_parity(vertices, faces, disk, incidence, frozenset(claimed))
    if disk is not None:
        return [disk], ""
    if depth == 0 or len(face_ids) < 2 * min_island:
        return [], message
    parts = _cut_in_two(vertices, faces, np.asarray(face_ids, dtype=np.int64), adjacency)
    done: list[np.ndarray] = []
    failed = "" if parts else message
    for k, part in enumerate(parts):
        others = set(claimed).union(*(p.tolist() for p in parts[k + 1 :]), *(d.tolist() for d in done))
        found, why = _disk_pieces(vertices, faces, part, incidence, adjacency, others, min_island, depth - 1)
        done += found
        failed = failed or why
    return done, failed


def prepare_regions(vertices, faces, mask, *, min_island: int = 4, max_hole: int = 4, smooth: int = 2, split_rings: bool = False, frozen=None):
    """Clean a face mask into disjoint disk components with even seams.

    `split_rings` cuts regions that are not disks (around a limb or an open hole) into pieces instead of skipping
    them. Faces in the `frozen` mask are never taken, however the mask is cleaned or grown.
    Returns (components, warnings): each component is a sorted face-id array.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    adjacency = face_adjacency(faces)
    mask = clean_mask(faces, mask, min_island, max_hole, adjacency)
    if smooth:
        mates = triangle_mates(vertices, faces)
        mask = clean_mask(faces, smooth_mask(faces, mask, smooth, adjacency, mates), min_island, max_hole, adjacency)
    claimed: set[int] = set()
    if frozen is not None:
        mask = mask & ~frozen
        claimed.update(np.flatnonzero(frozen).tolist())
    incidence = _Incidence(faces)
    components: list[np.ndarray] = []
    warnings: list[str] = []
    for index, component in enumerate(mask_components(faces, mask, adjacency)):
        others = claimed | (set(np.flatnonzero(mask).tolist()) - set(component.tolist()))
        if split_rings:
            pieces, message = _disk_pieces(vertices, faces, component, incidence, adjacency, others, min_island)
        else:
            disk, message = make_disk(faces, component, incidence, frozenset(others))
            if disk is not None:
                disk, message = fix_parity(vertices, faces, disk, incidence, frozenset(others))
            pieces = [] if disk is None else [disk]
        if not pieces:
            warnings.append(f"Region {index + 1} ({len(component)} faces) skipped: {message}")
        for piece in pieces:
            claimed.update(piece.tolist())
            components.append(piece)
    return components, warnings


# --------------------------------------------------------------------------- planar embedding


def _cg(matvec, rhs: np.ndarray, diagonal: np.ndarray, x0: np.ndarray, iterations: int = 4000, tol: float = 1e-11):
    """Jacobi-preconditioned conjugate gradients, one independent solve per column."""

    x = x0.copy()
    r = rhs - matvec(x)
    z = r / diagonal[:, None]
    p = z.copy()
    rz = np.einsum("ij,ij->j", r, z)
    scale = max(float(np.abs(rhs).max(initial=0.0)), 1e-30)
    for _ in range(iterations):
        if float(np.abs(r).max(initial=0.0)) <= tol * scale:
            break
        ap = matvec(p)
        alpha = rz / np.where(np.abs(np.einsum("ij,ij->j", p, ap)) > 1e-300, np.einsum("ij,ij->j", p, ap), 1.0)
        x += alpha * p
        r -= alpha * ap
        z = r / diagonal[:, None]
        rz_next = np.einsum("ij,ij->j", r, z)
        p = z + (rz_next / np.where(rz > 1e-300, rz, 1.0)) * p
        rz = rz_next
    return x


def _unique_edges(faces) -> np.ndarray:
    _face_of, a, b = _half_edges(faces)
    if not len(a):
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.sort(np.stack((a, b), axis=1), axis=1), axis=0)


def tutte_embedding(count: int, faces, boundary, boundary_uv, positions=None) -> np.ndarray:
    """Tutte barycentric embedding with the boundary pinned to `boundary_uv`.

    Each interior vertex is the weighted mean of its neighbours. With positive
    symmetric weights (1 / 3D edge length here) and a convex boundary polygon
    the map is a bijection (Tutte 1963, Floater 1997). Solved by CG.
    """

    edges = _unique_edges(faces)
    uv = np.zeros((count, 2))
    boundary = np.asarray(boundary, dtype=np.int64)
    uv[boundary] = boundary_uv
    fixed = np.zeros(count, dtype=bool)
    fixed[boundary] = True
    if fixed.all():
        return uv
    if positions is not None:
        lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
        weights = 1.0 / np.maximum(lengths, 1e-12 * max(float(lengths.max(initial=0.0)), 1e-30))
    else:
        weights = np.ones(len(edges))
    i, j = edges[:, 0], edges[:, 1]
    degree = np.bincount(i, weights, count) + np.bincount(j, weights, count)
    free = np.flatnonzero(~fixed)
    local = np.full(count, -1, dtype=np.int64)
    local[free] = np.arange(len(free))
    rhs = np.zeros((len(free), 2))
    for src, dst in ((i, j), (j, i)):
        link = ~fixed[src] & fixed[dst]
        for axis in range(2):
            rhs[:, axis] += np.bincount(local[src[link]], weights[link] * uv[dst[link], axis], len(free))
    inner = ~fixed[i] & ~fixed[j]
    ii, jj, ww = local[i[inner]], local[j[inner]], weights[inner]
    diagonal = np.maximum(degree[free], 1e-30)

    def matvec(x):
        out = diagonal[:, None] * x
        for axis in range(x.shape[1]):
            out[:, axis] -= np.bincount(ii, ww * x[jj, axis], len(free)) + np.bincount(jj, ww * x[ii, axis], len(free))
        return out

    x0 = np.tile(uv[boundary].mean(axis=0), (len(free), 1))
    uv[free] = _cg(matvec, rhs, diagonal, x0)
    return uv


def _arc_parameters(points: np.ndarray) -> np.ndarray:
    """Cumulative closed-polyline arc length in [0, 1) at each loop vertex."""

    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    total = max(float(lengths.sum()), 1e-30)
    return np.concatenate(([0.0], np.cumsum(lengths)[:-1])) / total


def circle_boundary(points: np.ndarray) -> np.ndarray:
    angle = 2.0 * math.pi * _arc_parameters(points)
    return np.stack((np.cos(angle), np.sin(angle)), axis=1)


def square_boundary(points: np.ndarray, corners) -> np.ndarray:
    """Loop vertices on the unit square, corners[k] at square corner k, sides by arc length."""

    count = len(points)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    uv = np.zeros((count, 2))
    for side in range(4):
        start, end = corners[side], corners[(side + 1) % 4]
        span = (end - start) % count or count
        idx = (start + np.arange(span + 1)) % count
        seg = np.linalg.norm(np.diff(points[idx], axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(seg))) / max(float(seg.sum()), 1e-30)
        uv[idx[:-1]] = square[side] + t[:-1, None] * (square[(side + 1) % 4] - square[side])
    return uv


def seam_turning(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Signed turning angle (radians) at each loop vertex in its tangent plane; > 0 turns towards the interior."""

    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    incoming = incoming - normals * np.einsum("ij,ij->i", incoming, normals)[:, None]
    outgoing = outgoing - normals * np.einsum("ij,ij->i", outgoing, normals)[:, None]
    sine = np.einsum("ij,ij->i", np.cross(incoming, outgoing), normals)
    return np.arctan2(sine, np.einsum("ij,ij->i", incoming, outgoing))


def skip_offset(turning: np.ndarray) -> int:
    """0 or 1: which alternate seam vertices become coarse-edge midpoints.

    A midpoint at a reflex (inward) corner leaves its quads concave in 3D,
    so the parity with the least reflex turning on its midpoints wins.
    """

    reflex = np.maximum(-np.asarray(turning), 0.0)
    return int(reflex[1::2].sum() > reflex[0::2].sum())


def four_corners(points: np.ndarray, normals: np.ndarray, corner_deg: float = 45.0, side_deg: float = 25.0):
    """Corner indices if the loop splits cleanly into 4 sides with equal opposite counts, else None.

    Turning is measured in the local tangent plane. Exactly four vertices must
    turn by >= `corner_deg`, every other vertex by <= `side_deg`.
    """

    turn = np.degrees(np.abs(seam_turning(points, normals)))
    corners = np.flatnonzero(turn >= corner_deg)
    if len(corners) != 4 or np.any(np.delete(turn, corners) > side_deg):
        return None
    count = len(points)
    sides = [(corners[(k + 1) % 4] - corners[k]) % count for k in range(4)]
    if sides[0] != sides[2] or sides[1] != sides[3] or min(sides) < 1:
        return None
    return [int(c) for c in corners]


class _Locator:
    """Point location in a planar triangulation via a uniform bucket grid."""

    def __init__(self, uv: np.ndarray, tris: np.ndarray):
        self.uv = np.asarray(uv, dtype=np.float64)
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        p = self.uv[tris]
        area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (
            p[:, 2, 0] - p[:, 0, 0]
        )
        span = float(np.ptp(self.uv, axis=0).max()) if len(self.uv) else 1.0
        self.ids = np.flatnonzero(np.abs(area) > 1e-14 * span * span)
        tris = tris[self.ids]
        self.tris = tris
        p = self.uv[tris]
        self.low = self.uv.min(axis=0) - 1e-9 * span
        cells = max(1, int(math.sqrt(max(len(tris), 1))))
        self.cells = cells
        self.size = (self.uv.max(axis=0) + 1e-9 * span - self.low) / cells
        lo = np.clip(((p.min(axis=1) - self.low) / self.size).astype(np.int64), 0, cells - 1)
        hi = np.clip(((p.max(axis=1) - self.low) / self.size).astype(np.int64), 0, cells - 1)
        wide, tall = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
        spans = wide * tall
        tri_ids = np.repeat(np.arange(len(tris)), spans)
        local = np.arange(len(tri_ids)) - np.repeat(np.cumsum(spans) - spans, spans)
        cell_ids = (lo[tri_ids, 0] + local // tall[tri_ids]) * cells + lo[tri_ids, 1] + local % tall[tri_ids]
        order = np.argsort(cell_ids, kind="stable")
        self.bucket = tri_ids[order]
        self.start = np.searchsorted(cell_ids[order], np.arange(cells * cells + 1))

    def _bary(self, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        p = self.uv[self.tris[t]]
        v0, v1, v2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], q - p[:, 0]
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        den = np.where(np.abs(den) > 1e-300, den, 1e-300)
        b1 = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / den
        b2 = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / den
        return np.stack((1.0 - b1 - b2, b1, b2), axis=1)

    def locate(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(source triangle index, clamped barycentric coordinates) for each query point.

        Every (query, bucket triangle) pair is scored at once; the triangle
        whose smallest barycentric coordinate is largest wins (containing
        triangles score >= 0).
        """

        q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        best_t = np.zeros(len(q), dtype=np.int64)
        best_b = np.full((len(q), 3), 1.0 / 3.0)
        best_score = np.full(len(q), -np.inf)
        if len(q) and len(self.tris):
            cell = np.clip(((q - self.low) / self.size).astype(np.int64), 0, self.cells - 1)
            cell = cell[:, 0] * self.cells + cell[:, 1]
            first, last = self.start[cell], self.start[cell + 1]
            counts = last - first
            rows = np.repeat(np.arange(len(q)), counts)
            if len(rows):
                offsets = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
                t = self.bucket[first[rows] + offsets]
                b = self._bary(q[rows], t)
                score = b.min(axis=1)
                # Per query, the pair with the highest score: sort by (row, score).
                order = np.lexsort((score, rows))
                last_of_row = order[np.cumsum(counts[counts > 0]) - 1]
                hit = rows[last_of_row]
                best_score[hit] = score[last_of_row]
                best_t[hit] = t[last_of_row]
                best_b[hit] = b[last_of_row]
        missing = ~np.isfinite(best_score)
        if missing.any() and len(self.tris):  # outside every bucket candidate: nearest centroid
            centroids = self.uv[self.tris].mean(axis=1)
            for row in np.flatnonzero(missing):
                t = int(np.argmin(np.einsum("ij,ij->i", centroids - q[row], centroids - q[row])))
                best_t[row] = t
                best_b[row] = self._bary(q[row : row + 1], np.array([t]))[0]
        best_b = np.clip(best_b, 0.0, None)
        best_b /= np.maximum(best_b.sum(axis=1, keepdims=True), 1e-30)
        return self.ids[best_t], best_b


# --------------------------------------------------------------------------- planar triangle mesh


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _incircle(a, b, c, d) -> float:
    """> 0 when d lies inside the circumcircle of the CCW triangle abc."""

    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    return (
        (ax * ax + ay * ay) * (bx * cy - cx * by)
        - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay)
    )


def _key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


class _Planar:
    """Small mutable CCW triangle mesh in the plane (split, insert, flip, smooth)."""

    def __init__(self, points, tris, fixed: int):
        self.p = [(float(x), float(y)) for x, y in points]
        self.t: list[list[int]] = []
        self.edges: dict[tuple[int, int], list[int]] = {}
        self.fixed = fixed  # vertices [0, fixed) never move
        for tri in tris:
            self._add(list(tri))

    def _link(self, index: int) -> None:
        tri = self.t[index]
        for k in range(3):
            self.edges.setdefault(_key(tri[k], tri[(k + 1) % 3]), []).append(index)

    def _unlink(self, index: int) -> None:
        tri = self.t[index]
        for k in range(3):
            key = _key(tri[k], tri[(k + 1) % 3])
            users = self.edges[key]
            users.remove(index)
            if not users:
                del self.edges[key]

    def _add(self, tri) -> int:
        self.t.append(list(tri))
        self._link(len(self.t) - 1)
        return len(self.t) - 1

    def _set(self, index: int, tri) -> None:
        self._unlink(index)
        self.t[index] = list(tri)
        self._link(index)

    def _directed(self, index: int, a: int, b: int) -> tuple[int, int, int]:
        """(x, y, opposite) with x -> y the edge {a, b} as it runs in triangle `index`."""

        tri = self.t[index]
        for k in range(3):
            if {tri[k], tri[(k + 1) % 3]} == {a, b}:
                return tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        raise KeyError((a, b))

    def split_edge(self, a: int, b: int, point) -> int:
        middle = len(self.p)
        self.p.append((float(point[0]), float(point[1])))
        for index in list(self.edges[_key(a, b)]):
            x, y, o = self._directed(index, a, b)
            self._set(index, (x, middle, o))
            self._add((middle, y, o))
        return middle

    def insert(self, index: int, point) -> list[int]:
        middle = len(self.p)
        self.p.append((float(point[0]), float(point[1])))
        a, b, c = self.t[index]
        self._set(index, (a, b, middle))
        return [index, self._add((b, c, middle)), self._add((c, a, middle))]

    def walk(self, q, start: int = 0) -> int:
        index = start
        for _ in range(len(self.t) + 8):
            tri = self.t[index]
            moved = False
            for k in range(3):
                x, y = tri[k], tri[(k + 1) % 3]
                if _orient(self.p[x], self.p[y], q) < 0.0:
                    users = self.edges[_key(x, y)]
                    if len(users) == 2:
                        index = users[0] if users[1] == index else users[1]
                        moved = True
                        break
            if not moved:
                return index
        return index

    def flips(self, stack=None, budget: int = 200_000) -> int:
        """Lawson flips towards the Delaunay triangulation."""

        stack = list(self.edges) if stack is None else list(stack)
        flipped = 0
        while stack and budget > 0:
            budget -= 1
            key = stack.pop()
            users = self.edges.get(key)
            if not users or len(users) != 2:
                continue
            a, b, c = self._directed(users[0], *key)
            _b, _a, d = self._directed(users[1], *key)
            pa, pb, pc, pd = self.p[a], self.p[b], self.p[c], self.p[d]
            scale = max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1]), 1e-30) ** 4
            if _incircle(pa, pb, pc, pd) <= 1e-12 * scale:
                continue
            # Sine test: a nearly straight corner would flip into a zero-area sliver along it.
            if _orient(pa, pd, pc) <= 1e-5 * math.dist(pa, pd) * math.dist(pd, pc):
                continue
            if _orient(pd, pb, pc) <= 1e-5 * math.dist(pd, pb) * math.dist(pd, pc):
                continue
            first, second = users
            self._set(first, (a, d, c))
            self._set(second, (d, b, c))
            flipped += 1
            stack.extend((_key(a, c), _key(c, b), _key(b, d), _key(d, a)))
        return flipped

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.p, dtype=np.float64), np.asarray(self.t, dtype=np.int64).reshape(-1, 3)

    def refine(self, size_fn, max_points: int, rounds: int = 48) -> None:
        """Split interior edges longer than 4/3 of the local size, longest first."""

        for _ in range(rounds):
            keys = [key for key, users in self.edges.items() if len(users) == 2]
            if not keys or len(self.p) >= max_points:
                return
            ends = np.asarray(keys, dtype=np.int64)
            points = np.asarray(self.p)
            middle = 0.5 * (points[ends[:, 0]] + points[ends[:, 1]])
            ratio = np.linalg.norm(points[ends[:, 0]] - points[ends[:, 1]], axis=1) / np.maximum(size_fn(middle), 1e-12)
            order = np.argsort(-ratio)
            order = order[ratio[order] > 4.0 / 3.0]
            if not len(order):
                return
            touched: set[int] = set()
            for row in order.tolist():
                key = keys[row]
                users = self.edges.get(key)
                if not users or len(users) != 2 or touched.intersection(users):
                    continue
                touched.update(users)
                before = len(self.t)
                self.split_edge(key[0], key[1], middle[row])
                touched.update(range(before, len(self.t)))
                if len(self.p) >= max_points:
                    break
            self.flips()

    def smooth(self, iterations: int = 4) -> None:
        """Laplacian smoothing of free vertices; moves that invert a triangle are undone."""

        points, tris = self.arrays()
        ends = np.asarray(list(self.edges), dtype=np.int64).reshape(-1, 2)
        count = len(points)
        degree = np.bincount(ends.ravel(), minlength=count).astype(np.float64)
        free = np.arange(count) >= self.fixed
        for _ in range(iterations):
            total = np.zeros_like(points)
            for axis in range(2):
                total[:, axis] = np.bincount(ends[:, 0], points[ends[:, 1], axis], count) + np.bincount(
                    ends[:, 1], points[ends[:, 0], axis], count
                )
            moved = np.where(free[:, None] & (degree[:, None] > 0), total / np.maximum(degree, 1)[:, None], points)
            for _attempt in range(4):
                p = moved[tris]
                area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (
                    p[:, 2, 0] - p[:, 0, 0]
                )
                bad = area <= 0.0
                if not bad.any():
                    break
                undo = np.unique(tris[bad])
                moved[undo] = points[undo]
            points = moved
        self.p = [tuple(row) for row in points.tolist()]


# --------------------------------------------------------------------------- quad fill core


@dataclass
class PatchResult:
    """One patch: delete `removed_faces`, append `points`, add `faces`.

    Face indices below `base` are existing vertices (the seam); index
    `base + k` is `points[k]`.
    """

    ok: bool
    message: str = ""
    base: int = 0
    removed_faces: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    seam: list[int] = field(default_factory=list)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    faces: list[tuple[int, ...]] = field(default_factory=list)
    face_source: list[int] = field(default_factory=list)
    face_uvs: list[tuple[tuple[float, float], ...]] | None = None
    strategy: str = ""
    closure: str = ""
    elapsed_ms: float = 0.0

    @property
    def quads(self) -> int:
        return sum(1 for face in self.faces if len(face) == 4)


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _planar_quads_valid(uv: np.ndarray, quads: np.ndarray) -> np.ndarray:
    lifted = np.concatenate((uv, np.zeros((len(uv), 1))), axis=1)
    return quads_locally_valid(lifted, quads)


def _smooth_quads(uv: np.ndarray, faces: list[tuple[int, ...]], fixed: np.ndarray, iterations: int, lift=None) -> np.ndarray:
    """Planar Laplacian relax of a quad mesh.

    Moves that fold a quad in the plane, or leave its lifted quality worse and
    poor (`quad_quality` < 0.2), are undone until none remain.
    """

    if not faces or iterations <= 0:
        return uv
    quads = np.asarray(faces, dtype=np.int64)
    ends = _unique_edges(faces)
    count = len(uv)
    degree = np.bincount(ends.ravel(), minlength=count).astype(np.float64)
    free = ~fixed & (degree > 0)
    for _ in range(iterations):
        total = np.zeros_like(uv)
        for axis in range(2):
            total[:, axis] = np.bincount(ends[:, 0], uv[ends[:, 1], axis], count) + np.bincount(
                ends[:, 1], uv[ends[:, 0], axis], count
            )
        moved = np.where(free[:, None], 0.5 * uv + 0.5 * total / np.maximum(degree, 1)[:, None], uv)
        simple = _planar_simple(uv[quads])
        quality = quad_quality(lift(uv)[quads]) if lift is not None else None
        for _attempt in range(64):  # reverting only returns towards `uv`, so this ends
            bad = simple & ~_planar_simple(moved[quads])
            if lift is not None:
                after = quad_quality(lift(moved)[quads])
                bad |= (after < quality - 1e-9) & (after < 0.2)
            if not bad.any():
                break
            undo = np.unique(quads[bad])
            moved[undo] = uv[undo]
        uv = moved
    return uv


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.sqrt(np.einsum("...j,...j->...", v, v))
    return v / np.where(length > 1e-30, length, 1.0)[..., None]


def quad_quality(points: np.ndarray) -> np.ndarray:
    """Per-quad quality for (..., 4, 3) corners: > 0 means convex and warped < 60 degrees.

    min(minimum scaled Jacobian about the mean normal, (cos warp - 0.5) / 0.5),
    the warp being the larger fold of the two diagonal splits. This mirrors
    `quads_locally_valid`, as a smooth-ish number an optimiser can climb.
    """

    p0, p1, p2, p3 = (points[..., k, :] for k in range(4))
    normal = _unit(_cross3(p2 - p0, p3 - p1))  # the quad's vector area direction
    sides = points[..., [1, 2, 3, 0], :] - points  # side k runs corner k -> k + 1
    following, previous = sides, -sides[..., [3, 0, 1, 2], :]
    square = np.einsum("...j,...j->...", sides, sides)
    lengths = np.sqrt(square * square[..., [3, 0, 1, 2]])
    jac = np.einsum("...kj,...j->...k", _cross3(following, previous), normal) / np.where(lengths > 1e-30, lengths, 1.0)
    jac = np.where(lengths > 1e-30, jac, -1.0).min(axis=-1)
    fold_a = np.einsum("...j,...j->...", _unit(_cross3(p1 - p0, p2 - p0)), _unit(_cross3(p2 - p0, p3 - p0)))
    fold_b = np.einsum("...j,...j->...", _unit(_cross3(p2 - p1, p3 - p1)), _unit(_cross3(p3 - p1, p0 - p1)))
    return np.minimum(jac, (np.minimum(fold_a, fold_b) - 0.5) * 2.0)


def _planar_simple(points: np.ndarray) -> np.ndarray:
    """(..., 4, 2) quads that are simple and CCW: one diagonal splits them into two CCW triangles."""

    def tri(a, b, c):
        u, w = points[..., b, :] - points[..., a, :], points[..., c, :] - points[..., a, :]
        return u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]

    return ((tri(0, 1, 2) > 0) & (tri(0, 2, 3) > 0)) | ((tri(1, 2, 3) > 0) & (tri(1, 3, 0) > 0))


def _quad_colouring(count: int, quads: np.ndarray) -> np.ndarray:
    """Greedy vertex colouring where corners of one quad never share a colour (~5-7 colours)."""

    mates: list[set[int]] = [set() for _ in range(count)]
    for quad in quads.tolist():
        for v in quad:
            mates[v].update(quad)
    colour = np.full(count, -1, dtype=np.int64)
    for v in range(count):
        used = {int(colour[u]) for u in mates[v]}
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    return colour


def _untangle(
    coords: np.ndarray, faces, fixed: np.ndarray, to3d, planar: bool, rounds: int = 12, threshold: float = 0.25
) -> np.ndarray:
    """Vectorised quality optimisation of free quad corners, seam (`fixed`) untouched.

    Each round takes the free corners of quads with `quad_quality` below
    `threshold`, processes them one colour class at a time (no two on one
    quad, so moves never interact) and, per class, scores ~20 candidate
    positions per vertex at once: the edge-ring mean, the mean parallelogram
    completion a + b - o of its quads, halfway blends, and eight short steps.
    The winner maximises the incident quads' minimum quality (ties broken by
    the mean), keeps them simple and CCW in the plane when `planar`, and must
    beat the current position. `to3d` lifts (N, 2|3) coordinates to 3D point-wise.
    """

    quads = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    if not len(quads):
        return coords
    coords = np.asarray(coords, dtype=np.float64).copy()
    count = len(coords)
    dim = coords.shape[1]
    inc_vertex = quads.ravel()
    order = np.argsort(inc_vertex, kind="stable")
    inc_vertex = inc_vertex[order]
    inc_quad, inc_corner = np.divmod(order, 4)
    starts = np.searchsorted(inc_vertex, np.arange(count + 1))
    if dim == 2:
        angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        steps = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    else:
        steps = np.concatenate((np.eye(dim), -np.eye(dim)))
    colour = _quad_colouring(count, quads)
    xyz = to3d(coords)

    deficit = np.inf
    for _round in range(rounds):
        quality = quad_quality(xyz[quads])
        previous, deficit = deficit, float((threshold - np.minimum(quality, threshold)).sum())
        if deficit > 0.98 * previous:  # under 2% progress in the last round: converged
            break
        poor = quads[quality < threshold]
        active = np.unique(poor)
        active = active[~fixed[active]]
        if not len(active):
            break
        moved_any = False
        for c in range(int(colour[active].max()) + 1):
            batch = active[colour[active] == c]
            if not len(batch):
                continue
            lo, hi = starts[batch], starts[batch + 1]
            owner = np.repeat(np.arange(len(batch)), hi - lo)
            sel = lo[owner] + np.arange(len(owner)) - np.repeat(np.cumsum(hi - lo) - (hi - lo), hi - lo)
            q_of, c_of = inc_quad[sel], inc_corner[sel]
            corners = quads[q_of]
            a = coords[corners[np.arange(len(sel)), (c_of + 1) % 4]]
            o = coords[corners[np.arange(len(sel)), (c_of + 2) % 4]]
            b = coords[corners[np.arange(len(sel)), (c_of + 3) % 4]]
            valence = (hi - lo).astype(np.float64)[:, None]
            parallelogram = np.zeros((len(batch), dim))
            ring = np.zeros((len(batch), dim))
            for axis in range(dim):
                parallelogram[:, axis] = np.bincount(owner, (a + b - o)[:, axis], len(batch))
                ring[:, axis] = np.bincount(owner, 0.5 * (a + b)[:, axis], len(batch))
            parallelogram /= valence
            ring /= valence
            current = coords[batch]
            reach = np.sqrt(np.bincount(owner, np.einsum("ij,ij->i", a - current[owner], a - current[owner]), len(batch)) / valence[:, 0])
            options = [current, ring, parallelogram, 0.5 * (current + ring), 0.5 * (current + parallelogram)]
            for scale in (0.15, 0.45):
                for step in steps:
                    options.append(current + scale * reach[:, None] * step)
            candidates = np.stack(options, axis=1)  # (batch, K, dim)
            kinds = candidates.shape[1]
            lifted = to3d(candidates.reshape(-1, dim)).reshape(len(batch), kinds, 3)
            base3 = np.broadcast_to(xyz[corners][:, None], (len(sel), kinds, 4, 3)).copy()
            rows, ks = np.arange(len(sel))[:, None], np.arange(kinds)[None, :]
            base3[rows, ks, c_of[:, None]] = lifted[owner]
            score = quad_quality(base3)
            if planar:
                base2 = np.broadcast_to(coords[corners][:, None], (len(sel), kinds, 4, dim)).copy()
                base2[rows, ks, c_of[:, None]] = candidates[owner]
                score = np.where(_planar_simple(base2[..., :2]), score, -10.0)
            group = np.concatenate(([0], np.cumsum(hi - lo)[:-1]))
            worst = np.minimum.reduceat(score, group, axis=0)
            mean = np.add.reduceat(score, group, axis=0) / valence
            inverted = np.add.reduceat((score <= 0.0).astype(np.float64), group, axis=0)
            value = worst + 0.05 * mean - inverted  # each concave or folded quad costs a full point
            pick = np.argmax(value, axis=1)
            better = value[np.arange(len(batch)), pick] > value[:, 0] + 1e-3
            if better.any():
                chosen = batch[better]
                xyz[chosen] = lifted[better, pick[better]]
                coords[chosen] = xyz[chosen] if dim == 3 else candidates[better, pick[better]]
                moved_any = True
        if not moved_any:
            break
    return coords


def coons_grid(loop_values: np.ndarray, corners) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Coons patch (bilinearly blended transfinite interpolation).

    `loop_values` (B, d) runs CCW; corners c0..c3 bound sides of n1, n2, n1, n2
    edges. Returns (nodes (n1+1, n2+1, d), loop index per node or -1 inside).
    """

    values = np.asarray(loop_values, dtype=np.float64)
    count = len(values)
    c0, c1, c2, c3 = corners
    n1, n2 = (c1 - c0) % count, (c2 - c1) % count
    i = np.arange(n1 + 1)
    j = np.arange(n2 + 1)
    bottom = (c0 + i) % count
    top = (c3 - i) % count
    left = (c0 - j) % count
    right = (c1 + j) % count
    u = (i / n1)[:, None, None]
    v = (j / n2)[None, :, None]
    p00, p10, p11, p01 = values[c0], values[c1], values[c2], values[c3]
    nodes = (
        (1 - v) * values[bottom][:, None, :]
        + v * values[top][:, None, :]
        + (1 - u) * values[left][None, :, :]
        + u * values[right][None, :, :]
        - ((1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + (1 - u) * v * p01 + u * v * p11)
    )
    ids = np.full((n1 + 1, n2 + 1), -1, dtype=np.int64)
    ids[:, 0] = bottom
    ids[:, -1] = top
    ids[0, :] = left
    ids[-1, :] = right
    return nodes, ids


def _coons_fill(loop_uv: np.ndarray, corners) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Coons grid in the planar domain: (uv for loop + new vertices, quads)."""

    count = len(loop_uv)
    nodes, ids = coons_grid(loop_uv, corners)
    inner = ids < 0
    ids[inner] = count + np.arange(int(inner.sum()))
    uv = np.concatenate((loop_uv, nodes[inner]))
    faces = [
        (int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1]))
        for i in range(ids.shape[0] - 1)
        for j in range(ids.shape[1] - 1)
    ]
    return uv, faces


def _side_polylines(loop_uv: np.ndarray, corners) -> list[np.ndarray]:
    count = len(loop_uv)
    sides = []
    for k in range(4):
        start, end = corners[k], corners[(k + 1) % 4]
        span = (end - start) % count or count
        sides.append(loop_uv[(start + np.arange(span + 1)) % count])
    return sides


def _coons_counts(loop_xyz: np.ndarray, corners, target: float):
    """Iso-line counts (m1, m2) for a 3D target edge length when they differ from the seam's, else None.

    The seam fixes n1 x n2 cells. The target wants about (opposite-side mean
    3D length) / target; a mismatch over 25% (and at least one cell) switches
    the Coons fill to resampled iso-lines (`_coons_points`).
    """

    if not target or target <= 0.0:
        return None
    sides = _side_polylines(loop_xyz, corners)
    counts, differs = [], False
    for first, second in ((0, 2), (1, 3)):
        cells = len(sides[first]) - 1
        length = 0.5 * sum(float(np.linalg.norm(np.diff(sides[k], axis=0), axis=1).sum()) for k in (first, second))
        wanted = max(1, round(length / target))
        counts.append(wanted)
        differs |= abs(wanted - cells) > max(1.0, 0.25 * cells)
    return tuple(counts) if differs else None


def _coons_points(loop_uv: np.ndarray, loop_xyz: np.ndarray, corners, counts) -> tuple[np.ndarray, float]:
    """Interior nodes of a Coons patch whose sides are resampled to `counts` edges; also the smallest uv spacing.

    Each side is resampled at equal 3D arc length (uv interpolated), so
    iso-lines cross the domain at the target density; the seam keeps its own
    counts and the planar mesher grades between the two.
    """

    sides_uv = _side_polylines(loop_uv, corners)
    sides_xyz = _side_polylines(loop_xyz, corners)
    loop = []
    for k in range(4):
        seg = np.linalg.norm(np.diff(sides_xyz[k], axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(seg))) / max(float(seg.sum()), 1e-30)
        samples = np.linspace(0.0, 1.0, counts[k % 2] + 1)[:-1]
        loop.append(np.stack([np.interp(samples, t, sides_uv[k][:, axis]) for axis in range(2)], axis=1))
    loop = np.concatenate(loop)
    n1, n2 = counts
    nodes, _ids = coons_grid(loop, (0, n1, n1 + n2, 2 * n1 + n2))
    spacing = float(np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1).min())
    return nodes[1:-1, 1:-1].reshape(-1, 2), spacing


def _coarse_planar(loop_uv: np.ndarray, size_fn, strategy: str, max_points: int, points=None) -> _Planar:
    """Triangulate the planar domain over every second seam vertex.

    GRID inserts an axis-aligned lattice at the interior target size, or the
    given interior `points` ((nodes, uv spacing): resampled Coons iso-lines) instead.
    """

    coarse = loop_uv[0::2]
    center = coarse.mean(axis=0)
    count = len(coarse)
    planar = _Planar(np.vstack((coarse, center)), [(k, (k + 1) % count, count) for k in range(count)], fixed=count)
    planar.flips()
    if strategy == "GRID":
        step = float(np.median(size_fn(coarse * 0.5 + center * 0.5)))
        step = max(step, math.sqrt(_polygon_area(coarse) / max(max_points, 1)))
        low, high = coarse.min(axis=0), coarse.max(axis=0)
        xs = np.arange(math.floor((low[0] - center[0]) / step), math.ceil((high[0] - center[0]) / step) + 1)
        ys = np.arange(math.floor((low[1] - center[1]) / step), math.ceil((high[1] - center[1]) / step) + 1)
        grid = center + step * np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        if points is not None:
            grid, step = points
        keep = _inside_polygon(grid, coarse) & (_polygon_distance(grid, coarse) > 0.6 * step)
        keep &= np.linalg.norm(grid - center, axis=1) > 1e-9 * step
        last = 0
        for q in grid[keep]:
            last = planar.walk(q, min(last, len(planar.t) - 1))
            fresh = planar.insert(last, q)
            planar.flips([_key(*sorted(e)) for t in fresh for e in _tri_edges(planar.t[t])])
    else:
        planar.refine(size_fn, max_points)
        planar.smooth(6)
        planar.flips()
    return planar


def _tri_edges(tri):
    return ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _inside_polygon(q: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(q), dtype=bool)
    a, b = polygon, np.roll(polygon, -1, axis=0)
    for (ax, ay), (bx, by) in zip(a, b):
        crosses = (ay > q[:, 1]) != (by > q[:, 1])
        x_at = ax + (q[:, 1] - ay) * (bx - ax) / np.where(by != ay, by - ay, 1e-300)
        inside ^= crosses & (q[:, 0] < x_at)
    return inside


def _polygon_distance(q: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("qkj,kj->qk", q[:, None, :] - a[None], ab) / np.maximum(np.einsum("kj,kj->k", ab, ab), 1e-300), 0, 1)
    nearest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(q[:, None, :] - nearest, axis=2).min(axis=1)


def _pair_and_close(
    uv: np.ndarray, xyz: np.ndarray, tris: np.ndarray, loop_count: int, seed_count: int, parity: bool,
    field_mode: str = "SMOOTH",
):
    """TRIAD-Q pairing in 3D, then fixed-seam closure in the planar domain.

    Coarse seam edge (2k, 2k + 2) is closed with seam vertex 2k + 1 as its
    frozen midpoint. ParityPath templates must be valid in the plane and in
    3D; `parity=False` splits every face (CornerSplit). Returns (uv of all
    vertices, quads, closure label); raises ValueError if ParityPath fails.
    """

    paired, _report = remesh_mesh(
        MeshData(vertices=xyz, faces=[tuple(t) for t in tris.tolist()], name="region_patch"),
        RemeshOptions(
            target_faces=max(1, len(tris) // 2), seed_count=seed_count, force_quads=False, field_mode=field_mode
        ),
    )
    faces: list[tuple[int, ...]] = []
    quads = [face for face in paired.faces if len(face) == 4]
    valid = _planar_quads_valid(uv, np.asarray(quads, dtype=np.int64).reshape(-1, 4)) if quads else []
    for face, ok in zip(quads, valid):
        if ok:
            faces.append(tuple(face))
        else:  # the pair's shared diagonal runs corner 1 -> 3
            faces.extend(((face[0], face[1], face[3]), (face[1], face[2], face[3])))
    faces.extend(tuple(face) for face in paired.faces if len(face) == 3)
    frozen = {_key(2 * k, (2 * k + 2) % loop_count): 2 * k + 1 for k in range(loop_count // 2)}
    planar = MeshData(vertices=np.concatenate((uv, np.zeros((len(uv), 1))), axis=1), faces=faces)
    if parity:
        try:
            closed = parity_path_quad_closure(planar, frozen_midpoints=frozen, validation_vertices=xyz)
        except ValueError:
            # Staircase seams: no template is valid in 3D at some seam cell. Take the
            # planar-valid one; the Jacobian pass (_untangle) repairs it in 3D.
            closed = parity_path_quad_closure(
                planar, frozen_midpoints=frozen, validation_vertices=xyz, validation_required=False
            )
        label = "ParityPath"
    else:
        closed = quad_only_closure(planar, frozen_midpoints=frozen)
        label = "CornerSplit"
    return closed.vertices[:, :2].copy(), [tuple(f) for f in closed.faces], label


def _fill_core(loop_uv, lift, size_fn, strategy, corners, seed_count, max_points, target=0.0, field_mode="SMOOTH"):
    """Unoptimised all-quad fill of the planar loop: (uv, quads, label); vertex k < B is loop vertex k."""

    count = len(loop_uv)
    if count == 4:
        return loop_uv.copy(), [(0, 1, 2, 3)], "Single"
    counts = loop_xyz = None
    if corners is not None:
        uv, faces = _coons_fill(loop_uv, corners)
        loop_xyz = lift(loop_uv, [])
        counts = _coons_counts(loop_xyz, corners, target)
        if counts is None and _planar_quads_valid(uv, np.asarray(faces)).all():
            return uv, faces, "Coons"
    for parity, factor in ((True, 1.0), (False, 2.0)):
        points = None
        if counts is not None:  # iso-lines at the target density (half as many for corner-splitting)
            points = _coons_points(loop_uv, loop_xyz, corners, [max(1, round(c / factor)) for c in counts])
            strategy = "GRID"
        planar = _coarse_planar(loop_uv, lambda q: size_fn(q, factor), strategy, max_points, points)
        points, tris = planar.arrays()
        coarse = count // 2
        # Planar vertex i < coarse is seam vertex 2i; later ones are new.
        remap = np.concatenate((2 * np.arange(coarse), count + np.arange(len(points) - coarse)))
        uv = np.concatenate((loop_uv, points[coarse:]))
        tris = remap[tris]
        xyz = lift(uv, [tuple(t) for t in tris.tolist()])
        try:
            return _pair_and_close(uv, xyz, tris, count, max(1, seed_count), parity, field_mode)
        except ValueError:
            continue
    raise ValueError("no closure could make the patch all quads")


def build_seam_layer(loop_uv: np.ndarray, angles=None, end_angle: float = 115.0):
    """Topology and start positions of a seam-conforming boundary layer.

    Returns (inner loop uv (M, 2), layer quads, M). Layer vertex j is index
    B + j. A side vertex k gets one rung to inner vertex o_k and quads
    (k, k + 1, o_k+1, o_k), so it meets two quads. A sharp convex seam corner
    (interior 3D angle < `end_angle` degrees, from `angles`) is an end: no
    rung, one quad (k - 1, k, k + 1, o) with o shared by both neighbours, so
    M = B - 2 * ends (parity holds). Ends are never adjacent. Inner points
    start on the loop scaled towards its centroid by ~one edge; the optimiser
    places them on the surface.
    """

    count = len(loop_uv)
    center = loop_uv.mean(axis=0)
    radius = float(np.linalg.norm(loop_uv - center, axis=1).mean())
    edge = float(np.linalg.norm(np.roll(loop_uv, -1, axis=0) - loop_uv, axis=1).mean())
    scaled = center + (loop_uv - center) * max(0.0, 1.0 - 0.8 * edge / max(radius, 1e-30))
    ends = np.zeros(count, dtype=bool)
    if angles is not None:
        angles = np.asarray(angles, dtype=np.float64)
        for k in np.argsort(angles).tolist():
            if angles[k] >= end_angle or count - 2 * (int(ends.sum()) + 1) < 4:
                break
            if not (ends[k - 1] or ends[(k + 1) % count]):
                ends[k] = True
    parent = list(range(count))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for k in np.flatnonzero(ends).tolist():
        parent[root((k + 1) % count)] = root(k - 1)
    order, members = [], {}
    for k in range(count):
        group = root(k - 1) if ends[k] else root(k)
        members.setdefault(group, []).append(k)
        if not ends[k] and (not order or order[-1] != group):
            order.append(group)
    if len(order) > 1 and order[0] == order[-1]:
        order.pop()
    slot = {group: j for j, group in enumerate(order)}
    inner = np.asarray([scaled[members[group]].mean(axis=0) for group in order])
    quads = []
    for k in range(count):
        following = (k + 1) % count
        if ends[k]:
            quads.append((k - 1 if k else count - 1, k, following, count + slot[root(k - 1)]))
        elif not ends[following]:
            quads.append((k, following, count + slot[root(following)], count + slot[root(k)]))
    return inner, quads, len(order)


def quad_fill(
    loop_uv: np.ndarray,
    lift,
    size_fn,
    *,
    strategy: str = "FLOW",
    corners=None,
    seed_count: int = 2,
    max_points: int = 20_000,
    smooth_iterations: int = 4,
    cheap_lift: bool = True,
    seam_layer: str = "AUTO",
    seam_angles=None,
    target: float = 0.0,
    field_mode: str = "SMOOTH",
):
    """All-quad fill of a planar disk whose CCW boundary is `loop_uv` (even count).

    `lift(uv, faces)` returns 3D positions for every row of uv (seam rows
    included); `size_fn(points, interior_factor)` is the coarse target edge
    length in the planar domain, the interior part scaled by the factor;
    `cheap_lift` lets smoothing re-check quads in 3D. Vertex k < B is seam
    vertex k. Returns (uv, xyz, quads, label).

    ParityPath keeps paired quads whole, so its coarse mesh uses the target
    size. If it cannot close the seam validly, the coarse mesh is rebuilt
    at twice the interior size and every face is corner-split instead.

    `seam_layer` ("AUTO", "ON", "OFF") adds a seam-conforming boundary layer
    (a pillow): the interior is filled against an inner copy of the loop (same
    count, so parity holds) and joined to the seam by one ring of quads
    (i, i + 1, o_i+1, o_i). Every seam vertex then has two quads, so a reflex
    seam corner (interior angle up to ~340 degrees in 3D) can be met by convex
    quads once the optimiser places o_i. AUTO uses it except where a clean
    four-sided Coons grid fits (that one gets the layer only if it leaves
    concave or warped quads); OFF is the plain fill.
    `seam_angles` (interior 3D angle in degrees per loop vertex) turns sharp
    convex corners into one-quad ends (see `build_seam_layer`). `target`
    (3D edge length) lets a Coons fill use more or fewer iso-lines than the
    seam has (`_coons_points`); 0 keeps the seam's counts. `field_mode`
    picks the TRIAD-Q cross field (`RemeshOptions.field_mode`).
    """

    count = len(loop_uv)
    if count < 4 or count % 2:
        raise ValueError(f"seam needs an even count >= 4 (got {count})")
    fixed_count = count

    def finish(uv, faces):
        fixed = np.zeros(len(uv), dtype=bool)
        fixed[:fixed_count] = True
        if not cheap_lift:
            uv = _smooth_quads(uv, faces, fixed, smooth_iterations)
            return uv, _untangle(lift(uv, faces), faces, fixed, lambda p: p, planar=False)

        def check(points):
            return lift(points, faces)

        uv = _smooth_quads(uv, faces, fixed, smooth_iterations, check)
        uv = _untangle(uv, faces, fixed, check, planar=True)
        return uv, lift(uv, faces)

    def bad_count(xyz, faces):
        return int((quad_quality(xyz[np.asarray(faces, dtype=np.int64)]) <= 0.0).sum())

    best = None
    plain_first = seam_layer == "OFF" or count < 8 or (seam_layer == "AUTO" and corners is not None)
    if plain_first:
        uv, faces, label = _fill_core(loop_uv, lift, size_fn, strategy, corners, seed_count, max_points, target, field_mode)
        uv, xyz = finish(uv, faces)
        best = (bad_count(xyz, faces), uv, xyz, faces, label)
        if seam_layer == "OFF" or count < 8 or best[0] == 0:
            return best[1:]
    variants = [seam_angles, None] if seam_angles is not None else [None]
    for angles in variants:  # with corner ends, then a plain pillow
        inner, layer, inner_count = build_seam_layer(loop_uv, angles)
        if angles is not None and inner_count == count and len(variants) > 1:
            continue  # no ends: identical to the plain pillow
        try:
            uv_in, faces_in, label = _fill_core(
                inner, lift, size_fn, strategy, corners if inner_count == count else None, seed_count, max_points, target,
                field_mode,
            )
        except ValueError:
            continue
        uv = np.concatenate((loop_uv, uv_in))
        faces = layer + [tuple(count + v for v in face) for face in faces_in]
        uv, xyz = finish(uv, faces)
        layered = (bad_count(xyz, faces), uv, xyz, faces, label + "+Layer")
        if best is None or layered[0] < best[0]:
            best = layered
        if best[0] == 0:
            break
    if best is None:  # no layer could be built: plain fill
        uv, faces, label = _fill_core(loop_uv, lift, size_fn, strategy, corners, seed_count, max_points, target, field_mode)
        uv, xyz = finish(uv, faces)
        best = (0, uv, xyz, faces, label)
    return best[1:]


# --------------------------------------------------------------------------- region remesh


def _triangulate_component(faces, face_ids, corner_uvs=None):
    """Fan-triangulate a face set. Returns (global vertex ids, local tris, tri source face, tri corner uvs)."""

    tris, source, uvs = [], [], []
    for face_index in face_ids:
        face = faces[int(face_index)]
        face_uv = corner_uvs[int(face_index)] if corner_uvs is not None else None
        for k in range(1, len(face) - 1):
            tris.append((face[0], face[k], face[k + 1]))
            source.append(int(face_index))
            if face_uv is not None:
                uvs.append((face_uv[0], face_uv[k], face_uv[k + 1]))
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    vertex_ids, local = np.unique(tris, return_inverse=True)
    tri_uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 3, 2) if corner_uvs is not None else None
    return vertex_ids, local.reshape(-1, 3), np.asarray(source, dtype=np.int64), tri_uv


def _graph_distance(count: int, edges: np.ndarray, lengths: np.ndarray, sources) -> tuple[np.ndarray, np.ndarray]:
    """Multi-source Dijkstra: (distance, nearest source) per vertex."""

    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for (a, b), length in zip(edges.tolist(), lengths.tolist()):
        neighbours[a].append((b, length))
        neighbours[b].append((a, length))
    distance = np.full(count, np.inf)
    nearest = np.full(count, -1, dtype=np.int64)
    heap = []
    for source in sources:
        distance[source] = 0.0
        nearest[source] = source
        heap.append((0.0, int(source)))
    heapq.heapify(heap)
    while heap:
        d, vertex = heapq.heappop(heap)
        if d > distance[vertex]:
            continue
        for other, length in neighbours[vertex]:
            candidate = d + length
            if candidate < distance[other]:
                distance[other] = candidate
                nearest[other] = nearest[vertex]
                heapq.heappush(heap, (candidate, other))
    return distance, nearest


def _corner_angle_sums(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Sum of triangle corner angles (degrees) at each vertex: a seam vertex's interior angle in 3D."""

    p = points[tris]
    total = np.zeros(len(points))
    for k in range(3):
        u = p[:, (k + 1) % 3] - p[:, k]
        w = p[:, (k + 2) % 3] - p[:, k]
        cos = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
        total += np.bincount(tris[:, k], np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), len(points))
    return total


def _vertex_normals(points: np.ndarray, faces) -> np.ndarray:
    normals = np.zeros_like(points)
    for size in {len(face) for face in faces}:
        group = np.asarray([face for face in faces if len(face) == size], dtype=np.int64)
        p = points[group]
        n = np.cross(p[:, 1:-1] - p[:, :1], p[:, 2:] - p[:, :1]).sum(axis=1)
        for k in range(size):
            for axis in range(3):
                normals[:, axis] += np.bincount(group[:, k], n[:, axis], len(points))
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 1e-30, length, 1.0)


def remesh_region(
    vertices,
    faces,
    face_ids,
    *,
    target_length: float = 0.0,
    strategy: str = "FLOW",
    falloff: float = 0.0,
    corner_uvs=None,
    seed_count: int = 2,
    max_faces: int = 20_000,
    seam_layer: str = "AUTO",
    field_mode: str = "SMOOTH",
) -> PatchResult:
    """Replace one disk-shaped face set by an all-quad patch on the same seam.

    `target_length` <= 0 uses the mean seam edge length. Near the seam the
    edge length lerps from the seam's local edge length to the target over
    `falloff` (smoothstep on graph distance from the seam).
    """

    started = time.perf_counter()
    vertices = np.asarray(vertices, dtype=np.float64)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    result = PatchResult(ok=False, base=len(vertices), removed_faces=face_ids, strategy=strategy)
    loops, pinched = boundary_loops(faces, face_ids)
    if pinched or len(loops) != 1:
        result.message = "region is not a disk with one seam loop"
        return result
    seam = loops[0]
    result.seam = list(seam)
    if len(seam) % 2:
        result.message = f"odd seam ({len(seam)} edges)"
        return result

    vertex_ids, tris, tri_source, tri_uv = _triangulate_component(faces, face_ids, corner_uvs)
    local_of = {int(v): k for k, v in enumerate(vertex_ids.tolist())}
    xyz = vertices[vertex_ids]
    tri_list = [tuple(t) for t in tris.tolist()]
    vertex_normals = _vertex_normals(xyz, tri_list)
    seam_local = np.asarray([local_of[v] for v in seam], dtype=np.int64)
    if skip_offset(seam_turning(xyz[seam_local], vertex_normals[seam_local])):
        seam = seam[1:] + seam[:1]
        seam_local = np.roll(seam_local, -1)
    seam_xyz = xyz[seam_local]
    seam_edges = np.linalg.norm(np.roll(seam_xyz, -1, axis=0) - seam_xyz, axis=1)
    target = float(target_length) if target_length > 0 else float(seam_edges.mean())
    falloff = float(falloff) if falloff > 0 else 2.0 * target

    corners = None
    if strategy == "GRID":
        corners = four_corners(seam_xyz, vertex_normals[seam_local])
    boundary_uv = square_boundary(seam_xyz, corners) if corners else circle_boundary(seam_xyz)
    uv = tutte_embedding(len(xyz), tris, seam_local, boundary_uv, xyz)
    locator = _Locator(uv, tris)

    # Coarse sizing: twice the local seam edge at the seam (closure halves it), target inside.
    edges = _unique_edges(tri_list)
    length3 = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    length2 = np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=1)
    distance, nearest = _graph_distance(len(xyz), edges, length3, seam_local.tolist())
    local_seam = np.zeros(len(xyz))
    local_seam[seam_local] = 0.5 * (seam_edges + np.roll(seam_edges, 1))
    blend = smoothstep(distance / max(falloff, 1e-30))
    seam_part = (1.0 - blend) * 2.0 * local_seam[np.maximum(nearest, 0)]
    ratio = np.bincount(edges.ravel(), np.repeat(length2 / np.maximum(length3, 1e-30), 2), len(xyz))
    ratio /= np.maximum(np.bincount(edges.ravel(), minlength=len(xyz)), 1)
    area3 = float(np.linalg.norm(np.cross(xyz[tris[:, 1]] - xyz[tris[:, 0]], xyz[tris[:, 2]] - xyz[tris[:, 0]]), axis=1).sum()) * 0.5
    budget = math.sqrt(max(1.0, area3 / max(target * target, 1e-30) / max_faces))

    def size_fn(points, factor):
        size2 = (seam_part + blend * target * factor) * ratio * budget
        t, b = locator.locate(points)
        return np.einsum("ij,ij->i", size2[tris[t]], b)

    def lift(points, _faces):
        t, b = locator.locate(points)
        return np.einsum("ijk,ij->ik", xyz[tris[t]], b)

    try:
        loop_uv = uv[seam_local]
        out_uv, out_xyz, out_faces, label = quad_fill(
            loop_uv, lift, size_fn, strategy=strategy, corners=corners, seed_count=seed_count, max_points=max_faces,
            seam_layer=seam_layer, seam_angles=_corner_angle_sums(xyz, tris)[seam_local], target=target,
            field_mode=field_mode,
        )
    except ValueError as exc:
        result.message = f"fill failed: {exc}"
        return result

    count = len(seam)
    used = sorted({v for face in out_faces for v in face if v >= count})
    new_index = {v: result.base + k for k, v in enumerate(used)}
    seam_global = list(seam)

    def to_global(v: int) -> int:
        return seam_global[v] if v < count else new_index[v]

    result.points = out_xyz[used] if used else np.zeros((0, 3))
    result.faces = [tuple(to_global(v) for v in face) for face in out_faces]
    centroid = np.asarray([out_uv[list(face)].mean(axis=0) for face in out_faces])
    t, _b = locator.locate(centroid)
    result.face_source = tri_source[t].tolist()
    if tri_uv is not None:
        corner_t, corner_b = locator.locate(out_uv)
        corner_uv = np.einsum("ijk,ij->ik", tri_uv[corner_t], corner_b)
        result.face_uvs = [tuple(tuple(corner_uv[v]) for v in face) for face in out_faces]
    result.closure = label
    result.ok = True
    result.message = f"{len(out_faces)} quads ({label})"
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


# --------------------------------------------------------------------------- hole fill


def _membrane(points: np.ndarray, faces, fixed: np.ndarray, extra_edges=None, bilaplacian: bool = True) -> np.ndarray:
    """Harmonic, then bilaplacian (least-squares fair) positions for the free vertices.

    Uniform Laplacian L x = x - mean(neighbours). The bilaplacian step
    minimises |L x|^2 over free and seam rows, so with the outside ring as
    extra fixed neighbours the fill continues the surrounding curvature.
    """

    count = len(points)
    edges = _unique_edges(faces)
    if extra_edges is not None and len(extra_edges):
        edges = np.concatenate((edges, np.asarray(extra_edges, dtype=np.int64).reshape(-1, 2)))
    i, j = edges[:, 0], edges[:, 1]
    degree = np.maximum(np.bincount(edges.ravel(), minlength=count), 1).astype(np.float64)
    free = np.flatnonzero(~fixed)
    if not len(free):
        return points
    x = points.copy()

    def neighbour_sum(v):
        out = np.zeros_like(v)
        for axis in range(v.shape[1]):
            out[:, axis] = np.bincount(i, v[j, axis], count) + np.bincount(j, v[i, axis], count)
        return out

    def laplacian(v):
        return v - neighbour_sum(v) / degree[:, None]

    def harmonic_matvec(v):  # (D - W) restricted to free rows: symmetric for CG
        full = np.zeros_like(x)
        full[free] = v
        return (degree[:, None] * laplacian(full))[free]

    base = x.copy()
    base[free] = 0.0
    rhs = -(degree[:, None] * laplacian(base))[free]
    start = np.tile(points[fixed].mean(axis=0), (len(free), 1))
    x[free] = _cg(harmonic_matvec, rhs, degree[free], start, 3000, 1e-10)
    if not bilaplacian:
        return x
    rows = np.zeros(count, dtype=bool)
    rows[free] = True
    rows[np.unique(edges[np.isin(edges, free).any(axis=1)])] = True
    rows &= np.bincount(edges.ravel(), minlength=count) > 0

    def transpose_laplacian(v):
        scaled = v / degree[:, None]
        return v - neighbour_sum(scaled)

    def normal_matvec(v):
        full = np.zeros_like(x)
        full[free] = v
        r = laplacian(full)
        r[~rows] = 0.0
        return transpose_laplacian(r)[free]

    r0 = laplacian(base)
    r0[~rows] = 0.0
    rhs = -transpose_laplacian(r0)[free]
    x[free] = _cg(normal_matvec, rhs, np.ones(len(free)), x[free], 3000, 1e-10)
    return x


def fill_loop(
    vertices,
    loop,
    *,
    ring_edges=None,
    target_length: float = 0.0,
    strategy: str = "FLOW",
    seed_count: int = 2,
    max_faces: int = 20_000,
    seam_layer: str = "AUTO",
    field_mode: str = "SMOOTH",
) -> PatchResult:
    """Fill a closed hole loop (CCW as the new faces should run) with all quads.

    `ring_edges` are (loop vertex id, outside neighbour id) pairs; when given,
    the fill is faired against them so it follows the surrounding curvature.
    """

    started = time.perf_counter()
    vertices = np.asarray(vertices, dtype=np.float64)
    loop = [int(v) for v in loop]
    result = PatchResult(ok=False, base=len(vertices), seam=list(loop), strategy=strategy)
    count = len(loop)
    if count % 2 or count < 4:
        result.message = f"hole loop needs an even edge count >= 4 (got {count}); add or dissolve one edge"
        return result
    loop_xyz = vertices[loop]
    center = loop_xyz.mean(axis=0)
    normal = np.cross(loop_xyz - center, np.roll(loop_xyz, -1, axis=0) - center).sum(axis=0)
    area = 0.5 * float(np.linalg.norm(normal))
    normal /= max(2.0 * area, 1e-30)
    if skip_offset(seam_turning(loop_xyz, np.tile(normal, (count, 1)))):
        loop = loop[1:] + loop[:1]
        loop_xyz = vertices[loop]
    edges = np.linalg.norm(np.roll(loop_xyz, -1, axis=0) - loop_xyz, axis=1)
    perimeter = float(edges.sum())
    target = float(target_length) if target_length > 0 else float(edges.mean())
    corners = None
    if strategy == "GRID":
        corners = four_corners(loop_xyz, np.tile(normal, (count, 1)))
    loop_uv = square_boundary(loop_xyz, corners) if corners else circle_boundary(loop_xyz)
    # Planar units per 3D unit from areas: a jagged seam would inflate a perimeter ratio.
    area = max(area, (perimeter / (4.0 * math.pi)) ** 2 * math.pi * 0.05)
    scale = math.sqrt((1.0 if corners else math.pi) / area)
    local_seam = 0.5 * (edges + np.roll(edges, 1))

    def size_fn(points, factor):
        depth = _polygon_distance(points, loop_uv) / scale
        nearest = np.argmin(np.linalg.norm(points[:, None, :] - loop_uv[None], axis=2), axis=1)
        blend = smoothstep(depth / max(2.0 * target, 1e-30))
        return ((1.0 - blend) * 2.0 * local_seam[nearest] + blend * target * factor) * scale

    ring = []
    if ring_edges:
        index_of = {v: k for k, v in enumerate(loop)}
        outside = sorted({b for _a, b in ring_edges if b not in index_of})
        ring = [(index_of[a], b) for a, b in ring_edges if a in index_of and b not in index_of]
    else:
        outside = []

    def lift(points, faces):
        n = len(points)
        positions = np.zeros((n, 3))
        positions[:count] = loop_xyz
        fixed = np.zeros(n, dtype=bool)
        fixed[:count] = True
        if not ring:
            return _membrane(positions, faces, fixed)
        offset = {v: n + k for k, v in enumerate(outside)}
        positions = np.vstack((positions, vertices[outside]))
        fixed = np.concatenate((fixed, np.ones(len(outside), dtype=bool)))
        extra = [(a, offset[b]) for a, b in ring]
        return _membrane(positions, faces, fixed, extra)[:n]

    try:
        out_uv, out_xyz, out_faces, label = quad_fill(
            loop_uv, lift, size_fn, strategy=strategy, corners=corners, seed_count=seed_count, max_points=max_faces,
            cheap_lift=False, seam_layer=seam_layer, target=target, field_mode=field_mode,
            seam_angles=180.0 - np.degrees(seam_turning(loop_xyz, np.tile(normal, (count, 1)))),
        )
    except ValueError as exc:
        result.message = f"fill failed: {exc}"
        return result
    used = sorted({v for face in out_faces for v in face if v >= count})
    new_index = {v: result.base + k for k, v in enumerate(used)}
    result.points = out_xyz[used] if used else np.zeros((0, 3))
    result.faces = [tuple(loop[v] if v < count else new_index[v] for v in face) for face in out_faces]
    result.face_source = [-1] * len(out_faces)
    result.closure = label
    result.ok = True
    result.message = f"{len(out_faces)} quads ({label})"
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


# --------------------------------------------------------------------------- fit + apply


def fit_to_surface(
    points, faces, pinned, project=None, iterations: int = 5, strength: float = 0.5, guard: bool = True
) -> np.ndarray:
    """Tangential Laplacian relax p' = p + l (I - n n^T)(avg - p), then project, pinned rows fixed.

    `project(points)` maps (N, 3) points onto the target surface (a BVH in
    Blender); None keeps the relax purely tangential. Moves that leave a quad
    worse and poor are undone; with `guard`, poor quads are then optimised
    (`_untangle` with projected candidates), so fitting keeps the quad quality.
    """

    points = np.asarray(points, dtype=np.float64).copy()
    pinned = np.asarray(pinned, dtype=bool)
    edges = _unique_edges(faces)
    count = len(points)
    if not len(edges):
        return points
    degree = np.bincount(edges.ravel(), minlength=count).astype(np.float64)
    free = ~pinned & (degree > 0)
    quads = np.asarray([face for face in faces if len(face) == 4], dtype=np.int64).reshape(-1, 4)
    for _ in range(max(0, int(iterations))):
        normals = _vertex_normals(points, faces)
        total = np.zeros_like(points)
        for axis in range(3):
            total[:, axis] = np.bincount(edges[:, 0], points[edges[:, 1], axis], count) + np.bincount(
                edges[:, 1], points[edges[:, 0], axis], count
            )
        delta = total / np.maximum(degree, 1)[:, None] - points
        delta -= normals * np.einsum("ij,ij->i", delta, normals)[:, None]
        moved = points + strength * delta
        if project is not None and free.any():
            moved[free] = project(moved[free])
        moved[~free] = points[~free]
        if len(quads):  # undo moves that leave a quad worse and poor (concave or warped)
            before = quad_quality(points[quads])
            for _attempt in range(64):  # reverting only ever returns towards `points`, so this ends
                after = quad_quality(moved[quads])
                bad = (after < before - 1e-9) & (after < 0.2)
                if not bad.any():
                    break
                undo = np.unique(quads[bad])
                moved[undo] = points[undo]
        points = moved
    if guard and len(quads) and free.any():
        points = _untangle(points, quads, ~free, project if project is not None else (lambda p: p), planar=False)
    return points


def apply_patches(vertices, faces, patches) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    """Merge patches into a copy of the mesh.

    Returns (vertices, faces, is_new_face); untouched faces keep their order
    and come first. Loose interior vertices of removed regions stay (callers
    with a real mesh delete them).
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    removed = np.zeros(len(faces), dtype=bool)
    out_points = [vertices]
    new_faces: list[tuple[int, ...]] = []
    offset = len(vertices)
    for patch in patches:
        if not patch.ok:
            continue
        removed[patch.removed_faces] = True
        shift = offset - patch.base
        new_faces.extend(tuple(v if v < patch.base else v + shift for v in face) for face in patch.faces)
        out_points.append(patch.points)
        offset += len(patch.points)
    kept = [tuple(face) for face, gone in zip(faces, removed) if not gone]
    flags = np.concatenate((np.zeros(len(kept), dtype=bool), np.ones(len(new_faces), dtype=bool)))
    return np.vstack(out_points), kept + new_faces, flags
