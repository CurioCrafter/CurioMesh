# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Flat Quads: spend almost no faces on flat areas, and all of them on detail.

A conforming all-quad disk has F = I + m/2 - 1 quads (I interior vertices, m
border vertices), so a densely sampled border forces many quads however flat
the inside is. This pass therefore works on the layout before pairing:

1. Grow near-planar regions, largest faces first, and merge neighbours whose
   common plane still fits every vertex.
2. Simplify each border chain once, so both regions sharing it agree.
3. Retriangulate each region from its simplified outline (holes kept). A chain
   next to kept detail stays dense, and density grades away from it.

The result is a conforming coarse mesh that TRIAD-Q pairs and closes as usual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .engine import RemeshOptions, RemeshReport, remesh_mesh
from .patch import _half_edges, _inside_polygon, _key, _labels, _orient, _Planar, _polygon_distance, _tri_edges, face_adjacency
from .topology import parity_path_quad_closure, quad_only_closure, topology_summary
from .types import MeshData


@dataclass(slots=True)
class FlatQuadOptions:
    tolerance: float = 0.001  # largest distance from a region's plane, as a fraction of the bounding-box diagonal
    angle_deg: float = 20.0  # largest face normal deviation from a region's plane
    min_faces: int = 12  # smaller regions stay detail: on rough scans QEM reduces them better and far faster
    min_saving: float = 0.3  # a rebuilt region must use at least this fraction fewer triangles
    grading: float = 0.6  # how fast density falls with distance from a dense chain
    max_outline: int = 20_000  # outlines above this stay detail (ear clipping is quadratic)
    seed_count: int = 2


@dataclass(slots=True)
class FlatQuadReport:
    input_faces: int = 0
    output_faces: int = 0
    regions: int = 0
    rejected_regions: int = 0
    flat_faces_in: int = 0
    flat_faces_out: int = 0
    detail_faces: int = 0
    tolerance: float = 0.0
    all_quads: bool = False
    message: str = ""
    flat_face_mask: list[bool] = field(default_factory=list)
    rejected_faces: dict[str, int] = field(default_factory=dict)  # faces in regions turned down, by reason


class _Unfit(ValueError):
    """A region's simplified outline cannot be retriangulated."""


def _face_frames(vertices: np.ndarray, faces) -> tuple[np.ndarray, np.ndarray]:
    """(unit normal, area) per polygon, by Newell's method."""

    face_of, a, b = _half_edges(faces)
    cross = np.cross(vertices[a], vertices[b])
    total = np.stack([np.bincount(face_of, cross[:, axis], len(faces)) for axis in range(3)], axis=1)
    length = np.linalg.norm(total, axis=1)
    return total / np.maximum(length, 1e-30)[:, None], 0.5 * length


def _fit_plane(points: np.ndarray, normal_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = points.mean(axis=0)
    centred = points - origin
    normal = np.linalg.eigh(centred.T @ centred)[1][:, 0]
    return origin, -normal if float(normal @ normal_hint) < 0.0 else normal


def _grow_regions(vertices, faces, normals, areas, materials, tol, cos_limit) -> np.ndarray:
    """Region id per face (every face starts or joins one), largest faces seeding first."""

    fa, fb = face_adjacency(faces)
    neighbours: list[list[int]] = [[] for _ in faces]
    for a, b in zip(fa.tolist(), fb.tolist()):
        neighbours[a].append(b)
        neighbours[b].append(a)
    labels = np.full(len(faces), -1, dtype=np.int64)
    next_label = 0
    for seed in np.argsort(-areas, kind="stable").tolist():
        if labels[seed] >= 0:
            continue
        labels[seed] = next_label
        members = [seed]
        ids = set(faces[seed])
        normal = normals[seed].copy()
        origin = vertices[list(faces[seed])].mean(axis=0)
        refit_at = 8
        frontier = list(neighbours[seed])
        while frontier:
            face = frontier.pop()
            if labels[face] >= 0 or materials[face] != materials[seed]:
                continue
            if float(normals[face] @ normal) < cos_limit:
                continue
            if np.abs((vertices[list(faces[face])] - origin) @ normal).max() > tol:
                continue
            labels[face] = next_label
            members.append(face)
            ids.update(faces[face])
            frontier.extend(neighbours[face])
            if len(members) >= refit_at:  # the seed's normal is noisy: settle on the fitted plane
                origin, fitted = _fit_plane(vertices[list(ids)], normal)
                if np.abs((vertices[list(ids)] - origin) @ fitted).max() <= tol:
                    normal = fitted
                refit_at *= 2
        next_label += 1
    return labels


def _merge_regions(vertices, faces, normals, labels, materials, fa, fb, tol, cos_limit) -> np.ndarray:
    """Merge neighbours whose common plane fits every vertex, so noise can't fragment a plate."""

    count = int(labels.max()) + 1
    parent = list(range(count))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    members: list[list[int]] = [[] for _ in range(count)]
    for face, label in enumerate(labels.tolist()):
        members[label].append(face)
    region_ids = [set(v for f in group for v in faces[f]) for group in members]
    differ = labels[fa] != labels[fb]
    pairs, shared = np.unique(np.sort(np.stack((labels[fa][differ], labels[fb][differ]), axis=1), axis=1), axis=0, return_counts=True)
    for row in np.argsort(-shared, kind="stable").tolist():
        a, b = root(int(pairs[row, 0])), root(int(pairs[row, 1]))
        if a == b or materials[members[a][0]] != materials[members[b][0]]:
            continue
        if len(members[a]) + len(members[b]) < 6:
            continue
        ids = list(region_ids[a] | region_ids[b])
        hint = normals[members[a] + members[b]].sum(axis=0)
        if float(np.linalg.norm(hint)) < 1e-12:
            continue
        origin, normal = _fit_plane(vertices[ids], hint)
        if np.abs((vertices[ids] - origin) @ normal).max() > tol:
            continue
        if (normals[members[a]] @ normal).min() < cos_limit or (normals[members[b]] @ normal).min() < cos_limit:
            continue
        parent[b] = a
        members[a] = members[a] + members[b]
        region_ids[a] = region_ids[a] | region_ids[b]
    return np.asarray([root(int(label)) for label in labels], dtype=np.int64)


def _douglas_peucker(points: np.ndarray, tol: float) -> list[int]:
    """Indices of the chain points kept (always both ends)."""

    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        a, b = points[first], points[last]
        span = b - a
        length2 = float(span @ span)
        inner = points[first + 1 : last] - a
        if length2 < 1e-30:
            distance = np.linalg.norm(inner, axis=1)
        else:
            along = np.clip(inner @ span / length2, 0.0, 1.0)
            distance = np.linalg.norm(inner - along[:, None] * span, axis=1)
        worst = int(np.argmax(distance))
        if distance[worst] > tol:
            middle = first + 1 + worst
            keep.add(middle)
            stack.extend(((first, middle), (middle, last)))
    return sorted(keep)


def _edge_faces(faces) -> dict[tuple[int, int], list[int]]:
    users: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for k, a in enumerate(face):
            b = face[(k + 1) % len(face)]
            users.setdefault((a, b) if a < b else (b, a), []).append(index)
    return users


def _kept_vertices(vertices, faces, edge_faces, labels, usable, tol) -> tuple[set[int], set[int]]:
    """(vertices every region must keep, the subset locked to detail or non-manifold edges)."""

    locked: set[int] = set()
    for index in np.flatnonzero(~usable[labels]).tolist():
        locked.update(faces[index])
    links: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for (a, b), users in edge_faces.items():
        if len(users) > 2:
            locked.update((a, b))
            continue
        first = int(labels[users[0]])
        second = int(labels[users[1]]) if len(users) == 2 else -1
        if first == second:
            continue
        kind = (first, second) if first < second else (second, first)
        links.setdefault(a, []).append((b, kind))
        links.setdefault(b, []).append((a, kind))
    dense = set(locked)
    corners = set(locked)
    for vertex, ends in links.items():
        if len(ends) != 2 or ends[0][1] != ends[1][1]:
            corners.add(vertex)

    kept = set(corners)
    visited: set[tuple[int, int]] = set()

    def walk(start: int, step: int) -> list[int]:
        chain = [start, step]
        visited.add((start, step) if start < step else (step, start))
        while chain[-1] not in corners and chain[-1] != start:
            here = chain[-1]
            following = next((v for v, _kind in links[here] if ((v, here) if v < here else (here, v)) not in visited), None)
            if following is None:
                break
            visited.add((following, here) if following < here else (here, following))
            chain.append(following)
        return chain

    for vertex in sorted(corners & set(links)):
        for step, _kind in links[vertex]:
            if ((vertex, step) if vertex < step else (step, vertex)) not in visited:
                chain = walk(vertex, step)
                picks = _douglas_peucker(vertices[chain], tol)
                kept.update(chain[i] for i in picks)
    for vertex in sorted(links):  # closed loops with no corner: anchor at the two farthest points
        if vertex in corners or all(((v, vertex) if v < vertex else (vertex, v)) in visited for v, _k in links[vertex]):
            continue
        loop = walk(vertex, links[vertex][0][0])
        if loop[-1] == loop[0]:
            loop = loop[:-1]
        points = vertices[loop]
        far = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
        third = int(np.argmax(np.linalg.norm(np.cross(points[far] - points[0], points - points[0]), axis=1)))
        anchors = sorted({0, far, third})
        for i, start in enumerate(anchors):
            stop = anchors[(i + 1) % len(anchors)]
            span = list(range(start, stop + 1)) if stop > start else list(range(start, len(loop))) + list(range(stop + 1))
            kept.update(loop[span[j]] for j in _douglas_peucker(points[span], tol))
    return kept, dense


def _largest_piece(group: list[int], faces) -> list[int]:
    """The biggest edge-connected part of a face list."""

    by_edge: dict[tuple[int, int], list[int]] = {}
    for face in group:
        polygon = faces[face]
        for k, a in enumerate(polygon):
            b = polygon[(k + 1) % len(polygon)]
            by_edge.setdefault((a, b) if a < b else (b, a), []).append(face)
    remaining = set(group)
    best: list[int] = []
    while remaining:
        stack = [remaining.pop()]
        piece = []
        while stack:
            face = stack.pop()
            piece.append(face)
            polygon = faces[face]
            for k, a in enumerate(polygon):
                b = polygon[(k + 1) % len(polygon)]
                for other in by_edge[(a, b) if a < b else (b, a)]:
                    if other in remaining:
                        remaining.discard(other)
                        stack.append(other)
        if len(piece) > len(best):
            best = piece
    return best


def _segments_cross(p1, p2, q1, q2) -> np.ndarray:
    """Whether segment p1-p2 properly crosses each segment q1[i]-q2[i]."""

    def side(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    d1, d2 = side(q1, q2, p1), side(q1, q2, p2)
    d3, d4 = side(p1, p2, q1), side(p1, p2, q2)
    return (d1 * d2 < 0.0) & (d3 * d4 < 0.0)


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _bridge_holes(points: np.ndarray, outer: list[int], holes: list[list[int]]) -> list[int]:
    """One weakly simple polygon: each hole is joined to the outline by a visible bridge."""

    polygon = list(outer)
    for hole in sorted(holes, key=lambda loop: -float(points[loop, 0].max())):
        start = int(np.argmax(points[hole, 0]))
        hole = hole[start:] + hole[:start]
        pivot = points[hole[0]]
        ring = np.asarray(polygon)
        others = [loop for loop in holes if loop is not hole]
        first = np.concatenate([ring] + [np.asarray(loop) for loop in others + [hole]])
        second = np.concatenate([np.roll(ring, -1)] + [np.roll(np.asarray(loop), -1) for loop in others + [hole]])
        chosen = None
        for candidate in np.argsort(np.linalg.norm(points[ring] - pivot, axis=1)).tolist():
            target = points[ring[candidate]]
            if not _segments_cross(pivot, target, points[first], points[second]).any():
                chosen = candidate
                break
        if chosen is None:
            raise _Unfit("no visible bridge from a hole to the outline")
        polygon[chosen + 1 : chosen + 1] = hole + [hole[0], polygon[chosen]]
    return polygon


def _ear_clip(points: np.ndarray, polygon: list[int]) -> list[tuple[int, int, int]]:
    """Triangles of a CCW weakly simple polygon; collinear runs never become ear tips."""

    scale = float(np.ptp(points, axis=0).max()) or 1.0
    eps = 1e-12 * scale * scale
    following = list(range(1, len(polygon))) + [0]
    previous = [len(polygon) - 1] + list(range(len(polygon) - 1))
    alive = len(polygon)

    def turn(k: int) -> float:
        return _orient(points[polygon[previous[k]]], points[polygon[k]], points[polygon[following[k]]])

    def is_ear(k: int) -> bool:
        a, b, c = polygon[previous[k]], polygon[k], polygon[following[k]]
        pa, pb, pc = points[a], points[b], points[c]
        if _orient(pa, pb, pc) <= 1e-9 * math.dist(pa, pb) * math.dist(pb, pc):
            return False
        ring = [following[k]]
        while ring[-1] != previous[k]:
            ring.append(following[ring[-1]])
        ids = np.asarray([polygon[i] for i in ring[1:-1]], dtype=np.int64)
        ids = ids[(ids != a) & (ids != b) & (ids != c)]
        if not len(ids):
            return True
        q = points[ids]
        ab = (pb[0] - pa[0]) * (q[:, 1] - pa[1]) - (pb[1] - pa[1]) * (q[:, 0] - pa[0])
        bc = (pc[0] - pb[0]) * (q[:, 1] - pb[1]) - (pc[1] - pb[1]) * (q[:, 0] - pb[0])
        ca = (pa[0] - pc[0]) * (q[:, 1] - pc[1]) - (pa[1] - pc[1]) * (q[:, 0] - pc[0])
        return not ((ab >= -eps) & (bc >= -eps) & (ca >= -eps)).any()

    triangles = []
    k = 0
    misses = 0
    while alive > 3:
        if misses > alive:
            raise _Unfit("outline cannot be clipped into triangles")
        if is_ear(k):
            triangles.append((polygon[previous[k]], polygon[k], polygon[following[k]]))
            following[previous[k]], previous[following[k]] = following[k], previous[k]
            alive -= 1
            misses = 0
            k = previous[k]
        else:
            misses += 1
            k = following[k]
    if turn(k) <= eps:
        raise _Unfit("outline ends in a degenerate triangle")
    triangles.append((polygon[previous[k]], polygon[k], polygon[following[k]]))
    return triangles


def _outline_is_simple(points: np.ndarray, loops: list[list[int]]) -> bool:
    first = np.concatenate([np.asarray(loop) for loop in loops])
    second = np.concatenate([np.roll(np.asarray(loop), -1) for loop in loops])
    return not any(_segments_cross(points[first[i]], points[second[i]], points[first], points[second]).any() for i in range(len(first)))


def _graded_sites(uv: np.ndarray, rings: list[list[int]], size_fn, max_sites: int) -> np.ndarray:
    """Interior points of a quadtree refined to the local size, kept clear of every outline.

    Staying a size away from the outline is what prevents slivers along long, straight, frozen borders.
    """

    low = uv.min(axis=0)
    side = float(np.ptp(uv, axis=0).max())
    centres = low + 0.5 * side
    centres = centres[None]
    half = 0.5 * side
    sites = []
    total = 0
    while len(centres) and half > 1e-9 * side and total < max_sites:
        sizes = size_fn(centres)
        split = sizes < 1.2 * 2.0 * half
        done = centres[~split]
        if len(done):
            sites.append(done)
            total += len(done)
        offsets = 0.5 * half * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        centres = (centres[split, None] + offsets).reshape(-1, 2)
        half *= 0.5
    if not sites:
        return np.zeros((0, 2))
    q = np.concatenate(sites)
    inside = _inside_polygon(q, uv[rings[0]])
    clearance = _polygon_distance(q, uv[rings[0]])
    for ring in rings[1:]:
        inside ^= _inside_polygon(q, uv[ring])
        clearance = np.minimum(clearance, _polygon_distance(q, uv[ring]))
    q = q[inside & (clearance > 0.7 * size_fn(q))]
    return q[np.lexsort((q[:, 0], q[:, 1]))]


def _rebuild_region(vertices, origin, normal, loops, dense, options, tol):
    """Coarse triangles for one region: (loop vertex ids, new 3D points, triangles over ids + new points)."""

    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    ids = sorted({vertex for loop in loops for vertex in loop})
    local = {vertex: k for k, vertex in enumerate(ids)}
    relative = vertices[ids] - origin
    uv = np.stack((relative @ u, relative @ v), axis=1)
    rings = [[local[vertex] for vertex in loop] for loop in loops]
    if any(len(ring) < 3 for ring in rings):
        raise _Unfit("outline collapsed")
    areas = [_signed_area(uv[ring]) for ring in rings]
    outer_at = int(np.argmax(np.abs(areas)))
    outer = rings[outer_at] if areas[outer_at] > 0.0 else rings[outer_at][::-1]
    holes = [ring if area < 0.0 else ring[::-1] for k, (ring, area) in enumerate(zip(rings, areas)) if k != outer_at]
    if abs(areas[outer_at]) < 1e-30 or not _outline_is_simple(uv, [outer] + holes):
        raise _Unfit("outline folds over itself")

    triangles = _ear_clip(uv, _bridge_holes(uv, outer, holes))
    planar = _Planar(uv, triangles, fixed=len(ids))
    planar.flips()
    dense_rows = [local[vertex] for vertex in ids if vertex in dense]
    if dense_rows:  # grade: dense next to kept detail, coarse away from it
        loop_edges = np.concatenate([np.linalg.norm(uv[np.roll(ring, -1)] - uv[ring], axis=1) for ring in rings])
        lengths = dict(zip((k for ring in rings for k in ring), loop_edges.tolist()))
        sources = uv[dense_rows]
        edge = np.asarray([lengths.get(k, float(loop_edges.mean())) for k in dense_rows])

        def size_fn(q: np.ndarray) -> np.ndarray:
            rows = [
                (edge + options.grading * np.linalg.norm(sources - q[i : i + 2048, None], axis=2)).min(axis=1)
                for i in range(0, len(q), 2048)
            ]
            return np.concatenate(rows) if rows else np.zeros(0)

        for q in _graded_sites(uv, rings, size_fn, max_sites=len(ids) * 4 + 64):
            at = planar.walk(q, len(planar.t) - 1)
            a, b, c = (planar.p[k] for k in planar.t[at])
            if min(_orient(a, b, q), _orient(b, c, q), _orient(c, a, q)) <= 0.0:
                continue  # walking stopped at a hole: this site is not reachable from here
            fresh = planar.insert(at, q)
            planar.flips([_key(*sorted(e)) for t in fresh for e in _tri_edges(planar.t[t])])
    points, tris = planar.arrays()
    extra = points[len(ids):]
    lifted = origin + extra[:, 0, None] * u + extra[:, 1, None] * v
    return ids, lifted, tris


def flatten_regions(mesh: MeshData, options: FlatQuadOptions | None = None) -> tuple[MeshData, FlatQuadReport]:
    """Replace flat regions by coarse conforming triangles; other faces pass through untouched."""

    options = options or FlatQuadOptions()
    vertices = mesh.vertices
    faces = mesh.faces
    report = FlatQuadReport(input_faces=len(faces))
    if not faces:
        report.message = "Input mesh has no faces."
        return mesh, report
    used = vertices[sorted({v for face in faces for v in face})]
    tol = float(options.tolerance) * float(np.linalg.norm(np.ptp(used, axis=0)))
    report.tolerance = tol
    cos_limit = math.cos(math.radians(options.angle_deg))
    normals, areas = _face_frames(vertices, faces)
    materials = list(mesh.face_materials)
    labels = _grow_regions(vertices, faces, normals, areas, materials, tol, cos_limit)
    fa, fb = face_adjacency(faces)
    if len(fa):
        labels = _merge_regions(vertices, faces, normals, labels, materials, fa, fb, tol, cos_limit)
    _unique, labels = np.unique(labels, return_inverse=True)
    region_count = int(labels.max()) + 1
    members: list[list[int]] = [[] for _ in range(region_count)]
    for face, label in enumerate(labels.tolist()):
        members[label].append(face)
    usable = np.asarray([len(group) >= options.min_faces for group in members] + [False], dtype=bool)
    spare_label = region_count  # faces released from a region join this detail label
    members.append([])

    def reject(region: int, reason: str) -> None:
        nonlocal failed
        usable[region] = False
        failed = True
        report.rejected_faces[reason] = report.rejected_faces.get(reason, 0) + len(members[region])

    edge_faces = _edge_faces(faces)
    built: dict[int, tuple] = {}
    cache: dict[int, tuple] = {}  # most outlines survive a round unchanged
    for _round in range(32):  # a failed region becomes detail, which re-locks its neighbours' chains
        kept, dense = _kept_vertices(vertices, faces, edge_faces, labels, usable, tol)
        failed = False
        built.clear()
        for region in np.flatnonzero(usable).tolist():
            group = members[region]
            halves = set()
            for face in group:
                polygon = faces[face]
                for k, a in enumerate(polygon):
                    halves.add((a, polygon[(k + 1) % len(polygon)]))
            successor: dict[int, list[int]] = {}
            for a, b in halves:
                if (b, a) not in halves:
                    successor.setdefault(a, []).append(b)
            pinched = {vertex for vertex, nxt in successor.items() if len(nxt) > 1}
            if pinched:  # the region touches itself at a vertex: give up only the faces around it
                keep = _largest_piece([face for face in group if not pinched.intersection(faces[face])], faces)
                labels[list(set(group) - set(keep))] = spare_label
                members[region] = keep
                failed = True
                if len(keep) < options.min_faces:
                    reject(region, "pinched")
                continue
            if not successor:
                reject(region, "no outline")
                continue
            remaining = set(successor)
            loops = []
            while remaining:
                start = min(remaining)
                loop, vertex = [], start
                while vertex in remaining:
                    remaining.discard(vertex)
                    if vertex in kept:
                        loop.append(vertex)
                    vertex = successor[vertex][0]
                loops.append(loop)
            outline = sum(len(loop) for loop in loops)
            source_triangles = sum(len(faces[face]) - 2 for face in group)
            allowed = (1.0 - options.min_saving) * source_triangles
            # A polygon with h holes needs m + 2h - 2 triangles before any grading.
            if outline > options.max_outline or outline + 2 * (len(loops) - 1) - 2 > allowed:
                reject(region, "outline too long")
                continue
            signature = (tuple(tuple(loop) for loop in loops), frozenset(v for loop in loops for v in loop if v in dense))
            known = cache.get(region)
            if known is not None and known[0] == signature:
                result = known[1]
            else:
                ids = list({v for face in group for v in faces[face]})
                origin, normal = _fit_plane(vertices[ids], normals[group].sum(axis=0))
                try:
                    result = _rebuild_region(vertices, origin, normal, loops, dense, options, tol)
                except _Unfit:
                    result = None
                cache[region] = (signature, result)
            if result is None:
                reject(region, "outline folds")
                continue
            if len(result[2]) > allowed:
                reject(region, "too little saving")
                continue
            built[region] = (result, materials[group[0]])
        if not failed:
            break
    else:
        built.clear()

    is_flat = np.asarray([usable[label] and int(label) in built for label in labels.tolist()], dtype=bool)
    out_points = [vertices]
    out_faces = [faces[index] for index in np.flatnonzero(~is_flat).tolist()]
    out_materials = [materials[index] for index in np.flatnonzero(~is_flat).tolist()]
    flags = [False] * len(out_faces)
    offset = len(vertices)
    for (ids, lifted, tris), material in built.values():
        table = np.concatenate((np.asarray(ids, dtype=np.int64), offset + np.arange(len(lifted), dtype=np.int64)))
        out_points.append(lifted)
        offset += len(lifted)
        for tri in table[tris].tolist():
            out_faces.append(tuple(tri))
            out_materials.append(material)
            flags.append(True)
    stacked = np.vstack(out_points)
    alive = sorted({v for face in out_faces for v in face})
    remap = np.full(len(stacked), -1, dtype=np.int64)
    remap[alive] = np.arange(len(alive))
    result = MeshData(
        vertices=stacked[alive],
        faces=[tuple(remap[list(face)].tolist()) for face in out_faces],
        face_materials=out_materials,
        name=mesh.name,
    )
    report.regions = len(built)
    report.rejected_regions = int(sum(1 for group in members if len(group) >= options.min_faces)) - len(built)
    report.flat_faces_in = int(is_flat.sum())
    report.flat_faces_out = int(sum(flags))
    report.detail_faces = int(len(flags) - sum(flags))
    report.output_faces = len(result.faces)
    report.flat_face_mask = flags
    report.message = f"{report.regions} flat regions: {report.flat_faces_in:,} faces became {report.flat_faces_out:,} triangles"
    return result, report


def make_closable(mesh: MeshData) -> tuple[MeshData, int]:
    """Make a dirty mesh certifiable for quad closure without moving anything.

    Faces with a repeated vertex, zero area, or the same vertices as an earlier
    face are dropped. Each vertex is then split into one copy per fan of faces
    joined by a clean edge (exactly two faces, opposite directions), which opens
    non-manifold edges, flipped normals and bow-ties into borders. Returns (mesh,
    vertices added).
    """

    vertices = mesh.vertices
    seen: set[tuple[int, ...]] = set()
    faces: list[tuple[int, ...]] = []
    materials: list[str] = []
    span = float(np.linalg.norm(np.ptp(vertices, axis=0))) if len(vertices) else 1.0
    _normals, areas = _face_frames(vertices, mesh.faces)
    for face, material, area in zip(mesh.faces, mesh.face_materials, areas.tolist()):
        key = tuple(sorted(face))
        if len(set(face)) < len(face) or key in seen or area <= 1e-14 * span * span:
            continue
        seen.add(key)
        faces.append(face)
        materials.append(material)

    directed: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for k, a in enumerate(face):
            directed.setdefault((a, face[(k + 1) % len(face)]), []).append(index)
    users_of: dict[int, list[int]] = {}
    for index, face in enumerate(faces):
        for vertex in face:
            users_of.setdefault(vertex, []).append(index)
    new_points: list[np.ndarray] = []
    corner_copy: dict[tuple[int, int], int] = {}
    next_vertex = len(vertices)
    for vertex, users in users_of.items():
        if len(users) < 2:
            continue
        parent = {face: face for face in users}

        def root(face: int) -> int:
            while parent[face] != face:
                parent[face] = parent[parent[face]]
                face = parent[face]
            return face

        for face in users:
            polygon = faces[face]
            following = polygon[(polygon.index(vertex) + 1) % len(polygon)]
            out = directed.get((vertex, following), [])
            back = directed.get((following, vertex), [])
            if len(out) == 1 and len(back) == 1:
                parent[root(face)] = root(back[0])
        fans = sorted({root(face) for face in users})
        if len(fans) < 2:
            continue
        copy_of = {fans[0]: vertex}
        for fan in fans[1:]:
            copy_of[fan] = next_vertex
            new_points.append(vertices[vertex])
            next_vertex += 1
        for face in users:
            corner_copy[(vertex, face)] = copy_of[root(face)]
    out_faces = [tuple(corner_copy.get((v, index), v) for v in face) for index, face in enumerate(faces)]
    stacked = np.vstack([vertices] + [p[None] for p in new_points]) if new_points else vertices
    return MeshData(vertices=stacked, faces=out_faces, face_materials=materials, name=mesh.name), len(new_points)


def pair_and_close(
    coarse: MeshData, *, seed_count: int = 2, force_quads: bool = True, field_mode: str = "SMOOTH",
    feature_angle_deg: float = 60.0,
) -> tuple[MeshData, RemeshReport]:
    """Repair, pair with TRIAD-Q, then close every connected piece on its own.

    Sparse ParityPath closure keeps each paired quad whole, so a big flat quad
    stays one face. Corner closure splits every face, so it only runs on a piece
    ParityPath refuses: one loose fragment can't cost the rest of the mesh its
    quads. A piece neither can close stays quad-dominant.
    """

    repaired, _added = make_closable(coarse)
    pair = RemeshOptions(
        target_faces=max(1, len(repaired.faces) // 2), seed_count=seed_count, force_quads=False, field_mode=field_mode,
        feature_angle_deg=feature_angle_deg, max_pairing=True,
    )
    paired, solver = remesh_mesh(repaired, pair)
    if not force_quads or not paired.faces:
        return paired, solver
    closed, sparse, split, open_pieces = _close_pieces(paired)
    solver.closure_applied = True
    solver.closure_method = "PARITY_PATH" if not split and not open_pieces else "PER_PIECE"
    solver.output_faces = len(closed.faces)
    solver.quads = sum(1 for face in closed.faces if len(face) == 4)
    solver.tris = sum(1 for face in closed.faces if len(face) == 3)
    solver.quad_ratio = solver.quads / max(1, len(closed.faces))
    solver.success = True
    solver.message = f"Closed {sparse} pieces sparsely, {split} by corner splitting, {open_pieces} left open."
    return closed, solver


def _close_pieces(paired: MeshData) -> tuple[MeshData, int, int, int]:
    """(closed mesh, pieces closed by ParityPath, by corner closure, left as they were)."""

    fa, fb = face_adjacency(paired.faces)
    labels = _labels(len(paired.faces), fa, fb)
    order = np.argsort(labels, kind="stable")
    _unique, starts = np.unique(labels[order], return_index=True)
    points: list[np.ndarray] = []
    faces: list[tuple[int, ...]] = []
    materials: list[str] = []
    offset = 0
    counts = [0, 0, 0]
    for group in np.split(order, starts[1:]):
        members = group.tolist()
        used = sorted({v for f in members for v in paired.faces[f]})
        local = {v: k for k, v in enumerate(used)}
        piece = MeshData(
            vertices=paired.vertices[used],
            faces=[tuple(local[v] for v in paired.faces[f]) for f in members],
            face_materials=[paired.face_materials[f] for f in members],
        )
        result = piece
        kind = 2
        if all(len(face) == 4 for face in piece.faces) and topology_summary(piece).all_quads:
            kind = 0
        else:
            for kind, closure in enumerate((parity_path_quad_closure, quad_only_closure)):
                try:
                    result = closure(piece)
                    break
                except ValueError:
                    continue
            else:
                kind = 2
        counts[kind] += 1
        points.append(result.vertices)
        faces.extend(tuple(v + offset for v in face) for face in result.faces)
        materials.extend(result.face_materials)
        offset += len(result.vertices)
    merged = MeshData(vertices=np.vstack(points), faces=faces, face_materials=materials, name=paired.name)
    return merged, counts[0], counts[1], counts[2]


def flat_quad_remesh(
    mesh: MeshData, options: FlatQuadOptions | None = None, *, force_quads: bool = True, field_mode: str = "SMOOTH",
) -> tuple[MeshData, FlatQuadReport, RemeshReport]:
    """Flatten regions, then pair and close with TRIAD-Q."""

    options = options or FlatQuadOptions()
    coarse, report = flatten_regions(mesh, options)
    paired, solver = pair_and_close(coarse, seed_count=options.seed_count, force_quads=force_quads, field_mode=field_mode)
    report.output_faces = len(paired.faces)
    report.all_quads = bool(paired.faces) and all(len(face) == 4 for face in paired.faces)
    return paired, report, solver


__all__ = ["FlatQuadOptions", "FlatQuadReport", "flat_quad_remesh", "flatten_regions", "make_closable", "pair_and_close"]

