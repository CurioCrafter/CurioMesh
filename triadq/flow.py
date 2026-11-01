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


def _collapse_folded(points: np.ndarray, faces: list) -> None:
    """Close each folded quad along its shorter diagonal, in place, unless that would pinch a neighbour. Closed quads become None."""

    quads = [k for k, f in enumerate(faces) if f is not None]
    if not quads:
        return
    array = np.asarray([faces[k] for k in quads], dtype=np.int64)
    bad = np.flatnonzero(quad_quality(points[array]) <= 0.0)
    users: dict[int, list[int]] = {}
    for k in quads:
        for v in faces[k]:
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
            if faces[f] is not None:
                faces[f] = tuple(first if v == second else v for v in faces[f])
        faces[k] = None


def _dissolve_doublets(faces: list) -> None:
    """A vertex with only two quads adds nothing: join them, in place. The second becomes None."""

    users: dict[int, list[int]] = {}
    for k, f in enumerate(faces):
        if f is not None:
            for v in f:
                users.setdefault(v, []).append(k)
    for vertex, around in users.items():
        if len(around) != 2 or faces[around[0]] is None or faces[around[1]] is None:
            continue
        one, two = faces[around[0]], faces[around[1]]
        if vertex not in one or vertex not in two:
            continue
        i, j = one.index(vertex), two.index(vertex)
        p, x, q = one[(i + 1) % 4], one[(i + 2) % 4], one[(i + 3) % 4]
        if two[(j + 1) % 4] != q or two[(j + 3) % 4] != p or two[(j + 2) % 4] == x:
            continue  # border vertex, or the two quads don't share both its edges
        faces[around[0]] = (p, x, q, two[(j + 2) % 4])
        faces[around[1]] = None


def _refill_folded(moved: np.ndarray, faces: list, most_rings: int = 4) -> np.ndarray:
    """Last resort, in place on `faces`: replace a folded quad and rings around it by a fan from their middle. Returns the points, with any new ones.

    The outline doesn't move, so nothing outside the area can fold because of it. The area grows until a fan fits.
    """

    extra: list[np.ndarray] = []
    for _pass in range(64):
        live = [k for k, f in enumerate(faces) if f is not None]
        if not live:
            break
        points = np.vstack([moved, *extra]) if extra else moved
        array = np.asarray([faces[k] for k in live], dtype=np.int64)
        bad = np.flatnonzero(quad_quality(points[array]) <= 0.0)
        if not len(bad):
            break
        users: dict[int, list[int]] = {}
        for k in live:
            for v in faces[k]:
                users.setdefault(v, []).append(k)
        group = {live[int(bad[0])]}
        done = False
        for _ring in range(most_rings):
            group |= {f for k in group for v in faces[k] for f in users[v]}
            halves = {(faces[k][i], faces[k][(i + 1) % 4]) for k in group for i in range(4)}
            following = {a: b for a, b in halves if (b, a) not in halves}
            border = [a for a, b in halves if (b, a) not in halves]
            if not border or len(following) != len(border) or len(border) % 2:
                continue  # touches itself, closed, or an odd outline: take more
            ring = [border[0]]
            while following[ring[-1]] != border[0] and len(ring) <= len(border):
                ring.append(following[ring[-1]])
            if len(ring) != len(border):
                continue
            middle = len(points)
            center = points[ring].mean(axis=0)
            trial = np.vstack((points, center[None]))
            for first in (0, 1):
                cells = [(middle, ring[(first + 2 * i) % len(ring)], ring[(first + 2 * i + 1) % len(ring)], ring[(first + 2 * i + 2) % len(ring)]) for i in range(len(ring) // 2)]
                if len(cells) <= len(group) and (quad_quality(trial[np.asarray(cells)]) > 0.0).all():
                    extra.append(center[None])
                    slots = sorted(group)
                    for k in slots:
                        faces[k] = None
                    for k, cell in zip(slots, cells):
                        faces[k] = cell
                    done = True
                    break
            if done:
                break
        if not done:
            break
    return np.vstack([moved, *extra]) if extra else moved


def _ease(moved: np.ndarray, faces: list, find_nearest, rounds: int, wider: int) -> None:
    """Move corners of folded quads between their neighbours, in place: `rounds` staying on the source, then `rounds` not, over wider rings."""

    quads = np.asarray([f for f in faces if f is not None], dtype=np.int64)
    if not len(quads):
        return
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
            for i in ids.tolist():
                found = find_nearest(moved[i].tolist())
                if found[0] is not None:
                    moved[i] = found[0]


def unfold(points: np.ndarray, faces, find_nearest, rounds: int = 6, cycles: int = 6) -> tuple[np.ndarray, list[tuple[int, ...]], list[int]]:
    """Leave no quad folded. Returns (points, faces, source index per face).

    Easing costs a little shape on the first rounds and a little more later. What it can't open is closed along a
    diagonal, which leaves one vertex fewer. Faces that aren't quads pass through.
    """

    moved = np.array(points, dtype=np.float64)
    all_faces = [tuple(f) for f in faces]
    working: list = [f if len(f) == 4 else None for f in all_faces]
    array = np.asarray([f for f in working if f is not None], dtype=np.int64)
    if not len(array) or not (quad_quality(moved[array]) <= 0.0).any():
        return moved, all_faces, list(range(len(all_faces)))
    for cycle in range(cycles):
        _ease(moved, working, find_nearest, rounds, 1 + cycle)
        array = np.asarray([f for f in working if f is not None], dtype=np.int64)
        if not len(array) or not (quad_quality(moved[array]) <= 0.0).any():
            break
        _collapse_folded(moved, working)
        _dissolve_doublets(working)
    moved = _refill_folded(moved, working)
    live = [f for f in working if f is not None]
    corners = Counter(tuple(sorted(f)) for f in live)
    # Closing can leave two quads back to back on the same corners, touching nothing else.
    keep = [k for k, f in enumerate(all_faces) if (len(f) != 4 or (working[k] is not None and corners[tuple(sorted(working[k]))] == 1))]
    final = [all_faces[k] if len(all_faces[k]) != 4 else working[k] for k in keep]
    alive = sorted({v for f in final for v in f})
    remap = {v: i for i, v in enumerate(alive)}
    return moved[alive], [tuple(remap[v] for v in f) for f in final], keep


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
