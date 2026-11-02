# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Clean up a finished all-quad mesh with small local patterns, without adding a vertex.

Three moves, the ones in Tarini et al. 2010 ("Practical quad mesh simplification"):

- turn: two quads sharing an edge are a hexagon, which can be split three ways. Pick the best way.
- collapse: merge two opposite corners of a quad (3-x-3-x becomes three regular vertices).
- doublet: a vertex in only two quads goes away and the two quads become one.

A move happens only if the corners get cheaper (`corner_cost`) and the new quads still lie on the old ones.
Faces that aren't quads and borders stay as they are. Plain Python on purpose: a move touches a handful of faces.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import MeshData


@dataclass(slots=True)
class TidyOptions:
    passes: int = 4
    flat_share: float = 0.03  # a hexagon may leave its best-fit plane by this share of its size
    turn_deg: float = 15.0  # a collapse may turn a neighbouring quad by this much
    fold_deg: float = 25.0  # a new quad may fold along a diagonal this much
    min_corner_deg: float = 25.0
    max_corner_deg: float = 165.0
    collapse: bool = True


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _normal(p, q, r, s):
    """Unit normal of a quad from its diagonals, or None if it has no area."""

    n = _cross(_sub(r, p), _sub(s, q))
    size = _len(n)
    return (n[0] / size, n[1] / size, n[2] / size) if size > 1e-30 else None


def _shape(p, q, r, s, cos_low: float, cos_high: float, cos_fold: float):
    """The quad's normal if it is convex enough and not folded, else None."""

    points = (p, q, r, s)
    n = _normal(p, q, r, s)
    if n is None:
        return None
    for k in range(4):
        here, out, back = points[k], points[(k + 1) % 4], points[k - 1]
        u, v = _sub(out, here), _sub(back, here)
        lu, lv = _len(u), _len(v)
        if lu < 1e-30 or lv < 1e-30:
            return None
        cosine = _dot(u, v) / (lu * lv)
        if cosine > cos_low or cosine < cos_high:  # too sharp or too open
            return None
        if _dot(_cross(u, v), n) <= 0.0:  # a dent
            return None
    for a, b, c in ((p, q, r), (q, r, s), (r, s, p), (s, p, q)):
        t = _cross(_sub(b, a), _sub(c, a))
        lt = _len(t)
        if lt < 1e-30 or _dot(t, n) / lt < cos_fold:
            return None
    return n


def _corner(valence: int, border: bool, faces: int) -> float:
    """Default cost of a corner: squared distance from a regular vertex, where a straight border counts as regular."""

    if border:
        return 0.0 if faces <= 2 else float((faces - 2) ** 2)
    return float((valence - 4) ** 2)


def tidy(mesh: MeshData, options: TidyOptions | None = None, corner_cost=None) -> tuple[MeshData, dict]:
    """(cleaned mesh, how many of each move). `corner_cost(valence, border, faces)` replaces the default cost."""

    options = options or TidyOptions()
    cost = corner_cost or _corner
    points = [tuple(p) for p in mesh.vertices.tolist()]
    faces: list[list[int] | None] = [list(f) for f in mesh.faces]
    kinds = list(mesh.face_materials)
    users: list[set[int]] = [set() for _ in points]
    for index, face in enumerate(faces):
        for v in face:
            users[v].add(index)
    cos_low, cos_high = math.cos(math.radians(options.min_corner_deg)), math.cos(math.radians(options.max_corner_deg))
    cos_fold, cos_turn = math.cos(math.radians(options.fold_deg)), math.cos(math.radians(options.turn_deg))

    def neighbours(v: int) -> set[int]:
        out: set[int] = set()
        for f in users[v]:
            face = faces[f]
            k = face.index(v)
            out.add(face[(k + 1) % len(face)])
            out.add(face[k - 1])
        return out

    def is_border(v: int, near: set[int]) -> bool:
        return len(near) != len(users[v])

    def corner(v: int, change: int = 0) -> float:
        near = neighbours(v)
        return cost(len(near) + change, is_border(v, near), len(users[v]) + change)

    def shape_ok(ids) -> object:
        return _shape(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], cos_low, cos_high, cos_fold)

    def quality(ids) -> float:
        worst = 2.0
        for k in range(4):
            here = points[ids[k]]
            u, v = _sub(points[ids[(k + 1) % 4]], here), _sub(points[ids[k - 1]], here)
            worst = min(worst, 1.0 - abs(_dot(u, v)) / max(_len(u) * _len(v), 1e-30))
        return worst

    def replace(index: int, ids: list[int]) -> None:
        for v in faces[index]:
            users[v].discard(index)
        faces[index] = ids
        for v in ids:
            users[v].add(index)

    def try_turn(f: int, k: int) -> bool:
        face = faces[f]
        u, v = face[k], face[(k + 1) % 4]
        shared = users[u] & users[v]
        if len(shared) != 2:
            return False
        shared.discard(f)
        g = shared.pop()
        other = faces[g]
        if len(other) != 4 or kinds[f] != kinds[g]:
            return False
        at = other.index(v)
        if other[(at + 1) % 4] != u:  # opposite windings: not a clean edge
            return False
        hexagon = [v, face[(k + 2) % 4], face[(k + 3) % 4], u, other[(at + 2) % 4], other[(at + 3) % 4]]
        if len(set(hexagon)) != 6:
            return False
        ends = (corner(u, -1), corner(v, -1))
        if is_border(u, neighbours(u)) or is_border(v, neighbours(v)):
            return False
        if len(neighbours(u)) <= 3 or len(neighbours(v)) <= 3:
            return False
        now = corner(u) + corner(v)
        best, best_gain = None, 1e-9
        old_quality = min(quality(face), quality(other))
        for shift in (1, 2):
            a, b = hexagon[shift], hexagon[shift + 3]
            if b in neighbours(a):
                continue
            gain = now + corner(a) + corner(b) - ends[0] - ends[1] - corner(a, 1) - corner(b, 1)
            first = [hexagon[(shift + i) % 6] for i in range(4)]
            second = [hexagon[(shift + 3 + i) % 6] for i in range(4)]
            if gain < -1e-9:
                continue
            new_quality = min(quality(first), quality(second))
            if gain <= 1e-9:  # same corners: only worth it for clearly better shape
                gain = 0.25 * max(new_quality - old_quality - 0.1, 0.0)
                if gain <= 0.0:
                    continue
            elif new_quality < 0.6 * old_quality:
                continue
            if gain > best_gain:
                best, best_gain = (first, second), gain
        if best is None:
            return False
        ring = np.asarray([points[i] for i in hexagon])
        centre = ring.mean(axis=0)
        normal = np.linalg.eigh((ring - centre).T @ (ring - centre))[1][:, 0]
        size = float(np.linalg.norm(np.ptp(ring, axis=0)))
        if float(np.abs((ring - centre) @ normal).max()) > options.flat_share * size:
            return False
        up = _normal(*(points[i] for i in face))
        for ids in best:
            n = shape_ok(ids)
            if n is None or up is None or _dot(n, up) < cos_turn:
                return False
        replace(f, best[0])
        replace(g, best[1])
        return True

    def try_collapse(f: int, k: int) -> bool:
        face = faces[f]
        a, b, c, d = (face[(k + i) % 4] for i in range(4))
        if users[a] & users[c] != {f}:
            return False
        near_a, near_c = neighbours(a), neighbours(c)
        if near_a & near_c != {b, d} or is_border(a, near_a) or is_border(c, near_c):
            return False
        near_b, near_d = neighbours(b), neighbours(d)
        if any(len(near) <= 3 for near in (near_b, near_d)) or is_border(b, near_b) or is_border(d, near_d):
            return False
        merged = len(near_a) + len(near_c) - 2
        before = corner(a) + corner(c) + corner(b) + corner(d)
        after = cost(merged, False, merged) + corner(b, -1) + corner(d, -1)
        if after >= before - 1e-9:
            return False
        around = sorted((users[a] | users[c]) - {f})
        if any(kinds[g] != kinds[f] or len(faces[g]) != 4 for g in around):
            return False
        pa, pc = points[a], points[c]
        middle = ((pa[0] + pc[0]) / 2.0, (pa[1] + pc[1]) / 2.0, (pa[2] + pc[2]) / 2.0)
        olds = [_normal(*(points[v] for v in faces[g])) for g in around]
        if any(n is None for n in olds):
            return False
        for place in (middle, pa, pc):  # the first spot that keeps every neighbour on its old surface
            saved = points[a]
            points[a] = place
            fine = True
            for g, old in zip(around, olds):
                ids = [a if v == c else v for v in faces[g]]
                n = shape_ok(ids)
                if n is None or _dot(n, old) < cos_turn:
                    fine = False
                    break
            if fine:
                break
            points[a] = saved
        else:
            return False
        for g in around:
            replace(g, [a if v == c else v for v in faces[g]])
        for v in face:
            users[v].discard(f)
        faces[f] = None
        return True

    def try_doublet(v: int) -> bool:
        if len(users[v]) != 2:
            return False
        f, g = users[v]
        first, second = faces[f], faces[g]
        if len(first) != 4 or len(second) != 4 or kinds[f] != kinds[g] or len(neighbours(v)) != 2:
            return False
        k, j = first.index(v), second.index(v)
        ring = [first[(k + 1) % 4], first[(k + 2) % 4], first[(k + 3) % 4], second[(j + 2) % 4]]
        if len(set(ring)) != 4 or second[(j + 1) % 4] != first[k - 1]:
            return False
        if shape_ok(ring) is None:
            return False
        replace(f, ring)
        for u in second:
            users[u].discard(g)
        faces[g] = None
        return True

    counts = {"turns": 0, "collapses": 0, "doublets": 0}
    for _pass in range(options.passes):
        moved = 0
        for v in range(len(points)):
            if len(users[v]) == 2 and try_doublet(v):
                counts["doublets"] += 1
                moved += 1
        for f in range(len(faces)):
            if faces[f] is None or len(faces[f]) != 4:
                continue
            if options.collapse:
                for k in (0, 1):
                    if try_collapse(f, k):
                        counts["collapses"] += 1
                        moved += 1
                        break
                if faces[f] is None:
                    continue
            for k in range(4):
                if try_turn(f, k):
                    counts["turns"] += 1
                    moved += 1
                    break
        if not moved:
            break
    alive = [i for i, face in enumerate(faces) if face is not None]
    used = sorted({v for i in alive for v in faces[i]})
    remap = {v: i for i, v in enumerate(used)}
    out = MeshData(
        vertices=np.asarray([points[v] for v in used], dtype=np.float64).reshape(-1, 3),
        faces=[tuple(remap[v] for v in faces[i]) for i in alive],
        face_materials=[kinds[i] for i in alive],
        name=mesh.name,
    )
    return out, counts


__all__ = ["TidyOptions", "tidy"]
