# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Smooth Grids: rounded areas come out as clean rows of quads.

Reducing and pairing triangles leaves a dome or a dished disc with scattered, slanted quads. This pass finds such areas
on the source: smooth doubly curved ones by how the normals spread (a dome from a cylinder or a plate, sign changes tell
it from rock), then whatever else has creases all around. It finds the same areas on the reduced mesh and rebuilds each
disk as a four-sided grid with concentric rows around it, and each ring as concentric rows. Every border vertex stays, or
gets a neighbour on the source where the border cuts a curve short, so neighbours keep conforming. New vertices land on
the source. The caller's camera check judges the whole result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .flatquad import _edge_faces, _face_frames
from .patch import (
    _Locator,
    _arc_parameters,
    _corner_angle_sums,
    _half_edges,
    _euler,
    _labels,
    boundary_loops,
    circle_boundary,
    square_boundary,
    coons_grid,
    face_adjacency,
    fit_to_surface,
    mask_components,
    quad_quality,
    tutte_embedding,
)
from .types import MeshData


@dataclass(slots=True)
class GridOptions:
    sharp_deg: float = 35.0  # an edge bent more than this ends a smooth area
    min_spread: float = 2e-4  # smallest eigenvalue of the normals around a face, as a share: 0 on a cylinder
    min_total_deg: float = 20.0  # total curvature an area needs, so nearly flat noise stays as it is
    min_agreement: float = 0.7  # edge length bending one way, minus the other way, as a share: rock fails this
    min_source_faces: int = 20
    min_border: int = 6  # border vertices a reduced area needs before a grid fits inside
    turned_deg: float = 20.0  # same as the camera's: turned further than this from the source, it looks different
    moved: float = 0.0115  # two of the camera's pixels, as a share of the object's size
    max_turned: float = 0.005  # a rebuilt area may have this much more of its surface turned than the triangles had
    band_skew: int = 4  # a band this far from equal counts has nothing to gain from rebuilding
    flat_turn: float = 5.0  # a plate turning less than this is flat, and needs no quads inside
    min_kept: float = 0.5  # a curved plate's grid must keep this share of the quads its triangles would pair into
    min_compact: float = 0.3  # area against border length: a thin bent strip is left to pairing
    plate_turn: float = 30.0  # a long plate that turns less than this still gets a grid
    corner_deg: float = 50.0  # a border vertex turning more than this is a plate's corner
    min_faces: int = 8
    min_size: float = 0.02  # share of the whole object's size below which an area isn't worth rebuilding
    max_bend: float = 10.0  # median degrees between neighbouring faces of a crease-bounded area: rougher is rock
    relax_rounds: int = 8
    max_ratio: float = 1.6  # how fast rows may lose vertices from one to the next
    max_rows: int = 12
    turn_per_row: float = 15.0  # degrees of surface turning one row of a disk's middle may cover
    row_sag: float = 0.001  # rows stop being added once a straight row strays this share of the object's size from the source
    rim_turn: float = 12.0  # a smooth border turning more than this at each vertex gets vertices between them
    rim_rounds: int = 2
    rim_size: float = 0.08  # only areas this big a share of the object get them: small parts are not worth the faces
    spike_deg: float = 100.0  # a border corner sharper than this gets tidied away
    plain_quality: float = 0.25  # a grid without rows around it must be this good, or a row goes on
    max_added: float = 0.03  # all grids together may add this share of the mesh's quads
    max_error: float = 1.5  # a grid may stray from the source this many times what the reduced triangles did
    max_sag: float = 0.02  # or this share of the area's size, which no camera at this density would notice


@dataclass(slots=True)
class GridReport:
    built: int = 0
    faces_in: int = 0
    quads_out: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    loops: list[int] = field(default_factory=list)


def smooth_areas(vertices: np.ndarray, faces, materials=None, options: GridOptions | None = None) -> np.ndarray:
    """Area id per source face, -1 where the surface isn't a smooth doubly curved area."""

    options = options or GridOptions()
    count = len(vertices)
    normals, areas = _face_frames(vertices, faces)
    face_of, a, b = _half_edges(faces)
    low, high = np.minimum(a, b), np.maximum(a, b)
    key = low * (count + 1) + high
    order = np.argsort(key, kind="stable")
    uniq, start, uses = np.unique(key[order], return_index=True, return_counts=True)
    two = uses == 2
    fa, fb = face_of[order][start[two]], face_of[order][start[two] + 1]
    va, vb = low[order][start[two]], high[order][start[two]]
    centre = np.stack([np.bincount(face_of, vertices[a, axis], len(faces)) for axis in range(3)], axis=1)
    centre /= np.bincount(face_of, minlength=len(faces))[:, None]
    bend = np.arctan2(np.linalg.norm(np.cross(normals[fa], normals[fb]), axis=1), np.einsum("ij,ij->i", normals[fa], normals[fb]))
    convex = np.einsum("ij,ij->i", centre[fb] - centre[fa], normals[fa] - normals[fb]) >= 0.0
    signed = np.where(convex, bend, -bend)
    lengths = np.linalg.norm(vertices[va] - vertices[vb], axis=1)
    sharp = bend > math.radians(options.sharp_deg)
    if materials is not None:
        kinds = np.unique(np.asarray(materials), return_inverse=True)[1]
        sharp |= kinds[fa] != kinds[fb]

    triangles = np.asarray([(face[0], face[k], face[k + 1]) for face in faces for k in range(1, len(face) - 1)], dtype=np.int64).reshape(-1, 3)
    defect = 2.0 * math.pi - np.radians(_corner_angle_sums(vertices, triangles))
    soft = ~sharp

    # Normals around a face: one direction on a plate, a great circle on a cylinder, a patch of the sphere on a dome.
    # A neighbour counts by its size relative to the face, squared, so a small dome face at the end of a long cylinder face barely counts.
    tensor = areas[:, None, None] * np.einsum("ij,ik->ijk", normals, normals)
    for here, there in ((fa[soft], fb[soft]), (fb[soft], fa[soft])):
        weight = np.minimum(areas[here], areas[there]) ** 2 / np.maximum(areas[here], 1e-30)
        np.add.at(tensor, here, weight[:, None, None] * np.einsum("ij,ik->ijk", normals[there], normals[there]))
    spread = np.linalg.eigvalsh(tensor)
    mask = spread[:, 0] > options.min_spread * np.maximum(spread.sum(axis=1), 1e-30)
    support = np.zeros(len(faces))
    np.add.at(support, fa[soft], mask[fb[soft]])
    np.add.at(support, fb[soft], mask[fa[soft]])
    mask |= support >= 2  # close notches in the outline

    keep = soft & mask[fa] & mask[fb]
    labels = _labels(len(faces), fa[keep], fb[keep])
    labels[~mask] = -1
    out = np.full(len(faces), -1, dtype=np.int64)
    ids, inverse, members = np.unique(labels, return_inverse=True, return_counts=True)
    total_curvature = np.zeros(len(ids))
    faces_at = np.maximum(np.bincount(a, minlength=count), 1)
    np.add.at(total_curvature, inverse[face_of], (np.maximum(defect, 0.0) / faces_at)[a])  # a vertex shares out its defect
    pull = np.zeros(len(ids))
    span = np.zeros(len(ids))
    inner = keep
    np.add.at(pull, inverse[fa[inner]], (signed * lengths)[inner])
    np.add.at(span, inverse[fa[inner]], (bend * lengths)[inner])
    next_id = 0
    for row, label in enumerate(ids.tolist()):
        if label < 0 or members[row] < options.min_source_faces:
            continue
        if total_curvature[row] < math.radians(options.min_total_deg):
            continue
        if abs(pull[row]) < options.min_agreement * max(span[row], 1e-30):
            continue
        out[inverse == row] = next_id
        next_id += 1
    return out


def crease_areas(vertices: np.ndarray, faces, skip: np.ndarray, options: GridOptions | None = None) -> np.ndarray:
    """Area id per source face: parts of the surface with creases all around, not rough, not in `skip`."""

    options = options or GridOptions()
    normals, _areas = _face_frames(vertices, faces)
    fa, fb = face_adjacency(faces)
    bend = np.degrees(np.arctan2(np.linalg.norm(np.cross(normals[fa], normals[fb]), axis=1), np.einsum("ij,ij->i", normals[fa], normals[fb])))
    soft = (bend <= options.sharp_deg) & ~skip[fa] & ~skip[fb]
    labels = _labels(len(faces), fa[soft], fb[soft])
    labels[skip] = -1
    ids, inverse, members = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(inverse[fa[soft]], kind="stable")
    held = inverse[fa[soft]][order]
    bends = bend[soft][order]
    out = np.full(len(faces), -1, dtype=np.int64)
    next_id = 0
    for row, label in enumerate(ids.tolist()):
        if label < 0 or members[row] < options.min_source_faces:
            continue
        inner = bends[np.searchsorted(held, row) : np.searchsorted(held, row, side="right")]
        if len(inner) and float(np.median(inner)) > options.max_bend:
            continue
        out[inverse == row] = next_id
        next_id += 1
    return out


def points_of(vertices: list, ids) -> np.ndarray:
    return np.asarray([vertices[int(v)] for v in ids], dtype=np.float64)


def _tidy_border(points: np.ndarray, faces: list, mask: np.ndarray, free: np.ndarray, links: list[list[int]], sharp_deg: float, rounds: int = 6) -> np.ndarray:
    """Take off spikes and fill notches: a triangle with two border edges meeting at a sharp angle changes sides.

    Reduced triangles don't follow an area's outline exactly, and a zigzag border folds the quads next to it.
    `free` marks the faces no other area wants.
    """

    limit = math.cos(math.radians(sharp_deg))
    mask = mask.copy()
    for _ in range(rounds):
        candidates = set(np.flatnonzero(mask).tolist())
        candidates.update(n for f in list(candidates) for n in links[f])
        flips = []
        for face in sorted(candidates):
            polygon = faces[face]
            if len(polygon) != 3:
                continue
            inside = [n for n in links[face] if mask[n]]
            if mask[face] and len(inside) == 1:
                apex = next(v for v in polygon if v not in faces[inside[0]])
            elif not mask[face] and len(inside) == 3:
                flips.append(face)
                continue
            elif not mask[face] and free[face] and len(inside) == 2:
                common = set(faces[inside[0]]) & set(faces[inside[1]]) & set(polygon)
                if len(common) != 1:
                    continue
                apex = common.pop()
            else:
                continue
            at = polygon.index(apex)
            u = points[polygon[(at + 1) % 3]] - points[apex]
            w = points[polygon[(at + 2) % 3]] - points[apex]
            if float(u @ w) > limit * float(np.linalg.norm(u) * np.linalg.norm(w)):
                flips.append(face)
        if not flips:
            break
        mask[flips] = ~mask[flips]
    return mask


class _Splits:
    """The mesh tables a border split changes, with an undo: an area that fails leaves nothing behind."""

    def __init__(self, vertices: list, faces: list, area_list: list, materials: list, members: dict, users: dict):
        self.vertices, self.faces, self.area_list, self.materials, self.members, self.users = vertices, faces, area_list, materials, members, users
        self.log: list[tuple] = []

    def _save(self, key: tuple[int, int]) -> None:
        here = self.users.get(key)
        self.log.append(("edge", key, None if here is None else list(here)))

    def split(self, v: int, w: int, point: np.ndarray) -> int:
        """Put a new vertex on an edge; each triangle on it becomes two, so the mesh stays conforming. All faces must be triangles."""

        vertices, faces, area_list, users = self.vertices, self.faces, self.area_list, self.users
        new = len(vertices)
        self.log.append(("vertex",))
        vertices.append(point)
        edge = (v, w) if v < w else (w, v)
        self._save(edge)
        for face in users.pop(edge):
            tri = faces[face]
            at = next(i for i in range(3) if {tri[i], tri[(i + 1) % 3]} == {v, w})
            p, q, r = tri[at], tri[(at + 1) % 3], tri[(at + 2) % 3]
            added = len(faces)
            self.log.append(("face", face, tri))
            faces[face] = (p, new, r)
            faces.append((new, q, r))
            area_list.append(area_list[face])
            self.materials.append(self.materials[face])
            if area_list[face] >= 0:
                self.members[area_list[face]].append(added)
            pair = (q, r) if q < r else (r, q)
            for key in ((p, new), (q, new), (r, new), pair):
                self._save((key[0], key[1]) if key[0] < key[1] else (key[1], key[0]))
            users.setdefault((p, new) if p < new else (new, p), []).append(face)
            users.setdefault((q, new) if q < new else (new, q), []).append(added)
            users.setdefault((r, new) if r < new else (new, r), []).extend((face, added))
            users[pair][users[pair].index(face)] = added
        return new

    def undo(self) -> None:
        for entry in reversed(self.log):
            if entry[0] == "vertex":
                self.vertices.pop()
            elif entry[0] == "face":
                _kind, face, tri = entry
                added = len(self.faces) - 1
                if self.area_list[added] >= 0:
                    self.members[self.area_list[added]].pop()
                self.faces.pop()
                self.area_list.pop()
                self.materials.pop()
                self.faces[face] = tri
            elif entry[2] is None:
                self.users.pop(entry[1], None)
            else:
                self.users[entry[1]] = entry[2]
        self.log.clear()


def _split_border(splits: _Splits, done: set, loop: list[int], project, only: range | None = None) -> list[int] | None:
    """The border with one vertex more, on its longest edge that can take one (of those `only` allows). A built area's border can't change."""

    points = points_of(splits.vertices, loop)
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    for k in np.argsort(-lengths).tolist():
        if only is not None and k not in only:
            continue
        v, w = loop[k], loop[(k + 1) % len(loop)]
        here = splits.users[(v, w) if v < w else (w, v)]
        if len(here) > 2 or any(len(splits.faces[f]) != 3 or splits.area_list[f] in done for f in here):
            continue
        new = splits.split(v, w, project(0.5 * (splits.vertices[v] + splits.vertices[w])[None])[0])
        return loop[: k + 1] + [new] + loop[k + 1 :]
    return None


def _four_corners(points: np.ndarray, corner_deg: float) -> list[int] | None:
    """Positions of a loop's corners, if it has exactly four. Nothing else may turn as much."""

    before, after = points - np.roll(points, 1, axis=0), np.roll(points, -1, axis=0) - points
    turn = np.degrees(np.arctan2(np.linalg.norm(np.cross(before, after), axis=1), np.einsum("ij,ij->i", before, after)))
    corners = np.flatnonzero(turn > corner_deg)
    return corners.tolist() if len(corners) == 4 else None


def _row_counts(dense: int, sparse: int, rows: int, ratio: float) -> list[int]:
    """Vertices per row from one loop to another, always the same parity, falling evenly. Rows are added until each step fits `ratio`."""

    rows = max(rows, 1, math.ceil(math.log(max(dense / max(sparse, 1), 1.0)) / math.log(ratio)))
    counts = [dense]
    for r in range(1, rows):
        wanted = dense * (sparse / dense) ** (r / rows)
        counts.append(max(sparse, min(counts[-1], int(round((wanted - dense) / 2.0)) * 2 + dense)))
    return counts + [sparse]


def _strip(first: np.ndarray, second: np.ndarray, next_id: int, first_at=None, second_at=None) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    """Quads between two vertex rows running the same way, the first with at least as many, both starting side by side.

    A step takes one edge from each row, or three from the first and one from the second. That second kind is what
    closes rows off: four quads around two new vertices, none with a flat corner, as a quad over two border edges in a
    straight line would have. `first_at`, `second_at` say how far around (0 to 1) each vertex is, so steps go wherever
    they keep the rungs straight; without them, evenly. Returns (quads, for each new vertex the two whose middle it
    starts at).
    """

    big, small = len(first), len(second)
    shrink = (big - small) // 2
    first_at = np.arange(big) / big if first_at is None else first_at
    second_at = np.arange(small) / small if second_at is None else second_at

    def along(k: int) -> float:
        return float(first_at[k % big]) + k // big

    quads, extra, i = [], [], 0
    for j in range(small):
        o0, o1 = int(first[i % big]), int(first[(i + 1) % big])
        i0, i1 = int(second[j % small]), int(second[(j + 1) % small])
        target = float(second_at[(j + 1) % small]) + (j + 1) // small
        if shrink >= small - j or (shrink > 0 and abs(along(i + 3) - target) < abs(along(i + 1) - target)):
            shrink -= 1
            o2, o3 = int(first[(i + 2) % big]), int(first[(i + 3) % big])
            left, right = next_id + len(extra), next_id + len(extra) + 1
            extra += [(o1, i0), (o2, i1)]
            quads += [(o0, o1, left, i0), (o1, o2, right, left), (o2, o3, i1, right), (left, right, i1, i0)]
            i += 3
        else:
            quads.append((o0, o1, i1, i0))
            i += 1
    return quads, extra


def _around(points: np.ndarray, centre: np.ndarray, u: np.ndarray, v: np.ndarray, zero: float | None = None) -> tuple[np.ndarray, float]:
    """How far around (0 to 1, rising) each loop vertex is from its first, by angle if the loop winds simply, else by length.

    Also the first vertex's angle. A loop running the other way round the axis still rises, so both loops of a ring must
    be given running the same way.
    """

    relative = points - centre
    angle = np.arctan2(relative @ v, relative @ u)
    first = float(angle[0]) if zero is None else zero
    steps = (np.diff(np.concatenate((angle, angle[:1]))) + math.pi) % (2.0 * math.pi) - math.pi
    total = float(steps.sum())
    if abs(abs(total) - 2.0 * math.pi) > 1e-6 or (steps * math.copysign(1.0, total) <= 0.0).any():
        return _arc_parameters(points), first
    return np.concatenate(([0.0], np.cumsum(steps)[:-1])) / total, first


def _at(points: np.ndarray, phase: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """A closed loop's positions at these `phase` values."""

    closed = np.vstack((points, points[:1]))
    knots = np.concatenate((phase, [1.0]))
    return np.stack([np.interp(wanted % 1.0, knots, closed[:, axis]) for axis in range(3)], axis=1)


def ring_fill(dense_xyz: np.ndarray, sparse_xyz: np.ndarray, project, options: GridOptions, tolerance: float):
    """Concentric rows between two loops running the same way, the first with at least as many vertices.

    Returns (xyz of both loops + new vertices, quads, turn); the second loop is rotated by `turn` to line up, and its
    seam ids must be rotated the same way. Rungs follow a common angle round the ring, so rows don't spiral. Rows come
    from how far the straight line between the loops strays from the surface and how fast the count has to fall, so a
    flat band stays one row of long quads.
    """

    dense, sparse = len(dense_xyz), len(sparse_xyz)
    centre = 0.5 * (dense_xyz.mean(axis=0) + sparse_xyz.mean(axis=0))
    axis = np.linalg.eigh(np.cov((dense_xyz - centre).T))[1]
    u, v = axis[:, 2], axis[:, 1]
    start = float(np.arctan2((dense_xyz[0] - centre) @ v, (dense_xyz[0] - centre) @ u))
    offsets = np.arctan2((sparse_xyz - centre) @ v, (sparse_xyz - centre) @ u) - start
    turn = int(np.argmin(np.abs((offsets + math.pi) % (2.0 * math.pi) - math.pi)))
    sparse_xyz = np.roll(sparse_xyz, -turn, axis=0)
    dense_at = _around(dense_xyz, centre, u, v)[0]
    sparse_at = _around(sparse_xyz, centre, u, v)[0]
    probes = np.arange(sparse) / sparse
    middle = 0.5 * (_at(dense_xyz, dense_at, probes) + _at(sparse_xyz, sparse_at, probes))
    sag = float(np.linalg.norm(middle - project(middle), axis=1).max())
    counts = _row_counts(dense, sparse, min(options.max_rows, math.ceil(math.sqrt(sag / max(tolerance, 1e-30)))), options.max_ratio)
    rows = len(counts) - 1
    table = [np.arange(dense)]
    phases = [dense_at]
    extra = []
    at = dense + sparse
    for r in range(1, rows):
        count = counts[r]
        t = r / rows
        wanted = np.arange(count) / count
        extra.append((1.0 - t) * _at(dense_xyz, dense_at, wanted) + t * _at(sparse_xyz, sparse_at, wanted))
        table.append(at + np.arange(count))
        phases.append(wanted)
        at += count
    table.append(dense + np.arange(sparse))
    phases.append(sparse_at)
    xyz = np.vstack([dense_xyz, sparse_xyz] + extra)
    quads, middles = [], []
    for r in range(rows):
        strip, added = _strip(table[r], table[r + 1], at + len(middles), phases[r], phases[r + 1])
        quads += strip
        middles += added
    if middles:
        pairs = np.asarray(middles, dtype=np.int64)
        xyz = np.vstack((xyz, 0.5 * (xyz[pairs[:, 0]] + xyz[pairs[:, 1]])))
    if len(xyz) == dense + sparse:
        return xyz, quads, turn
    xyz[dense + sparse :] = project(xyz[dense + sparse :])
    pinned = np.zeros(len(xyz), dtype=bool)
    pinned[: dense + sparse] = True
    return fit_to_surface(xyz, quads, pinned, project, iterations=options.relax_rounds), quads, turn


def cap_fill(loop_xyz: np.ndarray, tris: np.ndarray, points: np.ndarray, border: np.ndarray, project, inner_count: int | None = None, relax_rounds: int = 8, ratio: float = 1.6, corners=None):
    """A four-sided grid over a disk, with rows around it down to `inner_count` border vertices: (xyz of loop + new vertices, quads).

    None means no rows at all. With `corners` (a plate's own four, opposite sides equal) the disk flattens onto a square
    instead, and the grid runs straight from side to side. `points`, `tris` are the disk as it stands, with `border` its loop in the same order as
    `loop_xyz`. The disk is flattened onto a circle (Tutte), so nothing can fold however the disk curves, and new
    vertices are lifted back through it, then relaxed on the surface. Without rows, the grid's corners flatten to 180
    degrees on a round border; with them, they become four poles just inside.
    """

    count = len(loop_xyz)
    circle = square_boundary(loop_xyz, corners) if corners else circle_boundary(loop_xyz)
    uv = tutte_embedding(len(points), tris, border, circle, points)
    grid_loop = inner_count or count
    first = grid_loop // 4
    second = grid_loop // 2 - first
    counts = [count] if inner_count is None else _row_counts(count, inner_count, 1, ratio)
    rows = len(counts) - 1
    table = [np.arange(count)]
    phases = [_arc_parameters(loop_xyz)]
    at = count
    new_uv = []
    shrink = 1.0 / (rows + 0.5 * first + 0.5)  # rows and grid cells about as wide as each other
    for r in range(1, rows + 1):
        angle = 2.0 * np.pi * np.arange(counts[r]) / counts[r]
        new_uv.append((1.0 - r * shrink) * np.stack((np.cos(angle), np.sin(angle)), axis=1))
        table.append(at + np.arange(counts[r]))
        phases.append(np.arange(counts[r]) / counts[r])
        at += counts[r]
    quads, middles = [], []
    for r in range(rows):
        strip, added = _strip(table[r], table[r + 1], at + len(middles), phases[r], phases[r + 1])
        quads += strip
        middles += added
    if middles:
        known = np.concatenate([circle] + new_uv)
        pairs = np.asarray(middles, dtype=np.int64)
        row_uv = new_uv
        new_uv = row_uv + [0.5 * (known[pairs[:, 0]] + known[pairs[:, 1]])]
        at += len(middles)
    if rows:
        inner = (new_uv[rows - 1])
        corners = (0, first, first + second, 2 * first + second)
    elif corners:
        inner = circle
    else:  # the grid sits on the border itself: corners go where the sides come out closest to equal
        steps = np.linalg.norm(np.roll(loop_xyz, -1, axis=0) - loop_xyz, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(steps)))

        def run(start: int, length: int) -> float:
            start %= count
            end = start + length
            return cumulative[end % count] + cumulative[-1] * (end // count) - cumulative[start]

        offset = min(range(first + second), key=lambda s: float(np.ptp([run(s, first), run(s + first, second), run(s + first + second, first), run(s + 2 * first + second, second)])))
        inner = circle
        corners = (offset, (offset + first) % count, (offset + first + second) % count, (offset + 2 * first + second) % count)
    nodes, ids = coons_grid(inner, corners)
    free = ids < 0
    ids[~free] = table[-1][ids[~free]]
    ids[free] = at + np.arange(int(free.sum()))
    new_uv.append(nodes[free])
    for i in range(ids.shape[0] - 1):
        for j in range(ids.shape[1] - 1):
            quads.append((int(ids[i, j]), int(ids[i + 1, j]), int(ids[i + 1, j + 1]), int(ids[i, j + 1])))
    new_uv = np.concatenate(new_uv)
    if not len(new_uv):
        return loop_xyz, quads
    tri, weights = _Locator(uv, tris).locate(new_uv)
    xyz = np.vstack((loop_xyz, project(np.einsum("ij,ijk->ik", weights, points[tris[tri]]))))
    pinned = np.zeros(len(xyz), dtype=bool)
    pinned[:count] = True
    return fit_to_surface(xyz, quads, pinned, project, iterations=relax_rounds), quads


def grid_areas(coarse: MeshData, labels: np.ndarray, nearest, options: GridOptions | None = None, project_for=None, extent: float | None = None, normals: np.ndarray | None = None) -> tuple[MeshData, GridReport]:
    """Rebuild the reduced mesh's smooth areas as organized rows. Everything else passes through untouched.

    `labels` comes from `smooth_areas` or `crease_areas`; `nearest(points)` gives the closest source face and point per
    point. `project_for(area)`, if given, moves points onto that area alone; `extent` is the whole object's size;
    `normals`, the source's face normals, let an area be judged the way the camera will.
    Quads already in `coarse` stay out of it.
    """

    options = options or GridOptions()
    report = GridReport()
    faces = [tuple(face) for face in coarse.faces]
    materials = list(coarse.face_materials)
    if not faces or int(labels.max(initial=-1)) < 0:
        return coarse, report
    points = np.asarray(coarse.vertices, dtype=np.float64)
    vertices = list(points)
    extent = extent or float(np.ptp(points, axis=0).max())
    triangle = np.fromiter((len(face) == 3 for face in faces), dtype=bool, count=len(faces))
    face_of, a, _b = _half_edges(faces)
    centres = np.stack([np.bincount(face_of, points[a, axis], len(faces)) for axis in range(3)], axis=1)
    centres /= np.bincount(face_of, minlength=len(faces))[:, None]
    area_of = np.where(triangle, labels[nearest(centres)[0]], -1)
    adjacency = face_adjacency(faces)

    def reject(reason: str) -> None:
        report.rejected[reason] = report.rejected.get(reason, 0) + 1

    links: list[list[int]] = [[] for _ in faces]
    for first, second in zip(*(side.tolist() for side in adjacency)):
        links[first].append(second)
        links[second].append(first)

    # One disk or ring per area, small holes filled, the rest given back. Biggest first: they get first call on the budget.
    final = np.full(len(faces), -1, dtype=np.int64)
    order = np.argsort(area_of, kind="stable")
    sorted_areas = area_of[order]
    plans = []
    for area in np.unique(area_of[area_of >= 0]).tolist():
        group = order[np.searchsorted(sorted_areas, area) : np.searchsorted(sorted_areas, area, side="right")]
        size = float(np.linalg.norm(np.ptp(centres[group], axis=0)))
        if size < options.min_size * extent or len(group) < options.min_faces:
            reject("too small")  # triangles along a grid's border pair badly, so a small grid costs more than it gives
            continue
        mask = np.zeros(len(faces), dtype=bool)
        mask[group] = True
        mask = _tidy_border(points, faces, mask, area_of < 0, links, options.spike_deg)
        parts = mask_components(faces, mask, adjacency)
        if not parts:
            continue
        loops, pinched = boundary_loops(faces, parts[0])
        shape = (len(loops), _euler(faces, parts[0]))
        if pinched or shape not in ((1, 1), (2, 0)):
            reject("not a disk or a ring")
        elif len({materials[f] for f in parts[0].tolist()}) > 1:
            reject("mixed materials")
        elif not triangle[parts[0]].all():
            reject("has quads")
        elif min(len(loop) for loop in loops) < (options.min_border if len(loops) == 1 else 3) or max(len(loop) for loop in loops) < options.min_border:
            reject("border too short")
        else:
            plans.append((size, area, parts[0]))
    plans.sort(key=lambda row: -row[0])
    for _size, area, part in plans:
        final[part] = area
    area_list = final.tolist()
    members = {area: np.flatnonzero(final == area).tolist() for _size, area, _part in plans}
    users = _edge_faces(faces)

    tolerance = options.row_sag * extent
    splits = _Splits(vertices, faces, area_list, materials, members, users)

    def turned_share(corners: np.ndarray) -> float:
        """Share of this surface, by area, that the camera would call different: turned from the source, or moved off it.

        Each triangle (a quad is two) is looked at in four places, as one sample per big face misses what it cuts across.
        """

        tri = corners if corners.shape[1] == 3 else np.concatenate((corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]))
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        size = np.linalg.norm(cross, axis=1)
        middle = tri.mean(axis=1)
        samples = np.concatenate((middle, 0.5 * (middle + tri[:, 0]), 0.5 * (middle + tri[:, 1]), 0.5 * (middle + tri[:, 2])))
        found, closest = nearest(samples)
        repeat = np.tile(cross, (4, 1))
        weight = np.tile(size, 4)
        away = np.abs(np.einsum("ij,ij->i", repeat, normals[found])) < math.cos(math.radians(options.turned_deg)) * weight
        away |= np.linalg.norm(samples - closest, axis=1) > options.moved * extent
        return float(weight[away].sum() / max(float(weight.sum()), 1e-30))

    budget = options.max_added * len(faces) / 2.0
    built = []
    done: set[int] = set()
    queue = [(size, area, True) for size, area, _part in reversed(plans)]
    retry = None  # the last area, if it got rim vertices: should it fail, it goes again without them
    while queue or retry:
        splits.undo()  # whatever the last area split and didn't use
        if retry and retry[1] not in done:
            queue.append(retry)
        retry = None
        if not queue:
            break
        _size, area, with_rim = queue.pop()
        project = project_for(area) if project_for else (lambda query: nearest(query)[1])
        ids = members[area]
        loops = boundary_loops(faces, ids)[0]
        here = np.asarray([faces[f] for f in ids], dtype=np.int64)
        corners_xyz = points_of(vertices, here.ravel()).reshape(-1, 3, 3)
        surface = float(np.linalg.norm(np.cross(corners_xyz[:, 1] - corners_xyz[:, 0], corners_xyz[:, 2] - corners_xyz[:, 0]), axis=1).sum()) / 2.0
        if len(loops) == 2 and len(np.unique(here)) == len(loops[0]) + len(loops[1]) and abs(len(loops[0]) - len(loops[1])) <= options.band_skew:
            reject("already a band")  # one strip of triangles from border to border: pairing makes its rows
            continue
        if len(loops) == 1:
            outline = points_of(vertices, loops[0])
            perimeter = float(np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1).sum())
            if 4.0 * math.pi * surface / max(perimeter * perimeter, 1e-30) < options.min_compact and not _four_corners(outline, options.corner_deg):
                reject("a bent strip")
                continue
        # Coarse rims: where the border is a curve cut into long straight edges, it gets the vertices between them, on the
        # curve the neighbours imply. Sag won't tell: a chord across a flat disc lies on the disc. A built neighbour can't change.
        for _round in range(options.rim_rounds if with_rim and _size >= options.rim_size * extent else 0):
            changed = False
            for index, loop in enumerate(loops):
                outline = points_of(vertices, loop)
                before, after = outline - np.roll(outline, 1, axis=0), np.roll(outline, -1, axis=0) - outline
                bend = np.degrees(np.arctan2(np.linalg.norm(np.cross(before, after), axis=1), np.einsum("ij,ij->i", before, after)))
                curved = (bend > options.rim_turn) & (bend < options.corner_deg)
                wanted = curved & np.roll(curved, -1)
                if not wanted.any():
                    continue
                middle = (9.0 * (outline + np.roll(outline, -1, axis=0)) - np.roll(outline, 1, axis=0) - np.roll(outline, -2, axis=0)) / 16.0
                closest = nearest(middle)[1]
                grown = []
                for k, v in enumerate(loop):
                    grown.append(v)
                    w = loop[(k + 1) % len(loop)]
                    edge = (v, w) if v < w else (w, v)
                    if wanted[k] and len(users[edge]) <= 2 and not any(len(faces[f]) != 3 or area_list[f] in done for f in users[edge]):
                        grown.append(splits.split(v, w, closest[k]))
                        changed = True
                        retry = (_size, area, False)
                loops[index] = grown
            if not changed:
                break
        plate = None
        if len(loops) == 1:
            positions = _four_corners(points_of(vertices, loops[0]), options.corner_deg)
            if positions:
                # Opposite sides need the same count: the shorter ones get vertices. Too uneven, and it isn't a plate to grid.
                tips = [loops[0][k] for k in positions]
                for side in (0, 1):
                    while loops[0] is not None:
                        where = [loops[0].index(tip) for tip in tips]
                        one = (where[side + 1] - where[side]) % len(loops[0])
                        two = (where[(side + 3) % 4] - where[side + 2]) % len(loops[0])
                        if one == two or abs(one - two) > max(3, 0.5 * max(one, two)):
                            break
                        start = where[side] if one < two else where[side + 2]
                        allowed = range(start, start + min(one, two)) if start + min(one, two) <= len(loops[0]) else None
                        if allowed is None:
                            break
                        grown = _split_border(splits, done, loops[0], project, allowed)
                        if grown is None:
                            break
                        loops[0] = grown
                where = [loops[0].index(tip) for tip in tips]
                sides = [(where[(k + 1) % 4] - where[k]) % len(loops[0]) for k in range(4)]
                if sides[0] == sides[2] and sides[1] == sides[3]:
                    plate = where
        if len(loops) == 1 and len(loops[0]) % 2:
            loops[0] = _split_border(splits, done, loops[0], project)
        elif len(loops) == 2 and (len(loops[0]) + len(loops[1])) % 2:
            short = int(len(loops[1]) < len(loops[0]))
            loops[short] = _split_border(splits, done, loops[short], project)
        if any(loop is None for loop in loops):
            reject("odd border")
            continue
        ids = members[area]
        used = sorted({v for f in ids for v in faces[f]})
        local = {v: k for k, v in enumerate(used)}
        here = np.asarray([vertices[v] for v in used])
        tris = np.asarray([[local[v] for v in faces[f]] for f in ids], dtype=np.int64)
        was = here[tris].mean(axis=1)
        limit = max(options.max_error * float(np.linalg.norm(was - nearest(was)[1], axis=1).max()), options.max_sag * float(np.linalg.norm(np.ptp(here, axis=0))))
        turned_before = turned_share(here[tris]) if normals is not None else 0.0

        def fault(xyz, quads, floor: float = 0.0) -> str | None:
            quad_array = np.asarray(quads, dtype=np.int64)
            if quad_quality(xyz[quad_array]).min() <= floor:
                return "folded"
            centre = xyz[quad_array].mean(axis=1)
            if float(np.linalg.norm(centre - nearest(centre)[1], axis=1).max()) > limit + 1e-9:
                return "strays from the source"
            if normals is not None and turned_share(xyz[quad_array]) > turned_before + options.max_turned:
                return "turned from the source"
            if len(quads) - len(tris) / 2.0 > budget:
                return "too many quads"
            return None

        if len(loops) == 1:
            seam = loops[0]
            border = np.asarray([local[v] for v in seam], dtype=np.int64)
            # How much the disk turns sets how fine its middle has to be; a flat one needs few rows of quads there.
            cross = np.cross(here[tris[:, 1]] - here[tris[:, 0]], here[tris[:, 2]] - here[tris[:, 0]])
            middle_normal = cross.sum(axis=0)
            middle_normal /= max(float(np.linalg.norm(middle_normal)), 1e-30)
            unit = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-30)
            turned = 2.0 * math.degrees(math.acos(float(np.clip((unit @ middle_normal).min(), -1.0, 1.0))))
            inner = 4 * max(2, math.ceil(turned / options.turn_per_row / 4.0) * 2)
            if plate and turned > options.plate_turn and 4.0 * math.pi * surface / max(perimeter * perimeter, 1e-30) < options.min_compact:
                reject("a bent strip")
                continue
            if plate:
                xyz, quads = cap_fill(here[border], tris, here, border, project, None, options.relax_rounds, corners=plate)
                problem = fault(xyz, quads)
                if not problem and turned > options.flat_turn and len(quads) < options.min_kept * len(tris) / 2.0:
                    problem = "drops detail"  # a curved plate had its triangles for a reason
            elif len(seam) > inner * options.max_ratio:
                xyz, quads = cap_fill(here[border], tris, here, border, project, inner, options.relax_rounds, options.max_ratio)
                problem = fault(xyz, quads)
            else:
                xyz, quads = cap_fill(here[border], tris, here, border, project, None, options.relax_rounds)
                problem = fault(xyz, quads, options.plain_quality)
                if problem:
                    xyz, quads = cap_fill(here[border], tris, here, border, project, len(seam), options.relax_rounds)
                    problem = fault(xyz, quads)
        else:
            big = int(len(loops[1]) > len(loops[0]))
            dense, sparse = loops[big], loops[1 - big][::-1]  # opposite borders of a ring run opposite ways
            xyz, quads, turn = ring_fill(points_of(vertices, dense), points_of(vertices, sparse), project, options, tolerance)
            seam = dense + sparse[turn:] + sparse[:turn]
            problem = fault(xyz, quads)
        if problem:
            reject(problem)
            continue
        budget -= max(len(quads) - len(tris) / 2.0, 0.0)
        done.add(area)
        splits.log.clear()
        built.append((area, materials[ids[0]], seam, xyz, quads, len(ids)))
    splits.undo()
    gone = {area for area, *_rest in built}
    out_points = list(vertices)
    out_faces = [face for face, label in zip(faces, area_list) if label not in gone]
    out_materials = [material for material, label in zip(materials, area_list) if label not in gone]
    for _area, material, loop, xyz, quads, removed in built:
        offset = len(out_points) - len(loop)
        table = lambda v, offset=offset, loop=loop: loop[v] if v < len(loop) else v + offset  # noqa: E731
        out_points.extend(xyz[len(loop) :])
        out_faces.extend(tuple(table(v) for v in quad) for quad in quads)
        out_materials.extend([material] * len(quads))
        report.built += 1
        report.faces_in += removed
        report.quads_out += len(quads)
        report.loops.append(len(loop))
    if not built:
        return coarse, report
    stacked = np.asarray(out_points, dtype=np.float64)
    alive = np.zeros(len(stacked), dtype=bool)
    for face in out_faces:
        alive[list(face)] = True
    remap = np.cumsum(alive) - 1
    return MeshData(vertices=stacked[alive], faces=[tuple(remap[list(face)].tolist()) for face in out_faces], face_materials=out_materials, name=coarse.name), report


__all__ = ["GridOptions", "GridReport", "cap_fill", "crease_areas", "grid_areas", "ring_fill", "smooth_areas"]
