# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Exact checks on a finished result: non-manifold edges and vertices, flipped or duplicate faces, holes, and faces that
pass through each other. The same numbers are taken on the source, because a scan may already be broken: only what the
result adds counts. Every count can be split per source piece, so a repair can go back to the one piece that broke."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .topology import _turbo, topology_summary
from .types import MeshData

CHUNK = 50_000
TILE = 20_000  # faces tested per slice, so a big mesh never blocks Blender for long


@dataclass
class Inspect:
    nm_edges: int = 0
    nm_verts: int = 0
    flipped: int = 0
    dups: int = 0
    holes: int = 0  # boundary loops
    selfx: int | None = None  # face pairs passing through each other; None = not measured
    folded: int = 0  # quads bent past 60 degrees or concave (patch.quad_quality <= 0)
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))  # those pairs
    broken: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # face on a bad edge, flipped, duplicate or a folded quad
    crossing: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # face that passes through another
    per_piece: dict[int, tuple[int, int, int]] = field(default_factory=dict)  # piece -> (broken faces, holes, crossing faces)
    bent: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # the folded quads among `broken`
    new_cross: dict[int, int] | None = None  # piece -> crossings the source did not have there, once that is worked out


def edge_flags(faces) -> tuple[np.ndarray, int, int, int]:
    """(broken per face, non-manifold edges, flipped edges, duplicate faces)."""
    users: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    seen: set[tuple[int, ...]] = set()
    broken = np.zeros(len(faces), bool)
    dups = 0
    for f, face in enumerate(faces):
        key = tuple(sorted(face))
        if key in seen:
            dups += 1
            broken[f] = True
        seen.add(key)
        n = len(face)
        for i, a in enumerate(face):
            b = face[(i + 1) % n]
            users.setdefault((a, b) if a < b else (b, a), []).append((f, a < b))
    nm = flipped = 0
    for items in users.values():
        if len(items) > 2:
            nm += 1
            for f, _ in items:
                broken[f] = True
        elif len(items) == 2 and items[0][1] == items[1][1]:
            flipped += 1
            broken[items[0][0]] = broken[items[1][0]] = True
    return broken, nm, flipped, dups


def hole_counts(faces, piece: np.ndarray) -> tuple[int, dict[int, int]]:
    """Boundary loops overall and per piece."""
    count: dict[tuple[int, int], int] = {}
    owner: dict[tuple[int, int], int] = {}
    for f, face in enumerate(faces):
        n = len(face)
        for i, a in enumerate(face):
            b = face[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            count[key] = count.get(key, 0) + 1
            owner[key] = int(piece[f])
    parent: dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key, c in count.items():
        if c == 1:
            parent[find(key[0])] = find(key[1])
    roots: dict[int, int] = {}
    for key, c in count.items():
        if c == 1:
            roots[find(key[0])] = owner[key]
    per: dict[int, int] = {}
    for p in roots.values():
        per[p] = per.get(p, 0) + 1
    return len(roots), per


def drop_shared(faces, pairs: np.ndarray) -> np.ndarray:
    """Keep the face pairs that share no vertex: neighbours that merely touch are not crossings."""
    if not len(pairs):
        return pairs
    pad = faces if isinstance(faces, np.ndarray) else corner_table(faces)
    keep = np.ones(len(pairs), bool)
    for s in range(0, len(pairs), CHUNK):
        a, b = pad[pairs[s : s + CHUNK, 0]], pad[pairs[s : s + CHUNK, 1]]
        keep[s : s + CHUNK] = ~(a[:, :, None] == b[:, None, :]).any(axis=(1, 2))
    return pairs[keep]


def tiles(centers: np.ndarray, most: int = TILE) -> list[np.ndarray]:
    """Face ids cut into tiles of at most `most` faces, each cut at the middle of the widest axis."""
    out, stack = [], [np.arange(len(centers))]
    while stack:
        ids = stack.pop()
        if len(ids) <= most:
            out.append(ids)
            continue
        axis = int(np.ptp(centers[ids], axis=0).argmax())
        half = len(ids) // 2
        order = np.argpartition(centers[ids, axis], half)
        stack += [ids[order[:half]], ids[order[half:]]]
    return out


def face_boxes(co: np.ndarray, pad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Low and high corner of each face's box, from a table of corners (see `corner_table`)."""
    corners = co[pad]
    return corners.min(axis=1), corners.max(axis=1)


def corner_table(faces) -> np.ndarray:
    """Faces as a rectangular table, short faces padded with their own first corner."""
    width = max((len(f) for f in faces), default=3)
    return np.asarray([tuple(f) + (f[0],) * (width - len(f)) for f in faces], dtype=np.int64).reshape(-1, width)


def tree_of(co: np.ndarray, faces, ids: np.ndarray):
    """A search tree of just these faces, whose numbers are positions in `ids`."""
    from mathutils.bvhtree import BVHTree  # only inside Blender

    corners = np.fromiter((v for f in ids.tolist() for v in faces[f]), dtype=np.int64)
    used, flat = np.unique(corners, return_inverse=True)
    polys, at = [], 0
    for f in ids.tolist():
        size = len(faces[f])
        polys.append(flat[at : at + size].tolist())
        at += size
    return BVHTree.FromPolygons(co[used].tolist(), polys, epsilon=0.0)


def crossing_slices(co, faces):
    """Generator: face pairs that pass through each other, one tile per step. Returns the pairs when exhausted."""
    co = np.asarray(co, dtype=np.float64)
    if not len(faces):
        return np.zeros((0, 2), dtype=np.int64)
    tree = _turbo.face_tree(co, faces) if _turbo is not None else None
    if tree is not None:  # all of it at once, on all cores, and it stops when the job does
        return tree.crossing_pairs()
    pad = corner_table(faces)
    low, high = face_boxes(co, pad)
    found = []
    for ids in tiles((low + high) / 2.0):
        near = np.flatnonzero(((low <= high[ids].max(axis=0)) & (high >= low[ids].min(axis=0))).all(axis=1))
        hits = tree_of(co, faces, ids).overlap(tree_of(co, faces, near))
        if hits:
            hits = np.asarray(hits, dtype=np.int64)
            pairs = np.column_stack((ids[hits[:, 0]], near[hits[:, 1]]))
            found.append(pairs[pairs[:, 0] != pairs[:, 1]])
        yield
    if not found:
        return np.zeros((0, 2), dtype=np.int64)
    return drop_shared(pad, np.unique(np.sort(np.vstack(found), axis=1), axis=0))


def crossing_pairs(co, faces) -> np.ndarray:
    """Face pairs of this mesh that pass through each other, all at once."""
    slices = crossing_slices(co, faces)
    while True:
        try:
            next(slices)
        except StopIteration as done:
            return done.value


def folded_quads(co, faces) -> np.ndarray:
    """Per face: a quad bent past 60 degrees across a diagonal, or concave. A thin part cut to too few faces wraps its edges so."""
    from .patch import quad_quality

    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    out = np.zeros(len(faces), bool)
    quads = np.flatnonzero(sizes == 4)
    if len(quads):
        corners = np.asarray([faces[k] for k in quads.tolist()], dtype=np.int64)
        out[quads] = quad_quality(np.asarray(co, dtype=np.float64)[corners]) <= 0.0
    return out


def inspect(co, faces, piece, pairs=...) -> Inspect:
    """All the numbers for one mesh. `piece` gives each face's piece. `pairs` defaults to a BVH test (Blender only)."""
    faces = [tuple(f) for f in faces]
    piece = np.asarray(piece, dtype=np.int64)
    fast = _turbo.mesh_checks(faces, piece) if _turbo is not None else None
    if fast is not None:
        broken, nm, flipped, dups, holes, holes_by = fast
    else:
        broken, nm, flipped, dups = edge_flags(faces)
        holes, holes_by = hole_counts(faces, piece)
    folded = folded_quads(co, faces)
    broken = np.asarray(broken, dtype=bool) | folded  # a folded quad is a new problem like a flipped one: its part goes a size up
    summary = topology_summary(MeshData(co, faces))
    if pairs is ...:
        pairs = crossing_pairs(co, faces)
    crossing = np.zeros(len(faces), bool)
    if pairs is not None and len(pairs):
        crossing[pairs.ravel()] = True
    out = Inspect(nm, summary.non_manifold_vertices, flipped, dups, holes, None if pairs is None else len(pairs), int(folded.sum()), np.zeros((0, 2), np.int64) if pairs is None else pairs, broken, crossing)
    out.bent = folded
    # Crossing, per piece: pairs inside the piece plus the other pieces it crosses (a part passing through a
    # neighbour counts once for each of the two, however many faces meet).
    inside: dict[int, int] = {}
    others: dict[int, set[int]] = {}
    if pairs is not None and len(pairs):
        two = piece[pairs]
        same = two[:, 0] == two[:, 1]
        for p, n in zip(*np.unique(two[same, 0], return_counts=True)):
            inside[int(p)] = int(n)
        for a, b in np.unique(np.sort(two[~same], axis=1), axis=0).tolist():
            others.setdefault(a, set()).add(b)
            others.setdefault(b, set()).add(a)
    broken_by = np.bincount(piece, weights=broken, minlength=int(piece.max()) + 1 if len(piece) else 0)
    for p in np.unique(piece).tolist():
        out.per_piece[p] = (int(broken_by[p]), holes_by.get(p, 0), inside.get(p, 0) + len(others.get(p, ())))
    return out


def new_spots(before: Inspect, after: Inspect) -> list[int]:
    """Pieces of the result with more broken faces, holes or crossings than the same piece of the source, worst first."""
    bad = {}
    for p, mine in after.per_piece.items():
        theirs = before.per_piece.get(p, (0, 0, 0))
        extra = sum(max(0, a - b) for a, b in zip(mine, theirs))
        if after.new_cross is not None:
            extra += after.new_cross.get(p, 0) - max(0, mine[2] - theirs[2])
        if extra:
            bad[p] = extra
    return sorted(bad, key=lambda p: (-bad[p], p))


def describe(before: Inspect, after: Inspect, fixed: int, left: int) -> str:
    """The outcome in plain words. Only what the result added counts."""
    new = {
        "non-manifold edges": max(0, after.nm_edges - before.nm_edges),
        "non-manifold vertices": max(0, after.nm_verts - before.nm_verts),
        "flipped edges": max(0, after.flipped - before.flipped),
        "duplicate faces": max(0, after.dups - before.dups),
        "folded quads": max(0, after.folded - before.folded),
        "holes opened": max(0, after.holes - before.holes),
        "self-intersections": sum(after.new_cross.values()) if after.new_cross is not None else max(0, (after.selfx or 0) - (before.selfx or 0)),
    }
    found = ", ".join(f"{n:,} {name}" for name, n in new.items() if n)
    note = "" if after.selfx is not None else " (self-intersections not tested)"
    if left:
        return f"{left} spot{'s' if left != 1 else ''} could not be fixed and {'are' if left != 1 else 'is'} selected for you: {found or 'they cross or break more than your mesh did there'}{note}"
    if fixed:
        return f"Repaired {fixed} spot{'s' if fixed != 1 else ''}; no new problems{note}"
    return f"New problems: {found}{note}" if found else f"No new problems{note}"


# --------------------------------------------------------------------------- defects per face, against the source

FACING = -0.5  # a face whose normal is more than 120 degrees off the source's there is turned inward
FOLD = -0.866  # two faces on one edge more than 150 degrees apart lie folded over each other
SUNK = 0.5  # a face whose middle is inside the source by more than this share of its longest side has sunk into it
LOOK_AHEAD = 1e-3  # of a face's longest side: how far in front of it the source is asked which way is out
FLAT_AREA = 1e-6  # of the mean squared side: a face without area
MOST_SUNK = 20_000  # faces tested by winding number, deepest first: it looks at every source triangle
OVERLAP_SPLITS = 4  # random halves tried for faces lying on each other: one misses a pair half the time


@dataclass
class Defects:
    """Per result face: what is wrong with it. Each mask is as long as the faces."""

    inward: np.ndarray  # normal turned against the source's at the nearest source point
    sunk: np.ndarray  # middle inside the source volume, deeper than a chord would go
    crossing: np.ndarray  # passes through a face it shares no corner with
    folded_over: np.ndarray  # on an edge where it turns back over its neighbour (the source doesn't)
    bent: np.ndarray  # a quad bent past 60 degrees or concave
    flat: np.ndarray  # no area, or a corner used twice
    duplicate: np.ndarray  # the same corners as an earlier face
    overlap: np.ndarray  # lies on another face, in its plane, without sharing a corner
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))  # the crossing pairs

    KINDS = ("inward", "sunk", "crossing", "folded_over", "bent", "flat", "duplicate", "overlap")

    def any(self) -> np.ndarray:
        out = np.zeros(len(self.inward), bool)
        for k in self.KINDS:
            out |= getattr(self, k)
        return out

    def counts(self, where: np.ndarray | None = None) -> dict[str, int]:
        """Faces of each kind, of all faces or of the faces in `where`."""
        pick = slice(None) if where is None else where
        out = {k: int(getattr(self, k)[pick].sum()) for k in self.KINDS}
        out["any"] = int(self.any()[pick].sum())
        return out


def _normals(co: np.ndarray, pad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unit normal, area, middle) per face of a corner table, by Newell's sum. Padded corners add nothing to the normal."""
    p = co[pad]
    n = 0.5 * np.cross(p, np.roll(p, -1, axis=1)).sum(axis=1)
    area = np.linalg.norm(n, axis=1)
    return n / np.maximum(area, 1e-300)[:, None], area, p.mean(axis=1)


def _edge_faces(pad: np.ndarray) -> np.ndarray:
    """(M, 2) faces on each edge that has exactly two."""
    a, b = pad.ravel(), np.roll(pad, -1, axis=1).ravel()
    face = np.repeat(np.arange(len(pad)), pad.shape[1])
    real = a != b  # padding repeats the first corner
    lo, hi, face = np.minimum(a, b)[real], np.maximum(a, b)[real], face[real]
    order = np.lexsort((hi, lo))
    lo, hi, face = lo[order], hi[order], face[order]
    start = np.flatnonzero(np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])])
    two = start[np.diff(np.r_[start, len(lo)]) == 2]
    return np.column_stack((face[two], face[two + 1]))


def _overlapping(co, faces, pad, normal, middle, size, seed: int = 0) -> np.ndarray:
    """Faces whose middle lies on another face, in its plane, that shares no corner with it."""
    out = np.zeros(len(pad), bool)
    if _turbo is None or len(pad) < 2:
        return out
    rng = np.random.default_rng(seed)
    for _ in range(OVERLAP_SPLITS):  # ponytail: random halves miss a pair 1/16 of the time; a k-nearest query would not
        half = rng.random(len(pad)) < 0.5
        here, there = np.flatnonzero(half), np.flatnonzero(~half)
        if not len(here) or not len(there):
            continue
        tree = _turbo.face_tree(co, [faces[f] for f in there.tolist()])
        if tree is None:
            return out
        _at, hit, away = tree.nearest(middle[here])
        hit = there[np.asarray(hit, dtype=np.int64)]
        close = (np.asarray(away) < 0.02 * size[here]) & (np.abs((normal[here] * normal[hit]).sum(axis=1)) > 0.95)
        if close.any():
            a, b = here[close], hit[close]
            apart = ~(pad[a][:, :, None] == pad[b][:, None, :]).any(axis=(1, 2))
            out[a[apart]] = out[b[apart]] = True
    return out


def defects(co, faces, source_co=None, source_faces=None, pairs=None) -> Defects:
    """Everything wrong with each face of a result, the source saying which way is out. The source lookups, the crossing
    test and the overlap test need the native core: without it those masks stay empty. `pairs`: crossings found already."""
    co = np.asarray(co, dtype=np.float64)
    faces = [tuple(f) for f in faces]
    count = len(faces)
    empty = np.zeros(count, bool)
    if not count:
        return Defects(*(empty.copy() for _ in Defects.KINDS))
    pad = corner_table(faces)
    normal, area, middle = _normals(co, pad)
    sides = np.linalg.norm(co[pad] - co[np.roll(pad, -1, axis=1)], axis=2)
    size = sides.max(axis=1)
    mean_side = float(sides[sides > 0].mean()) if (sides > 0).any() else 1.0
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=count)
    flat = (area < FLAT_AREA * mean_side * mean_side) | np.fromiter((len(set(f)) < len(f) for f in faces), dtype=bool, count=count)
    _keys, first, inverse = np.unique(np.column_stack((np.sort(pad, axis=1), sizes)), axis=0, return_index=True, return_inverse=True)
    original = first[inverse.ravel()]
    duplicate = original != np.arange(count)
    bent = folded_quads(co, faces)
    if pairs is None:
        tree = _turbo.face_tree(co, faces) if _turbo is not None else None
        pairs = tree.crossing_pairs() if tree is not None else np.zeros((0, 2), np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    crossing = empty.copy()
    crossing[pairs.ravel()] = True
    inward, sunk = empty.copy(), empty.copy()
    edge = _edge_faces(pad)
    fold = (normal[edge[:, 0]] * normal[edge[:, 1]]).sum(axis=1) < FOLD
    tree = None
    if source_co is not None and _turbo is not None:
        source_co = np.asarray(source_co, dtype=np.float64)
        source_faces = [tuple(f) for f in source_faces]
        tree = _turbo.face_tree(source_co, source_faces)
    if tree is not None:
        source_pad = corner_table(source_faces)
        source_normals, _area, source_middle = _normals(source_co, source_pad)
        source_sides = np.linalg.norm(source_co[source_pad] - source_co[np.roll(source_pad, -1, axis=1)], axis=2).max(axis=1)
        stacked = _overlapping(source_co, source_faces, source_pad, source_normals, source_middle, source_sides)
        at, hit, away = tree.nearest(middle)
        hit = np.asarray(hit, dtype=np.int64)
        source_normal = source_normals[hit]
        # Which way is out, from the middle and halfway to each corner: a face across a sharp edge of the source gets
        # both sides' say. A face lying on two source faces back to back (a sheet with both sides) can't be turned. The
        # points sit just in front of the face, so one midway between the two sides of a plate asks the side it faces.
        towards = 0.5 * (middle[:, None, :] + co[pad]) + (LOOK_AHEAD * size)[:, None, None] * normal[:, None, :]
        _at, near, _away = tree.nearest(towards.reshape(-1, 3))
        near = np.asarray(near, dtype=np.int64).reshape(len(pad), -1)
        votes = np.where(stacked[near][:, :, None], 0.0, source_normals[near]).sum(axis=1) + np.where(stacked[hit][:, None], 0.0, source_normal)
        length = np.linalg.norm(votes, axis=1)
        inward = (length > 0.0) & ((normal * votes).sum(axis=1) < FACING * np.maximum(length, 1e-300))
        depth = -((middle - at) * source_normal).sum(axis=1)
        deep = np.flatnonzero((depth > SUNK * size) & (np.asarray(away) > 0.99 * SUNK * size))
        deep = deep[np.argsort(-depth[deep] / size[deep])][:MOST_SUNK]
        if len(deep):
            tris = np.asarray([(f[0], f[i], f[i + 1]) for f in source_faces for i in range(1, len(f) - 1)], dtype=np.int64)
            sunk[deep] = _turbo.inside_volume(middle[deep], source_co[tris])
        fold &= (source_normal[edge[:, 0]] * source_normal[edge[:, 1]]).sum(axis=1) > FACING  # a knife edge of the source folds so too
    folded_over = empty.copy()
    folded_over[edge[fold].ravel()] = True
    overlap = _overlapping(co, faces, pad, normal, middle, size) & ~duplicate
    overlap[original[duplicate]] = False  # the first of two duplicates is not also an overlap
    if tree is not None:  # where the source passes through itself or lies on itself, the result may too
        old = stacked.copy()
        old[tree.crossing_pairs().ravel()] = True
        if old.any() and (len(pairs) or overlap.any()):
            _near, gap = _turbo.point_tree(source_middle[old]).nearest(middle)
            new = np.asarray(gap) > 2.0 * size
            pairs = pairs[new[pairs[:, 0]] | new[pairs[:, 1]]]
            crossing[:] = False
            crossing[pairs.ravel()] = True
            overlap &= new
    return Defects(inward, sunk, crossing, folded_over, bent, flat, duplicate, overlap, pairs)
