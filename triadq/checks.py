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
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))  # those pairs
    broken: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # face on a bad edge, flipped or duplicate
    crossing: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # face that passes through another
    per_piece: dict[int, tuple[int, int, int]] = field(default_factory=dict)  # piece -> (broken faces, holes, crossing faces)
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
    summary = topology_summary(MeshData(co, faces))
    if pairs is ...:
        pairs = crossing_pairs(co, faces)
    crossing = np.zeros(len(faces), bool)
    if pairs is not None and len(pairs):
        crossing[pairs.ravel()] = True
    out = Inspect(nm, summary.non_manifold_vertices, flipped, dups, holes, None if pairs is None else len(pairs), np.zeros((0, 2), np.int64) if pairs is None else pairs, broken, crossing)
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
