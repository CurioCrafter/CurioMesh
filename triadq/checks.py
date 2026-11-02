# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Exact checks on a finished result: non-manifold edges and vertices, flipped or duplicate faces, holes, and faces that
pass through each other. The same numbers are taken on the source, because a scan may already be broken: only what the
result adds counts. Every count can be split per source piece, so a repair can go back to the one piece that broke."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .topology import topology_summary
from .types import MeshData

SELF_LIMIT = 400_000  # faces: above this the self-intersection test is skipped and the summary says so
CHUNK = 50_000


@dataclass
class Inspect:
    nm_edges: int = 0
    nm_verts: int = 0
    flipped: int = 0
    dups: int = 0
    holes: int = 0  # boundary loops
    selfx: int | None = None  # face pairs passing through each other; None = not measured
    broken: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # face on a bad edge, flipped or duplicate
    crossing: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # face that passes through another
    per_piece: dict[int, tuple[int, int, int]] = field(default_factory=dict)  # piece -> (broken faces, holes, crossing faces)


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
    width = max(len(f) for f in faces)
    pad = np.full((len(faces), width), -1, dtype=np.int64)
    for i, face in enumerate(faces):
        pad[i, : len(face)] = face
    keep = np.ones(len(pairs), bool)
    for s in range(0, len(pairs), CHUNK):
        a, b = pad[pairs[s : s + CHUNK, 0]], pad[pairs[s : s + CHUNK, 1]]
        keep[s : s + CHUNK] = ~(((a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)).any(axis=(1, 2)))
    return pairs[keep]


def crossing_pairs(co, faces) -> np.ndarray | None:
    """Face pairs of this mesh that pass through each other, or None when it is too big to test."""
    from mathutils.bvhtree import BVHTree  # only inside Blender

    if len(faces) > SELF_LIMIT:
        return None
    tree = BVHTree.FromPolygons(np.asarray(co).tolist(), [list(f) for f in faces], epsilon=0.0)
    pairs = np.asarray(tree.overlap(tree), dtype=np.int64).reshape(-1, 2)
    return drop_shared(faces, pairs[pairs[:, 0] < pairs[:, 1]])


def inspect(co, faces, piece, pairs=...) -> Inspect:
    """All the numbers for one mesh. `piece` gives each face's piece. `pairs` defaults to a BVH test (Blender only)."""
    faces = [tuple(f) for f in faces]
    piece = np.asarray(piece, dtype=np.int64)
    broken, nm, flipped, dups = edge_flags(faces)
    holes, holes_by = hole_counts(faces, piece)
    summary = topology_summary(MeshData(co, faces))
    if pairs is ...:
        pairs = crossing_pairs(co, faces)
    crossing = np.zeros(len(faces), bool)
    if pairs is not None and len(pairs):
        crossing[pairs.ravel()] = True
    out = Inspect(nm, summary.non_manifold_vertices, flipped, dups, holes, None if pairs is None else len(pairs), broken, crossing)
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
        extra = sum(max(0, a - b) for a, b in zip(mine, before.per_piece.get(p, (0, 0, 0))))
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
        "self-intersections": max(0, (after.selfx or 0) - (before.selfx or 0)),
    }
    found = ", ".join(f"{n:,} {name}" for name, n in new.items() if n)
    note = "" if after.selfx is not None else " (self-intersections not tested: too many faces)"
    if left:
        return f"{left} spot{'s' if left != 1 else ''} could not be fixed and {'are' if left != 1 else 'is'} selected for you: {found or 'they cross or break more than your mesh did there'}{note}"
    if fixed:
        return f"Repaired {fixed} spot{'s' if fixed != 1 else ''}; no new problems{note}"
    return f"New problems: {found}{note}" if found else f"No new problems{note}"
