# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Sharing faces between the pieces of a mesh: most visible detail per face, with a floor no piece goes below."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Level:
    faces: int
    loss: float  # visible area that looks different at this size


def allocate(pieces: list[list[Level]], allowed: float, piece_share: float, seen: list[float]) -> list[int]:
    """Pick a level per piece with the fewest faces whose total loss is at most `allowed`.

    Each piece's levels go from fewest faces to most. `seen` is each piece's visible area. No piece may lose more than
    `piece_share` of it, so a small part can't be given up to save a few faces. From there, each upgrade is taken in
    order of loss removed per face added. A level that's no better than a smaller one is skipped.
    """

    choice = []
    for levels, area in zip(pieces, seen):
        index = next((k for k, level in enumerate(levels) if level.loss <= piece_share * area), len(levels) - 1)
        choice.append(index)
    total = sum(levels[k].loss for levels, k in zip(pieces, choice))
    if total <= allowed:
        return choice

    def best_upgrade(piece: int):
        levels, at = pieces[piece], choice[piece]
        best = None
        for k in range(at + 1, len(levels)):
            gain = levels[at].loss - levels[k].loss
            cost = max(levels[k].faces - levels[at].faces, 1)
            if gain > 0 and (best is None or gain / cost > best[0]):
                best = (gain / cost, k)
        return best

    heap = []
    for piece in range(len(pieces)):
        upgrade = best_upgrade(piece)
        if upgrade is not None:
            heap.append((-upgrade[0], piece, upgrade[1]))
    heapq.heapify(heap)
    while heap and total > allowed:
        _rate, piece, k = heapq.heappop(heap)
        total -= pieces[piece][choice[piece]].loss - pieces[piece][k].loss
        choice[piece] = k
        upgrade = best_upgrade(piece)
        if upgrade is not None:
            heapq.heappush(heap, (-upgrade[0], piece, upgrade[1]))
    return choice


def allocate_to_faces(pieces: list[list[Level]], faces: float, piece_share: float, seen: list[float]) -> list[int]:
    """Pick levels whose faces add up to about `faces`, losing as little as `allocate` can for that many.

    More allowed loss always means fewer faces, so the allowed loss is searched by halving. When the levels can't
    reach `faces` (the smallest ones are still above it, or the biggest are below), the nearest total is returned.
    """

    def total(choice: list[int]) -> int:
        return sum(levels[k].faces for levels, k in zip(pieces, choice))

    low, high = 0.0, sum(max(level.loss for level in levels) for levels in pieces)
    best = allocate(pieces, high, piece_share, seen)
    for _ in range(24):
        mid = (low + high) / 2.0
        choice = allocate(pieces, mid, piece_share, seen)
        if abs(total(choice) - faces) < abs(total(best) - faces):
            best = choice
        if total(choice) > faces:
            low = mid
        else:
            high = mid
    return best


def next_ask(tries: list[tuple[int, int]], wanted: int, low: int, high: int) -> int:
    """What to ask a layout for next, from (asked, got) pairs, so it gives `wanted` faces.

    A part of the result doesn't scale with what is asked for (small parts paired and closed), so after two tries the
    line through them is followed rather than the ratio of the last one.
    """

    (a0, g0), (a1, g1) = tries[-2:] if len(tries) > 1 else ((0, 0), tries[-1])
    slope = (g1 - g0) / (a1 - a0) if a1 != a0 and g1 != g0 else g1 / max(a1, 1)
    return min(max(int(round(a1 + (wanted - g1) / max(slope, 0.1))), low), high)


def density_drop(paint: np.ndarray, strength: float) -> np.ndarray:
    """How far below 1 a vertex's Decimate weight goes for a painted amount 0..1.

    Decimate leaves a vertex alone at weight 0 and collapses it freely at 1, and the pull is steep: 0.99 keeps about
    three times the faces, 0.98 four times, 0.9 six times. So a full coat of paint at strength 1 drops the weight by 0.02.
    """
    return np.clip(0.02 * strength * np.clip(paint, 0.0, 1.0), 0.0, 0.5)


def best_levels(levels: list[Level]) -> list[int]:
    """Indices of the levels, by faces, that lose less than every level with as few faces or fewer."""

    kept, lowest = [], float("inf")
    for index in sorted(range(len(levels)), key=lambda k: (levels[k].faces, levels[k].loss)):
        if levels[index].loss < lowest:
            kept.append(index)
            lowest = levels[index].loss
    return kept


def vertex_heat(vertices: np.ndarray, spots: np.ndarray, cell: float) -> np.ndarray:
    """For each vertex, how many of the `spots` lie in its grid cell or a neighboring one."""

    heat = np.zeros(len(vertices), dtype=np.int64)
    if not len(spots) or not len(vertices):
        return heat
    low = np.minimum(vertices.min(axis=0), spots.min(axis=0)) - 2.0 * cell
    wide = 3 + int(np.ceil((np.maximum(vertices.max(axis=0), spots.max(axis=0)) - low).max() / cell))
    stride = np.asarray((1, wide, wide * wide), dtype=np.int64)
    filled, counts = np.unique(((spots - low) / cell).astype(np.int64) @ stride, return_counts=True)
    at = ((vertices - low) / cell).astype(np.int64) @ stride
    for step in product((-1, 0, 1), repeat=3):
        key = at + int(np.asarray(step) @ stride)
        slot = np.minimum(np.searchsorted(filled, key), len(filled) - 1)
        match = filled[slot] == key
        heat[match] += counts[slot[match]]
    return heat


def hottest(heat: np.ndarray, count: float) -> np.ndarray:
    """Mask of the `count` hottest vertices, leaving out any with no heat."""

    keep = np.zeros(len(heat), dtype=bool)
    keep[np.argsort(-heat, kind="stable")[: min(int(count), int((heat > 0).sum()))]] = True
    return keep
