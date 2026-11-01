# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Sharing faces between the pieces of a mesh: most visible detail per face, with a floor no piece goes below."""

from __future__ import annotations

from dataclasses import dataclass
import heapq


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


def best_levels(levels: list[Level]) -> list[int]:
    """Indices of the levels, by faces, that lose less than every level with as few faces or fewer."""

    kept, lowest = [], float("inf")
    for index in sorted(range(len(levels)), key=lambda k: (levels[k].faces, levels[k].loss)):
        if levels[index].loss < lowest:
            kept.append(index)
            lowest = levels[index].loss
    return kept
