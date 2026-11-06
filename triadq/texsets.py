# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Group faces into texture sets, or into pieces under a game engine's triangle limit.

Both keep whole parts together where they can: a part is only cut, along its longest side, when it alone is too big.
"""

from __future__ import annotations

import numpy as np


def _cut(faces: np.ndarray, weight: np.ndarray, centres: np.ndarray, most: float) -> list[np.ndarray]:
    """Cut a group of faces along its longest side, `most` at a time, until each piece weighs at most `most`."""
    out, todo = [], [faces]
    while todo:
        group = todo.pop()
        if weight[group].sum() <= most or len(group) < 2:
            out.append(group)
            continue
        c = centres[group]
        axis = int(np.argmax(c.max(0) - c.min(0)))
        order = group[np.argsort(c[:, axis], kind="stable")]
        half = int(np.searchsorted(np.cumsum(weight[order]), most, side="right"))
        half = min(max(half, 1), len(order) - 1)
        todo += [order[:half], order[half:]]
    return out


def _groups(labels: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(labels, kind="stable")
    cuts = np.flatnonzero(np.diff(labels[order])) + 1
    return np.split(order, cuts)


def texture_sets(labels, area, centres, count: int) -> np.ndarray:
    """Per face, which of `count` sets it goes in: whole parts, sets of about equal area. Fewer sets if there is less to share."""
    labels, area, centres = np.asarray(labels), np.asarray(area, float), np.asarray(centres, float)
    count = max(1, int(count))
    if len(labels) == 0 or count == 1:
        return np.zeros(len(labels), np.int32)
    share = area.sum() / count
    pieces = [p for g in _groups(labels) for p in _cut(g, area, centres, share)]
    pieces.sort(key=lambda p: -area[p].sum())
    sums = np.zeros(min(count, len(pieces)))
    out = np.zeros(len(labels), np.int32)
    for p in pieces:  # biggest first into the emptiest set
        k = int(np.argmin(sums))
        out[p] = k
        sums[k] += area[p].sum()
    return out


def split_under(labels, tris, centres, limit: int) -> np.ndarray:
    """Per face, which piece it goes in so no piece has more than `limit` triangles. A piece never mixes two labels."""
    labels, tris, centres = np.asarray(labels), np.asarray(tris, float), np.asarray(centres, float)
    out = np.zeros(len(labels), np.int32)
    if len(labels) == 0:
        return out
    if limit <= 0 or tris.sum() <= limit and len(np.unique(labels)) == 1:
        return out
    made = 0
    for group in _groups(labels):
        pieces = sorted(_cut(group, tris, centres, limit), key=lambda p: -tris[p].sum())
        room: list[float] = []
        for p in pieces:  # first fit, biggest first
            k = next((i for i, r in enumerate(room) if r >= tris[p].sum()), None)
            if k is None:
                room.append(limit)
                k = len(room) - 1
            room[k] -= tris[p].sum()
            out[p] = made + k
        made += len(room)
    return out
