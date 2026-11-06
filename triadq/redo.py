# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Redo Rough Spots: the pure parts of redoing a small region of a finished result.

A region is carved round a rough spot (faces turned inward, folded over, crossing). Its inner corners are put back
towards the source, part or all of the way (undoing what fitting and repairs moved there), while every corner a face
outside the region uses stays put, so the region goes back with nothing to join. `carve` grows the region,
`free_corners` says which corners may move, `toward` moves them and `better` says whether a redo is kept. Plain NumPy.

Remaking a region's faces finer was tried first: cut finer, put on the source, reduced and paired again. Closing the
patch into quads splits its outline (the edge of a patch is where a leftover triangle is cheapest to pair), so it can't
go back without a T-junction, and on the spots measured it left more bad faces than it removed. Relaxing the corners
towards their neighbours bent many quads on rough scans.
"""

from __future__ import annotations

import numpy as np


def corner_users(faces) -> dict[int, list[int]]:
    """The faces round each corner, for `carve`."""
    users: dict[int, list[int]] = {}
    for f, face in enumerate(faces):
        for v in face:
            users.setdefault(v, []).append(f)
    return users


def carve(faces, seeds, rings: int, most: int, taken: np.ndarray | None = None, users: dict | None = None) -> np.ndarray:
    """Face numbers of the region: `seeds` grown `rings` times over shared corners, at most `most` faces.

    Faces in `taken` (a mask) are never added, so regions carved one after another don't overlap. Pass `users`
    (`corner_users`) when carving many regions of one mesh.
    """
    users = corner_users(faces) if users is None else users
    inside = {int(f) for f in seeds if taken is None or not taken[f]}
    front = set(inside)
    for _ring in range(rings):
        grown = set()
        for f in front:
            for v in faces[f]:
                grown.update(g for g in users[v] if g not in inside and (taken is None or not taken[g]))
        if not grown or len(inside) + len(grown) > most:
            break
        inside |= grown
        front = grown
    return np.asarray(sorted(inside), dtype=np.int64)


def free_corners(faces, region, count: int) -> np.ndarray:
    """True for the corners only faces of the region use: the ones a redo may move without touching the rest."""
    inside = np.zeros(len(faces), bool)
    inside[region] = True
    used_in, used_out = np.zeros(count, bool), np.zeros(count, bool)
    for f, face in enumerate(faces):
        (used_in if inside[f] else used_out)[list(face)] = True
    return used_in & ~used_out


def toward(points: np.ndarray, targets: np.ndarray, near: np.ndarray, share: float) -> np.ndarray:
    """`points` moved `share` of the way to their `targets` (their nearest source points), where `near` says the target
    is close enough to trust: a target across a thin part would turn the faces over."""
    out = np.array(points, dtype=np.float64)
    out[near] += share * (np.asarray(targets)[near] - out[near])
    return out


def better(old: tuple[int, int], new: tuple[int, int]) -> bool:
    """Keep a redo only if it leaves fewer bad faces, and no more bent quads (bad faces, bent quads)."""
    return new[0] < old[0] and new[1] <= old[1]
