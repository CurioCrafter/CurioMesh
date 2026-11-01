# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Plain-NumPy helpers for the segment view: what kind a part is, cutting a part out of a result, putting a new one in."""

from __future__ import annotations

import numpy as np

KINDS = ("TUBE", "FLAT", "ROUGH")
KIND_WORDS = {"TUBE": "Tube", "FLAT": "Flat", "ROUGH": "Rough"}


def classify(points: np.ndarray, faces) -> str:
    """TUBE (long and round), FLAT (a thin sheet) or ROUGH (anything else), from the spread of the part's vertices.

    A rough guide for picking parts in bulk, not a measurement: `s1 >= s2 >= s3` are the spreads along the three axes
    of the vertex cloud. Under ten vertices there is nothing to tell, so it is ROUGH.
    """

    used = np.unique(np.fromiter((v for f in faces for v in f), dtype=np.int64))
    if len(used) < 10:
        return "ROUGH"
    cloud = np.asarray(points, dtype=np.float64)[used]
    spread = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov((cloud - cloud.mean(axis=0)).T)), 0.0))[::-1]
    s1, s2, s3 = (float(x) for x in spread)
    if s1 <= 1e-12:
        return "ROUGH"
    if s3 < 0.08 * s1 and s3 < 0.4 * s2:
        return "FLAT"
    if s2 < 0.45 * s1 and s3 > 0.5 * s2:
        return "TUBE"
    return "ROUGH"


def cut(co: np.ndarray, faces, mat, face_piece: np.ndarray, piece: int):
    """One part of a result on its own: (points, faces, materials) with its own vertex numbers."""

    ids = np.flatnonzero(np.asarray(face_piece) == piece)
    used, flat = np.unique(np.fromiter((v for i in ids.tolist() for v in faces[i]), dtype=np.int64), return_inverse=True)
    out, at = [], 0
    for i in ids.tolist():
        size = len(faces[i])
        out.append(tuple(int(v) for v in flat[at : at + size]))
        at += size
    return np.asarray(co)[used], out, np.asarray(mat)[ids]


def splice(co: np.ndarray, faces, mat, face_piece: np.ndarray, builds: dict):
    """The result with every part in `builds` ({piece: (points, faces, materials)}) replaced.

    Returns (points, faces, materials, face_piece). Vertices nobody uses any more are dropped.
    """

    face_piece = np.asarray(face_piece)
    keep = ~np.isin(face_piece, list(builds))
    all_faces = [f for f, k in zip(faces, keep) if k]
    all_mat = list(np.asarray(mat)[keep])
    all_piece = list(face_piece[keep])
    pieces = [co]
    offset = len(co)
    for piece, (points, new_faces, new_mat) in builds.items():
        pieces.append(np.asarray(points, dtype=np.float64))
        all_faces += [tuple(int(v) + offset for v in f) for f in new_faces]
        all_mat += [int(m) if str(m).lstrip("-").isdigit() else 0 for m in new_mat]
        all_piece += [piece] * len(new_faces)
        offset += len(points)
    every = np.vstack(pieces)
    used, flat = np.unique(np.fromiter((v for f in all_faces for v in f), dtype=np.int64), return_inverse=True)
    out, at = [], 0
    for f in all_faces:
        out.append(tuple(int(v) for v in flat[at : at + len(f)]))
        at += len(f)
    return every[used], out, np.asarray(all_mat, dtype=np.int32), np.asarray(all_piece, dtype=np.int64)


def worst_first(changes, count: int | None = None) -> list[int]:
    """Part numbers, the part whose view changed most first. Parts with no number (None) come last."""

    order = sorted(range(len(changes)), key=lambda i: (changes[i] is None, -(changes[i] or 0.0), i))
    return order if count is None else order[:count]


def heat_colour(value: float | None, top: float) -> tuple[float, float, float]:
    """Green for no change through yellow to red at `top` percent or more. Grey when unknown."""

    if value is None:
        return 0.55, 0.55, 0.55
    t = min(max(value / max(top, 1e-9), 0.0), 1.0)
    return (min(2.0 * t, 1.0), min(2.0 * (1.0 - t), 1.0) * 0.85 + 0.1, 0.15)
