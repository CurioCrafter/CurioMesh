# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The maths of each Tidy Tool, as a `trial.Trial` subclass (plain NumPy, no bpy).

Smooth Corners is the small example every tool follows: it slides each free corner part of the way toward the middle
of its neighbours (umbrella smoothing, Taubin 1995 without the shrink step, as the projection undoes the shrink) and
puts it back on your original mesh after each step, so nothing shrinks and every corner stays on the source surface.
"""

from __future__ import annotations

import numpy as np

from .trial import Base, Candidate, Trial, edges_of

NOTHING_FREE = "Nothing can move here: every corner is held (borders, hard edges or the mirror plane)."


class SmoothCorners(Trial):
    name = "Smooth Corners"
    group = "RELAX"
    handles = ("strength",)
    words = ("0", "1")
    defaults = {"strength": 0.6, "steps": 6}
    blendable = True
    stages = ("poles", "move", "look")

    def check(self, base: Base) -> str:
        return "" if base.free.any() else NOTHING_FREE

    def run(self, base: Base, settings: dict, rough: bool = False):
        strength = float(settings.get("strength", self.defaults["strength"]))
        steps = int(settings.get("steps", self.defaults["steps"]))
        steps = max(1, steps // 4) if rough else max(1, steps)
        free = base.free
        co = base.co.copy()
        edges = edges_of(base.faces)
        count = np.bincount(edges.ravel(), minlength=len(co)).astype(np.float64)
        count[count == 0] = 1.0
        for step in range(steps):
            middle = np.stack([np.bincount(edges[:, 0], co[edges[:, 1], k], len(co)) + np.bincount(edges[:, 1], co[edges[:, 0], k], len(co)) for k in range(3)], axis=1) / count[:, None]
            co[free] += 0.5 * strength * (middle[free] - co[free])
            yield (step + 0.5) / steps
            if base.project is not None:
                co[free] = base.project(co[free])
            yield (step + 1.0) / steps
        return Candidate(co, strength=strength, exact=not rough)


# --------------------------------------------------------------------------- Look First for tools that already existed


def swap_parent(old_piece, new_piece) -> np.ndarray:
    """For each face of a result with some parts swapped, a face of the result before it in the same part (the first
    one), so the preview can tell the swapped parts' new faces from the rest. -1 where the part had no face before."""
    old_piece, new_piece = np.asarray(old_piece, dtype=np.int64), np.asarray(new_piece, dtype=np.int64)
    if not len(new_piece):
        return np.zeros(0, dtype=np.int64)
    top = int(max(old_piece.max(initial=-1), new_piece.max(initial=-1))) + 1
    first = np.full(top, -1, dtype=np.int64)
    order = np.arange(len(old_piece), dtype=np.int64)[::-1]
    first[old_piece[order]] = order  # the last write wins: the first face of each part
    return first[new_piece]


class SwapParts(Trial):
    """Some parts of the result with other faces: a ready Quad Size, your own faces, or the reduced ones again. The
    faces are worked out before the tool opens (they already exist as a part's try); this only shows them."""

    name = "Swap Parts"
    group = "LOOPS"
    handles = ()
    defaults: dict = {}
    blendable = False
    layers = ("scope", "ghost", "poles", "held")
    stages = ("faces", "poles", "look")
    swap = None  # (co, faces, new piece per face, old piece per face, parts swapped): set on a copy of the class

    def check(self, base: Base) -> str:
        if self.swap is None:
            return "Nothing to swap here: these parts already show these faces."
        return ""

    def run(self, base: Base, settings: dict, rough: bool = False):
        co, faces, new_piece, old_piece, parts = self.swap
        parent = swap_parent(old_piece, new_piece)
        yield 0.5
        inside = np.isin(np.asarray(new_piece), list(parts))
        parent = np.where(parent >= 0, parent, 0)
        before, after = int(base.scope.sum()), int(inside.sum())
        word = "parts" if len(parts) != 1 else "part"
        return Candidate(np.asarray(co, dtype=np.float64), faces=list(faces), parent=parent, strength=1.0, note=f"this {word} {before:,} -> {after:,} faces")


# Detail Markers. Decimate weight drop -> about how many times the faces it keeps (the numbers `budget.density_drop`
# is written from); a marker at strength s drops the weight by 0.02 * Strength * s at its middle.
MARKER_DROP = ((0.0, 1.0), (0.01, 3.0), (0.02, 4.0), (0.1, 6.0))
GRID_MOST = 40  # sample quads across the ring at most: a drawing cap so a tiny size never draws millions of lines


def marker_faces_factor(strength: float, density: float = 1.0) -> float:
    """About how many times the faces a marker asks for at its middle (More Detail; Less Detail asks for as many
    times fewer). An estimate: the search still decides."""
    drop = min(0.02 * max(float(density), 0.0) * abs(float(strength)), 0.5)
    x, y = zip(*MARKER_DROP)
    return float(np.interp(drop, x, y))


def marker_scale(strength: float, density: float = 1.0, less: bool = False) -> float:
    """How many times longer a quad's side gets at the marker's middle: under 1 for More Detail, over 1 for Less."""
    side = float(np.sqrt(marker_faces_factor(strength, density)))
    return side if less else 1.0 / side


def marker_extra_faces(radius: float, edge: float, strength: float, density: float = 1.0, less: bool = False) -> int:
    """About how many faces the marker adds (negative: takes away) on a flat disc of `radius` covered by quads of side
    `edge`. The ask falls off as the ball's weight does (1 in the middle, 0 at the edge: (1 - d^2)^2), ring by ring."""
    if radius <= 0.0 or edge <= 0.0:
        return 0
    rings = np.linspace(0.0, 1.0, 33)
    mid = 0.5 * (rings[:-1] + rings[1:])
    area = np.pi * radius**2 * (rings[1:] ** 2 - rings[:-1] ** 2)
    weight = (1.0 - mid**2) ** 2
    factor = np.array([marker_faces_factor(strength * w, density) for w in weight])
    now = area / edge**2
    after = now / factor if less else now * factor
    return int(round(float((after - now).sum())))


def sample_grid(radius: float, size: float):
    """A square grid of quads of side `size` inside a circle of `radius`, in its own plane: ((k, 2) points, (m, 4)
    quads). The middle of the circle is a grid corner. At most GRID_MOST quads across (bigger squares past that)."""
    if radius <= 0.0 or size <= 0.0:
        return np.zeros((0, 2)), np.zeros((0, 4), dtype=np.int64)
    size = max(size, 2.0 * radius / GRID_MOST)
    n = int(np.ceil(radius / size))
    ticks = np.arange(-n, n + 1) * size
    u, v = np.meshgrid(ticks, ticks, indexing="xy")
    points = np.stack((u.ravel(), v.ravel()), axis=1)
    side = 2 * n + 1
    i, j = np.meshgrid(np.arange(2 * n), np.arange(2 * n), indexing="xy")
    a = (j * side + i).ravel()
    quads = np.stack((a, a + 1, a + side + 1, a + side), axis=1)
    centre = points[quads].mean(axis=1)
    keep = (centre**2).sum(axis=1) <= radius**2
    quads = quads[keep]
    used = np.unique(quads)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[quads]


# Hard Edges on Apply: which edges of the result Apply will mark sharp (or hold with loops), shown before Apply.
ANGLE_LOW, ANGLE_HIGH = 5.0, 95.0  # degrees at the dial's two ends (the setting itself goes to 180 in the panel)


def edge_angles(co, faces) -> tuple[np.ndarray, np.ndarray]:
    """((e, 2) edges used by exactly two faces, (e,) degrees between those two faces' normals). Edges on a border or
    shared by more than two faces are left out, as Apply leaves them out."""
    co = np.asarray(co, dtype=np.float64)
    if not faces:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    normals = np.zeros((len(faces), 3))
    by_size: dict[int, list] = {}
    for i, f in enumerate(faces):
        by_size.setdefault(len(f), []).append(i)
    rows, pairs = [], []
    for _size, which in by_size.items():  # one NumPy pass per face size (quads, triangles...)
        which = np.asarray(which, dtype=np.int64)
        ids = np.asarray([faces[i] for i in which.tolist()], dtype=np.int64)
        p = co[ids]
        normals[which] = np.cross(p, np.roll(p, -1, axis=1)).sum(axis=1)  # Newell's normal
        nxt = np.roll(ids, -1, axis=1)
        pairs.append(np.stack((np.minimum(ids, nxt).ravel(), np.maximum(ids, nxt).ravel()), axis=1))
        rows.append(np.repeat(which, ids.shape[1]))
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)
    pairs, rows = np.concatenate(pairs), np.concatenate(rows)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, rows = pairs[order], rows[order]
    found, start, count = np.unique(pairs, axis=0, return_index=True, return_counts=True)
    two = count == 2
    a, b = rows[start[two]], rows[start[two] + 1]
    cos = np.clip((normals[a] * normals[b]).sum(axis=1), -1.0, 1.0)
    return found[two], np.degrees(np.arccos(cos))


def angle_of(value: float) -> float:
    """Degrees for a dial value 0..1."""
    return ANGLE_LOW + (ANGLE_HIGH - ANGLE_LOW) * min(max(float(value), 0.0), 1.0)


def value_of(angle: float) -> float:
    """The dial value 0..1 for an angle in degrees."""
    return min(max((float(angle) - ANGLE_LOW) / (ANGLE_HIGH - ANGLE_LOW), 0.0), 1.0)


class HardEdges(Trial):
    """Every edge of the result whose two faces meet at more than the angle, drawn pink: what Apply marks sharp (or
    holds with loops). Nothing moves; Enter keeps the angle for Apply."""

    name = "Hard Edges"
    group = "HARD"
    handles = ("angle",)
    words = (f"{ANGLE_LOW:.0f}", f"{ANGLE_HIGH:.0f}")
    defaults = {"angle": value_of(40.0)}
    blendable = False
    layers = ("scope", "kept")
    stages = ()

    def value_text(self, value: float) -> str:
        return f"Angle {angle_of(value):.0f} degrees"

    def check(self, base: Base) -> str:
        return "" if len(base.faces) > 1 else "No hard edges in this area."

    def run(self, base: Base, settings: dict, rough: bool = False):
        if "edge_angles" not in base.cache:
            base.cache["edge_angles"] = edge_angles(base.co, base.faces)
            yield 0.8
        edges, angles = base.cache["edge_angles"]
        angle = angle_of(settings.get("angle", self.defaults["angle"]))
        hard = edges[angles > angle]
        word = "edge" if len(hard) == 1 else "edges"
        return Candidate(base.co.copy(), strength=float(settings.get("angle", 0.0)), lines=base.co[hard], note=f"{len(hard):,} hard {word} at {angle:.0f} degrees, in pink")
