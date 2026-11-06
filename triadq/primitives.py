# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Low-poly primitives fitted to a part: an oriented box, a cylinder and an ellipsoid, all quads.

Every fit is one shape in the same form: a centre, three unit axes (rows) and a half size along each, so a unit
primitive in [-1, 1]^3 is scaled, turned and moved onto the part. The box is the smallest-volume one found by turning
the principal axes; the cylinder and the ellipsoid share its frame. Which fit is used is the camera's call (`view_miss`).
Plain NumPy.
"""

from __future__ import annotations

import numpy as np

TURN_STEP_DEG = 3.0  # the box search turns the frame in steps this fine round each axis, then finer round the best
SEARCH_POINTS = 3000  # the box search looks at this many of the points; the final box holds all of them
THINNEST = 0.005  # a half size is at least this share of the largest, so a flat part still gives a solid


def _frame(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    return np.linalg.svd(centred, full_matrices=False)[2] if len(points) >= 3 else np.eye(3)


def _volume(points: np.ndarray, axes: np.ndarray) -> float:
    return float(np.prod(np.ptp(points @ axes.T, axis=0)))


def _turned(axes: np.ndarray, k: int, angle: float) -> np.ndarray:
    i, j = (k + 1) % 3, (k + 2) % 3
    out = axes.copy()
    c, s = np.cos(angle), np.sin(angle)
    out[i], out[j] = c * axes[i] + s * axes[j], -s * axes[i] + c * axes[j]
    return out


def _shape(points: np.ndarray, axes: np.ndarray) -> dict:
    """The box of `points` in this frame, longest axis first and right-handed."""
    local = points @ axes.T
    low, high = local.min(axis=0), local.max(axis=0)
    half = (high - low) / 2.0
    order = np.argsort(-half, kind="stable")
    centre = (low + high) / 2.0 @ axes  # the middle in the frame, back in the world
    axes, half = axes[order], half[order]
    axes[2] = np.cross(axes[0], axes[1])
    return {"center": centre, "axes": axes, "half": np.maximum(half, THINNEST * float(half.max()) + 1e-12)}


def fit_box(points: np.ndarray) -> dict:
    """The oriented box of smallest volume found by turning the principal frame round each of its axes."""
    points = np.asarray(points, dtype=np.float64)
    some = points[np.random.default_rng(0).choice(len(points), SEARCH_POINTS, replace=False)] if len(points) > SEARCH_POINTS else points
    axes = _frame(some)
    best = _volume(some, axes)
    for step, span, rounds in ((TURN_STEP_DEG, 90.0, 4), (0.5, TURN_STEP_DEG, 2)):  # coarse turns, then fine ones round the best
        angles = np.radians(np.arange(-span, span + 1e-9, step))
        for _round in range(rounds):
            for k in range(3):
                for angle in angles:
                    trial = _turned(axes, k, angle)
                    volume = _volume(some, trial)
                    if volume < best * (1.0 - 1e-9):
                        best, axes = volume, trial
    return dict(_shape(points, axes), kind="box")


def fit_cylinder(points: np.ndarray, box: dict | None = None) -> dict:
    """The cylinder round one of the box's axes that holds the points in the least volume. Its axis is axes[2]."""
    points = np.asarray(points, dtype=np.float64)
    box = fit_box(points) if box is None else box
    local = (points - box["center"]) @ box["axes"].T
    best = None
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        radius = float(np.percentile(np.hypot(local[:, i], local[:, j]), 99.5))
        length = float(box["half"][k])
        volume = radius * radius * length
        if best is None or volume < best[0]:
            best = (volume, (i, j, k), radius, length)
    _volume_, order, radius, length = best
    axes = box["axes"][list(order)].copy()
    axes[2] = np.cross(axes[0], axes[1])
    return {"kind": "cylinder", "center": box["center"].copy(), "axes": axes, "half": np.array([radius, radius, length])}


def fit_ellipsoid(points: np.ndarray, box: dict | None = None) -> dict:
    """The ellipsoid touching the box's sides: points on a sphere give back that sphere."""
    box = fit_box(points) if box is None else box
    return {"kind": "ellipsoid", "center": box["center"].copy(), "axes": box["axes"].copy(), "half": box["half"].copy()}


def unit_box() -> tuple[np.ndarray, list[tuple[int, ...]]]:
    v = np.array([(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    return v, [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def unit_cylinder(sides: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Radius 1 round z, from -1 to 1. Even sides; each cap is a fan of quads round its middle."""
    sides = max(4, int(sides) + int(sides) % 2)
    t = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.stack([np.cos(t), np.sin(t)], axis=1)
    v = np.vstack([np.column_stack([ring, np.full(sides, -1.0)]), np.column_stack([ring, np.full(sides, 1.0)]), [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]])
    faces = [(i, (i + 1) % sides, sides + (i + 1) % sides, sides + i) for i in range(sides)]
    low, high = 2 * sides, 2 * sides + 1
    for i in range(0, sides, 2):
        faces.append((low, (i + 2) % sides, i + 1, i))
        faces.append((high, sides + i, sides + i + 1, sides + (i + 2) % sides))
    return v, faces


def unit_sphere(sides: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """A quad sphere: a cube of n by n quads a side pushed out to radius 1, with n = sides / 4 (6 n^2 quads)."""
    n = max(1, int(round(sides / 4.0)))
    index: dict[tuple, int] = {}
    points: list = []
    faces = []
    grid = np.linspace(-1.0, 1.0, n + 1)

    def at(p):
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(points)
            points.append(p)
        return index[key]

    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, w = (axis + 1) % 3, (axis + 2) % 3
            ids = np.empty((n + 1, n + 1), dtype=np.int64)
            for a in range(n + 1):
                for b in range(n + 1):
                    p = np.zeros(3)
                    p[axis], p[u], p[w] = sign, grid[a], grid[b]
                    ids[a, b] = at(p)
            for a in range(n):
                for b in range(n):
                    faces.append((int(ids[a, b]), int(ids[a + 1, b]), int(ids[a + 1, b + 1]), int(ids[a, b + 1])))
    v = np.asarray(points)
    return v / np.linalg.norm(v, axis=1, keepdims=True), faces


def outward(points: np.ndarray, faces, centre: np.ndarray) -> list[tuple[int, ...]]:
    """Faces of a convex shape wound to face away from its middle."""
    out = []
    for face in faces:
        p = points[list(face)]
        normal = np.cross(p[1] - p[0], p[2] - p[0]) + (np.cross(p[2] - p[0], p[3] - p[0]) if len(face) > 3 else 0.0)
        out.append(tuple(face) if float(normal @ (p.mean(axis=0) - centre)) >= 0.0 else tuple(reversed(face)))
    return out


def build(fit: dict, sides: int = 12) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """The fit as a mesh: (points, quads)."""
    unit = {"box": unit_box, "cylinder": lambda: unit_cylinder(sides), "ellipsoid": lambda: unit_sphere(sides)}[fit["kind"]]
    v, faces = unit()
    points = (v * fit["half"]) @ fit["axes"] + fit["center"]
    return points, outward(points, faces, fit["center"])


def fits(points: np.ndarray, box: dict | None = None) -> list[dict]:
    """Every primitive fitted to these points: box, cylinder, ellipsoid."""
    box = fit_box(points) if box is None else box
    return [box, fit_cylinder(points, box), fit_ellipsoid(points, box)]


def spread_points(points: np.ndarray, count: int) -> np.ndarray:
    """Indices of `count` points spread out as far as they go (farthest point sampling), for a hull of fewer faces."""
    points = np.asarray(points, dtype=np.float64)
    if count >= len(points):
        return np.arange(len(points))
    picked = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    gap = np.linalg.norm(points - points[picked[0]], axis=1)
    for _ in range(count - 1):
        picked.append(int(np.argmax(gap)))
        gap = np.minimum(gap, np.linalg.norm(points - points[picked[-1]], axis=1))
    return np.asarray(picked)


def view_miss(reference, other, depth_tol: float) -> float:
    """Share of the pixels over the six pictures where the outline differs or the surface moved past `depth_tol`.
    Looser than the strict camera check on purpose: a blockout is judged by its silhouette and its bulk, not its shading."""
    wrong = total = 0
    for (hit_a, _na, depth_a, _fa), (hit_b, _nb, depth_b, _fb) in zip(reference, other):
        both = hit_a & hit_b
        wrong += int((hit_a ^ hit_b).sum()) + int((np.abs(depth_a - depth_b)[both] > depth_tol).sum())
        total += int((hit_a | hit_b).sum())
    return wrong / max(total, 1)
