# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared set-up for the UV unit tests: the add-on folder on the path, `addon_module`, and small test meshes."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def addon_module(name: str):
    """Import `name` (e.g. "uv_layers") from the add-on folder as part of a bare package, so its relative imports work without bpy."""

    import importlib
    import types

    if "uv_addon_pkg" not in sys.modules:
        package = types.ModuleType("uv_addon_pkg")
        package.__path__ = [str(ROOT)]
        sys.modules["uv_addon_pkg"] = package
    return importlib.import_module(f"uv_addon_pkg.{name}")


def plane(nx, ny, w, h):
    """A flat nx x ny quad grid of size w x h: (points, indptr, flat)."""

    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    points = np.stack([w * i.ravel() / nx, h * j.ravel() / ny, 0.0 * i.ravel()], 1).astype(float)
    ids = (j * (nx + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + nx + 2, ids + nx + 1], 1).ravel()
    return points, np.arange(0, len(flat) + 1, 4), flat


def two_islands():
    """8x8 plane in two islands split at x = 4: left UV = (x, y)/8, right UV = (x/8 + 0.25, y/8)."""

    points, indptr, flat = plane(8, 8, 8.0, 8.0)
    right = np.repeat((np.arange(64) % 8) >= 4, 4)
    uv = np.stack([points[flat, 0] / 8 + 0.25 * right, points[flat, 1] / 8], 1).ravel()
    return points, indptr, flat, uv


def grid(nx, ny, h=1.0, z=lambda x, y: 0.0 * x):
    """An nx x ny quad grid with spacing h and height z(x, y): (points, indptr, flat)."""

    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    x, y = i.ravel() * h, j.ravel() * h
    points = np.stack([x, y, z(x, y)], 1)
    ids = (j * (nx + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + nx + 2, ids + nx + 1], 1).ravel()
    return points, np.arange(0, len(flat) + 1, 4), flat


def cube(k):
    """A unit cube of k x k quads per side, welded: (points, indptr, flat)."""

    ids, pts, flat = {}, [], []
    for ax, sg in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        if sg < 0:
            u, v = v, u

        def vid(i, j):
            c = [0, 0, 0]
            c[ax], c[u], c[v] = (k if sg > 0 else 0), i, j
            c = tuple(c)
            if c not in ids:
                ids[c] = len(pts)
                pts.append([x / k for x in c])
            return ids[c]

        for j in range(k):
            for i in range(k):
                flat += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
    return np.array(pts, float), np.arange(0, len(flat) + 1, 4), np.array(flat)
