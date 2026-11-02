# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triadq import MeshData, topology_summary  # noqa: E402
from triadq.flatquad import pair_and_close  # noqa: E402
from triadq.patch import _half_edges, quad_quality  # noqa: E402
from triadq.smoothgrid import _strip, grid_areas, smooth_areas  # noqa: E402


def capsule(sides: int = 24, stacks: int = 6, length: float = 4.0):
    """Closed triangulated capsule along Y: unit hemispheres joined by one long band."""

    rows = []
    for y0, sign in ((-length / 2.0, -1.0), (length / 2.0, 1.0)):
        angles = np.linspace(np.pi / 2.0, 0.0, stacks + 1)[1:] if sign < 0 else np.linspace(0.0, np.pi / 2.0, stacks + 1)[:-1]
        rows.extend((y0 + sign * np.sin(t), np.cos(t)) for t in angles)
    around = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    points = [(radius * np.cos(t), y, radius * np.sin(t)) for y, radius in rows for t in around]
    faces = []
    for r in range(len(rows) - 1):
        for k in range(sides):
            a, b = r * sides + k, r * sides + (k + 1) % sides
            c, d = a + sides, b + sides
            faces += [(a, b, d), (a, d, c)]
    south, north = len(points), len(points) + 1
    points += [(0.0, -length / 2.0 - 1.0, 0.0), (0.0, length / 2.0 + 1.0, 0.0)]
    top = (len(rows) - 1) * sides
    for k in range(sides):
        faces.append((south, (k + 1) % sides, k))
        faces.append((north, top + k, top + (k + 1) % sides))
    vertices = np.asarray(points, dtype=np.float64)
    return vertices, faces, length


def lookups(vertices, faces, length):
    middles = vertices[np.asarray(faces)].mean(axis=1)

    def on_capsule(points):
        along = np.clip(points[:, 1], -length / 2.0, length / 2.0)
        axis = np.stack((np.zeros(len(points)), along, np.zeros(len(points))), axis=1)
        offset = points - axis
        return axis + offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-12)

    def nearest(points):
        rows = np.argmin(np.linalg.norm(points[:, None] - middles[None], axis=2), axis=1)
        return rows, on_capsule(points)

    return nearest


class SmoothGridTests(unittest.TestCase):
    def test_finds_domes_not_the_band(self):
        vertices, faces, length = capsule()
        labels = smooth_areas(vertices, faces)
        self.assertEqual(int(labels.max()), 1)
        band = np.ptp(vertices[np.asarray(faces)][:, :, 1], axis=1) > 0.5 * length
        self.assertTrue((labels[band] < 0).all())
        self.assertGreater((labels >= 0).mean(), 0.9)

    def test_rough_surface_is_left_alone(self):
        rng = np.random.default_rng(3)
        vertices, faces, _length = capsule()
        rough = vertices * (1.0 + 0.05 * rng.standard_normal((len(vertices), 1)))
        self.assertLess(int(smooth_areas(rough, faces).max()), 0)

    def test_domes_become_grids_that_conform(self):
        vertices, faces, length = capsule()
        labels = smooth_areas(vertices, faces)
        nearest = lookups(vertices, faces, length)
        coarse = MeshData(vertices, faces, ["0"] * len(faces))
        result, report = grid_areas(coarse, labels, nearest)
        self.assertEqual(report.built, 2)
        quads = np.asarray([face for face in result.faces if len(face) == 4])
        self.assertGreater((quad_quality(result.vertices[quads]) > 0.0).mean(), 0.999)
        closed = pair_and_close(result, seed_count=1, keep_quads=True)[0]
        summary = topology_summary(closed)
        self.assertTrue(summary.all_quads)
        self.assertEqual(summary.boundary_edges, 0)
        # Inside a dome, only the grid's four poles are irregular.
        _face, used, _next = _half_edges(closed.faces)
        valence = np.bincount(used)
        points = closed.vertices
        dome = np.abs(points[:, 1]) > 0.5 * length + 0.45
        self.assertLessEqual(int((valence[dome] != 4).sum()), 2 * 8)

    def test_strip_closes_rows_without_flat_corners(self):
        quads, extra = _strip(np.arange(12), 100 + np.arange(8), 200)
        self.assertEqual(len(extra), 4)
        uses = {}
        for quad in quads:
            for a, b in zip(quad, quad[1:] + quad[:1]):
                uses[(min(a, b), max(a, b))] = uses.get((min(a, b), max(a, b)), 0) + 1
        self.assertEqual(sum(1 for k, n in uses.items() if n == 1), 20)  # just the two rows are open
        for quad in quads:
            self.assertLessEqual(sum(v < 12 for v in quad), 2)  # never three in a line along a row


if __name__ == "__main__":
    unittest.main()
