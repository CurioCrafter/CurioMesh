# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.fieldmesh import field_quads, is_clean, long_creases  # noqa: E402
from triadq.patterns import measure  # noqa: E402


def capsule(rows: int = 40, around: int = 60):
    """A closed smooth tube with domes, as triangles: nothing like a grid in its triangulation."""

    angle = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    lat = np.linspace(-1.0, 1.0, rows)
    radius = np.sqrt(np.maximum(1.0 - lat**8, 0.0))
    points = [(0.0, 0.0, -2.0)]
    for z, r in zip(lat[1:-1], radius[1:-1]):
        points += [(r * np.cos(a), r * np.sin(a), 2.0 * z) for a in angle]
    points.append((0.0, 0.0, 2.0))
    faces, at = [], lambda i, j: 1 + i * around + j % around
    for j in range(around):
        faces.append((0, at(0, j + 1), at(0, j)))
        faces.append((len(points) - 1, at(rows - 3, j), at(rows - 3, j + 1)))
        for i in range(rows - 3):
            faces += [(at(i, j), at(i, j + 1), at(i + 1, j + 1)), (at(i, j), at(i + 1, j + 1), at(i + 1, j))]
    return np.asarray(points), faces


class FieldMeshTests(unittest.TestCase):
    def test_rows_on_a_smooth_tube(self) -> None:
        points, faces = capsule()
        out, quads = field_quads(points, faces, 400)
        found = measure(out, quads)
        self.assertGreater(found["quads"], 240)
        self.assertLess(found["quads"], 600)
        self.assertLess(found["tris"], 0.05 * found["quads"])
        self.assertEqual(found["non_manifold_edges"], 0)
        self.assertLess(found["irregular_share_of_vertices"], 0.25)
        self.assertGreater(found["loop_median"], 5)
        away = np.linalg.norm(out[:, :2], axis=1)
        self.assertLess(float(away.max()), 1.05)  # stays on the surface

    def test_cells_can_follow_bends(self) -> None:
        points, faces = capsule()
        self.assertGreater(len(field_quads(points, faces, 400, 1.5)[1]), 200)

    def test_clean_means_all_quads_unfolded_and_manifold(self) -> None:
        points = np.array([(0.0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)])
        self.assertTrue(is_clean(points, [(0, 1, 2, 3), (1, 4, 5, 2)]))
        self.assertFalse(is_clean(points, [(0, 1, 2), (1, 4, 5, 2)]))
        self.assertFalse(is_clean(points, [(0, 2, 1, 3)]))
        self.assertFalse(is_clean(points, [(0, 1, 2, 3), (1, 4, 5, 2), (1, 2, 3, 4)]))

    def test_short_or_crossed_creases_are_not_lines(self) -> None:
        points = np.array([(float(k), 0.0, 0.0) for k in range(11)] + [(5.0, 1.0, 0.0), (5.0, -1.0, 0.0)])
        a, b = np.array([0, 1, 2, 3, 5, 5, 8]), np.array([1, 2, 3, 4, 11, 12, 9])
        found = long_creases(points, a, b, np.radians(np.full(7, 30.0)), share=0.3)
        self.assertEqual(found.tolist(), [True, True, True, True, False, False, False])

    def test_too_many_quads_are_refused_with_a_message(self) -> None:
        points, faces = capsule(8, 8)
        with self.assertRaisesRegex(ValueError, "stop at"):
            field_quads(points, faces, 10_000_000)

    def test_refuses_a_grid_too_small_to_lay(self) -> None:
        points, faces = capsule(8, 8)
        with self.assertRaises(ValueError):
            field_quads(points, faces, 8)


if __name__ == "__main__":
    unittest.main()
