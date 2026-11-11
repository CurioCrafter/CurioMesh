# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _nobvh  # noqa: E402,F401  (no BVH outside Blender)

from triadq.fit import _run, corner_normals_steps, fit_vertices, untangled  # noqa: E402


def grid(count: int, z: float = 0.0):
    x, y = np.meshgrid(np.arange(count, dtype=np.float64), np.arange(count, dtype=np.float64), indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), np.full(count * count, z)], axis=1)
    quads = [(i * count + j, (i + 1) * count + j, (i + 1) * count + j + 1, i * count + j + 1) for i in range(count - 1) for j in range(count - 1)]
    return points, np.asarray(quads)


class FitVerticesTests(unittest.TestCase):
    def test_flat_area_moves_along_its_normal_only(self):
        points, quads = grid(3, z=0.1)
        sample = np.array([[1.2, 1.1, 0.0], [0.9, 0.8, 0.0], [1.1, 0.9, 0.0]])
        normal = np.tile([0.0, 0.0, 1.0], (3, 1))
        moved = fit_vertices(points, quads, sample, normal, np.ones(3), np.array([4, 4, 4]))
        np.testing.assert_allclose(moved[4], [1.0, 1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(np.delete(moved, 4, axis=0), np.delete(points, 4, axis=0))

    def test_a_ridge_vertex_lands_on_the_ridge_without_rounding_it(self):
        points, quads = grid(3)
        points[4] = [1.1, 1.0, 0.1]
        # A roof along y with its ridge over x=1, z=0.
        slope = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]) / np.sqrt(2.0)
        sample = np.array([[0.9, 1.0, -0.1], [1.1, 1.0, -0.1]])
        moved = fit_vertices(points, quads, sample, slope, np.ones(2), np.array([4, 4]))
        np.testing.assert_allclose(moved[4], [1.0, 1.0, 0.0], atol=1e-9)

    def test_sample_facing_away_is_the_far_side_of_a_thin_wall(self):
        points, quads = grid(3)
        sample = np.array([[1.0, 1.0, -0.2]])
        moved = fit_vertices(points, quads, sample, np.array([[0.0, 0.0, -1.0]]), np.ones(1), np.array([4]))
        np.testing.assert_allclose(moved, points)

    def test_a_move_that_would_fold_a_quad_goes_back(self):
        points, quads = grid(3)
        # A wide plane tilted hard, pulling one vertex far past its neighbours is cut to the move limit or given back.
        sample = np.array([[1.0, 1.0, 100.0]])
        moved = fit_vertices(points, quads, sample, np.array([[0.0, 0.0, 1.0]]), np.ones(1), np.array([4]))
        self.assertLessEqual(np.linalg.norm(moved[4] - points[4]), 0.3 + 1e-9)

    def test_corners_that_fold_a_quad_over_its_neighbour_go_back(self):
        points, quads = grid(4)
        after = points.copy()
        far = [12, 13, 14, 15]
        after[far, 0] = 1.5  # the far row of quads swung back over the row before it
        after[far, 2] = 0.2
        after[0, 2] = 0.01  # a harmless move away from it stays
        kept = untangled(points, after, quads)
        np.testing.assert_allclose(kept[far], points[far])
        self.assertEqual(kept[0, 2], 0.01)


class CornerNormalTests(unittest.TestCase):
    def test_another_loose_part_is_not_heard(self):
        points, quads = grid(2)
        normals = np.array([[0.0, 0.3, 1.0], [0.0, -0.3, 1.0]])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        out = _run(corner_normals_steps(points, [tuple(q) for q in quads.tolist()], normals, np.ones(2), np.array([0, 1]), np.zeros(4, dtype=np.int64), lambda p, c: [(None, 0, 0.1), (None, 1, 0.1)], 2))
        np.testing.assert_allclose(np.abs(out), np.tile(np.abs(normals[0]), (4, 1)), atol=2e-3)

    def test_a_corner_hears_its_own_side_of_a_hard_edge(self):
        # Two faces meeting in a right angle along x=0: floor (z=0, x>0) and wall (x=0, z>0).
        points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=np.float64)
        faces = [(0, 3, 2, 1), (0, 1, 5, 4)]
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        def find_n(point, count):
            on_floor = point[2] < point[0]
            return [(None, 0 if on_floor else 1, 0.1)]

        out = _run(corner_normals_steps(points, faces, normals, np.ones(2), np.zeros(2, dtype=np.int64), np.zeros(6, dtype=np.int64), find_n, 1))
        np.testing.assert_allclose(np.abs(out[:4]), np.tile([0.0, 0.0, 1.0], (4, 1)), atol=1e-9)
        np.testing.assert_allclose(np.abs(out[4:]), np.tile([1.0, 0.0, 0.0], (4, 1)), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
