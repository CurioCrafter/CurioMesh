# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.shading import blended_normals, corner_normals, corner_table, sharp_edges  # noqa: E402


class ShadingTests(unittest.TestCase):
    def test_flat_quad_shades_flat(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        self.assertTrue(np.allclose(corner_normals(points, [(0, 1, 2, 3)]), [0, 0, 1]))

    def test_a_ridge_blends_when_gentle_and_stays_hard_past_the_angle(self):
        # Two faces meeting along x = 0, each 45 degrees off level, so 90 degrees between them.
        points = np.array([[0, 0, 0], [0, 1, 0], [-1, 1, -1], [-1, 0, -1], [1, 0, -1], [1, 1, -1]], dtype=np.float64)
        polys = [(3, 0, 1, 2), (0, 4, 5, 1)]
        slope = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        hard = corner_normals(points, polys)
        self.assertTrue(np.allclose(hard[4], slope))  # the second face's corner on the ridge keeps its own normal
        soft = corner_normals(points, polys, 120.0)
        self.assertTrue(np.allclose(soft[4], [0, 0, 1]))
        at = np.array([[0.0, 0.5, 0.0], [1.0, 0.25, -1.0], [1.0, 0.75, -1.0]])
        blended = blended_normals(points, corner_table(polys), soft, np.array([1, 1, 1]), at)
        self.assertTrue(np.allclose(blended, [[0, 0, 1], slope, slope]))

    def test_sharp_edges_past_the_angle_or_shared_by_many(self):
        normals = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
        # Edge 0 joins faces 0 and 1 (flat), edge 1 joins 1 and 2 (a right angle), edge 2 has three faces, edge 3 one.
        edges = np.array([0, 0, 1, 1, 2, 2, 2, 3])
        owners = np.array([0, 1, 1, 2, 0, 1, 2, 2])
        self.assertEqual(sharp_edges(edges, owners, normals, 4).tolist(), [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
