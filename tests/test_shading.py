# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.shading import blended_normals, corner_table, vertex_normals  # noqa: E402


class ShadingTests(unittest.TestCase):
    def test_flat_quad_shades_flat(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        normals = vertex_normals(points, [(0, 1, 2, 3)])
        self.assertTrue(np.allclose(normals, [0, 0, 1]))

    def test_the_normal_blends_across_a_ridge_in_either_half_of_a_quad(self):
        # Two faces meeting at a right angle along x = 0: the ridge's normal lies halfway between them.
        points = np.array([[0, 0, 0], [0, 1, 0], [-1, 1, -1], [-1, 0, -1], [1, 0, -1], [1, 1, -1]], dtype=np.float64)
        polys = [(3, 0, 1, 2), (0, 4, 5, 1)]
        normals = vertex_normals(points, polys)
        self.assertTrue(np.allclose(normals[0], [0, 0, 1]))
        at = np.array([[0.0, 0.5, 0.0], [1.0, 0.25, -1.0], [1.0, 0.75, -1.0]])
        blended = blended_normals(points, corner_table(polys), normals, np.array([1, 1, 1]), at)
        slope = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        self.assertTrue(np.allclose(blended, [[0, 0, 1], slope, slope]))


if __name__ == "__main__":
    unittest.main()
