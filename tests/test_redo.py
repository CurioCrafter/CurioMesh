# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors
import unittest

import numpy as np

from triadq import redo


def _grid(n: int):
    points = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return points, faces


class RedoTest(unittest.TestCase):
    def test_region_put_back_inside_its_outline(self):
        points, faces = _grid(6)
        middle = 3 * 6 + 3
        region = redo.carve(faces, [middle], rings=1, most=100)
        self.assertEqual(len(region), 9)  # the face and the eight round it
        free = redo.free_corners(faces, region, len(points))
        self.assertEqual(int(free.sum()), 4)  # the inner corners of the 3x3 block; its outline is shared with the rest
        bumped = points.copy()
        bumped[free, 2] = [0.8, -0.8, 0.8, -0.8]  # folds the quads over
        targets = points.copy()
        targets[free, 2] += [0.0, 0.0, 0.0, 5.0]  # the last one's nearest source point is too far to trust
        out = redo.toward(bumped[free], targets[free], np.array([True, True, True, False]), 0.5)
        self.assertTrue(np.allclose(out[:, 2], [0.4, -0.4, 0.4, -0.8]))

    def test_carving_skips_taken_faces_and_stops_at_its_size(self):
        _points, faces = _grid(6)
        taken = np.zeros(len(faces), bool)
        taken[:6] = True
        self.assertLessEqual(len(redo.carve(faces, [8], rings=3, most=20)), 20)
        self.assertFalse(taken[redo.carve(faces, [8], rings=1, most=100, taken=taken)].any())

    def test_redo_kept_only_when_it_clears_faces_without_bending_quads(self):
        self.assertTrue(redo.better((10, 0), (2, 0)))
        self.assertFalse(redo.better((10, 0), (10, 0)))
        self.assertFalse(redo.better((10, 0), (2, 1)))


if __name__ == "__main__":
    unittest.main()
