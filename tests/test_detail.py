# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles import choose, levels_for  # noqa: E402
from triadq.detail import folded, hard_edges, look, mesh_facts, settle  # noqa: E402


def grid(n):
    points = np.array([(x, y, 0.0) for y in range(n) for x in range(n)], dtype=np.float64)
    quads = np.array([(y * n + x, y * n + x + 1, (y + 1) * n + x + 1, (y + 1) * n + x) for y in range(n - 1) for x in range(n - 1)])
    return points, quads


class DetailTests(unittest.TestCase):
    def test_the_far_side_of_a_thin_part_is_never_taken(self):
        points, normals = np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
        walls = {1.0: ((0.0, 0.0, 0.4), (0.0, 0.0, -1.0), 0, 0.4), -1.0: ((0.0, 0.0, -0.5), (0.0, 0.0, 1.0), 0, 0.5)}  # in front, facing away: a far wall
        targets, found = look(points, normals, 1.0, lambda p, d, far: walls[d[2]], lambda p, far: (None, None, None, None))
        self.assertTrue(found[0])
        self.assertAlmostEqual(targets[0, 2], -0.5)

    def test_nothing_found_takes_the_neighbours_move_and_a_fold_is_undone(self):
        points, quads = grid(5)
        targets = points + (0.0, 0.0, 1.0)
        found = np.ones(len(points), dtype=bool)
        found[12] = False  # the middle
        targets[12] = points[12]
        moved, lost, left = settle(points, targets, found, quads)
        self.assertEqual(lost, 1)
        self.assertEqual(left, 0)
        self.assertAlmostEqual(moved[12, 2], 1.0)
        targets = points.copy()
        targets[12, 0] += 3.0  # jumps over its neighbours
        self.assertTrue(folded(points, targets, quads).any())
        moved, _lost, left = settle(points, targets, np.ones(len(points), dtype=bool), quads)
        self.assertEqual(left, 0)
        self.assertFalse(folded(points, moved, quads).any())

    def test_hard_edges_and_mesh_facts(self):
        # A flat square of two triangles, and a second, far part folded at 90 degrees.
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0], [5, 1, 1], [6, 1, 1]], dtype=np.float64)
        tris = np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (7, 6, 9), (7, 9, 8)])
        facts = mesh_facts(points, tris)
        self.assertEqual(facts["parts"], 2)
        self.assertAlmostEqual(facts["biggest_part"], 4 / 6)
        quad_corners = np.array([4, 5, 6, 7, 7, 6, 9, 8])
        self.assertEqual(hard_edges(points, quad_corners, np.array([4, 4]), 50.0), [(6, 7)])

    def test_auto_picks_by_mesh_and_memory(self):
        base = {"faces": 20000, "parts": 1, "biggest_part": 1.0, "flat": 0.1, "sharp": 0.1, "median_angle": 7.0, "open": 0.0}
        self.assertEqual(choose(base, 16.0)["use"], "ANIMATION")
        self.assertGreater(choose(base, 16.0)["levels"], 0)
        self.assertEqual(choose(dict(base, parts=200, biggest_part=0.05), 16.0)["levels"], 0)
        self.assertEqual(choose(dict(base, flat=0.4), 16.0)["use"], "GAME")
        self.assertEqual(choose(dict(base, flat=0.03), 16.0)["kind"], "scan")
        self.assertEqual(levels_for(12000, 0.5), 2)
        self.assertEqual(levels_for(5000, 16.0, 20000), 2)
        self.assertEqual(levels_for(5000, 16.0, 6000), 1)


if __name__ == "__main__":
    unittest.main()
