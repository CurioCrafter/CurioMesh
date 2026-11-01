# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import unittest

import numpy as np

from triadq.uvs import absorb_small_kinds, chart_codes, simplify_borders, uv_overlap


class UvHelpersTest(unittest.TestCase):
    def test_chart_codes_round_trip(self):
        materials, islands = np.array([0, 0, 1, 0]), np.array([5, 7, 5, 5])
        codes, by_material, by_island = chart_codes(materials, islands)
        self.assertEqual(by_material[codes].tolist(), materials.tolist())
        self.assertEqual(by_island[codes].tolist(), islands.tolist())
        self.assertEqual(codes[0], codes[3])
        self.assertEqual(len(set(codes.tolist())), 3)

    def test_straight_border_loses_its_middle_but_stays_a_border(self):
        # Two rows of four quads, split by a straight border along y = 1, with a bend at x = 2 when asked.
        for bend, expected in ((0.0, 2), (0.5, 3)):
            points = np.array([[x, y, bend * (y == 1) * (1 - abs(x - 2) / 2)] for y in range(3) for x in range(5)], dtype=float)
            polys = [(y * 5 + x, y * 5 + x + 1, (y + 1) * 5 + x + 1, (y + 1) * 5 + x) for y in range(2) for x in range(4)]
            kinds = ["a"] * 4 + ["b"] * 4
            out_points, out_polys, out_kinds, free = simplify_borders(points, polys, kinds, 0.01)
            self.assertLess(len(out_points), len(points) + 1)
            on_border = set(range(len(out_points))) - set(free)
            self.assertEqual(len(on_border), expected)
            self.assertTrue(all(abs(out_points[v][1] - 1) < 1e-9 for v in on_border))
            shared = {v for p, k in zip(out_polys, out_kinds) if k == "a" for v in p} & {v for p, k in zip(out_polys, out_kinds) if k == "b" for v in p}
            self.assertEqual(shared, on_border)

    def test_small_kind_joins_its_neighbour_of_the_same_group(self):
        # A row of quads: a large "0", a small "1" and a small "2" of another material
        polys = [(0, 1, 3, 2), (2, 3, 5, 4), (4, 5, 7, 6)]
        areas = np.array([10.0, 1.0, 1.0])
        groups = [0, 0, 1]
        kinds, smeared = absorb_small_kinds(areas, polys, ["0", "1", "2"], 2.0, groups)
        self.assertEqual(kinds, ["0", "0", "2"])
        self.assertEqual(smeared, 1.0)

    def test_overlap(self):
        left = [[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]]
        right = [[1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
        self.assertEqual(uv_overlap(np.array([left, right])), 0.0)
        self.assertGreater(uv_overlap(np.array([left, left])), 0.9)
        self.assertEqual(uv_overlap(np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])), 1.0)


if __name__ == "__main__":
    unittest.main()
