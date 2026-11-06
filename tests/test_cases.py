# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.cases import classify, curve, explain, features, first_size, load_table

TABLE = {"cases": {
    "all": {"shares": {"0.02": {"parts": 9, "loss": [0.5, 0.9, 1.0]}, "0.04": {"parts": 9, "loss": [0.2, 0.3, 0.4]}, "0.08": {"parts": 9, "loss": [0.0, 0.1, 0.2]}}, "revolve": {"fits": 0, "wins": 0}},
    "tube": {"shares": {"0.02": {"parts": 9, "loss": [0.4, 0.5, 0.6]}, "0.04": {"parts": 9, "loss": [0.1, 0.2, 0.3]}, "0.08": {"parts": 9, "loss": [0.0, 0.01, 0.02]}}, "revolve": {"fits": 8, "wins": 6}},
}}


def tube(rings=40, spin=24, length=10.0):
    points = [(math.cos(2 * math.pi * k / spin), math.sin(2 * math.pi * k / spin), length * r / (rings - 1)) for r in range(rings) for k in range(spin)]
    polys = [(r * spin + k, r * spin + (k + 1) % spin, (r + 1) * spin + (k + 1) % spin, (r + 1) * spin + k) for r in range(rings - 1) for k in range(spin)]
    return np.asarray(points), polys


def plate(side=30):
    points = [(float(x), float(y), 0.0) for y in range(side) for x in range(side)]
    polys = [(y * side + x, y * side + x + 1, (y + 1) * side + x + 1, (y + 1) * side + x) for y in range(side - 1) for x in range(side - 1)]
    return np.asarray(points), polys


class CaseTests(unittest.TestCase):
    def test_simple_shapes_get_their_case(self):
        points, polys = tube()
        f = features(points, polys, 10.0)
        self.assertEqual(f["boundary_loops"], 2)
        self.assertEqual(f["euler"], 0)
        self.assertGreater(f["revolve_fit"], 0.9)
        self.assertEqual(classify(f), "tube")
        points, polys = plate()
        f = features(points, polys, 30.0)
        self.assertEqual(f["boundary_loops"], 1)
        self.assertEqual(f["euler"], 1)
        self.assertEqual(classify(f), "flat plate")

    def test_first_size_skips_only_what_even_lucky_parts_failed(self):
        self.assertEqual(first_size("tube", [0.02, 0.04, 0.08], 0.2, TABLE), 1)
        self.assertEqual(first_size("tube", [0.02, 0.04, 0.08], 0.5, TABLE), 0)
        self.assertEqual(first_size("tube", [0.02, 0.04, 0.08], -1.0, TABLE), 2)  # nothing fits: the biggest
        self.assertEqual(curve("no such case", [0.02], TABLE)[0], (0.5, 0.9, 1.0))

    def test_words(self):
        self.assertIn("Looks like a tube", explain("tube", 120, 0.04, 2, 3))
        self.assertIn("Looks like an ornament", explain("ornament", 120, 0.04, 0, 3))
        self.assertIn("as it was", explain("box", 120, 0.0, 0, 3, whole=True))
        self.assertIn("an estimated 4.0%", explain("box", 120, 0.04, 0, 3, guessed=True))
        self.assertIn("Made to measure", explain("box", 120, None, 0, 3))

    def test_shipped_table_has_every_part_of_the_answer(self):
        table = load_table()
        self.assertIn("all", table["cases"])
        for row in table["cases"].values():
            self.assertGreaterEqual(len(row["shares"]), 1)
            self.assertIn("revolve", row)


if __name__ == "__main__":
    unittest.main()
