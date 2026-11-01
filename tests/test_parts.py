# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.parts import build_box, build_revolve, fit_box, fit_revolve  # noqa: E402


def capped_cylinder(sides=48, rings=12, radius=0.5, height=4.0):
    """A closed triangulated tube: rings of vertices, a centre vertex on each end."""

    pts = [(radius * math.cos(2 * math.pi * j / sides), radius * math.sin(2 * math.pi * j / sides), height * i / (rings - 1)) for i in range(rings) for j in range(sides)]
    pts += [(0.0, 0.0, 0.0), (0.0, 0.0, height)]
    tris = []
    for i in range(rings - 1):
        for j in range(sides):
            a, b, c, d = i * sides + j, i * sides + (j + 1) % sides, (i + 1) * sides + j, (i + 1) * sides + (j + 1) % sides
            tris += [(a, b, d), (a, d, c)]
    for j in range(sides):
        tris.append((rings * sides, (j + 1) % sides, j))
        tris.append((rings * sides + 1, (rings - 1) * sides + j, (rings - 1) * sides + (j + 1) % sides))
    return np.asarray(pts), tris


def closed(polys):
    edges = Counter()
    for p in polys:
        for i in range(len(p)):
            edges[(min(p[i], p[i - 1]), max(p[i], p[i - 1]))] += 1
    return all(c == 2 for c in edges.values())


class RevolveTests(unittest.TestCase):
    def test_tube_becomes_a_few_clean_rings_of_quads(self):
        pts, tris = capped_cylinder()
        fit = fit_revolve(pts, tris)
        self.assertIsNotNone(fit)
        co, polys, _ = build_revolve(fit, 16, 0.01)
        self.assertTrue(all(len(p) == 4 for p in polys))
        self.assertTrue(closed(polys))
        self.assertLess(len(polys), 100)
        self.assertAlmostEqual(float(np.ptp(co[:, 2])), 4.0, places=3)


class BoxTests(unittest.TestCase):
    def test_box_is_found_and_rebuilt_as_six_quads(self):
        pts, polys, _ = build_box({"frame": np.eye(3), "low": np.zeros(3), "high": np.array([4.0, 1.0, 0.5])}, (4, 2, 2))
        fit = fit_box(pts, polys)
        self.assertGreater(fit["share"], 0.99)
        co, quads, _ = build_box(fit, (1, 1, 1))
        self.assertEqual(len(quads), 6)
        self.assertTrue(closed(quads))
        self.assertTrue(np.allclose(np.sort(np.ptp(co, axis=0)), [0.5, 1.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
