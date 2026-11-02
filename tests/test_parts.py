# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import parts  # noqa: E402
from triadq.parts import build_revolve, fit_revolve  # noqa: E402


def grid(nx, ny, scale=(1.0, 1.0)):
    co = np.array([(i * scale[0], j * scale[1], 0.0) for j in range(ny + 1) for i in range(nx + 1)])
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i) for j in range(ny) for i in range(nx)]
    return co, faces


def cylinder(sides=12, rings=20, radius=0.1, length=4.0):
    co = np.array([(radius * np.cos(2 * np.pi * i / sides), radius * np.sin(2 * np.pi * i / sides), length * r / rings) for r in range(rings + 1) for i in range(sides)])
    faces = [(r * sides + i, r * sides + (i + 1) % sides, (r + 1) * sides + (i + 1) % sides, (r + 1) * sides + i) for r in range(rings) for i in range(sides)]
    return co, faces


def sphere(lat=10, lon=12):
    pts = [(np.sin(np.pi * (a + 1) / (lat + 1)) * np.cos(2 * np.pi * b / lon), np.sin(np.pi * (a + 1) / (lat + 1)) * np.sin(2 * np.pi * b / lon), np.cos(np.pi * (a + 1) / (lat + 1))) for a in range(lat) for b in range(lon)]
    faces = [(a * lon + b, a * lon + (b + 1) % lon, (a + 1) * lon + (b + 1) % lon, (a + 1) * lon + b) for a in range(lat - 1) for b in range(lon)]
    return np.array(pts), faces


class ClassifyTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(parts.classify(*grid(6, 6)), "FLAT")
        self.assertEqual(parts.classify(*cylinder()), "TUBE")
        self.assertEqual(parts.classify(*sphere()), "ROUGH")

    def test_tiny_part_is_rough(self):
        self.assertEqual(parts.classify(*grid(1, 1)), "ROUGH")


class SpliceTests(unittest.TestCase):
    def test_replaces_only_the_named_part(self):
        co_a, f_a = grid(2, 2)
        co_b, f_b = grid(2, 2)
        co = np.vstack((co_a, co_b + [10, 0, 0]))
        faces = f_a + [tuple(v + len(co_a) for v in f) for f in f_b]
        piece = np.array([0] * len(f_a) + [1] * len(f_b))
        mat = np.zeros(len(faces), dtype=np.int32)
        one_quad = (np.array([[10, 0, 0], [11, 0, 0], [11, 1, 0], [10, 1, 0.0]]), [(0, 1, 2, 3)], [0])
        out_co, out_faces, out_mat, out_piece = parts.splice(co, faces, mat, piece, {1: one_quad})
        self.assertEqual(list(out_piece), [0] * 4 + [1])
        self.assertEqual(len(out_faces), 5)
        self.assertEqual(len(out_co), 9 + 4)
        self.assertEqual(max(v for f in out_faces for v in f), len(out_co) - 1)
        again = parts.cut(out_co, out_faces, out_mat, out_piece, 1)
        self.assertEqual(len(again[1]), 1)
        self.assertEqual(len(again[0]), 4)


class WorstTests(unittest.TestCase):
    def test_order_and_unknown_last(self):
        self.assertEqual(parts.worst_first([1.0, None, 9.0, 4.0]), [2, 3, 0, 1])
        self.assertEqual(parts.worst_first([1.0, None, 9.0, 4.0], 2), [2, 3])

    def test_heat_runs_green_to_red(self):
        low, high = parts.heat_colour(0.0, 10.0), parts.heat_colour(10.0, 10.0)
        self.assertGreater(low[1], low[0])
        self.assertGreater(high[0], high[1])



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


if __name__ == "__main__":
    unittest.main()
