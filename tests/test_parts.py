# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import parts  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
