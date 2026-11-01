# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import symmetry  # noqa: E402
from triadq import flatquad  # noqa: E402
from triadq.types import Cancelled, MeshData  # noqa: E402


def _sheet(n=12):
    vertices = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)])
    faces = []
    for y in range(n):
        for x in range(n):
            a, b, c, d = y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x
            faces += [(a, b, c), (a, c, d)]
    return MeshData(vertices, faces)


class CancelTests(unittest.TestCase):
    def test_pairing_stops_when_asked(self):
        with self.assertRaises(Cancelled):
            flatquad.pair_and_close(_sheet(), cancelled=lambda: True)
        out, _report = flatquad.pair_and_close(_sheet())  # the stop flag does not leak into the next call on this thread
        self.assertTrue(out.faces)


def _cube(n=5):
    """A closed triangulated cube from -1 to 1 whose grid does not have a vertex plane at 0."""
    index, co, faces = {}, [], []
    for face_axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = (face_axis + 1) % 3, (face_axis + 2) % 3
            ids = {}
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0.0, 0.0, 0.0]
                    p[face_axis], p[u], p[v] = sign, -1.0 + 2.0 * i / n, -1.0 + 2.0 * j / n
                    key = tuple(round(x, 9) for x in p)
                    if key not in index:
                        index[key] = len(co)
                        co.append(p)
                    ids[i, j] = index[key]
            for i in range(n):
                for j in range(n):
                    a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
                    faces += [(a, b, c), (a, c, d)] if sign > 0 else [(a, c, b), (a, d, c)]
    return np.asarray(co), faces


class SymmetryTests(unittest.TestCase):
    def test_cut_and_mirror_gives_an_exact_closed_mirror(self):
        co, faces = _cube()
        half, half_faces, source = symmetry.cut_half(co, faces, 0, 0.0)
        self.assertGreaterEqual(half[:, 0].min(), 0.0)
        self.assertEqual(len(source), len(half_faces))
        out, out_faces, _mat, snapped = symmetry.mirror_result(half, half_faces, None, 0, 0.0, 3.5)
        self.assertGreater(snapped, 0)
        self.assertFalse(symmetry.border_vertices(out_faces, len(out)).any())  # no gap along the seam
        mirrored = out * np.array([-1.0, 1.0, 1.0])
        self.assertEqual({tuple(p) for p in out.tolist()}, {tuple(p) for p in mirrored.tolist()})  # exactly mirrored


if __name__ == "__main__":
    unittest.main()
