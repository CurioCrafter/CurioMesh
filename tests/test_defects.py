# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import checks, repair  # noqa: E402

CUBE = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
POINTS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], float)
NATIVE = checks._turbo is not None


def found(result, kind):
    return np.flatnonzero(getattr(result, kind)).tolist()


class DefectTests(unittest.TestCase):
    def test_a_clean_result_has_none(self):
        result = checks.defects(POINTS, CUBE, POINTS, CUBE)
        self.assertEqual(result.counts()["any"], 0)

    def test_bent_flat_and_duplicate_faces(self):
        co = np.vstack([POINTS, [(5, 0, 0), (6, 0, 0), (6, 0, 1), (5, 1, 0), (9, 0, 0), (9, 1, 0), (9, 2, 0)]])
        faces = CUBE + [(8, 9, 10, 11), (12, 13, 14), CUBE[2]]  # bent 90 degrees across a diagonal, all on one line, a copy
        result = checks.defects(co, faces)
        self.assertEqual((found(result, "bent"), found(result, "flat"), found(result, "duplicate")), ([6], [7], [8]))
        self.assertEqual(found(result, "overlap"), [])  # a duplicate is not also an overlap

    def test_a_face_folded_back_over_its_neighbour(self):
        co = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.1, 0, 0.05), (0.1, 1, 0.05)], float)
        faces = [(0, 1, 2, 3), (0, 3, 5, 4)]  # the second turns 170 degrees round the shared edge, back over the first
        self.assertEqual(found(checks.defects(co, faces), "folded_over"), [0, 1])
        level = co.copy()
        level[4:, 0] = -1.0  # opened out flat: fine
        level[4:, 2] = 0.0
        self.assertEqual(found(checks.defects(level, faces), "folded_over"), [])

    @unittest.skipUnless(NATIVE, "needs the native core")
    def test_turned_and_sunk_faces_against_the_source(self):
        faces = list(CUBE)
        faces[1] = faces[1][::-1]  # the top turned inward
        co = np.vstack([POINTS, [(0.4, 0.4, 0.3), (0.6, 0.4, 0.3), (0.6, 0.6, 0.3), (0.4, 0.6, 0.3)]])
        faces.append((8, 9, 10, 11))  # a small face deep inside, facing up: away from the floor it is nearest to
        result = checks.defects(co, faces, POINTS, CUBE)
        self.assertEqual(found(result, "inward"), [1, 6])
        self.assertEqual(found(result, "sunk"), [6])

    @unittest.skipUnless(NATIVE, "needs the native core")
    def test_crossing_and_overlapping_faces(self):
        co = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, 1.5, 0.5), (0.5, 1.5, -0.5), (0.2, 0.2, 0.001), (1.2, 0.2, 0.001), (1.2, 1.2, 0.001), (0.2, 1.2, 0.001)], float)
        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]  # a floor, a wall through it, a copy of the floor lying on it
        result = checks.defects(co, faces)
        self.assertEqual(result.pairs.tolist(), [[0, 1], [1, 2]])
        self.assertEqual(found(result, "overlap"), [0, 2])


class UntangleTests(unittest.TestCase):
    def test_a_corner_pulled_over_its_neighbours_goes_back(self):
        co = np.array([(i, j, 0.0) for j in range(5) for i in range(5)])
        faces = [(j * 5 + i, j * 5 + i + 1, (j + 1) * 5 + i + 1, (j + 1) * 5 + i) for j in range(4) for i in range(4)]
        co[12] = (3.9, 3.9, 0.0)  # the middle corner dragged past a neighbour: faces turn over

        def near(_piece, where):  # the source is the plane z = 0, facing up
            where = np.asarray(where, dtype=float).reshape(-1, 3)
            return where * (1, 1, 0), np.tile((0.0, 0.0, 1.0), (len(where), 1))

        normals = checks._normals(co, checks.corner_table(faces))[0]
        bad = normals[:, 2] < 0
        self.assertTrue(bad.any())
        fixed, moved = repair.untangle(co, faces, np.zeros(len(faces), int), bad, near)
        self.assertEqual(moved, 1)
        self.assertTrue((checks._normals(fixed, checks.corner_table(faces))[0][:, 2] > 0.99).all())
        self.assertTrue((np.delete(fixed, 12, axis=0) == np.delete(co, 12, axis=0)).all())  # nothing else moved


if __name__ == "__main__":
    unittest.main()
