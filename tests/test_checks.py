# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles import PROFILES, detail_words, preset  # noqa: E402
from triadq import checks  # noqa: E402

CUBE = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
POINTS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], float)


class CheckTests(unittest.TestCase):
    def test_closed_cube_has_no_problems(self):
        result = checks.inspect(POINTS, CUBE, np.zeros(6, int), pairs=np.zeros((0, 2), int))
        self.assertEqual((result.nm_edges, result.flipped, result.dups, result.holes), (0, 0, 0, 0))

    def test_flip_duplicate_and_hole_are_counted(self):
        faces = list(CUBE)
        faces[1] = faces[1][::-1]  # one face the wrong way round
        faces.append(CUBE[0])  # a duplicate
        result = checks.inspect(POINTS, faces, np.zeros(len(faces), int), pairs=np.zeros((0, 2), int))
        self.assertGreater(result.flipped, 0)
        self.assertEqual(result.dups, 1)
        opened = checks.inspect(POINTS, CUBE[:5], np.zeros(5, int), pairs=np.zeros((0, 2), int))
        self.assertEqual(opened.holes, 1)

    def test_neighbours_that_share_a_vertex_are_not_crossings(self):
        pairs = np.array([[0, 2], [0, 1]])
        self.assertEqual(checks.drop_shared(CUBE, pairs).tolist(), [[0, 1]])

    def test_only_new_problems_count_and_name_the_piece(self):
        clean = checks.Inspect(selfx=0, per_piece={0: (0, 0, 0), 1: (0, 0, 0)})
        worse = checks.Inspect(nm_edges=2, selfx=0, per_piece={0: (0, 0, 0), 1: (4, 0, 0)})
        self.assertEqual(checks.new_spots(clean, worse), [1])
        scan = checks.Inspect(nm_edges=2, selfx=0, per_piece={0: (4, 0, 0), 1: (0, 0, 0)})
        self.assertEqual(checks.new_spots(scan, scan), [])
        self.assertEqual(checks.describe(scan, scan, 0, 0), "No new problems")
        self.assertIn("1 spot could not be fixed", checks.describe(clean, worse, 0, 1))

    def test_presets_keep_the_default_and_cover_a_wide_range(self):
        self.assertEqual(preset("GAME"), (6.0, "AUTO", "QUADS"))
        self.assertIsNone(preset("CUSTOM"))
        self.assertEqual(preset("LOW_POLY")[2], "TRIS")
        self.assertEqual(len({v[2:] for v in PROFILES.values()}), len(PROFILES))  # no two uses are the same preset
        self.assertNotEqual(detail_words(0.2), detail_words(40.0))


if __name__ == "__main__":
    unittest.main()
