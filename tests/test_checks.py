# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiles import PROFILES, detail_words, lod_name, lod_targets, preset  # noqa: E402
from triadq import checks, repair  # noqa: E402

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

    def test_a_quad_bent_round_an_edge_counts_as_broken(self):
        co = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [6, 0, 1], [5, 1, 0]], float)
        flat, bent = (0, 1, 2, 3), (4, 5, 6, 7)  # the second turns 90 degrees across its diagonal
        result = checks.inspect(co, [flat, bent], np.array([0, 1]), pairs=np.zeros((0, 2), int))
        self.assertEqual((result.folded, result.broken.tolist()), (1, [False, True]))
        level = co.copy()
        level[6] = (6, 1, 0)  # the same quad lying flat, as the source had it
        source = checks.inspect(level, [flat, bent], np.array([0, 1]), pairs=np.zeros((0, 2), int))
        self.assertEqual(checks.new_spots(source, result), [1])
        self.assertIn("1 folded quads", checks.describe(source, result, 0, 0))

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
        self.assertEqual(checks.describe(scan, scan, 0, 0, overlap=22), "New problems: 22 faces lying on each other")
        self.assertIn("1 spot could not be fixed", checks.describe(clean, worse, 0, 1))

    def test_presets_keep_the_default_and_cover_a_wide_range(self):
        self.assertEqual(preset("GAME"), (6.0, "AUTO", "QUADS"))
        self.assertIsNone(preset("CUSTOM"))
        self.assertEqual(preset("LOW_POLY")[2], "TRIS")
        self.assertEqual(len({v[2:] for v in PROFILES.values()}), len(PROFILES))  # no two uses are the same preset
        self.assertNotEqual(detail_words(0.2), detail_words(40.0))


class CrossingTests(unittest.TestCase):
    def test_tiles_cover_every_face_once_and_stay_small(self):
        centers = np.random.default_rng(1).random((1000, 3))
        found = checks.tiles(centers, 64)
        self.assertLessEqual(max(len(t) for t in found), 64)
        self.assertEqual(np.sort(np.concatenate(found)).tolist(), list(range(1000)))

    def test_short_faces_are_padded_with_their_own_corner(self):
        self.assertEqual(checks.corner_table([(0, 1, 2), (3, 4, 5, 6)]).tolist(), [[0, 1, 2, 0], [3, 4, 5, 6]])

    def test_pairs_that_share_a_face_are_one_spot(self):
        self.assertEqual(sorted(len(s) for s in repair.spots_of([(0, 1), (1, 2), (5, 6)])), [1, 2])

    def test_only_what_the_source_did_not_have_is_new(self):
        co = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (9, 0, 0), (10, 0, 0), (10, 1, 0), (0, 5, 0), (1, 5, 0), (1, 6, 0)], float)
        faces = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        known = repair.Known(co, faces, np.array([0, 0, 1]), np.array([[0, 1], [0, 2]]))  # piece 0 crosses itself, and piece 1
        work = repair._Work(co, faces, ["0"] * 3, np.array([0, 0, 1]), known, None, False)
        self.assertFalse(work.is_new(0, 2))  # those two pieces already cross
        self.assertFalse(work.is_new(0, 1))  # right where the source crosses itself
        work.piece[:] = [0, 2, 1]
        self.assertTrue(work.is_new(1, 2))  # pieces that did not cross


class LodTests(unittest.TestCase):
    def test_lod_names_and_sizes(self):
        self.assertEqual(lod_name("Crate", 2), "Crate_LOD2")
        self.assertEqual(lod_targets((50.0, 25.0, 0.001), 10000), [5000, 2500, 12])


if __name__ == "__main__":
    unittest.main()
