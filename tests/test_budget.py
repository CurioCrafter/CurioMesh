# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.budget import Level, allocate, allocate_to_faces  # noqa: E402


class AllocateTests(unittest.TestCase):
    def test_spends_faces_where_they_remove_the_most_loss(self):
        cheap_fix = [Level(10, 50.0), Level(20, 0.0)]
        costly_fix = [Level(10, 50.0), Level(1000, 0.0)]
        self.assertEqual(allocate([cheap_fix, costly_fix], 50.0, 1.0, [100.0, 100.0]), [1, 0])

    def test_no_piece_goes_below_its_floor_even_when_the_total_is_fine(self):
        bolt = [Level(12, 9.0), Level(24, 1.0)]
        tank = [Level(100, 0.0)]
        self.assertEqual(allocate([bolt, tank], 1000.0, 0.2, [10.0, 1000.0]), [1, 0])

    def test_stops_as_soon_as_the_total_is_met(self):
        piece = [Level(10, 30.0), Level(20, 10.0), Level(40, 0.0)]
        self.assertEqual(allocate([piece], 15.0, 1.0, [100.0]), [1])

    def test_keeps_the_most_faces_when_nothing_reaches_the_total(self):
        piece = [Level(10, 30.0), Level(20, 10.0)]
        self.assertEqual(allocate([piece], 0.0, 1.0, [100.0]), [1])


class AllocateToFacesTests(unittest.TestCase):
    def pieces(self):
        # twelve pieces, each with sizes 100 / 200 / 400 faces and less loss the more faces
        return [[Level(100, 40.0 + i), Level(200, 15.0 + i), Level(400, 0.0)] for i in range(12)], [100.0] * 12

    def total(self, pieces, choice):
        return sum(levels[k].faces for levels, k in zip(pieces, choice))

    def test_lands_within_ten_percent_of_the_faces_asked_for(self):
        pieces, seen = self.pieces()
        for faces in (1500, 2400, 3300, 4500):
            choice = allocate_to_faces(pieces, faces, 1.0, seen)
            self.assertLessEqual(abs(self.total(pieces, choice) - faces), 0.1 * faces)

    def test_asking_for_less_never_gives_more(self):
        pieces, seen = self.pieces()
        totals = [self.total(pieces, allocate_to_faces(pieces, f, 1.0, seen)) for f in (1300, 2000, 3000, 4500)]
        self.assertEqual(totals, sorted(totals))

    def test_gives_the_nearest_when_the_target_is_out_of_reach(self):
        pieces, seen = self.pieces()
        self.assertEqual(self.total(pieces, allocate_to_faces(pieces, 10, 1.0, seen)), 1200)
        self.assertEqual(self.total(pieces, allocate_to_faces(pieces, 99999, 1.0, seen)), 4800)


if __name__ == "__main__":
    unittest.main()
