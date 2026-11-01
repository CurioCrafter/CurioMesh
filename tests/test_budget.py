# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.budget import Level, allocate, best_levels  # noqa: E402


class AllocateTests(unittest.TestCase):
    def test_a_size_no_better_than_a_smaller_one_is_dropped(self):
        levels = [Level(100, 5.0), Level(50, 9.0), Level(70, 9.5), Level(200, 5.0), Level(300, 0.0)]
        self.assertEqual(best_levels(levels), [1, 0, 4])

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


if __name__ == "__main__":
    unittest.main()
