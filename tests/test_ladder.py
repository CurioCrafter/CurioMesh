# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import ladder as L  # noqa: E402


class LadderTest(unittest.TestCase):
    def test_slider_maps_round_trip_and_grows(self):
        self.assertAlmostEqual(L.size_detail(6.0, 0.5), 6.0)
        self.assertAlmostEqual(L.size_detail(6.0, 0.0), 1.5)
        self.assertAlmostEqual(L.size_detail(6.0, 1.0), 24.0)
        values = [L.size_detail(6.0, s / 10) for s in range(11)]
        self.assertEqual(values, sorted(values))
        for s in (0.0, 0.2, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(L.size_of(6.0, L.size_detail(6.0, s)), s)

    def test_plan_covers_every_rung_once_coarse_first(self):
        order = L.plan(6.0, "DETAIL")
        self.assertEqual(len(order), len(set(order)))
        self.assertNotIn(("DETAIL", 6.0), order)
        self.assertEqual(order[0], ("DETAIL", 9.54))
        mine = [d for s, d in order if s == "DETAIL"]
        self.assertEqual(mine[:3], [9.54, 15.12, 24.0])  # quick ones first
        self.assertEqual(mine[3], 1.5)  # then the finest, which the ones between reuse
        self.assertEqual({s for s, _d in order}, {"DETAIL", "BALANCED", "ROWS"})
        self.assertEqual(order[-2:], [("ROWS", 6.0), ("ROWS", 15.12)])
        self.assertNotIn("ROWS", {s for s, _d in L.plan(6.0, "DETAIL", rows=False)})
        self.assertTrue(all(L.LOWEST <= d <= L.HIGHEST for _s, d in L.plan(60.0, "BALANCED")))

    def test_rung_choice(self):
        ready = [1.5, 6.0, 24.0]
        self.assertEqual(L.nearest(ready, 3.0), 1.5)  # ties go to the finer one
        self.assertEqual(L.nearest(ready, 3.1), 6.0)
        self.assertIsNone(L.nearest([], 3.0))
        self.assertEqual(L.step(ready, 6.0, True), 24.0)
        self.assertEqual(L.step(ready, 6.0, False), 1.5)
        self.assertIsNone(L.step(ready, 24.0, True))

    def test_useful_drops_beaten_rungs(self):
        rungs = [(6.0, 14286, 6.2), (9.54, 14488, 8.5), (15.1, 9000, 14.0), (3.8, 20000, 4.0)]
        self.assertEqual(L.useful(rungs), [6.0, 15.1, 3.8])
        self.assertEqual(L.useful([(1, 10, 1.0), (2, 10, 1.0)]), [1, 2])  # ties both stay

    def test_bar_between_styles(self):
        self.assertEqual(L.between(0.0), ("DETAIL", "BALANCED", 0.0))
        self.assertEqual(L.between(0.25), ("DETAIL", "BALANCED", 0.5))
        self.assertEqual(L.between(0.75), ("BALANCED", "ROWS", 0.5))
        self.assertEqual(L.between(1.0), ("BALANCED", "ROWS", 1.0))
        self.assertEqual(L.order_of("ROWS"), 1.0)

    def test_pieces_switch_one_at_a_time_smallest_first(self):
        sizes = [100, 10, 40]
        seen = [L.switched(sizes, t / 20) for t in range(21)]
        for a, b in zip(seen, seen[1:]):
            self.assertTrue(set(a) <= set(b))
        self.assertEqual(seen[0], [])
        self.assertEqual(seen[-1], [0, 1, 2])
        self.assertEqual(L.switched(sizes, 0.1), [1])
        self.assertEqual(L.switched(sizes, 0.2), [1, 2])
        self.assertEqual(L.switched([5], 0.49), [])
        self.assertEqual(L.switched([5], 0.5), [0])


if __name__ == "__main__":
    unittest.main()
