# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_flowstroke import grid  # noqa: E402
from test_organize import ball, onto_ball  # noqa: E402
from triadq import loopscore as L  # noqa: E402
from triadq import organize as O  # noqa: E402
from triadq.types import drain  # noqa: E402


class LoopScoreTest(unittest.TestCase):
    def test_a_clean_grid_scores_high_and_a_shaken_one_low(self):
        co, faces = grid(10)
        clean = L.loop_quality(co, faces, with_symmetry=True)
        n = clean["numbers"]
        self.assertEqual(n["loops"], 22)  # 11 rows, 11 columns, each one loop
        self.assertAlmostEqual(n["turn_mean"], 0.0, places=6)
        self.assertAlmostEqual(n["spacing_cv"], 0.0, places=6)
        self.assertEqual(n["zigzag"], 0.0)
        self.assertGreater(n["symmetry"], 0.99)
        shaken = L.loop_quality(*grid(10, wobble=0.2))
        self.assertGreater(clean["quality"], shaken["quality"] + 20)
        self.assertGreater(shaken["numbers"]["zigzag"], 0.2)

    def test_rings_spirals_and_poles(self):
        co, faces = ball(6)
        loops, closed, label = L.walk(L.Topology(co, faces))
        self.assertEqual(int((label < 0).sum()), 0)  # every edge in one loop
        self.assertGreater(sum(closed), 0)
        n = L.measure(co, faces)
        self.assertEqual(n["poles"], 8)
        self.assertEqual(n["spirals"], 0)
        self.assertGreaterEqual(n["pole_gap"], 3)

    def test_meter(self):
        a = {"quality": 41.2, "parts": {"straight": 40, "spacing": 52}}
        b = {"quality": 63, "parts": {"straight": 70, "spacing": 80}}
        self.assertEqual(L.meter(a, b), "Loop Quality 41 -> 63 (straight 40 -> 70, even spacing 52 -> 80)")

    def test_straighten_raises_loop_quality_and_stays_on_the_surface(self):
        co, faces = ball(10, jitter=0.3)
        before = L.loop_quality(co, faces)["quality"]
        last = drain(O.settle_steps(co, faces, onto_ball, method="straighten"))[-1][1]
        self.assertGreater(L.loop_quality(last, faces)["quality"], before + 20)
        self.assertLess(float(np.abs(np.linalg.norm(last, axis=1) - 1.0).max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
