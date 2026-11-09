# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import importlib.util
import sys
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location("cm_verdict", Path(__file__).resolve().parents[1] / "verdict.py")
V = importlib.util.module_from_spec(_spec)
sys.modules["cm_verdict"] = V
_spec.loader.exec_module(V)


class VerdictTest(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(V.verdict("Animation", 6000, 1.5, 1.5, 0, 0), ("GOOD", "Good for animation"))
        self.assertEqual(V.verdict("Game Asset", 6000, 7.0, 6.0, 0, 0)[0], "GOOD")  # within 1.2 x asked
        self.assertEqual(V.verdict("Game Asset", 6000, 8.0, 6.0, 0, 0)[0], "CHECK")
        self.assertEqual(V.verdict("Game Asset", 6000, 13.0, 6.0, 0, 0)[0], "BAD")
        self.assertEqual(V.verdict("Game Asset", 1000, 1.0, 6.0, 3, 1), ("CHECK", "Check 1 spot"))
        self.assertEqual(V.verdict("Game Asset", 1000, 1.0, 6.0, 6, 2), ("BAD", "2 spots with bad faces left"))

    def test_heavier_than_source_says_so(self):
        self.assertEqual(V.verdict("Game Asset", 3000, 1.0, 6.0, 0, 0, 2500), ("CHECK", V.HEAVIER))
        self.assertEqual(V.verdict("Game Asset", 2000, 1.0, 6.0, 0, 0, 2500)[0], "GOOD")

    def test_face_count_aim_ignores_change(self):
        self.assertEqual(V.verdict("Low Poly", 500, 40.0, None, 0, 0)[0], "GOOD")

    def test_serious_and_bands(self):
        self.assertEqual(V.serious_count({"inward": 2, "folded_over": 1, "sliver": 50}), 3)
        self.assertEqual(V.serious_count(None), 0)
        self.assertEqual(V.tidiness_words(10), "rough")
        self.assertEqual(V.tidiness_words(56), "like a quick hand retopo")
        self.assertEqual(V.tidiness_words(95), "hand-made rows")


if __name__ == "__main__":
    unittest.main()
