# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.bakescore import picture_counts  # noqa: E402


def square(size=64, inset=8, depth=1.0):
    hit = np.zeros((size, size), bool)
    hit[inset:-inset, inset:-inset] = True
    normal = np.zeros((size, size, 3))
    normal[..., 2] = 1.0
    return hit, normal, np.full((size, size), depth), np.zeros((size, size), np.int64)


class BakeScoreTests(unittest.TestCase):
    def test_a_shift_of_fine_bumps_does_not_count(self):
        rng = np.random.default_rng(1)
        hit, normal, depth, face = square()
        bumpy = normal.copy()
        bumpy[..., :2] = rng.normal(0.0, 0.5, (64, 64, 2))
        shifted = np.roll(bumpy, 1, axis=0)
        wrong, _shown = picture_counts([(hit, bumpy, depth, face)], [(hit, shifted, depth, face)], 0.1)
        self.assertEqual(wrong, 0)

    def test_a_large_turn_counts(self):
        hit, normal, depth, face = square()
        turned = np.zeros_like(normal)
        turned[..., 0] = 1.0
        wrong, shown = picture_counts([(hit, normal, depth, face)], [(hit, turned, depth, face)], 0.1)
        self.assertEqual(wrong, shown)

    def test_a_surface_far_away_counts_and_a_near_one_does_not(self):
        hit, normal, depth, face = square()
        near = picture_counts([(hit, normal, depth, face)], [(hit, normal, depth + 0.2, face)], 0.1)
        far = picture_counts([(hit, normal, depth, face)], [(hit, normal, depth + 0.5, face)], 0.1)
        self.assertEqual(near[0], 0)
        self.assertEqual(far[0], far[1])

    def test_a_lost_thin_part_outweighs_its_few_pixels(self):
        hit, normal, depth, face = square()
        hit[2:8, 30:32] = True  # a spike on top of the square
        lost = hit.copy()
        lost[2:8, 30:32] = False
        wrong, shown = picture_counts([(hit, normal, depth, face)], [(lost, normal, depth, face)], 0.1)
        self.assertGreater(wrong, 2 * int((hit ^ lost).sum()))
        self.assertLess(wrong, shown)


if __name__ == "__main__":
    unittest.main()
