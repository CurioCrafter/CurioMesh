# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq.texsets import split_under, texture_sets  # noqa: E402


def strip(n, parts):
    """n unit faces in a row along x, labelled into `parts` equal parts."""
    centres = np.stack([np.arange(n) + 0.5, np.zeros(n), np.zeros(n)], 1)
    return np.repeat(np.arange(parts), n // parts), np.ones(n), centres


class TextureSetTests(unittest.TestCase):
    def test_whole_parts_into_even_sets(self):
        labels, area, centres = strip(40, 4)
        sets = texture_sets(labels, area, centres, 2)
        self.assertEqual(sorted(np.bincount(sets)), [20, 20])
        for part in range(4):
            self.assertEqual(len(np.unique(sets[labels == part])), 1)  # a part is not cut when it fits

    def test_one_big_part_is_cut(self):
        labels, area, centres = strip(30, 1)
        sets = texture_sets(labels, area, centres, 3)
        self.assertEqual(sorted(np.bincount(sets)), [10, 10, 10])

    def test_one_set_is_everything(self):
        labels, area, centres = strip(10, 2)
        self.assertEqual(texture_sets(labels, area, centres, 1).max(), 0)


class SplitTests(unittest.TestCase):
    def test_no_piece_over_the_limit_and_none_mixes_sets(self):
        labels, tris, centres = strip(100, 2)
        tris = tris * 2
        pieces = split_under(labels, tris, centres, 30)
        for p in np.unique(pieces):
            self.assertLessEqual(tris[pieces == p].sum(), 30)
            self.assertEqual(len(np.unique(labels[pieces == p])), 1)

    def test_under_the_limit_stays_whole(self):
        labels, tris, centres = strip(10, 1)
        self.assertEqual(split_under(labels, tris, centres, 20).max(), 0)


if __name__ == "__main__":
    unittest.main()
