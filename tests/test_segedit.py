# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Live segment editing: the size controls, painted size, locks, split in two, shrink and the debounce."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parallel  # noqa: E402
from triadq import segment  # noqa: E402
from test_segment import dumbbell, grid  # noqa: E402


def count(labels) -> int:
    return len(np.unique(labels))


class SizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.co, cls.faces = dumbbell(sides=32, rings=60)
        cls.tree = segment.build(cls.co, cls.faces, cells=24)

    def test_plain_cut_is_unchanged(self):
        ones = np.ones(len(self.tree.leaf_area))
        np.testing.assert_array_equal(np.unique(segment.cut(self.tree, 0.7), return_inverse=True)[1], np.unique(segment.cut(self.tree, 0.7, scale=ones), return_inverse=True)[1])

    def test_red_paint_makes_more_segments(self):
        plain = count(segment.cut(self.tree, 0.5))
        red = segment.patch_scale(self.tree, np.ones(len(self.faces)))
        blue = segment.patch_scale(self.tree, np.zeros(len(self.faces)))
        self.assertGreater(count(segment.cut(self.tree, 0.5, scale=red)), plain)
        self.assertLessEqual(count(segment.cut(self.tree, 0.5, scale=blue)), plain)

    def test_paint_only_changes_where_painted(self):
        _n, _a, centre = segment.face_frames(self.co, self.faces)
        weight = np.full(len(self.faces), 0.5)
        weight[centre[:, 2] > 1.4] = 1.0  # red on one ball only
        parent = segment.cut(self.tree, 0.4, scale=segment.patch_scale(self.tree, weight))
        labels = parent[self.tree.leaf]
        top, bottom = labels[centre[:, 2] > 1.5], labels[centre[:, 2] < 0.5]
        self.assertGreater(count(top), count(bottom))

    def test_largest_caps_the_segment(self):
        whole = count(segment.cut(self.tree, 0.0))
        capped = segment.cut(self.tree, 0.0, largest=0.3)
        self.assertGreater(count(capped), whole)
        area = np.bincount(capped, self.tree.leaf_area)
        self.assertLessEqual(area.max(), 0.3 * self.tree.leaf_area.sum() + 1e-9)

    def test_smallest_joins_small_ones(self):
        self.assertLessEqual(count(segment.cut(self.tree, 1.0, smallest=0.2)), count(segment.cut(self.tree, 1.0, smallest=0.0)))

    def test_crease_strength_merges_again(self):
        stronger = segment.remerge(self.tree, segment.crease_factor(1.0))
        self.assertEqual(len(stronger.merges), len(self.tree.merges))
        self.assertTrue(np.all(np.diff(stronger.costs) >= 0))
        self.assertIs(segment.remerge(self.tree, 1.0), self.tree)
        self.assertAlmostEqual(segment.crease_factor(0.5), 1.0)

    def test_face_paint_is_corner_mean(self):
        co, faces = grid(2, 1)
        weight = np.zeros(len(co))
        weight[[0, 3]] = 1.0  # the left edge of the first quad
        np.testing.assert_allclose(segment.face_paint(faces, weight), [0.5, 0.0])


class EditTests(unittest.TestCase):
    def test_locked_segment_survives_a_new_cut(self):
        co, faces = grid(10, 10)
        fa, fb, _u, _v = segment.face_links(faces)
        old = np.zeros(len(faces), dtype=np.int64)
        old[[0, 1, 10, 11]] = 3  # a locked corner
        new = np.zeros(len(faces), dtype=np.int64)
        new[50:] = 1
        out = segment.keep_locked(new, old, {3}, fa, fb)
        corner = out[[0, 1, 10, 11]]
        self.assertEqual(len(set(corner.tolist())), 1)
        self.assertEqual(int((out == corner[0]).sum()), 4)
        self.assertEqual(count(out), 3)

    def test_lock_cutting_a_segment_makes_two(self):
        co, faces = grid(10, 1)
        fa, fb, _u, _v = segment.face_links(faces)
        old = np.zeros(10, dtype=np.int64)
        old[4:6] = 1
        out = segment.keep_locked(np.zeros(10, dtype=np.int64), old, {1}, fa, fb)
        self.assertEqual(count(out), 3)

    def test_split_in_two_halves_the_dumbbell(self):
        co, faces = dumbbell()
        tree = segment.build(co, faces, cells=20)
        labels = np.zeros(len(faces), dtype=np.int64)
        side = segment.split_in_two(tree, labels, 0)
        self.assertIsNotNone(side)
        self.assertTrue(side.any() and not side.all())
        _n, _a, centre = segment.face_frames(co, faces)
        self.assertNotEqual(side[centre[:, 2] < 0.3].mean() > 0.5, side[centre[:, 2] > 1.7].mean() > 0.5)

    def test_shrink_gives_faces_to_the_neighbour(self):
        co, faces = grid(10, 1)
        fa, fb, _u, _v = segment.face_links(faces)
        labels = np.array([0] * 5 + [1] * 5)
        mask = np.zeros(10, dtype=bool)
        mask[2:5] = True
        out = segment.shrink_into(labels, mask, 0, fa, fb)
        np.testing.assert_array_equal(out, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_shrink_with_no_neighbour_keeps_faces(self):
        co, faces = grid(4, 1)
        fa, fb, _u, _v = segment.face_links(faces)
        out = segment.shrink_into(np.zeros(4, dtype=np.int64), np.ones(4, dtype=bool), 0, fa, fb)
        np.testing.assert_array_equal(out, 0)


class DebounceTests(unittest.TestCase):
    def test_waits_for_quiet_and_newest_wins(self):
        d = parallel.Debounce(0.2)
        self.assertFalse(d.due(0.0))
        d.poke(1.0)
        d.poke(1.1)
        self.assertFalse(d.due(1.25))
        self.assertTrue(d.due(1.35))
        ticket = d.take()
        self.assertFalse(d.due(5.0))
        self.assertFalse(d.stale(ticket))
        d.poke(1.4)
        self.assertTrue(d.stale(ticket))


if __name__ == "__main__":
    unittest.main()
