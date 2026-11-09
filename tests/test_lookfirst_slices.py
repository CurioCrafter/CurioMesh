# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Look First's own measuring in short slices, and the splitting-tool flag."""
import unittest

import numpy as np

from triadq import autofix, loopscore, patterns, trial
from triadq.trial import Candidate, Spot, Trial

N = 12


def _sphere(n=8):
    """A cube sphere of quads, outward."""
    idx, pts, faces = {}, [], []
    t = np.linspace(-1, 1, n + 1)

    def vid(p):
        k = tuple(np.round(p, 9))
        if k not in idx:
            idx[k] = len(pts)
            pts.append(p)
        return idx[k]

    for ax in range(3):
        for sgn in (-1, 1):
            for i in range(n):
                for j in range(n):
                    def P(a, b):
                        p = np.zeros(3)
                        p[ax], p[(ax + 1) % 3], p[(ax + 2) % 3] = sgn, t[a], t[b]
                        return p / np.linalg.norm(p)

                    q = [vid(P(i, j)), vid(P(i + 1, j)), vid(P(i + 1, j + 1)), vid(P(i, j + 1))]
                    faces.append(tuple(q[::-1] if sgn < 0 else q))
    return np.array(pts), faces


def _grid(n=N):
    pts = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return pts, faces


def _count(work):
    n = 0
    while True:
        try:
            next(work)
            n += 1
        except StopIteration as done:
            return n, done.value


class SlicesTest(unittest.TestCase):
    def test_steps_give_the_plain_answers_and_really_yield(self):
        co, faces = _sphere(9)
        base = trial.make_base(co, faces)
        n, tidy = _count(trial._tidiness_steps(co, faces))
        self.assertGreater(n, 3)
        self.assertEqual(tidy, trial._tidiness(co, faces))
        n, loops = _count(trial._loops_steps(co, faces, None))
        self.assertGreater(n, 3)
        self.assertEqual(loops, trial._loops(co, faces, None))
        self.assertEqual(loops, float(loopscore.loop_quality(co, faces, None)["quality"]))
        self.assertEqual(patterns.measure(co, faces)["quads"], len(faces))
        trial.BAD_CHUNK, keep = 100, trial.BAD_CHUNK
        try:
            co2 = co.copy()
            co2[5] += 0.9  # some faces turned over or bent
            n, bad = _count(trial.bad_faces_steps(base, co2))
        finally:
            trial.BAD_CHUNK = keep
        self.assertGreater(n, 3)
        self.assertTrue(bad.any())
        np.testing.assert_array_equal(bad, trial.bad_faces(base, co2))

    def test_measure_is_the_same_with_steps(self):
        co, faces = _sphere(6)
        base = trial.make_base(co, faces)
        cand = Candidate(co * 1.01)
        trial.finish(trial.measure(base, cand, ("poles", "tidy", "loops", "bad")))
        self.assertEqual(cand.numbers["tidy"], (trial._tidiness(co, faces), trial._tidiness(co * 1.01, faces)))
        self.assertEqual(cand.numbers["bad"], (0, 0))


class Split(Trial):
    """Toy: splits the quad at spot_face into two triangles; `fold` also drags a corner of a neighbour so it folds."""

    name = "Split"
    handles = ()
    defaults = {}
    blendable = False
    stages = ("faces", "poles")
    kinds = (("cross", "face", "faces"),)
    own = "whole"
    changes_faces = True
    splits_faces = True

    def __init__(self, face=20, fold=False):
        self.face, self.fold = face, fold

    def spots(self, base):
        yield 1.0
        return [Spot("cross", np.array([self.face]), base.co[list(base.faces[self.face])].mean(axis=0), 1.0)]

    def own_number(self, base, co, faces, spot):
        return float(sum(len(faces[f]) == 4 for f in spot.faces))  # quads across the "seam" left

    def run(self, base, settings, rough=False):
        faces, parent = [], []
        for f, face in enumerate(base.faces):
            if f == self.face:
                a, b, c, d = face
                faces += [(a, b, c), (a, c, d)]
                parent += [f, f]
            else:
                faces.append(face)
                parent.append(f)
        co = base.co.copy()
        if self.fold:  # drags a corner of the split quad itself past the middle: the pieces turn over
            co[base.faces[self.face][2]] = co[base.faces[self.face][0]] - np.array([1.0, 1.0, 0.0])
        yield 0.5
        return Candidate(co, faces=faces, parent=np.array(parent))

    least = 0.5


class SplitFlagTest(unittest.TestCase):
    def _run(self, trial_, flag=True):
        co, faces = _grid()
        base = trial.make_base(co, faces)
        trial_.splits_faces = flag
        return trial.finish(autofix.propose(trial_, base, {}, spots=trial.finish(trial_.spots(base))))

    def test_split_triangles_are_allowed_with_the_flag(self):
        p = self._run(Split())
        self.assertEqual(p.reason, "")
        self.assertEqual(p.after.bad, 0)

    def test_without_the_flag_the_same_split_is_counted_and_a_pole_appears(self):
        p = self._run(Split(), flag=False)
        self.assertEqual(p.reason, "worse")  # the pole and Loop Quality drop of the split edges: what Seam Fit ran into

    def test_a_real_fold_is_still_refused_with_the_flag(self):
        p = self._run(Split(fold=True))
        self.assertEqual([why for _sp, why in p.left], ["bad faces"])  # the spot is put back, and says why
        self.assertEqual(p.cand.faces, None)

    def test_a_zero_area_piece_of_a_split_is_not_turned_but_a_real_turn_is(self):
        co = np.array([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0), (1, 1e-14, 0)], dtype=np.float64)
        base = trial.make_base(co[:4], [(0, 1, 2, 3)])
        flat = [(0, 1, 2), (2, 3, 0)]
        sliver = [(0, 2, 3), (1, 0, 4)]
        parent = np.array([0, 0])
        self.assertTrue(trial.bad_faces(base, co, sliver, parent).any())
        self.assertFalse(trial.bad_faces(base, co, sliver, parent, splits=True).any())
        turned = [(0, 2, 3), (0, 3, 2)]
        self.assertTrue(trial.bad_faces(base, co, turned, parent, splits=True).any())
        self.assertFalse(trial.bad_faces(base, co, flat, parent, splits=True).any())

    def test_bad_by_parent_counts_a_split_once(self):
        bad = np.array([True, True, False, True])
        self.assertEqual(trial.bad_by_parent(bad, np.array([0, 0, 1, 2])), 2)


if __name__ == "__main__":
    unittest.main()
