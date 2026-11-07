# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
import unittest

import numpy as np

from triadq import history, tidytools, trial


def _grid(n: int):
    points = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x) for y in range(n) for x in range(n)]
    return points, faces


def _flat(points):
    out = np.array(points, dtype=np.float64)
    out[:, 2] = 0.0
    return out


class TrialTest(unittest.TestCase):
    def _bumped(self):
        points, faces = _grid(4)
        rng = np.random.default_rng(3)
        inner = [v for v in range(len(points)) if 0 < points[v, 0] < 4 and 0 < points[v, 1] < 4]
        points[inner, :2] += rng.uniform(-0.3, 0.3, (len(inner), 2))
        return trial.make_base(points, faces, project=_flat), inner

    def test_run_yields_often_and_keeps_points_on_the_surface(self):
        base, inner = self._bumped()
        work = tidytools.SmoothCorners().run(base, {"strength": 1.0, "steps": 4})
        yields = 0
        while True:
            try:
                next(work)
                yields += 1
            except StopIteration as done:
                cand = done.value
                break
        self.assertGreater(yields, 1)
        self.assertTrue(np.allclose(cand.co[:, 2], 0.0))
        border = [v for v in range(len(base.co)) if v not in inner]
        self.assertTrue(np.allclose(cand.co[border], base.co[border]))  # the edge of the scope never moves
        self.assertGreater(np.abs(cand.co[inner] - base.co[inner]).max(), 1e-3)

    def test_blend_ends_match_before_and_full(self):
        base, _ = self._bumped()
        full = trial.finish(tidytools.SmoothCorners().run(base, {"strength": 1.0}))
        self.assertTrue(np.allclose(trial.blend(base, full, 0.0).co, base.co))
        self.assertTrue(np.allclose(trial.blend(base, full, 1.0).co, full.co))
        half = trial.blend(base, full, 0.5)
        self.assertFalse(half.exact)
        self.assertIsNotNone(half.moved)

    def test_diff_finds_a_pole_that_goes(self):
        points, faces = _grid(3)
        base = trial.make_base(points, faces[1:])  # a corner quad missing: a border corner with three faces, a pole
        cand = trial.diff(base, trial.Candidate(points, faces=faces))
        self.assertEqual(len(cand.poles_gone), 1)
        self.assertEqual(len(cand.poles_new), 0)

    def test_scope_edge_is_held_and_nothing_free_refuses(self):
        points, faces = _grid(4)
        scope = np.zeros(len(faces), dtype=bool)
        scope[5] = True  # one inner face: all four of its corners touch faces outside
        base = trial.make_base(points, faces, scope=scope)
        self.assertFalse(base.free.any())
        self.assertEqual(tidytools.SmoothCorners().check(base), tidytools.NOTHING_FREE)

    def test_measure_and_readout(self):
        base, _ = self._bumped()
        cand = trial.diff(base, trial.finish(tidytools.SmoothCorners().run(base, {"strength": 1.0})))

        def look(co, faces):
            yield
            return float(np.abs(co[:, 0]).sum() % 3)

        trial.finish(trial.measure(base, cand, ("poles", "move", "look"), look))
        self.assertIn("poles", cand.numbers)
        self.assertIn("look", cand.numbers)
        title, numbers, _third = trial.readout(cand, tidytools.SmoothCorners(), {"strength": 0.6}, "ready")
        self.assertIn("Smooth Corners", title)
        self.assertIn("Strength 0.60", title)
        self.assertEqual(len(numbers), 3)
        self.assertEqual(trial.number_text("poles", (5, 3)), ("2 poles gone", 1))
        self.assertEqual(trial.number_text("look", (1.0, 3.0), cap=2.0)[1], -1)


class StepsTest(unittest.TestCase):
    def test_steps_truncate_merge_and_let_go(self):
        steps = history.Steps()
        a = np.zeros(1000)
        s1 = steps.add(0, "Tidy: Again", "", {"co": a})
        s2 = steps.add(s1, "Quad Size", "", {"co": np.ones(1000)}, kind="size", now=100.0)
        s3 = steps.add(s2, "Quad Size", "", {"co": np.ones(1000)}, kind="size", now=100.5)
        self.assertEqual((s1, s2, s3), (1, 2, 2))  # one drag is one step
        self.assertTrue(steps.renumber("size", "4,880 faces"))
        self.assertEqual(steps.at(2).numbers, "4,880 faces")
        self.assertEqual(steps.at(1).size, 8000)
        s4 = steps.add(1, "Smooth Corners", "", {"co": a})  # undone to step 1, then a new step: step 2 is gone
        self.assertEqual(s4, 2)
        self.assertEqual([e.name for e in steps.entries], ["Tidy: Again", "Smooth Corners"])
        self.assertEqual(steps.at(2).size, 0)  # the same array as step 1: not counted twice
        self.assertEqual(steps.let_go_oldest(1), 1)
        self.assertEqual(steps.let_go, 1)
        self.assertEqual(steps.let_go_oldest(10**9), 0)  # the newest always stays
        self.assertIsNone(steps.at(1))


if __name__ == "__main__":
    unittest.main()
