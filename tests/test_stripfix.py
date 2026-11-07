# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Face Loops' fixes as Tidy Tools with their Auto Fix spots (bench/tools_design/AUTO_FIX.md 6.2, builder D)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Redraw in Rows uses the add-on's flowstroke

from triadq import autofix, trial  # noqa: E402
from triadq import stripfix as SF  # noqa: E402
from triadq.trial import Spot  # noqa: E402


def grid(nx, ny):
    at = {(i, j): j * (nx + 1) + i for j in range(ny + 1) for i in range(nx + 1)}
    co = np.array([(float(i), float(j), 0.0) for j in range(ny + 1) for i in range(nx + 1)])
    faces = [(at[i, j], at[i + 1, j], at[i + 1, j + 1], at[i, j + 1]) for j in range(ny) for i in range(nx)]
    return co, faces, at


def flat(points):
    out = np.array(points, dtype=np.float64)
    out[:, 2] = 0.0
    return out


def spin(faces, at, i, j):
    """The edge between quads (i, j) and (i + 1, j) spun inside them: two 3s and two 5s round it."""
    faces = list(faces)
    f = faces.index((at[i, j], at[i + 1, j], at[i + 1, j + 1], at[i, j + 1]))
    g = faces.index((at[i + 1, j], at[i + 2, j], at[i + 2, j + 1], at[i + 1, j + 1]))
    faces[f] = (at[i, j], at[i + 1, j], at[i + 2, j], at[i + 2, j + 1])
    faces[g] = (at[i, j], at[i + 2, j + 1], at[i + 1, j + 1], at[i, j + 1])
    return faces


def wiggly(cols, n=16):
    """An n x n grid whose strips between each pair of columns in `cols` snake left and right."""
    co, faces, at = grid(n, n)
    co = co.copy()
    for j in range(1, n):
        for i in cols:
            co[at[i, j], 0] += 0.3 * np.sin(2 * np.pi * j / 4)
    return co, faces, at


def held(base, spot):
    """`base` with every corner of `spot` held: a spot nothing can make better."""
    stay = np.zeros(len(base.co), dtype=bool)
    stay[np.unique([v for f in spot.faces.tolist() for v in base.faces[f]])] = True
    return trial.make_base(base.co, base.faces, stay=stay, project=base.project, mirror=base.mirror)


def run(work):
    return trial.finish(work)


class StraightenTest(unittest.TestCase):
    def test_finds_the_wiggly_strips_fixes_one_and_puts_back_the_held_one(self):
        co, faces, _at = wiggly((2, 3, 12, 13))
        base = trial.make_base(co, faces, project=flat)
        tool = SF.Straighten()
        spots = run(tool.spots(base))
        self.assertEqual(len(spots), 2)
        self.assertTrue(all(sp.kind == "wiggly" and sp.bad > 1.0 for sp in spots))
        free = run(autofix.propose(tool, base, {}, spots))
        self.assertEqual(free.reason, "")
        for sp in spots:
            self.assertLess(tool.own_number(base, free.cand.co, faces, sp), tool.own_number(base, co, faces, sp))
        right = max(spots, key=lambda sp: sp.at[0])
        p = run(autofix.propose(tool, held(base, right), {}, spots))
        self.assertEqual([sp.at[0] for sp in p.fixed], [min(sp.at[0] for sp in spots)])
        self.assertEqual([sp.at[0] for sp, _why in p.left], [right.at[0]])
        self.assertIn(p.left[0][1], ("no better", "held"))  # held: nothing in it may move
        self.assertTrue(np.allclose(p.cand.co[[v for f in right.faces.tolist() for v in faces[f]]], co[[v for f in right.faces.tolist() for v in faces[f]]]))

    def test_the_manual_run_is_the_same_maths(self):
        co, faces, _at = wiggly((5, 6))
        base = trial.make_base(co, faces, project=flat)
        tool = SF.Straighten()
        spot = run(tool.spots(base))[0]
        element = spot.pick["strips"][0]
        self.assertEqual(tool.waiting(tool.defaults), "strip")
        cand = run(tool.run(base, dict(tool.defaults, strip=element)))
        self.assertFalse(cand.refusal)
        self.assertEqual(cand.numbers["strips"][1], 1)
        self.assertTrue(np.allclose(cand.co[:, 2], 0.0))
        self.assertTrue(np.allclose(cand.co[base.stay], co[base.stay]))
        clean, cfaces, _ = grid(6, 6)
        self.assertEqual(run(tool.run(trial.make_base(clean, cfaces, project=flat), dict(tool.defaults))).refusal, SF.PICK_FIRST)

    def test_mirrored_result_stays_mirrored(self):
        flat_grid, faces, at = grid(16, 16)
        partner = np.array([at[16 - i, j] for j in range(17) for i in range(17)])
        co, _faces, _at = wiggly((13, 14))
        left = flat_grid[:, 0] < 8.0
        co[left] = co[partner[left]] * (-1, 1, 1) + (16.0, 0.0, 0.0)  # the wiggle on both halves, exactly mirrored
        base = trial.make_base(co, faces, project=flat, mirror=(0, 8.0))
        tool = SF.Straighten()
        spots = run(tool.spots(base))
        self.assertEqual(len(spots), 1)  # found once, on the half the mirror copies from
        p = run(autofix.propose(tool, base, {}, spots))
        self.assertEqual(p.reason, "")
        new = p.cand.co
        self.assertTrue(np.allclose(new[partner, 0], 16.0 - new[:, 0]) and np.allclose(new[partner, 1], new[:, 1]))
        self.assertGreater(float(np.abs(new[left] - co[left]).max()), 0.05)  # the other half moved too


class EvenWidthTest(unittest.TestCase):
    def test_an_uneven_strip_gets_even(self):
        co, faces, at = grid(12, 16)
        co = co.copy()
        for j in range(1, 16):
            w = 0.3 * np.sin(2 * np.pi * j / 8)
            co[at[6, j], 0] += w
            co[at[5, j], 0] -= w
        base = trial.make_base(co, faces, project=flat)
        tool = SF.EvenWidth()
        spots = run(tool.spots(base))
        self.assertEqual([sp.kind for sp in spots], ["uneven"])
        self.assertEqual(run(SF.Straighten().spots(base)), [])  # straight, only uneven
        p = run(autofix.propose(tool, base, {}, spots))
        self.assertEqual(p.reason, "")
        self.assertLess(tool.own_number(base, p.cand.co, faces, spots[0]), 0.5 * spots[0].bad)

    def test_fix_every_strip_is_the_pair_through_every_spot(self):
        co, faces, _at = wiggly((2, 3))
        base = trial.make_base(co, faces, project=flat)
        every = autofix.every(SF.FixStrips)()
        self.assertIn(("keep_going", "Keep going"), every.toggles)
        self.assertFalse(SF.FixStrips.auto)  # Straighten and Even Width are Auto Fix's passes
        cand = run(every.run(base, {}))
        self.assertFalse(cand.refusal)
        self.assertGreater(float(np.abs(cand.co - co).max()), 0.05)
        clean, cfaces, _ = grid(8, 8)
        nothing = run(every.run(trial.make_base(clean, cfaces, project=flat), {}))
        self.assertTrue(nothing.refusal.startswith("No "))


class RemoveDeadEndTest(unittest.TestCase):
    def test_a_dead_end_between_poles_goes(self):
        co, faces, at = grid(12, 12)
        faces = spin(faces, at, 5, 5)
        base = trial.make_base(co, faces, project=flat)
        tool = SF.RemoveDeadEnd()
        spots = run(tool.spots(base))
        self.assertTrue(spots and all(sp.kind == "dead end" for sp in spots))
        p = run(autofix.propose(tool, base, {}, spots))
        self.assertEqual(p.reason, "")
        self.assertLess(p.after.poles, p.before.poles)
        self.assertLess(len(p.cand.faces), len(faces))
        self.assertEqual(len(p.cand.parent), len(p.cand.faces))
        self.assertTrue(p.left)  # the dead ends that would not make anything better are left as they were
        self.assertTrue(all(why in ("no better", "held", "bad faces") for _sp, why in p.left))

    def test_faceloops_remove_is_the_same_maths(self):
        co, faces, _at = grid(6, 4)
        base = trial.make_base(co, faces, project=flat)
        tool = SF.RemoveDeadEnd()
        cand = run(tool.run(base, {"strip": 5}))
        self.assertFalse(cand.refusal)
        self.assertEqual(len(cand.faces), len(faces) - len(tool._strips(base).walk(5, past_poles=True)["faces"]))
        self.assertTrue(all(len(set(f)) == 4 for f in cand.faces))


class RedrawInRowsTest(unittest.TestCase):
    def _messy(self):
        n = 24
        co, faces, at = grid(n, n)
        for i, j in ((9, 9), (12, 12), (9, 14), (13, 8)):
            faces = spin(faces, at, i, j)
        piece = np.array([1 if 7 <= k % n <= 16 and 7 <= k // n <= 16 else 0 for k in range(len(faces))])
        return trial.make_base(co, faces, project=flat, labels={"piece": piece})

    def test_a_messy_patch_is_found_and_redrawn_and_a_tiny_one_is_left(self):
        base = self._messy()
        spots = run(SF.messy_patches(base))
        self.assertEqual(len(spots), 1)
        self.assertEqual(spots[0].kind, "messy")
        tool = SF.RedrawInRows()
        tiny = Spot("messy", np.arange(4), base.co[:4].mean(axis=0), 0.5)  # too small for rows
        p = run(autofix.propose(tool, base, {}, spots + [tiny]))
        self.assertEqual(p.reason, "")
        self.assertLess(p.after.poles, p.before.poles)
        self.assertLessEqual(p.after.bad, p.before.bad)
        self.assertEqual([why for _sp, why in p.left], ["held"])

    def test_a_tidy_result_has_no_messy_patch(self):
        co, faces, _at = grid(12, 12)
        self.assertEqual(run(SF.messy_patches(trial.make_base(co, faces, project=flat))), [])


class EdgeCasesTest(unittest.TestCase):
    TOOLS = (SF.Straighten, SF.EvenWidth, SF.FixStrips, SF.RemoveDeadEnd, SF.RedrawInRows)

    def test_empty_mesh_and_nothing_free_find_no_spots(self):
        empty = trial.make_base(np.zeros((0, 3)), [], project=flat)
        co, faces, _at = wiggly((2, 3))
        boxed = trial.make_base(co, faces, scope=np.zeros(len(faces), dtype=bool), project=flat)
        for cls in self.TOOLS:
            for base in (empty, boxed):
                self.assertEqual(run(cls().spots(base)), [], cls.__name__)
                self.assertNotEqual(cls().check(base), "", cls.__name__)

    def test_no_spots_is_no_better(self):
        co, faces, _at = grid(6, 6)
        base = trial.make_base(co, faces, project=flat)
        for cls in (SF.Straighten, SF.RemoveDeadEnd):
            p = run(autofix.propose(cls(), base, {}, []))
            self.assertEqual(p.reason, "no better", cls.__name__)

    def test_auto_fix_plan_picks_them_up(self):
        names = [cls.__name__ for cls, _kinds in autofix.plan(set(autofix.GROUPS))]
        for name in ("RemoveDeadEnd", "RedrawInRows", "Straighten", "EvenWidth", "EvenTheBand", "OpenUpPoles", "CancelPolePair"):
            self.assertIn(name, names)
        self.assertNotIn("FixStrips", names)


if __name__ == "__main__":
    unittest.main()
