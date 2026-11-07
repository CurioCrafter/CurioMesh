# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
import unittest

import numpy as np

from triadq import loopspoles as L
from triadq import trial
from triadq.patterns import Topology


def _grid(nx: int, ny: int, xs=None):
    xs = np.arange(nx + 1, dtype=np.float64) if xs is None else np.asarray(xs, dtype=np.float64)
    at = {(i, j): j * (nx + 1) + i for j in range(ny + 1) for i in range(nx + 1)}
    co = np.array([(xs[i], float(j), 0.0) for j in range(ny + 1) for i in range(nx + 1)])
    faces = [(at[i, j], at[i + 1, j], at[i + 1, j + 1], at[i, j + 1]) for j in range(ny) for i in range(nx)]
    return co, faces, at


def _flat(points):
    out = np.array(points, dtype=np.float64)
    out[:, 2] = 0.0
    return out


def _done(work):
    slices = 0
    while True:
        try:
            next(work)
            slices += 1
        except StopIteration as end:
            return end.value, slices


class EvenTheBandTest(unittest.TestCase):
    def test_bunched_strips_get_even_and_stay_on_the_lines(self):
        co, faces, at = _grid(6, 4, xs=[0.0, 0.3, 0.6, 2.0, 3.5, 5.0, 6.0])  # columns of very different widths
        base = trial.make_base(co, faces, project=_flat)
        tool = L.EvenTheBand()
        self.assertEqual(tool.waiting(tool.defaults), "strip")
        self.assertEqual(tool.check(base), "")
        face = 2 * 6 + 2  # a quad in the middle, its vertical edge picked: the strip running up and down
        p = base.co[list(faces[face])]
        element = tool.pick_at(base, {}, "strip", face, 0.5 * (p[1] + p[2]))
        self.assertGreaterEqual(element, 0)
        cand, slices = _done(tool.run(base, dict(tool.defaults, strip=element)))
        self.assertFalse(cand.refusal)
        self.assertGreater(slices, 0)
        self.assertEqual(cand.numbers["spread"][1], 1)  # better
        new = cand.co
        rows = new[np.flatnonzero((base.co[:, 1] > 0) & (base.co[:, 1] < 4))]
        self.assertTrue(np.allclose(rows[:, 1], np.round(rows[:, 1])))  # points only slid along their rung lines
        self.assertTrue(np.allclose(new[:, 2], 0.0))
        self.assertTrue(np.allclose(new[base.stay], base.co[base.stay]))  # the border holds
        middle = new[[at[i, 2] for i in range(7)], 0]
        widths = np.diff(middle)
        self.assertLess(widths.std() / widths.mean(), 0.25)
        self.assertGreater(len(tool.picked_lines(base, {"strip": element})), 0)

    def test_half_strength_is_half_way(self):
        co, faces, _at = _grid(6, 4, xs=[0.0, 0.3, 0.6, 2.0, 3.5, 5.0, 6.0])
        base = trial.make_base(co, faces, project=_flat)
        tool = L.EvenTheBand()
        element = tool.pick_at(base, {}, "strip", 14, base.co[list(faces[14])][1:3].mean(axis=0))
        full, _ = _done(tool.run(base, {"strength": 1.0, "strip": element}))
        half, _ = _done(tool.run(base, {"strength": 0.5, "strip": element}))
        self.assertTrue(np.allclose(half.co, 0.5 * (base.co + full.co), atol=1e-9))

    def test_a_strip_with_no_neighbours_refuses(self):
        co, faces, _at = _grid(1, 3)  # one column: the border is on both sides of every strip
        base = trial.make_base(co, faces, project=_flat)
        tool = L.EvenTheBand()
        element = tool.pick_at(base, {}, "strip", 1, base.co[list(faces[1])][1:3].mean(axis=0))
        cand, _ = _done(tool.run(base, {"strength": 1.0, "strip": element}))
        self.assertEqual(cand.refusal, L.NO_NEIGHBOURS)


def _spun_grid():
    """A flat 6 x 6 grid with one inner edge spun inside its two quads: two 3s and two 5s round it."""
    co, faces, at = _grid(6, 6)
    faces = list(faces)
    f, g = faces.index((at[2, 2], at[3, 2], at[3, 3], at[2, 3])), faces.index((at[3, 2], at[4, 2], at[4, 3], at[3, 3]))
    faces[f] = (at[2, 2], at[3, 2], at[4, 2], at[4, 3])
    faces[g] = (at[2, 2], at[4, 3], at[3, 3], at[2, 3])
    return co, faces, at


class CancelPolePairTest(unittest.TestCase):
    def test_a_three_and_a_five_cancel_with_the_same_faces(self):
        co, faces, at = _spun_grid()
        base = trial.make_base(co, faces, project=_flat)
        tool = L.CancelPolePair()
        self.assertEqual(tool.check(base), "")
        poles, labels = tool.targets(base, tool.defaults, "pole")
        self.assertEqual(sorted(labels), ["3", "3", "5", "5"])
        three = at[3, 2]
        settings = dict(tool.defaults, pole=three)
        self.assertEqual(tool.waiting(settings), "partner")
        partners, numbers = tool.targets(base, settings, "partner")
        self.assertTrue(len(partners) and all(base.topo.code[w] == 5 for w in partners.tolist()))
        self.assertEqual(numbers[0], "1")
        cand, slices = _done(tool.run(base, dict(settings, partner=int(partners[0]))))
        self.assertFalse(cand.refusal)
        self.assertGreater(slices, 1)
        self.assertEqual(len(cand.faces), len(faces))
        self.assertTrue(all(len(set(f)) == 4 for f in cand.faces))
        after = Topology(cand.co, cand.faces)
        self.assertEqual(int(after.code[three]), 4)
        self.assertEqual(int(after.code[partners[0]]), 4)
        self.assertLess(int(after.poles.sum()), int(base.topo.poles.sum()))
        self.assertTrue(np.allclose(cand.co[:, 2], 0.0))
        trial.diff(base, cand)
        self.assertGreaterEqual(len(cand.poles_gone), 2)

    def test_no_poles_and_no_partner_refuse(self):
        co, faces, _at = _grid(4, 4)
        tool = L.CancelPolePair()
        self.assertEqual(tool.check(trial.make_base(co, faces, project=_flat)), L.NO_POLES)
        co, faces, at = _spun_grid()
        base = trial.make_base(co, faces, project=_flat)
        cand, _ = _done(tool.run(base, dict(tool.defaults, pole=at[3, 2], partner=at[3, 3])))  # two 3s cannot cancel
        self.assertEqual(cand.refusal, L.NO_PARTNER)

    def test_mirrored_results_refuse(self):
        co, faces, _at = _spun_grid()
        base = trial.make_base(co, faces, project=_flat, mirror=(0, 3.0))
        self.assertEqual(L.CancelPolePair().check(base), L.MIRRORED)


class OpenUpPolesTest(unittest.TestCase):
    def test_a_pinched_pole_opens_and_stays_on_the_surface(self):
        co, faces, at = _spun_grid()
        co = co.copy()
        co[at[3, 3]] += (0.35, -0.3, 0.0)  # pinch the quads round the poles
        base = trial.make_base(co, faces, project=_flat)
        tool = L.OpenUpPoles()
        self.assertEqual(tool.check(base), "")
        cand, slices = _done(tool.run(base, dict(tool.defaults, strength=1.0)))
        self.assertGreater(slices, 1)
        self.assertEqual(cand.numbers["narrow"][1], 1)
        fans = tool._poles(base)
        self.assertGreater(L._mean_narrowest(cand.co, fans), L._mean_narrowest(base.co, fans) + 1.0)
        self.assertTrue(np.allclose(cand.co[:, 2], 0.0))
        self.assertTrue(np.allclose(cand.co[base.stay], base.co[base.stay]))
        half, _ = _done(tool.run(base, dict(tool.defaults, strength=0.5)))
        self.assertTrue(np.allclose(half.co, 0.5 * (base.co + cand.co)))

    def test_no_poles_refuses(self):
        co, faces, _at = _grid(4, 4)
        self.assertEqual(L.OpenUpPoles().check(trial.make_base(co, faces, project=_flat)), L.NO_POLES)


class RingsTest(unittest.TestCase):
    def test_rings_grow_one_face_ring_at_a_time(self):
        co, faces, at = _grid(6, 6)
        patch, ring = L.rings_of(faces, len(co), [at[3, 3]], 1)
        self.assertEqual(len(patch), 4)
        self.assertEqual(int(ring[at[2, 2]]), 1)
        patch, ring = L.rings_of(faces, len(co), [at[3, 3]], 2)
        self.assertEqual(len(patch), 16)


if __name__ == "__main__":
    unittest.main()
