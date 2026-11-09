# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
import unittest

import numpy as np

from triadq import autofix, trial
from triadq import loopspoles as L
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


def _run(work):
    return trial.finish(work)


def _held(base, spot):
    """`base` with every corner of `spot` held: a spot nothing can make better."""
    stay = np.zeros(len(base.co), dtype=bool)
    stay[np.unique([v for f in spot.faces.tolist() for v in base.faces[f]])] = True
    return trial.make_base(base.co, base.faces, stay=stay, project=base.project, mirror=base.mirror)


def _spun_at(n: int, spins):
    co, faces, at = _grid(n, n)
    faces = list(faces)
    for i, j in spins:
        f, g = faces.index((at[i, j], at[i + 1, j], at[i + 1, j + 1], at[i, j + 1])), faces.index((at[i + 1, j], at[i + 2, j], at[i + 2, j + 1], at[i + 1, j + 1]))
        faces[f] = (at[i, j], at[i + 1, j], at[i + 2, j], at[i + 2, j + 1])
        faces[g] = (at[i, j], at[i + 2, j + 1], at[i + 1, j + 1], at[i, j + 1])
    return co, faces, at


class SpotsTest(unittest.TestCase):
    """Auto Fix (AUTO_FIX.md 6.2): each tool finds its planted problem, makes its own number better, and puts back
    a planted spot that cannot improve."""

    def test_even_the_band_finds_the_bunched_bands(self):
        xs = [0.0, 0.3, 0.6, 2.0, 3.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 12.2, 12.4, 14.0, 15.6, 17.0, 18.0]
        co, faces, _at = _grid(18, 6, xs=xs)  # two bunched bands, a pole-free run between them
        base = trial.make_base(co, faces, project=_flat)
        tool = L.EvenTheBand()
        spots = _run(tool.spots(base))
        self.assertTrue(spots and all(sp.kind == "bunched" and sp.bad > L.BUNCHED for sp in spots))
        p = _run(autofix.propose(tool, base, {}, spots))
        self.assertEqual(p.reason, "")
        for sp in spots:
            self.assertLess(tool.own_number(base, p.cand.co, faces, sp), sp.bad)
        p = _run(autofix.propose(tool, _held(base, spots[0]), {}, spots[:1]))
        self.assertEqual([why for _sp, why in p.left], [p.left[0][1]] if p.left else ["missing"])
        self.assertIn(p.left[0][1], ("no better", "held"))
        self.assertEqual(p.reason, "no better")

    def test_open_up_poles_finds_the_pinched_pole(self):
        co, faces, at = _spun_at(12, [(5, 5)])
        co = co.copy()
        co[at[6, 6]] += (0.35, -0.3, 0.0)
        base = trial.make_base(co, faces, project=_flat)
        tool = L.OpenUpPoles()
        spots = _run(tool.spots(base))
        self.assertEqual([sp.kind for sp in spots], ["pinched"])
        p = _run(autofix.propose(tool, base, {}, spots))
        self.assertEqual(len(p.fixed), 1)  # its own number got better (Loop Quality may still say no)
        self.assertLess(tool.own_number(base, p.cand.co, faces, spots[0]), tool.own_number(base, co, faces, spots[0]))
        p = _run(autofix.propose(tool, _held(base, spots[0]), {}, spots))
        self.assertTrue(p.left and not p.fixed)
        self.assertEqual(_run(tool.spots(trial.make_base(*_grid(4, 4)[:2], project=_flat))), [])

    def test_cancel_pole_pair_pairs_and_cancels(self):
        co, faces, _at = _spun_at(12, [(5, 5)])
        base = trial.make_base(co, faces, project=_flat)
        tool = L.CancelPolePair()
        spots = _run(tool.spots(base))
        self.assertEqual(len(spots), 2)  # two 3-5 pairs round the spun edge
        self.assertTrue(all(sp.pick["pole"] != sp.pick["partner"] for sp in spots))
        p = _run(autofix.propose(tool, base, {}, spots))
        self.assertEqual(p.reason, "")
        self.assertEqual(p.after.poles, 0)
        self.assertEqual(len(p.cand.faces), len(faces))
        self.assertEqual([why for _sp, why in p.left], ["held"])  # the second pair went with the first walk
        self.assertEqual(_run(tool.spots(trial.make_base(co, faces, project=_flat, mirror=(0, 6.0)))), [])

    def test_tries_and_reach_are_settings(self):
        co, faces, at = _spun_grid()
        base = trial.make_base(co, faces, project=_flat)
        tool = L.CancelPolePair()
        self.assertIn("tries", tool.defaults)
        self.assertIn("reach", tool.defaults)
        three = at[3, 2]
        partner = int(tool.targets(base, dict(tool.defaults, pole=three), "partner")[0][0])
        self.assertFalse(_run(tool.run(base, dict(tool.defaults, pole=three, partner=partner, tries=100000))).refusal)

    def test_empty_mesh_nothing_free_and_mirrored(self):
        empty = trial.make_base(np.zeros((0, 3)), [], project=_flat)
        co, faces, _at = _spun_at(8, [(3, 3)])
        boxed = trial.make_base(co, faces, scope=np.zeros(len(faces), dtype=bool), project=_flat)
        for cls in (L.EvenTheBand, L.OpenUpPoles, L.CancelPolePair):
            for base in (empty, boxed):
                self.assertEqual(_run(cls().spots(base)), [], cls.__name__)
        xs = [0.0, 1.0, 2.0, 2.2, 2.4, 4.0, 5.6, 6.0, 6.4, 8.0, 9.6, 10.0, 10.2, 10.4, 12.0, 13.0, 14.0]
        co, faces, _at = _grid(16, 4, xs=xs)  # bunched the same way on both sides of x = 7
        base = trial.make_base(co, faces, project=_flat, mirror=(0, 7.0))
        spots = _run(L.EvenTheBand().spots(base))
        self.assertTrue(spots)
        self.assertTrue(all(sp.at[0] >= 7.0 - 1e-9 for sp in spots))  # found on the half the mirror copies from


if __name__ == "__main__":
    unittest.main()
