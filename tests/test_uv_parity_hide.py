# SPDX-License-Identifier: GPL-3.0-or-later
"""Seams out of sight (R7hide): the NumPy twin on its own, then twin against the compiled kernel.

Exact parity where the moves do not depend on a free layout (no move, or meshes the slides leave alone); on meshes
where charts with poles are laid out again the two flattens differ within their own parity tolerance (LSCM start in
the kernel, Tutte in the twin), so there both must make moves that keep every chart a disk and lower the seen cut.
"""

from __future__ import annotations

import math
import re
import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from test_uv_parity_chart import cube, speckled, tube
from triadq import uvchart, uvhide

from uv_parity_common import turbo  # noqa: E402  (None in a build without it)


def seen_px(mesh, seed=3):
    t = uvchart.build(*mesh, use_fast=False, merge=False).topo
    return np.random.default_rng(seed).uniform(0.2, 4.0, t.ne)


def slid(tree, x):
    raw = tree._level(x, hide=False)
    return raw, tree.level(x)


class Twin(unittest.TestCase):
    def test_min_cut_takes_the_cheap_side(self):
        g = uvhide._Flow(3)  # s - x - t with a cheap arc x - t
        g.add(0, 2, 5)
        g.add(2, 1, 2)
        flow, reach = uvhide._drive(g.run(0, 1))
        self.assertEqual(flow, 2)
        self.assertEqual(reach, [True, False, True])

    def test_stretch_reads_one_on_a_flat_layout_and_huge_when_squeezed(self):
        pts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        ip, fl = np.array([0, 4]), np.array([0, 1, 2, 3])
        q = uvhide._l2(pts, ip, fl, pts[:, :2].ravel() * 3.0)
        self.assertAlmostEqual(q[0], 1.0, places=12)
        self.assertEqual(q[2], 0.0)
        squeezed = np.array([0.0, 0, 1, 0, 1, 0.01, 0, 0.01])
        self.assertGreater(uvhide._l2(pts, ip, fl, squeezed)[1], 5.0)  # the UV Check's L2: a flattened side reads large
        self.assertEqual(uvhide._min_box(pts[:, :2].ravel()), (1.0, 1.0))

    def test_moves_yield_before_the_level_is_made(self):
        # the moves' first step must not make the whole level (replay, outline smoothing, chart check) at once: on a
        # 20k-face mesh that one step took 1.3 s with no compiled core (the bake unwrap froze Blender)
        mesh = speckled(16, 7, 0.4)
        tree = uvchart.build(*mesh, use_fast=False, seen_edge_px=seen_px(mesh), out_of_sight=True, px_per_unit=100.0, build_stretch=0.05)
        tree._smooth, tree._cache = None, {}
        gen = uvhide.moves(tree)
        next(gen)
        self.assertIsNone(tree._smooth)  # still being made, in later steps
        whole = uvhide._drive(gen)
        tree._smooth, tree._cache = None, {}
        again = uvhide._drive(uvhide.moves(tree))
        self.assertEqual(whole.moved, again.moved)
        tree._hide, tree._cache = again, {}
        level = tree.level_steps(5.0)  # the flatten's level, with the slides applied, is made in steps too
        next(level)
        self.assertEqual(tree._cache, {})
        got = uvhide._drive(level)
        tree._cache = {}
        want = tree.level(5.0)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
        np.testing.assert_array_equal(got[2][:5], want[2][:5])  # the last number is the time it took

    def test_slides_lower_the_seen_cut_and_keep_disks(self):
        mesh = speckled(16, 7, 0.4)
        seen = seen_px(mesh)
        tree = uvchart.build(*mesh, use_fast=False, seen_edge_px=seen, out_of_sight=True, px_per_unit=100.0, build_stretch=0.05)
        (rc, rcut, rn), (c, cut, n) = slid(tree, 5.0)
        self.assertGreater(uvhide.of(tree).moved, 0)
        self.assertEqual(n[1], 0, "every chart a disk")
        self.assertLess(seen[cut].sum(), 0.9 * seen[rcut].sum())
        self.assertLessEqual(n[0], rn[0])
        for x in (0.0, 2.0):  # lower levels stay disks too
            self.assertEqual(tree.level(x)[2][1], 0)

    def test_array_helpers_match_their_loops(self):
        rng = np.random.default_rng(2)
        for n in (1, 7, 60):
            a, b = rng.integers(0, n, 2 * n), rng.integers(0, n, 2 * n)
            d = uvchart._Dsu(n)
            for x, y in zip(a.tolist(), b.tolist()):
                d.union(x, y)
            self.assertEqual(uvhide._components(n, a, b).tolist(), [d.find(i) for i in range(n)])
            x, y, c = rng.integers(0, n + 2, 3 * n), rng.integers(0, n + 2, 3 * n), rng.integers(1, 9, 3 * n)
            keep = x != y
            g = uvhide._Flow(n + 2)
            for p, q, w in zip(x[keep].tolist(), y[keep].tolist(), c[keep].tolist()):
                g.add(p, q, w)
            h = uvhide._Flow.from_arcs(n + 2, x[keep], y[keep], c[keep])
            self.assertEqual((g.head, g.to, g.nxt, g.cap), (h.head, h.to, h.nxt, h.cap))
        square = [(0, 1), (1, 2), (2, 3), (3, 0)]
        self.assertFalse(uvhide._border_crosses(np.array([0.0, 0, 1, 0, 1, 1, 0, 1]), square))
        self.assertTrue(uvhide._border_crosses(np.array([0.0, 0, 1, 0, 0, 1, 1, 1]), square))  # a bow tie
        self.assertEqual(uvhide._min_box(np.array([0.0, 0, 2, 0, 2, 1, 0, 1, 1, 0.5]), [0, 1, 2, 3]), (2.0, 2.0))

    def test_twin_names_no_compiled_core(self):
        text = (common.ROOT / "triadq" / "uvhide.py").read_text(encoding="utf-8").lower()
        for word in ("turbo", "native", "rust", "gpu"):
            self.assertNotIn(word, text)


@needs_native
class Parity(unittest.TestCase):
    def kernel(self, mesh, **kw):
        job = turbo.uvchart_build(*mesh, **kw)
        self.assertIsNotNone(job)
        job._thread.join(120)
        return job.result()

    def test_constants_match(self):
        rs = common.rs_constants("hide")
        for name, value in rs.items():
            self.assertEqual(float(getattr(uvhide, name)), value, name)
        self.assertGreaterEqual(len(rs), 9)
        text = (common.ROOT / "native" / "src" / "uvhide.rs").read_text(encoding="utf-8")
        rings = re.search(r"^pub const RINGS: \[usize; \d+\] = \[([0-9, ]+)\];", text, re.M).group(1)
        self.assertEqual(tuple(int(x) for x in rings.split(",")), uvhide.RINGS)

    def test_levels_match_exactly_where_nothing_slides(self):
        for mesh in (tube(16, 5), cube(4)):
            seen = seen_px(mesh)
            kw = dict(seen_edge_px=seen, out_of_sight=True, px_per_unit=100.0)
            twin = uvchart.build(*mesh, use_fast=False, **kw)
            kern = self.kernel(mesh, **kw)
            for x in (0.0, 10.0):
                tc, tcut, tn = twin.level(x)
                kc, kcut, kn = kern.level(x)
                np.testing.assert_array_equal(kc, tc)
                np.testing.assert_array_equal(kcut, tcut)
                np.testing.assert_array_equal(kn[:5], tn[:5])

    def test_both_slide_borders_out_of_sight(self):
        mesh = speckled(16, 7, 0.4)
        seen = seen_px(mesh)
        kw = dict(seen_edge_px=seen, out_of_sight=True, px_per_unit=100.0, build_stretch=0.05)
        twin = uvchart.build(*mesh, use_fast=False, **kw)
        kern = self.kernel(mesh, **kw)
        (_, rcut, rn), (_, tcut, tn) = slid(twin, 5.0)
        kc, kcut, kn = kern.level(5.0)
        raw = seen[rcut].sum()
        for cut, n in ((tcut, tn), (np.asarray(kcut, bool), kn)):
            self.assertEqual(n[1], 0)
            self.assertLessEqual(n[0], rn[0])
            self.assertLess(seen[cut].sum(), 0.9 * raw)
        a, b = seen[tcut].sum(), seen[np.asarray(kcut, bool)].sum()
        self.assertLess(abs(a - b) / max(a, b), 0.25, (a, b))
        self.assertTrue(math.isfinite(a))


@needs_native
class Late(unittest.TestCase):
    """R8late: the first result comes from the charts before the slides, the slid ones are swapped in when ready."""

    def test_first_result_before_the_slides_then_the_slid_charts(self):
        fast = common.addon_module("uv_fast")
        kn = common.addon_module("triadq.uvnew")
        kt = common.addon_module("turbo")
        try:
            self.assertTrue(fast.install())
            mesh = speckled(16, 7, 0.4)
            seen = seen_px(mesh)
            job = kt.uvchart_build(*mesh, seen_edge_px=seen, out_of_sight=True, px_per_unit=100.0, build_stretch=0.05)
            job._thread.join(120)
            tree = job.result()
            self.assertFalse(kt.uvhide_ready(tree))
            settings = kn.Settings(out_of_sight=False, hidden_pct=100.0, texture_px=256, stretch_allowed=5.0)
            stage = kn.NewUVs(*mesh, settings=settings, _keep={"tree": tree})
            shown = None
            while not stage.done:
                stage.step()
                if stage.first_ready and shown is None:
                    shown = (stage.revision, stage.result)
                if stage.waiting:
                    time.sleep(0.0005)
            self.assertIsNone(stage.error)
            self.assertTrue(kt.uvhide_ready(tree))
            self.assertGreater(stage.revision, shown[0] + 1, "the slid layout came after the first one")
            self.assertIn("hide_s", stage.result.timing)
            first_cut, last_cut = (seen[np.asarray(kn.cut_edges(mesh[1], mesh[2], r.uv), bool)[: len(seen)]].sum() for r in (shown[1], stage.result))
            self.assertLess(last_cut, first_cut)
            again = stage.again(stretch_allowed=4.0)  # the moves are made: straight to the slid charts
            again.run()
            self.assertTrue(again._hidden)
            nxt = stage.keep_going()  # Keep going packs the slid charts again
            nxt.run()
            np.testing.assert_array_equal(nxt.result.chart, stage.result.chart)
        finally:
            fast.uninstall()


if __name__ == "__main__":
    unittest.main()
