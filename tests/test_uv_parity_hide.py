# SPDX-License-Identifier: GPL-3.0-or-later
"""Seams out of sight (R7hide): the NumPy twin on its own, then twin against the compiled kernel.

Exact parity where the moves do not depend on a free layout (no move, or meshes the slides leave alone); on meshes
where charts with poles are laid out again the two flattens differ within their own parity tolerance (LSCM start in
the kernel, Tutte in the twin), so there both must make moves that keep every chart a disk and lower the seen cut.
"""

from __future__ import annotations

import math
import re
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from test_uv_parity_chart import cube, speckled, tube
from triadq import uvchart, uvhide

import turbo


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


if __name__ == "__main__":
    unittest.main()
