# SPDX-License-Identifier: GPL-3.0-or-later
"""UV Trials (A3): Seam Fit through Look First's Trial shape, the applied fit equals the previewed one (hash), the UV-only
Trials (New, New page layout, Bake those spots) give valid layouts in slices, Keep going never lowers space used."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_parity_common as common

from test_uv_keep import grid, two_island_source

UT = common.addon_module("triadq.uvtrial")
T = common.addon_module("triadq.trial")
keep = common.addon_module("triadq.uvkeep")
uvscore = common.addon_module("triadq.uvscore")
uvnew = common.addon_module("triadq.uvnew")


def faces_of(ip, fl):
    return [tuple(fl[ip[i]:ip[i + 1]].tolist()) for i in range(len(ip) - 1)]


def run_sliced(work):
    """Finish a generator, returning (value, longest step in s, steps)."""
    longest, steps = 0.0, 0
    while True:
        t = time.perf_counter()
        try:
            next(work)
        except StopIteration as fin:
            return fin.value, max(longest, time.perf_counter() - t), steps
        longest = max(longest, time.perf_counter() - t)
        steps += 1


class SeamFitTrial(unittest.TestCase):
    def setUp(self):
        self.sa = two_island_source()
        s = keep.Settings(mode="light", allowance=1.0)
        self.plan = keep.make_plan(self.sa, 25.0, s)
        UT.FIT["inputs"] = lambda: dict(source=self.sa, plan=self.plan, settings=s, material=None)

    def tearDown(self):
        UT.FIT["inputs"] = None
        keep.CANDIDATES.clear()
        UT.APPLIED_EXTRA.clear()

    def test_preview_then_apply_writes_the_same_fit(self):
        p, ip, fl = grid(5)
        base = T.make_base(p, faces_of(ip, fl))
        trial = UT.SeamFit()
        self.assertEqual(trial.check(base), "")
        cand, longest, steps = run_sliced(trial.run(base, dict(trial.defaults, faces=1.0)))
        self.assertFalse(cand.refusal, cand.refusal)
        self.assertGreaterEqual(len(cand.faces), len(base.faces))  # the side-line snap may fit this grid with no split
        self.assertEqual(len(cand.parent), len(cand.faces))
        self.assertTrue(cand.lines.dtype.kind in "iu")  # split lines, if any
        self.assertTrue(len(cand.lines) or cand.numbers["split"][1])  # something was split or moved
        self.assertGreater(steps, 1)
        # the result after Apply is the fitted mesh: its own Apply must find and write exactly this candidate
        key = UT.remember_applied(cand)
        f = common.addon_module("triadq.types").FlatFaces.of(cand.faces)
        self.assertEqual(key, keep.result_key(cand.co, f.indptr, f.flat))
        self.assertEqual(keep.CANDIDATES[key].key, cand.fit.key)
        from uv_hooks_fake import FakeMesh

        layers = common.addon_module("uv_layers")
        mesh = FakeMesh()
        c = keep.CANDIDATES[key]
        layers.apply_candidate(mesh, c.points, c.indptr, c.flat, c.material, c.layers, c.active)
        pts, iptr, flat, _uv, mat = layers.geometry(mesh)
        got = keep.candidate_hash(pts, iptr, flat, mat, [(x.name, layers.read_uv(mesh, x.name)) for x in mesh.uv_layers])
        self.assertEqual(got, cand.fit.key)

    def test_auto_fix_spot_is_found_and_fixed(self):
        """Auto Fix's pattern: SeamFit made with its inputs, one spot, own number to 0 after the fit."""
        af = common.addon_module("triadq.autofix")
        p, ip, fl = grid(5)
        base = T.make_base(p, faces_of(ip, fl))
        trial = UT.SeamFit(UT.FIT["inputs"]())
        UT.FIT["inputs"] = None  # the trial needs nothing the panel sets
        spots = UT.finish(trial.spots(base))
        self.assertEqual(len(spots), 1)
        self.assertGreater(spots[0].bad, trial.least)
        prop = UT.finish(af.propose(trial, base, {}, spots))
        self.assertEqual(len(prop.fixed), 1, prop.left)
        for cls in (UT.SeamFit, UT.BakeSpots):  # what autofix.plan asks of a pass
            self.assertTrue(cls.auto and cls.spots is not T.Trial.spots)
        self.assertEqual(UT.finish(UT.BakeSpots().spots(base)), [])  # no UVs before Apply: no spots

    def test_no_uvs_refuses_with_the_message(self):
        UT.FIT["inputs"] = lambda: None
        p, ip, fl = grid(3)
        self.assertEqual(UT.SeamFit().check(T.make_base(p, faces_of(ip, fl))), UT.NO_UVS)

    def test_dial_end_is_any(self):
        self.assertIsNone(UT.allowance_of(1.0))
        self.assertAlmostEqual(UT.allowance_of(0.15), 0.15)
        self.assertEqual(UT.SeamFit().value_text(1.0), "Extra faces any")


def cube_mesh(n=4):
    """A closed box made of n x n quads per side (welded)."""
    pts, idx, faces = [], {}, []

    def vid(q):
        q = tuple(np.round(q, 9))
        if q not in idx:
            idx[q] = len(pts)
            pts.append(q)
        return idx[q]

    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = [a for a in range(3) if a != axis]
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        q = np.zeros(3)
                        q[axis], q[u], q[v] = sign, -1 + 2 * (i + di) / n, -1 + 2 * (j + dj) / n
                        quad.append(vid(q))
                    faces.append(quad if sign > 0 else quad[::-1])
    flat = np.array([v for f in faces for v in f], np.int64)
    return np.array(pts), np.arange(0, len(flat) + 1, 4), flat


class UVOnly(unittest.TestCase):
    def test_new_then_layout_keep_going(self):
        p, ip, fl = cube_mesh(4)
        base = UT.make_uv_base(p, ip, fl, texture_px=512)
        trial = UT.NewUVs()
        settings = dict(trial.defaults, new=uvnew.Settings(out_of_sight=False, hidden_pct=100.0, time_per_pass=0.5))
        cand, longest, _ = run_sliced(trial.run(base, settings))
        self.assertFalse(cand.refusal, cand.refusal)
        self.assertEqual(cand.uv.shape, (len(fl), 2))
        cand, _, _ = run_sliced(UT.measure(trial, base, cand))
        self.assertIn(cand.verdict[0], ("GOOD", "CHECK", "BAD"))
        self.assertEqual(cand.numbers["flipped_faces"], 0)
        # New page layout on that layout, then Keep going: never lower space used, islands kept
        lb = UT.make_uv_base(p, ip, fl, uv=cand.uv, texture_px=512)
        lt = UT.PageLayout()
        first, _, _ = run_sliced(lt.run(lb, dict(lt.defaults, time=0.5)))
        self.assertFalse(first.refusal, first.refusal)
        more, _, _ = run_sliced(lt.run(lb, dict(lt.defaults, time=0.5, keep_going=True), first))
        self.assertGreaterEqual(more.gain, first.gain)
        a, na = UT.islands(ip, fl, cand.uv)
        b, nb = UT.islands(ip, fl, more.uv)
        self.assertEqual(na, nb)
        more, _, _ = run_sliced(UT.measure(lt, lb, more))
        self.assertLess(more.numbers["overlap_pct"], 0.01)

    def test_cut_toggle(self):
        cuts = UT.toggle_cut((), 5, 2)
        self.assertEqual(cuts, ((2, 5),))
        self.assertEqual(UT.toggle_cut(cuts, 2, 5), ())

    def test_bake_spots_gives_straddlers_their_own_islands(self):
        sa = two_island_source()
        p, ip, fl = grid(5)
        s = keep.Settings(mode="light", allowance=1.0)
        plan = keep.make_plan(sa, 25.0, s)
        alone = keep.run(keep.FitStage(sa, plan, p, ip, fl, None, s, only_transfer=True, use_fast=False))  # UVs carried over, no fit
        src = uvscore.mesh(sa.points, sa.indptr, sa.flat, sa.layers[0][1], sa.material)
        base = UT.make_uv_base(alone.points, alone.indptr, alone.flat, uv=alone.layers[0][1], texture_px=1024, source=src)
        trial = UT.BakeSpotsUV()
        self.assertEqual(trial.check(base), "")
        cand, _, _ = run_sliced(trial.run(base, dict(trial.defaults)))
        self.assertFalse(cand.refusal, cand.refusal)
        straddle = UT.finish(UT.straddlers(base))
        self.assertEqual(set(cand.faces.tolist()), set(np.flatnonzero(straddle).tolist()))
        lab, _ = UT.islands(alone.indptr, alone.flat, cand.uv)
        others = set(lab[~straddle].tolist())
        self.assertFalse(others & set(lab[cand.faces].tolist()))  # the new islands are their own
        cand, _, _ = run_sliced(UT.measure(trial, base, cand))
        self.assertLess(cand.numbers["overlap_pct"], 0.01 + base.cache["numbers"]["overlap_pct"])


class KindsAsIslands(unittest.TestCase):
    def test_every_code_names_a_real_island_and_codes_stay_apart(self):
        hooks = common.addon_module("uv_hooks")
        codes = np.array([0, 0, 0, 1, 1, 2])
        island = np.array([5, 5, 7, 7, 7, 9])  # code 0 lies on islands 5 and 7, code 1 on 7, code 2 on 9
        _c, _m, out = hooks.as_islands((codes, np.zeros(3, np.int64), np.arange(3)), island)
        self.assertEqual(out.tolist(), [5, 7, 9])
        self.assertTrue(set(out.tolist()) <= set(island.tolist()))


class Names(unittest.TestCase):
    AVOID = ("New UV Map", "Keep Seams", "Keep My Seams", "Pack Islands", "Pack Again", "Repack", "Hide Cuts", "Mark Seam", "Protect UV",
             "Fewer Islands", "Low Stretch", "Bake Into Existing", "Even Out UVs", "Utilization", "Straighten", "Rectify")

    def test_no_avoided_names_in_the_ui_files(self):
        for name in ("uv_panel.py", "uv_draw.py", "triadq/uvtrial.py"):
            text = (common.ROOT / name).read_text(encoding="utf-8")
            for word in self.AVOID:
                self.assertNotIn(word, text, f"{word!r} in {name}")

    def test_no_engine_words_in_the_core(self):
        import re

        text = (common.ROOT / "triadq" / "uvtrial.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"(?i)turbo|native|rust|\bgpu\b", text))


if __name__ == "__main__":
    unittest.main()
