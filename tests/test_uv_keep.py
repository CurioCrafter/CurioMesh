# SPDX-License-Identifier: GPL-3.0-or-later
"""Add-on Keep (A1): kinds, reconciliation, the fit stage on a small plane, and that the written mesh hashes like the fit."""

from __future__ import annotations

import unittest

import numpy as np

import uv_common as common  # noqa: F401  (puts the add-on folder on the path)

from triadq import uvkeep, uvscore, uvseams


def grid(n, size=1.0):
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    points = np.stack([size * i.ravel() / n, size * j.ravel() / n, 0.0 * i.ravel()], 1)
    ids = (j * (n + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + n + 2, ids + n + 1], 1).ravel()
    return points, np.arange(0, len(flat) + 1, 4), flat


def two_island_source(n=16):
    """A plane whose left half and right half are separate UV islands with a 0.1 gap (about 100 px at 1024)."""
    p, ip, fl = grid(n)
    cx = p[fl][:, 0].reshape(-1, 4)
    uv = p[fl][:, :2].copy()
    right = np.repeat(cx.mean(1) > 0.5, 4)
    uv[:, 0] = np.where(right, 0.55 + (uv[:, 0] - 0.5) * 0.9, uv[:, 0] * 0.9)
    return uvkeep.Source(p, ip, fl, np.zeros(len(ip) - 1, np.int64), [("UVMap", uv)], 0, ("t", 1, "x"))


def straddlers(sa, p, ip, fl, uv):
    src = uvscore.mesh(sa.points, sa.indptr, sa.flat, sa.layers[0][1], sa.material)
    res = uvscore.mesh(p, ip, fl, uv, np.zeros(len(ip) - 1, np.int64))
    return uvscore.as_dict(uvscore.check(res, src).numbers)["straddle_faces"]


class Fit(unittest.TestCase):
    def test_fit_removes_straddlers_and_hash_is_written(self):
        sa = two_island_source()
        p, ip, fl = grid(5)  # edges at 0.2 steps: the seam at 0.5 runs through the middle column
        s = uvkeep.Settings(mode="light", allowance=1.0)
        plan = uvkeep.make_plan(sa, 25.0, s)
        self.assertIsNotNone(plan.codes)
        alone = uvkeep.run(uvkeep.FitStage(sa, plan, p, ip, fl, None, s, only_transfer=True))
        fit = uvkeep.run(uvkeep.FitStage(sa, plan, p, ip, fl, None, s))
        before = straddlers(sa, alone.points, alone.indptr, alone.flat, alone.layers[0][1])
        after = straddlers(sa, fit.points, fit.indptr, fit.flat, fit.layers[0][1])
        self.assertGreater(before, 0)
        self.assertLessEqual(after, before / 2, (before, after))
        self.assertGreater(fit.extra_faces + sum(fit.numbers.get(k, 0) for k in ("moved", "side_moved", "ball_moved", "ball_after_moved")), 0)  # cut or slid
        # Apply writes exactly the fitted candidate: the hash read back from the mesh equals the stage's
        from uv_hooks_fake import FakeMesh

        layers = common.addon_module("uv_layers")
        mesh = FakeMesh()
        layers.apply_candidate(mesh, fit.points, fit.indptr, fit.flat, fit.material, fit.layers, fit.active)
        pts, iptr, flat, _uv, mat = layers.geometry(mesh)
        got = uvkeep.candidate_hash(pts, iptr, flat, mat, [(x.name, layers.read_uv(mesh, x.name)) for x in mesh.uv_layers])
        self.assertEqual(got, fit.key)

    def test_again_continues_from_the_result(self):
        sa = two_island_source()
        p, ip, fl = grid(5)
        s = uvkeep.Settings(mode="light", allowance=1.0)
        plan = uvkeep.make_plan(sa, 25.0, s)
        first = uvkeep.run(uvkeep.FitStage(sa, plan, p, ip, fl, None, s))
        second = uvkeep.run(uvkeep.again(first, sa, plan, s))
        self.assertLessEqual(straddlers(sa, second.points, second.indptr, second.flat, second.layers[0][1]),
                             straddlers(sa, first.points, first.indptr, first.flat, first.layers[0][1]))


def fake_seams(classes, n_islands=2, pairs=None):
    n = len(classes)
    pairs = pairs or [(0, 1)] * n
    return {"n_islands": n_islands, "island": np.arange(n_islands), "chain_island": np.array(pairs, np.int64).ravel(),
            "chain_material": np.zeros(2 * n, np.int64), "chain_length": np.full(n, 1.0), "run_indptr": np.arange(n + 1),
            "gap_px": np.ones(n), "class": np.array(classes, np.int64)}


class SeamCheckTwin(unittest.TestCase):
    def test_sliced_check_matches_the_uv_check(self):
        sa = two_island_source()
        p, ip, fl = grid(5)
        p = p + np.random.default_rng(3).normal(0.0, 0.01, p.shape) * [1, 1, 0]
        s = uvkeep.Settings(mode="light")
        plan = uvkeep.make_plan(sa, 25.0, s)
        alone = uvkeep.run(uvkeep.FitStage(sa, plan, p, ip, fl, None, s, only_transfer=True))
        uv = alone.layers[0][1].copy()
        uv[:4] = uv[:4][::-1]  # one face turned over
        want = uvscore.check(uvscore.mesh(alone.points, alone.indptr, alone.flat, uv), uvscore.mesh(sa.points, sa.indptr, sa.flat, sa.layers[0][1], sa.material),
                             texture_px=s.texture_px)
        check = uvkeep.SeamCheck(sa, sa.layers[0][1], s.texture_px)
        check.CHUNK = 3  # many slices
        gen = check.steps(alone.points, alone.indptr, alone.flat, uv)
        while True:
            try:
                next(gen)
            except StopIteration as done:
                straddle, flipped = done.value
                break
        np.testing.assert_array_equal(straddle, want.straddle)
        np.testing.assert_array_equal(flipped, want.flipped)
        self.assertTrue(want.straddle.any() and want.flipped.any())


class Halves(unittest.TestCase):
    def test_halve_split_and_close(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
        ip, fl = np.array([0, 4, 7]), np.array([0, 1, 2, 3, 4, 5, 6])  # a quad, a zero-area triangle
        self.assertEqual(uvkeep.halve_mask(P, ip, fl, [0.0, 0.0]).tolist(), [True, False])
        self.assertEqual(uvkeep.halve_mask(P, ip, fl, [1.0, 0.0]).tolist(), [False, False])
        nip, nfl, org = uvkeep.fan_split(ip, fl, np.array([7, 8]), np.array([True, False]))
        self.assertEqual((nip.tolist(), nfl.tolist(), org.tolist()), ([0, 3, 6, 9], [0, 1, 2, 0, 2, 3, 4, 5, 6], [7, 7, 8]))
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.1, 0.1], [0.2, 0.3], [0.4, 0.2]], float)
        uv[1] = [0.5, 0.5 + 1e-12]  # first fan triangle of the quad: next to no UV area
        out = uvkeep.close_flat(P, ip, fl, uv)
        np.testing.assert_array_equal(out[4:], np.repeat(uv[4:].mean(0)[None], 3, 0))  # no area: one UV
        self.assertTrue(np.array_equal(out[1], out[0]) or np.array_equal(out[1], out[2]))
        np.testing.assert_array_equal(out[[0, 2, 3]], uv[[0, 2, 3]])
        self.assertEqual(uvkeep.still_folded(ip, out, [1.0, 1.0]).tolist(), [True, False])


class Reconcile(unittest.TestCase):
    def test_pair_joins_when_it_fits(self):
        seams, rep = fake_seams([uvseams.HARD, uvseams.SOFT]), {}
        uvkeep._reconcile(seams, 1.0, 0.0, 10.0, rep)
        self.assertEqual(list(seams["class"]), [uvseams.HARD, uvseams.HARD])
        self.assertEqual((rep["hard_conflicts_kept"], rep["hard_conflicts_soft"]), (1, 0))

    def test_hard_goes_soft_when_it_does_not_fit_and_self_seam(self):
        seams, rep = fake_seams([uvseams.HARD, uvseams.SOFT]), {}
        uvkeep._reconcile(seams, 1.0, 0.0, 0.0, rep)
        self.assertEqual(list(seams["class"]), [uvseams.SOFT, uvseams.SOFT])
        self.assertEqual((rep["hard_conflicts_kept"], rep["hard_conflicts_soft"]), (0, 1))
        seams, rep = fake_seams([uvseams.HARD], 1, [(0, 0)]), {}
        uvkeep._reconcile(seams, 1.0, 0.0, 10.0, rep)
        self.assertEqual(list(seams["class"]), [uvseams.SOFT])

    def test_kinds_split_only_at_hard_chains(self):
        # islands 0 | 1 | 2 in a row (Hard, Hard) and a loose island 3: 0 and 2 share a kind, so does 3 (no border to it)
        seams = fake_seams([uvseams.HARD, uvseams.HARD], 4, [(0, 1), (1, 2)])
        k = uvkeep._kinds(seams)
        self.assertNotEqual(k[0], k[1])
        self.assertNotEqual(k[1], k[2])
        self.assertEqual(k[0], k[2])
        self.assertEqual(k[3], k[0])

    def test_too_many_kinds_reports_instead_of_dropping(self):
        sa = two_island_source()
        nf = len(sa.indptr) - 1
        seams = fake_seams([uvseams.HARD, uvseams.HARD, uvseams.HARD], 3, [(0, 1), (1, 2), (0, 2)])  # three kinds: each meets both others
        seams["island"] = np.zeros(nf, np.int64)
        seams["island"][nf // 3:2 * nf // 3], seams["island"][2 * nf // 3:] = 1, 2
        real, old = uvkeep.chart_codes, uvkeep._triaged
        uvkeep.chart_codes = lambda m, k: None if len(np.unique(k)) > 2 else real(m, k)
        uvkeep._triaged = lambda *a: (seams, {"hard_islands": [], "hard_cost": [], "left_islands": [], "left_cost": [], "spent": 0.0, "budget": 0.0, "edge_est": 1.0})
        try:
            plan = uvkeep.make_plan(sa, 10.0, uvkeep.Settings(mode="light"))
        finally:
            uvkeep.chart_codes, uvkeep._triaged = real, old
        self.assertIsNotNone(plan.codes)
        self.assertTrue(any("Too many seams" in m for m in plan.messages), plan.messages)


class AfterApply(unittest.TestCase):
    """uv_hooks.kinds and after_apply on stand-in objects: the hook path and the fallback path call the same function."""

    def setUp(self):
        from uv_hooks_fake import FakeMesh

        self.sa = two_island_source()
        n = len(self.sa.points)
        self.source = type("Obj", (), {})()
        self.source.data = FakeMesh()
        faces = [tuple(self.sa.flat[self.sa.indptr[i]:self.sa.indptr[i + 1]]) for i in range(len(self.sa.indptr) - 1)]
        self.source.data.from_pydata(self.sa.points, [], faces)
        self.source.data.uv_layers.new("UVMap")
        self.source.data.uv_layers[0].data.foreach_set("uv", self.sa.layers[0][1].ravel())
        flatfaces = common.addon_module("triadq.types").FlatFaces
        self.src = {"co": self.sa.points, "faces": flatfaces(self.sa.indptr, self.sa.flat), "loops": flatfaces(self.sa.indptr, np.arange(len(self.sa.flat))),
                    "mat": self.sa.material, "sig": ("fake", n)}
        self.hooks = common.addon_module("uv_hooks")
        self.hooks.SOURCE_DATA = lambda obj: self.src
        self.keep = common.addon_module("triadq.uvkeep")
        self.keep.OVERRIDE.clear()
        self.keep.OVERRIDE.update(mode="light", allowance=1.0)
        self.gs = type("GS", (), {"keep_uvs": False, "bake_detail": "OFF"})()

    def tearDown(self):
        self.keep.OVERRIDE.clear()
        self.keep.CANDIDATES.clear()
        self.hooks.TRANSFERS.clear()
        self.hooks.SOURCE_DATA = None

    def result(self):
        p, ip, fl = grid(5)
        return {"co": p, "faces": [tuple(fl[ip[i]:ip[i + 1]]) for i in range(len(ip) - 1)], "mat": np.zeros(len(ip) - 1, np.int32)}

    def test_kinds_modes(self):
        sentinel = ("old",)
        self.keep.OVERRIDE["mode"] = "exact"
        self.assertEqual(self.hooks.kinds(lambda s, r: sentinel, self.source, self.src, self.gs), sentinel)
        self.keep.OVERRIDE["mode"] = "off"
        self.assertIsNone(self.hooks.kinds(lambda s, r: sentinel, self.source, self.src, self.gs))
        self.keep.OVERRIDE["mode"] = "light"
        codes = self.hooks.kinds(lambda s, r: sentinel, self.source, self.src, self.gs)
        self.assertEqual(len(codes[0]), len(self.sa.indptr) - 1)

    def test_hook_and_fallback_write_the_same_hash(self):
        from uv_hooks_fake import FakeMesh

        self.hooks.kinds(None, self.source, self.src, self.gs)  # the search's call: makes the plan
        res = self.result()
        stage = self.keep.fit_stage(self.source, self.src, res, self.gs)
        fit = self.keep.run(stage)
        hook_mesh, late_mesh = FakeMesh(), FakeMesh()
        common.addon_module("sliced").finish(self.hooks.after_apply_steps(hook_mesh, self.source, self.result(), self.gs))  # hook path: right after the mesh is made
        self.assertTrue(self.hooks.LAST["fitted"])
        self.assertEqual(self.hooks.LAST["hash"], fit.key)
        hook_hash = self.hooks.LAST["hash"]
        common.addon_module("sliced").finish(self.hooks.after_apply_steps(late_mesh, self.source, self.result(), self.gs))  # fallback path: the Apply operator, same function
        self.assertEqual(self.hooks.LAST["hash"], hook_hash)

    def test_an_unkept_seam_fit_preview_is_not_written(self):
        """Seam Fit opened on the result, then cancelled (or its Apply undone): the result's Apply writes the plain
        transfer, never the fit the user did not keep."""
        from uv_hooks_fake import FakeMesh

        UT, T = common.addon_module("triadq.uvtrial"), common.addon_module("triadq.trial")
        self.hooks.kinds(None, self.source, self.src, self.gs)
        res = self.result()
        sa, plan = self.keep.plan_for(self.source, self.src, self.gs)
        trial = UT.SeamFit(dict(source=sa, plan=plan, settings=self.keep.settings_of(self.gs), material=res["mat"]))
        try:
            cand = T.finish(trial.run(T.make_base(res["co"], res["faces"]), dict(trial.defaults, faces=1.0)))
            self.assertFalse(cand.refusal, cand.refusal)
            common.addon_module("sliced").finish(self.hooks.after_apply_steps(FakeMesh(), self.source, self.result(), self.gs))
            self.assertFalse(self.hooks.LAST["fitted"])
        finally:
            UT.APPLIED_EXTRA.clear()

    def test_a_kept_transfer_is_not_reused_after_your_uvs_change(self):
        from uv_hooks_fake import FakeMesh

        self.hooks.kinds(None, self.source, self.src, self.gs)
        common.addon_module("sliced").finish(self.hooks.after_apply_steps(FakeMesh(), self.source, self.result(), self.gs))
        self.source.data.uv_layers[0].data.foreach_set("uv", (self.sa.layers[0][1] * 0.5).ravel())  # you unwrapped your mesh again
        self.assertGreater(len(list(self.hooks.after_apply_steps(FakeMesh(), self.source, self.result(), self.gs))), 0)

    def test_without_a_fit_only_the_transfer_is_written(self):
        from uv_hooks_fake import FakeMesh

        self.hooks.kinds(None, self.source, self.src, self.gs)
        res = self.result()
        before = len(res["faces"])
        mesh = FakeMesh()
        common.addon_module("sliced").finish(self.hooks.after_apply_steps(mesh, self.source, res, self.gs))
        self.assertFalse(self.hooks.LAST["fitted"])
        self.assertEqual(len(res["faces"]), before)
        self.assertEqual(len(mesh.polygons), before)
        self.assertTrue(any("not run" in m for m in self.hooks.LAST["messages"]))


    def test_sliced_apply_writes_at_the_end_and_keeps_the_transfer(self):
        from uv_hooks_fake import FakeMesh

        self.hooks.kinds(None, self.source, self.src, self.gs)
        mesh, work = FakeMesh(), None
        work = self.hooks.after_apply_steps(mesh, self.source, self.result(), self.gs)
        steps = 0
        for _ in work:
            steps += 1
            self.assertEqual(len(mesh.polygons), 0)  # nothing written while the transfer runs
        self.assertGreater(steps, 1)
        sliced = self.hooks.LAST["hash"]
        self.hooks.TRANSFERS.clear()
        common.addon_module("sliced").finish(self.hooks.after_apply_steps(FakeMesh(), self.source, self.result(), self.gs))
        self.assertEqual(self.hooks.LAST["hash"], sliced)
        again = list(self.hooks.after_apply_steps(FakeMesh(), self.source, self.result(), self.gs))
        self.assertEqual(again, [])  # the same result again: the kept transfer is written at once
        self.assertEqual(self.hooks.LAST["hash"], sliced)
        self.hooks.TRANSFERS.clear()
        self.assertGreater(len(list(self.hooks.prepare_steps(self.source, self.result(), self.gs))), 1)  # made ahead
        self.assertEqual(list(self.hooks.after_apply_steps(FakeMesh(), self.source, self.result(), self.gs)), [])
        self.assertEqual(self.hooks.LAST["hash"], sliced)


class ViewLimit(unittest.TestCase):
    def test_camera_check_holds_moves_past_most_detail_loss(self):
        p, ip, fl = grid(4)
        moved = p.copy()
        moved[12, 2] = 0.3  # the middle vertex lifted off the surface: its faces get wrong pixels
        moved[4, 1] += 0.01  # a far corner slid along the surface: no wrong pixels

        class Judge:
            total = 1000

            def wrong(self, corners):
                return 100.0 * sum(abs(c[2]) for c in corners)

        out = {}
        for _ in uvkeep.view_limit(Judge(), p, moved, ip, fl, np.full(len(p), 0.25), 0.0, out):
            pass
        self.assertEqual((out["limit"][12], out["limit"][4], out["held"]), (0.0, 0.25, 1))
        for _ in uvkeep.view_limit(Judge(), p, moved, ip, fl, np.full(len(p), 0.25), 50.0, out):
            pass
        self.assertEqual(out["held"], 0)  # 120 wrong pixels fit in half of the 5 touched faces' 312 pixels



class BruteTree:
    """Closest points on a few polygons (their fan triangles), the face tree's form."""

    def __init__(self, points, faces):
        from triadq.uvseams import _closest_bary_many

        P = np.asarray(points, float).reshape(-1, 3)
        ip, fl = np.asarray(faces.indptr), np.asarray(faces.flat)
        tri, owner = [], []
        for f in range(len(ip) - 1):
            for k in range(ip[f] + 1, ip[f + 1] - 1):
                tri.append((fl[ip[f]], fl[k], fl[k + 1]))
                owner.append(f)
        self.a, self.b, self.c = (P[np.array(tri)[:, i]] for i in range(3))
        self.owner, self.bary = np.array(owner), _closest_bary_many

    def nearest(self, q):
        at, g, d = [], [], []
        for x in np.asarray(q, float).reshape(-1, 3):
            w = self.bary(x, self.a, self.b, self.c)
            pts = w[:, :1] * self.a + w[:, 1:2] * self.b + w[:, 2:] * self.c
            i = int(np.argmin(((pts - x) ** 2).sum(1)))
            at.append(pts[i])
            g.append(self.owner[i])
            d.append(float(np.sqrt(((pts[i] - x) ** 2).sum())))
        return np.array(at), np.array(g), np.array(d)


class BallSnap(unittest.TestCase):
    def test_corners_move_onto_the_side_the_others_hold(self):
        from triadq import uvsnap
        from triadq.types import FlatFaces

        sa = two_island_source(16)  # sides meet at x = 0.5
        region = (sa.points[sa.flat].reshape(-1, 4, 3)[:, :, 0].mean(1) > 0.5).astype(np.int64)
        trees = uvsnap.RegionTrees(sa.points, sa.indptr, sa.flat, region, lambda a, b: BruteTree(a, b))
        p, ip, fl = grid(4)  # vertices at x = 0.25, 0.5, 0.75 ...
        p[p[:, 0] == 0.5, 0] = 0.56  # the middle column just over the line: the faces left of it straddle
        whole = BruteTree(sa.points, FlatFaces(sa.indptr, sa.flat))
        _at, g, d0 = whole.nearest(p)
        label = region[g]
        out = {}
        for _ in uvsnap.ball_snap_steps(p, ip, fl, None, trees, label, d0, np.full(len(p), 0.1), out):
            pass
        q = out["points"].reshape(-1, 3)
        self.assertEqual(len(out["moved"]), 5)
        self.assertTrue(np.allclose(q[out["moved"], 0], 0.5))  # onto the line where the sides meet
        out = {}
        for _ in uvsnap.ball_snap_steps(p, ip, fl, None, trees, label, d0, np.full(len(p), 0.05), out):
            pass
        self.assertEqual(len(out["moved"]), 0)  # the move limit holds

    def test_a_slide_never_closes_an_edge(self):
        from triadq import uvsnap

        p, ip, fl = grid(2)
        mesh = uvsnap._Mesh(p, ip, fl)
        pos = [tuple(x) for x in p.tolist()]
        self.assertTrue(uvsnap._keeps_edges(pos, mesh, 4, (0.6, 0.5, 0.0)))
        self.assertFalse(uvsnap._keeps_edges(pos, mesh, 4, (0.995, 0.5, 0.0)))  # onto its neighbour at x = 1


class Specks(unittest.TestCase):
    def test_streak_leaves_no_fan_triangle_a_winding(self):
        quad = np.array([[0.0, 0.0], [0.01, 0.0001], [0.0101, 0.0012], [0.0, 0.001]])
        got = uvkeep.streak(quad)
        for k in (1, 2):
            a, b = got[k] - got[0], got[k + 1] - got[0]
            self.assertEqual(a[0] * b[1] - a[1] * b[0], 0.0)
            f = np.asarray(got, np.float32)  # and as the mesh stores it
            a, b = f[k] - f[0], f[k + 1] - f[0]
            self.assertEqual(float(a[0] * b[1] - a[1] * b[0]), 0.0)
        self.assertLess(np.max(np.linalg.norm(got - quad, axis=1)), 0.0007)  # the short edges shrink, the long way stays
        tri = uvkeep.streak(quad[:3])
        self.assertEqual(uvkeep.uv_area_px(tri), 0.0)

    def test_texel_area_counts_what_a_face_shows(self):
        p, ip, fl = grid(4)  # 16 faces of 1/16 each, the plane laid out on a quarter of a 1024 page side: 64 x 64 texels each
        uv = p[fl][:, :2] * 0.25
        shows, dens = uvkeep.texel_area(p, ip, fl, uv, 1024.0)
        self.assertAlmostEqual(dens, 256.0)
        self.assertTrue(np.allclose(shows, 64.0 * 64.0))


if __name__ == "__main__":
    unittest.main()
