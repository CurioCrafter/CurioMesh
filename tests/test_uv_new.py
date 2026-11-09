# SPDX-License-Identifier: GPL-3.0-or-later
"""New UVs (A2): the pipeline on small meshes (NumPy path, then with the compiled kernels when built), the gap rule,
the seam step check on a synthetic cut, and the slice length."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from triadq import uvbake, uvchart, uvflat, uvnew, uvpack, uvscore
from test_uv_parity_chart import cube, grid


def run(mesh, **kw):
    settings = uvnew.Settings(out_of_sight=False, hidden_pct=100.0, texture_px=256, time_per_pass=1.0, **kw)  # no pictures: no Blender here
    stage = uvnew.NewUVs(*mesh, settings=settings)
    longest = 0.0
    while not stage.done:
        t = time.perf_counter()
        stage.step()
        longest = max(longest, time.perf_counter() - t)
    if stage.error:
        raise stage.error
    return stage, stage.result, longest


class Rules(unittest.TestCase):
    def test_gap_is_the_spec_rule(self):
        self.assertEqual((uvnew.gap_texels(2048), uvnew.gap_texels(1024), uvnew.gap_texels(256)), (16, 8, 2))
        self.assertEqual(uvbake.gap_px(2048), 16)  # the bake margin is the gap

    def test_calipers_find_the_box(self):
        p = np.array([[0, 0], [4, 1], [3, 3], [-1, 2.0]])
        a = uvnew.min_area_angle(p @ np.array([[np.cos(0.5), np.sin(0.5)], [-np.sin(0.5), np.cos(0.5)]]))
        q = p @ np.array([[np.cos(0.5), np.sin(0.5)], [-np.sin(0.5), np.cos(0.5)]])
        r = q @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        box = (r.max(0) - r.min(0)).prod()
        self.assertLessEqual(box, (p.max(0) - p.min(0)).prod() + 1e-9)

    def test_hull_of_many_points_is_the_hull_of_all_of_them(self):
        rng = np.random.default_rng(3)
        p = rng.normal(size=(4000, 2)) * [3.0, 1.0]
        h = uvnew._hull(p)
        a, b = h, np.roll(h, -1, axis=0)
        side = (b[:, 0] - a[:, 0])[None] * (p[:, 1:2] - a[:, 1][None]) - (b[:, 1] - a[:, 1])[None] * (p[:, 0:1] - a[:, 0][None])
        self.assertTrue((side >= -1e-9).all())  # every point lies left of (or on) every hull edge: counter-clockwise and nothing outside
        self.assertTrue(all(any(np.allclose(v, q) for q in p[:: 1]) for v in h[:5]))

    def test_angles_of_all_charts_at_once_match_one_by_one(self):
        rng = np.random.default_rng(5)
        sizes = [1, 2, 3, 3, 7, 40, 600, 5, 100, 300]
        uv = np.concatenate([rng.normal(size=(n, 2)) * [2.0, 0.5] + rng.normal(size=2) for n in sizes])
        uv[sizes[0] + sizes[1] + 1] = uv[sizes[0] + sizes[1]]  # repeated points (corners of one vertex)
        label = np.repeat(np.arange(len(sizes)), sizes)
        order = np.argsort(label, kind="stable")
        bounds = np.searchsorted(label[order], np.arange(len(sizes) + 1))
        gen = uvnew.min_area_angles(uv, order, bounds, list(range(len(sizes))))
        try:
            while True:
                next(gen)
        except StopIteration as done:
            got = done.value
        want = [uvnew.min_area_angle(uv[label == c]) if (label == c).sum() > 2 else 0.0 for c in range(len(sizes))]
        np.testing.assert_array_equal(got, want)

    def test_cut_edges_mark_only_uv_splits(self):
        mesh = grid(2, 1)
        uv = mesh[0][mesh[2]][:, :2].copy()
        self.assertFalse(uvnew.cut_edges(mesh[1], mesh[2], uv).any())
        uv[4:] += 5.0  # the second face moves away: its shared edge is now a seam
        self.assertEqual(int(uvnew.cut_edges(mesh[1], mesh[2], uv).sum()), 1)

    def test_exact_overlap_finds_a_small_fold_and_not_shared_edges(self):
        g = grid(6, 6)
        uv = g[0][g[2]][:, :2].copy()
        t = uv[uvnew._fan(g[1], np.arange(36))]
        self.assertFalse(any(uvnew.overlap_pairs(t, 1e-9)))  # neighbours share edges and corners only
        t2 = np.concatenate([t, t[:1] + [0.3, 0.3]])  # one triangle laid over another by a fraction of a face
        self.assertTrue(any(uvnew.overlap_pairs(t2, 1e-9)))

    def test_all_charts_at_once_find_what_one_by_one_finds(self):
        g = grid(6, 6)
        uv = g[0][g[2]][:, :2].copy()
        t = uv[uvnew._fan(g[1], np.arange(36))]  # 72 triangles, one flat chart
        run_ = lambda tt, cc, n: uvnew._drive(uvnew.overlapping_charts(tt, cc, n))  # noqa: E731
        self.assertFalse(run_(t, np.zeros(len(t), np.int64), 1).any())
        bad = np.concatenate([t, t[:1] + [0.3, 0.3]])  # one triangle laid over another by a fraction of a face
        flags = run_(np.concatenate([bad, t + [50.0, 0.0]]), np.concatenate([np.zeros(len(bad), np.int64), np.ones(len(t), np.int64)]), 2)
        self.assertEqual(flags.tolist(), [True, False])  # only the chart that folds is named
        self.assertTrue(run_(np.concatenate([t[:1], t[1:2][:, ::-1]]), np.zeros(2, np.int64), 1)[0])  # a face wound against the rest
        self.assertTrue(run_(np.zeros((1, 3, 2)), np.zeros(1, np.int64), 1)[0])  # no area

    def test_edge_table_and_seams_match_the_sorted_reference(self):
        points, indptr, flat = grid(7, 5)
        nxt = np.arange(1, len(flat) + 1)
        nxt[indptr[1:] - 1] = indptr[:-1]
        keys = np.minimum(flat, flat[nxt]).astype(np.uint64) << np.uint64(32) | np.maximum(flat, flat[nxt]).astype(np.uint64)
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        rank = np.empty(len(first), np.int64)
        rank[np.argsort(first, kind="stable")] = np.arange(len(first))
        ids, ne = uvnew.edge_ids(indptr, flat)
        self.assertEqual((ids.tolist(), ne), (rank[inverse.reshape(-1)].tolist(), len(first)))
        old = uvnew.STRIPE_ENTRIES
        try:
            uvnew.STRIPE_ENTRIES = 16  # many stripes
            self.assertEqual(uvnew.edge_ids(indptr, flat)[0].tolist(), ids.tolist())
        finally:
            uvnew.STRIPE_ENTRIES = old

    def test_the_box_layout_is_shown_before_the_outline_packing_ends(self):
        mesh = cube(4)
        stage = uvnew.NewUVs(*mesh, settings=uvnew.Settings(out_of_sight=False, hidden_pct=100.0, texture_px=256, time_per_pass=1.0))
        seen = None
        while not stage.done:
            stage.step()
            if stage.first_ready and seen is None:
                seen = stage.result.uv.copy()
            if stage.waiting:
                time.sleep(0.0005)
        self.assertIsNotNone(seen)
        self.assertEqual(seen.shape, stage.result.uv.shape)
        self.assertLessEqual(stage.result.timing["first_s"], stage.result.timing["total_s"])

    def test_a_small_chart_joins_its_neighbour_without_overlap(self):
        points, indptr, flat = grid(20, 1)  # twenty faces in a row
        stage = uvnew.NewUVs(points, indptr, flat)
        uv = points[flat][:, :2].copy()
        uv[76:] = uv[76:] @ np.array([[0.0, -1.0], [1.0, 0.0]]) + [9.0, 9.0]  # the last face: its own chart, turned and moved
        chart = np.array([0] * 19 + [1])
        gen = stage._attach_small(chart, uv, np.array([1, 1]))
        try:
            while True:
                next(gen)
        except StopIteration as fin:
            chart2, uv2, rows2 = fin.value
        self.assertEqual(len(rows2), 1)
        self.assertFalse(uvnew.cut_edges(indptr, flat, uv2).any())  # glued back: no seam left

    def test_seam_step_reads_both_sides_of_a_cut(self):
        points, indptr, flat = grid(2, 1)
        uv = np.zeros((8, 2))
        uv[0:4] = [[0, 0], [0.25, 0], [0.25, 0.5], [0, 0.5]]
        uv[4:8] = [[0.75, 0], [1.0, 0], [1.0, 0.5], [0.75, 0.5]]
        normal = np.zeros((8, 8, 3))
        normal[:] = [0.5, 0.5, 1.0]  # flat
        normal[:, 4:] = [1.0, 0.5, 0.5]  # a tilted half on the right: 90 degrees away from flat
        out = uvbake.seam_step(points, indptr, flat, uv, normal)
        self.assertEqual(out["edges"], 1)
        self.assertGreater(out["normal_p95"], 60.0)


class Pipeline(unittest.TestCase):
    def check_cube(self, label):
        stage, r, longest = run(cube(4))
        self.assertGreaterEqual(r.charts, 1)
        self.assertLessEqual(r.charts, 6, label)
        self.assertTrue(np.isfinite(r.uv).all() and r.uv.min() >= -1e-9 and r.uv.max() <= 1 + 1e-9, label)
        n = uvscore.as_dict(stage.check(texture_px=256).numbers)
        self.assertEqual(n["flipped_faces"], 0, label)
        self.assertEqual(n["overlap_pct"], 0, label)
        self.assertLessEqual(n["stretch_worst1"], 1.6, label)
        self.assertGreaterEqual(n["space_used_pct"], 35.0, label)  # a cube is a 4-face belt and two caps: 37 % is its box limit
        self.assertEqual(n["straight_rows_pct"], 100.0, label)
        return longest

    def test_cube_numpy(self):
        uvchart.FAST.clear(), uvflat.FAST.clear(), uvpack.FAST.clear(), uvscore.FAST.clear()
        self.check_cube("numpy warm-up")  # the first run pays NumPy's start-up cost in one slice
        self.assertLess(self.check_cube("numpy"), 0.010)

    def test_plane_is_one_chart_with_straight_rows(self):
        stage, r, _ = run(grid(8, 8))
        self.assertEqual(r.charts, 1)
        self.assertTrue(r.on_chart.all())

    def test_again_keeps_the_tree_when_only_the_layout_changes(self):
        stage, r, _ = run(cube(4))
        second = stage.again(texture_px=512)
        self.assertIs(second.tree, None)  # set when the run reaches the charts stage
        second.run()
        self.assertIs(second.tree, stage.tree)
        self.assertEqual(second.result.charts, r.charts)

    def test_nothing_to_unwrap_is_a_message_not_a_crash(self):
        stage = uvnew.NewUVs(np.zeros((0, 3)), [0], [])
        while not stage.done:
            stage.step()
        self.assertIsInstance(stage.error, ValueError)


@needs_native
class WithKernels(unittest.TestCase):
    def test_cube_with_the_faster_callables(self):
        fast = common.addon_module("uv_fast")
        try:
            self.assertTrue(fast.install())
            Pipeline("check_cube").check_cube("kernels")
        finally:
            fast.uninstall()

    def test_the_same_mesh_gives_the_same_layout_and_keep_going_never_lowers_it(self):
        fast = common.addon_module("uv_fast")
        kn = common.addon_module("triadq.uvnew")  # the add-on's own copy of the modules, the one `fast` installs the kernels into
        kp = common.addon_module("triadq.uvpack")
        try:
            self.assertTrue(fast.install())
            self.assertIn("pack_into", kp.FAST)

            def make(**kw):
                return kn.NewUVs(*cube(4), settings=kn.Settings(out_of_sight=False, hidden_pct=100.0, texture_px=256, **kw))

            first = make()
            r1 = first.run()
            r2 = make(time_per_pass=0.0001).run()  # no clock anywhere in the layout: a tiny Time per pass changes nothing
            self.assertTrue(np.array_equal(r1.uv, r2.uv) and r1.scale == r2.scale)
            stage, passes = first, 0
            while not stage.exhausted:
                nxt = stage.keep_going()
                self.assertIsNotNone(nxt)
                while not nxt.done:
                    nxt.step()
                    if nxt.waiting:
                        time.sleep(0.0005)
                self.assertIsNone(nxt.error)
                self.assertGreaterEqual(nxt.result.scale, stage.result.scale)
                self.assertEqual(nxt.result.nest_effort, stage.settings.nest_effort + 1)
                stage, passes = nxt, passes + 1
                if not stage.gained:
                    break  # a pass that gains nothing ends it
            self.assertLessEqual(passes, len(kp.ORDER_MODES) - 1)
            self.assertIsNone(make(density=100.0).keep_going())  # not finished (and a fixed density packs boxes): nothing to continue
        finally:
            fast.uninstall()


if __name__ == "__main__":
    unittest.main()
