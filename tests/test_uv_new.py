# SPDX-License-Identifier: GPL-3.0-or-later
"""New UVs (A2): the pipeline on small meshes (NumPy path, then with the compiled kernels when built), the gap rule,
the seam step check on a synthetic cut, and the slice length."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from triadq import uvbake, uvchart, uvflat, uvnew, uvpack, uvscore, uvsight
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


if __name__ == "__main__":
    unittest.main()
