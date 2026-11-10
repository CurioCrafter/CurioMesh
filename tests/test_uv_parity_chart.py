# SPDX-License-Identifier: GPL-3.0-or-later
"""Row Charts (K6): the NumPy twin on its own, then twin against the compiled kernel (parity) and the job wrappers."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from triadq import uvchart

from uv_parity_common import turbo  # noqa: E402  (None in a build without it)


def grid(nx, ny, h=1.0, z=lambda x, y: 0.0 * x):
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    x, y = i.ravel() * h, j.ravel() * h
    points = np.stack([x, y, z(x, y)], 1)
    ids = (j * (nx + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + nx + 2, ids + nx + 1], 1).ravel()
    return points, np.arange(0, len(flat) + 1, 4), flat


def cube(k):
    ids, pts, flat = {}, [], []
    for ax, sg in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        if sg < 0:
            u, v = v, u

        def vid(i, j):
            c = [0, 0, 0]
            c[ax], c[u], c[v] = (k if sg > 0 else 0), i, j
            c = tuple(c)
            if c not in ids:
                ids[c] = len(pts)
                pts.append([x / k for x in c])
            return ids[c]

        for j in range(k):
            for i in range(k):
                flat += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
    return np.array(pts, float), np.arange(0, len(flat) + 1, 4), np.array(flat)


def tube(around, rows, r=1.0):
    a = 2 * np.pi * np.arange(around) / around
    pts = np.array([[r * np.cos(t), r * np.sin(t), j * 0.3] for j in range(rows + 1) for t in a])
    flat = []
    for j in range(rows):
        for i in range(around):
            flat += [j * around + i, j * around + (i + 1) % around, (j + 1) * around + (i + 1) % around, (j + 1) * around + i]
    return pts, np.arange(0, len(flat) + 1, 4), np.array(flat)


def speckled(n, every, amp):
    """A wavy n x n grid with every `every`-th quad split in two triangles: many irregular points, ragged patches."""

    points, indptr, flat = grid(n, n, 1.0 / n, lambda x, y: amp * np.sin(5 * x) * np.cos(4 * y))
    faces = []
    for k, q in enumerate(flat.reshape(-1, 4)):
        faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if k % every == 0 else [list(q)]
    return points, np.concatenate([[0], np.cumsum([len(f) for f in faces])]), np.array([v for f in faces for v in f])


def l_parts():
    """A flat 10x8 grid with an L-shaped part (faces i < 4, j < 5) and a second block (i >= 7, j >= 4): grid unions only."""

    points, indptr, flat = grid(10, 8)
    f = np.arange(80)
    i, j = f % 10, f // 10
    label = np.where((i < 4) & (j < 5), 1, np.where((i >= 7) & (j >= 4), 2, 0))
    return points, indptr, flat, label


class Twin(unittest.TestCase):
    def test_cube_six_patches(self):
        tree = uvchart.build(*cube(4), use_fast=False)
        n = dict(zip(uvchart.NUMBER_COLUMNS, tree.numbers()))
        self.assertEqual(n["patches"], 6)
        self.assertEqual(n["irregular"], 8)
        self.assertEqual(n["segments"], 12)
        for x in (0.0, 10.0, 100.0):
            self.assertEqual(tree.level(x)[2][1], 0, f"not-disk charts at level {x}")

    def test_cylinder_one_cut(self):
        tree = uvchart.build(*tube(16, 5), use_fast=False)
        n = dict(zip(uvchart.NUMBER_COLUMNS, tree.numbers()))
        self.assertEqual((n["patches"], n["euler_cut_patches"], n["euler_cut_edges"], n["not_disk_patches"]), (1, 1, 5, 0))

    def test_forced_line_turning_vertex_starts_traces(self):
        points, indptr, flat = grid(8, 8)
        vid = lambda i, j: j * 9 + i  # noqa: E731
        forced = [[vid(i, 2), vid(i + 1, 2)] for i in range(2, 5)] + [[vid(5, j), vid(5, j + 1)] for j in range(2, 6)]
        tree = uvchart.build(points, indptr, flat, forced=np.array(forced).ravel(), use_fast=False)
        self.assertTrue(tree.turning[vid(5, 2)])
        self.assertTrue(tree.irregular[vid(2, 2)] and tree.irregular[vid(5, 6)])
        t = tree.topo
        sep = {(t.ea[e], t.eb[e]) for e in range(t.ne) if tree.sep[e]}
        self.assertIn((vid(5, 2), vid(6, 2)), sep)
        self.assertIn((vid(5, 1), vid(5, 2)), sep)

    def test_level_skips_over_threshold_merges(self):
        points, indptr, flat, label = l_parts()
        tree = uvchart.build(points, indptr, flat, part_label=label, use_fast=False)
        self.assertGreaterEqual(len(tree.merges), 2)
        # Make the first recorded merge expensive: the later ones still apply at a level below it.
        first = tree.merges[0]
        tree.merges[0] = (first[0], 1.5, first[2], first[3])
        tree._cache.clear()
        used = tree.level(10.0)[2][2]
        self.assertEqual(used, len(tree.merges) - 1)

    def test_border_crossing_found_and_touching_ignored(self):
        p, q = np.array([[0.0, 0.0], [1.0, -1.0]]), np.array([[2.0, 0.0], [1.0, 1.0]])
        self.assertTrue(uvchart._sides_cross(p, q, np.array([[0, 1], [2, 3]])))
        q2 = np.array([[2.0, 0.0], [2.0, 1.0]])
        self.assertFalse(uvchart._sides_cross(np.array([[0.0, 0.0], [2.0, 0.0]]), q2, np.array([[0, 1], [1, 3]])))
        self.assertEqual(uvchart.round_caps(1.0)[0], 0.0125)
        self.assertEqual(uvchart.round_caps(1.0)[-1], 1.0)

    def test_folds_weigh_less_and_compact_unions_rank_first(self):
        # R4: a cut along a 90 deg fold weighs (1 - FOLD_DISCOUNT) of its length; the queue takes the most compact union first.
        tree = uvchart.build(*cube(3), use_fast=False, merge=False)
        for sg in tree.segments:
            self.assertAlmostEqual(sg.weight, (1.0 - uvchart.FOLD_DISCOUNT) * sg.length, places=9)
        square, strip = uvchart._irregularity(4.0, 1.0), uvchart._irregularity(26.0, 12.0)  # a 12:1 strip
        self.assertAlmostEqual(square, 4.0 / np.pi)
        self.assertLess(uvchart._key(0.1, 1.0, square), uvchart._key(0.0, 5.0, strip))
        self.assertGreater(strip, uvchart.SHAPE_MAX)

    def test_thin_tube_becomes_strip_pieces(self):
        # R5: a long thin open tube (40 rows, 6 around) is cut into two pieces, each cut once along its length.
        mesh = tube(6, 40, 0.2)
        tree = uvchart.build(*mesh, use_fast=False)
        n = dict(zip(uvchart.NUMBER_COLUMNS, tree.numbers()))
        self.assertEqual((n["tube_faces"], n["patches"], n["not_disk_patches"]), (240, 2, 0))
        self.assertEqual(tree.level(120.0)[2][:2].tolist(), [2, 0])
        # boxes: a square's smallest box is its side; a thin diagonal strip's is the strip
        self.assertEqual(uvchart._min_box([[0, 0], [1, 0], [1, 1], [0, 1]]), (1.0, 1.0))
        w, h = uvchart._min_box([[0, 0], [10, 10], [10.1, 9.9], [0.1, -0.1]])
        self.assertAlmostEqual(min(w, h), np.hypot(0.1, 0.1), places=6)

    def test_slices_stay_short(self):
        warm = uvchart.Charts(*cube(4), use_fast=False)  # the first run pays one-time start-up (imports, NumPy) in one slice
        while not warm.done:
            warm.step()
        stage = uvchart.Charts(*cube(4), use_fast=False)
        longest = 0.0
        while not stage.done:
            t = time.perf_counter()
            stage.step()
            longest = max(longest, time.perf_counter() - t)
        self.assertLess(longest, 0.010, f"a slice ran {longest * 1e3:.1f} ms")


@needs_native
class Parity(unittest.TestCase):
    def run_job(self, job):
        self.assertIsNotNone(job)
        job._thread.join(120)
        return job.result()

    def kernel(self, points, indptr, flat, **kw):
        return self.run_job(turbo.uvchart_build(points, indptr, flat, **kw))

    def same_topology(self, mesh, **kw):
        twin = uvchart.build(*mesh, use_fast=False, merge=False, **kw)
        kern = self.kernel(*mesh, **kw)
        a, b, sep, fixed, cost = kern.edges()
        t = twin.topo
        np.testing.assert_array_equal(a, t.ea)
        np.testing.assert_array_equal(b, t.eb)
        np.testing.assert_array_equal(sep, twin.sep)
        np.testing.assert_array_equal(fixed, twin.fixed)
        np.testing.assert_allclose(cost, twin.cost, rtol=1e-12)
        np.testing.assert_array_equal(kern.patches(), twin.patch)
        kirr = kern.irregular()
        np.testing.assert_array_equal(kirr > 0, twin.irregular)
        np.testing.assert_array_equal(kirr == 2, twin.turning)
        p, q, scost, seen, ptr, ids = kern.segments()
        self.assertEqual(len(p), len(twin.segments))
        for k, sg in enumerate(twin.segments):
            self.assertEqual((p[k], q[k]), (sg.p, sg.q))
            self.assertEqual(sorted(ids[ptr[k]:ptr[k + 1]]), sorted(sg.edges))
            self.assertAlmostEqual(scost[k], sg.cost, places=6)
        kn, tn = kern.numbers(), twin.numbers()
        for col in (0, 1, 2, 3, 4, 5, 6, 7, 18):
            self.assertEqual(kn[col], tn[col], uvchart.NUMBER_COLUMNS[col])
        return twin, kern

    def test_topology_matches_on_test_meshes(self):
        twin, kern = self.same_topology(tube(6, 40, 0.2))  # R5: a thin tube cut into strip pieces
        self.assertEqual(kern.numbers()[21], 240)
        self.same_topology(cube(4))
        self.same_topology(tube(16, 5))
        points, indptr, flat, label = l_parts()
        self.same_topology((points, indptr, flat), part_label=label)

    def test_topology_matches_with_seen_pixels_and_hidden_cuts(self):
        mesh = cube(3)
        twin = uvchart.build(*mesh, use_fast=False, merge=False)
        seen = np.linspace(0.5, 3.0, twin.topo.ne)
        self.same_topology(mesh, seen_edge_px=seen, out_of_sight=True, px_per_unit=100.0)

    def test_merge_tree_and_levels_match_on_grid_unions(self):
        points, indptr, flat, label = l_parts()
        twin = uvchart.build(points, indptr, flat, part_label=label, use_fast=False)
        kern = self.kernel(points, indptr, flat, part_label=label)
        stretch, cost, pole, ptr, ids = kern.merges()
        self.assertEqual(len(stretch), len(twin.merges))
        self.assertGreater(len(stretch), 2)
        for k, (segs, s, c, p) in enumerate(twin.merges):
            self.assertEqual(list(ids[ptr[k]:ptr[k + 1]]), list(segs))
            self.assertAlmostEqual(stretch[k], s, places=6)
            self.assertEqual(bool(pole[k]), p)
        for x in (0.0, 10.0):
            kc, kcut, kn = kern.level(x)
            tc, tcut, tn = twin.level(x)
            np.testing.assert_array_equal(kc, tc)
            np.testing.assert_array_equal(kcut, tcut)
            np.testing.assert_array_equal(kn[:5], tn[:5])

    def test_regions_of_a_large_piece_match(self):
        """R10big: one piece of more than REGION_MIN_FACES faces (the L-parts tile 14 x 16 times) runs its merge rounds
        region by region, then across the region borders: the kernel (regions on worker threads) and the twin make the
        same merges in the same order."""

        tiles_x, tiles_y = 14, 16
        points, indptr, flat = grid(10 * tiles_x, 8 * tiles_y)
        f = np.arange(len(indptr) - 1)
        i, j = f % (10 * tiles_x) % 10, f // (10 * tiles_x) % 8
        label = np.where((i < 4) & (j < 5), 1, np.where((i >= 7) & (j >= 4), 2, 0)) + 3 * (f % (10 * tiles_x) // 10 + tiles_x * (f // (10 * tiles_x) // 8))
        self.assertGreater(len(f), uvchart.REGION_MIN_FACES)
        twin = uvchart.build(points, indptr, flat, part_label=label, use_fast=False)
        members = [[] for _ in range(twin.n_patches)]
        for face, p in enumerate(twin.patch):
            members[p].append(face)
        self.assertEqual(len(set(uvchart._regions(twin.topo, twin.points.tolist(), members))), 4)
        kern = self.kernel(points, indptr, flat, part_label=label)
        stretch, _, pole, ptr, ids = kern.merges()
        self.assertEqual(len(stretch), len(twin.merges))
        self.assertGreater(len(stretch), 100)
        for k, (segs, s, _, p) in enumerate(twin.merges):
            self.assertEqual(list(ids[ptr[k]:ptr[k + 1]]), list(segs))
            self.assertAlmostEqual(stretch[k], s, places=6)
            self.assertEqual(bool(pole[k]), p)
        np.testing.assert_array_equal(kern.level(10.0)[0], twin.level(10.0)[0])

    def test_levels_nest_and_stay_disks_with_smoother_outlines(self):
        """R6: the outline smoothing moves patches (fewer cut outline weight) and, with the last merge round, keeps every
        level's charts disks, each inside one chart of every level above it (the redo of a chart that lies over itself
        takes the charts of a lower level inside it), in the kernel and in the twin."""

        mesh = speckled(16, 7, 0.4)
        twin = uvchart.build(*mesh, use_fast=False, build_stretch=0.05)
        moves, _ = twin._smoothing()
        self.assertGreater(len(moves), 0)
        root, glued, _, _, _ = twin._replay(float("inf"))
        before = sum(w for e, w in enumerate(twin.outline) if twin.cut_base[e] and not glued[e])
        _, cut, _ = twin.level(1000.0)
        self.assertLess(sum(w for e, w in enumerate(twin.outline) if cut[e]), before)
        for tree in (self.kernel(*mesh, build_stretch=0.05), twin):
            levels = [tree.level(x) for x in (0.0, 1.0, 2.5, 5.0, 1000.0)]
            for (lo, _, nlo), (hi, _, nhi) in zip(levels, levels[1:]):
                self.assertEqual((nlo[1], nhi[1]), (0, 0))
                for c in np.unique(lo):
                    self.assertEqual(len(np.unique(hi[lo == c])), 1)

    def test_stage_uses_the_fast_callable(self):
        uv_fast = common.addon_module("uv_fast")
        fast_chart = common.addon_module("triadq.uvchart")
        self.assertTrue(uv_fast.install())
        try:
            stage = fast_chart.Charts(*cube(4))
            while not stage.done:
                stage.step()
            chart, cut, numbers = stage.tree.level(10.0)
            self.assertEqual(numbers[1], 0)
            self.assertEqual(stage.tree.numbers()[0], 6)
        finally:
            uv_fast.uninstall()

    def test_wrappers_never_block_and_new_uvs_are_flip_free(self):
        points, indptr, flat = grid(150, 150, 0.01, lambda x, y: 0.2 * np.sin(3 * x) * np.cos(2 * y))
        t = time.perf_counter()
        job = turbo.uvchart_build(points, indptr, flat)
        took = time.perf_counter() - t
        self.assertLess(took, 0.010, f"wrapper blocked {took * 1e3:.1f} ms")
        tree = self.run_job(job)
        t = time.perf_counter()
        job = turbo.uvchart_uvs(tree, 10.0, rows=0)
        self.assertLess(time.perf_counter() - t, 0.010)
        uv, chart, stats, ratio = self.run_job(job)
        stats = stats.reshape(-1, turbo.native().UVFLAT_STATS_LEN)
        self.assertEqual(int(np.nansum(stats[:, 7])), 0)
        self.assertTrue(np.isfinite(uv).all())

    def test_keep_going_accepts_only_score_drops(self):
        points, indptr, flat = grid(12, 12, 0.2, lambda x, y: 0.5 * np.sin(2 * x) * np.cos(1.5 * y))
        tree = self.run_job(turbo.uvchart_build(points, indptr, flat))
        cuts = self.run_job(turbo.uvchart_cuts(tree, 5.0, rows=0, stretch_allowed=0.01))
        accepted = 0
        for _ in range(8):
            before = cuts.state()
            kind, ok, s0, s1, ms = self.run_job(turbo.uvchart_pass(tree, cuts))
            after = cuts.state()
            if ok:
                accepted += 1
                self.assertLess(s1, s0)
                self.assertAlmostEqual(after[4], s1)
            else:
                np.testing.assert_array_equal(after[2], before[2])
                self.assertEqual(after[4], before[4])
                break


if __name__ == "__main__":
    unittest.main()
