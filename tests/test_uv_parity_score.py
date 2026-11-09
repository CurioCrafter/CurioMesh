# SPDX-License-Identifier: GPL-3.0-or-later
"""UV Check (K3): the NumPy twin on its own, the verdict table, then twin against the compiled kernel (parity)."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_parity_common as common  # noqa: F401  (puts the add-on folder on the path)
from uv_parity_common import needs_native

from triadq import uvscore
from triadq.uvscore import mesh

from uv_parity_common import turbo  # noqa: E402  (None in a build without it)


def plane(n, split=False, z=0.0, size=8.0):
    """n x n quads on [0, size]^2; UV = xy scaled to 0.9, the right half shifted by 0.05 when `split` (two islands)."""

    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    h = size / n
    points = np.stack([i.ravel() * h, j.ravel() * h, np.full(i.size, z)], 1)
    ids = (j * (n + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + n + 2, ids + n + 1], 1).ravel()
    return mesh(points, np.arange(0, len(flat) + 1, 4), flat, plane_uv(points, flat, split, size))


def plane_uv(points, flat, split, size=8.0):
    """UV per corner from the position; `split` shifts the right half by 0.05 (a seam at x = size / 2)."""

    corner = points[flat]
    cell_x = corner.reshape(-1, 4, 3)[:, :, 0].mean(1)
    uv = np.stack([corner[:, 0] / size * 0.9, corner[:, 1] / size * 0.9], 1)
    if split:
        uv[np.repeat(cell_x > size / 2, 4), 0] += 0.05
    return uv


def numbers(check):
    return uvscore.as_dict(check.numbers)


class Twin(unittest.TestCase):
    def test_sliced_check_gives_the_same_numbers(self):
        res, src = plane(12, False, z=0.01), plane(16, True)
        want = uvscore.check(res, src, use_fast=False)
        saved = uvscore.QUERIES_PER_STEP, uvscore.TRIS_PER_STEP
        uvscore.QUERIES_PER_STEP, uvscore.TRIS_PER_STEP = 7, 5  # many slices
        try:
            gen, steps = uvscore.check_steps(res, src), 0
            while True:
                try:
                    next(gen)
                    steps += 1
                except StopIteration as done:
                    got = done.value
                    break
        finally:
            uvscore.QUERIES_PER_STEP, uvscore.TRIS_PER_STEP = saved
        self.assertGreater(steps, 20)
        np.testing.assert_array_equal(got.numbers, want.numbers)
        np.testing.assert_array_equal(got.straddle, want.straddle)
        np.testing.assert_array_equal(got.flipped, want.flipped)

    def test_source_against_itself_is_clean(self):
        m = plane(8, True)
        out = numbers(uvscore.check(m, m, use_fast=False))
        self.assertEqual(out["straddle_faces"], 0)
        self.assertLess(out["drift_p99"], 1e-6)
        self.assertEqual(out["flipped_faces"], 0)
        self.assertEqual(out["islands"], 2)

    def test_faces_across_a_seam_straddle(self):
        res = uvscore.check(plane(8, False), plane(8, True), use_fast=False)
        self.assertGreaterEqual(numbers(res)["straddle_faces"], 8)
        self.assertGreater(numbers(res)["drift_p99"], 10)
        self.assertEqual(int(res.straddle.sum()), numbers(res)["straddle_faces"])

    def test_offset_plane_keeps_drift_under_one_pixel(self):
        # The result is a finer plane 0.2 edges above the source, cut on the same seam, per-side UVs: no straddle and no drift.
        src = plane(8, True)
        n = 12
        res_plane = plane(n, True, z=0.2 * 8.0 / n)
        out = numbers(uvscore.check(res_plane, src, use_fast=False))
        self.assertEqual(out["straddle_faces"], 0)
        self.assertLess(out["drift_p99"], 1.0)

    def test_overlap_per_page_and_tile(self):
        m = plane(4)
        folded = mesh(m.points, m.indptr, m.flat, np.stack([0.45 - np.abs(m.uv[:, 0] - 0.45), m.uv[:, 1]], 1))
        self.assertGreater(numbers(uvscore.check(folded, texture_px=256, use_fast=False))["overlap_pct"], 5)
        self.assertEqual(numbers(uvscore.check(m, texture_px=256, use_fast=False))["overlap_pct"], 0)
        # the same island moved one tile over is not an overlap with the first tile's content
        both = mesh(np.concatenate([m.points, m.points]), np.r_[m.indptr, m.indptr[1:] + m.indptr[-1]],
                    np.r_[m.flat, m.flat + len(m.points)], np.concatenate([m.uv, m.uv + [1.0, 0.0]]))
        out = numbers(uvscore.check(both, texture_px=256, use_fast=False))
        self.assertEqual(out["overlap_pct"], 0)
        self.assertAlmostEqual(out["space_used_pct"], 81.0, delta=1.0)
        # intended overlap (mirrored on purpose) is not counted
        allow = np.ones((len(folded.indptr) - 1,), bool)
        self.assertEqual(numbers(uvscore.check(folded, texture_px=256, allow_overlap=allow, use_fast=False))["overlap_pct"], 0)
        # materials are separate pages: the same UVs on two materials do not overlap
        stacked = mesh(both.points, both.indptr, both.flat, np.concatenate([m.uv, m.uv]), np.r_[np.zeros(16, int), np.ones(16, int)])
        self.assertEqual(numbers(uvscore.check(stacked, texture_px=256, use_fast=False))["overlap_pct"], 0)
        self.assertGreater(numbers(uvscore.check(mesh(*stacked[:4]), texture_px=256, use_fast=False))["overlap_pct"], 90)

    def test_per_chart_stretch_ignores_chart_scale(self):
        # two islands, the second drawn 3x bigger on purpose: no stretch (intended scale divided out), pixel spread shows the 3x
        m = plane(8, True)
        uv = m.uv.copy()
        right = np.repeat(m.points[m.flat].reshape(-1, 4, 3)[:, :, 0].mean(1) > 4, 4)
        uv[right] = (uv[right] - 0.5) * 3 + 0.5
        out = numbers(uvscore.check(mesh(m.points, m.indptr, m.flat, uv), use_fast=False))
        self.assertAlmostEqual(out["stretch_med"], 1.0, places=6)
        self.assertGreater(out["pixel_spread"], 2.5)
        dens = np.where(right[::4], 3.0, 1.0)  # the intended density says the right half should be 3x
        out = numbers(uvscore.check(mesh(m.points, m.indptr, m.flat, uv), density=dens, use_fast=False))
        self.assertLess(out["pixel_spread"], 1.01)

    def test_flipped_against_chart_and_seen_split(self):
        m = plane(6)
        uv = m.uv.copy()
        vertex = 3 * 7 + 3  # an interior vertex dragged past its neighbours folds the faces around it
        uv[m.flat == vertex, 0] += 2.5 * 0.9 * 8.0 / 6 / 8.0
        folded = mesh(m.points, m.indptr, m.flat, uv)
        seen = np.ones(36)
        res = uvscore.check(folded, seen_face=seen, use_fast=False)
        out = numbers(res)
        self.assertGreaterEqual(out["flipped_faces"], 1)
        self.assertEqual(out["flipped_seen_faces"], out["flipped_faces"])
        seen[res.flipped] = 0.0
        out = numbers(uvscore.check(folded, seen_face=seen, use_fast=False))
        self.assertGreaterEqual(out["flipped_faces"], 1)
        self.assertEqual(out["flipped_seen_faces"], 0)

    def test_seams_in_sight_with_synthetic_seen(self):
        m = plane(8, True)
        a, b = uvscore._edges(m)[:2]
        length = np.linalg.norm(m.points[a] - m.points[b], axis=1)
        out = numbers(uvscore.check(m, seen_edge_px=3.0 * length, use_fast=False))
        self.assertAlmostEqual(out["seams_in_sight"], 1.0, places=9)
        self.assertGreater(out["cut_per_sqrt_area"], 0.5)

    def test_straight_rows_and_bend(self):
        m = plane(6)
        n = 6
        col, row = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        col, row = col.ravel()[m.flat], row.ravel()[m.flat]
        on = np.ones(36, bool)
        out = numbers(uvscore.check(m, col=col, row=row, on_chart=on, use_fast=False))
        self.assertEqual(out["straight_rows_pct"], 100.0)
        uv = m.uv.copy()
        uv[:, 1] += 0.01 * np.sin(col * 1.3)  # rows wobble by ~10 texels
        wob = mesh(m.points, m.indptr, m.flat, uv)
        out = numbers(uvscore.check(wob, col=col, row=row, on_chart=on, use_fast=False))
        self.assertLess(out["straight_rows_pct"], 50.0)
        out = numbers(uvscore.check(wob, col=col, row=row, on_chart=~on, use_fast=False))
        self.assertGreater(out["row_bend_p95"], 5.0)

    def test_extra_and_split_faces(self):
        m = plane(4)
        out = numbers(uvscore.check(m, faces_off=10, split=np.arange(16) < 3, use_fast=False))
        self.assertEqual((out["extra_faces"], out["split_faces"]), (6, 3))

    def test_empty_mesh(self):
        m = mesh(np.zeros((0, 3)), [0], [], np.zeros((0, 2)))
        out = numbers(uvscore.check(m, use_fast=False))
        self.assertEqual(out["faces"], 0)


def base_numbers(**kw):
    n = dict.fromkeys(uvscore.NAMES, float("nan"))
    n.update(faces=1000.0, tris=2000.0, straddle_pct=0.0, drift_p99=0.0, flipped_faces=0.0, flipped_seen_faces=0.0, overlap_pct=0.0,
             stretch_med=1.0, stretch_worst1=1.2, stretch_worst1_seen=1.2, pixel_spread=1.2, space_used_pct=80.0,
             seams_in_sight=0.2, straight_rows_pct=95.0)
    n.update(kw)
    return n


class Verdict(unittest.TestCase):
    def level(self, mode="new", source=None, **kw):
        return uvscore.uv_verdict(base_numbers(**kw), mode, source)

    def test_table_new(self):
        self.assertEqual(self.level()[0], "GOOD")
        self.assertEqual(self.level(stretch_med=1.2)[0], "CHECK")
        self.assertEqual(self.level(stretch_med=1.5)[0], "BAD")
        self.assertEqual(self.level(stretch_worst1=2.0, stretch_worst1_seen=2.0)[0], "CHECK")
        self.assertEqual(self.level(space_used_pct=60.0)[0], "CHECK")
        self.assertEqual(self.level(space_used_pct=40.0)[0], "BAD")
        self.assertEqual(self.level(pixel_spread=2.0)[0], "CHECK")
        self.assertEqual(self.level(pixel_spread=3.0)[0], "BAD")
        self.assertEqual(self.level(straight_rows_pct=70.0)[0], "CHECK")
        self.assertEqual(self.level(straight_rows_pct=30.0)[0], "BAD")
        self.assertEqual(self.level(seams_in_sight=0.4)[0], "CHECK")
        self.assertEqual(self.level(seams_in_sight=0.7)[0], "BAD")
        self.assertEqual(self.level(overlap_pct=0.3)[0], "CHECK")
        self.assertEqual(self.level(overlap_pct=2.0)[0], "BAD")

    def test_table_keep(self):
        self.assertEqual(self.level("keep")[0], "GOOD")
        self.assertEqual(self.level("keep", straddle_pct=3.4)[0], "CHECK")
        self.assertIn("3.4 %", self.level("keep", straddle_pct=3.4)[1])
        self.assertEqual(self.level("keep", straddle_pct=12.0)[0], "BAD")
        self.assertEqual(self.level("keep", drift_p99=5.0)[0], "CHECK")
        self.assertEqual(self.level("keep", drift_p99=9.0)[0], "BAD")

    def test_keep_rules_are_relative_to_the_source(self):
        src = base_numbers(stretch_med=1.78, stretch_worst1=3.0, pixel_spread=4.0, overlap_pct=0.4, flipped_seen_faces=3.0, flipped_faces=3.0)
        same = dict(stretch_med=1.78, stretch_worst1=3.0, stretch_worst1_seen=3.0, pixel_spread=4.0, overlap_pct=0.4, flipped_seen_faces=3.0, flipped_faces=3.0)
        self.assertEqual(self.level("keep", src, **same)[0], "GOOD")  # the source's own density and flips are not Seam Fit's doing
        self.assertEqual(self.level("keep", None, **same)[0], "BAD")
        worse = dict(same, stretch_med=2.2)
        self.assertNotEqual(self.level("keep", src, **worse)[0], "GOOD")

    def test_unseen_problems_never_pass_check(self):
        self.assertEqual(self.level(flipped_faces=20.0, flipped_seen_faces=0.0)[0], "CHECK")
        self.assertEqual(self.level(flipped_faces=20.0, flipped_seen_faces=20.0)[0], "BAD")
        self.assertEqual(self.level(flipped_faces=1.0, flipped_seen_faces=1.0)[0], "CHECK")
        self.assertEqual(self.level(stretch_worst1=4.0, stretch_worst1_seen=1.3)[0], "CHECK")
        self.assertEqual(self.level(stretch_worst1=4.0, stretch_worst1_seen=4.0)[0], "BAD")

    def test_title_names_the_worst_rule(self):
        level, title, _ = self.level("keep", straddle_pct=3.4)
        self.assertEqual((level, title), ("CHECK", "Check: 3.4 % of faces cross a seam"))
        self.assertEqual(self.level()[1], "Good for baking")
        self.assertEqual(self.level(seam_step=None)[0], "GOOD")
        self.assertEqual(uvscore.uv_verdict(base_numbers(), "new", None, seam_step=4.0)[0], "CHECK")


@needs_native
class Parity(unittest.TestCase):
    def native(self, res, src=None, **kw):
        def arr(a, dt=np.float64):
            return np.zeros(0, dt) if a is None else np.asarray(a, dtype=dt).ravel()

        extra = (src.points, src.indptr, src.flat, src.uv, arr(src.material, np.int64)) if src is not None else ((), (), (), (), ())
        job = turbo.uvscore_check(res.points, res.indptr, res.flat, res.uv, arr(res.material, np.int64), *extra,
                                  src_region=arr(kw.get("src_region"), np.int64), seen_face=arr(kw.get("seen_face")),
                                  seen_edge_px=arr(kw.get("seen_edge_px")), allow_overlap=arr(kw.get("allow_overlap"), bool),
                                  col=arr(kw.get("col"), np.int64), row=arr(kw.get("row"), np.int64), on_chart=arr(kw.get("on_chart"), bool),
                                  density=arr(kw.get("density")), faces_off=kw.get("faces_off", 0), split=arr(kw.get("split"), bool),
                                  texture_px=kw.get("texture_px", 1024.0))
        self.assertIsNotNone(job)
        while not job.done():
            time.sleep(0.001)
        return job.result()

    def same(self, res, src=None, **kw):
        twin = uvscore.check(res, src, use_fast=False, **kw)
        n, straddle, flipped = self.native(res, src, **kw)
        np.testing.assert_allclose(n, twin.numbers, rtol=1e-6, atol=1e-6, equal_nan=True, err_msg=str(list(zip(uvscore.NAMES, n, twin.numbers))))
        np.testing.assert_array_equal(straddle, twin.straddle)
        np.testing.assert_array_equal(flipped, twin.flipped)
        return n

    def test_names_and_constants(self):
        self.assertEqual(turbo.uvscore_names(), uvscore.NAMES)

    def test_source_and_result_planes(self):
        src = plane(8, True)
        self.same(src, src)
        self.same(plane(8, False), src)
        n = self.same(plane(12, True, z=0.2 * 8 / 12), src)
        self.assertLess(n[uvscore.NAMES.index("drift_p99")], 1.0)

    def test_rotated_noisy_result(self):
        rng = np.random.default_rng(3)
        src = plane(10, True)
        res = plane(7, True, z=0.05)
        points = res.points + rng.normal(0, 0.03, res.points.shape)
        uv = res.uv + rng.normal(0, 0.002, res.uv.shape)
        self.same(mesh(points, res.indptr, res.flat, uv), src, seen_face=rng.random(len(res.indptr) - 1))

    def test_overlap_rows_density_and_edges(self):
        m = plane(6, True)
        n = 6
        col, row = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        col, row = col.ravel()[m.flat], row.ravel()[m.flat]
        uv = m.uv.copy()
        uv[:, 1] += 0.004 * np.sin(col * 1.3)
        w = mesh(m.points, m.indptr, m.flat, uv)
        a, b = uvscore._edges(w)[:2]
        px = np.linalg.norm(w.points[a] - w.points[b], axis=1) * 2.5
        self.same(w, None, col=col, row=row, on_chart=np.arange(36) % 2 == 0, density=np.linspace(1, 2, 36), seen_edge_px=px,
                  seen_face=np.linspace(0, 1, 36), faces_off=30, split=np.arange(36) < 4, texture_px=300.0)
        folded = mesh(m.points, m.indptr, m.flat, np.stack([0.45 - np.abs(m.uv[:, 0] - 0.45), m.uv[:, 1]], 1))
        self.same(folded, None, allow_overlap=np.arange(36) < 6, texture_px=256.0)

    def test_materials_and_regions(self):
        src = plane(8, True)
        res = plane(8, False)
        mat = (np.arange(64) % 3).astype(np.int64)
        self.same(mesh(res.points, res.indptr, res.flat, res.uv, mat), mesh(src.points, src.indptr, src.flat, src.uv, mat))
        region = np.zeros(64, np.int64)  # all one side region: nothing straddles
        n = self.same(res, src, src_region=region)
        self.assertEqual(n[uvscore.NAMES.index("straddle_faces")], 0)

    def test_empty_inputs(self):
        m = mesh(np.zeros((0, 3)), [0], [], np.zeros((0, 2)))
        n, _, _ = self.native(m)
        self.assertEqual(n[0], 0)

    def test_speed_at_23k_faces(self):
        src = plane(152, True)
        n = 152
        # a coarser, shifted remesh of the same plane
        res = plane(120, True, z=0.0)
        t = time.perf_counter()
        job = turbo.uvscore_check(res.points, res.indptr, res.flat, res.uv, (), src.points, src.indptr, src.flat, src.uv)
        started = time.perf_counter() - t
        self.assertLess(started, 0.010, "the wrapper blocked the caller")
        while not job.done():
            time.sleep(0.001)
        took = time.perf_counter() - t
        job.result()
        print(f"\nuvscore_check: {len(src.indptr) - 1} source faces, {len(res.indptr) - 1} result faces: {took:.3f} s")
        self.assertLess(took, 0.2)
        del n


if __name__ == "__main__":
    unittest.main()
