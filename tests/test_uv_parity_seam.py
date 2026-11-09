# SPDX-License-Identifier: GPL-3.0-or-later
"""Seam kernel (K1): the NumPy twin on its own, then twin against the compiled kernel (parity) and the job wrapper."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

import uv_parity_common as common  # noqa: F401  (puts the add-on folder on the path)
from uv_parity_common import needs_native

from triadq import uvseams

from uv_parity_common import turbo  # noqa: E402  (None in a build without it)


def grid(nx, ny, w, h):
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    points = np.stack([w * i.ravel() / nx, h * j.ravel() / ny, 0.0 * i.ravel()], 1).astype(float)
    ids = (j * (nx + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + nx + 2, ids + nx + 1], 1).ravel()
    return points, np.arange(0, len(flat) + 1, 4), flat


def two_islands():
    """8x8 plane in two islands split at x = 4: left UV = (x, y)/8, right UV = (x/8 + 0.25, y/8)."""

    points, indptr, flat = grid(8, 8, 8.0, 8.0)
    right = np.repeat((np.arange(64) % 8) >= 4, 4)
    uv = np.stack([points[flat, 0] / 8 + 0.25 * right, points[flat, 1] / 8], 1).ravel()
    return points, indptr, flat, uv


def tube(nu, nv, turn, wrap=1.0):
    """Open tube around z, welded points, u = angle / 2 pi from 0 to `wrap` (one cut at angle `turn`), v = z."""

    pts = []
    for j in range(nv + 1):
        for i in range(nu):
            a = 2 * math.pi * (i + turn) / nu
            pts.append((math.cos(a), math.sin(a), j * 0.5))
    flat, uv = [], []
    for j in range(nv):
        for i in range(nu):
            i1 = (i + 1) % nu
            ids = (j * nu + i, j * nu + i1, (j + 1) * nu + i1, (j + 1) * nu + i)
            for k, (us, vs) in enumerate(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                flat.append(ids[k])
                uv.extend((wrap * (us + turn) / nu, vs / nv))
    flat = np.array(flat)
    return np.array(pts), np.arange(0, len(flat) + 1, 4), flat, np.array(uv)


def fold_case():
    """8x8 plane, middle vertex with a non-convex 1-ring and a UV thrown outside it."""

    points, indptr, flat = grid(8, 8, 8.0, 8.0)
    points[39, 0] += 0.8
    points[41, 1] -= 0.6
    uv = np.stack([points[flat, 0] / 8, points[flat, 1] / 8], 1).ravel()
    before = uv.copy()
    uv[2 * np.flatnonzero(flat == 40)] = 0.4
    uv[2 * np.flatnonzero(flat == 40) + 1] = 0.7
    return points, indptr, flat, uv, before


def sliver_case():
    """8x8 plane, middle vertex read 0.2 px past its right neighbour's column: the faces on its right fold by a sliver."""

    points, indptr, flat = grid(8, 8, 8.0, 8.0)
    uv = np.stack([points[flat, 0] / 8, points[flat, 1] / 8], 1).ravel()
    before = uv.copy()
    uv[2 * np.flatnonzero(flat == 40)] = 5 / 8 + 0.2 / 1024
    return points, indptr, flat, uv, before


def crease():
    """A floor (z = 0, x in 0..4) and a wall (x = 4, z in 0..2) at 90 deg in one island: floor UV (x, y) / 8, wall UV
    (4 + z, y) / 8 (the wall unrolled beside the floor, same winding as seen from outside the corner)."""

    pts, flat, uv = [], [], []
    for j in range(3):
        for i in range(5):
            pts.append((float(i), float(j), 0.0))
    for j in range(3):
        for k in range(1, 3):
            pts.append((4.0, float(j), float(k)))
    def fid(i, j):
        return j * 5 + i

    def wid(k, j):
        return fid(4, j) if k == 0 else 15 + j * 2 + (k - 1)
    for j in range(2):
        for i in range(4):
            ids = (fid(i, j), fid(i + 1, j), fid(i + 1, j + 1), fid(i, j + 1))
            flat.extend(ids)
            uv.extend(c for v in ids for c in (pts[v][0] / 8, pts[v][1] / 8))
        for k in range(2):
            ids = (wid(k, j), wid(k + 1, j), wid(k + 1, j + 1), wid(k, j + 1))  # x = 4 wall, normal +x after the turn
            ids = (ids[0], ids[3], ids[2], ids[1])
            flat.extend(ids)
            uv.extend(c for v in ids for c in ((4 + pts[v][2]) / 8, pts[v][1] / 8))
    flat = np.array(flat)
    return np.array(pts), np.arange(0, len(flat) + 1, 4), flat, np.array(uv)


def drift_px(uv, want):
    return float(np.max(np.abs(np.asarray(uv) - np.asarray(want)))) * 1024.0


class Twin(unittest.TestCase):
    def test_plane_two_islands(self):
        points, indptr, flat, uv = two_islands()
        seams = uvseams.extract(points, indptr, flat, uv, use_fast=False)
        self.assertEqual((seams["n_islands"], len(seams["class"]), len(seams["run_vertex"])), (2, 1, 9))
        self.assertEqual(seams["class"][0], uvseams.SOFT)
        self.assertAlmostEqual(seams["gap_px"][0], 256.0)
        out = uvseams.transfer(points, indptr, flat, uv, points, indptr, flat, use_fast=False)
        self.assertLess(drift_px(out["uv"], uv), 1e-6)  # every face on its own side: 0 straddlers, split corners per side
        self.assertEqual(out["folded_before"], 0)

    def test_plane_offset_reads_its_side(self):
        points, indptr, flat, uv = two_islands()
        res = points.copy()
        res[:, 0] = 4 + (res[:, 0] - 4) * 1.01
        res[:, 2] = 0.2
        out = uvseams.transfer(points, indptr, flat, uv, res, indptr, flat, use_fast=False)
        right = np.repeat((np.arange(64) % 8) >= 4, 4)
        want = np.clip(res[flat, 0], 0, 8) / 8 + 0.25 * right
        self.assertLess(drift_px(out["uv"][0::2], want), 1e-6)

    def test_self_seam_tube_reads_each_side(self):
        points, indptr, flat, uv = tube(12, 4, 0.0)
        seams = uvseams.extract(points, indptr, flat, uv, repeat=[False], use_fast=False)
        self.assertEqual((seams["n_islands"], len(seams["class"]), int(seams["class"][0])), (1, 1, uvseams.SOFT))
        out = uvseams.transfer(points, indptr, flat, uv, points, indptr, flat, repeat=[False], use_fast=False)
        self.assertLessEqual(drift_px(out["uv"], uv), 1.0)  # p99 <= 1 px holds for the largest one
        self.assertEqual(out["folded_after"], 0)

    def test_wrap_tube_shift_on_repeat(self):
        points, indptr, flat, uv = tube(12, 4, 0.0)
        self.assertEqual(int(uvseams.extract(points, indptr, flat, uv, repeat=[True], use_fast=False)["class"][0]), uvseams.SHIFT)
        self.assertEqual(int(uvseams.extract(points, indptr, flat, uv, repeat=[False], use_fast=False)["class"][0]), uvseams.SOFT)
        rp, ri, rf, _ = tube(12, 4, 0.5)
        out = uvseams.transfer(points, indptr, flat, uv, rp, ri, rf, repeat=[True], use_fast=False)
        u = out["uv"][0::2].reshape(-1, 4)
        self.assertLess(float(np.max(np.ptp(u, 1))), 0.1)  # faces across the wrap read continuous texture
        offsets = np.tile([0.5, 1.5, 1.5, 0.5], 48) + np.repeat(np.arange(48) % 12, 4)
        want = (offsets / 12) % 1.0
        got = out["uv"][0::2] % 1.0
        d = np.abs(got - want)
        self.assertLess(float(np.max(np.minimum(d, 1 - d))) * 1024, 1.0)
        self.assertTrue(np.any(out["shift"]))

    def test_walk_stays_on_its_side_of_a_thin_tube_cut(self):
        # a coarse result tube whose faces sit across the source tube's one cut: walking the long way round would read the
        # far side of the cut and fold the face; the walk's UV check keeps every corner on the reference side
        points, indptr, flat, uv = tube(24, 4, 0.0)
        res, rip, rfl, _ = tube(4, 2, 0.5)
        out = uvseams.transfer(points, indptr, flat, uv, res, rip, rfl, repeat=[False], fix_folds=False, use_fast=False)
        self.assertEqual(sum(uvseams.folded(rip, out["uv"], out["expected"], f) for f in range(len(rip) - 1)), 0)

    def test_coarse_faces_at_a_crease_read_the_surface_they_lie_on(self):
        # result quads on the floor whose corners sit on the crease line (the wall ties with the floor there), one quad
        # bent over the crease (its fan triangles want opposite windings: no common winding) and a zero-area triangle
        points, indptr, flat, uv = crease()
        res = np.array([[3.0, 0.2, 0.0], [4.0, 0.2, 0.0], [4.0, 0.8, 0.0], [3.0, 0.8, 0.0],
                        [3.4, 1.2, 0.0], [4.0, 1.2, 0.6], [4.0, 1.8, 0.6], [3.4, 1.8, 0.0],
                        [1.0, 1.0, 0.0], [1.5, 1.0, 0.0], [2.0, 1.0, 0.0]])
        rip, rfl = np.array([0, 4, 8, 11]), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        out = uvseams.transfer(points, indptr, flat, uv, res, rip, rfl, fix_folds=False, use_fast=False)
        self.assertEqual(out["orient"][0], 1.0)  # on the floor, read as the floor
        self.assertLess(drift_px(out["uv"][:8], res[[0, 1, 2, 3], :2].ravel() / 8), 1e-6)
        self.assertEqual(out["orient"][2], 0.0)  # no area: no winding

    def test_reread_unfolds_a_face_through_its_reference_triangle(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        sflat, suv = np.array([0, 1, 2, 3]), np.array([0, 0, 1, 0, 1, 1, 0, 1], float)
        res = np.array([[0.2, 0.2, 0.01], [0.6, 0.2, 0.0], [0.6, 0.6, 0.0], [0.2, 0.6, 0.0]])
        rip, rfl = np.array([0, 4]), np.array([0, 1, 2, 3])
        uv = np.array([0.2, 0.2, 0.2, 0.6, 0.6, 0.6, 0.6, 0.2])  # read mirrored: folded against the source's winding
        look = dict(ref_face=np.array([0]), ref_ids=np.array([0, 1, 2]))
        self.assertEqual(uvseams.reread_folded(src, sflat, suv, look, res, rip, rfl, uv, np.array([1.0])), 1)
        np.testing.assert_allclose(uv, [0.2, 0.2, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6], atol=1e-12)

    def test_fold_fix_folds_no_neighbour(self):
        points, indptr, flat, uv, before = fold_case()
        nf = len(indptr) - 1
        region, expected = np.zeros(nf, np.int64), np.ones(nf)
        folds = sum(uvseams.folded(indptr, uv, expected, f) for f in range(nf))
        self.assertGreater(folds, 0)
        report = uvseams.fold_fix(points, indptr, flat, uv, region, expected)
        self.assertEqual((report["before"], report["after"]), (folds, 0))
        self.assertGreater(report["untangled"], 0)  # untangling moved the thrown vertex back
        cheb = np.maximum(np.abs(flat % 9 - 4), np.abs(flat // 9 - 4)).reshape(-1, 4).min(1)
        far = np.repeat(cheb >= 3, 8)
        np.testing.assert_array_equal(uv[far], before[far])

    def test_fold_fix_moves_a_sliver_only_as_far_as_needed(self):
        points, indptr, flat, uv, before = sliver_case()
        nf = len(indptr) - 1
        region, expected = np.zeros(nf, np.int64), np.ones(nf)
        thrown = uv.copy()
        report = uvseams.fold_fix(points, indptr, flat, uv, region, expected)
        self.assertGreater(report["before"], 0)
        self.assertEqual(report["after"], 0)
        moved = np.abs(uv - thrown).reshape(-1, 2).max(1) * 1024
        self.assertTrue(np.all(moved[flat != 40] == 0))  # nothing else moves
        self.assertLess(moved[flat == 40].max(), 0.5)  # 0.2 px back over the line, 0.25 px clear: not to the ring's middle

    def test_triage_small_island(self):
        points, indptr, flat = grid(8, 8, 8.0, 8.0)
        small = np.repeat(np.arange(64) == 27, 4)
        uv = np.stack([points[flat, 0] / 8 + 0.5 * small, points[flat, 1] / 8], 1).ravel()
        seams = uvseams.extract(points, indptr, flat, uv, use_fast=False)
        self.assertEqual((len(seams["class"]), len(seams["run_vertex"])), (1, 5))  # a closed loop, first vertex repeated
        t = uvseams.triage(seams, points, indptr, flat, 64.0, 0.5)
        self.assertEqual(t["hard_islands"].tolist(), [1])
        self.assertAlmostEqual(float(t["hard_cost"][0]), 8.0)
        self.assertEqual(int(seams["class"][0]), uvseams.HARD)

    def test_stage_slices_stay_short(self):
        points, indptr, flat, uv = tube(24, 8, 0.0)
        rp, ri, rf, _ = tube(24, 8, 0.3)
        runs = []
        for _ in range(3):  # best of three: a busy PC can preempt one slice of a low-priority test
            stage = uvseams.Transfer(points, indptr, flat, uv, rp, ri, rf, use_fast=False)
            longest = 0.0
            while not stage.done:
                t = time.perf_counter()
                stage.step()
                longest = max(longest, time.perf_counter() - t)
            runs.append(longest)
            self.assertTrue(stage.result["done"])
        self.assertLess(min(runs), 0.010)


@needs_native
class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        turbo.native()  # load once, so the wrapper timing below measures the call only

    def run_job(self, job):
        self.assertIsNotNone(job)
        job._thread.join(60)
        return job.result()

    def same_dict(self, a, b, keys):
        for key in keys:
            x, y = np.asarray(a[key]), np.asarray(b[key])
            self.assertEqual(x.shape, y.shape, key)
            if x.dtype.kind == "f":
                np.testing.assert_allclose(x, y, atol=1e-6, rtol=0, err_msg=key)
            else:
                np.testing.assert_array_equal(x, y, err_msg=key)

    def test_extract_matches(self):
        keys = ("weld", "island", "region", "edge_a", "edge_b", "edge_chain", "edge_face", "corner", "run_indptr", "run_vertex",
                "uv_a", "uv_b", "gap_px", "class", "chain_island", "chain_material", "chain_length", "island_tile")
        for points, indptr, flat, uv in (two_islands(), tube(12, 4, 0.0)):
            for repeat in ([True], [False]):
                twin = uvseams.extract(points, indptr, flat, uv, repeat=repeat, use_fast=False)
                kern = self.run_job(turbo.uvseam_extract(points, indptr, flat, uv, repeat=repeat))
                self.same_dict(twin, kern, keys)

    def test_transfer_matches(self):
        p, i, f, uv = two_islands()
        off = p.copy()
        off[:, 2] = 0.2
        off[:, 0] += 0.3 * np.sin(off[:, 1])  # bent: the seam is no longer on a grid line
        tp, ti, tf, tuv = tube(12, 4, 0.0)
        rp, ri, rf, _ = tube(12, 4, 0.5)
        keys = ("uv", "expected", "ids", "w", "shift", "slid", "ref_ids", "ref_w", "ref_face", "orient", "region", "join")
        for src, res, repeat in (((p, i, f, uv), (off, i, f), []), ((tp, ti, tf, tuv), (rp, ri, rf), [True]), ((tp, ti, tf, tuv), (rp, ri, rf), [False])):
            twin = uvseams.transfer(*src, *res, repeat=repeat, use_fast=False)
            kern = self.run_job(turbo.uvseam_transfer(*src, *res, repeat=repeat))
            self.same_dict(twin, kern, keys)
            self.assertEqual((twin["folded_before"], twin["folded_after"]), (kern["folded_before"], kern["folded_after"]))

    def test_crease_reads_match(self):
        points, indptr, flat, uv = crease()
        keys = ("uv", "expected", "ids", "w", "shift", "slid", "ref_ids", "ref_w", "ref_face", "orient", "region")
        g = np.linspace(0.0, 1.0, 7)
        res = np.array([(3.0 + 1.4 * a, 2.0 * b, max(0.0, 1.4 * a - 0.6) * 1.5) for b in g for a in g])  # bent across x = 4
        res[24, 2] += 0.4  # one corner lifted: a quad folded in 3D
        ids = np.arange(49).reshape(7, 7)
        rfl = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]], -1).ravel()
        rip = np.arange(0, len(rfl) + 1, 4)
        twin = uvseams.transfer(points, indptr, flat, uv, res, rip, rfl, use_fast=False)
        kern = self.run_job(turbo.uvseam_transfer(points, indptr, flat, uv, res, rip, rfl))
        self.same_dict(twin, kern, keys)
        self.assertIn(0.0, set(np.asarray(kern["orient"]).tolist()))  # the vote's no-common-winding branch ran

    def test_fold_fix_and_layer_match(self):
        points, indptr, flat, uv, _ = fold_case()
        look = self.run_job(turbo.uvseam_transfer(points, indptr, flat, uv, points, indptr, flat, fix_folds=False))
        k_uv, k_exp, k_fold = self.run_job(turbo.uvseam_layer(indptr, flat, uv, points, indptr, flat, look))
        t_uv, t_exp, t_fold = uvseams.layer(indptr, flat, uv, points, indptr, flat, look, use_fast=False)
        np.testing.assert_allclose(t_uv, k_uv, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(t_exp, k_exp)
        self.assertEqual({k: t_fold[k] for k in k_fold}, k_fold)

    def test_sliver_fold_fix_matches(self):
        points, indptr, flat, uv, _ = sliver_case()
        look = self.run_job(turbo.uvseam_transfer(points, indptr, flat, uv, points, indptr, flat, fix_folds=False))
        k_uv, k_exp, k_fold = self.run_job(turbo.uvseam_layer(indptr, flat, uv, points, indptr, flat, look))
        t_uv, t_exp, t_fold = uvseams.layer(indptr, flat, uv, points, indptr, flat, look, use_fast=False)
        np.testing.assert_allclose(t_uv, k_uv, atol=1e-6, rtol=0)
        self.assertEqual({k: t_fold[k] for k in k_fold}, k_fold)
        self.assertGreater(k_fold["folded_before"], 0)
        self.assertEqual(k_fold["folded_after"], 0)

    def test_triage_matches(self):
        points, indptr, flat = grid(8, 8, 8.0, 8.0)
        small = np.repeat(np.isin(np.arange(64), [27, 28, 9]), 4)
        uv = np.stack([points[flat, 0] / 8 + 0.5 * small, points[flat, 1] / 8], 1).ravel()
        kern = self.run_job(turbo.uvseam_triage(points, indptr, flat, uv, 64.0, 0.2))
        seams = uvseams.extract(points, indptr, flat, uv, use_fast=False)
        twin = uvseams.triage(seams, points, indptr, flat, 64.0, 0.2)
        for key in ("hard_islands", "left_islands"):
            np.testing.assert_array_equal(twin[key], kern[key])
        np.testing.assert_allclose(twin["hard_cost"], kern["hard_cost"], atol=1e-9)
        np.testing.assert_array_equal(seams["class"], kern["class"])

    def test_wrapper_never_blocks(self):
        points, indptr, flat = grid(200, 200, 8.0, 8.0)
        uv = np.stack([points[flat, 0] / 8, points[flat, 1] / 8], 1).ravel()
        res = points.copy()
        res[:, 2] = 0.01
        started = time.perf_counter()
        job = turbo.uvseam_transfer(points, indptr, flat, uv, res, indptr, flat)
        self.assertLess(time.perf_counter() - started, 0.010)
        out = self.run_job(job)
        self.assertTrue(out["done"])

    def test_stop_hands_back_partial(self):
        points, indptr, flat, uv = two_islands()
        job = turbo.uvseam_transfer(points, indptr, flat, uv, points, indptr, flat)
        job.stop()
        out = self.run_job(job)
        if not out["done"]:
            self.assertTrue(np.any(out["ref_face"] < 0) or len(out["ref_face"]) == 0)


if __name__ == "__main__":
    unittest.main()
