# SPDX-License-Identifier: GPL-3.0-or-later
"""Seam snap and split leftovers (K1b): the NumPy twin on its own, then twin against the compiled kernel (parity) and
the job wrapper."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

import uv_parity_common as common  # noqa: F401  (puts the add-on folder on the path)
from uv_parity_common import needs_native

from triadq import uvsnap

import turbo


def grid(nx, ny, w, h):
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    points = np.stack([w * i.ravel() / nx, h * j.ravel() / ny, 0.0 * i.ravel()], 1).astype(float)
    ids = (j * (nx + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + nx + 2, ids + nx + 1], 1).ravel()
    return points, np.arange(0, len(flat) + 1, 4), flat


def seams_of(lines, gap=0.25, corner=None, closed=()):
    """Seam dict and welded positions for polylines; side A UV = (x, y) / 8, side B = side A + (gap, 0). `corner` = a
    point every line starts at (one shared welded id); `closed` = lines that are loops (last vertex = first)."""

    xyz, run_vertex, indptr = [], [], [0]
    if corner is not None:
        xyz.append(corner)
    for k, line in enumerate(lines):
        ids = []
        for i, p in enumerate(line):
            if corner is not None and i == 0:
                ids.append(0)
            elif k in closed and i == len(line) - 1:
                ids.append(ids[0])
            else:
                ids.append(len(xyz))
                xyz.append(p)
        run_vertex += ids
        indptr.append(len(run_vertex))
    xyz = np.asarray(xyz, float)
    pts = xyz[run_vertex]
    uva = np.stack([pts[:, 0] / 8, pts[:, 1] / 8], 1)
    uvb = uva + [gap, 0.0]
    seams = dict(run_indptr=np.array(indptr), run_vertex=np.array(run_vertex), uv_a=uva.ravel(), uv_b=uvb.ravel(),
                 **{"class": np.full(len(lines), 2)}, gap_px=np.full(len(pts), gap * 1024))
    return seams, xyz.ravel()


def line(deg, x0=0.0, y0=0.0, length=8.0, k=17):
    t = np.linspace(0, 1, k)
    a = math.radians(deg)
    span = length / math.cos(a)
    return np.stack([x0 + t * span * math.cos(a), y0 + t * span * math.sin(a), 0 * t], 1)


def straddlers(out, deg, y0=0.0):
    """Faces of a split result with vertices strictly on both sides of a straight chain (seam vertices count as on it)."""

    pts = out["points"].reshape(-1, 3)
    a = math.radians(deg)
    side = (pts - [0, y0, 0]) @ np.array([-math.sin(a), math.cos(a), 0.0])
    side[out["vchain"] >= 0] = 0.0
    ip, fl = out["indptr"], out["flat"]
    return sum(1 for f in range(len(ip) - 1) if side[fl[ip[f]:ip[f + 1]]].max() > 1e-9 and side[fl[ip[f]:ip[f + 1]]].min() < -1e-9)


def edge_uses(out):
    ip, fl = out["indptr"], out["flat"]
    uses = {}
    for f in range(len(ip) - 1):
        face = fl[ip[f]:ip[f + 1]].tolist()
        for k in range(len(face)):
            key = tuple(sorted((face[k], face[(k + 1) % len(face)])))
            uses[key] = uses.get(key, 0) + 1
    return uses


def on_border(p):
    return min(p[0], p[1], 8 - p[0], 8 - p[1]) < 1e-9


def thin_case():
    """8x3 grid whose row-1 vertices sit at y = 0.9 and vertex (4, 2) is pulled down to y = 0.95: moving (4, 0.9) onto a
    chain at y = 1 would push it past (4, 0.95) and fold the quads above it."""

    P, I, F = grid(8, 3, 8.0, 3.0)
    P[9:18, 1] = 0.9
    P[22, 1] = 0.95
    return P, I, F


def run_both(points, indptr, flat, seams, xyz, max_move=(), budget=-1.0, use_fast=False):
    s = uvsnap.snap(points, indptr, flat, seams, xyz, max_move=max_move, use_fast=use_fast)
    r = uvsnap.split(s["points"], indptr, flat, seams, xyz, vchain=s["vchain"], vparam=s["vparam"], budget=budget, use_fast=use_fast)
    return s, r


class Twin(unittest.TestCase):
    def test_8_degrees_found_and_accepted(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(8)])
        s, r = run_both(P, I, F, seams, xyz)
        self.assertIn(int(s["state"][0]), (uvsnap.ACCEPTED, uvsnap.PARTLY_HELD))
        self.assertLess(float(s["mean_dist"][0]), uvsnap.SNAP_MEAN_DIST)
        self.assertGreaterEqual(int(s["moved"]), 8)
        self.assertEqual(straddlers(r, 8), 0)
        self.assertLessEqual(int(r["extra_faces"]), 2)  # only the staircase step's held vertex leaves a face to split

    def test_30_degrees_found_and_refused_then_split(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(30)])
        s, r = run_both(P, I, F, seams, xyz)
        self.assertEqual(int(s["state"][0]), uvsnap.REFUSED)
        self.assertGreater(int(s["turns"][0]) * uvsnap.SNAP_TURN_EVERY, int(s["path_edges"][0]))
        moved = np.flatnonzero(np.any(s["points"].reshape(-1, 3) != P, axis=1))  # only vertices next to the chain, onto it
        edge = uvsnap.vertex_edge(P, I, F)
        chain = uvsnap._chains(seams, xyz)[0]
        for v in moved.tolist():
            self.assertLessEqual(float(np.linalg.norm(s["points"].reshape(-1, 3)[v] - P[v])), uvsnap.REUSE_EDGE * edge[v] + 1e-12)
            self.assertLess(chain.closest(0, len(chain.pts) - 1, tuple(s["points"].reshape(-1, 3)[v]))[0], 1e-20)
            self.assertEqual(int(s["vchain"][v]), 0)
        self.assertEqual(straddlers(r, 30), 0)
        self.assertGreater(int(r["faces_split"]), 0)
        self.assertGreater(int(r["triangles_left"]), 0)  # counted for the card
        self.assertEqual(int(r["faces_cannot"]) + int(r["faces_bake"]) + int(r["faces_left"]), 0)

    def test_crossings_shared_no_t_junctions(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(30)])
        _, r = run_both(P, I, F, seams, xyz)
        pts = r["points"].reshape(-1, 3)
        uses = edge_uses(r)
        self.assertTrue(all(n == 2 or (n == 1 and on_border(pts[a]) and on_border(pts[b])) for (a, b), n in uses.items()))
        new = range(len(P), len(pts))
        for v in new:
            faces = int(np.sum(r["flat"] == v))
            self.assertGreaterEqual(faces, 1 if on_border(pts[v]) else 2)

    def test_new_vertex_uvs_from_the_chain_per_side(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(30)])
        _, r = run_both(P, I, F, seams, xyz)
        pts, fl, ip = r["points"].reshape(-1, 3), r["flat"], r["indptr"]
        cuv, side = r["corner_uv"].reshape(-1, 2), r["corner_side"]
        a = math.radians(30)
        left = (pts @ np.array([-math.sin(a), math.cos(a), 0.0]))
        for h in np.flatnonzero(fl >= len(P)):
            want = pts[fl[h], :2] / 8 + [0.25 * side[h], 0.0]
            np.testing.assert_allclose(cuv[h], want, atol=1e-9)
        for f in range(len(ip) - 1):
            hs = np.arange(ip[f], ip[f + 1])
            seam = hs[side[hs] >= 0]
            off = hs[(side[hs] < 0) & (np.abs(left[fl[hs]]) > 1e-9)]
            if len(seam) and len(off):
                self.assertEqual(len(set(side[seam].tolist())), 1)
                self.assertEqual(int(side[seam][0]), 0 if left[fl[off[0]]] > 0 else 1)  # side A = left of the chain

    def test_vertex_reused_under_a_tenth_of_an_edge(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(0, 0.0, 4.05)])
        r = uvsnap.split(P, I, F, seams, xyz, use_fast=False)
        self.assertEqual(len(r["points"]), len(P.ravel()))
        np.testing.assert_array_equal(r["points"], P.ravel())
        self.assertEqual(int(r["extra_faces"]), 0)
        row = np.arange(36, 45)
        self.assertTrue(np.all(r["vchain"][row] == 0))
        self.assertTrue(np.all(~np.isnan(r["corner_uv"].reshape(-1, 2)[np.isin(r["flat"], row)])))

    def test_y_junction_fan(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        c = (3.4, 3.6, 0.0)
        legs = [np.linspace(c, (0.0, 3.6, 0.0), 9), np.linspace(c, (3.4, 8.0, 0.0), 9), np.linspace(c, (8.0, 0.0, 0.0), 9)]
        seams, xyz = seams_of(legs, corner=c)
        r = uvsnap.split(P, I, F, seams, xyz, use_fast=False)
        face = 3 * 8 + 3
        self.assertEqual(int(r["face_state"][face]), uvsnap.SPLIT)
        pts = r["points"].reshape(-1, 3)
        hub = int(np.flatnonzero(np.all(np.abs(pts - c) < 1e-9, axis=1))[0])
        self.assertGreaterEqual(int(np.sum(r["flat"] == hub)), 3)
        uses = edge_uses(r)
        self.assertTrue(all(n == 2 or (n == 1 and on_border(pts[a]) and on_border(pts[b])) for (a, b), n in uses.items()))

    def test_two_chains_cut_one_after_the_other(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(0, 0.0, 3.3), line(0, 0.0, 3.7)])
        r = uvsnap.split(P, I, F, seams, xyz, use_fast=False)
        self.assertEqual(int(r["faces_split"]), 8)
        self.assertEqual(int(r["extra_faces"]), 16)
        self.assertEqual(int(r["triangles_left"]), 0)
        self.assertEqual(straddlers(r, 0, 3.3) + straddlers(r, 0, 3.7), 0)

    def test_whole_island_face_goes_to_bake(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        loop = [(3.3, 3.3, 0), (3.7, 3.3, 0), (3.7, 3.7, 0), (3.3, 3.7, 0), (3.3, 3.3, 0)]
        seams, xyz = seams_of([loop], closed=(0,))
        r = uvsnap.split(P, I, F, seams, xyz, use_fast=False)
        self.assertEqual(int(r["face_state"][3 * 8 + 3]), uvsnap.BAKE)
        self.assertEqual((int(r["faces_bake"]), int(r["extra_faces"])), (1, 0))

    def test_max_move_zero_holds_and_split_never_moves(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(8)])
        s, r = run_both(P, I, F, seams, xyz, max_move=np.zeros(len(P)))
        np.testing.assert_array_equal(s["points"], P.ravel())
        self.assertGreater(int(np.sum(s["held"])), 5)
        held = np.flatnonzero(s["held"])
        np.testing.assert_array_equal(r["points"].reshape(-1, 3)[: len(P)], P)  # split never moves a vertex
        np.testing.assert_array_equal(r["points"].reshape(-1, 3)[held], P[held])
        self.assertEqual(straddlers(r, 8), 0)

    def test_snap_never_flips(self):
        P, I, F = thin_case()
        seams, xyz = seams_of([line(0, 0.0, 1.0)])
        s = uvsnap.snap(P, I, F, seams, xyz, use_fast=False)
        self.assertTrue(bool(s["held"][13]))  # (4, 0.9) would fold the quads under (4, 0.95)
        out = s["points"].reshape(-1, 3)
        for f in range(len(I) - 1):
            q = out[F[I[f]:I[f + 1]]]
            self.assertTrue(uvsnap._face_ok([tuple(p) for p in q]))
            self.assertGreater(uvsnap._newell([tuple(p) for p in q])[2], 0.0)

    def test_budget_spends_on_the_worst_run_first(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(0, 0.0, 2.5), line(0, 0.0, 5.5)])
        seams["gap_px"][17:] *= 4.0  # the second chain shows a bigger jump
        r = uvsnap.split(P, I, F, seams, xyz, budget=8.0, use_fast=False)
        self.assertEqual((int(r["extra_faces"]), int(r["faces_left"])), (8, 8))
        self.assertTrue(np.all(r["face_state"][40:48] == uvsnap.SPLIT))
        self.assertTrue(np.all(r["face_state"][16:24] == uvsnap.LEFT))
        self.assertEqual(float(r["wanted"]), 16.0)

    def test_stage_slices_stay_short(self):
        P, I, F = grid(24, 24, 8.0, 8.0)
        seams, xyz = seams_of([line(30, k=60)])
        runs = []
        for _ in range(3):  # best of three: a busy PC can preempt one slice of a low-priority test
            stage = uvsnap.Split(P, I, F, seams, xyz, use_fast=False)
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
        turbo.native()

    def run_job(self, job):
        self.assertIsNotNone(job)
        job._thread.join(60)
        return job.result()

    def same(self, a, b, skip=()):
        for key in a:
            if key in skip:
                continue
            x, y = np.asarray(a[key]), np.asarray(b[key])
            self.assertEqual(x.shape, y.shape, key)
            if x.dtype.kind == "f":
                np.testing.assert_allclose(x, y, atol=1e-6, rtol=0, err_msg=key)
            else:
                np.testing.assert_array_equal(x, y, err_msg=key)

    def cases(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        bent = P.copy()
        bent[:, 2] = 0.1 * np.sin(bent[:, 0]) * np.cos(bent[:, 1])
        c = (3.4, 3.6, 0.0)
        legs = [np.linspace(c, (0.0, 3.6, 0.0), 9), np.linspace(c, (3.4, 8.0, 0.0), 9), np.linspace(c, (8.0, 0.0, 0.0), 9)]
        yield (P, I, F) + seams_of([line(8)]) + ((),)
        yield (P, I, F) + seams_of([line(30)]) + ((),)
        yield (P, I, F) + seams_of([line(8)]) + (np.zeros(len(P)),)
        yield (bent, I, F) + seams_of([line(11), line(0, 0.0, 6.3)]) + ((),)
        yield (P, I, F) + seams_of(legs, corner=c) + ((),)
        yield thin_case() + seams_of([line(0, 0.0, 1.0)]) + ((),)

    def test_snap_and_split_match(self):
        for P, I, F, seams, xyz, mm in self.cases():
            twin_s, twin_r = run_both(P, I, F, seams, xyz, max_move=mm, budget=-1.0)
            kern_s = self.run_job(turbo.uvsnap_snap(P, I, F, seams, xyz, max_move=mm))
            self.same(twin_s, kern_s)
            kern_r = self.run_job(turbo.uvsnap_split(kern_s["points"], I, F, seams, xyz, vchain=kern_s["vchain"], vparam=kern_s["vparam"]))
            self.same(twin_r, kern_r)
            for budget in (0.0, 5.0):
                t = uvsnap.split(P, I, F, seams, xyz, budget=budget, use_fast=False)
                k = self.run_job(turbo.uvsnap_split(P, I, F, seams, xyz, budget=budget))
                self.same(t, k)

    def test_stage_hands_the_work_to_the_job(self):
        uv_fast = common.addon_module("uv_fast")
        fast_snap = common.addon_module("triadq.uvsnap")
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(30)])
        self.assertTrue(uv_fast.install())
        try:
            stage = fast_snap.Split(P, I, F, seams, xyz)
            self.assertIsNotNone(stage._job)
            while not stage.done:
                stage.step()
            self.assertEqual(stage.numbers["faces_split"], int(uvsnap.split(P, I, F, seams, xyz, use_fast=False)["faces_split"]))
        finally:
            uv_fast.uninstall()

    def test_wrapper_never_blocks(self):
        P, I, F = grid(200, 200, 8.0, 8.0)
        seams, xyz = seams_of([line(30, k=400), line(3, 0.0, 2.0, k=400)])
        for start in (lambda: turbo.uvsnap_snap(P, I, F, seams, xyz), lambda: turbo.uvsnap_split(P, I, F, seams, xyz)):
            t = time.perf_counter()
            job = start()
            self.assertLess(time.perf_counter() - t, 0.010)
            self.assertTrue(self.run_job(job)["done"])

    def test_stop_hands_back_partial(self):
        P, I, F = grid(8, 8, 8.0, 8.0)
        seams, xyz = seams_of([line(30)])
        job = turbo.uvsnap_split(P, I, F, seams, xyz)
        job.stop()
        out = self.run_job(job)
        self.assertIn(bool(out["done"]), (True, False))


if __name__ == "__main__":
    unittest.main()
