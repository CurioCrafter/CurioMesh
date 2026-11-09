# SPDX-License-Identifier: GPL-3.0-or-later
"""Packer (K7): the twin on its own, the twin against the compiled kernel (parity), the job wrapper, and P0's Smart Project boxes."""

from __future__ import annotations

from pathlib import Path
import random
import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from triadq import uvpack

import turbo

BENCH = Path(__file__).resolve().parents[3] / "bench" / "uvplan"


def wait(job):
    """The result of a worker job (the tests may block; the add-on never does)."""

    while not job.done():
        time.sleep(0.001)
    return job.result()


def random_boxes(n, seed, lo=0.1, hi=1.0):
    rng = random.Random(seed)
    return [rng.uniform(lo, hi) for _ in range(n)], [rng.uniform(lo, hi) for _ in range(n)]


def sizes(w, h, result):
    out = []
    for i, (x, y, turned) in enumerate(result["places"]):
        a, b = uvpack.snap(w[i] * result["scale"]), uvpack.snap(h[i] * result["scale"])
        out.append((x, y, b, a) if turned else (x, y, a, b))
    return out


def assert_no_overlap(testcase, rects, page):
    for i, (x, y, w, h) in enumerate(rects):
        testcase.assertTrue(x >= 0 and y >= 0 and x + w <= page[0] and y + h <= page[1], f"box {i} outside the page")
        for x2, y2, w2, h2 in rects[:i]:
            testcase.assertTrue(x + w <= x2 or x2 + w2 <= x or y + h <= y2 or y2 + h2 <= y, "overlap")


def mip_clean(rects, page, k):
    """Rasterise every box at each reduced size (blocks of 2^j texels), grow it by one block, and require that no block is claimed twice."""

    for j in range(0, max(k, 0) + 1):
        s = 1 << j
        count = np.zeros((-(-page[1] // s) + 3, -(-page[0] // s) + 3), np.int32)
        for x, y, w, h in rects:
            count[y // s:(y + h - 1) // s + 3, x // s:(x + w - 1) // s + 3] += 1  # block b sits at index b + 1; one block of growth each way
        if count.max() > 1:
            return False, j
    return True, None


def no_overlap_raster(rects, page):
    """The same check as `assert_no_overlap`, by painting the page (for thousands of boxes)."""

    count = np.zeros((page[1], page[0]), np.int32)
    for x, y, w, h in rects:
        if x < 0 or y < 0 or x + w > page[0] or y + h > page[1]:
            return False
        count[y:y + h, x:x + w] += 1
    return int(count.max()) <= 1


def random_charts(n, seed):
    """n charts of one to three triangles (some boxy), in texels, as `pack_into` takes them: (tris, tri_chart, as_box)."""

    rng = random.Random(seed)
    tris, charts, flags = [], [], []
    for c in range(n):
        w, h = rng.uniform(4, 40), rng.uniform(4, 40)
        x0, y0 = rng.uniform(0, 5), rng.uniform(0, 5)
        tris += [x0, y0, x0 + w, y0, x0, y0 + h]
        charts.append(c)
        if rng.random() < 0.6:
            tris += [x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
            charts.append(c)
        flags.append(int(rng.random() < 0.4))
    return tris, charts, flags


def touched_blocks(tris, charts, placed, c, s):
    """Blocks of s texels chart c touches after `pack_into` placed it: a set of (bx, by)."""

    ok, dx, dy, q = placed[c]
    mine = [v for t, ch in enumerate(charts) if ch == c for v in tris[6 * t:6 * t + 6]]
    pts = [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(uvpack.turn_tris(uvpack.spin_tris(mine, q >> 2), q))]  # angle step, then quarters
    y0 = int(min(pts[1::2]) // 1)
    out = set()
    for y, rows in enumerate(uvpack.raster(pts, y0, int(-(-max(pts[1::2]) // 1)) - y0), y0):
        for a, b in rows:
            out |= {(bx, y // s) for bx in range(a // s, (b - 1) // s + 1)}
    return out


def block_gap_breaks(tris, charts, placed, page, k):
    """Pairs of charts whose touched blocks, each grown by one block, meet at some mip level up to k (and charts off the page)."""

    bad = []
    n = len(placed)
    for j in range(0, k + 1):
        s = 1 << j
        grown = []
        for c in range(n):
            blocks = touched_blocks(tris, charts, placed, c, s)
            if any(bx < 1 or by < 1 or bx > page // s - 2 or by > page // s - 2 for bx, by in blocks):
                bad.append((c, "page edge", j))
            grown.append({(bx + i, by + o) for bx, by in blocks for i in (-1, 0, 1) for o in (-1, 0, 1)})
        for a in range(n):
            for b in range(a):
                if grown[a] & grown[b]:
                    bad.append((a, b, j))
    return bad


class Twin(unittest.TestCase):
    def test_four_equal_squares_fill_the_page(self):
        r = uvpack.fill_page([1, 1, 1, 1], [1, 1, 1, 1], page=(1024, 1024), k=-1)
        self.assertAlmostEqual(r["used"], 1.0, places=12)
        assert_no_overlap(self, sizes([1] * 4, [1] * 4, r), (1024, 1024))

    def test_gap_holds_at_every_mip_level_including_unaligned_origins(self):
        for k, page in ((0, (257, 263)), (1, (259, 251)), (3, (509, 499))):
            w, h = random_boxes(30, 5 + k)
            r = uvpack.fill_page(w, h, page=page, k=k, passes=1)
            rects = sizes(w, h, r)
            assert_no_overlap(self, rects, page)
            clean, level = mip_clean(rects, page, k)
            self.assertTrue(clean, f"k {k}: boxes meet at mip level {level}")
            self.assertTrue(all(x % (1 << k) == 0 and y % (1 << k) == 0 for x, y, _, _ in rects), "boxes sit on block edges")

    def test_keep_going_never_lowers_space_used(self):
        w, h = random_boxes(40, 3)
        r = uvpack.fill_page(w, h, page=(1024, 1024), k=1, passes=1)
        for _ in range(3):
            nxt = uvpack.fill_page(w, h, page=(1024, 1024), k=1, order=r["order"], policy=r["policy"], scale=r["scale"], passes=1)
            self.assertGreaterEqual(nxt["used"], r["used"])
            r = nxt

    def test_fill_the_page_does_not_oscillate(self):
        w, h = random_boxes(25, 9)
        r = uvpack.fill_page(w, h, page=(512, 512), k=1, passes=1)
        for factor in (0.1, 0.5, 3.0, 40.0):  # a start far off on either side settles on a layout at least as good
            nxt = uvpack.fill_page(w, h, page=(512, 512), k=1, order=r["order"], policy=r["policy"], scale=r["scale"] * factor, passes=-1)
            # feasibility is not monotone in the scale, so a start far off may settle on a neighbouring plateau: within 1 %
            self.assertGreaterEqual(nxt["scale"], r["scale"] * 0.79, factor)
            self.assertIsNotNone(uvpack.pack(w, h, [True] * 25, nxt["order"], nxt["policy"], nxt["scale"], (512, 512), 1))
            r = nxt

    def test_pack_into_never_overlaps_the_existing_layout(self):
        seed = [10, 10, 90, 10, 10, 90, 90, 10, 90, 90, 10, 90]
        tris, charts = [], []
        for c in range(6):
            s = 10.0 + 4.0 * c
            tris += [0.0, 0.0, s, 0.0, 0.0, s]
            charts.append(c)
        placed = uvpack.pack_into((128, 128), 1, seed, tris, charts, 6, as_box=[0, 1, 0, 1, 0, 1], turn_ok=[1] * 6)
        self.assertTrue(all(p[0] for p in placed))
        taken = np.zeros((128, 128), bool)
        for y, rows in enumerate(uvpack.raster(seed, 10, 80), 10):
            for a, b in rows:
                taken[y, a:b] = True
        for c, (ok, dx, dy, q) in enumerate(placed):
            t = uvpack.turn_tris(tris[6 * c:6 * c + 6], q)
            t = [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(t)]
            y0 = int(min(t[1::2]) // 1)
            for y, rows in enumerate(uvpack.raster(t, y0, int(-(-max(t[1::2]) // 1)) - y0), y0):
                for a, b in rows:
                    self.assertTrue(0 <= a and b <= 128 and 0 <= y < 128)
                    self.assertFalse(taken[y, a:b].any(), f"chart {c} overlaps")
                    taken[y, a:b] = True

    def test_pack_into_keeps_the_block_gap_and_is_the_same_every_time(self):
        tris, charts, flags = random_charts(30, 5)
        for k in (0, 1, 2):
            first = uvpack.pack_into((256, 256), k, [], tris, charts, 30, as_box=flags, turn_ok=[1] * 30)
            self.assertEqual(first, uvpack.pack_into((256, 256), k, [], tris, charts, 30, as_box=flags, turn_ok=[1] * 30), "same input, same layout")
            self.assertTrue(all(p[0] for p in first), f"k {k}")
            self.assertEqual(block_gap_breaks(tris, charts, first, 256, k), [], f"k {k}: blocks two charts touch, grown by one block, must never meet")

    def test_a_small_chart_sits_in_the_bay_of_a_big_one(self):
        # an L: 80 x 80 minus the 60 x 60 corner; a 20 x 20 square has no room beside it on a 96 page, but fits in the notch
        big = [0, 0, 80, 0, 0, 20, 80, 0, 80, 20, 0, 20, 0, 20, 20, 20, 0, 80, 20, 20, 20, 80, 0, 80]
        small = [0, 0, 20, 0, 0, 20, 20, 0, 20, 20, 0, 20]
        placed = uvpack.pack_into((96, 96), 1, [], big + small, [0] * 4 + [1] * 2, 2, as_box=[0, 0], turn_ok=[0, 0])
        self.assertTrue(all(p[0] for p in placed))
        (_, bx, by, _), (_, sx, sy, _) = placed
        self.assertTrue(sx >= bx + 20 and sy >= by + 20 and sx + 20 <= bx + 80 and sy + 20 <= by + 80, "the small square sits in the notch of the L")
        self.assertEqual(block_gap_breaks(big + small, [0] * 4 + [1] * 2, placed, 96, 1), [])

    def test_slices_stay_short(self):
        w, h = random_boxes(60, 4)
        stage = uvpack.FillPage(w, h, page=(1024, 1024), k=1, passes=1, use_fast=False)
        longest = 0.0
        while not stage.done:
            t = time.perf_counter()
            stage.step()
            longest = max(longest, time.perf_counter() - t)
        self.assertIsNotNone(stage.result)
        self.assertLess(longest, 0.010, "a slice ran longer than 10 ms")


@needs_native
class Parity(unittest.TestCase):
    def test_fill_page_matches_the_twin(self):
        for n, k, page, seed in ((12, 1, (256, 256), 1), (30, 3, (509, 499), 2), (20, -1, (300, 200), 3)):
            w, h = random_boxes(n, seed)
            ref = uvpack.fill_page(w, h, page=page, k=k, passes=1)
            job = turbo.uvpack_fill_page(w, h, [True] * n, [1.0] * n, page[0], page[1], k, None, 0, 0.0, 1, 0.0)
            info, x, y, t, order = wait(job)
            self.assertAlmostEqual(info[0], ref["scale"], delta=1e-12 * ref["scale"], msg=f"scale n={n} k={k}")
            self.assertEqual([(int(a), int(b), bool(c)) for a, b, c in zip(x, y, t)], ref["places"])
            self.assertEqual([int(i) for i in order], ref["order"])
            self.assertAlmostEqual(info[1], ref["used"], places=12)

    def test_pack_into_matches_the_twin(self):
        seed = [10, 10, 90, 10, 10, 90, 90, 10, 90, 90, 10, 90]
        tris, charts = [], []
        for c in range(6):
            s = 10.0 + 4.0 * c
            tris += [0.5, 0.25, s + 0.5, 0.25, 0.5, s + 0.25]
            charts.append(c)
        flags = [0, 1, 0, 1, 0, 1]
        for k in (-1, 1, 2):
            ref = uvpack.pack_into((128, 128), k, seed, tris, charts, 6, as_box=flags, turn_ok=[1] * 6)
            ok, dx, dy, q = wait(turbo.uvpack_pack_into(seed, tris, charts, 6, flags, [1] * 6, 128, 128, k))
            self.assertEqual([(bool(a), int(b), int(c), int(d)) for a, b, c, d in zip(ok, dx, dy, q)], [(bool(a), b, c, d) for a, b, c, d in ref], f"k {k}")

    def test_pack_into_with_effort_matches_the_twin_when_it_fits_and_when_it_does_not(self):
        for n, page in ((30, 256), (40, 160)):  # roomy, then too tight for every chart
            tris, charts, flags = random_charts(n, 11 + n)
            for k in (-1, 1, 2):
                for effort in range(4):
                    ref = uvpack.pack_into((page, page), k + uvpack.EFFORT_STEP * effort, [], tris, charts, n, as_box=flags, turn_ok=[1] * n)
                    ok, dx, dy, q = wait(turbo.uvpack_pack_into([], tris, charts, n, flags, [1] * n, page, page, k + uvpack.EFFORT_STEP * effort))
                    self.assertEqual([(bool(a), int(b), int(c), int(d)) for a, b, c, d in zip(ok, dx, dy, q)], [(bool(a), b, c, d) for a, b, c, d in ref], f"n {n} k {k} effort {effort}")
        # effort never places fewer charts
        tris, charts, flags = random_charts(40, 51)
        counts = [sum(p[0] for p in uvpack.pack_into((160, 160), 1 + uvpack.EFFORT_STEP * e, [], tris, charts, 40, as_box=flags)) for e in range(4)]
        self.assertEqual(counts, sorted(counts))

    def test_free_turns_match_the_twin_keep_the_gap_and_fit_what_quarters_cannot(self):
        # two long strips lying at +30 and -30 degrees: quarter turns never make them parallel, turned to lie flat they fit
        tris, charts = [], []
        c30 = 0.8660254037844386
        for c, (L, W, s30) in enumerate(((150.0, 22.0, 0.5), (140.0, 20.0, -0.5))):
            pts = [(x * c30 - y * s30, x * s30 + y * c30) for x, y in ((0, 0), (L, 0), (L, W), (0, W))]
            for a, b, d in ((0, 1, 2), (0, 2, 3)):
                tris += [*pts[a], *pts[b], *pts[d]]
                charts.append(c)
        small, more, _ = random_charts(6, 5)
        tris += small
        charts += [c + 2 for c in more]
        n = 8
        quarters = uvpack.pack_into((160, 160), 1, [], tris, charts, n, turn_ok=[1] * n)
        self.assertFalse(all(p[0] for p in quarters), "quarter turns alone do not fit the strips")
        for effort in (0, 1):
            k = 1 + uvpack.EFFORT_STEP * effort + uvpack.FREE_TURNS
            ref = uvpack.pack_into((160, 160), k, [], tris, charts, n, turn_ok=[1] * n)
            self.assertTrue(all(p[0] for p in ref), f"free turns fit everything (effort {effort})")
            self.assertEqual(block_gap_breaks(tris, charts, ref, 160, 1), [])
            ok, dx, dy, q = wait(turbo.uvpack_pack_into([], tris, charts, n, [0] * n, [1] * n, 160, 160, k))
            self.assertEqual([(bool(a), int(b), int(c), int(d)) for a, b, c, d in zip(ok, dx, dy, q)], [(bool(a), b, c, d) for a, b, c, d in ref], f"effort {effort}")

    def test_every_policy_matches_the_twin_and_keeps_the_gap(self):
        # both placement rules (best short side fit, bottom-left) x every orientation policy, native against twin
        w, h = random_boxes(40, 12)
        page, k = (400, 400), 2
        scale = uvpack.fill_page(w, h, page=page, k=k)["scale"] * 0.7
        order = uvpack.make_order(w, h, [1.0] * 40, 0)
        for pol in range(uvpack.POLICIES):
            ref = uvpack.pack(w, h, [True] * 40, order, pol, scale, page, k)
            ok, x, y, t = wait(turbo.uvpack_pack_boxes(w, h, [True] * 40, order, pol, scale, page[0], page[1], k))
            self.assertTrue(ok and ref is not None, pol)
            self.assertEqual([(int(a), int(b), bool(c)) for a, b, c in zip(x, y, t)], ref, f"policy {pol}")
            rects = sizes(w, h, dict(places=ref, scale=scale))
            assert_no_overlap(self, rects, page)
            self.assertTrue(mip_clean(rects, page, k)[0], f"policy {pol}")

    def test_wrapper_never_blocks_the_caller(self):
        w, h = random_boxes(400, 6)
        t = time.perf_counter()
        job = turbo.uvpack_fill_page(w, h, [True] * 400, [1.0] * 400, 1024, 1024, 1, None, 0, 0.0, -1, 0.0)
        self.assertLess(time.perf_counter() - t, 0.010)
        job.stop()
        info, *_ = wait(job)  # a stop returns the best layout so far, not an error
        self.assertTrue(info[5] == 1.0 or info[6] == 0.0 or job.done())

    def test_wrapper_pack_boxes_and_stop(self):
        w, h = random_boxes(20, 8)
        order = uvpack.make_order(w, h, [1.0] * 20, 0)
        ok, x, y, t = wait(turbo.uvpack_pack_boxes(w, h, [True] * 20, order, 0, 1.0, 1024, 1024, 1))
        self.assertTrue(ok)
        self.assertEqual(len(x), 20)
        ok, *_ = wait(turbo.uvpack_pack_boxes(w, h, [True] * 20, order, 0, 1000.0, 1024, 1024, 1))
        self.assertFalse(ok)

    def test_native_keep_going_never_lowers(self):
        w, h = random_boxes(60, 21)
        info, x, y, t, order = wait(turbo.uvpack_fill_page(w, h, [True] * 60, [1.0] * 60, 1024, 1024, 1, None, 0, 0.0, 1, 0.0))
        for _ in range(3):
            nxt, x, y, t, order2 = wait(turbo.uvpack_fill_page(w, h, [True] * 60, [1.0] * 60, 1024, 1024, 1, order, int(info[2]), float(info[0]), 1, 0.0))
            self.assertGreaterEqual(nxt[1], info[1])
            info, order = nxt, order2


def smart_project_boxes(path):
    """Island boxes (width, height in UV units) of a P0 Smart Project result."""

    import sys

    sys.path.insert(0, str(BENCH))
    import uvstats

    m = uvstats.load(str(path))
    FP, FT, tf = uvstats.tris(m)
    U = m["cuv"][FT]
    lab = uvstats._topo(m["P"], m["cuv"], FP, FT)["lab"]
    n = lab.max() + 1
    lo, hi = np.full((n, 2), np.inf), np.full((n, 2), -np.inf)
    for c in range(3):
        np.minimum.at(lo, lab, U[:, c])
        np.maximum.at(hi, lab, U[:, c])
    size = hi - lo
    return size[:, 0], size[:, 1]


@needs_native
class SmartProjectBoxes(unittest.TestCase):
    def test_space_used_at_four_px(self):
        files = sorted((BENCH / "_p0" / "out").glob("*_smart.obj"))
        if not files:
            self.skipTest("P0's Smart Project results are not on this machine")
        report = {}
        for path in files:
            w, h = smart_project_boxes(path)
            keep = (w > 0) & (h > 0)
            w, h, n = w[keep], h[keep], int(keep.sum())
            t0 = time.perf_counter()
            info, x, y, t, order = wait(turbo.uvpack_fill_page(w, h, np.ones(n, bool), np.ones(n), 1024, 1024, 1, None, 0, 0.0, 0, 0.0))
            t_first = time.perf_counter() - t0
            self.assertEqual(info[6], 1.0, path.name)
            rects = []
            for i in range(n):
                a, b = uvpack.snap(w[i] * info[0]), uvpack.snap(h[i] * info[0])
                rects.append((int(x[i]), int(y[i]), b, a) if t[i] else (int(x[i]), int(y[i]), a, b))
            self.assertTrue(no_overlap_raster(rects, (1024, 1024)), path.name)
            mip, level = mip_clean(rects, (1024, 1024), 1)
            self.assertTrue(mip, f"{path.name}: boxes meet at mip level {level}")
            # Keep going from that layout (one swap pass, 5 s budget): never lower
            t0 = time.perf_counter()
            more, *_ = wait(turbo.uvpack_fill_page(w, h, np.ones(n, bool), np.ones(n), 1024, 1024, 1, order, int(info[2]), float(info[0]), 1, 5.0))
            t_more = time.perf_counter() - t0
            self.assertGreaterEqual(more[1], info[1])
            # what no packer can beat at this gap: every box perfectly tiled with its 2-texel margin (k = 1, 4 px between boxes)
            cw = np.maximum(np.ceil(w * info[0] - 1e-9), 1)
            ch = np.maximum(np.ceil(h * info[0] - 1e-9), 1)
            bound = float((cw * ch).sum() / ((cw + 4) * (ch + 4)).sum())
            free, *_ = wait(turbo.uvpack_fill_page(w, h, np.ones(n, bool), np.ones(n), 1024, 1024, -1, None, 0, 0.0, 0, 0.0))  # the same boxes with no gap
            # and what no packer can beat when every box also fills whole mip blocks (a box ends on a block edge, so an odd size
            # wastes a texel before its neighbour): the same tiling with each side rounded up to a multiple of the block
            ew = np.ceil(cw / 2) * 2
            eh = np.ceil(ch / 2) * 2
            aligned = float((cw * ch).sum() / ((ew + 4) * (eh + 4)).sum())
            report[path.name] = (n, round(float(info[1]), 4), round(float(more[1]), 4), round(min(bound, float(free[1])), 4), bool(more[4]),
                                 round(aligned, 4), round(t_first, 2), round(t_more, 2))
        print("space used at 4 px, 1024 page (islands, first fill, after one Keep going pass, bound = lower of the gap-only bound and the no-gap result, budget hit, "
              "block-rounded bound, seconds of the first fill, seconds of the Keep going pass):", report)
        for name, v in report.items():
            print(f"  {name:40} {v[0]:5} islands  first {100 * v[1] / v[3]:5.1f} %  Keep going {100 * v[2] / v[3]:5.1f} % of the bound  ({100 * v[2] / v[5]:5.1f} % of the block-rounded bound)  {v[6]:5.2f} s + {v[7]:5.2f} s")
        # The gate: 95 % of the bound (SPEC 14, K7). The block-rounded bound is a true upper bound, so where it is itself below 95 % of
        # the raw bound (many tiny islands of odd size) the raw bound cannot be reached by any packer and the packer must reach 98 % of
        # the block-rounded one. Proof: mip-clean means the blocks two boxes touch, each grown by one block, never meet. A box of
        # w x h texels touches at least ceil(w / 2) x ceil(h / 2) blocks whatever its origin, so with its growth it owns at least
        # (ceil(w / 2) + 2) x (ceil(h / 2) + 2) blocks, i.e. (ew + 4) x (eh + 4) texels, and these owned sets are disjoint. They fit in the
        # page plus the growth ring that may hang over its edge (one block each side: (1024 + 4)^2 texels), so
        # used <= aligned * (1028 / 1024)^2. Checked below on every set.
        for name, (n, first, more, bound, _, aligned, *_) in report.items():
            self.assertGreaterEqual(more, first)
            self.assertLessEqual(more, aligned * (1028 / 1024) ** 2 + 1e-9, f"{name}: above the block-rounded bound, so the proof or the packer is wrong")
            self.assertTrue(more >= 0.95 * bound or more >= 0.98 * aligned, f"{name}: {more} (bound {bound}, block-rounded {aligned})")
        self.assertGreaterEqual(sum(v[2] >= 0.95 * v[3] for v in report.values()), 17)

if __name__ == "__main__":
    unittest.main()
