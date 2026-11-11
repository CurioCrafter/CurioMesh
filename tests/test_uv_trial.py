# SPDX-License-Identifier: GPL-3.0-or-later
"""UV Trials (A3): the UV-only Trials (New, New page layout, Bake those spots) give valid layouts in slices, Keep going
never lowers space used."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_common as common

UT = common.addon_module("triadq.uvtrial")
uvseams = common.addon_module("triadq.uvseams")
uvscore = common.addon_module("triadq.uvscore")
uvnew = common.addon_module("triadq.uvnew")


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
    return p, ip, fl, uv


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
    def test_a_dropped_run_stops(self):
        """A drag, Again or Esc drops the preview's generator mid-run: the run stops, so no worker job is left going."""
        p, ip, fl = cube_mesh(4)
        run = uvnew.NewUVs(p, ip, fl, settings=uvnew.Settings(out_of_sight=False, hidden_pct=100.0), budget=UT.NEW_SLICE)
        work = UT._poll_new(run)
        for _ in range(3):
            next(work)
        work.close()
        self.assertTrue(run._stop)

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
        sp, si, sf, suv = two_island_source()
        p, ip, fl = grid(5)  # edges at 0.2 steps: the seam at 0.5 runs through the middle column
        uv = uvseams.transfer(sp, si, sf, suv.ravel(), p, ip, fl)["uv"]  # UVs carried over side by side, no fit
        src = uvscore.mesh(sp, si, sf, suv, np.zeros(len(si) - 1, np.int64))
        base = UT.make_uv_base(p, ip, fl, uv=uv, texture_px=1024, source=src)
        trial = UT.BakeSpotsUV()
        self.assertEqual(trial.check(base), "")
        cand, _, _ = run_sliced(trial.run(base, dict(trial.defaults)))
        self.assertFalse(cand.refusal, cand.refusal)
        straddle = UT.finish(UT.straddlers(base))
        self.assertEqual(set(cand.faces.tolist()), set(np.flatnonzero(straddle).tolist()))
        lab, _ = UT.islands(ip, fl, cand.uv)
        others = set(lab[~straddle].tolist())
        self.assertFalse(others & set(lab[cand.faces].tolist()))  # the new islands are their own
        cand, _, _ = run_sliced(UT.measure(trial, base, cand))
        self.assertLess(cand.numbers["overlap_pct"], 0.01 + base.cache["numbers"]["overlap_pct"])
        # the bake counts only faces that show a texel: on a 1-pixel page none does
        tiny = UT.make_uv_base(p, ip, fl, uv=uv, texture_px=1, source=src)
        self.assertTrue(UT.finish(UT.straddlers(tiny)).any())
        self.assertFalse(UT.shows_texture(tiny).any())


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
