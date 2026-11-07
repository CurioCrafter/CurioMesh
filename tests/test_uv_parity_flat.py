# SPDX-License-Identifier: GPL-3.0-or-later
"""Flatten (K5): the NumPy twin on its own, then twin against the compiled kernel (parity) and the job wrapper."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from triadq import uvflat

import turbo


def grid(nx, ny, h=1.0, z=lambda x, y: 0.0 * x):
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    x, y = i.ravel() * h, j.ravel() * h
    points = np.stack([x, y, z(x, y)], 1)
    ids = (j * (nx + 1) + i)[:-1, :-1].ravel()
    flat = np.stack([ids, ids + 1, ids + nx + 2, ids + nx + 1], 1).ravel()
    return points, np.arange(0, len(flat) + 1, 4), flat


def curved():
    return grid(10, 6, 0.3, lambda x, y: np.sin(0.8 * x) + 0.1 * np.cos(y))


class Twin(unittest.TestCase):
    def test_flat_grid_unchanged(self):
        points, indptr, flat = grid(8, 8, 0.5)
        uv, col, row = uvflat.grid_layout(points, indptr, flat)
        np.testing.assert_allclose(uv, points[:, :2], atol=1e-12)
        for rows in ("off", "soft", "on"):
            out, numbers = uvflat.flatten_chart(points, indptr, flat, uv, col=col, row=row, rows=rows, steps=5, use_fast=False)
            np.testing.assert_allclose(out, uv, atol=1e-9, err_msg=rows)

    def test_folded_start_ends_flip_free_in_short_slices(self):
        points, indptr, flat = grid(8, 8, 1.0, lambda x, y: 0.2 * np.sin(0.5 * x) * np.cos(0.4 * y))
        start = points[:, :2].copy()
        start[start[:, 0] > 4, 0] = 8 - start[start[:, 0] > 4, 0]
        self.assertGreater(uvflat.count_flips(indptr, flat, start), 0)
        stage = uvflat.Flatten(points, indptr, flat, start, rows="off", steps=30, use_fast=False)
        longest = 0.0
        while not stage.done:
            t = time.perf_counter()
            stage.step()
            longest = max(longest, time.perf_counter() - t)
        self.assertEqual(stage.numbers["flips"], 0)
        self.assertEqual(stage.numbers["start"], "tutte")
        self.assertLess(longest, 0.010, "a slice ran longer than 10 ms")

    def test_on_keeps_rows_on_one_value(self):
        points, indptr, flat = curved()
        uv, col, row = uvflat.grid_layout(points, indptr, flat)
        out, numbers = uvflat.flatten_chart(points, indptr, flat, uv, col=col, row=row, rows="on", steps=10, use_fast=False)
        self.assertEqual(numbers["rows"], "on")
        for labels, axis in ((col, 0), (row, 1)):
            for k in np.unique(labels):
                self.assertEqual(np.ptp(out[labels == k, axis]), 0.0)

    def test_pole_is_not_a_grid(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 0, 0], [-0.5, -1, 0], [0.5, -1, 0]], dtype=float)
        self.assertIsNone(uvflat.grid_layout(points, [0, 4, 8, 12], [0, 1, 2, 3, 0, 3, 4, 5, 0, 5, 6, 1]))


@needs_native
class Parity(unittest.TestCase):
    def run_job(self, job):
        self.assertIsNotNone(job)
        job._thread.join(60)
        return job.result()

    def test_grid_layout_and_stretch_match(self):
        points, indptr, flat = curved()
        uv, col, row = uvflat.grid_layout(points, indptr, flat)
        kuv, kcol, krow = self.run_job(turbo.uvflat_grid_layout(points, indptr, flat))
        np.testing.assert_allclose(kuv.reshape(-1, 2), uv, atol=1e-12)
        np.testing.assert_array_equal(kcol, col)
        np.testing.assert_array_equal(krow, row)
        bent = uv + 0.05 * np.sin(uv[:, ::-1] * 3)
        q, per = uvflat.stretch(points, indptr, flat, bent)
        kq, kper = self.run_job(turbo.uvflat_stretch(points, indptr, flat, bent))
        self.assertAlmostEqual(q, kq, places=6)
        np.testing.assert_allclose(kper, per, atol=1e-6)

    def test_arap_twin_matches_kernel_within_1e3_of_chart_size(self):
        points, indptr, flat = curved()
        uv, col, row = uvflat.grid_layout(points, indptr, flat)
        size = float(np.linalg.norm(np.ptp(uv, axis=0)))
        for rows in ("off", "on"):
            twin, numbers = uvflat.flatten_chart(points, indptr, flat, uv, col=col, row=row, rows=rows, steps=20, use_fast=False)
            chart = np.zeros(len(indptr) - 1, np.int64)
            k_uv, stats = self.run_job(turbo.uvflat_flatten(points, indptr, flat, chart, uv[flat].ravel(), col[flat], row[flat], rows=uvflat.ROWS[rows], steps=20))
            kernel = np.full_like(uv, np.nan)
            kernel[flat] = k_uv.reshape(-1, 2)
            gap = float(np.abs(kernel - twin).max()) / size
            self.assertLess(gap, 1e-3, rows)
            self.assertEqual(stats[4], uvflat.ROWS[rows])
            self.assertEqual(stats[7], 0)
            self.assertEqual(numbers["flips"], 0)

    def test_stage_uses_the_fast_callable(self):
        uv_fast = common.addon_module("uv_fast")
        fast_flat = common.addon_module("triadq.uvflat")
        self.assertTrue(uv_fast.install())
        try:
            points, indptr, flat = curved()
            uv, col, row = fast_flat.grid_layout(points, indptr, flat)
            out, numbers = fast_flat.flatten_chart(points, indptr, flat, uv, col=col, row=row, rows="auto", steps=5)
            self.assertIn(numbers["rows"], ("on", "soft"))
            self.assertEqual(numbers["flips"], 0)
            self.assertTrue(np.isfinite(out).all())
        finally:
            uv_fast.uninstall()

    def test_wrapper_never_blocks(self):
        points, indptr, flat = grid(300, 300, 0.01, lambda x, y: 0.2 * np.sin(3 * x) * np.cos(2 * y))
        chart = np.zeros(len(indptr) - 1, np.int64)
        start = np.full(2 * len(flat), np.nan)
        t = time.perf_counter()
        job = turbo.uvflat_flatten(points, indptr, flat, chart, start, [], [], rows=0, start_mode=2, steps=3)
        took = time.perf_counter() - t
        self.assertLess(took, 0.010, f"wrapper blocked {took * 1e3:.1f} ms")
        job.stop()
        uv, stats = self.run_job(job)
        self.assertEqual(len(stats), turbo.native().UVFLAT_STATS_LEN)


if __name__ == "__main__":
    unittest.main()
