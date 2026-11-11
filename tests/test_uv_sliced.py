# SPDX-License-Identifier: GPL-3.0-or-later
"""The Apply UV steps in timer slices: the bake unwrap stepped in short pieces gives the same UVs as the old one-call
run, and a Classic transfer stopped half-way hands back to Blender's transfer (None)."""

from __future__ import annotations

import unittest

import numpy as np

import uv_common  # noqa: F401  (puts the add-on on the path)
from test_uv_classic import Obj, two_islands
from uv_common import cube

from triadq import uvclassic, uvnew


class Sliced(unittest.TestCase):
    def test_stepped_unwrap_matches_one_call(self):
        mesh = cube(6)
        settings = uvnew.Settings(out_of_sight=False, hidden_pct=100.0, texture_px=256, time_per_pass=1.0)
        whole = uvnew.NewUVs(*mesh, settings=settings, budget=1e9).run()  # what the bake did before
        run = uvnew.NewUVs(*mesh, settings=settings)  # what uvbake.ensure_target_uvs_steps does now
        run._whole = True
        steps = 0
        while not run.done:
            run.step()
            steps += 1
        self.assertGreater(steps, 1)
        self.assertTrue(np.array_equal(run.result.uv, whole.uv))

    def test_stopped_transfer_hands_back(self):
        points, indptr, flat, uv = two_islands()
        source = Obj(points, indptr, flat, {"A": uv.reshape(-1, 2)})
        target = Obj(points, indptr, flat, {})
        config = type("C", (), {"texture_mode": "PROJECT"})()
        asked = []
        steps = uvclassic.preserve_textures_steps(source, target, config, stopped=lambda: asked.append(1) or len(asked) > 2)
        with self.assertRaises(StopIteration) as end:
            while True:
                next(steps)
        self.assertIsNone(end.exception.value)


if __name__ == "__main__":
    unittest.main()
