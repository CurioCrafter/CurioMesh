# SPDX-License-Identifier: GPL-3.0-or-later
"""Classic Remesh (A1c): the per-side transfer keeps each island's UVs, on every layer."""

from __future__ import annotations

import unittest

import numpy as np

import uv_common as common  # noqa: F401  (puts the add-on folder on the path)
from uv_common import two_islands
from triadq import uvclassic


def run(steps):
    while True:
        try:
            next(steps)
        except StopIteration as end:
            return end.value


class Obj:
    """Stand-in for a Blender mesh object built from polygons."""

    def __init__(self, points, indptr, flat, layers):
        cols = {"co": points.ravel(), "loop_total": np.diff(indptr), "vertex_index": flat, "material_index": np.zeros(len(indptr) - 1, np.int32)}

        class Seq(list):
            def foreach_get(self, key, out):
                out[:] = cols[key]

        class Layer:
            def __init__(self, name, uv):
                self.name, self.data, self.values = name, self, np.array(uv, np.float32).ravel()

            def foreach_get(self, key, out):
                out[:] = self.values

            def foreach_set(self, key, values):
                self.values = np.array(values, np.float32)

        class Layers(list):
            active = None

            def get(self, name):
                return next((x for x in self if x.name == name), None)

            def new(self, name):
                self.append(Layer(name, np.zeros((len(flat), 2))))
                return self[-1]

        mesh = type("Mesh", (), {})()
        mesh.vertices, mesh.polygons, mesh.loops = Seq(range(len(points))), Seq(range(len(indptr) - 1)), Seq(range(len(flat)))
        mesh.uv_layers = Layers(Layer(name, uv) for name, uv in layers.items())
        mesh.uv_layers.active = mesh.uv_layers[0] if mesh.uv_layers else None
        self.data, self.matrix_world = mesh, None

    def __setitem__(self, key, value):
        pass


class Transfer(unittest.TestCase):
    def test_island_aware_drift_and_all_layers(self):
        points, indptr, flat, uv = two_islands()
        res = points.copy()
        res[:, 0] = 4 + (res[:, 0] - 4) * 1.01
        res[:, 2] = 0.2
        source = Obj(points, indptr, flat, {"A": uv.reshape(-1, 2), "B": uv.reshape(-1, 2) * 0.5})
        target = Obj(res, indptr, flat, {})
        config = type("C", (), {"texture_mode": "PROJECT"})()
        self.assertEqual(run(uvclassic.preserve_textures_steps(source, target, config)), "TRANSFERRED")
        right = np.repeat((np.arange(64) % 8) >= 4, 4)
        want = np.clip(res[flat, 0], 0, 8) / 8 + 0.25 * right
        got = target.data.uv_layers.get("A").values.reshape(-1, 2)
        self.assertLess(float(np.percentile(np.abs(got[:, 0] - want), 99)) * 1024, 8.0)
        self.assertEqual(target.data.uv_layers.active.name, "A")
        b = target.data.uv_layers.get("B").values.reshape(-1, 2)
        self.assertLess(float(np.abs(b - got * 0.5).max()) * 1024, 8.0)

    def test_without_uvs_defers_else_transfers(self):
        points, indptr, flat, uv = two_islands()
        source = Obj(points, indptr, flat, {"A": uv.reshape(-1, 2)})
        target = Obj(points, indptr, flat, {})
        config = type("C", (), {"texture_mode": "PROJECT"})()
        self.assertIsNone(run(uvclassic.preserve_textures_steps(Obj(points, indptr, flat, {}), target, config)))  # no source UVs
        config.texture_mode = "BAKE"
        self.assertIsNone(run(uvclassic.preserve_textures_steps(source, target, config)))  # not a transfer mode
        config.texture_mode = "PROJECT"
        self.assertEqual(run(uvclassic.preserve_textures_steps(source, target, config)), "TRANSFERRED")
        self.assertTrue(np.isfinite(target.data.uv_layers.get("A").values).all())

if __name__ == "__main__":
    unittest.main()
