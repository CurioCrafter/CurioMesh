# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake those spots: Ctrl+Z gives float (HDR) texture pixels back exactly, and each image is baked with the data it
holds (normal only into normal maps)."""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np

import uv_parity_common as common

uvbake = common.addon_module("triadq.uvbake")


class FakePixels:
    def __init__(self, values):
        self.values = np.asarray(values, np.float32).reshape(-1)

    def foreach_get(self, out):
        out[:] = self.values

    def foreach_set(self, values):
        self.values = np.asarray(values, np.float32).reshape(-1).copy()


class FakeImage:
    def __init__(self, name, values, is_float):
        self.name, self.size, self.is_float = name, (2, 1), is_float
        self.pixels = FakePixels(values)

    def update(self):
        pass


class FloatUndo(unittest.TestCase):
    def test_hdr_pixels_come_back_exactly(self):
        hdr = [4.5, -0.25, 0.3333, 1.0, 12.0, 0.5, 0.0, 1.0]
        image = FakeImage("hdr", hdr, is_float=True)
        scene: dict = {}
        fake = types.ModuleType("bpy")
        fake.data = types.SimpleNamespace(images={"hdr": image})
        old = sys.modules.get("bpy")
        sys.modules["bpy"] = fake
        try:
            uvbake._states.clear(), uvbake._applied.clear()
            uvbake.remember_images(scene, [image], before=True)
            image.pixels.foreach_set(np.zeros(8))
            uvbake.remember_images(scene, [image])
            scene[uvbake.HISTORY_KEY] = 0  # Ctrl+Z
            self.assertEqual(uvbake.restore(scene), ["hdr"])
            np.testing.assert_array_equal(image.pixels.values, np.asarray(hdr, np.float32))
        finally:
            uvbake._states.clear(), uvbake._applied.clear()
            if old is None:
                sys.modules.pop("bpy", None)
            else:
                sys.modules["bpy"] = old


if __name__ == "__main__":
    unittest.main()
