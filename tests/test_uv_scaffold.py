# SPDX-License-Identifier: GPL-3.0-or-later
"""UV scaffold (K0): constants parity, marked blocks, banned words in the twins, worker-job shape, layer helpers."""

from __future__ import annotations

import importlib
import re
import threading
import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import KERNELS, ROOT, TWINS, needs_native

import turbo


class Layer:
    def __init__(self, name, n):
        self.name, self.data = name, self
        self.values = np.zeros(n * 2, dtype=np.float32)

    def foreach_get(self, key, out):
        out[:] = self.values

    def foreach_set(self, key, values):
        self.values = np.array(values, dtype=np.float32)


class Layers(list):
    active = None

    def get(self, name):
        return next((layer for layer in self if layer.name == name), None)

    def new(self, name):
        layer = Layer(name, self.n)
        self.append(layer)
        self.active = self.active or layer
        return layer


class Mesh:
    def __init__(self, n):
        self.loops = range(n)
        self.uv_layers = Layers()
        self.uv_layers.n = n


class Scaffold(unittest.TestCase):
    def test_constants_match_between_rust_and_twin(self):
        for kernel, twin in TWINS.items():
            module = importlib.import_module(f"triadq.{twin}")
            for name, value in common.rs_constants(kernel).items():
                self.assertEqual(getattr(module, name), value, f"{twin}.{name}")

    def test_twins_avoid_banned_words(self):
        for twin in TWINS.values():
            text = (ROOT / "triadq" / f"{twin}.py").read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"turbo|native|rust|gpu", text, re.I), twin)

    def test_every_kernel_has_balanced_blocks(self):
        for path in ("native/src/lib.rs", "turbo.py", "uv_fast.py"):
            text = (ROOT / path).read_text(encoding="utf-8")
            for kernel in KERNELS:
                self.assertEqual(len(re.findall(rf"main-only: uv begin {kernel}\s", text)), 1, f"{path} {kernel} begin")
                self.assertEqual(len(re.findall(rf"main-only: uv end {kernel}\s", text)), 1, f"{path} {kernel} end")
        lib = (ROOT / "native/src/lib.rs").read_text(encoding="utf-8")
        for kernel in KERNELS:
            self.assertIn(f"register_uv{kernel}(m)?;", lib)

    def test_job_never_blocks_and_hands_back_result_or_error(self):
        gate = threading.Event()
        started = time.perf_counter()
        job = turbo.UvJob(lambda stop: (gate.wait(5), 42)[1])
        self.assertLess(time.perf_counter() - started, 0.01)
        self.assertFalse(job.done())
        with self.assertRaises(RuntimeError):
            job.result()
        gate.set()
        job._thread.join(5)
        self.assertTrue(job.done())
        self.assertEqual(job.result(), 42)
        bad = turbo.UvJob(lambda stop: 1 / 0)
        bad._thread.join(5)
        with self.assertRaises(ZeroDivisionError):
            bad.result()

    def test_job_stop_sets_the_flag(self):
        class Flag:
            hit = False

            def set(self):
                self.hit = True

        flag = Flag()
        job = turbo.UvJob(lambda stop: None, stop=flag)
        job.stop()
        self.assertTrue(flag.hit)

    def test_wrappers_return_none_without_the_module(self):
        turbo.set_enabled(False)
        try:
            self.assertIsNone(turbo.uvseam_probe())
            self.assertIsNone(turbo.uv_job("uvseam_probe"))
        finally:
            turbo.set_enabled(True)

    def test_layer_helpers_round_trip(self):
        import uv_layers

        mesh = Mesh(4)
        self.assertIsNone(uv_layers.read_uv(mesh))
        self.assertIsNone(uv_layers.write_uv(mesh, np.zeros((4, 2)), create=False))
        uv = np.arange(8, dtype=np.float64).reshape(4, 2) / 8
        self.assertEqual(uv_layers.write_uv(mesh, uv), "UVMap")
        self.assertEqual(uv_layers.layer_names(mesh), ["UVMap"])
        np.testing.assert_allclose(uv_layers.read_uv(mesh), uv, atol=1e-7)
        with self.assertRaises(ValueError):
            uv_layers.write_uv(mesh, np.zeros((3, 2)))


@needs_native
class Probes(unittest.TestCase):
    def test_every_compile_probe_passes_through_the_binding(self):
        for kernel in KERNELS:
            self.assertEqual(getattr(turbo, f"uv{kernel}_probe")(), 1, kernel)

    def test_adapter_installs_and_removes(self):
        uv_fast = common.addon_module("uv_fast")
        uvseams = common.addon_module("triadq.uvseams")

        self.assertTrue(uv_fast.install())
        self.assertEqual(uvseams.FAST["probe"](), 1)
        uv_fast.uninstall()
        self.assertEqual(uvseams.FAST, {})


if __name__ == "__main__":
    unittest.main()
