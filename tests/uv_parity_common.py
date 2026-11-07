# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared loader and skip for the per-kernel UV parity tests (`tests/test_uv_parity_<kernel>.py`).

`native_module()` returns the compiled kernel module or None; `needs_native` skips a test class when it is missing.
`rs_constants(kernel)` reads the `pub const NAME: type = value;` table at the top of `native/src/<kernel>.rs`.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import turbo  # noqa: E402

KERNELS = ("seam", "snap", "qem", "score", "sight", "flat", "chart", "pack")
TWINS = {"seam": "uvseams", "snap": "uvsnap", "score": "uvscore", "sight": "uvsight", "flat": "uvflat", "chart": "uvchart", "pack": "uvpack"}


def addon_module(name: str):
    """Import `name` (e.g. "uv_fast") from the add-on folder as part of a bare package, so its relative imports work without bpy."""

    import importlib
    import types

    if "uv_addon_pkg" not in sys.modules:
        package = types.ModuleType("uv_addon_pkg")
        package.__path__ = [str(ROOT)]
        sys.modules["uv_addon_pkg"] = package
    return importlib.import_module(f"uv_addon_pkg.{name}")


def native_module():
    return turbo.native()


needs_native = unittest.skipIf(native_module() is None, "native module not built")


def rs_constants(kernel: str) -> dict:
    """`{NAME: number}` for the numeric `pub const` lines of native/src/uv<kernel>.rs."""

    text = (ROOT / "native" / "src" / f"uv{kernel}.rs").read_text(encoding="utf-8")
    found = re.findall(r"^pub const (\w+): (?:f64|f32|usize|u32|u64|i32|i64) = ([-+0-9._e]+);", text, re.M)
    return {name: float(value.replace("_", "")) for name, value in found}
