# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Test helper: load an add-on module that uses relative imports, without the add-on's __init__ (it needs bpy).

The module is loaded as part of a stand-in package that shares the top-level `triadq` and `guides` the other tests
import, so the types they pass around are the same objects. Only the tests do this; the add-on never touches sys.
"""

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    if "addon" not in sys.modules:
        import guides
        import triadq

        for sub in ("issues", "parts", "patch", "segment"):
            importlib.import_module("triadq." + sub)
        package = types.ModuleType("addon")
        package.__path__ = [str(ROOT)]
        package.triadq, package.guides = triadq, guides
        sys.modules.update({"addon": package, "addon.triadq": triadq, "addon.guides": guides})
        for sub in ("issues", "parts", "patch", "segment"):
            sys.modules["addon.triadq." + sub] = sys.modules["triadq." + sub]
    return importlib.import_module("addon." + name)
