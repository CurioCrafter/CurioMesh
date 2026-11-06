# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Entry script of a helper Blender, run with `--python` by `parallel.helper_command`. The add-on never imports it.

After "--" the command line holds the add-on's package, a module and a function of it, then that function's own
arguments. Only the two helper entries in ENTRIES can be named. The script enables CurioMesh by its package name, the
way Blender loads it, and calls the function with Blender's argv minus those three words. No code is built at runtime.
"""

import importlib
import sys

import addon_utils

at = sys.argv.index("--") + 1
package, module, function = sys.argv[at : at + 3]
ENTRIES = {("parallel", "worker_main"), ("rows_worker", "main")}
if (module, function) not in ENTRIES:
    raise SystemExit(f"not a CurioMesh helper entry: {module}.{function}")
addon_utils.enable(package, default_set=False)
getattr(importlib.import_module(f"{package}.{module}"), function)(sys.argv[:at] + sys.argv[at + 3 :])
