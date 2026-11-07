# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Fail when a tree carries the GitHub-only UV features (the Blender-site build must not).

Usage: python scripts/check_no_uv.py [ROOT]   (ROOT defaults to the repo root)
Checks: any UV file, any marker line or marked block, a UV property group, the UV popover.
Search strings are built from parts and this file is skipped, so it passes on itself.
"""
import fnmatch
import os
import re
import sys
from pathlib import Path

MARK = "main-only" + ": uv"
PATTERNS = [
    re.compile(re.escape(MARK).encode()),
    re.compile(rb"\bcuriomesh" + rb"_uv\b"),
    re.compile(rb"\bCURIOMESH_PT_pop" + rb"_uv\b"),
]
# Paths relative to the root; triadq/uvs.py is the old, shipped module and is allowed.
UV_FILES = [
    "native/src/uv*.rs", "standalone/src/uv*.rs", "standalone/src/gui_uv.rs", "standalone/src/unwrap.rs",
    "triadq/uv*.py", "uv_layers.py", "uv_panel.py", "uv_draw.py", "uv_fast.py", "uv_hooks.py",
    "tests/test_uv_*.py", "tests/uv_*.py", "tests/blender_uv_*.py",
]
ALLOWED = {"triadq/uvs.py", "tests/test_uvs.py"}
SKIP_DIRS = {".git", "__pycache__", ".claude", "target"}


def check(root):
    root = Path(root).resolve()
    me = Path(__file__).resolve()
    problems = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if path.resolve() == me or rel in ALLOWED:
                continue
            if any(fnmatch.fnmatchcase(rel, p) for p in UV_FILES):
                problems.append(f"{rel}: UV file")
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            for pat in PATTERNS:
                m = pat.search(data)
                if m:
                    line = data.count(b"\n", 0, m.start()) + 1
                    problems.append(f"{rel}:{line}: {m.group().decode()}")
                    break
    return problems


def main(argv):
    root = argv[1] if len(argv) > 1 else Path(__file__).resolve().parents[1]
    problems = check(root)
    for p in problems:
        print(p)
    print("UV guard: FAIL (%d)" % len(problems) if problems else "UV guard: ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
