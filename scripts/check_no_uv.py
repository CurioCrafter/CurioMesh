# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Fail when a tree carries UV parts this Blender-site build leaves out (exit 1 on any hit).

Usage: python scripts/check_no_uv.py [ROOT]   (ROOT defaults to the repo root)

The site build includes New UVs, New page layout, Cut here, Bake those spots, the bake's own unwrap, Classic Remesh's
per-side UV transfer and the UVs popover with Seam Fit Exact / Off. It leaves out Seam Fit Light (the fit after the
search, its Look First tool, its Auto Fix passes and its Apply work-ahead) and the compiled UV kernels.
Checks: a left-out UV file, any marker line or marked block, the names of the left-out parts.
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
    re.compile(rb"\buv" + rb"keep\b|\buv" + rb"snap\b|\buv" + rb"auto\b|\buv" + rb"_hooks\b|\buv" + rb"_fast\b"),
    re.compile(rb"\buv" + rb"_seam_fit\b|\bSeam" + rb"Fit\b|\bSeam" + rb"Snap\b"),
    re.compile(rb"Seam Fit: " + rb"Light"),
]
# Paths relative to the root: the left-out modules and the compiled kernels.
UV_FILES = [
    "native/src/uv*.rs", "standalone/src/uv*.rs", "standalone/src/gui_uv.rs", "standalone/src/unwrap.rs",
    *(f"triadq/uv{name}.py" for name in ("keep", "snap", "auto")), *(f"uv_{name}.py" for name in ("fast", "hooks")),
    "tests/test_uv_parity_*.py", "tests/uv_parity_common.py", *(f"tests/test_uv_{name}.py" for name in ("keep", "auto", "scaffold")),
    "tests/uv_hooks_fake.py",
]
SKIP_DIRS = {".git", "__pycache__", ".claude", ".ruff_cache", "target", "dist"}


def check(root):
    root = Path(root).resolve()
    me = Path(__file__).resolve()
    problems = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if path.resolve() == me or rel == "tests/test_check_no_uv.py":
                continue
            if any(fnmatch.fnmatchcase(rel, p) for p in UV_FILES):
                problems.append(f"{rel}: UV file left out of this build")
                continue
            if rel.startswith(("native/", "standalone/")):
                continue  # the GitHub build's compiled sources: never in the zip, not this build's code
            try:
                data = path.read_bytes()
            except OSError:
                continue
            for pat in PATTERNS:
                m = pat.search(data)
                if m:
                    line = data.count(b"\n", 0, m.start()) + 1
                    problems.append(f"{rel}:{line}: {m.group().decode(errors='replace')}")
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
