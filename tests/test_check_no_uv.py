# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_no_uv  # noqa: E402

MARK = "main-only" + ": uv"


def tree(files):
    d = Path(tempfile.mkdtemp())
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    return d


class GuardTests(unittest.TestCase):
    def test_clean_and_dirty_trees(self):
        kept = {"triadq/uvs.py": "x", "triadq/uvnew.py": "x", "uv_panel.py": "curiomesh_uv = 1", "bridge.py": "curiomesh_uv_status"}
        self.assertEqual(check_no_uv.check(tree(kept)), [])
        for rel, text in [("uv_fast.py", ""), ("triadq/uvkeep.py", ""), ("native/src/uvpack.rs", ""), ("a.py", "x  # " + MARK + "\n"),
                          ("b.md", "<!-- " + MARK + " begin -->"), ("c.py", "from .triadq import uv" + "keep"),
                          ("d.py", "bpy.ops.curiomesh.uv" + "_seam_fit()"), ("e.py", "Seam Fit: " + "Light")]:
            self.assertEqual(len(check_no_uv.check(tree({rel: text}))), 1, rel)

    def test_passes_on_itself(self):
        src = Path(check_no_uv.__file__).read_text()
        self.assertEqual(check_no_uv.check(tree({"scripts/check_no_uv.py": src, "tests/test_check_no_uv.py": Path(__file__).read_text()})), [])


if __name__ == "__main__":
    unittest.main()
