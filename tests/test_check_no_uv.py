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
        self.assertEqual(check_no_uv.check(tree({"triadq/uvs.py": "x", "bridge.py": "curiomesh_uv_status"})), [])
        for rel, text in [("uv_fast.py", ""), ("native/src/uvpack.rs", ""), ("a.py", "x  # " + MARK + "\n"),
                          ("b.md", "<!-- " + MARK + " begin -->"), ("c.py", "curiomesh" + "_uv = 1")]:
            self.assertEqual(len(check_no_uv.check(tree({rel: text}))), 1, rel)

    def test_passes_on_this_repo(self):
        self.assertEqual(check_no_uv.check(Path(__file__).resolve().parents[1]), [])


if __name__ == "__main__":
    unittest.main()
