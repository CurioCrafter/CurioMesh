# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Send as a Problem link: only version, Blender version and Use, with ids from the issue form. No Blender needed."""

import re
import types
import unittest
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "parts_export.py").read_text(encoding="utf-8")


def _load():
    # issue_url is pure: run just the block between the markers, no bpy.
    block = SRC[SRC.index("ISSUE_URL ="):SRC.index("# ---", SRC.index("def issue_url"))]
    mod = types.ModuleType("blk")
    exec("import urllib.parse\n" + block, mod.__dict__)
    return mod


class SendProblemUrl(unittest.TestCase):
    def test_only_three_form_fields(self):
        m = _load()
        url = m.issue_url("0.9.9", "4.2.1 LTS", "GAME")
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(set(q), {"template", "version", "blender", "purpose"})
        self.assertEqual(q["purpose"], ["Game asset"])
        self.assertEqual(q["version"], ["0.9.9"])
        self.assertTrue(url.startswith("https://github.com/CurioCrafter/CurioMesh/issues/new?template=bad-spot.yml&"))

    def test_unknown_use_leaves_purpose_out(self):
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(_load().issue_url("1", "4", "CUSTOM")).query)
        self.assertNotIn("purpose", q)

    def test_ids_match_yaml(self):
        m = _load()
        yml = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bad-spot.yml").read_text(encoding="utf-8")
        ids = re.findall(r"^\s+id: (\w+)", yml, re.M)
        for fid in m.FORM_FIELDS.values():
            self.assertIn(fid, ids)
        # every Use maps to a real dropdown option
        opts = re.search(r"id: purpose.*?options:\n(.*?)\n\s+validations", yml, re.S).group(1)
        options = re.findall(r"- (.+)", opts)
        for v in m._FORM_PURPOSE.values():
            self.assertIn(v, options)


if __name__ == "__main__":
    unittest.main()
