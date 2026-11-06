# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "skills" / "curiomesh-retopo" / "scripts"))

import cm_agent as cm  # noqa: E402
from scripts.build_release import build_skills  # noqa: E402


class AgentSkillsTest(unittest.TestCase):
    def test_report_helpers(self):
        self.assertEqual(cm.parse_use("low-poly"), "LOW_POLY")
        self.assertEqual(cm.parse_reduce("25%"), ("PERCENT", 25.0))
        self.assertEqual(cm.parse_faces("1,5,10-12"), [1, 5, 10, 11, 12])
        record = {"totals": {"inward": 3, "pole": 9}, "spots": [{"part": 2, "kinds": {"inward": 3}, "faces": [1], "face_count": 3, "score": 5.0}]}
        summary = cm.issues_summary(record)
        self.assertEqual(summary["per_part"], {"2": {"inward": 3}})
        self.assertEqual(cm.defect_faces(summary), 3)
        good = {"faces": 1000, "detail": 6, "view_changed_pct": 5.9, "tris": 0, "issues": {"totals": {}}}
        self.assertTrue(cm.verdict(good)["ok"])
        bad = dict(good, view_changed_pct=14.0, issues=summary)
        self.assertFalse(cm.verdict(bad)["ok"])
        self.assertTrue(cm.better({"bad": 5}, {"bad": 4}))
        self.assertFalse(cm.better({"bad": 5}, {"bad": 5}))  # equal is not better: the step is undone

    def test_skill_files_and_zip(self):
        skills = sorted((ROOT / "skills").glob("*/SKILL.md"))
        self.assertGreaterEqual(len(skills), 4)
        for path in skills:
            head = path.read_text(encoding="utf-8").split("---")[1]
            name = re.search(r"^name: (.+)$", head, re.M).group(1)
            self.assertEqual(name, path.parent.name)
            self.assertRegex(name, r"^[a-z0-9]+(-[a-z0-9]+)*$")
            self.assertIn("description:", head)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(build_skills(ROOT, Path(tmp))) as archive:
                names = archive.namelist()
        self.assertIn("curiomesh-retopo/SKILL.md", names)
        self.assertIn("curiomesh-retopo/scripts/curiomesh_run.py", names)
        self.assertFalse([n for n in names if "__pycache__" in n])


if __name__ == "__main__":
    unittest.main()
