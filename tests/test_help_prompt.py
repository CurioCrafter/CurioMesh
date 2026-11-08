# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import promptgen  # noqa: E402

# "help" is also a stdlib name, so load the add-on's file by path
_spec = importlib.util.spec_from_file_location("cm_help", ROOT / "help.py")
cm_help = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm_help)

TOPICS = ("empty", "dashboard", "use", "detail", "start", "working", "verdict", "adjust", "tools", "fix", "finish",
          "bake", "lod", "export", "history", "undo", "pro", "prompt", "agent")


class HelpPrompt(unittest.TestCase):
    def test_help_keys(self):
        for key in TOPICS:
            title, body = cm_help.HELP[key]
            self.assertTrue(title and len(body) > 20, key)

    def test_build(self):
        for goal in promptgen.STEPS:
            text = promptgen.build({"goal": goal, "object": "Crate", "route": "headless"})
            self.assertIn("curiomesh-agent skill", text)
            self.assertIn("Model: Crate", text)
            self.assertIn(promptgen.SAFETY, text)
        mail = promptgen.build({"goal": "FIX", "route": "mailbox", "folder": "C:/x/agent", "numbers": {"faces": 900, "verdict": {"title": "Check"}}})
        self.assertIn("--folder", mail)
        self.assertIn("faces 900", mail)
        self.assertIn("issues.redo", mail)
        self.assertNotIn("no live link", mail)
        self.assertIn("no live link", promptgen.build({}))

    def test_real_command_names(self):
        src = "".join(p.read_text(encoding="utf-8") for p in [*ROOT.glob("agent*.py"), ROOT / "parts_export.py", ROOT / "parts_import.py"])
        steps = "\n".join(s for v in promptgen.STEPS.values() for s in v) + promptgen.CONNECT
        pat = r"\b((?:auto|use|settings|result|job|history|parts|issues|autofix|tools|tidy|loops|lod|export|session|commands)\.[a-z_]+)\b"
        for name in set(re.findall(pat, steps)):
            self.assertIn(f'"{name}"', src, name)


if __name__ == "__main__":
    unittest.main()
