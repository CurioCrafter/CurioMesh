# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The pure helpers of agentops_run (no Blender): property values to JSON, range and choice checks, History list shaping."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agentops_run as R  # noqa: E402
import jobpack  # noqa: E402


def prop(kind, identifier="p", **kw):
    base = dict(type=kind, identifier=identifier, name=identifier.title(), description="help", is_readonly=False, options=set(), array_length=0, hard_min=-3.4e38, hard_max=3.4e38, soft_min=-3.4e38, soft_max=3.4e38, enum_items=[], is_enum_flag=False, length_max=0)
    base.update(kw)
    return NS(**base)


def refuses(info, value):
    with unittest.TestCase().assertRaises(R.AgentError) as cm:
        R.check_setting(info, value)
    assert cm.exception.agent_code == "BAD_ARGS"
    return cm.exception


class Values(unittest.TestCase):
    def test_json(self):
        self.assertEqual(R.to_json((1.0, 2.0)), [1.0, 2.0])
        self.assertEqual(R.to_json({"b", "a"}), ["a", "b"])
        self.assertEqual(R.to_json(None), None)

    def test_info_float_range(self):
        info = R.prop_info(prop("FLOAT", hard_min=0.1, hard_max=80.0, soft_min=0.1, soft_max=20.0), 6.0, 6.0)
        self.assertEqual((info["min"], info["max"], info["soft_max"]), (0.1, 80.0, 20.0))
        self.assertEqual(R.prop_info(prop("FLOAT", hard_min=0.10000000149011612), 1.0)["min"], 0.1)  # float32 noise trimmed
        self.assertEqual(R.prop_info(prop("FLOAT"), 1.0)["min"], None)  # no limit is None, not -3.4e38
        self.assertEqual(R.prop_info(prop("INT", hard_min=-(2**31), hard_max=2**31 - 1), 3)["max"], None)

    def test_info_enum(self):
        items = [NS(identifier="A", name="Aa", description="first"), NS(identifier="B", name="Bb", description="second")]
        info = R.prop_info(prop("ENUM", enum_items=items), "A", "B")
        self.assertEqual([i["id"] for i in info["items"]], ["A", "B"])
        self.assertEqual(info["items"][1]["help"], "second")
        json.dumps(info)

    def test_hidden_and_readonly(self):
        self.assertTrue(R.prop_info(prop("INT", options={"HIDDEN"}), 0)["hidden"])
        refuses(R.prop_info(prop("INT", options={"HIDDEN"}), 0), 1)
        refuses(R.prop_info(prop("BOOLEAN", is_readonly=True), True), False)

    def test_links_refused(self):
        refuses({"name": "obj", "type": "POINTER"}, "x")
        refuses({"name": "list", "type": "COLLECTION"}, [])

    def test_number_range(self):
        info = R.prop_info(prop("FLOAT", hard_min=0.1, hard_max=80.0), 6.0)
        self.assertEqual(R.check_setting(info, 7), 7.0)
        err = refuses(info, 81)
        self.assertEqual((err.details["min"], err.details["max"]), (0.1, 80.0))
        refuses(info, 0.0)
        refuses(info, "6")
        refuses(info, True)
        refuses(info, float("nan"))

    def test_int(self):
        info = R.prop_info(prop("INT", hard_min=1, hard_max=10), 3)
        self.assertEqual(R.check_setting(info, 4.0), 4)
        refuses(info, 4.5)
        refuses(info, 11)

    def test_unbounded_float_takes_big_values(self):
        self.assertEqual(R.check_setting(R.prop_info(prop("FLOAT"), 0.0), 1e9), 1e9)

    def test_enum(self):
        items = [NS(identifier="A", name="Aa", description=""), NS(identifier="B", name="Bb", description="")]
        info = R.prop_info(prop("ENUM", enum_items=items), "A")
        self.assertEqual(R.check_setting(info, "B"), "B")
        self.assertEqual(refuses(info, "C").details["choices"], ["A", "B"])
        flags = R.prop_info(prop("ENUM", enum_items=items, is_enum_flag=True), set())
        self.assertEqual(R.check_setting(flags, ["A", "B"]), {"A", "B"})
        refuses(flags, ["A", "Z"])

    def test_bool_string_array(self):
        self.assertIs(R.check_setting(R.prop_info(prop("BOOLEAN"), False), True), True)
        refuses(R.prop_info(prop("BOOLEAN"), False), 1)
        info = R.prop_info(prop("STRING", length_max=4), "")
        self.assertEqual(R.check_setting(info, "abc"), "abc")
        refuses(info, "abcdef")
        vec = R.prop_info(prop("FLOAT", array_length=3, hard_min=0.0, hard_max=1.0), [0, 0, 0])
        self.assertEqual(R.check_setting(vec, [0, 0.5, 1]), [0.0, 0.5, 1.0])
        refuses(vec, [0, 1])
        refuses(vec, [0, 0, 2])


class HistoryRows(unittest.TestCase):
    ROWS = [
        {"index": 0, "step": 1, "name": "New result", "numbers": "", "kind": "", "faces": 900, "on_screen": False, "undone": False},
        {"index": 1, "step": 2, "name": "Tidy Up", "numbers": "", "kind": "tidy", "faces": 850, "on_screen": True, "undone": False},
        {"index": 2, "step": 3, "name": "Quad Size", "numbers": "", "kind": "size", "faces": 700, "on_screen": False, "undone": True},
    ]

    def test_shape(self):
        out = R.history_rows(self.ROWS, 2)
        self.assertEqual(out["current"]["name"], "Tidy Up")
        self.assertTrue(out["can_undo"] and out["can_redo"])
        first = R.history_rows(self.ROWS, 1)
        self.assertFalse(first["can_undo"])
        last = R.history_rows(self.ROWS, 3)
        self.assertFalse(last["can_redo"])
        self.assertEqual(R.history_rows([], 0)["current"], None)

    def test_find(self):
        self.assertEqual(R.find_step(self.ROWS, step=2)["faces"], 850)
        self.assertEqual(R.find_step(self.ROWS, index=0)["faces"], 900)
        for bad in ({"step": 9}, {"index": 7}, {"index": -1}, {}, {"step": 1, "index": 1}):
            with self.assertRaises(R.AgentError) as cm:
                R.find_step(self.ROWS, **bad)
            self.assertIn(cm.exception.agent_code, ("NOT_FOUND", "BAD_ARGS"))


class Shapes(unittest.TestCase):
    def test_status(self):
        idle = R.status_shape([])
        self.assertFalse(idle["running"])
        self.assertIsNone(idle["percent"])
        busy = R.status_shape(["auto", "tidy"], 0.256, "Part 3 of 12", 4.44)
        self.assertEqual((busy["kind"], busy["percent"], busy["seconds"]), ("auto", 25.6, 4.4))

    def test_counts(self):
        out = R.face_counts([(0, 1, 2, 3), (0, 1, 2), (0, 1, 2, 3, 4)])
        self.assertEqual((out["quads"], out["tris"], out["ngons"], out["quad_pct"]), (1, 1, 1, 33.3))

    def test_issues(self):
        rec = {"totals": {"folded_over": 3}, "spots": [{"part": 2, "kinds": {"folded_over": 2}, "face_count": 5, "score": 1.0}, {"part": 2, "kinds": {"inward": 1}}]}
        out = R.issues_summary(rec, top=1)
        self.assertEqual(out["per_part"], {"2": {"folded_over": 2, "inward": 1}})
        self.assertEqual(len(out["worst"]), 1)
        self.assertEqual(R.issues_summary(None)["spots"], 0)

    def test_tidy_levels(self):
        rows = R.tidy_levels({0.0: {"tidy": 50.0, "view": 3.0, "poles": 20, "out": object()}, 0.5: {"tidy": 61.234, "view": 3.2, "poles": 9, "faces": 800}}, 0.5)
        self.assertEqual([r["strength"] for r in rows], [0.0, 0.5])
        self.assertTrue(rows[1]["shown"] and not rows[0]["shown"])
        self.assertEqual(rows[1]["tidy"], 61.23)
        json.dumps(rows)

    def test_commands_table(self):
        self.assertGreaterEqual(len(R.COMMANDS), 35)
        for name, cmd in R.COMMANDS.items():
            self.assertTrue(callable(cmd["fn"]) and cmd["help"] and cmd["args"]["type"] == "object", name)
            self.assertIn("object", cmd["args"]["properties"], name)
            for need in cmd["args"]["required"]:
                self.assertIn(need, cmd["args"]["properties"], name)
        for must in ("settings.set", "auto.start", "job.wait", "result.numbers", "tidy.keep_going", "history.jump", "history.undo", "export.game", "job.import_result"):
            self.assertIn(must, R.COMMANDS)


class JobEffort(unittest.TestCase):
    def test_set_effort(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "job.json").write_text(json.dumps({"effort": "FULL", "detail": 6.0}))
            (Path(d) / "README.txt").write_text("It is set to FULL now.")
            jobpack.set_effort(d, "LIGHT")
            self.assertEqual(json.loads((Path(d) / "job.json").read_text()), {"effort": "LIGHT", "detail": 6.0})
            self.assertIn("It is set to LIGHT now", (Path(d) / "README.txt").read_text())


if __name__ == "__main__":
    unittest.main()
