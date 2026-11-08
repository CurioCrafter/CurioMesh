# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agentops_tools as A  # noqa: E402


def prop(identifier, kind, default, **kw):
    base = dict(identifier=identifier, name=identifier.title(), description="help " + identifier, type=kind, default=default, is_hidden=False, is_readonly=False, hard_min=-3.4e38, hard_max=3.4e38, soft_min=-3.4e38, soft_max=3.4e38, enum_items=[])
    base.update(kw)
    return NS(**base)


PROPS = [
    prop("rna_type", "POINTER", None),
    prop("strength", "FLOAT", 0.6, hard_min=0.0, hard_max=1.0, soft_min=0.0, soft_max=1.0),
    prop("steps", "INT", 8, hard_min=1, hard_max=2**31 - 1, soft_min=1, soft_max=50),
    prop("keep_going", "BOOLEAN", False),
    prop("first", "BOOLEAN", False, is_hidden=True),
    prop("mode", "ENUM", "A", enum_items=[NS(identifier="A", name="A", description="a"), NS(identifier="B", name="B", description="b")]),
]


class Cand:
    def __init__(self, moved=None, faces=None, co=True, refusal="", numbers=None):
        self.moved, self.faces, self.refusal, self.numbers = moved, faces, refusal, numbers or {}
        self.co = np.zeros((3, 3)) if co else None


class AgentOpsToolsTest(unittest.TestCase):
    def test_schema_shape(self):
        schema = A.settings_schema(PROPS)
        self.assertEqual([e["name"] for e in schema], ["strength", "steps", "keep_going", "mode"])  # rna_type and hidden left out
        steps = schema[1]
        self.assertEqual((steps["min"], steps["max"], steps["soft_max"]), (1, None, 50))  # INT_MAX means no hard limit
        self.assertEqual(schema[0]["max"], 1.0)
        self.assertEqual([c["value"] for c in schema[3]["choices"]], ["A", "B"])
        self.assertTrue(all({"name", "label", "help", "type", "default"} <= set(e) for e in schema))

    def test_list_row(self):
        tool = NS(idname="curiomesh.tool_source_relax", label="Source Relax", group="RELAX", developer=True)
        row = A.tool_entry(tool, "help", A.settings_schema(PROPS), "Relax", True, False, ("strip",))
        self.assertEqual((row["tool"], row["group"], row["developer_only"], row["picks"]), ("tool_source_relax", "RELAX", True, ["strip"]))

    def check(self, given):
        return A.check_settings(A.settings_schema(PROPS), given)

    def bad(self, given):
        with self.assertRaises(A.ToolError) as cm:
            self.check(given)
        self.assertEqual(cm.exception.agent_code, "BAD_ARGS")
        return cm.exception.details

    def test_settings_validation(self):
        self.assertEqual(self.check({"strength": 1, "steps": 3.0, "keep_going": True, "mode": "B"}), {"strength": 1.0, "steps": 3, "keep_going": True, "mode": "B"})
        self.assertEqual(self.bad({"strength": 1.5})["maximum"], 1.0)
        self.assertEqual(self.bad({"steps": 0})["minimum"], 1)
        self.check({"steps": 5000})  # above the soft maximum is allowed: only the hard range counts
        self.assertIn("strength", self.bad({"strengh": 0.5})["close"])
        self.assertEqual(self.bad({"mode": "C"})["allowed"], ["A", "B"])
        self.bad({"keep_going": 1})
        self.bad({"strength": True})
        self.bad({"strength": float("nan")})
        self.bad({"steps": 2.5})
        self.bad({"strength": "x"})

    def test_gain(self):
        self.assertTrue(A.gained(Cand(moved=np.array([0.0, 0.2]))))
        self.assertFalse(A.gained(Cand(moved=np.array([0.0, 0.001]))))
        self.assertTrue(A.gained(Cand(faces=[(0, 1, 2)])))
        self.assertFalse(A.gained(Cand(co=False)))
        self.assertFalse(A.gained(Cand(moved=np.array([1.0]), refusal="no")))
        self.assertTrue(A.gained(Cand(moved=np.array([0.0]), numbers={"tidy": (25.0, 28.0)})))  # measured: the numbers decide
        self.assertTrue(A.gained(Cand(numbers={"poles": (5, 3), "tidy": (25.0, 25.0)})))
        self.assertFalse(A.gained(Cand(moved=np.array([2.0]), faces=[(0, 1, 2)], numbers={"tidy": (25.0, 24.0), "poles": (5, 5)})))  # moved a lot, nothing better

    def test_keep_going_stops_when_a_pass_gains_nothing(self):
        gains = iter([True, True, False, True])
        seen = []

        def one():
            g = next(gains)
            seen.append(g)
            return {"gain": g}

        passes, why = A.run_passes(one, True)
        self.assertEqual((len(passes), why, seen), (3, "gained nothing", [True, True, False]))  # the fourth is never run
        passes, why = A.run_passes(lambda: {"gain": True}, False)
        self.assertEqual((len(passes), why), (1, "single"))
        passes, why = A.run_passes(lambda: {"gain": False}, True)
        self.assertEqual((len(passes), why), (1, "gained nothing"))

    def test_keep_going_says_when_time_is_up(self):
        t = iter(range(0, 100, 10))
        passes, why = A.run_passes(lambda: {"gain": True}, True, seconds=25, clock=lambda: next(t))
        self.assertEqual(why, "time")
        self.assertEqual(len(passes), 3)  # started at 0, then looked at 10, 20, 30

    def test_commands_table(self):
        for name, c in A.COMMANDS.items():
            self.assertTrue(callable(c["fn"]) and c["help"] and c["args"]["type"] == "object", name)
            self.assertIn("object", c["args"]["properties"], name)
            self.assertTrue(set(c["args"]["required"]) <= set(c["args"]["properties"]), name)
        for name in ("tools.list", "tools.scope", "tools.preview", "tools.run", "tools.again", "tools.every", "loops.strips", "loops.edit", "loops.fix_all", "loops.ring", "loops.redraw", "loops.move_pole", "flow.add", "flow.list", "flow.remove", "flow.clear", "autofix.groups", "autofix.set", "autofix.run", "autofix.keep_going", "autofix.stop", "autofix.undo", "autofix.findings"):
            self.assertIn(name, A.COMMANDS)

    def test_error_shape(self):
        e = A.ToolError("REFUSED", "worse", "hint", {"a": 1})
        self.assertEqual((e.agent_code, e.hint, e.details, str(e)), ("REFUSED", "hint", {"a": 1}, "worse"))


if __name__ == "__main__":
    unittest.main()
