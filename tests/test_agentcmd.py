# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_mailbox  # noqa: E402
import agentcmd  # noqa: E402


class Boom(Exception):
    agent_code, hint, details = "BORDER_MOVED", "keep the border", {"part": 7}


def _deep(n):
    if n:
        return _deep(n - 1)
    raise ValueError("deep")


SCHEMA = {"type": "object", "required": ["use"], "properties": {
    "use": {"type": "string", "enum": ["GAME", "SCAN"]},
    "detail": {"type": "number", "minimum": 0, "maximum": 100},
    "parts": {"type": "array", "items": {"type": "integer"}},
    "keep": {"type": "boolean"}}}


class RunTests(unittest.TestCase):
    def setUp(self):
        agentcmd._registry = {
            "t.ok": {"fn": lambda c, use, **k: {"use": use}, "args": SCHEMA, "help": "", "edits": False},
            "t.boom": {"fn": lambda c: (_ for _ in ()).throw(Boom("Part 8: moved")), "args": {}, "help": "", "edits": True},
            "t.bug": {"fn": lambda c: _deep(20), "args": {}, "help": "", "edits": True},
            "tidy.again": {"fn": lambda c: {}, "args": {}, "help": "", "edits": True},
            "tidy.keep_going": {"fn": lambda c: {}, "args": {}, "help": "", "edits": True},
        }
        self.addCleanup(setattr, agentcmd, "_registry", None)

    def code(self, name, args=None):
        return agentcmd.run(name, args, context=object())

    def test_ok_reply(self):
        r = self.code("t.ok", {"use": "GAME", "detail": 5, "parts": [1, 2], "keep": True})
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"], {"use": "GAME"})
        self.assertIn("seconds", r)

    def test_schema(self):
        for args, bit in [({}, "missing"), ({"use": "NOPE"}, "one of"), ({"use": "GAME", "detail": 101}, "between"),
                          ({"use": "GAME", "detail": True}, "number"), ({"use": "GAME", "parts": [1, "x"]}, "parts[1]"),
                          ({"use": "GAME", "zzz": 1}, "unknown")]:
            r = self.code("t.ok", args)
            self.assertFalse(r["ok"], args)
            self.assertEqual(r["error"]["code"], "BAD_ARGS")
            self.assertIn(bit, r["error"]["message"])
        self.assertEqual(self.code("t.ok", {"use": "X"})["error"]["details"]["allowed"], ["GAME", "SCAN"])
        self.assertEqual(self.code("t.ok", [1])["error"]["code"], "BAD_ARGS")

    def test_unknown_command_close_names(self):
        e = self.code("tidy.agian")["error"]
        self.assertEqual(e["code"], "UNKNOWN_COMMAND")
        self.assertEqual(e["details"]["close"][0], "tidy.again")

    def test_agent_code_duck_typing(self):
        e = self.code("t.boom")["error"]
        self.assertEqual((e["code"], e["hint"], e["details"]), ("BORDER_MOVED", "keep the border", {"part": 7}))
        self.assertIn("moved", e["message"])

    def test_internal_trace_is_trimmed(self):
        e = self.code("t.bug")["error"]
        self.assertEqual(e["code"], "INTERNAL")
        self.assertLessEqual(len(e["details"]["trace"]), 8)
        self.assertTrue(e["details"]["trace"][-1].startswith("ValueError"))

    def test_missing_handler_modules_are_skipped(self):
        agentcmd._registry = None
        names = agentcmd.commands(reload=True)
        self.assertIn("commands.list", names)
        self.assertIn("session.info", names)

    def test_codes_table(self):
        self.assertEqual(len(agentcmd.CODES), 17)
        self.assertEqual(agentcmd.CODES.keys() & {"BAD_ARGS", "INTERNAL", "NEEDS_WINDOW"}, {"BAD_ARGS", "INTERNAL", "NEEDS_WINDOW"})

    def test_commands_list(self):
        agentcmd._registry.update(agentcmd.BUILTIN)
        r = self.code("commands.list")
        self.assertTrue(r["ok"])
        self.assertEqual([c["name"] for c in r["result"]["commands"]], sorted(agentcmd._registry))


class MailboxTests(unittest.TestCase):
    def test_pending_order_and_skips(self):
        names = ["0010.json", "0002.json.tmp", "0002.json", ".hidden.json", "notes.txt", "0001.json"]
        self.assertEqual(agent_mailbox.pending(names), ["0001.json", "0002.json", "0010.json"])

    def test_parse(self):
        self.assertEqual(agent_mailbox.parse('{"command":"a.b","args":{"x":1}}', "0007.json"), ("a.b", {"x": 1}, "0007"))
        self.assertEqual(agent_mailbox.parse('{"id":"q","command":"a.b"}', "0007.json")[2], "q")
        for bad, code in [("{nope", "FILE"), ("[1]", "BAD_ARGS"), ('{"args":{}}', "BAD_ARGS")]:
            c, _a, reply = agent_mailbox.parse(bad, "0007.json")
            self.assertIsNone(c)
            self.assertEqual((reply["ok"], reply["error"]["code"], reply["id"]), (False, code, "0007"))

    def test_handle_one_and_serve(self):
        agentcmd._registry = {"t.echo": {"fn": lambda c, n: {"n": n}, "args": {"required": ["n"], "properties": {"n": {"type": "integer"}}}, "help": "", "edits": False},
                              "session.quit": agentcmd.BUILTIN["session.quit"]}
        self.addCleanup(setattr, agentcmd, "_registry", None)
        with tempfile.TemporaryDirectory() as d:
            inbox = Path(d, "inbox")
            inbox.mkdir()
            (inbox / "0002.json").write_text(json.dumps({"command": "t.echo", "args": {"n": 2}}))
            (inbox / "0001.json").write_text(json.dumps({"command": "t.echo", "args": {"n": 1}}))
            (inbox / "0003.json.tmp").write_text("{half")
            (inbox / "0004.json").write_text("garbage")
            self.assertEqual(agent_mailbox.handle_one(d, object()), "t.echo")
            self.assertTrue(Path(d, "outbox", "0001.json").exists())
            self.assertFalse((inbox / "0001.json").exists())
            self.assertTrue((inbox / "0002.json").exists())  # one per call
            (inbox / "0005.json").write_text(json.dumps({"command": "session.quit"}))
            agent_mailbox.serve(d, object(), poll=0.001)
            out = {p.name: json.loads(p.read_text()) for p in Path(d, "outbox").iterdir()}
            self.assertEqual(sorted(out), ["0001.json", "0002.json", "0004.json", "0005.json"])
            self.assertEqual(out["0002.json"]["result"], {"n": 2})
            self.assertEqual(out["0004.json"]["error"]["code"], "FILE")
            self.assertEqual(out["0002.json"]["id"], "0002")
            self.assertTrue((inbox / "0003.json.tmp").exists())  # never touched
            self.assertFalse(any(p.suffix == ".tmp" for p in Path(d, "outbox").iterdir()))


if __name__ == "__main__":
    unittest.main()
