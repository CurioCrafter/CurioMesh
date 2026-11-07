# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""agentops_parts without Blender: part list shaping, id validation and the command table, on a fake segment table."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agentops_parts as A  # noqa: E402


def _record(i, change, **more):
    base = {"id": i, "kind": "ROUGH", "before": 1000, "after": 100 + i, "change": change, "method": "AUTO", "tries": [{"method": "AUTO", "after": 100 + i, "change": change}], "cur": 0, "goal": "REDUCE", "goal_detail": 0.5, "topology": None}
    return dict(base, **more)


class PartsShapeTest(unittest.TestCase):
    def setUp(self):
        self.table = [_record(0, 2.0), _record(1, None), _record(2, 9.5, name="Handle", locked=True), _record(3, 5.0)]

    def test_list_rows_and_worst_first(self):
        spots = [{"part": 2}, {"part": 2}, {"part": 0}, {"part": -1}]
        out = A.shape_parts(self.table, spots, {3})
        self.assertEqual([r["name"] for r in out["parts"]], ["Part 1", "Part 2", "Handle", "Part 4"])
        self.assertEqual(out["worst_first"], [2, 3, 0, 1])  # unmeasured last
        self.assertEqual([r["bad_spots"] for r in out["parts"]], [1, 0, 2, 0])
        self.assertTrue(out["parts"][2]["locked"])
        self.assertEqual(out["selected"], [3])
        self.assertTrue(out["parts"][3]["selected"] and not out["parts"][0]["selected"])
        self.assertEqual(out["parts"][0]["faces_before"], 1000)
        self.assertEqual(out["parts"][0]["faces_after"], 100)

    def test_animation_part_is_ranked_by_its_tidiness(self):
        judged = lambda r: 0.2 * (100.0 - r["topology"]) if r.get("topology") is not None else r["change"]  # noqa: E731
        table = [_record(0, 1.0), _record(1, 1.0, topology=10.0)]
        self.assertEqual(A.shape_parts(table, [], set(), judged)["worst_first"], [1, 0])

    def test_tries_mark_the_one_shown(self):
        record = _record(0, 3.0)
        record["tries"].append({"method": "LOWPOLY", "detail": 0.5, "after": 40, "change": 7.0, "topology": 80.0})
        record["cur"] = 1
        rows = A.shape_tries(record)
        self.assertEqual([r["current"] for r in rows], [False, True])
        self.assertEqual(rows[1]["faces"], 40)

    def test_spot_row_names_the_worst_kind(self):
        spot = {"kinds": {"pole": 9, "inward": 2}, "faces": [4, 5], "center": [0, 1, 2], "radius": 0.1, "part": 3, "score": 7.0}
        row = A.shape_spot(0, spot, {"pole": 0.1, "inward": 5.0})
        self.assertEqual((row["kind"], row["part"], row["face_count"], row["centre"]), ("inward", 3, 2, [0, 1, 2]))

    def test_compare_reply_drops_face_lists(self):
        diff = {"fixed": [{"kinds": {"inward": 1}, "part": 0, "center": [0, 0, 0], "score": 1.0, "faces": [1, 2]}], "still": [], "new": [], "totals": {"inward": {"old": 1, "new": 0}}}
        out = A.shape_diff(diff)
        self.assertEqual((out["fixed"], out["still"], out["new"]), (1, 0, 0))
        self.assertNotIn("faces", out["fixed_spots"][0])


class IdCheckTest(unittest.TestCase):
    def test_valid_ids_keep_order_and_lose_repeats(self):
        self.assertEqual(A.check_ids([3, 1, 3, 0], 4), [3, 1, 0])
        self.assertEqual(A.check_ids([], 4), [])

    def test_missing_part_is_not_found_with_the_range(self):
        with self.assertRaises(A.PartsError) as cm:
            A.check_ids([1, 7, 9], 4)
        e = cm.exception
        self.assertEqual(e.agent_code, "NOT_FOUND")
        self.assertEqual(e.details["missing"], [7, 9])
        self.assertIn("0 to 3", e.hint)

    def test_negative_and_no_parts(self):
        with self.assertRaises(A.PartsError) as cm:
            A.check_ids([-1], 4)
        self.assertEqual(cm.exception.agent_code, "NOT_FOUND")
        with self.assertRaises(A.PartsError) as cm:
            A.check_ids([0], 0)
        self.assertEqual(cm.exception.agent_code, "NOT_FOUND")

    def test_not_a_whole_number_is_bad_args(self):
        for bad in ("2", 1.5, True, None):
            with self.assertRaises(A.PartsError) as cm:
                A.check_ids([bad], 4)
            self.assertEqual(cm.exception.agent_code, "BAD_ARGS")

    def test_use_names(self):
        self.assertEqual(A.use_of("LOW_POLY"), "LOWPOLY")
        self.assertEqual(A.use_of("subd"), "SUBD")
        self.assertEqual(A.use_of("ANIMATION"), "ANIMATION")
        with self.assertRaises(A.PartsError) as cm:
            A.use_of("GAME")
        self.assertEqual(cm.exception.agent_code, "BAD_ARGS")


class CommandTableTest(unittest.TestCase):
    def test_every_command_is_complete(self):
        for name, c in A.COMMANDS.items():
            self.assertTrue(callable(c["fn"]), name)
            self.assertTrue(c["help"], name)
            self.assertEqual(c["args"]["type"], "object", name)
            self.assertIn("object", c["args"]["properties"], name)
            self.assertTrue(set(c["args"]["required"]) <= set(c["args"]["properties"]), name)
            self.assertIsInstance(c["edits"], bool, name)

    def test_the_named_commands_are_there_and_the_exchange_ones_are_not(self):
        want = "issues.list issues.redo issues.keep_original issues.put_back issues.save issues.compare fix.find fix.run parts.list parts.select parts.worst parts.rename parts.lock parts.merge parts.halve parts.from_faces parts.redo parts.tries parts.use_try parts.keep_original".split()
        self.assertEqual(sorted(A.COMMANDS), sorted(want))

    def test_use_and_method_lists_match_the_add_on(self):
        text = (Path(__file__).resolve().parents[1] / "segment_goals.py").read_text(encoding="utf-8")
        for use in A.USES:
            self.assertIn(f'("{use}", ', text)
        profiles = (Path(__file__).resolve().parents[1] / "profiles.py").read_text(encoding="utf-8")
        for method in A.METHODS:
            self.assertIn(f'("{method}", ', profiles)


if __name__ == "__main__":
    unittest.main()
