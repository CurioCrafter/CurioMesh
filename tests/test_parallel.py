# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hardware  # noqa: E402
import parallel  # noqa: E402

BIG = {"cores": 24, "ram_gb": 64.0, "free_gb": 40.0}

# A stand-in helper: plain Python, same protocol. Part 3 fails, part 5 kills the helper, the rest are measured.
FAKE = r"""
import sys, time
import parallel
from pathlib import Path
folder = Path(sys.argv[1])
for line in sys.stdin:
    name, kind, part, spent = line.split()
    part = int(part)
    if part == 5:
        raise SystemExit(1)
    time.sleep(0.05 * (part % 3))  # finish out of order
    result = {"error": "no"} if part == 3 else {"sizes": [part], "levels": [], "ready": [], "seen": float(spent), "looks": 1, "field": 0.5, "notes": [f"n{part}"], "case": None, "timings": {"pictures": 1.0}, "peak_mb": 10.0 + part}
    parallel._write(folder / f"{name}.npz", result)
"""


class WorkerCountTests(unittest.TestCase):
    def test_small_jobs_stay_serial_and_one_helper_is_never_started(self):
        self.assertEqual(hardware.worker_count(BIG, "Full", 5_000, 1_000, 5_000, 30), 0)
        self.assertEqual(hardware.worker_count(BIG, "Full", 90_000, 1_000, 80_000, 1), 0)
        self.assertEqual(hardware.worker_count({"cores": 2, "ram_gb": 8.0, "free_gb": 6.0}, "Full", 90_000, 1_000, 80_000, 50), 0)

    def test_effort_sets_how_many_and_memory_caps_it(self):
        full, normal, light = (hardware.worker_count(BIG, e, 400_000, 100_000, 400_000, 600) for e in ("Full", "Normal", "Light"))
        self.assertEqual(full, 23)
        self.assertEqual(normal, 11)
        self.assertEqual(light, 2)
        self.assertEqual(hardware.worker_count(BIG, "Full", 400_000, 100_000, 400_000, 600, busy=12.0), 11)
        self.assertEqual(hardware.worker_count(BIG, "Full", 400_000, 100_000, 400_000, 600, most=4), 4)
        tight = {"cores": 24, "ram_gb": 64.0, "free_gb": 14.0}  # 4.4 GB above the 85% line
        each = hardware.worker_mb(400_000, 100_000)
        count = hardware.worker_count(tight, "Full", 400_000, 100_000, 400_000, 600)
        self.assertEqual(count, int((14.0 - 0.15 * 64.0) * 1024 // each))
        self.assertLessEqual(count * each, (14.0 - 0.15 * 64.0) * 1024)
        self.assertEqual(hardware.worker_count({"cores": 24, "ram_gb": 64.0, "free_gb": 10.0}, "Full", 400_000, 100_000, 400_000, 600), 0)


class PoolTests(unittest.TestCase):
    def test_parts_come_back_out_of_order_failures_are_reported_and_close_kills_every_helper(self):
        code = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(code))  # the stand-in is plain Python; real helpers enable the add-on instead
        pool = parallel.Pool([sys.executable, "-c", FAKE], 2, {"anything": 1}, env, tempfile.mkdtemp())
        try:
            todo, got = [0, 1, 2, 3, 4, 5, 6], {}
            ends = time.perf_counter() + 30.0
            while (todo or pool.busy) and time.perf_counter() < ends:
                pool.poll()
                while todo and pool.idle():
                    part = todo.pop(0)
                    self.assertTrue(pool.give(part, f"measure {part} 2.5"))
                for part in list(pool.ready):
                    self.assertTrue(pool.waiting_for(part))
                    got[part] = pool.take(part)
                if not pool.alive():
                    break
                time.sleep(0.01)
            self.assertIn(3, pool.failed)  # an error inside the helper
            self.assertIn(5, pool.failed)  # the helper died on it
            self.assertFalse(pool.waiting_for(5))
            self.assertEqual(pool.lost, 1)
            self.assertTrue({0, 1, 2, 4} <= set(got))
            self.assertEqual(got[2]["seen"], 2.5)
            self.assertIsNone(pool.take(0))  # taken once
        finally:
            procs = [p for p in pool.workers if p is not None]
            folder = pool.folder
            pool.close()
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertFalse(folder.exists())

    def test_files_round_trip_exactly_and_other_code_is_refused(self):
        import tempfile

        from triadq.budget import Level
        from triadq.types import FlatFaces

        faces = FlatFaces(np.array([0, 3, 7]), np.arange(7))
        value = {
            "co": np.arange(6.0).reshape(2, 3), "n": np.int32(3), "f": np.float64(0.5), 2: "two", "t": (1, [2.0, None], "x"),
            "s": {1, 2}, "fs": frozenset({"a"}), "faces": faces, "levels": [Level(4, 0.25)], "cfg": hardware.EFFORTS["LIGHT"],
            "nan": float("nan"), "ok": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npz"
            parallel._write(path, value)
            back = parallel._read(path)
            self.assertEqual(list(Path(tmp).iterdir()), [path])  # no temporary left
            self.assertTrue(np.array_equal(back["co"], value["co"]))
            self.assertIs(type(back["n"]), np.int32)
            self.assertIs(type(back["f"]), np.float64)
            self.assertEqual((back[2], back["t"], back["s"], back["fs"], back["levels"], back["cfg"], back["ok"]), ("two", (1, [2.0, None], "x"), {1, 2}, frozenset({"a"}), [Level(4, 0.25)], hardware.EFFORTS["LIGHT"], True))
            self.assertIsInstance(back["faces"], FlatFaces)
            self.assertEqual(list(back["faces"]), list(faces))
            self.assertTrue(np.isnan(back["nan"]))
            with self.assertRaises(TypeError):
                parallel._write(path, SimpleNamespace(a=1))  # not NumPy, not plain Python, not the add-on's
            np.savez(path, tree=np.asarray('{"$c": "SimpleNamespace", "f": {}}'))
            with self.assertRaises(ValueError):
                parallel._read(path)

    def test_helper_command_enables_the_addon_by_its_package_name(self):
        command, env = parallel.helper_command("rows_worker.main", ["folder", "0", "3"], binary="blender")
        self.assertEqual(command[:4], ["blender", "--background", "--factory-startup", "--python"])
        self.assertTrue(os.path.isfile(command[4]) and command[4].endswith("helper_entry.py"))  # a shipped script, no code string
        self.assertEqual(command[5:], ["--", parallel.PACKAGE, "rows_worker", "main", "folder", "0", "3"])
        self.assertEqual(env, dict(os.environ))  # the child's environment is not changed


class MergeTests(unittest.TestCase):
    def test_a_helpers_part_changes_the_job_as_measuring_it_here_would(self):
        job = SimpleNamespace(looks=2, field_spent=1.0, notes=["a"], timings={"pictures": 1.0}, part_cases={})
        parallel.apply_result(job, 7, {"looks": 3, "field": 0.5, "notes": ["b"], "timings": {"pictures": 2.0, "decimate": 1.5}, "case": ("box", 1, 2)})
        self.assertEqual((job.looks, job.field_spent, job.notes), (5, 1.5, ["a", "b"]))
        self.assertEqual(job.timings, {"pictures": 3.0, "decimate": 1.5})
        self.assertEqual(job.part_cases, {7: ("box", 1, 2)})

    def test_the_helper_side_returns_what_the_part_changed_and_numbers_it_like_the_job(self):
        class Job:
            def __init__(self):
                self.looks, self.field_spent, self.notes, self.timings, self.part_cases = 0, 0.0, [], {}, {}
                self.pieces, self.picture_sizes = [None] * 4, [64] * 4

            def _measure_piece(self, ids, pixels):
                part = len(self.sizes)  # the job numbers a part by how many came before it
                self.looks += 2
                self.field_spent += 1.0
                self.notes.append(f"part {part}")
                self.timings["pictures"] = self.timings.get("pictures", 0.0) + 0.25
                self.part_cases[part] = ("tube", 0, 2)
                yield
                self._finish_piece("key", ["small", "whole"], [1, 2], [False, False], 9.0, None, True)

        job = Job()
        first = parallel.measure_in_worker(job, 2, 10.0)
        second = parallel.measure_in_worker(job, 3, 0.0)
        self.assertEqual(first["sizes"], ["small", "whole"])
        self.assertEqual((first["looks"], first["field"], first["notes"], first["case"], first["timings"]), (2, 1.0, ["part 2"], ("tube", 0, 2), {"pictures": 0.25}))
        self.assertEqual((second["notes"], second["case"]), (["part 3"], ("tube", 0, 2)))
        self.assertEqual(job.notes, [])
        job._measure_piece = lambda ids, pixels: iter(())  # never finishes the part
        self.assertIn("error", parallel.measure_in_worker(job, 1, 0.0))


class FindMayStartTests(unittest.TestCase):
    def test_find_replaces_only_a_find(self):
        busy = lambda find: SimpleNamespace(busy=True, is_find=find)  # noqa: E731
        self.assertTrue(parallel.find_may_start(None))
        self.assertTrue(parallel.find_may_start(SimpleNamespace(busy=False)))
        self.assertTrue(parallel.find_may_start(busy(True)))
        self.assertFalse(parallel.find_may_start(SimpleNamespace(busy=True)))  # a search or a redo


if __name__ == "__main__":
    unittest.main()
