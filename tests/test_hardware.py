# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hardware  # noqa: E402
import jobpack  # noqa: E402

BIG = {"os": "Linux", "cores": 16, "ram_gb": 64.0, "free_gb": 48.0}
SMALL = {"os": "Windows", "cores": 4, "ram_gb": 8.0, "free_gb": 3.0}


class ChunkTests(unittest.TestCase):
    def test_every_face_lands_in_exactly_one_small_compact_chunk(self):
        centers = np.random.default_rng(1).random((1000, 3))
        chunks = hardware.split_chunks(centers, 130)
        self.assertTrue(all(len(c) <= 130 for c in chunks))
        self.assertEqual(sorted(np.concatenate(chunks).tolist()), list(range(1000)))
        self.assertEqual(len(hardware.split_chunks(centers, 5000)), 1)

    def test_chunk_borders_are_found_and_welded_back_with_no_gap(self):
        faces = [(0, 1, 2), (1, 3, 2)]  # two triangles sharing the edge 1-2
        label = np.array([0, 1])
        self.assertEqual(hardware.border_vertices(faces, label, 4).tolist(), [False, True, True, False])
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)  # the border twice
        welded, joined = hardware.weld(points, [(0, 1, 2), (3, 4, 5)])
        self.assertEqual(len(welded), 4)
        self.assertEqual(len(set(joined[0]) & set(joined[1])), 2)

    def test_region_names_say_where_the_part_is(self):
        low, high = np.zeros(3), np.ones(3)
        self.assertEqual(hardware.region_name(np.array([0.5, 0.5, 0.5]), low, high), "middle")
        self.assertEqual(hardware.region_name(np.array([0.1, 0.5, 0.9]), low, high), "upper left")


class SettingsTests(unittest.TestCase):
    def test_auto_picks_by_the_machine_and_light_is_lighter(self):
        self.assertEqual(hardware.pick_effort(SMALL), "LIGHT")
        self.assertEqual(hardware.pick_effort(BIG), "FULL")
        self.assertEqual(hardware.pick_effort({"os": "x", "cores": 6, "free_gb": 10.0}), "NORMAL")
        light, full = hardware.settings_for("LIGHT", BIG), hardware.settings_for("FULL", BIG)
        self.assertLess(light.chunk_faces, full.chunk_faces)
        self.assertLess(light.max_rays, full.max_rays)
        self.assertGreater(light.gap_s, full.gap_s)
        self.assertEqual(hardware.settings_for("AUTO", SMALL).name, "Light")

    def test_chunks_shrink_to_fit_free_memory(self):
        tight = {"os": "x", "cores": 8, "free_gb": 1.0}
        self.assertLess(hardware.settings_for("FULL", tight).chunk_faces, hardware.EFFORTS["FULL"].chunk_faces)

    def test_big_job_on_small_machine_is_rated_large_and_slower_than_on_a_big_one(self):
        light = hardware.settings_for("AUTO", SMALL)
        slow = hardware.estimate(1_500_000, 3, light)
        fast = hardware.estimate(1_500_000, 3, hardware.settings_for("AUTO", BIG))
        self.assertGreater(slow["seconds"], 0)
        self.assertEqual(hardware.rate(SMALL, slow, light)[0], "TOO_BIG")  # 5 GB does not fit in 3 GB free
        level, text = hardware.rate(dict(SMALL, free_gb=12.0), slow, light)
        self.assertEqual(level, "LARGE")
        self.assertIn("large for this PC", text)
        self.assertIn("working in small steps", text)
        self.assertEqual(hardware.rate(BIG, hardware.estimate(20_000, 5, hardware.EFFORTS["NORMAL"]), hardware.EFFORTS["NORMAL"])[0], "OK")
        self.assertGreater(slow["gb"], 0)
        self.assertGreater(fast["gb"], 0)


class SavedProgressTests(unittest.TestCase):
    def test_a_saved_part_comes_back_the_same_and_a_bad_file_is_ignored(self):
        sizes = [(np.random.default_rng(2).random((5, 3)), [(0, 1, 2), (2, 3, 4, 0)], ["0", "3"]), (np.zeros((3, 3)), [(0, 1, 2)], ["1"])]
        with tempfile.TemporaryDirectory() as folder:
            path = hardware.piece_file(Path(folder), b"key")
            hardware.save_piece(path, 6.0, sizes, [(2, 1.5), (1, 0.0)], 9.0)
            detail, back, levels, seen = hardware.load_piece(path)
            self.assertEqual((detail, seen, levels), (6.0, 9.0, [(2, 1.5), (1, 0.0)]))
            self.assertEqual(back[0][1], sizes[0][1])
            self.assertEqual(back[0][2], ["0", "3"])
            np.testing.assert_allclose(back[0][0], sizes[0][0])
            path.write_bytes(b"not a zip")
            self.assertIsNone(hardware.load_piece(path))


class JobFolderTests(unittest.TestCase):
    def test_export_writes_a_runnable_folder_and_the_result_loads_back(self):
        co = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        faces = [(0, 1, 2, 3)]
        with tempfile.TemporaryDirectory() as folder:
            out = jobpack.export_job(Path(folder) / "job", co, faces, [0], 6.0, 0, "FULL", Path(__file__).resolve().parents[1])
            for name in ("mesh.npz", "job.json", "run_job.py", "README.txt", "curiomesh_engine/autoretopo.py", "curiomesh_engine/triadq/budget.py"):
                self.assertTrue((out / name).exists(), name)
            compile((out / "run_job.py").read_text(), "run_job.py", "exec")
            output = {"co": co, "faces": faces, "mat": np.zeros(1, np.int32), "parts": np.zeros(1, np.int64)}
            report = {"faces": 1, "quads": 1, "loss": 0.0, "seconds": 1.0}
            jobpack.save_result(out / "result.npz", output, report, 1, 4)
            res, back = jobpack.load_result(out / "result.npz", 1, 4)
            self.assertEqual(res["faces"], faces)
            self.assertEqual(back["quads"], 1)
            with self.assertRaises(ValueError):
                jobpack.load_result(out / "result.npz", 2, 4)


if __name__ == "__main__":
    unittest.main()
