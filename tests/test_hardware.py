# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hardware  # noqa: E402
import _addon  # noqa: E402

jobpack = _addon.load("jobpack")

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

    def test_a_piece_split_for_helpers_joins_at_its_border_and_leaves_other_pieces_alone(self):
        sizes = [len(c) for c in hardware.split_chunks(np.random.default_rng(2).random((380_000, 3)), 120_000)]
        self.assertTrue(all(60_000 <= n <= 120_000 for n in sizes))  # every region stays big enough for the native reducer
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]], float)
        only = np.array([True] * 6 + [False] * 3)  # two regions of a cut piece, then a loose piece touching them
        welded, joined = hardware.weld(points, [(0, 1, 2), (3, 4, 5), (6, 7, 8)], only)
        self.assertEqual(len(welded), 7)
        self.assertEqual(len(set(joined[0]) & set(joined[1])), 2)
        self.assertFalse(set(joined[2]) & (set(joined[0]) | set(joined[1])))
        self.assertTrue(np.array_equal(welded[list(joined[2])], points[6:]))

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


class SafetyTests(unittest.TestCase):
    def test_checkpoints_go_in_a_private_folder_and_a_planted_temp_name_is_left_alone(self):
        with tempfile.TemporaryDirectory() as root:
            folder = hardware.checkpoint_dir(np.zeros((3, 3)), 1, root)
            self.assertEqual(folder.parent.parent, Path(root))
            self.assertFalse(folder.parent.exists())  # made by the first saved part, so a job leaves no empty folder
            folder.mkdir(parents=True)
            (folder / "x.tmp").write_bytes(b"planted")
            path = hardware.piece_file(folder, b"k")
            hardware.save_piece(path, 1.0, [(np.zeros((3, 3)), [(0, 1, 2)], ["0"])], [(1, 0.0)], 1.0)
            self.assertEqual((folder / "x.tmp").read_bytes(), b"planted")
            self.assertEqual([p.name for p in folder.glob("*.tmp")], ["x.tmp"])
            self.assertIsNotNone(hardware.load_piece(path))

    def test_old_saved_progress_is_cleared_and_new_is_kept(self):
        with tempfile.TemporaryDirectory() as root:
            old, new = Path(root) / "old", Path(root) / "new"
            for folder in (old, new):
                folder.mkdir()
                (folder / "a.npz").write_bytes(b"12345")
            os.utime(old, (0, 0))
            self.assertEqual(hardware._clear_old(Path(root), hardware.STALE_RESUME_S), 5)
            self.assertEqual(sorted(p.name for p in Path(root).iterdir()), ["new"])
            self.assertEqual(hardware._clear_old(Path(root), 0.0), 5)

    def test_the_machine_is_read_without_outside_programs_or_ctypes(self):
        text = (Path(__file__).resolve().parents[1] / "hardware.py").read_text(encoding="utf-8")
        for word in ("subprocess", "ctypes", "nvidia-smi", "sysctl"):
            self.assertNotIn(word, text)
        machine = hardware.detect()
        self.assertGreaterEqual(machine["cores"], 1)
        self.assertTrue(hardware.describe(machine))

    def test_a_job_folder_name_cannot_leave_the_chosen_folder_and_a_bad_result_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            folder = jobpack.job_folder(root, "../../evil/..\\x")
            self.assertEqual(folder.parent, Path(root).resolve())
            out = {"co": np.zeros((3, 3)), "faces": [(0, 1, 7)], "mat": np.zeros(1, np.int32), "parts": np.zeros(1, np.int64)}
            report = {"faces": 1, "quads": 0, "loss": 0.0, "seconds": 1.0}
            jobpack.save_result(Path(root) / "ok.npz", out, report, 1, 3)  # vertex 7 does not exist
            with self.assertRaises(ValueError):
                jobpack.load_result(Path(root) / "ok.npz", 1, 3)
            report_text = np.asarray('{"faces": 1, "quads": 0, "loss": 0.0, "seconds": 1.0}')
            np.savez(Path(root) / "float.npz", co=np.zeros((3, 3)), lens=np.array([3.0]), flat=np.array([0.0, 1.0, 2.0]), mat=np.zeros(1, np.int32), parts=np.zeros(1, np.int64), source=np.array([1, 3]), report=report_text)
            with self.assertRaises(ValueError):  # face sizes as floats would fail later with a TypeError
                jobpack.load_result(Path(root) / "float.npz", 1, 3)


class NpzLimitTests(unittest.TestCase):
    def test_a_file_that_unpacks_too_big_is_refused_before_reading(self):
        import zipfile

        with tempfile.TemporaryDirectory() as folder:
            bomb = Path(folder) / "bomb.npz"
            with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as packed:
                packed.writestr("a.npy", bytes(hardware.MOST_NPZ_BYTES + 1))  # compresses to about a megabyte
            with self.assertRaises(ValueError):
                hardware.load_npz(bomb)
            (Path(folder) / "junk.npz").write_bytes(b"not a zip")
            with self.assertRaises(ValueError):
                hardware.load_npz(Path(folder) / "junk.npz")


class JobFolderTests(unittest.TestCase):
    def test_export_writes_a_runnable_folder_and_the_result_loads_back(self):
        co = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        faces = [(0, 1, 2, 3)]
        with tempfile.TemporaryDirectory() as folder:
            out = jobpack.export_job(Path(folder) / "job", co, faces, [0], 6.0, 0, "FULL", mirror=(0, 0.5, 1.0, 2, 1), lines=[np.zeros((2, 3))], drop=np.zeros(4), source=(2, 6))
            for name in ("mesh.npz", "guides.npz", "job.json", "run_job.py", "README.txt"):
                self.assertTrue((out / name).exists(), name)
            settings = __import__("json").loads((out / "job.json").read_text())
            self.assertEqual((settings["mirror"][3], settings["source_faces"]), (2, 2))
            self.assertIn("drop", np.load(out / "mesh.npz").files)
            self.assertFalse((out / "curiomesh_engine").exists())  # the installed add-on runs it
            run_job = (out / "run_job.py").read_text()
            compile(run_job, "run_job.py", "exec")
            self.assertNotIn("sys.path", run_job)
            output = {"co": co, "faces": faces, "mat": np.zeros(1, np.int32), "parts": np.zeros(1, np.int64)}
            report = {"faces": 1, "quads": 1, "loss": 0.0, "seconds": 1.0}
            jobpack.save_result(out / "result.npz", output, report, 1, 4)
            res, back = jobpack.load_result(out / "result.npz", 1, 4)
            self.assertEqual(res["faces"], faces)
            self.assertEqual(back["quads"], 1)
            with self.assertRaises(ValueError):
                jobpack.load_result(out / "result.npz", 2, 4)


class CostTests(unittest.TestCase):
    def test_bake_time_grows_with_pixels_maps_and_sets_and_shrinks_with_cores(self):
        base = hardware.bake_seconds(1024, "NORMAL", 1, SMALL)
        self.assertEqual(hardware.bake_seconds(1024, "OFF", 1, SMALL), 0.0)
        self.assertGreater(hardware.bake_seconds(2048, "NORMAL", 1, SMALL), 3.0 * base)  # four times the pixels, less the fixed part
        self.assertGreater(hardware.bake_seconds(1024, "NORMAL_COLOR", 1, SMALL), base)
        self.assertGreater(hardware.bake_seconds(1024, "NORMAL_COLOR", 4, SMALL), 3.0 * hardware.bake_seconds(1024, "NORMAL_COLOR", 1, SMALL))
        self.assertLess(hardware.bake_seconds(1024, "NORMAL", 1, BIG), base)
        self.assertEqual(hardware.bake_seconds(1024, "NORMAL", 3, SMALL), base)  # sets only apply to Normal + Color

    def test_style_and_tags_and_the_sentence(self):
        self.assertEqual(hardware.style_factor("DETAIL"), 1.0)
        self.assertGreater(hardware.style_factor("ROWS"), 1.0)
        self.assertGreater(hardware.style_factor("DETAIL", "QUADRIFLOW"), 1.0)
        self.assertEqual(hardware.cost_tag(5), ("", False))
        self.assertEqual(hardware.cost_tag(60), ("Slow: about 60 seconds", False))
        self.assertEqual(hardware.cost_tag(400, 3)[1], True)
        self.assertEqual(hardware.with_bake("About 3 minutes", 5), "About 3 minutes")
        self.assertIn("plus about 2 minutes of baking that blocks Blender", hardware.with_bake("x", 120))

    def test_asking_first(self):
        self.assertFalse(hardware.needs_confirm("OK", 100, 100))
        self.assertTrue(hardware.needs_confirm("LARGE", 100))
        self.assertTrue(hardware.needs_confirm("OK", 400, 300))
        self.assertTrue(hardware.needs_confirm("OK", 1, 130, applying=True))
        self.assertFalse(hardware.needs_confirm("TOO_BIG", 1, 100, applying=True))  # Apply only cares about its bake
        ways = hardware.cheaper_ways("ROWS", "NORMAL_COLOR", 4096, "NORMAL")
        self.assertEqual(len(ways), 4)
        self.assertTrue(ways[-1].startswith("Export Job"))
        self.assertEqual(len(hardware.cheaper_ways("DETAIL", "OFF", 512, "LIGHT")), 1)

    def test_weak_pc_notice(self):
        self.assertIn("4 cores", hardware.weak_pc(dict(SMALL, gpu="x")))
        self.assertEqual(hardware.weak_pc(dict(BIG)), "")  # the card doesn't matter: bakes use the scene's Cycles device
        self.assertEqual(hardware.weak_pc(dict(BIG, gpu="RTX")), "")
        self.assertEqual(hardware.weak_pc(None), "")


if __name__ == "__main__":
    unittest.main()
