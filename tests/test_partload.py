# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""partload: a grid in two parts, the part folder written by hand, and every check an import makes."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import partload as L  # noqa: E402
from triadq import parts as P  # noqa: E402
from triadq import segment as S  # noqa: E402
from triadq.patch import boundary_loops  # noqa: E402


def grid(nx=6, ny=4):
    co = np.array([(i * 0.5, j * 0.5, 0.0) for j in range(ny + 1) for i in range(nx + 1)])
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i) for j in range(ny) for i in range(nx)]
    piece = np.array([0 if i < nx // 2 else 1 for j in range(ny) for i in range(nx)])
    return co, faces, np.zeros(len(faces), np.int32), piece


def obj(points, faces, mats=None):
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in np.asarray(points, dtype=np.float32).astype(np.float64).tolist()]
    last = None
    for i, f in enumerate(faces):
        if mats is not None and int(mats[i]) != last:
            last = int(mats[i])
            lines.append(f"usemtl m{last}")
        lines.append("f " + " ".join(str(v + 1) for v in f))
    return "\n".join(lines) + "\n"


class Folder:
    """The grid's part 1 written in the folder format, as an exporter would."""

    def __init__(self, root):
        self.co, self.faces, self.mat, self.piece = grid()
        self.ids = np.flatnonzero(self.piece == 1)
        self.pts, self.local, self.mats = P.cut(self.co, self.faces, self.mat, self.piece, 1)
        self.loops = S.seam_loops(self.co, self.faces, self.ids)
        self.size = float(np.ptp(self.co, axis=0).max())
        self.root = Path(root)
        (self.root / "returned").mkdir(parents=True)
        index = {tuple(p): i for i, p in enumerate(self.pts.tolist())}
        border = [[index[tuple(p)] for p in loop.tolist()] for loop in self.loops]
        self.manifest = {
            "format": "curiomesh-parts", "version": [1, 0], "made_by": "CurioMesh 0.9.9", "object": "Grid",
            "result": {"source_faces": 24, "source_verts": 35, "result_faces": len(self.faces), "result_key": L.result_key(len(self.faces), np.vstack(self.loops))},
            "parts": [{"id": 1, "name": "Part 2", "kind": "FLAT", "faces": len(self.local), "border": [b + b[:1] for b in border], "border_points": np.vstack(self.loops).tolist(), "materials": [0]}],
        }
        self.write_manifest()
        (self.root / "part_1.obj").write_text(obj(self.pts, self.local, self.mats))

    def write_manifest(self):
        (self.root / "parts.json").write_text(json.dumps(self.manifest))

    def give_back(self, points, faces):
        (self.root / "returned" / "part_1.obj").write_text(obj(points, faces))

    def bring_back(self):
        """What parts_import does without Blender: load, check the key, match, splice and weld, check holes and shape."""
        manifest = L.load_manifest(self.root)
        L.check_result(manifest, len(self.faces), np.vstack(self.loops), [0, 1])
        points, faces, mats = L.returned(self.root, 1)
        points, faces = L.match_border(points, faces, self.loops, self.size, part=1)
        co, out, _mat, _piece = P.splice(self.co, self.faces, self.mat, self.piece, {1: (points, faces, mats)})
        co, out = S.weld_border(co, out)
        L.check_no_hole(self.faces, out, part=1)
        L.check_shape(self.pts, self.local, points, faces, part=1)
        return co, out


def interior(f):
    border = {v for loop in boundary_loops(f.local, range(len(f.local)))[0] for v in loop}
    return [v for v in range(len(f.pts)) if v not in border]


class PartLoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Folder(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_interior_moved_stitches(self):
        f = self.f
        pts = f.pts.copy()
        inside = interior(f)
        self.assertTrue(inside)
        pts[inside] += [0.07, -0.05, 0.02]
        f.give_back(pts, f.local)
        co, out = f.bring_back()
        self.assertEqual(S.open_edges(out), S.open_edges(f.faces))
        self.assertEqual(len(co), len(f.co))  # the seam welded: no doubled vertices

    def test_b_border_point_moved(self):
        f = self.f
        pts = f.pts.copy()
        at = next(v for v in range(len(pts)) if v not in interior(f))
        pts[at] += [0.004, 0.0, 0.0]
        f.give_back(pts, f.local)
        with self.assertRaises(L.BorderMoved) as caught:
            f.bring_back()
        self.assertEqual(caught.exception.agent_code, "BORDER_MOVED")
        self.assertAlmostEqual(caught.exception.details["off"], 0.004, places=6)
        self.assertIn("loop", caught.exception.details)

    def test_c_missing_face_is_a_hole(self):
        f = self.f
        inside = set(interior(f))
        drop = next(i for i, face in enumerate(f.local) if all(v in inside for v in face))
        faces = [face for i, face in enumerate(f.local) if i != drop]
        f.give_back(f.pts, faces)
        with self.assertRaises((L.HoleError, L.BorderMoved)) as caught:
            f.bring_back()
        # a hole in the middle shows as a new border loop: the border check sees the extra open edges first
        self.assertIn(caught.exception.agent_code, ("HOLE", "BORDER_MOVED"))
        # the HOLE guard on its own: the same hole after splicing
        co, out, _m, _p = P.splice(f.co, f.faces, f.mat, f.piece, {1: (f.pts, faces, np.zeros(len(faces), np.int32))})
        co, out = S.weld_border(co, out)
        with self.assertRaises(L.HoleError) as caught:
            L.check_no_hole(f.faces, out, part=1)
        self.assertEqual(caught.exception.agent_code, "HOLE")
        self.assertGreater(caught.exception.details["open_after"], caught.exception.details["open_before"])

    def test_d_newer_version(self):
        f = self.f
        f.manifest["version"] = [2, 0]
        f.write_manifest()
        with self.assertRaises(L.VersionError) as caught:
            L.load_manifest(f.root)
        self.assertEqual(caught.exception.agent_code, "VERSION")
        self.assertIn("1.x", str(caught.exception))
        f.manifest["version"] = [1, 7]  # a newer minor only adds keys: still read
        f.manifest["new_key"] = {"anything": 1}
        f.write_manifest()
        L.load_manifest(f.root)

    def test_e_other_result(self):
        f = self.f
        f.manifest["result"]["result_key"] = "00000000"
        f.write_manifest()
        with self.assertRaises(L.MismatchError) as caught:
            f.bring_back()
        self.assertEqual(caught.exception.agent_code, "MISMATCH")
        with self.assertRaises(L.MismatchError):
            L.check_result(L.load_manifest(f.root), len(f.faces), np.vstack(f.loops), [0])  # part 1 is not in the table

    def test_f_renumbered_and_turned_over(self):
        f = self.f
        order = np.random.default_rng(3).permutation(len(f.pts))
        back = np.argsort(order)  # new number of each old vertex
        pts = f.pts[order]
        faces = [tuple(int(back[v]) for v in reversed(face)) for face in f.local]  # every face wound the other way
        f.give_back(pts, faces)
        co, out = f.bring_back()
        self.assertEqual(S.open_edges(out), S.open_edges(f.faces))
        half = [(a, b) for face in out for a, b in zip(face, face[1:] + face[:1])]
        self.assertEqual(len(half), len(set(half)), "faces turned back so every edge runs both ways once")

    def test_bad_files(self):
        f = self.f
        cases = {
            "v 0 0 nan\nv 1 0 0\nv 0 1 0\nf 1 2 3\n": "finite",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n": "not there",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 2\n": "same corner",
            "v 0 0 0\nv 1 0 0\nf 1 2\n": "corners",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl steel\nf 1 2 3\n": "usemtl",
            "v 0 0 0\n": "no faces",
        }
        for text, words in cases.items():
            (f.root / "returned" / "part_1.obj").write_text(text)
            with self.assertRaises(L.PartError) as caught:
                L.returned(f.root, 1)
            self.assertEqual(caught.exception.agent_code, "FILE")
            self.assertIn(words, str(caught.exception))
        (f.root / "returned" / "part_1.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nusemtl m2\nf -3 -1 -2\n")
        _pts, faces, mats = L.returned(f.root, 1, default_mat=5)
        self.assertEqual(mats.tolist(), [5, 2])
        self.assertEqual(faces[1], (1, 3, 2))
        self.assertIsNone(L.returned(f.root, 9))

    def test_shape_lost(self):
        f = self.f
        pts = f.pts.copy()
        pts[:, 1] *= 0.3
        with self.assertRaises(L.ShapeLost) as caught:
            L.check_shape(f.pts, f.local, pts, f.local, part=1)
        self.assertLess(caught.exception.details["area_share"], 0.7)


if __name__ == "__main__":
    unittest.main()
