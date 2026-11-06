# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import guides  # noqa: E402
import _addon  # noqa: E402
from triadq import flatquad  # noqa: E402
from triadq.budget import density_drop  # noqa: E402
from triadq.types import Cancelled, MeshData  # noqa: E402

symmetry = _addon.load("symmetry")


def _sheet(n=12):
    vertices = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)])
    faces = []
    for y in range(n):
        for x in range(n):
            a, b, c, d = y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x
            faces += [(a, b, c), (a, c, d)]
    return MeshData(vertices, faces)


class CancelTests(unittest.TestCase):
    def test_pairing_stops_when_asked(self):
        with self.assertRaises(Cancelled):
            flatquad.pair_and_close(_sheet(), cancelled=lambda: True)
        out, _report = flatquad.pair_and_close(_sheet())  # the stop flag does not leak into the next call on this thread
        self.assertTrue(out.faces)


def _cube(n=5):
    """A closed triangulated cube from -1 to 1 whose grid does not have a vertex plane at 0."""
    index, co, faces = {}, [], []
    for face_axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = (face_axis + 1) % 3, (face_axis + 2) % 3
            ids = {}
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0.0, 0.0, 0.0]
                    p[face_axis], p[u], p[v] = sign, -1.0 + 2.0 * i / n, -1.0 + 2.0 * j / n
                    key = tuple(round(x, 9) for x in p)
                    if key not in index:
                        index[key] = len(co)
                        co.append(p)
                    ids[i, j] = index[key]
            for i in range(n):
                for j in range(n):
                    a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
                    faces += [(a, b, c), (a, c, d)] if sign > 0 else [(a, c, b), (a, d, c)]
    return np.asarray(co), faces


class SymmetryTests(unittest.TestCase):
    def test_cut_and_mirror_gives_an_exact_closed_mirror(self):
        co, faces = _cube()
        half, half_faces, source, origin = symmetry.cut_half(co, faces, 0, 0.0)
        self.assertGreaterEqual(half[:, 0].min(), 0.0)
        self.assertEqual(len(source), len(half_faces))
        self.assertEqual(len(origin), len(half))
        out, out_faces, _mat, snapped, _kept = symmetry.mirror_result(half, half_faces, None, 0, 0.0, 3.5)
        self.assertGreater(snapped, 0)
        self.assertFalse(symmetry.border_vertices(out_faces, len(out)).any())  # no gap along the seam
        mirrored = out * np.array([-1.0, 1.0, 1.0])
        self.assertEqual({tuple(p) for p in out.tolist()}, {tuple(p) for p in mirrored.tolist()})  # exactly mirrored


    def test_seam_is_clean_when_the_half_strays_over_the_plane(self):
        co, faces = _cube()
        half, half_faces, _source, _origin = symmetry.cut_half(co, faces, 0, 0.0)
        half = half.copy()
        inner = (half[:, 0] > 0.3) & (half[:, 0] < 0.7)
        half[np.flatnonzero(inner)[:1], 0] = -0.05  # a solved vertex that crossed the plane
        on = np.flatnonzero(half[:, 0] == 0.0)
        a, b = next((f[k - 1], f[k]) for f in half_faces for k in range(len(f)) if f[k - 1] in on and f[k] in on)
        half[a] = 0.99 * half[b] + 0.01 * half[a]  # a seam vertex almost on top of its neighbour: a sliver once welded
        half[on, 0] += 0.003  # the border drifted a little off the plane
        out, out_faces, _mat, _snapped, kept = symmetry.mirror_result(half, half_faces, None, 0, 0.0, 3.5, 1)
        self.assertLess(len(kept), len(half_faces))  # the faces left without area by merging the two are gone
        check = symmetry.seam_check(out, out_faces, 0, 0.0, 3.5)
        self.assertGreater(check["seam_vertices"], 0)
        for kind in ("open", "non_manifold", "across", "duplicate"):
            self.assertEqual(check[kind], 0, kind)
        mirrored = out * np.array([-1.0, 1.0, 1.0])
        self.assertEqual({tuple(p) for p in out.tolist()}, {tuple(p) for p in mirrored.tolist()})

    def test_seam_check_counts_an_open_seam(self):
        co, faces = _cube()
        half, half_faces, _source, _origin = symmetry.cut_half(co, faces, 0, 0.0)
        self.assertGreater(symmetry.seam_check(half, half_faces, 0, 0.0, 3.5)["open"], 0)  # one half alone is open


class GuideTests(unittest.TestCase):
    def test_quads_keep_a_stroke_as_edges(self):
        sheet = _sheet(8)  # coarse triangles, 1 unit apart: the stroke runs between their vertices
        co = sheet.vertices / 8.0
        tris = np.asarray(sheet.faces)
        stroke = np.array([[0.1, 0.2, 0.0], [0.9, 0.5, 0.0]])
        co, tris, mat = guides.refine(co, tris, np.zeros(len(tris), dtype=np.int64), guides.resample(stroke, 0.05), 0.125)
        chains, keys = guides.chain(co, tris, [stroke], 0.125)
        co = guides.pull(co, tris, chains, [stroke], 0.125)
        self.assertTrue(keys)
        on_line = np.abs(co[np.unique(np.asarray(chains[0]))][:, 1] - (0.2 + 0.375 * (co[np.unique(np.asarray(chains[0]))][:, 0] - 0.1)))
        self.assertLess(on_line.max(), 0.02)  # the chain lies on the stroke
        out, _report = flatquad.pair_and_close(MeshData(co, [tuple(t) for t in tris.tolist()]), guide_edges=tuple(sorted(keys)))
        seg = [(out.vertices[f[k]], out.vertices[f[(k + 1) % len(f)]]) for f in out.faces for k in range(len(f))]
        p0, p1 = np.array([s[0] for s in seg]), np.array([s[1] for s in seg])
        d = p1 - p0
        dd = np.maximum((d * d).sum(axis=1), 1e-30)

        def on_an_edge(x):  # the output edges may split a guide edge in two, but they must still cover it
            t = np.clip(((x - p0) * d).sum(axis=1) / dd, 0.0, 1.0)
            return np.linalg.norm(p0 + t[:, None] * d - x, axis=1).min() < 1e-9

        kept = [on_an_edge(co[a] + (co[b] - co[a]) * t) for a, b in keys for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertTrue(all(kept))  # no quad was made across the stroke


class DensityTests(unittest.TestCase):
    def test_paint_lowers_the_decimate_weight_and_stays_small(self):
        drop = density_drop(np.array([0.0, 0.5, 1.0, 9.0]), 1.0)
        self.assertEqual(drop[0], 0.0)  # no paint: collapsed freely
        self.assertTrue(np.all(np.diff(drop) >= 0))
        self.assertAlmostEqual(drop[2], 0.02)  # full red at strength 1 (Decimate treats 0.98 as four times the faces)
        self.assertLessEqual(density_drop(np.array([1.0]), 4.0)[0], 0.5)  # never as low as 0, which locks the vertex

    def test_less_detail_collapses_first_and_more_detail_last(self):
        drop = density_drop(np.array([-1.0, 0.0, 1.0, -3.0]), 1.0)
        self.assertEqual(drop[0], 0.0)  # Less Detail: free
        self.assertEqual(drop[3], 0.0)
        self.assertAlmostEqual(drop[1], 0.02)  # unmarked holds on as More Detail alone would
        self.assertAlmostEqual(drop[2], 0.04)


if __name__ == "__main__":
    unittest.main()
