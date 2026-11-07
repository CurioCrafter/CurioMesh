# SPDX-License-Identifier: GPL-3.0-or-later
"""Classic path (A1c): pairing on UV meshes, no seams = same as the native pairing, per-side transfer drift."""

from __future__ import annotations

import unittest

import numpy as np

import uv_parity_common as common  # noqa: F401  (puts the add-on folder on the path)
from test_uv_parity_seam import grid, two_islands
from triadq import MeshData, uvclassic, uvseams
import triadq.engine as E


def plane_tris(split: bool):
    """8x8 plane as triangles; with `split` the UVs form two islands cut at x = 4 (a seam of shared vertices)."""

    points, indptr, flat = grid(8, 8, 8.0, 8.0)
    faces, uv_ids, uvs, ids = [], [], [], {}
    for q in range(64):
        a, b, c, d = flat[4 * q:4 * q + 4]
        side = int(split and q % 8 >= 4)
        for tri in ((a, b, c), (a, c, d)):
            faces.append(tuple(int(v) for v in tri))
            row = []
            for v in tri:
                key = (int(v), side)
                if key not in ids:
                    ids[key] = len(uvs)
                    uvs.append((points[v, 0] / 8 + 0.25 * side, points[v, 1] / 8))
                row.append(ids[key])
            uv_ids.append(tuple(row))
    return MeshData(points, faces, uvs=np.array(uvs), face_uvs=uv_ids)


class FakeTurbo:
    """pair_seeds stand-in: first-come pairing over the candidate list (same row layout as the real kernel)."""

    def pair_seeds(self, n, face_a, face_b, quads, scores, minimum, seeds):
        mate, found = {}, []
        for a, b, q in zip(face_a.tolist(), face_b.tolist(), quads.tolist()):
            if a not in mate and b not in mate:
                mate[a], mate[b] = b, a
                found.append((a, q))
        tris = np.array([i for i in range(n) if i not in mate], np.int64)
        owner = np.array([a for a, _ in found] + tris.tolist(), np.int64)
        return [(np.array([q for _, q in found], np.int64).reshape(-1, 4), tris, owner, 0) for _ in seeds]


def prepared(mesh, **options):
    tri = E._triangulate(mesh)
    opts = E.RemeshOptions(**options)
    mode = E._classify_mesh(tri, opts)
    return tri, E._pairing_prep(tri, E._options_for_mode(opts, mode), mode)


class Pairing(unittest.TestCase):
    def setUp(self):
        self.real = E._turbo

    def tearDown(self):
        E._turbo = self.real

    def test_no_seams_is_bit_identical(self):
        flat = plane_tris(False)
        flat.uvs = flat.face_uvs = None
        tri, prep = prepared(flat)
        for turbo in (self.real, FakeTurbo()):
            E._turbo = turbo
            want, got = E._pair_natively(tri, prep, [0, 1]), uvclassic.pair(tri, prep, [0, 1])
            self.assertEqual(want is None, got is None)
            if want is not None:
                for seed in (0, 1):
                    self.assertEqual(got[seed][0].faces, want[seed][0].faces)
                    self.assertEqual(got[seed][0].face_materials, want[seed][0].face_materials)
                    self.assertEqual(got[seed][1:], want[seed][1:])

    def test_seam_pairs_fall_back_per_face_python_and_native(self):
        mesh = plane_tris(True)
        tri, prep = prepared(mesh, preserve_uv_seams=False)  # seam edges are ordinary candidates
        for turbo in (None, FakeTurbo()):
            E._turbo = turbo
            got = uvclassic.pair(tri, prep, [0])[0][0]
            self.assertIsNotNone(got.face_uvs, "UVs must survive")
            self.assertGreater(sum(len(f) == 4 for f in got.faces), 20)
            for face, uvf in zip(got.faces, got.face_uvs):
                self.assertEqual(len(face), len(uvf))
                for u, v in zip(uvf, face):
                    self.assertAlmostEqual(got.uvs[u][1], got.vertices[v, 1] / 8)
                    self.assertIn(round(got.uvs[u][0] - got.vertices[v, 0] / 8, 6), (0.0, 0.25))
                if len(face) == 4:  # a pair across the seam would put one island's UV on the other's vertex
                    self.assertEqual(len({round(got.uvs[u][0] - got.vertices[v, 0] / 8, 6) for u, v in zip(uvf, face)}), 1)


    def test_out_of_range_uv_ids_stay_triangles_and_in_range(self):
        mesh = plane_tris(False)
        mesh.face_uvs[10] = (len(mesh.uvs) + 5, -1, 0)  # a triangle with garbage UV ids
        mesh.face_uvs[11] = (0, 0, 10 ** 6)
        tri, prep = prepared(mesh)
        bad_corners = {tuple(mesh.faces[10]), tuple(mesh.faces[11])}
        for turbo in (None, self.real, FakeTurbo()):
            E._turbo = turbo
            got = uvclassic.pair(tri, prep, [0])[0][0]
            self.assertIsNotNone(got.face_uvs)
            self.assertLessEqual(max(max(f) for f in got.face_uvs), len(got.uvs) - 1)
            self.assertGreaterEqual(min(min(f) for f in got.face_uvs), 0)
            tris_out = {tuple(f) for f in got.faces if len(f) == 3}
            self.assertTrue(bad_corners <= tris_out, "faces with bad UV ids must stay triangles")


class Obj:
    """Stand-in for a Blender mesh object built from polygons."""

    def __init__(self, points, indptr, flat, layers):
        cols = {"co": points.ravel(), "loop_total": np.diff(indptr), "vertex_index": flat, "material_index": np.zeros(len(indptr) - 1, np.int32)}

        class Seq(list):
            def foreach_get(self, key, out):
                out[:] = cols[key]

        class Layer:
            def __init__(self, name, uv):
                self.name, self.data, self.values = name, self, np.array(uv, np.float32).ravel()

            def foreach_get(self, key, out):
                out[:] = self.values

            def foreach_set(self, key, values):
                self.values = np.array(values, np.float32)

        class Layers(list):
            active = None

            def get(self, name):
                return next((l for l in self if l.name == name), None)

            def new(self, name):
                self.append(Layer(name, np.zeros((len(flat), 2))))
                return self[-1]

        mesh = type("Mesh", (), {})()
        mesh.vertices, mesh.polygons, mesh.loops = Seq(range(len(points))), Seq(range(len(indptr) - 1)), Seq(range(len(flat)))
        mesh.uv_layers = Layers(Layer(name, uv) for name, uv in layers.items())
        mesh.uv_layers.active = mesh.uv_layers[0] if mesh.uv_layers else None
        self.data, self.matrix_world = mesh, None

    def __setitem__(self, key, value):
        pass


class Transfer(unittest.TestCase):
    def setUp(self):
        self.saved = dict(uvseams.FAST), uvseams.transfer, uvseams.layer
        uvseams.FAST["transfer"] = object()  # "native core present"; the calls below run the NumPy twin
        t, l = self.saved[1:]
        uvseams.transfer = lambda *a, **k: t(*a, use_fast=False, **k)
        uvseams.layer = lambda *a, **k: l(*a, use_fast=False, **k)

    def tearDown(self):
        uvseams.FAST.clear()
        uvseams.FAST.update(self.saved[0])
        uvseams.transfer, uvseams.layer = self.saved[1:]

    def test_island_aware_drift_and_all_layers(self):
        points, indptr, flat, uv = two_islands()
        res = points.copy()
        res[:, 0] = 4 + (res[:, 0] - 4) * 1.01
        res[:, 2] = 0.2
        source = Obj(points, indptr, flat, {"A": uv.reshape(-1, 2), "B": uv.reshape(-1, 2) * 0.5})
        target = Obj(res, indptr, flat, {})
        config = type("C", (), {"texture_mode": "PROJECT"})()
        self.assertEqual(uvclassic.preserve_textures(source, target, config), "TRANSFERRED")
        right = np.repeat((np.arange(64) % 8) >= 4, 4)
        want = np.clip(res[flat, 0], 0, 8) / 8 + 0.25 * right
        got = target.data.uv_layers.get("A").values.reshape(-1, 2)
        self.assertLess(float(np.percentile(np.abs(got[:, 0] - want), 99)) * 1024, 8.0)
        self.assertEqual(target.data.uv_layers.active.name, "A")
        b = target.data.uv_layers.get("B").values.reshape(-1, 2)
        self.assertLess(float(np.abs(b - got * 0.5).max()) * 1024, 8.0)

    def test_without_native_or_uvs_defers_to_old_code(self):
        points, indptr, flat, uv = two_islands()
        source = Obj(points, indptr, flat, {"A": uv.reshape(-1, 2)})
        target = Obj(points, indptr, flat, {})
        config = type("C", (), {"texture_mode": "PROJECT"})()
        self.assertIsNone(uvclassic.preserve_textures(Obj(points, indptr, flat, {}), target, config))  # no source UVs
        config.texture_mode = "BAKE"
        self.assertIsNone(uvclassic.preserve_textures(source, target, config))  # not a transfer mode
        config.texture_mode = "PROJECT"
        uvseams.FAST.clear()
        self.assertIsNone(uvclassic.preserve_textures(source, target, config))  # no native core


if __name__ == "__main__":
    unittest.main()
