# SPDX-License-Identifier: GPL-3.0-or-later
"""CurioMesh Turbo: native kernels must match the Python code they replace.

Skips cleanly when the native module is not built. `python tests/test_turbo.py
--bench` prints the full Python-vs-native table (50k / 500k faces, minutes).
"""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import random
import sys
import time
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import turbo  # noqa: E402
from triadq import MeshData  # noqa: E402
import triadq.engine as E  # noqa: E402
import triadq.field as F  # noqa: E402
import triadq.flatquad as FQ  # noqa: E402
import triadq.patch as P  # noqa: E402
import triadq.quadreduce as QR  # noqa: E402
import triadq.topology as T  # noqa: E402


def torus(nu: int, nv: int, noise: float = 0.01, seed: int = 0, quads: float = 0.0) -> MeshData:
    """Noisy torus; each grid cell is a quad with probability `quads`, else two triangles."""

    rng = np.random.default_rng(seed)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ring = 1.0 + 0.35 * np.cos(vv)
    verts = np.stack([ring * np.cos(uu), ring * np.sin(uu), 0.35 * np.sin(vv)], -1).reshape(-1, 3)
    verts += rng.normal(scale=noise * 0.35 * 2 * np.pi / nv, size=verts.shape)
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a, b = i * nv + j, ((i + 1) % nu) * nv + j
    c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
    flip = rng.random(a.shape) < 0.5
    keep_quad = rng.random(a.shape) < quads
    faces = []
    for cell in zip(a.ravel(), b.ravel(), c.ravel(), d.ravel(), flip.ravel(), keep_quad.ravel()):
        p, q, r, s = map(int, cell[:4])
        f, k = bool(cell[4]), bool(cell[5])
        if k:
            faces.append((p, q, r, s))
        elif f:
            faces += [(p, q, r), (p, r, s)]
        else:
            faces += [(p, q, s), (q, r, s)]
    return MeshData(vertices=verts, faces=faces)


def grid(n: int, seed: int = 0, quads: float = 0.5, with_uvs: bool = False) -> MeshData:
    """Open n x n patch (has a boundary) with random tri/quad cells and per-face UV corners."""

    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n + 1, dtype=float), np.arange(n + 1, dtype=float), indexing="ij")
    verts = np.stack([xs, ys, 0.05 * rng.standard_normal(xs.shape)], -1).reshape(-1, 3)
    faces, mats = [], []
    for x in range(n):
        for y in range(n):
            a, b, c, d = x * (n + 1) + y, (x + 1) * (n + 1) + y, (x + 1) * (n + 1) + y + 1, x * (n + 1) + y + 1
            mat = f"m{(x // 3 + y // 4) % 3}"
            if rng.random() < quads:
                faces.append((a, b, c, d))
                mats.append(mat)
            else:
                faces += [(a, b, c), (a, c, d)]
                mats += [mat, mat]
    if not with_uvs:
        return MeshData(vertices=verts, faces=faces, face_materials=mats)
    uvs, face_uvs = [], []
    for face in faces:
        face_uvs.append(tuple(range(len(uvs), len(uvs) + len(face))))
        uvs.extend(verts[list(face), :2] / n)
    return MeshData(vertices=verts, faces=faces, face_materials=mats, uvs=np.asarray(uvs), face_uvs=face_uvs)


def messy(seed: int) -> MeshData:
    """Random polygon soup with bow-ties, duplicates, flipped faces and non-manifold edges."""

    rng = random.Random(seed)
    count = 40
    faces = []
    for _ in range(120):
        size = rng.choice((3, 3, 3, 4, 4, 5))
        faces.append(tuple(rng.sample(range(count), size)))
    faces += [faces[3], tuple(reversed(faces[7])), faces[9][1:] + faces[9][:1]]
    faces += [(0, 1, 2), (0, 2, 1), (0, 1, 3), (0, 1, 4), (5, 6, 7), (5, 8, 9)]
    verts = np.random.default_rng(seed).standard_normal((count + 3, 3))  # 3 unused vertices
    return MeshData(vertices=verts, faces=faces)


def meshes() -> list[MeshData]:
    return [
        torus(24, 16, seed=1),
        torus(20, 12, seed=2, quads=0.6),
        grid(14, seed=3),
        grid(9, seed=4, quads=1.0),
        *(messy(seed) for seed in range(6)),
        MeshData(vertices=np.zeros((4, 3)), faces=[(0, 1, 2), (0, 2, 3)]),  # degenerate
    ]


def _pair_stage(mesh: MeshData, seed: int = 0) -> MeshData:
    """Engine output before closure: a mixed tri/quad mesh, the ParityPath input."""

    tri = E._triangulate(mesh)
    mode = E._classify_mesh(tri, E.RemeshOptions())
    prep = E._pairing_prep(tri, E._options_for_mode(E.RemeshOptions(), mode), mode)
    return E._pair_triangles(tri, prep, seed)[0]


def py(fn, *args, **kwargs):
    """Run the pure-Python reference: triadq calls Turbo itself once it is wired in."""

    turbo.set_enabled(False)
    try:
        return fn(*args, **kwargs)
    finally:
        turbo.set_enabled(True)


def _outcome(fn, *args, **kwargs):
    try:
        return "ok", fn(*args, **kwargs)
    except ValueError as exc:
        return "error", str(exc)


def _same_mesh(test: unittest.TestCase, a: MeshData, b: MeshData) -> None:
    test.assertTrue(np.array_equal(a.vertices, b.vertices))
    test.assertEqual(a.faces, b.faces)
    test.assertEqual(a.face_materials, b.face_materials)
    test.assertEqual(a.name, b.name)
    test.assertEqual(a.face_uvs, b.face_uvs)
    test.assertEqual(a.uvs is None, b.uvs is None)
    if a.uvs is not None:
        test.assertTrue(np.array_equal(a.uvs, b.uvs))


@unittest.skipUnless(turbo.available(), f"native module not built: {turbo.status()}")
class TurboEquivalenceTests(unittest.TestCase):
    def test_topology_summary(self) -> None:
        for mesh in meshes():
            self.assertEqual(turbo.topology_summary(mesh), py(T.topology_summary, mesh))

    def test_edge_occurrences_and_fans(self) -> None:
        for mesh in meshes():
            expected = T._edge_occurrences(mesh.faces)
            fast = turbo.edge_occurrences(mesh.faces)
            self.assertEqual(list(fast.items()), list(expected.items()))
            self.assertEqual(
                turbo.non_manifold_vertex_count(mesh.faces), T._non_manifold_vertex_count(mesh.faces, expected)
            )

    def test_quad_only_closure(self) -> None:
        cases = [*meshes(), grid(8, seed=5, with_uvs=True)]
        broken_uv = grid(6, seed=6, with_uvs=True)
        broken_uv.face_uvs[10] = (0, 1)  # invalid corners: face UVs are dropped from here on
        cases.append(broken_uv)
        base = grid(5, seed=7)
        # Boundary edge (0, 1) gets an existing midpoint vertex, as fixed-boundary patches do.
        middle = (base.vertices[0] + base.vertices[1]) * 0.5
        frozen_mesh = base.copy_with(vertices=np.vstack((base.vertices, middle)))
        frozen = {(0, 1): len(base.vertices)}
        for mesh in cases:
            expected = _outcome(py, T.quad_only_closure, mesh, name="x")
            fast = _outcome(turbo.quad_only_closure, mesh, name="x")
            self.assertEqual(fast[0], expected[0], fast[1] if fast[0] == "error" else expected[1])
            if expected[0] == "error":
                self.assertEqual(fast[1], expected[1])
            else:
                _same_mesh(self, fast[1], expected[1])
        _same_mesh(
            self,
            turbo.quad_only_closure(frozen_mesh, frozen_midpoints=frozen),
            py(T.quad_only_closure, frozen_mesh, frozen_midpoints=frozen),
        )

    def test_parity_path_edges(self) -> None:
        cases = [_pair_stage(torus(24, 16, seed=s)) for s in range(3)]
        cases += [_pair_stage(grid(12, seed=s)) for s in range(3)]
        cases += [grid(10, seed=8), torus(16, 10, seed=9, quads=0.7), messy(1)]
        for mesh in cases:
            expected = _outcome(T._parity_path_edges, mesh)
            self.assertEqual(_outcome(turbo.parity_path_edges, mesh), expected)
        frozen = frozenset({(0, 1), (1, 2)})
        self.assertEqual(turbo.parity_path_edges(cases[3], frozen), T._parity_path_edges(cases[3], frozen))

    def test_matching(self) -> None:
        graphs = []
        for seed in range(25):  # random graphs with odd cycles (blossoms)
            rng = random.Random(seed)
            n = rng.randint(6, 80)
            allowed = defaultdict(list)
            for _ in range(rng.randint(n, 3 * n)):
                a, b = rng.sample(range(n), 2)
                score = rng.random()
                if b not in {item[1] for item in allowed[a]}:
                    allowed[a].append((score, b, ()))
                    allowed[b].append((score, a, ()))
            for linked in allowed.values():
                linked.sort(key=lambda item: item[0], reverse=True)
            order = list(range(n))
            rng.shuffle(order)
            graphs.append((dict(allowed), order))
        tri = E._triangulate(torus(30, 20, seed=3))
        prep = E._pairing_prep(tri, E.RemeshOptions(), "OrganicFlow")
        order = list(range(len(tri.faces)))
        random.Random(1).shuffle(order)
        graphs.append((prep.allowed, order))
        for allowed, order in graphs:
            mate: dict[int, int] = {}
            for face in order:  # the engine's greedy start
                if face in mate:
                    continue
                for _score, other, _quad in allowed.get(face, ()):
                    if other not in mate:
                        mate[face], mate[other] = other, face
                        break
            python_mate, native_mate = dict(mate), dict(mate)
            rescued = E._augment_matching(python_mate, allowed, order)
            self.assertEqual(turbo.augment_matching(native_mate, allowed, order), rescued)
            self.assertEqual(native_mate, python_mate)

    def test_winding_number(self) -> None:
        rng = np.random.default_rng(0)
        mesh = torus(18, 12, seed=4)
        triangles = mesh.vertices[np.asarray(mesh.faces)]
        triangles[::7] = triangles[::7, ::-1]  # some flipped triangles
        points = rng.uniform(-1.6, 1.6, (3000, 3))
        np.testing.assert_allclose(
            turbo.winding_number(points, triangles), py(P.winding_number, points, triangles), rtol=0, atol=1e-9
        )
        self.assertTrue(np.array_equal(turbo.inside_volume(points, triangles), py(P.inside_volume, points, triangles)))

    def _field_inputs(self, mesh: MeshData):
        tri = E._triangulate(mesh)
        adj = E._adjacency(tri)
        features = E._feature_mask(tri, adj, E.RemeshOptions())
        e1, e2, sides, side_angle = F._frames(tri, adj)
        link = (adj.counts == 2) & ~features
        half_a, half_b = adj.edge_halves[link].T
        face_a, corner_a = np.divmod(half_a, 3)
        face_b, corner_b = np.divmod(half_b, 3)
        transport = np.exp(4j * (side_angle[face_a, corner_a] - side_angle[face_b, corner_b]))
        rng = np.random.default_rng(1)
        cross = np.exp(1j * rng.uniform(0, 2 * np.pi, len(adj.faces))) * rng.uniform(0.05, 1.0, len(adj.faces))
        pinned = rng.random(len(adj.faces)) < 0.1
        return face_a, face_b, transport, cross, pinned

    def test_cross_field_loops(self) -> None:
        face_a, face_b, transport, cross0, pinned = self._field_inputs(torus(30, 20, seed=5))
        count = len(cross0)
        # engine._estimate_field loop, verbatim
        degree = 1.0 + np.bincount(face_a, minlength=count) + np.bincount(face_b, minlength=count)
        cross = cross0.copy()
        for _ in range(48):
            into_a = cross[face_b] * transport
            into_b = cross[face_a] * np.conj(transport)
            real = np.bincount(face_a, weights=into_a.real, minlength=count) + np.bincount(
                face_b, weights=into_b.real, minlength=count
            )
            imag = np.bincount(face_a, weights=into_a.imag, minlength=count) + np.bincount(
                face_b, weights=into_b.imag, minlength=count
            )
            cross = np.where(pinned, cross, (cross + real + 1j * imag) / degree)
        fast = turbo.cross_field_jacobi(cross0, face_a, face_b, transport, pinned, degree, 48)
        np.testing.assert_allclose(fast, cross, rtol=0, atol=1e-12)

        # field.estimate_field_gl loop, verbatim
        rng = np.random.default_rng(2)
        pull = np.where(pinned, 0.0, 0.5 * rng.random(count))
        curv = np.exp(1j * rng.uniform(0, 2 * np.pi, count))
        idx = np.concatenate((face_a, face_b))
        denom = np.bincount(idx, minlength=count) + 1.0 + pull
        pin_term = pull * curv
        steps = 48
        cross = cross0.copy()
        for step in range(steps):
            val = np.concatenate((cross[face_b] * transport, cross[face_a] * np.conj(transport)))
            acc = np.bincount(idx, weights=val.real, minlength=count) + 1j * np.bincount(
                idx, weights=val.imag, minlength=count
            )
            relaxed = (acc + 1.0 * cross + pin_term) / denom
            if step >= steps // 2:
                relaxed = relaxed + 0.4 * (1.0 - np.abs(relaxed) ** 2) * relaxed
            cross = np.where(pinned, cross, relaxed)
        fast = turbo.cross_field_gl(cross0, face_a, face_b, transport, pinned, denom, pin_term, steps)
        np.testing.assert_allclose(fast, cross, rtol=0, atol=1e-12)

    def test_wired_remesh_matches_python(self) -> None:
        """triadq with Turbo wired in gives the same remesh as pure Python."""

        for mesh, options in (
            (torus(30, 20, seed=7), E.RemeshOptions(target_faces=600, force_quads=True, seed_count=2)),
            (grid(12, seed=8, with_uvs=True), E.RemeshOptions(target_faces=150, force_quads=True, seed_count=2)),
        ):
            fast, fast_report = E.remesh_mesh(mesh, options)
            slow, slow_report = py(E.remesh_mesh, mesh, options)
            _same_mesh(self, fast, slow)
            self.assertEqual(fast_report.closure_method, slow_report.closure_method)

    def test_disable_switch(self) -> None:
        os.environ["CURIOMESH_TURBO"] = "0"
        try:
            self.assertFalse(turbo.available())
            self.assertIsNone(turbo.topology_summary(torus(6, 4)))
        finally:
            del os.environ["CURIOMESH_TURBO"]
        self.assertTrue(turbo.available())

    def test_speed_report(self) -> None:
        """Printed, never asserted."""

        mesh = torus(100, 100)
        rows = [
            ("topology_summary 20k", lambda: py(T.topology_summary, mesh), lambda: turbo.topology_summary(mesh)),
            ("quad_only_closure 20k", lambda: py(T.quad_only_closure, mesh), lambda: turbo.quad_only_closure(mesh)),
        ]
        print()
        for label, slow, fast in rows:
            print(_timed_row(label, slow, fast))


def _timed_row(label: str, slow, fast) -> str:
    start = time.perf_counter()
    slow()
    python = time.perf_counter() - start
    start = time.perf_counter()
    fast()
    native = time.perf_counter() - start
    return f"  {label:<36} python {python:8.2f}s  native {native:7.3f}s  x{python / max(native, 1e-9):7.1f}"


def bench() -> None:
    """Full table for the report: per function and end to end, 50k and 500k faces."""

    print(turbo.status())
    m50, m500 = torus(250, 100), torus(500, 500)
    rows = [
        ("topology_summary 50k", lambda: py(T.topology_summary, m50), lambda: turbo.topology_summary(m50)),
        ("topology_summary 500k", lambda: py(T.topology_summary, m500), lambda: turbo.topology_summary(m500)),
        (
            "_edge_occurrences 500k",
            lambda: T._edge_occurrences(m500.faces),
            lambda: turbo.edge_occurrences(m500.faces),
        ),
        (
            "_non_manifold_vertex_count 500k",
            lambda: T._non_manifold_vertex_count(m500.faces, T._edge_occurrences(m500.faces)),
            lambda: turbo.non_manifold_vertex_count(m500.faces),
        ),
        ("quad_only_closure 500k", lambda: py(T.quad_only_closure, m500), lambda: turbo.quad_only_closure(m500)),
    ]
    mixed = _pair_stage(m50)
    rows.append(("_parity_path_edges 50k", lambda: T._parity_path_edges(mixed), lambda: turbo.parity_path_edges(mixed)))
    tri = E._triangulate(m50)
    prep = E._pairing_prep(tri, E.RemeshOptions(), "OrganicFlow")
    order = list(range(len(tri.faces)))
    random.Random(0).shuffle(order)
    greedy: dict[int, int] = {}
    for face in order:
        if face in greedy:
            continue
        for _s, other, _q in prep.allowed.get(face, ()):
            if other not in greedy:
                greedy[face], greedy[other] = other, face
                break
    rows.append(
        (
            "_augment_matching 50k",
            lambda: E._augment_matching(dict(greedy), prep.allowed, order),
            lambda: turbo.augment_matching(dict(greedy), prep.allowed, order),
        )
    )
    small = torus(60, 40)
    triangles = small.vertices[np.asarray(small.faces)]
    points = np.random.default_rng(1).uniform(-1.5, 1.5, (20000, 3))
    rows.append(
        (
            "inside_volume 20k pts x 4.8k tris",
            lambda: py(P.inside_volume, points, triangles),
            lambda: turbo.inside_volume(points, triangles),
        )
    )
    for label, slow, fast in rows:
        print(_timed_row(label, slow, fast), flush=True)

    # End to end: triadq calls the native kernels itself.
    options = E.RemeshOptions(target_faces=25000, force_quads=True)
    print(
        _timed_row(
            "remesh_mesh 50k (8 seeds, quads)",
            lambda: py(E.remesh_mesh, m50, options),
            lambda: E.remesh_mesh(m50, options),
        ),
        flush=True,
    )


@unittest.skipUnless(turbo.available(), f"native module not built: {turbo.status()}")
class ClosureValidityTests(unittest.TestCase):
    def test_quad_validity_matches_python(self):
        rng = np.random.default_rng(11)
        square = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        points = square + rng.normal(scale=0.4, size=(20000, 4, 3))
        points[::50, 2] = points[::50, 0]  # repeated corner
        points[::77] = points[::77, ::-1]  # flipped
        points[3, 1, 1] = np.nan
        for epsilon in (1e-20, 1e-3, np.linspace(0.0, 0.2, len(points))):
            np.testing.assert_array_equal(T._quad_points_valid(points, epsilon), py(T._quad_points_valid, points, epsilon))

    def test_templates_match_python(self):
        for mesh in meshes():
            if any(len(face) not in (3, 4) for face in mesh.faces):
                continue
            selected = py(T._parity_path_edges, mesh, frozenset())
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            mids = {edge: len(vertices) + k for k, edge in enumerate(sorted(selected))}
            middles = [(vertices[a] + vertices[b]) * 0.5 for a, b in sorted(selected)]
            base = np.vstack([vertices, *middles]) if middles else vertices
            self.assertEqual(
                T._parity_template_validity(mesh, vertices, base, selected, mids),
                py(T._parity_template_validity, mesh, vertices, base, selected, mids),
            )


class PicturesTests(unittest.TestCase):
    def test_falls_back_without_native_or_on_ngons(self):
        square = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 2, 0]])
        self.assertIsNone(py(turbo.pictures, square, [(0, 1, 2)], np.zeros(3), 2.0, 8))
        self.assertIsNone(turbo.pictures(square, [(0, 1, 2, 4, 3)], np.zeros(3), 2.0, 8))

    @unittest.skipUnless(turbo.available(), f"native module not built: {turbo.status()}")
    def test_bent_quad_reports_its_own_normal(self):
        bent = np.array([[-1.0, -1, 0.5], [1, -1, -0.5], [1, 1, 0.5], [-1, 1, -0.5]])
        hit, normal, _depth, _face = turbo.pictures(bent, [(0, 1, 2, 3)], np.zeros(3), 2.0, 8)[4]  # from above
        self.assertTrue(hit.any())
        np.testing.assert_allclose(np.abs(normal[hit]), [[0.0, 0.0, 1.0]] * int(hit.sum()), atol=1e-9)

    @unittest.skipUnless(turbo.available(), f"native module not built: {turbo.status()}")
    def test_same_as_one_ray_per_pixel(self):
        mesh = torus(16, 10, seed=3)
        center, half, pixels = np.array([0.1, -0.05, 0.02]), 1.6, 24
        fast = turbo.pictures(mesh.vertices, mesh.faces, center, half, pixels)
        tris = np.array([(f[0], f[k], f[k + 1]) for f in mesh.faces for k in range(1, len(f) - 1)])
        a, b, c = np.moveaxis(mesh.vertices[tris], 1, 0)
        view = 0
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for sign in (1.0, -1.0):
                grid = ((np.arange(pixels) + 0.5) / pixels * 2.0 - 1.0) * half
                origins = np.zeros((pixels * pixels, 3))
                origins[:, axis] = center[axis] + sign * 2.0 * half
                origins[:, u] = center[u] + np.repeat(grid, pixels)
                origins[:, v] = center[v] + np.tile(grid, pixels)
                direction = np.zeros(3)
                direction[axis] = -sign
                # Moller-Trumbore, every ray against every triangle.
                e1, e2 = b - a, c - a
                h = np.cross(direction, e2)
                det = (e1 * h).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = origins[:, None] - a
                    bary_u = (s * h).sum(axis=2) / det
                    q = np.cross(s, e1)
                    bary_v = (q * direction).sum(axis=2) / det
                    t = (q * e2).sum(axis=2) / det
                t[~((bary_u >= 0) & (bary_v >= 0) & (bary_u + bary_v <= 1) & (t >= 0))] = np.inf
                first = t.argmin(axis=1)
                depth = t.min(axis=1).reshape(pixels, pixels)
                hit = np.isfinite(depth)
                normal = np.cross(e1, e2)[first]
                normal = (normal / np.linalg.norm(normal, axis=1, keepdims=True)).reshape(pixels, pixels, 3)
                got_hit, got_normal, got_depth, got_face = fast[view]
                self.assertTrue(((got_face >= 0) & (got_face < len(mesh.faces))).all())
                both = hit & got_hit
                self.assertTrue(hit.any())
                self.assertLessEqual(int((hit ^ got_hit).sum()), 1)  # a sample exactly on an outline may go either way
                np.testing.assert_allclose(got_depth[both], depth[both], atol=1e-9)
                self.assertGreater(float(np.abs((got_normal[both] * normal[both]).sum(axis=1)).min()), 0.999)
                view += 1


class GraphicsCardPicturesTests(unittest.TestCase):
    """The card draws `turbo.pictures` too. Pixels may pick a neighbouring triangle on a depth tie (the card works in f32),
    so the check is what the camera counts: `views.picture_counts` of CPU pictures against card pictures."""

    @staticmethod
    def _inputs():
        mesh = torus(90, 60, seed=5)
        tris = np.array([(f[0], f[k], f[k + 1]) for f in mesh.faces for k in range(1, len(f) - 1)], dtype=np.int64)
        a, b, c = np.moveaxis(mesh.vertices[tris], 1, 0)
        return mesh.vertices, tris, np.cross(b - a, c - a), [0.1, -0.05, 0.02], 1.6

    @unittest.skipUnless(turbo.available() and hasattr(turbo.native(), "gpu_info"), "native module has no card code")
    def test_card_pictures_count_the_same(self):
        module = turbo.native()
        if not module.gpu_info()[0]:
            self.skipTest("no graphics card: " + module.gpu_info()[1])
        from triadq import views

        vertices, tris, normals, center, half = self._inputs()
        for pixels in (32, 96):
            args = (vertices.reshape(-1), tris.reshape(-1), normals.reshape(-1), center, half, pixels)
            cpu = module.pictures(*args)
            try:
                card = module.pictures(*args, True)
            except RuntimeError as exc:  # an adapter that opens but cannot run the shader (a virtual machine's)
                self.skipTest(f"graphics card failed: {exc}")
            shape = (6, pixels, pixels)
            as_views = lambda out: [(out[0].reshape(shape)[i], out[1].reshape((*shape, 3))[i], out[2].reshape(shape)[i], out[3].reshape(shape)[i]) for i in range(6)]  # noqa: E731
            wrong, total = views.picture_counts(as_views(cpu), as_views(card), 2.0 * half / pixels)
            self.assertGreater(total, 100)
            self.assertLess(wrong / total, 0.01)

    @unittest.skipUnless(turbo.available() and hasattr(turbo.native(), "gpu_info"), "native module has no card code")
    def test_card_error_falls_back_and_says_so_once(self):
        from unittest import mock

        module = turbo.native()
        vertices, tris, _normals, center, half = self._inputs()
        polys = [tuple(t) for t in tris]
        real = module.pictures

        def broken(*args):
            if len(args) > 6 and args[6]:
                raise RuntimeError("the driver is on fire")
            return real(*args)

        want = turbo.pictures(vertices, polys, np.array(center), half, 24)
        old = (turbo.GPU_MIN_TRIANGLES, turbo._gpu_broken, turbo._gpu_unsaid)
        try:
            turbo.GPU_MIN_TRIANGLES, turbo._gpu_broken, turbo._gpu_unsaid = 0, "", False
            with mock.patch.object(module, "pictures", broken):
                got = turbo.pictures(vertices, polys, np.array(center), half, 24)
                self.assertIn("driver is on fire", turbo.gpu_note())
                self.assertIsNone(turbo.gpu_note())  # told once
                turbo.pictures(vertices, polys, np.array(center), half, 24)  # and not tried again
                self.assertIsNone(turbo.gpu_note())
            for a, b in zip(want, got):
                np.testing.assert_array_equal(a[0], b[0])
            turbo.set_gpu(False)
            self.assertIsNone(turbo.gpu_name())
        finally:
            turbo.set_gpu(True)
            turbo.GPU_MIN_TRIANGLES, turbo._gpu_broken, turbo._gpu_unsaid = old


def _drained(generator) -> None:
    for _ in generator:
        pass


@unittest.skipUnless(hasattr(turbo.native(), "pair_seeds"), "native module without the pairing kernels")
class SpeedPathTests(unittest.TestCase):
    """The kernels that make a big scan quicker must give what their Python functions give, face for face."""

    def test_shuffle_is_python_s(self) -> None:
        for count, seed in ((2, 0), (10, 1), (1000, 0), (99_991, 7)):
            order = list(range(count))
            random.Random(seed).shuffle(order)
            self.assertEqual(turbo.native().shuffled(count, seed).tolist(), order)

    def test_pairing_side_by_side(self) -> None:
        for mesh in meshes():
            if not all(len(face) == 3 for face in mesh.faces):
                mesh = MeshData(mesh.vertices, [tri for tri in E._triangulate(mesh).faces])
            tri = E._triangulate(mesh)
            for options in (E.RemeshOptions(), E.RemeshOptions(max_pairing=True)):
                mode = E._classify_mesh(tri, options)
                prep = E._pairing_prep(tri, E._options_for_mode(options, mode), mode)
                native = E._pair_natively(tri, prep, [0, 1, 2])
                for seed in range(3):
                    slow = py(E._pair_triangles, tri, prep, seed)
                    _same_mesh(self, native[seed][0], slow[0])
                    self.assertEqual(native[seed][2], slow[2])

    def test_make_closable(self) -> None:
        for mesh in meshes():
            fast, slow = FQ.make_closable(mesh), py(FQ.make_closable, mesh)
            _same_mesh(self, fast[0], slow[0])
            self.assertEqual(fast[1], slow[1])

    def test_close_pieces(self) -> None:
        for mesh in meshes():
            paired = FQ.make_closable(_pair_stage(mesh))[0]
            if not paired.faces:
                continue
            slow = _outcome(py, FQ._close_pieces, paired)
            fast = _outcome(FQ._close_pieces, paired)
            self.assertEqual(fast[0], slow[0])
            if slow[0] == "ok":
                _same_mesh(self, fast[1][0], slow[1][0])
                self.assertEqual(fast[1][1:], slow[1][1:])

    def test_close_pieces_many_parts(self) -> None:
        parts = [torus(10 + k, 8, seed=k) for k in range(6)]
        points, faces, offset = [], [], 0
        for part in parts:
            points.append(part.vertices + 5.0 * len(points))
            faces += [tuple(v + offset for v in face) for face in part.faces]
            offset += len(part.vertices)
        paired = _pair_stage(MeshData(np.vstack(points), faces))
        fast, slow = FQ._close_pieces(paired), py(FQ._close_pieces, paired)
        _same_mesh(self, fast[0], slow[0])
        self.assertEqual(fast[1:], slow[1:])

    def test_join_loose_triangles(self) -> None:
        for mesh in meshes():
            paired = FQ.make_closable(_pair_stage(mesh))[0]
            if not paired.faces:
                continue
            fast, slow = QR.join_loose_triangles(paired), py(QR.join_loose_triangles, paired)
            _same_mesh(self, fast[0], slow[0])
            self.assertEqual(fast[1], slow[1])

    def test_quad_reducer(self) -> None:
        closed = FQ.pair_and_close(torus(40, 24, seed=5))[0]
        limit = QR.DISSOLVE_SHARE * float(np.linalg.norm(np.ptp(closed.vertices, axis=0)))
        fast, slow = turbo.quad_reducer(closed, limit, QR.MOST_TURNED), py(QR.QuadReducer, closed)
        for step in range(2):
            for reducer in (fast, slow):
                _drained(reducer.remove(int(0.1 * reducer.alive)))
            _same_mesh(self, fast.result(), slow.result())
            self.assertEqual(fast.alive, slow.alive)
            self.assertEqual(fast.refused, len(slow.refused))
            if step == 0:
                saved_fast, saved_slow = fast.save(), slow.save()
        some = list(fast.result().faces[0])
        self.assertEqual(fast.original_ids(some), slow.original_ids(some))
        fast.restore(saved_fast)
        slow.restore(saved_slow)
        fast.freeze(some)
        slow.freeze(some)
        fast.retry_refused()
        slow.retry_refused()
        for reducer in (fast, slow):
            _drained(reducer.remove(int(0.05 * reducer.alive)))
        _same_mesh(self, fast.result(), slow.result())

    def test_flat_faces_read_like_a_list(self) -> None:
        faces = [(0, 1, 2), (3, 4, 5, 6), (7, 8, 9)]
        flat = T.FlatFaces.of(faces)
        self.assertEqual(list(flat), faces)
        self.assertEqual((flat[1], flat[-1], flat[1:], len(flat)), ((3, 4, 5, 6), (7, 8, 9), faces[1:], 3))
        self.assertEqual(flat, faces)
        self.assertEqual(P._half_edges(faces)[2].tolist(), P._half_edges(flat)[2].tolist())

    def test_pairing_per_chunk_welds_its_seams(self) -> None:
        mesh = torus(60, 30, seed=8)
        whole = FQ.pair_and_close(mesh)[0]
        chunked, report = FQ.pair_and_close(mesh, chunk_faces=400)
        again = FQ.pair_and_close(mesh, chunk_faces=400)[0]
        _same_mesh(self, chunked, again)  # threads don't change it
        summary = T.topology_summary(MeshData(chunked.vertices, list(chunked.faces)))
        self.assertTrue(summary.all_quads)
        self.assertEqual((summary.boundary_edges, summary.non_manifold_edges, summary.connected_components, summary.euler_characteristic), (0, 0, 1, 0))
        self.assertLess(len(chunked.faces), 1.7 * len(whole.faces))  # small regions on a small mesh: on a real scan, under 1% more
        slow = py(FQ.pair_and_close, mesh, chunk_faces=400)[0]
        _same_mesh(self, chunked, slow)

    def test_reducer_sizes_are_one_run(self) -> None:
        mesh = torus(60, 40, noise=0.2, seed=6)
        run = turbo.reducer(mesh.vertices, mesh.faces, None, 12)
        twice = turbo.reducer(mesh.vertices, mesh.faces, None, 12)
        last = len(mesh.faces) + 1
        for wanted in (3000, 1000, 200, 40):
            points, tris = run.level(wanted)
            again = twice.level(wanted)
            self.assertTrue(np.array_equal(points, again[0]) and np.array_equal(tris, again[1]))  # the same every time
            tris = tris.reshape(-1, 3)
            self.assertLessEqual(len(tris), wanted)
            self.assertGreaterEqual(len(tris), wanted - 2)
            self.assertLess(len(tris), last)
            last = len(tris)
            counts = T.topology_summary(MeshData(points.reshape(-1, 3), [tuple(t) for t in tris.tolist()]))
            self.assertEqual((counts.non_manifold_edges, counts.boundary_edges, counts.euler_characteristic), (0, 0, 0))  # still a closed torus

    def test_reducer_keeps_locked_vertices(self) -> None:
        mesh = torus(40, 24, noise=0.2, seed=7)
        locked = np.zeros(len(mesh.vertices), dtype=bool)
        locked[::9] = True
        points = turbo.reducer(mesh.vertices, mesh.faces, locked, 200).level(200)[0].reshape(-1, 3)
        found = {tuple(p) for p in points.tolist()}
        self.assertTrue(all(tuple(p) in found for p in mesh.vertices[locked].tolist()))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        unittest.main()
