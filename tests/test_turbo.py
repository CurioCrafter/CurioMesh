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
import triadq.patch as P  # noqa: E402
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


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        unittest.main()
