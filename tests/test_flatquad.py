# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triadq import MeshData, flat_quad_remesh, topology_summary  # noqa: E402
from triadq.flatquad import FlatQuadOptions, flatten_regions, make_closable  # noqa: E402


def plate(n: int, size: float = 2.0, height=None, offset=(0.0, 0.0, 0.0), skip=None):
    """Triangulated n x n plate; `height(x, y)` lifts it, `skip(x, y)` cuts cells out."""

    xs = np.linspace(-size / 2.0, size / 2.0, n + 1)
    vertices = np.array([(x, y, height(x, y) if height else 0.0) for y in xs for x in xs]) + offset
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            if skip and skip((xs[i] + xs[i + 1]) / 2.0, (xs[j] + xs[j + 1]) / 2.0):
                continue
            faces += [(a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)]
    return MeshData(vertices=vertices, faces=faces)


def cube(n: int) -> MeshData:
    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}
    faces = []
    xs = np.linspace(-1.0, 1.0, n + 1)

    def vertex(p):
        key = tuple(round(float(c), 9) for c in p)
        if key not in index:
            index[key] = len(vertices)
            vertices.append(key)
        return index[key]

    for axis in range(3):
        for sign in (-1.0, 1.0):
            for j in range(n):
                for i in range(n):
                    corners = []
                    for u, v in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = [0.0, 0.0, 0.0]
                        p[axis] = sign
                        p[(axis + 1) % 3] = xs[u]
                        p[(axis + 2) % 3] = xs[v]
                        corners.append(vertex(p))
                    if sign < 0.0:
                        corners.reverse()
                    faces.append(tuple(corners))
    return MeshData(vertices=np.array(vertices), faces=faces)


def boundary_loop_count(mesh: MeshData) -> int:
    halves = {(f[k], f[(k + 1) % len(f)]) for f in mesh.faces for k in range(len(f))}
    following = {a: b for a, b in halves if (b, a) not in halves}
    loops = 0
    while following:
        start = next(iter(following))
        vertex = start
        while vertex in following:
            vertex = following.pop(vertex)
        loops += 1
    return loops


class FlatRegionTests(unittest.TestCase):
    def test_flat_plate_becomes_one_quad(self):
        result, report, _solver = flat_quad_remesh(plate(40))
        self.assertEqual(len(result.faces), 1)
        self.assertTrue(report.all_quads)
        spread = np.abs(result.vertices[:, :2])
        self.assertTrue(np.allclose(spread, 1.0))  # the four true corners survive

    def test_long_rectangle_stays_one_quad(self):
        mesh = plate(40)
        mesh.vertices[:, 0] *= 50.0
        result, _report, _solver = flat_quad_remesh(mesh)
        self.assertEqual(len(result.faces), 1)

    def test_noisy_plate_merges_within_tolerance(self):
        rng = np.random.default_rng(3)
        mesh = plate(40)
        mesh.vertices[:, 2] += rng.normal(0.0, 2e-4, len(mesh.vertices))
        coarse, report = flatten_regions(mesh, FlatQuadOptions(tolerance=0.001))
        self.assertEqual(report.regions, 1)
        self.assertEqual(len(coarse.faces), 2)

    def test_tolerance_is_respected(self):
        mesh = plate(40, height=lambda x, y: 0.05 * x * x)
        coarse, report = flatten_regions(mesh, FlatQuadOptions(tolerance=0.001))
        self.assertGreater(report.regions, 1)
        limit = report.tolerance * 2.0 + 1e-9
        self.assertLess(np.abs(coarse.vertices[:, 2] - 0.05 * coarse.vertices[:, 0] ** 2).max(), limit)

    def test_hole_is_kept(self):
        mesh = plate(40, skip=lambda x, y: abs(x) < 0.3 and abs(y) < 0.3)
        result, report, _solver = flat_quad_remesh(mesh)
        self.assertTrue(report.all_quads)
        self.assertEqual(boundary_loop_count(result), 2)
        self.assertLess(len(result.faces), 16)
        kept = result.vertices
        self.assertTrue(((np.abs(kept[:, 0]) - 0.3) ** 2 + (np.abs(kept[:, 1]) - 0.3) ** 2 < 1e-3).any())

    def test_closed_box_becomes_six_quads(self):
        result, report, _solver = flat_quad_remesh(cube(12))
        self.assertTrue(report.all_quads)
        self.assertEqual(len(result.faces), 6)
        summary = topology_summary(result)
        self.assertEqual(summary.boundary_edges, 0)
        self.assertEqual(summary.non_manifold_edges, 0)

    def test_two_nearby_plates_stay_separate(self):
        first = plate(20)
        second = plate(20, offset=(0.0, 0.0, 0.002))
        base = len(first.vertices)
        mesh = MeshData(
            vertices=np.vstack((first.vertices, second.vertices)),
            faces=first.faces + [tuple(v + base for v in f) for f in second.faces],
        )
        result, _report, _solver = flat_quad_remesh(mesh)
        self.assertEqual(len(result.faces), 2)
        self.assertEqual(sorted(set(np.round(result.vertices[:, 2], 6).tolist())), [0.0, 0.002])

    def test_detail_stays_and_density_grades_away_from_it(self):
        bump = lambda x, y: 0.2 * np.exp(-((x * x + y * y) / 0.02)) if x * x + y * y < 0.09 else 0.0  # noqa: E731
        mesh = plate(60, height=bump)
        coarse, report = flatten_regions(mesh)
        self.assertGreaterEqual(report.regions, 1)
        self.assertLess(report.output_faces, len(mesh.faces) * 0.6)
        self.assertEqual(topology_summary(coarse).non_manifold_edges, 0)
        self.assertGreater(coarse.vertices[:, 2].max(), 0.19)  # the peak is untouched
        result, final, _solver = flat_quad_remesh(mesh)
        self.assertTrue(final.all_quads)
        self.assertEqual(boundary_loop_count(result), 1)

    def test_shared_outline_stays_conforming(self):
        # An L of two plates meeting along a dense crease: both must agree on the simplified chain.
        n = 16
        xs = np.linspace(0.0, 1.0, n + 1)
        vertices = [(x, y, 0.0) for y in xs for x in xs] + [(x, 0.0, -z) for z in xs[1:] for x in xs]
        faces = []
        for j in range(n):
            for i in range(n):
                a = j * (n + 1) + i
                faces += [(a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)]
        base = (n + 1) * (n + 1)
        for j in range(n):
            for i in range(n):
                top = i if j == 0 else base + (j - 1) * (n + 1) + i
                low = base + j * (n + 1) + i
                faces += [(top, low, low + 1), (top, low + 1, top + 1)]
        result, report, _solver = flat_quad_remesh(MeshData(vertices=np.array(vertices), faces=faces))
        self.assertEqual(report.regions, 2)
        self.assertEqual(len(result.faces), 2)
        self.assertEqual(topology_summary(result).non_manifold_edges, 0)

    def test_non_manifold_mesh_still_closes_to_all_quads(self):
        mesh = plate(10)
        extra = len(mesh.vertices)
        mesh = MeshData(
            vertices=np.vstack((mesh.vertices, [[0.0, 0.0, 1.0]])),
            faces=mesh.faces + [(0, 1, extra)],
        )
        result, report, _solver = flat_quad_remesh(mesh)
        self.assertTrue(report.all_quads)
        self.assertLess(len(result.faces), len(mesh.faces))
        self.assertEqual(topology_summary(result).non_manifold_edges, 0)

    def test_make_closable_opens_flipped_and_duplicate_faces_without_moving_anything(self):
        mesh = plate(4)
        flipped = [tuple(reversed(mesh.faces[0]))] + mesh.faces[1:] + [mesh.faces[1]]
        fixed, added = make_closable(MeshData(vertices=mesh.vertices, faces=flipped))
        summary = topology_summary(fixed)
        self.assertEqual(len(fixed.faces), len(mesh.faces))  # the duplicate is gone
        self.assertGreater(added, 0)
        self.assertEqual(summary.inconsistent_edges, 0)
        self.assertEqual(summary.duplicate_faces, 0)
        self.assertTrue(np.array_equal(fixed.vertices[: len(mesh.vertices)], mesh.vertices))

    def test_a_loose_tetrahedron_does_not_split_the_big_quads(self):
        # A tetrahedron pairs into two quads on the same four vertices, which sparse closure refuses.
        big = plate(20)
        base = len(big.vertices)
        tetra = np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 0.0, 1.0]])
        faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
        mesh = MeshData(
            vertices=np.vstack((big.vertices, tetra)),
            faces=big.faces + [tuple(v + base for v in f) for f in faces],
        )
        result, report, _solver = flat_quad_remesh(mesh)
        self.assertTrue(report.all_quads)
        self.assertLess(len(result.faces), 30)  # 1 plate quad plus the fragment's own

    def test_a_region_touching_itself_at_a_vertex_keeps_its_faces(self):
        # Two cells cut out so that they meet at one corner: the plate's outline is pinched there.
        mesh = plate(20, skip=lambda x, y: (0.0 < x < 0.1 and 0.0 < y < 0.1) or (-0.1 < x < 0.0 and -0.1 < y < 0.0))
        coarse, report = flatten_regions(mesh)
        self.assertGreaterEqual(report.regions, 1)
        self.assertLess(len(coarse.faces), len(mesh.faces) * 0.5)


if __name__ == "__main__":
    unittest.main()
