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

from triadq import MeshData, topology_summary  # noqa: E402
from triadq import patch  # noqa: E402
from triadq.topology import parity_path_quad_closure, quad_only_closure, quads_locally_valid  # noqa: E402


def grid(n: int, triangles: bool = False, bump: float = 0.0):
    xs = np.linspace(-1.0, 1.0, n + 1)
    vertices = np.array([(x, y, bump * np.exp(-4.0 * (x * x + y * y))) for y in xs for x in xs])
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + n + 2, a + n + 1
            faces.extend([(a, b, c), (a, c, d)] if triangles else [(a, b, c, d)])
    return vertices, faces


def centroids(vertices, faces):
    return np.array([vertices[list(face)].mean(axis=0) for face in faces])


def box_mask(vertices, faces, hx=0.5, hy=0.4):
    c = centroids(vertices, faces)
    return (np.abs(c[:, 0]) < hx) & (np.abs(c[:, 1]) < hy)


class MaskTests(unittest.TestCase):
    def test_clean_mask_drops_islands_and_fills_holes(self):
        vertices, faces = grid(10)
        mask = box_mask(vertices, faces, 0.6, 0.6)
        hole = int(np.argmin(np.linalg.norm(centroids(vertices, faces) - [0.1, 0.1, 0.0], axis=1)))
        speck = 0  # corner face, far from the region
        mask[hole] = False
        mask[speck] = True
        cleaned = patch.clean_mask(faces, mask, min_island=2, max_hole=2)
        self.assertTrue(cleaned[hole])
        self.assertFalse(cleaned[speck])
        self.assertEqual(len(patch.mask_components(faces, cleaned)), 1)

    def test_components_split(self):
        vertices, faces = grid(10)
        c = centroids(vertices, faces)
        mask = (np.abs(c[:, 1]) < 0.3) & (np.abs(np.abs(c[:, 0]) - 0.6) < 0.25)
        components = patch.mask_components(faces, mask)
        self.assertEqual(len(components), 2)
        self.assertEqual(sum(len(x) for x in components), int(mask.sum()))

    def test_combine_masks_in_order(self):
        a = np.array([1, 1, 0, 0], bool)
        b = np.array([0, 1, 1, 0], bool)
        self.assertEqual(patch.combine_masks(4, [("UNION", a), ("UNION", b)]).tolist(), [1, 1, 1, 0])
        self.assertEqual(patch.combine_masks(4, [("UNION", a), ("SUBTRACT", b)]).tolist(), [1, 0, 0, 0])
        self.assertEqual(patch.combine_masks(4, [("UNION", a), ("INTERSECT", b)]).tolist(), [0, 1, 0, 0])

    def test_winding_number_and_box(self):
        lattice = np.stack(np.meshgrid(*[np.linspace(-1, 1, 3)] * 3, indexing="ij"), axis=-1)
        lattice[..., 0] *= 1.0 + 0.3 * lattice[..., 2]  # deform: a frustum
        hull = patch.lattice_hull(lattice)
        points = np.array([[0, 0, 0], [0.9, 0, 0], [1.2, 0, 0.9], [1.2, 0, -0.9], [5, 5, 5]], float)
        w = patch.winding_number(points, hull)
        np.testing.assert_allclose(w[[0, 1, 2, 4]], [1, 1, 1, 0], atol=1e-9)
        self.assertEqual(patch.inside_volume(points, hull).tolist(), [True, True, True, False, False])
        to_local = np.diag([2.0, 2.0, 2.0, 1.0])  # a cage of half size 0.5
        self.assertEqual(patch.inside_box(np.array([[0.4, 0, 0], [0.6, 0, 0]]), to_local).tolist(), [True, False])


class LoopTests(unittest.TestCase):
    def test_boundary_loop_follows_face_winding(self):
        vertices, faces = grid(8)
        mask = box_mask(vertices, faces)
        loops, pinched = patch.boundary_loops(faces, np.flatnonzero(mask))
        self.assertEqual(pinched, [])
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        self.assertEqual(len(loop), 2 * (4 + 4))  # a 4 x 4 block of quads
        x, y = vertices[loop, 0], vertices[loop, 1]
        area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
        self.assertGreater(area, 0.0)  # counter-clockwise like the faces

    def test_parity_fix_makes_seam_even(self):
        vertices, faces = grid(12, triangles=True)
        mask = box_mask(vertices, faces, 0.5, 0.5)
        extra = np.flatnonzero(~mask & box_mask(vertices, faces, 0.62, 0.5))[:1]
        mask[extra] = True
        loops, _ = patch.boundary_loops(faces, np.flatnonzero(mask))
        self.assertEqual(len(loops[0]) % 2, 1)
        components, warnings = patch.prepare_regions(vertices, faces, mask, min_island=1, smooth=0)
        self.assertEqual(warnings, [])
        loops, _ = patch.boundary_loops(faces, components[0])
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]) % 2, 0)


class PlanarTests(unittest.TestCase):
    def test_coons_grid_reproduces_bilinear(self):
        corners = np.array([[0, 0], [3, 0], [3, 2], [0, 2]], float)
        loop = [corners[0] + (corners[1] - corners[0]) * t for t in np.arange(3) / 3]
        loop += [corners[1] + (corners[2] - corners[1]) * t for t in np.arange(2) / 2]
        loop += [corners[2] + (corners[3] - corners[2]) * t for t in np.arange(3) / 3]
        loop += [corners[3] + (corners[0] - corners[3]) * t for t in np.arange(2) / 2]
        nodes, ids = patch.coons_grid(np.array(loop), [0, 3, 5, 8])
        self.assertEqual(nodes.shape, (4, 3, 2))
        np.testing.assert_allclose(nodes[1, 1], [1.0, 1.0])
        self.assertEqual(int((ids < 0).sum()), 2)

    def test_tutte_embedding_is_inside_and_unflipped(self):
        vertices, faces = grid(8, triangles=True, bump=0.5)
        loops, _ = patch.boundary_loops(faces, np.arange(len(faces)))
        loop = loops[0]
        uv = patch.tutte_embedding(len(vertices), np.asarray(faces), loop, patch.circle_boundary(vertices[loop]), vertices)
        self.assertLess(float(np.linalg.norm(uv, axis=1).max()), 1.0 + 1e-9)
        p = uv[np.asarray(faces)]
        area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
        self.assertTrue(bool((area > 0).all()))

    def test_frozen_midpoint_closure_keeps_seam_vertex(self):
        # Square split into two triangles; vertex 4 must become the midpoint of seam edge (0, 1).
        vertices = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, -0.1, 0]], float)
        mesh = MeshData(vertices=vertices, faces=[(0, 1, 2), (0, 2, 3)])
        for closure in (quad_only_closure, parity_path_quad_closure):
            closed = closure(mesh, frozen_midpoints={(0, 1): 4})
            used = {v for face in closed.faces for v in face}
            self.assertIn(4, used)
            self.assertTrue(all(len(face) == 4 for face in closed.faces))
            np.testing.assert_allclose(closed.vertices[4], vertices[4])

    def test_parity_path_relaxed_validation(self):
        # Valid in the plane, but the 3D copy folds every template (a staircase seam corner).
        vertices = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, -0.1, 0]], float)
        folded = vertices.copy()
        folded[[1, 3], 2] = 6.0
        mesh = MeshData(vertices=vertices, faces=[(0, 1, 2), (0, 2, 3)])
        with self.assertRaises(ValueError):
            parity_path_quad_closure(mesh, frozen_midpoints={(0, 1): 4}, validation_vertices=folded)
        closed = parity_path_quad_closure(
            mesh, frozen_midpoints={(0, 1): 4}, validation_vertices=folded, validation_required=False
        )
        self.assertTrue(all(len(face) == 4 for face in closed.faces))
        self.assertIn(4, {v for face in closed.faces for v in face})


class PatchTests(unittest.TestCase):
    def check_patch(self, vertices, faces, results):
        merged, merged_faces, new = patch.apply_patches(vertices, faces, results)
        before = topology_summary(MeshData(vertices=vertices, faces=faces))
        after = topology_summary(MeshData(vertices=merged, faces=merged_faces))
        self.assertEqual(after.boundary_edges, before.boundary_edges)  # no T-junction cracks
        self.assertEqual(after.non_manifold_edges, 0)
        self.assertEqual(after.non_manifold_vertices, 0)
        self.assertEqual(after.inconsistent_edges, 0)
        self.assertEqual(after.euler_characteristic, before.euler_characteristic)
        self.assertTrue(all(len(merged_faces[i]) == 4 for i in np.flatnonzero(new)))
        for result in results:
            used = {v for face in result.faces for v in face if v < result.base}
            self.assertEqual(used, set(result.seam))  # the seam is reused exactly
        return merged, merged_faces, new

    def test_region_remesh_flow_and_grid(self):
        for triangles in (False, True):
            vertices, faces = grid(16, triangles=triangles, bump=0.3)
            components, warnings = patch.prepare_regions(vertices, faces, box_mask(vertices, faces))
            self.assertEqual((len(components), warnings), (1, []))
            for strategy in ("FLOW", "GRID"):
                with self.subTest(triangles=triangles, strategy=strategy):
                    result = patch.remesh_region(vertices, faces, components[0], strategy=strategy, target_length=0.12)
                    self.assertTrue(result.ok, result.message)
                    self.check_patch(vertices, faces, [result])

    def test_grid_on_four_clean_sides_is_coons(self):
        vertices, faces = grid(12)
        components, _ = patch.prepare_regions(vertices, faces, box_mask(vertices, faces))
        result = patch.remesh_region(vertices, faces, components[0], strategy="GRID")
        self.assertEqual(result.closure, "Coons")
        self.assertEqual(len(result.faces), int(box_mask(vertices, faces).sum()))

    def test_fill_loop_closes_hole(self):
        vertices, faces = grid(12, bump=0.3)
        hole = box_mask(vertices, faces)
        loops, _ = patch.boundary_loops(faces, np.flatnonzero(hole))
        kept = [face for face, gone in zip(faces, hole) if not gone]
        seam = set(loops[0])
        ring = sorted({(a, b) for face in kept for k, a in enumerate(face) for b in (face[k - 1], face[(k + 1) % 4])
                       if a in seam and b not in seam})
        for strategy in ("FLOW", "GRID"):
            with self.subTest(strategy=strategy):
                result = patch.fill_loop(vertices, loops[0], ring_edges=ring, strategy=strategy)
                self.assertTrue(result.ok, result.message)
                merged, merged_faces, _new = patch.apply_patches(vertices, kept, [result])
                summary = topology_summary(MeshData(vertices=merged, faces=merged_faces))
                self.assertEqual(summary.boundary_edges, 4 * 12)  # only the outer border is open
                self.assertEqual(summary.non_manifold_edges, 0)
                self.assertTrue(all(len(face) == 4 for face in result.faces))

    def test_odd_hole_is_refused(self):
        vertices = np.array([[np.cos(t), np.sin(t), 0] for t in np.linspace(0, 2 * np.pi, 7)[:-1]] + [[0, 0, 1]])
        result = patch.fill_loop(vertices, [0, 1, 2, 3, 4])
        self.assertFalse(result.ok)

    def test_fit_to_surface_pins_and_projects(self):
        vertices, faces = grid(6)
        noisy = vertices + np.random.default_rng(1).normal(0, 0.02, vertices.shape)
        pinned = np.zeros(len(vertices), bool)
        pinned[:7] = True

        def flatten(points):
            return points * np.array([1.0, 1.0, 0.0])

        out = patch.fit_to_surface(noisy, faces, pinned, flatten, iterations=3)
        np.testing.assert_allclose(out[:7], noisy[:7])
        np.testing.assert_allclose(out[7:, 2], 0.0, atol=1e-12)

    def test_staircase_seam_gets_layer_and_valid_quads(self):
        for triangles in (True, False):
            with self.subTest(triangles=triangles):
                vertices, faces = grid(24, triangles=triangles, bump=0.4)
                c = centroids(vertices, faces)
                mask = (c[:, 0] ** 2 / 0.55**2 + c[:, 1] ** 2 / 0.4**2) < 1  # cut along faces: jagged seam
                components, _ = patch.prepare_regions(vertices, faces, mask, smooth=0)
                result = patch.remesh_region(vertices, faces, components[0])
                self.assertTrue(result.ok, result.message)
                self.assertIn("ParityPath", result.closure)
                self.assertIn("Layer", result.closure)
                self.check_patch(vertices, faces, [result])
                merged = np.vstack((vertices, result.points))
                valid = quads_locally_valid(merged, np.asarray(result.faces))
                self.assertTrue(valid.all(), f"{int((~valid).sum())} concave/warped quads")
                plain = patch.remesh_region(vertices, faces, components[0], seam_layer="OFF")
                self.assertNotIn("Layer", plain.closure)
                self.check_patch(vertices, faces, [plain])

    def test_coons_follows_target_edge_length(self):
        vertices, faces = grid(12)
        components, _ = patch.prepare_regions(vertices, faces, box_mask(vertices, faces))
        seam_spacing = 1.0 / 6.0
        same = patch.remesh_region(vertices, faces, components[0], strategy="GRID", target_length=seam_spacing)
        self.assertEqual(same.closure, "Coons")
        fine = patch.remesh_region(vertices, faces, components[0], strategy="GRID", target_length=0.5 * seam_spacing)
        self.assertTrue(fine.ok, fine.message)
        self.assertNotEqual(fine.closure, "Coons")
        self.assertGreater(len(fine.faces), 2 * len(same.faces))  # interior at the finer target
        self.check_patch(vertices, faces, [fine])

    def test_field_mode_curvature_runs(self):
        vertices, faces = grid(16, triangles=True, bump=0.3)
        components, _ = patch.prepare_regions(vertices, faces, box_mask(vertices, faces))
        result = patch.remesh_region(vertices, faces, components[0], field_mode="CURVATURE")
        self.assertTrue(result.ok, result.message)
        self.check_patch(vertices, faces, [result])


class OptimiserTests(unittest.TestCase):
    def test_untangle_repairs_concave_quads(self):
        vertices, faces = grid(4)
        uv = vertices[:, :2].copy()
        fixed = np.zeros(len(uv), bool)
        border = (np.abs(uv) > 0.999).any(axis=1)
        fixed[border] = True
        uv[6] = [0.2, 0.25]  # drag an interior vertex past its neighbours: concave quads
        def lift(p):
            return np.concatenate((p, np.zeros((len(p), 1))), axis=1)
        quads = np.asarray(faces)
        self.assertTrue((patch.quad_quality(lift(uv)[quads]) <= 0).any())
        out = patch._untangle(uv, faces, fixed, lift, planar=True)
        np.testing.assert_allclose(out[border], uv[border])
        self.assertTrue((patch.quad_quality(lift(out)[quads]) > 0).all())

    def test_quad_quality_matches_validity(self):
        square = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]], float)
        dart = np.array([[[0, 0, 0], [1, 0, 0], [0.2, 0.2, 0], [0, 1, 0]]], float)
        folded = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 2], [0, 1, 0]]], float)
        self.assertAlmostEqual(float(patch.quad_quality(square)[0]), 1.0)
        for quad in (dart, folded):
            self.assertLessEqual(float(patch.quad_quality(quad)[0]), 0.0)
            self.assertFalse(quads_locally_valid(quad[0], np.array([[0, 1, 2, 3]]))[0])

    def test_seam_layer_topology(self):
        count = 12
        t = np.linspace(0, 2 * np.pi, count, endpoint=False)
        loop = np.stack((np.cos(t), np.sin(t)), axis=1)
        inner, quads, m = patch.build_seam_layer(loop)
        self.assertEqual((m, len(quads)), (count, count))
        angles = np.full(count, 180.0)
        angles[[2, 3, 7]] = (80.0, 70.0, 90.0)  # 2 and 3 adjacent: only the sharper becomes an end
        inner, quads, m = patch.build_seam_layer(loop, angles)
        self.assertEqual(m, count - 4)
        self.assertEqual(len(inner), m)
        edges = {}
        for quad in quads:
            for k in range(4):
                key = tuple(sorted((quad[k], quad[(k + 1) % 4])))
                edges[key] = edges.get(key, 0) + 1
        for k in range(count):  # every seam edge in exactly one layer quad
            self.assertEqual(edges[tuple(sorted((k, (k + 1) % count)))], 1)
        inner_edges = [e for e, n in edges.items() if n == 1 and min(e) >= count]
        self.assertEqual(len(inner_edges), m)  # the open inner side is the inner loop

    def test_fit_to_surface_keeps_quads_valid(self):
        vertices, faces = grid(8, bump=0.4)
        pinned = (np.abs(vertices[:, :2]) > 0.999).any(axis=1)
        before = quads_locally_valid(vertices, np.asarray(faces))
        out = patch.fit_to_surface(vertices, faces, pinned, None, iterations=5, strength=0.9)
        after = quads_locally_valid(out, np.asarray(faces))
        self.assertGreaterEqual(int(after.sum()), int(before.sum()))


if __name__ == "__main__":
    unittest.main()
