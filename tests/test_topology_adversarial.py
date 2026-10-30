# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

from __future__ import annotations

import unittest

import numpy as np

from triadq import (
    MeshData,
    component_topology_preserved,
    parity_path_quad_closure,
    split_non_manifold_vertex_fans,
    topology_summary,
)


def _mesh(vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]) -> MeshData:
    return MeshData(
        vertices=np.asarray(vertices, dtype=float),
        faces=faces,
        face_materials=["test"] * len(faces),
        name="adversarial",
    )


class AdversarialTopologyTests(unittest.TestCase):
    def test_parity_path_rejects_invalid_source_without_mutating_it(self) -> None:
        mesh = _mesh(
            [(0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)],
            [(0, 1, 2, 3)],
        )
        original_vertices = mesh.vertices.copy()
        original_faces = list(mesh.faces)

        with self.assertRaisesRegex(ValueError, "invalid_quads=1"):
            parity_path_quad_closure(mesh)

        np.testing.assert_array_equal(mesh.vertices, original_vertices)
        self.assertEqual(mesh.faces, original_faces)

    def test_parity_path_preserves_each_disconnected_component_signature(self) -> None:
        mesh = _mesh(
            [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (3, 0, 0),
                (5, 0, 0),
                (3, 2, 0),
            ],
            [(0, 1, 2, 3), (4, 5, 6)],
        )

        output = parity_path_quad_closure(mesh)

        self.assertTrue(topology_summary(output).all_quads)
        self.assertEqual(
            topology_summary(output).component_signature,
            topology_summary(mesh).component_signature,
        )

    def test_vertex_only_bow_tie_is_not_manifold(self) -> None:
        mesh = _mesh(
            [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (-1, 0, 0),
                (-1, -1, 0),
                (0, -1, 0),
            ],
            [(0, 1, 2, 3), (0, 4, 5, 6)],
        )
        summary = topology_summary(mesh)
        self.assertEqual(summary.non_manifold_edges, 0)
        self.assertEqual(summary.non_manifold_vertices, 1)
        self.assertEqual(summary.connected_components, 2)
        self.assertEqual(summary.component_euler, (1, 1))
        self.assertFalse(summary.all_quads)

    def test_vertex_only_bow_tie_split_preserves_geometry_and_face_data(self) -> None:
        mesh = MeshData(
            vertices=np.asarray(
                [
                    (0, 0, 0),
                    (1, 0, 0),
                    (1, 1, 0),
                    (0, 1, 0),
                    (-1, 0, 0),
                    (-1, -1, 0),
                    (0, -1, 0),
                ],
                dtype=float,
            ),
            faces=[(0, 1, 2, 3), (0, 4, 5, 6)],
            face_materials=["first", "second"],
            uvs=np.asarray(
                [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (0, 1)],
                dtype=float,
            ),
            face_uvs=[(0, 1, 2, 3), (4, 5, 6, 7)],
            name="bow_tie",
        )

        repaired, split_count = split_non_manifold_vertex_fans(mesh)
        repaired_summary = topology_summary(repaired)

        self.assertEqual(split_count, 1)
        self.assertEqual(repaired.vertex_count, mesh.vertex_count + 1)
        np.testing.assert_array_equal(repaired.vertices[:-1], mesh.vertices)
        np.testing.assert_array_equal(repaired.vertices[-1], mesh.vertices[0])
        self.assertEqual(repaired.faces[0], mesh.faces[0])
        self.assertEqual(repaired.faces[1][0], mesh.vertex_count)
        self.assertEqual(repaired.face_materials, mesh.face_materials)
        self.assertEqual(repaired.face_uvs, mesh.face_uvs)
        np.testing.assert_array_equal(repaired.uvs, mesh.uvs)
        self.assertEqual(repaired_summary.non_manifold_vertices, 0)
        self.assertTrue(repaired_summary.all_quads)
        self.assertEqual(topology_summary(mesh).non_manifold_vertices, 1)

    def test_vertex_fan_split_handles_a_globally_connected_bow_tie(self) -> None:
        mesh = _mesh(
            [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, -1, 0),
                (-1, -1, 0),
                (-1, 0, 0),
            ],
            [(0, 1, 2, 3), (0, 6, 5, 4), (2, 1, 4, 5)],
        )
        before = topology_summary(mesh)
        repaired, split_count = split_non_manifold_vertex_fans(mesh)
        after = topology_summary(repaired)

        self.assertEqual(before.connected_components, 1)
        self.assertEqual(before.non_manifold_vertices, 1)
        self.assertEqual(split_count, 1)
        self.assertEqual(after.connected_components, 1)
        self.assertEqual(after.non_manifold_vertices, 0)

    def test_vertex_fan_split_is_deterministic_for_three_fans(self) -> None:
        mesh = _mesh(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (-1, 0, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 1, 1),
            ],
            [(0, 1, 2), (0, 3, 4), (0, 5, 6)],
        )
        repaired, split_count = split_non_manifold_vertex_fans(mesh)

        self.assertEqual(split_count, 2)
        self.assertEqual(repaired.faces[0][0], 0)
        self.assertEqual(repaired.faces[1][0], mesh.vertex_count)
        self.assertEqual(repaired.faces[2][0], mesh.vertex_count + 1)
        self.assertEqual(topology_summary(repaired).non_manifold_vertices, 0)

    def test_vertex_fan_split_refuses_edge_non_manifold_input(self) -> None:
        mesh = _mesh(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)],
            [(0, 1, 2), (1, 0, 3), (0, 1, 4)],
        )
        repaired, split_count = split_non_manifold_vertex_fans(mesh)

        self.assertIs(repaired, mesh)
        self.assertEqual(split_count, 0)
        self.assertEqual(topology_summary(repaired).non_manifold_edges, 1)

    def test_duplicate_face_is_rejected_independent_of_winding(self) -> None:
        mesh = _mesh(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            [(0, 1, 2, 3), (3, 2, 1, 0)],
        )
        summary = topology_summary(mesh)
        self.assertEqual(summary.duplicate_faces, 1)
        self.assertFalse(summary.all_quads)

    def test_inconsistent_shared_edge_winding_is_rejected(self) -> None:
        mesh = _mesh(
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)],
            [(0, 1, 4, 3), (1, 4, 5, 2)],
        )
        summary = topology_summary(mesh)
        self.assertEqual(summary.inconsistent_edges, 1)
        self.assertFalse(summary.all_quads)

    def test_concave_and_crossed_quads_are_rejected(self) -> None:
        concave = _mesh(
            [(0, 0, 0), (1, 0, 0), (0.25, 0.25, 0), (0, 1, 0)],
            [(0, 1, 2, 3)],
        )
        crossed = _mesh(
            [(0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)],
            [(0, 1, 2, 3)],
        )
        self.assertEqual(topology_summary(concave).invalid_quads, 1)
        self.assertEqual(topology_summary(crossed).invalid_quads, 1)
        self.assertFalse(topology_summary(concave).all_quads)
        self.assertFalse(topology_summary(crossed).all_quads)

    def test_extreme_saddle_warpage_is_rejected(self) -> None:
        warped = _mesh(
            [(-1, -1, 100), (1, -1, -100), (1, 1, 100), (-1, 1, -100)],
            [(0, 1, 2, 3)],
        )
        summary = topology_summary(warped)
        self.assertEqual(summary.invalid_quads, 1)
        self.assertFalse(summary.all_quads)

    def test_disconnected_components_are_counted(self) -> None:
        mesh = _mesh(
            [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (3, 0, 0),
                (4, 0, 0),
                (4, 1, 0),
                (3, 1, 0),
            ],
            [(0, 1, 2, 3), (4, 5, 6, 7)],
        )
        summary = topology_summary(mesh)
        self.assertEqual(summary.connected_components, 2)
        self.assertEqual(summary.boundary_components, 2)
        self.assertEqual(summary.component_signature, ((1, 1), (1, 1)))
        self.assertTrue(summary.all_quads)

    def test_boundary_cycles_cannot_redistribute_between_equal_euler_components(self) -> None:
        source_signature = ((-2, 0), (-2, 4))
        redistributed_signature = ((-2, 2), (-2, 2))

        # These are exactly the legacy preservation invariants: they cannot
        # distinguish a closed genus-two component plus a four-boundary sphere
        # from two genus-one components with two boundaries each.
        self.assertEqual(
            tuple(component[0] for component in source_signature),
            tuple(component[0] for component in redistributed_signature),
        )
        self.assertEqual(
            sum(component[1] for component in source_signature),
            sum(component[1] for component in redistributed_signature),
        )

        self.assertFalse(component_topology_preserved(source_signature, redistributed_signature))


if __name__ == "__main__":
    unittest.main()
