from __future__ import annotations

import unittest

import numpy as np

from triadq import MeshData, component_topology_preserved, topology_summary


def _mesh(vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]) -> MeshData:
    return MeshData(
        vertices=np.asarray(vertices, dtype=float),
        faces=faces,
        face_materials=["test"] * len(faces),
        name="adversarial",
    )


class AdversarialTopologyTests(unittest.TestCase):
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
