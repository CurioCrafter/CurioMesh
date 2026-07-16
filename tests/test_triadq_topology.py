from __future__ import annotations

import gc
import json
from pathlib import Path
import sys
import unittest
import weakref
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triadq import MeshData, RemeshOptions, quad_only_closure, remesh_mesh, topology_summary  # noqa: E402
import triadq.engine as triad_engine  # noqa: E402


class QuadClosureTests(unittest.TestCase):
    def test_flow_modes_change_field_and_pairing_parameters(self) -> None:
        base = RemeshOptions(feature_angle_deg=35.0, smooth_field_iters=4)
        organic = triad_engine._options_for_mode(base, "OrganicFlow")
        patch = triad_engine._options_for_mode(base, "PatchFlow")
        dirty = triad_engine._options_for_mode(base, "DirtyFlow")
        self.assertGreater(organic.smooth_field_iters, patch.smooth_field_iters)
        self.assertGreater(organic.feature_angle_deg, patch.feature_angle_deg)
        self.assertEqual(dirty.smooth_field_iters, 1)
        self.assertNotEqual(
            triad_engine._pair_weights("OrganicFlow"),
            triad_engine._pair_weights("TextureFlow"),
        )

    def test_open_square_is_conforming_and_preserves_euler(self) -> None:
        source = MeshData(
            vertices=np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=float),
            faces=[(0, 1, 2), (0, 2, 3)],
            face_materials=["a", "b"],
            name="square",
        )
        output = quad_only_closure(source)
        source_summary = topology_summary(source)
        output_summary = topology_summary(output)

        self.assertTrue(output_summary.all_quads)
        self.assertEqual(output_summary.faces, 6)
        self.assertEqual(output_summary.euler_characteristic, source_summary.euler_characteristic)
        self.assertEqual(output_summary.boundary_components, 1)
        self.assertEqual(output_summary.non_manifold_edges, 0)
        self.assertEqual(output.face_materials.count("a"), 3)
        self.assertEqual(output.face_materials.count("b"), 3)

        diagonal_midpoints = [
            index
            for index, vertex in enumerate(output.vertices)
            if np.allclose(vertex, np.asarray((0.5, 0.5, 0.0)))
        ]
        self.assertEqual(len(diagonal_midpoints), 1)
        shared = diagonal_midpoints[0]
        self.assertEqual(sum(shared in face for face in output.faces), 4)

    def test_closed_tetrahedron_obeys_quad_defect_charge(self) -> None:
        source = MeshData(
            vertices=np.asarray(
                [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)],
                dtype=float,
            ),
            faces=[(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
            face_materials=["stone"] * 4,
            name="tetra",
        )
        output = quad_only_closure(source)
        summary = topology_summary(output)
        self.assertTrue(summary.all_quads)
        self.assertEqual(summary.boundary_edges, 0)
        self.assertEqual(summary.euler_characteristic, 2)
        self.assertEqual(summary.defect_charge, 8)

    def test_uv_seams_remain_face_local(self) -> None:
        source = MeshData(
            vertices=np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=float),
            faces=[(0, 1, 2), (0, 2, 3)],
            face_materials=["a", "b"],
            uvs=np.asarray([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)], dtype=float),
            face_uvs=[(0, 1, 2), (3, 4, 5)],
            name="uv_square",
        )
        output = quad_only_closure(source)
        self.assertIsNotNone(output.uvs)
        self.assertIsNotNone(output.face_uvs)
        self.assertEqual(len(output.face_uvs or []), len(output.faces))
        self.assertTrue(all(len(face) == 4 for face in output.face_uvs or []))

    def test_engine_force_quads_uses_global_closure(self) -> None:
        source = MeshData(
            vertices=np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=float),
            faces=[(0, 1, 2), (0, 2, 3)],
            face_materials=["scan", "scan"],
            name="paired_square",
        )
        output, report = remesh_mesh(source, RemeshOptions(target_faces=4, seed_count=2, force_quads=True))
        self.assertTrue(report.success)
        self.assertTrue(report.closure_applied)
        self.assertTrue(report.quad_only_verified)
        self.assertTrue(topology_summary(output).all_quads)
        self.assertEqual(report.euler_characteristic, 1)

    def test_pairing_skips_concave_quad_and_closes_triangles_safely(self) -> None:
        source = MeshData(
            vertices=np.asarray(
                [(0, 0, 0), (1, 0, 0), (0.25, 0.25, 0), (0, 1, 0)],
                dtype=float,
            ),
            faces=[(0, 1, 2), (0, 2, 3)],
            face_materials=["scan", "scan"],
            name="concave_pair",
        )
        output, report = remesh_mesh(
            source,
            RemeshOptions(target_faces=6, seed_count=3, force_quads=True),
        )
        self.assertTrue(report.success)
        self.assertTrue(report.quad_only_verified)
        self.assertEqual(len(output.faces), 6)
        self.assertTrue(topology_summary(output).all_quads)

    def test_engine_force_quads_preserves_paired_and_seamed_uv_material_faces(self) -> None:
        source = MeshData(
            vertices=np.asarray(
                [
                    (0, 0, 0),
                    (1, 0, 0),
                    (2, 0, 0),
                    (0, 1, 0),
                    (1, 1, 0),
                    (2, 1, 0),
                ],
                dtype=float,
            ),
            faces=[
                (0, 1, 4),
                (0, 4, 3),
                (1, 2, 5),
                (1, 5, 4),
            ],
            face_materials=["paired", "paired", "seam_a", "seam_b"],
            uvs=np.asarray(
                [
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (0.25, 0.25),
                    (0.75, 0.75),
                    (0.25, 0.75),
                ],
                dtype=float,
            ),
            face_uvs=[
                (0, 1, 2),
                (0, 2, 3),
                (4, 5, 6),
                (7, 8, 9),
            ],
            name="paired_and_seamed_strip",
        )

        output, report = remesh_mesh(
            source,
            RemeshOptions(target_faces=10, seed_count=3, force_quads=True),
        )

        self.assertTrue(report.success)
        self.assertTrue(report.quad_only_verified)
        self.assertEqual(output.face_count, 10)
        self.assertEqual(output.face_materials.count("paired"), 4)
        self.assertEqual(output.face_materials.count("seam_a"), 3)
        self.assertEqual(output.face_materials.count("seam_b"), 3)
        self.assertIsNotNone(output.uvs)
        self.assertIsNotNone(output.face_uvs)
        self.assertEqual(len(output.face_uvs or []), output.face_count)
        self.assertTrue(all(len(face_uvs) == 4 for face_uvs in output.face_uvs or []))
        np.testing.assert_allclose(output.uvs[: len(source.uvs)], source.uvs)

        first_corner_uvs: dict[str, set[int]] = {"paired": set(), "seam_a": set(), "seam_b": set()}
        for material, face_uvs in zip(output.face_materials, output.face_uvs or [], strict=True):
            first_corner_uvs[material].add(face_uvs[0])
        self.assertEqual(first_corner_uvs["paired"], {0, 1, 2, 3})
        self.assertEqual(first_corner_uvs["seam_a"], {4, 5, 6})
        self.assertEqual(first_corner_uvs["seam_b"], {7, 8, 9})

    def test_seed_sweep_releases_non_best_candidate_meshes_and_keeps_reports(self) -> None:
        source = MeshData(
            vertices=np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=float),
            faces=[(0, 1, 2)],
            face_materials=["scan"],
            name="candidate_memory",
        )
        vertex_refs: list[weakref.ReferenceType[np.ndarray]] = []
        live_before_each_candidate: list[int] = []

        def fake_candidate(
            _mesh: object,
            _options: object,
            mode: str,
            seed: int,
        ) -> triad_engine._Candidate:
            gc.collect()
            live_before_each_candidate.append(sum(reference() is not None for reference in vertex_refs))
            vertices = np.asarray(
                [(seed, 0, 0), (seed + 1, 0, 0), (seed, 1, 0)],
                dtype=float,
            )
            vertex_refs.append(weakref.ref(vertices))
            mesh = MeshData(vertices=vertices, faces=[(0, 1, 2)], face_materials=["scan"])
            report = triad_engine.RemeshReport(
                success=True,
                mode=mode,
                selected_seed=seed,
                output_faces=1,
                score=-float(seed),
            )
            return triad_engine._Candidate(mesh, report)

        with patch.object(triad_engine, "_run_candidate", side_effect=fake_candidate):
            output, report = remesh_mesh(source, RemeshOptions(seed_count=6))

        gc.collect()
        self.assertEqual(output.vertices[0, 0], 0.0)
        self.assertLessEqual(max(live_before_each_candidate), 2)
        self.assertEqual(sum(reference() is not None for reference in vertex_refs), 1)
        self.assertEqual([candidate["selected_seed"] for candidate in report.candidates], list(range(6)))
        json.dumps(report.to_dict(), allow_nan=False)

    def test_seed_sweep_prefers_success_over_higher_scoring_failure(self) -> None:
        source = MeshData(
            vertices=np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=float),
            faces=[(0, 1, 2)],
            face_materials=["scan"],
            name="candidate_success",
        )

        def fake_candidate(
            _mesh: object,
            _options: object,
            mode: str,
            seed: int,
        ) -> triad_engine._Candidate:
            mesh = MeshData(
                vertices=np.asarray(
                    [(seed, 0, 0), (seed + 1, 0, 0), (seed, 1, 0)],
                    dtype=float,
                ),
                faces=[(0, 1, 2)],
                face_materials=["scan"],
            )
            report = triad_engine.RemeshReport(
                success=seed != 0,
                mode=mode,
                selected_seed=seed,
                output_faces=1,
                score=100.0 if seed == 0 else 10.0 - seed,
                message="closure failed" if seed == 0 else "",
            )
            return triad_engine._Candidate(mesh, report)

        with patch.object(triad_engine, "_run_candidate", side_effect=fake_candidate):
            output, report = remesh_mesh(source, RemeshOptions(seed_count=3))

        self.assertTrue(report.success)
        self.assertEqual(report.selected_seed, 1)
        self.assertEqual(output.vertices[0, 0], 1.0)
        self.assertEqual([candidate["success"] for candidate in report.candidates], [False, True, True])

    def test_non_manifold_closure_fails_without_throwing(self) -> None:
        source = MeshData(
            vertices=np.asarray(
                [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)],
                dtype=float,
            ),
            faces=[(0, 1, 2), (1, 0, 3), (0, 1, 4)],
            face_materials=["a", "b", "c"],
            name="non_manifold",
        )
        _output, report = remesh_mesh(
            source,
            RemeshOptions(target_faces=12, seed_count=2, force_quads=True),
        )
        self.assertFalse(report.success)
        self.assertFalse(report.quad_only_verified)
        self.assertIn("rejected", report.message.lower())


if __name__ == "__main__":
    unittest.main()
