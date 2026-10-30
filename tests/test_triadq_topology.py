# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

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

from triadq import (  # noqa: E402
    MeshData,
    RemeshOptions,
    parity_path_quad_closure,
    quad_only_closure,
    remesh_mesh,
    topology_summary,
)
import triadq.engine as triad_engine  # noqa: E402


class QuadClosureTests(unittest.TestCase):
    @staticmethod
    def _transition_patch(selected_sides: tuple[int, ...], *, with_uvs: bool = False) -> MeshData:
        """Build a square quad with outward triangle residuals on chosen sides."""

        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, -1.0, 0.0),
            (2.0, 0.5, 0.0),
            (0.5, 2.0, 0.0),
            (-1.0, 0.5, 0.0),
        ]
        side_faces = {
            0: (1, 0, 4),
            1: (2, 1, 5),
            2: (3, 2, 6),
            3: (0, 3, 7),
        }
        faces = [(0, 1, 2, 3), *(side_faces[side] for side in selected_sides)]
        materials = ["center", *(f"flap_{side}" for side in selected_sides)]
        if not with_uvs:
            return MeshData(
                vertices=np.asarray(vertices, dtype=float),
                faces=faces,
                face_materials=materials,
                name="transition_patch",
            )

        # Every face owns its UV corners. In particular, the bottom triangle
        # deliberately disagrees with the center along their shared edge.
        uv_values: list[tuple[float, float]] = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        face_uvs: list[tuple[int, ...]] = [(0, 1, 2, 3)]
        for side in selected_sides:
            start = len(uv_values)
            base = 10.0 * (side + 1)
            uv_values.extend(
                [
                    (base + 2.0, base),
                    (base, base),
                    (base + 1.0, base - 2.0),
                ]
            )
            face_uvs.append((start, start + 1, start + 2))
        return MeshData(
            vertices=np.asarray(vertices, dtype=float),
            faces=faces,
            face_materials=materials,
            uvs=np.asarray(uv_values, dtype=float),
            face_uvs=face_uvs,
            name="transition_patch_uv",
        )

    def test_parity_path_leaves_an_unaffected_quad_exactly_untouched(self) -> None:
        source = MeshData(
            vertices=np.asarray(
                [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                dtype=float,
            ),
            faces=[(0, 1, 2, 3)],
            face_materials=["untouched"],
            uvs=np.asarray([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float),
            face_uvs=[(0, 1, 2, 3)],
            name="untouched_quad",
        )

        output = parity_path_quad_closure(source)

        np.testing.assert_array_equal(output.vertices, source.vertices)
        np.testing.assert_array_equal(output.uvs, source.uvs)
        self.assertEqual(output.faces, source.faces)
        self.assertEqual(output.face_materials, source.face_materials)
        self.assertEqual(output.face_uvs, source.face_uvs)
        self.assertEqual(
            topology_summary(output).component_signature,
            topology_summary(source).component_signature,
        )

    def test_parity_path_closes_a_triangle_residual_deterministically(self) -> None:
        source = MeshData(
            vertices=np.asarray([(0, 0, 0), (2, 0, 0), (0, 2, 0)], dtype=float),
            faces=[(0, 1, 2)],
            face_materials=["residual"],
            uvs=np.asarray([(0, 0), (1, 0), (0, 1)], dtype=float),
            face_uvs=[(0, 1, 2)],
            name="triangle_residual",
        )

        first = parity_path_quad_closure(source)
        second = parity_path_quad_closure(source)
        summary = topology_summary(first)

        self.assertTrue(summary.all_quads)
        self.assertEqual(first.face_count, 3)
        self.assertEqual(first.vertex_count, 7)
        self.assertEqual(first.face_materials, ["residual"] * 3)
        self.assertEqual(summary.component_signature, topology_summary(source).component_signature)
        self.assertIsNotNone(first.face_uvs)
        self.assertTrue(all(len(face_uv) == 4 for face_uv in first.face_uvs or []))
        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.uvs, second.uvs)
        self.assertEqual(first.faces, second.faces)
        self.assertEqual(first.face_uvs, second.face_uvs)

    def test_parity_path_uses_three_quad_template_for_adjacent_splits(self) -> None:
        source = self._transition_patch((0, 1))
        output = parity_path_quad_closure(source)
        summary = topology_summary(output)

        self.assertTrue(summary.all_quads)
        self.assertEqual(output.face_count, 9)
        self.assertEqual(output.face_materials.count("center"), 3)
        self.assertEqual(output.face_materials.count("flap_0"), 3)
        self.assertEqual(output.face_materials.count("flap_1"), 3)
        self.assertEqual(summary.component_signature, topology_summary(source).component_signature)

    def test_parity_path_uses_two_quad_template_for_opposite_splits(self) -> None:
        source = self._transition_patch((0, 2))
        output = parity_path_quad_closure(source)
        summary = topology_summary(output)

        self.assertTrue(summary.all_quads)
        self.assertEqual(output.face_count, 8)
        self.assertEqual(output.face_materials.count("center"), 2)
        self.assertEqual(output.face_materials.count("flap_0"), 3)
        self.assertEqual(output.face_materials.count("flap_2"), 3)
        self.assertEqual(summary.component_signature, topology_summary(source).component_signature)

    def test_parity_path_k4_template_preserves_uv_seams_materials_and_topology(self) -> None:
        source = self._transition_patch((0, 1, 2, 3), with_uvs=True)
        first = parity_path_quad_closure(source)
        second = parity_path_quad_closure(source)
        summary = topology_summary(first)

        self.assertTrue(summary.all_quads)
        self.assertEqual(first.face_count, 16)
        self.assertEqual(first.face_materials.count("center"), 4)
        for side in range(4):
            self.assertEqual(first.face_materials.count(f"flap_{side}"), 3)
        self.assertEqual(summary.component_signature, topology_summary(source).component_signature)
        self.assertIsNotNone(first.uvs)
        self.assertIsNotNone(first.face_uvs)
        np.testing.assert_array_equal(first.uvs[: len(source.uvs)], source.uvs)
        self.assertTrue(all(len(face_uv) == 4 for face_uv in first.face_uvs or []))

        midpoint = next(
            index
            for index, vertex in enumerate(first.vertices)
            if np.allclose(vertex, np.asarray((0.5, 0.0, 0.0)))
        )
        seam_uvs = {
            tuple(first.uvs[uv_index])
            for face, face_uv in zip(first.faces, first.face_uvs or [], strict=True)
            for vertex, uv_index in zip(face, face_uv, strict=True)
            if vertex == midpoint
        }
        self.assertIn((0.5, 0.0), seam_uvs)
        self.assertIn((11.0, 10.0), seam_uvs)
        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.uvs, second.uvs)
        self.assertEqual(first.faces, second.faces)
        self.assertEqual(first.face_uvs, second.face_uvs)

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

    def test_engine_force_quads_uses_parity_path_closure(self) -> None:
        source = MeshData(
            vertices=np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=float),
            faces=[(0, 1, 2), (0, 2, 3)],
            face_materials=["scan", "scan"],
            name="paired_square",
        )
        output, report = remesh_mesh(source, RemeshOptions(target_faces=1, seed_count=2, force_quads=True))
        self.assertTrue(report.success)
        self.assertTrue(report.closure_applied)
        self.assertEqual(report.closure_method, "PARITY_PATH")
        self.assertTrue(report.quad_only_verified)
        self.assertTrue(topology_summary(output).all_quads)
        self.assertEqual(report.euler_characteristic, 1)
        self.assertIn("TRIAD-Q ParityPath completed", report.message)
        self.assertTrue(
            all(
                candidate["closure_method"] == "PARITY_PATH"
                for candidate in report.candidates
            )
        )

    def test_engine_reports_global_target_closure_when_parity_path_rejects(self) -> None:
        source = MeshData(
            vertices=np.asarray(
                [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                dtype=float,
            ),
            faces=[(0, 1, 2), (0, 2, 3)],
            face_materials=["scan", "scan"],
            name="forced_fallback",
        )

        with patch.object(
            triad_engine,
            "parity_path_quad_closure",
            side_effect=ValueError("forced parity rejection"),
        ):
            output, report = remesh_mesh(
                source,
                RemeshOptions(target_faces=4, seed_count=1, force_quads=True),
            )

        self.assertTrue(report.success)
        self.assertTrue(report.quad_only_verified)
        self.assertEqual(report.closure_method, "GLOBAL_CORNER_TARGET")
        self.assertIn("global corner target closure", report.message)
        self.assertTrue(topology_summary(output).all_quads)

    def test_engine_force_quads_normalizes_vertex_only_bow_tie(self) -> None:
        source = MeshData(
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
            name="engine_bow_tie",
        )
        output, report = remesh_mesh(
            source,
            RemeshOptions(target_faces=2, seed_count=2, force_quads=True),
        )

        self.assertTrue(report.success)
        self.assertEqual(report.vertex_fan_splits, 1)
        self.assertTrue(topology_summary(output).all_quads)

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
        self.assertEqual(report.closure_method, "GLOBAL_CORNER_TARGET")
        self.assertIn("global corner target closure", report.message)
        self.assertIsNotNone(output.uvs)
        self.assertIsNotNone(output.face_uvs)
        self.assertEqual(len(output.face_uvs or []), output.face_count)
        self.assertTrue(all(len(face_uvs) == 4 for face_uvs in output.face_uvs or []))
        np.testing.assert_allclose(output.uvs[: len(source.uvs)], source.uvs)

        used_uvs: dict[str, set[int]] = {"paired": set(), "seam_a": set(), "seam_b": set()}
        for material, face_uvs in zip(output.face_materials, output.face_uvs or [], strict=True):
            used_uvs[material].update(face_uvs)
        self.assertTrue({0, 1, 2, 3}.issubset(used_uvs["paired"]))
        self.assertTrue({4, 5, 6}.issubset(used_uvs["seam_a"]))
        self.assertTrue({7, 8, 9}.issubset(used_uvs["seam_b"]))

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
        self.assertEqual(report.vertex_fan_splits, 0)
        self.assertIn("rejected", report.message.lower())

    def test_augmenting_path_rescues_greedy_strand(self) -> None:
        # Dual chain 0-1-2-3: greedy took the middle pair and stranded both ends.
        allowed = {
            0: [(1.0, 1, ())],
            1: [(1.0, 0, ()), (1.0, 2, ())],
            2: [(1.0, 1, ()), (1.0, 3, ())],
            3: [(1.0, 2, ())],
        }
        mate = {1: 2, 2: 1}
        rescued = triad_engine._augment_matching(mate, allowed, [0, 1, 2, 3])
        self.assertEqual(rescued, 1)
        self.assertEqual(mate, {0: 1, 1: 0, 2: 3, 3: 2})

    @staticmethod
    def _allowed_from_edges(edges: list[tuple[int, int]]) -> dict[int, list[tuple[float, int, tuple]]]:
        allowed: dict[int, list[tuple[float, int, tuple]]] = {}
        for a, b in edges:
            allowed.setdefault(a, []).append((1.0, b, ()))
            allowed.setdefault(b, []).append((1.0, a, ()))
        return allowed

    def test_blossom_matching_finds_path_through_odd_cycles(self) -> None:
        # Two 5-cycles (blossoms) based at free faces 0 and 9, bridged 1-8.
        # The only augmenting path, 0-4=3-2=1-8=7-6=5-9, walks each cycle the
        # long way round; a search without blossom contraction misses it.
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (9, 8), (8, 7), (7, 6), (6, 5), (5, 9), (1, 8)]
        mate = {1: 2, 2: 1, 3: 4, 4: 3, 8: 7, 7: 8, 6: 5, 5: 6}
        rescued = triad_engine._augment_matching(mate, self._allowed_from_edges(edges), list(range(10)))
        self.assertEqual(rescued, 1)
        self.assertEqual(len(mate), 10)
        for a, b in mate.items():
            self.assertEqual(mate[b], a)
            self.assertIn(tuple(sorted((a, b))), {tuple(sorted(edge)) for edge in edges})

    def test_blossom_matching_is_maximum_on_random_graphs(self) -> None:
        def brute_force(vertices: list[int], edges: set[tuple[int, int]]) -> int:
            if not vertices:
                return 0
            first, rest = vertices[0], vertices[1:]
            best = brute_force(rest, edges)
            for other in rest:
                if tuple(sorted((first, other))) in edges:
                    best = max(best, 1 + brute_force([v for v in rest if v != other], edges))
            return best

        rng = np.random.default_rng(7)
        for _trial in range(60):
            count = int(rng.integers(4, 11))
            edges = {
                (a, b) for a in range(count) for b in range(a + 1, count) if rng.random() < 0.3
            }
            mate: dict[int, int] = {}
            for a, b in sorted(edges, key=lambda _edge: rng.random()):
                if a not in mate and b not in mate:
                    mate[a] = b
                    mate[b] = a
            start = len(mate) // 2
            order = [int(v) for v in rng.permutation(count)]
            rescued = triad_engine._augment_matching(mate, self._allowed_from_edges(sorted(edges)), order)
            self.assertEqual(len(mate) // 2, brute_force(list(range(count)), edges))
            self.assertEqual(rescued, len(mate) // 2 - start)
            for a, b in mate.items():
                self.assertEqual(mate[b], a)
                self.assertIn(tuple(sorted((a, b))), edges)

    def test_cross_field_aligns_with_straight_feature_line(self) -> None:
        # Flat grid rotated 30 degrees with random cell diagonals; a material
        # boundary along one grid row is the only constraint. Longest-edge
        # initial guesses sit 45 degrees off the line in 4-RoSy terms.
        size = 16
        angle = np.radians(30.0)
        axis_u = np.array([np.cos(angle), np.sin(angle), 0.0])
        axis_v = np.array([-np.sin(angle), np.cos(angle), 0.0])
        vertices = np.asarray(
            [i * axis_u + j * axis_v for j in range(size + 1) for i in range(size + 1)]
        )
        rng = np.random.default_rng(3)
        faces: list[tuple[int, int, int]] = []
        materials: list[str] = []
        for j in range(size):
            for i in range(size):
                a = j * (size + 1) + i
                b, c, d = a + 1, a + size + 2, a + size + 1
                cell = [(a, b, c), (a, c, d)] if rng.random() < 0.5 else [(a, b, d), (b, c, d)]
                faces.extend(cell)
                materials.extend(["top" if j >= size // 2 else "bottom"] * 2)
        tri = triad_engine._triangulate(
            MeshData(vertices=vertices, faces=faces, face_materials=materials)
        )
        adj = triad_engine._adjacency(tri)
        line = triad_engine._material_edge_mask(adj)
        self.assertEqual(int(line.sum()), size)
        directions = triad_engine._estimate_field(tri, adj, line, smooth_iters=4)
        cosines = np.clip(directions @ axis_u, -1.0, 1.0)
        offsets = np.degrees(np.arccos(np.abs(cosines)))
        offsets = np.minimum(offsets, 90.0 - offsets)  # any of the four cross directions
        self.assertLess(float(offsets.max()), 1.0)


class ComponentDetailTests(unittest.TestCase):
    def test_component_detail_counts_faces_per_shell(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]], float)
        summary = topology_summary(MeshData(vertices=vertices, faces=[(0, 1, 2, 3), (4, 5, 6)]))
        self.assertEqual(summary.component_detail, ((1, 1, 1), (1, 1, 1)))
        self.assertEqual(summary.component_signature, ((1, 1), (1, 1)))
        self.assertEqual(summary.connected_components, 2)


if __name__ == "__main__":
    unittest.main()
