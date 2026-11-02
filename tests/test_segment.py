# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import patch, segment  # noqa: E402


def dumbbell(sides=24, rings=40):
    """Two balls on a thin bar, as one closed triangle mesh along z."""

    def radius(t):
        ball_a = max(0.0, 1.0 - ((t - 0.2) / 0.2) ** 2) ** 0.5
        ball_b = max(0.0, 1.0 - ((t - 0.8) / 0.2) ** 2) ** 0.5
        return max(ball_a, ball_b, 0.12)

    rows = [(np.cos(2 * np.pi * i / sides) * radius(r / rings) * 0.4, np.sin(2 * np.pi * i / sides) * radius(r / rings) * 0.4, 2.0 * r / rings) for r in range(rings + 1) for i in range(sides)]
    co = np.array(rows + [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)])
    faces = []
    for r in range(rings):
        for i in range(sides):
            a, b, c, d = r * sides + i, r * sides + (i + 1) % sides, (r + 1) * sides + (i + 1) % sides, (r + 1) * sides + i
            faces += [(a, b, c), (a, c, d)]
    south, north = len(co) - 2, len(co) - 1
    for i in range(sides):
        faces.append((south, (i + 1) % sides, i))
        faces.append((north, rings * sides + i, rings * sides + (i + 1) % sides))
    return co, faces


def grid(nx, ny):
    co = np.array([(float(i), float(j), 0.0) for j in range(ny + 1) for i in range(nx + 1)])
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i) for j in range(ny) for i in range(nx)]
    return co, faces


class FindTests(unittest.TestCase):
    def test_splits_at_the_neck(self):
        co, faces = dumbbell()
        tree = segment.build(co, faces, cells=20)
        labels = segment.find(tree, faces, 0.6)
        self.assertGreaterEqual(labels.max() + 1, 2)
        _n, _a, centroid = segment.face_frames(co, faces)
        low, high = labels[centroid[:, 2] < 0.3], labels[centroid[:, 2] > 1.7]
        self.assertNotEqual(np.bincount(low).argmax(), np.bincount(high).argmax())

    def test_slider_is_monotonic_enough(self):
        co, faces = dumbbell()
        tree = segment.build(co, faces, cells=20)
        self.assertLessEqual(len(np.unique(segment.cut(tree, 0.0))), len(np.unique(segment.cut(tree, 1.0))))

    def test_small_loose_piece_stays_whole(self):
        co, faces = grid(3, 3)
        labels = segment.find(segment.build(co, faces), faces, 1.0)
        self.assertEqual(labels.max(), 0)

    def test_every_segment_is_one_piece(self):
        co, faces = grid(8, 8)
        labels = np.zeros(len(faces), dtype=np.int64)
        labels[[0, 1, 40, 41]] = 1  # two islands of one label
        fa, fb, _u, _v = segment.face_links(faces)
        out = segment.one_piece_each(labels, fa, fb)
        for value in np.unique(out):
            self.assertEqual(len(patch.mask_components(faces, out == value)), 1)


class LineTests(unittest.TestCase):
    def test_sides_and_endless_ends(self):
        stroke = [(0.0, 0.0), (1.0, 0.0)]
        points = [(0.5, 1.0), (0.5, -1.0), (50.0, 1.0), (-50.0, -1.0)]
        self.assertEqual(segment.side_of_line(points, stroke).tolist(), [True, False, True, False])


class SeamTests(unittest.TestCase):
    def test_a_stepped_seam_on_a_flat_sheet_gets_straighter(self):
        co, faces = grid(12, 12)
        centres = co[np.asarray(faces)].mean(axis=1)
        labels = (centres[:, 0] + (centres[:, 1].astype(int) % 2) > 6.0).astype(np.int64)
        moved = segment.straighten_seams(co, faces, labels)
        fa, fb, u, v = segment.face_links(faces)
        across = labels[fa] != labels[fb]
        before = np.linalg.norm(co[u[across]] - co[v[across]], axis=1).sum()
        after = np.linalg.norm(moved[u[across]] - moved[v[across]], axis=1).sum()
        self.assertLess(after, before)

    def test_a_seam_on_a_crease_stays(self):
        co, faces = grid(8, 8)
        co[:, 2] = np.abs(co[:, 0] - 4.0)  # a roof ridge along x = 4
        centres = co[np.asarray(faces)].mean(axis=1)
        moved = segment.straighten_seams(co, faces, (centres[:, 0] > 4.0).astype(np.int64))
        ridge = np.flatnonzero(co[:, 0] == 4.0)
        self.assertTrue(np.array_equal(moved[ridge], co[ridge]))


class JoinTests(unittest.TestCase):
    def _seam(self, co, faces, ids):
        return segment.seam_loops(co, faces, ids)

    def test_bent_seam_still_closes_to_all_quads(self):
        co, faces = grid(10, 10)
        co[:, 2] = 0.8 * np.sin(co[:, 0]) * np.cos(co[:, 1])  # bent enough to refuse some templates
        tris = [t for f in faces for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))]
        seam = self._seam(co, faces, range(len(faces)))
        points, made, _mats = segment.pair_on_seams(co, tris, ["0"] * len(tris), seam)
        self.assertEqual(sum(len(f) != 4 for f in made), 0)
        self.assertEqual(segment.open_edges(made), len(seam[0]))
        self.assertEqual({tuple(p) for p in points[patch.boundary_loops(made, range(len(made)))[0][0]].tolist()}, {tuple(p) for p in seam[0].tolist()})

    def test_guide_edges_stay_edges(self):
        co, faces = grid(10, 10)
        tris = [t for f in faces for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))]
        row = [(5 * 11 + i, 5 * 11 + i + 1) for i in range(2, 8)]
        points, made, _mats = segment.pair_on_seams(co, tris, ["0"] * len(tris), self._seam(co, faces, range(len(faces))), guide_edges=row)
        fa, _fb, u, v = segment.face_links(made)
        have = {(tuple(points[a]), tuple(points[b])) for a, b in zip(u.tolist(), v.tolist())}
        kept = sum((tuple(co[a]), tuple(co[b])) in have or (tuple(co[b]), tuple(co[a])) in have for a, b in row)
        self.assertGreaterEqual(kept, 4)  # closure may split one, but the line is still there

    def test_conform_coarse_part_gets_every_seam_vertex(self):
        co, faces = grid(4, 4)
        fine = [np.array([(i * 0.5, 0.0, 0.0) for i in range(8)] + [(4.0, j * 0.5, 0.0) for j in range(8)] + [(4.0 - i * 0.5, 4.0, 0.0) for i in range(8)] + [(0.0, 4.0 - j * 0.5, 0.0) for j in range(8)])]
        points, out, origin, ids = segment.conform(co, faces, fine)
        self.assertEqual(len(origin), len(out))
        loops, pinched = patch.boundary_loops(out, range(len(out)))
        self.assertFalse(pinched)
        self.assertEqual(len(loops[0]), 32)
        self.assertEqual(sorted(loops[0]), sorted(ids[0].tolist()))
        self.assertTrue(np.array_equal(np.sort(points[ids[0]], axis=0), np.sort(fine[0], axis=0)))

    def test_conform_dense_part_collapses_onto_a_coarse_seam(self):
        co, faces = grid(8, 8)
        coarse = [np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (8.0, 0.0, 0.0), (8.0, 4.0, 0.0), (8.0, 8.0, 0.0), (4.0, 8.0, 0.0), (0.0, 8.0, 0.0), (0.0, 4.0, 0.0)])]
        _p, out, _o, ids = segment.conform(co, faces, coarse)
        loops, _pinched = patch.boundary_loops(out, range(len(out)))
        self.assertEqual(sorted(loops[0]), sorted(ids[0].tolist()))

    def test_refill_keeps_the_mesh_closed(self):
        co, faces = grid(10, 10)
        ids = np.flatnonzero([(i % 10) < 5 for i in range(len(faces))])
        rest = [faces[i] for i in range(len(faces)) if i not in set(ids.tolist())]
        points, made, _warn = segment.fill_part(co, faces, ids, "SUBD", 1.5)
        joined = np.vstack((co, points))
        final = rest + [tuple(v + len(co) for v in f) for f in made]
        welded, welded_faces = segment.weld_border(joined, final)
        self.assertEqual(segment.open_edges(welded_faces), segment.open_edges(faces))
        self.assertTrue(all(len(f) == 4 for f in made))
        self.assertLess(len(welded), len(joined))

    def test_pair_on_seams_is_all_quads_with_the_seam_kept(self):
        co, faces = grid(10, 10)
        tris = [t for f in faces for t in ((f[0], f[1], f[2]), (f[0], f[2], f[3]))]
        seam = self._seam(co, faces, range(len(faces)))
        points, made, _mats = segment.pair_on_seams(co, tris, ["0"] * len(tris), seam)
        loops, pinched = patch.boundary_loops(made, range(len(made)))
        self.assertFalse(pinched)
        self.assertEqual(len(loops[0]), len(seam[0]))
        self.assertEqual(sum(len(f) != 4 for f in made), 0)
        kept = {tuple(p) for p in points[loops[0]].tolist()}
        self.assertEqual(kept, {tuple(p) for p in seam[0].tolist()})


if __name__ == "__main__":
    unittest.main()
