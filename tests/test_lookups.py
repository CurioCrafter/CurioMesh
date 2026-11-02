# SPDX-License-Identifier: GPL-3.0-or-later
"""The native lookup trees and checks must give what slow, plain code gives. Skips when the native core isn't built."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import turbo  # noqa: E402
from triadq import checks  # noqa: E402
from triadq.types import Cancelled  # noqa: E402

LOOKUPS = hasattr(turbo.native(), "FaceLookup")


def rumpled_grid(n: int, seed: int = 0) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """A bumpy sheet of quads and triangles."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(n + 1.0), np.arange(n + 1.0), indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), rng.normal(0.0, 0.3, (n + 1) ** 2)))
    faces: list[tuple[int, ...]] = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1
            faces += [(a, b, c, d)] if (i + j) % 3 else [(a, b, c), (a, c, d)]
    return points, faces


def cut(points, poly) -> list[tuple[int, int, int]]:
    """A polygon's triangles as Blender's BVH cuts it: see `turbo.pictures`."""
    if len(poly) == 3:
        return [poly]
    a, b, c, d = points[list(poly)]
    facing = np.cross(c - a, d - b)
    if np.cross(a - d, b - a) @ facing < 0.0 or np.cross(c - b, d - c) @ facing < 0.0:
        return [(poly[0], poly[1], poly[2]), (poly[2], poly[3], poly[0])]
    return [(poly[3], poly[0], poly[1]), (poly[1], poly[2], poly[3])]


def closest_on_triangle(p, a, b, c) -> np.ndarray:
    """By brute force: the best of the plane's foot, when inside, and the three clamped sides."""
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    foot = p - n * ((p - a) @ n) / (n @ n)
    v, w = np.linalg.lstsq(np.column_stack((ab, ac)), foot - a, rcond=None)[0]
    options = [foot] if v >= 0 and w >= 0 and v + w <= 1 else []
    for s, e in ((a, b), (b, c), (c, a)):
        options.append(s + (e - s) * np.clip((p - s) @ (e - s) / ((e - s) @ (e - s)), 0.0, 1.0))
    return min(options, key=lambda q: float((q - p) @ (q - p)))


def pure_cross(a, b) -> bool:
    """Do two triangles cross? Their pieces on each other's planes must overlap along the line where the planes meet."""
    na, nb = np.cross(a[1] - a[0], a[2] - a[0]), np.cross(b[1] - b[0], b[2] - b[0])
    line = np.cross(na, nb)
    if line @ line == 0.0:
        return False

    def span(t, n, on):
        side = (t - on) @ n
        if (side > 0).all() or (side < 0).all():
            return None
        ends = []
        for i in range(3):
            j = (i + 1) % 3
            if side[i] * side[j] <= 0.0 and side[i] != side[j]:
                ends.append((t[i] + (t[j] - t[i]) * side[i] / (side[i] - side[j])) @ line)
        return (min(ends), max(ends)) if ends else None

    sa, sb = span(a, nb, b[0]), span(b, na, a[0])
    return sa is not None and sb is not None and sa[0] < sb[1] and sb[0] < sa[1]


@unittest.skipUnless(LOOKUPS, f"native module without the lookups: {turbo.status()}")
class LookupTests(unittest.TestCase):
    def test_nearest_points_match_brute_force(self):
        rng = np.random.default_rng(1)
        points, queries = np.round(rng.random((3000, 3)), 2), rng.random((400, 3))  # rounded: many exact ties
        ids, away = turbo.point_tree(points).nearest(queries)
        gaps = np.linalg.norm(queries[:, None] - points[None], axis=2)
        np.testing.assert_allclose(away, gaps.min(axis=1), rtol=1e-12)
        np.testing.assert_array_equal(ids, gaps.argmin(axis=1))  # argmin takes the first: ties go to the lowest index

    def test_nearest_of_own_kind_looks_only_so_far(self):
        rng = np.random.default_rng(2)
        points, queries = rng.random((800, 3)), rng.random((100, 3))
        point_kind, query_kind = rng.integers(0, 6, 800), rng.integers(0, 6, 100)
        out = turbo.point_tree(points).first_of_kind(queries, query_kind, point_kind, 8)
        for q, kind, got in zip(queries, query_kind, out):
            order = np.argsort(np.linalg.norm(points - q, axis=1))[:8]
            same = order[point_kind[order] == kind]
            self.assertEqual(got, same[0] if len(same) else -1)

    def test_nearest_faces_match_brute_force(self):
        points, faces = rumpled_grid(12)
        points = points.astype(np.float32).astype(np.float64)  # the tree keeps single precision
        rng = np.random.default_rng(3)
        queries = rng.random((200, 3)) * [12, 12, 4] - [0, 0, 2]
        at, face, away = turbo.face_tree(points, faces).nearest(queries)
        tris = [(f, tri) for f, poly in enumerate(faces) for tri in cut(points, poly)]
        for k, q in enumerate(queries):
            found = [closest_on_triangle(q, *points[list(t)]) for _f, t in tris]
            gaps = [np.linalg.norm(p - q) for p in found]
            self.assertAlmostEqual(away[k], min(gaps), places=9)
            np.testing.assert_allclose(at[k], found[int(np.argmin(gaps))], atol=1e-6)
            self.assertAlmostEqual(np.linalg.norm(at[k] - q), away[k], places=9)
            self.assertIn(face[k], [f for (f, _t), g in zip(tris, gaps) if g <= min(gaps) + 1e-9])

    def test_many_sided_faces_are_covered(self):
        ring = np.array([[0.0, 0, 0], [4, 0, 0], [4, 4, 0], [2, 1, 0], [0, 4, 0]])  # not convex
        tree = turbo.face_tree(ring, [(0, 1, 2, 3, 4)])
        _at, _face, away = tree.nearest(np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [2.0, 3.0, 0.0]]))
        np.testing.assert_allclose(away[:2], 0.0, atol=1e-9)
        self.assertGreater(away[2], 0.5)  # over the notch

    def test_crossing_pairs_match_brute_force(self):
        points, faces = rumpled_grid(5, seed=4)
        extra = np.column_stack((points[:, 0] + 0.37, points[:, 2] * 3.0 + 4.0, points[:, 1] * 0.4 - 1.0))  # a second sheet, through the first. Off the grid: edges that only touch are rounding-dependent
        offset = len(points)
        points = np.vstack((points, extra)).astype(np.float32).astype(np.float64)
        faces = faces + [tuple(v + offset for v in f) for f in faces]

        expected = set()
        for i, f in enumerate(faces):
            for j in range(i + 1, len(faces)):
                g = faces[j]
                if set(f) & set(g):
                    continue
                if any(pure_cross(points[list(a)], points[list(b)]) for a in cut(points, f) for b in cut(points, g)):
                    expected.add((i, j))
        self.assertGreater(len(expected), 3)
        found = turbo.face_tree(points, faces).crossing_pairs()
        self.assertEqual({tuple(p) for p in found.tolist()}, expected)
        self.assertEqual(found.tolist(), sorted(found.tolist()))

    def test_mesh_checks_match_the_python_ones(self):
        rng = np.random.default_rng(5)
        points, faces = rumpled_grid(10, seed=5)
        faces = list(faces)
        faces[3] = faces[3][::-1]  # a flip
        faces += [faces[7], faces[7][::-1], (0, 1, 50)]  # duplicates, a non-manifold edge
        del faces[20:24]  # holes
        piece = rng.integers(0, 5, len(faces))
        broken, bad_edges, flipped, dups, holes, per = turbo.mesh_checks(faces, piece)
        want_broken, want_bad, want_flipped, want_dups = checks.edge_flags(faces)
        want_holes, want_per = checks.hole_counts(faces, piece)
        np.testing.assert_array_equal(broken, want_broken)
        self.assertEqual((bad_edges, flipped, dups, holes, per), (want_bad, want_flipped, want_dups, want_holes, want_per))

    def test_a_stop_ends_a_running_call_and_not_the_next(self):
        points = np.random.default_rng(6).random((1000, 3))
        tree = turbo.point_tree(points)
        turbo._stoppable(tree._core.nearest, turbo._query(points))  # makes the flag
        flag = turbo._stop
        turbo.stop_native()
        with self.assertRaises(InterruptedError):
            tree._core.nearest(turbo._query(points), flag)
        self.assertEqual(len(tree.nearest(points)[0]), 1000)
        real = turbo._stop
        turbo._stop = flag  # as a call that had already started when the stop came
        try:
            with self.assertRaises(Cancelled):
                tree.nearest(points)
        finally:
            turbo._stop = real


class FallbackTests(unittest.TestCase):
    def test_no_core_means_blender_or_python(self):
        turbo.set_enabled(False)
        try:
            points, faces = rumpled_grid(3)
            self.assertIsNone(turbo.point_tree(points))
            self.assertIsNone(turbo.face_tree(points, faces))
            self.assertIsNone(turbo.mesh_checks(faces, np.zeros(len(faces), dtype=np.int64)))
            turbo.stop_native()  # harmless
            result = checks.inspect(points, faces, np.zeros(len(faces), dtype=np.int64), pairs=np.zeros((0, 2), dtype=np.int64))
            self.assertEqual((result.nm_edges, result.flipped, result.dups, result.holes), (0, 0, 0, 1))
        finally:
            turbo.set_enabled(True)


if __name__ == "__main__":
    unittest.main()
