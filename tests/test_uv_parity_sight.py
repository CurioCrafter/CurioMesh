# SPDX-License-Identifier: GPL-3.0-or-later
"""UV visibility (K4): cube, pocket, view counts, held pictures, native/NumPy parity, speed, wrapper never blocks."""

from __future__ import annotations

import time
import unittest

import numpy as np

import uv_parity_common as common
from uv_parity_common import needs_native

from uv_parity_common import turbo  # noqa: E402  (None in a build without it)

uvsight = common.addon_module("triadq.uvsight")


def cube(side=1.0, shift=0.0, inward=False):
    pts = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=np.float64) * side + shift
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    if inward:
        quads = [q[::-1] for q in quads]
    return pts, quads


def pocket():
    """A cube with a closed hollow inside: the inner faces point at the hollow and no view can reach them."""

    a, qa = cube(1.0)
    b, qb = cube(0.4, 0.3, inward=True)
    return np.vstack((a, b)), qa + [tuple(v + 8 for v in q) for q in qb]


def lumpy_sphere(rings=40, around=60, seed=3):
    rng = np.random.default_rng(seed)
    pts = [(0.0, 0.0, 1.0)]
    for r in range(1, rings):
        t = np.pi * r / rings
        for c in range(around):
            p = 2 * np.pi * c / around
            pts.append((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)))
    pts.append((0.0, 0.0, -1.0))
    pts = np.array(pts)
    pts *= (1.0 + 0.15 * np.sin(5 * pts[:, 0] + 2 * pts[:, 1]) * pts[:, 2] + rng.normal(0, 0.004, len(pts)))[:, None]
    quads = []
    def ring(r, c):
        return 1 + (r - 1) * around + c % around

    for r in range(1, rings - 1):
        for c in range(around):
            quads.append((ring(r, c), ring(r, c + 1), ring(r + 1, c + 1), ring(r + 1, c)))
    for c in range(around):
        quads.append((0, ring(1, c + 1), ring(1, c)))
        quads.append((len(pts) - 1, ring(rings - 1, c), ring(rings - 1, c + 1)))
    return pts, quads


def run(job, limit=60.0):
    """Wait for a job; returns (result, the longest time any single call here took)."""

    worst, end = 0.0, time.perf_counter() + limit
    while True:
        t = time.perf_counter()
        done = job.done()
        worst = max(worst, time.perf_counter() - t)
        if done:
            return job.result(), worst
        if time.perf_counter() > end:
            raise TimeoutError
        time.sleep(0.001)


def seen_native(points, polys, views=6, pixels=0, all_sides=True):
    pic = turbo.uvsight_pictures(points, polys, views, pixels, all_sides)
    run(turbo.uvsight_fill(pic))
    return pic, run(turbo.uvsight_seen(pic))[0]


def edge_list(polys):
    """Edges in the kernel's order (first occurrence over the corners) with the faces using them."""

    order, faces = {}, {}
    for f, p in enumerate(polys):
        for k in range(len(p)):
            key = tuple(sorted((p[k], p[(k + 1) % len(p)])))
            order.setdefault(key, len(order))
            faces.setdefault(key, []).append(f)
    return list(order), faces


@needs_native
class Cube(unittest.TestCase):
    def test_cube_is_seen_once(self):
        pts, quads = cube()
        _, (seen, _, _) = seen_native(pts, quads, 6, 512)
        self.assertGreater(seen.min(), 0.98)
        self.assertLessEqual(seen.max(), 1.0)

    def test_inside_pixels_match_the_analytic_length(self):
        pts, quads = cube(1.0)
        keys, faces = edge_list(quads)
        normals = np.array([np.cross(pts[q[1]] - pts[q[0]], pts[q[3]] - pts[q[0]]) for q in quads])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for views in (14, 26):
            pic, (_, inside, outline) = seen_native(pts, quads, views, 1024)
            size, _, half = pic.frame()
            pixel = 2 * half / size
            dirs = np.array(uvsight.view_dirs(views))
            want = np.zeros(len(keys))
            for d in dirs:
                for e, key in enumerate(keys):
                    a, b = faces[key]
                    if normals[a] @ d > 1e-9 and normals[b] @ d > 1e-9:
                        v = pts[key[1]] - pts[key[0]]
                        want[e] += np.linalg.norm(v - (v @ d) * d) / pixel
            want /= len(dirs)
            self.assertTrue((want > 0).any())
            hit = want > 0
            np.testing.assert_allclose(inside[hit], want[hit], rtol=0.02, err_msg=f"{views} views")
            self.assertLess(inside[~hit].max(initial=0.0), 0.02 * want.max())
            self.assertTrue((outline > 0).all())  # every cube edge is on an outline in some view

    def test_pocket_face_is_not_seen(self):
        pts, quads = pocket()
        _, (seen, _, _) = seen_native(pts, quads, 14, 512)
        self.assertGreater(seen[:6].min(), 0.98)
        self.assertLess(seen[6:].max(), 0.05)

    def test_any_view_count_and_growing_keeps_the_pictures(self):
        pts, quads = lumpy_sphere(12, 16)
        for views in (1, 6, 14, 26, 40):
            pic, (seen, inside, outline) = seen_native(pts, quads, views, 128)
            self.assertEqual(pic.drawn, views)
            self.assertEqual(len(seen), len(quads))
            self.assertTrue(np.isfinite(seen).all() and np.isfinite(inside).all() and np.isfinite(outline).all())
        pic = turbo.uvsight_pictures(pts, quads, 6, 128)
        run(turbo.uvsight_fill(pic))
        run(turbo.uvsight_seen(pic))
        self.assertEqual(pic.renders, 6)
        pic.set_views(14, True)
        run(turbo.uvsight_fill(pic))
        self.assertEqual((pic.drawn, pic.renders), (14, 14))  # only the 8 new views were drawn

    def test_reading_again_draws_nothing(self):
        pts, quads = lumpy_sphere(12, 16)
        pic, first = seen_native(pts, quads, 6, 128)
        drawn = pic.renders
        for _ in range(3):
            run(turbo.uvsight_fill(pic))
            again = run(turbo.uvsight_seen(pic))[0]
        pic.set_all_sides(False)  # new weights, same pictures
        low = run(turbo.uvsight_seen(pic))[0]
        self.assertEqual(pic.renders, drawn)
        np.testing.assert_array_equal(first[0], again[0])
        self.assertEqual(len(low[0]), len(first[0]))

    def test_stopped_fill_resumes(self):
        pts, quads = lumpy_sphere(30, 40)
        pic = turbo.uvsight_pictures(pts, quads, 26, 1024)
        job = turbo.uvsight_fill(pic)
        job.stop()
        try:
            run(job)
        except InterruptedError:
            pass
        run(turbo.uvsight_fill(pic))
        self.assertEqual(pic.drawn, 26)
        self.assertEqual(len(run(turbo.uvsight_seen(pic))[0][0]), len(quads))

    def test_empty_and_bad_input(self):
        pic = turbo.uvsight_pictures(np.zeros((0, 3)), [], 6, 64)
        run(turbo.uvsight_fill(pic))
        seen, inside, outline = run(turbo.uvsight_seen(pic))[0]
        self.assertEqual((len(seen), len(inside), len(outline)), (0, 0, 0))
        with self.assertRaises(ValueError):
            turbo.uvsight_pictures(np.zeros((3, 3)), [(0, 1, 5)], 6, 64)
        with self.assertRaises(ValueError):
            turbo.uvsight_pictures(np.zeros((3, 3)), [(0, 1, 2)], 100, 20000)  # would not fit in memory


@needs_native
class Parity(unittest.TestCase):
    def test_numpy_reader_matches_the_kernel_on_the_same_pictures(self):
        pts, quads = lumpy_sphere(14, 18)
        for views, all_sides in ((6, True), (14, False)):
            pic, (seen, inside, outline) = seen_native(pts, quads, views, 160, all_sides)
            size, center, half = pic.frame()
            dirs, weights = pic.views()
            held = uvsight.Pictures(dirs.reshape(-1, 3), weights, size, center, half, [pic.face(k).reshape(size, size) for k in range(views)], [pic.depth(k).reshape(size, size) for k in range(views)])
            twin = uvsight.Sight(pts, quads, pictures=held).run()
            np.testing.assert_allclose(twin.seen_face, seen, atol=1e-6)
            np.testing.assert_allclose(twin.inside_px, inside, atol=1e-6)
            np.testing.assert_allclose(twin.outline_px, outline, atol=1e-6)
            self.assertEqual(twin.renders, 0)

    def test_numpy_views_and_weights_match(self):
        for count in (6, 14, 26, 40):
            native_dirs, native_weights = turbo.native().uvsight_dirs(count, False)
            np.testing.assert_allclose(native_dirs.reshape(-1, 3), uvsight.view_dirs(count), atol=1e-12)
            np.testing.assert_allclose(native_weights, uvsight.view_weights(uvsight.view_dirs(count), False), atol=1e-12)

    def test_fallback_draws_with_a_six_view_function(self):
        pts, quads = cube()
        pic, (seen, inside, _) = seen_native(pts, quads, 14, 256)
        twin = uvsight.Sight(pts, quads, views=14, pixels=256, draw=turbo.pictures).run()
        self.assertEqual(twin.renders, 14)
        self.assertGreater(twin.seen_face.min(), 0.97)
        np.testing.assert_allclose(twin.inside_px.sum(), inside.sum(), rtol=0.03)

    def test_slices_are_short(self):
        pts, quads = lumpy_sphere(14, 18)
        twin = uvsight.Sight(pts, quads, views=6, pixels=256, draw=turbo.pictures)
        worst, steps = 0.0, 0
        while True:
            t = time.perf_counter()
            done = twin.step()
            worst = max(worst, time.perf_counter() - t)
            steps += 1
            if done:
                break
        print(f"numpy slices: {steps} steps, longest {worst * 1000:.1f} ms")
        self.assertLess(worst, 0.25)  # the six-view draw of one view is the longest slice


@needs_native
class Speed(unittest.TestCase):
    def test_six_views_1024_on_23k_faces(self):
        pts, quads = lumpy_sphere(152, 152)
        self.assertGreaterEqual(len(quads), 22000)
        started = time.perf_counter()
        t = time.perf_counter()
        pic = turbo.uvsight_pictures(pts, quads, 6, 1024)
        make = time.perf_counter() - t
        fill_job = turbo.uvsight_fill(pic)
        _, w1 = run(fill_job)
        seen_job = turbo.uvsight_seen(pic)
        _, w2 = run(seen_job)
        total = time.perf_counter() - started
        print(f"{len(quads)} faces: make {make * 1000:.1f} ms, 6 views at 1024 px total {total:.3f} s, longest poll {max(w1, w2) * 1000:.2f} ms")
        self.assertLessEqual(total, 0.5)
        self.assertLess(make, 0.01)
        self.assertLess(max(w1, w2), 0.01)
        t = time.perf_counter()
        job = turbo.uvsight_seen(pic)
        self.assertLess(time.perf_counter() - t, 0.01)
        run(job)

    def test_draw_job_hands_over_pictures_in_short_slices(self):
        pts, quads = lumpy_sphere(152, 152)
        uv_fast = common.addon_module("uv_fast")
        job = uv_fast._DrawJob(pts, quads, 6, 2048, None)
        while not job.done():
            time.sleep(0.001)
        gen, worst = job.result(), 0.0
        while True:
            t = time.perf_counter()
            try:
                next(gen)
            except StopIteration as end:
                faces, depths = end.value
                break
            finally:
                worst = max(worst, time.perf_counter() - t)
        print(f"picture hand-over: longest slice {worst * 1000:.2f} ms")
        self.assertLess(worst, 0.010)
        self.assertEqual(len(faces), 6)
        np.testing.assert_array_equal(faces[3].ravel(), job._pictures.face(3))
        np.testing.assert_array_equal(depths[3].ravel(), job._pictures.depth(3).astype(np.float32))


if __name__ == "__main__":
    unittest.main()
