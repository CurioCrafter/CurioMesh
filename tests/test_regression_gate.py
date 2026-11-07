# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Regression gate for bad faces (0.9.1 turned revolved triangle parts inside out). Pure Python."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _nobvh  # noqa: E402,F401  (no BVH outside Blender)

from triadq import checks, parts, special  # noqa: E402
from triadq.types import drain  # noqa: E402

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "turned_tube.json").read_text())
SOURCE = (np.asarray(FIXTURE["source_co"]), [tuple(f) for f in FIXTURE["source_faces"]])
FROZEN = (np.asarray(FIXTURE["result_co"]), [tuple(f) for f in FIXTURE["result_faces"]])
FROZEN_FACES = 42  # the frozen result's size: a rebuild that grows past a third more fails the gate
PURE = ("bent", "flat", "duplicate", "folded_over")  # the masks that need no source


def turned_in(co, faces, axis_point=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)) -> int:
    """Faces of a tube whose normal points toward its axis (caps, with no sideways part, never count)."""
    co = np.asarray(co, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    pad = checks.corner_table(faces)
    normal, _area, middle = checks._normals(co, pad)
    out = (middle - axis_point) - ((middle - axis_point) @ axis)[:, None] * axis
    return int(((normal * out).sum(axis=1) < -1e-9).sum())


def volume(co, faces) -> float:
    """Signed volume of a closed mesh: negative when it is inside out."""
    p = np.asarray(co, dtype=np.float64)
    return float(sum(np.dot(p[f[0]], np.cross(p[f[i]], p[f[i + 1]])) for f in faces for i in range(1, len(f) - 1)) / 6.0)


class GateTests(unittest.TestCase):
    def test_the_source_of_triangles_is_what_the_gate_assumes(self):
        self.assertEqual(turned_in(*SOURCE), 0)
        self.assertGreater(volume(*SOURCE), 0.0)

    def test_every_rebuild_of_a_triangle_part_faces_out(self):
        built = list(parts.rebuilds("0", drain(parts.fit_revolve_steps(*SOURCE))))
        self.assertGreaterEqual(len(built), 3)  # a round part starts at 16 steps (parts.min_spin): 16, 24, 32
        for co, polys, _kinds in built:
            turned = parts.orient_like(co, polys, *SOURCE)
            self.assertEqual(turned_in(co, turned), 0)
            self.assertGreater(volume(co, turned), 0.0)
            backwards = [f[::-1] for f in turned]  # what 0.9.1 shipped: the vote must still turn it back
            self.assertEqual(parts.orient_like(co, backwards, *SOURCE), turned)

    def test_copies_face_like_their_own_source(self):
        turn = np.array([(0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
        for mirror in (False, True):
            twin = SOURCE[0] @ (turn @ np.diag([1.0, 1.0, -1.0 if mirror else 1.0])).T + (7.0, 3.0, 1.0)
            move = special.match_copy(SOURCE[0], twin)
            self.assertIsNotNone(move)
            co = special.moved(FROZEN[0], move)
            faces = special.wound(FROZEN[1], move)
            self.assertGreater(volume(co, faces), 0.0)  # mirroring without winding again would be negative
            self.assertEqual(checks.defects(co, faces).counts()["any"], 0)

    def test_the_frozen_result_has_no_defects(self):
        self.assertEqual(turned_in(*FROZEN), 0)
        counts = checks.defects(*FROZEN).counts()
        self.assertEqual({k: counts[k] for k in PURE}, dict.fromkeys(PURE, 0))
        self.assertLessEqual(len(FROZEN[1]), FROZEN_FACES * 4 // 3)


if __name__ == "__main__":
    unittest.main()
