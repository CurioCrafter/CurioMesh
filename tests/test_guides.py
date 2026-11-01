# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triadq import flatquad  # noqa: E402
from triadq.types import Cancelled, MeshData  # noqa: E402


def _sheet(n=12):
    vertices = np.array([(x, y, 0.0) for y in range(n + 1) for x in range(n + 1)])
    faces = []
    for y in range(n):
        for x in range(n):
            a, b, c, d = y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x
            faces += [(a, b, c), (a, c, d)]
    return MeshData(vertices, faces)


class CancelTests(unittest.TestCase):
    def test_pairing_stops_when_asked(self):
        with self.assertRaises(Cancelled):
            flatquad.pair_and_close(_sheet(), cancelled=lambda: True)
        out, _report = flatquad.pair_and_close(_sheet())  # the stop flag does not leak into the next call on this thread
        self.assertTrue(out.faces)


if __name__ == "__main__":
    unittest.main()
