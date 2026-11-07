# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors
"""Test helper: outside Blender there is no mathutils BVH, so the checks that need one find nothing (the three lookups below).

The add-on itself has no such fallback; only the tests swap these lookups.
"""

import numpy as np

from triadq import checks

try:
    import mathutils  # noqa: F401  (inside Blender: the real lookups stay)
except ImportError:

    def _no_source(*args, **kwargs):
        return None
        yield

    def _no_overlap(co, faces, pad, *rest, **kwargs):
        return np.zeros(len(pad), bool)
        yield

    checks.source_steps = _no_source
    checks.crossing_pairs = lambda co, faces: np.zeros((0, 2), np.int64)
    checks._overlapping_steps = _no_overlap
