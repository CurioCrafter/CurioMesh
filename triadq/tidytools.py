# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The maths of each Tidy Tool, as a `trial.Trial` subclass (plain NumPy, no bpy).

Smooth Corners is the small example every tool follows: it slides each free corner part of the way toward the middle
of its neighbours (umbrella smoothing, Taubin 1995 without the shrink step, as the projection undoes the shrink) and
puts it back on your original mesh after each step, so nothing shrinks and every corner stays on the source surface.
"""

from __future__ import annotations

import numpy as np

from .trial import Base, Candidate, Trial, edges_of

NOTHING_FREE = "Nothing can move here: every corner is held (borders, hard edges or the mirror plane)."


class SmoothCorners(Trial):
    name = "Smooth Corners"
    group = "RELAX"
    handles = ("strength",)
    words = ("0", "1")
    defaults = {"strength": 0.6, "steps": 6}
    blendable = True
    stages = ("poles", "move", "look")

    def check(self, base: Base) -> str:
        return "" if base.free.any() else NOTHING_FREE

    def run(self, base: Base, settings: dict, rough: bool = False):
        strength = float(settings.get("strength", self.defaults["strength"]))
        steps = int(settings.get("steps", self.defaults["steps"]))
        steps = max(1, steps // 4) if rough else max(1, steps)
        free = base.free
        co = base.co.copy()
        edges = edges_of(base.faces)
        count = np.bincount(edges.ravel(), minlength=len(co)).astype(np.float64)
        count[count == 0] = 1.0
        for step in range(steps):
            middle = np.stack([np.bincount(edges[:, 0], co[edges[:, 1], k], len(co)) + np.bincount(edges[:, 1], co[edges[:, 0], k], len(co)) for k in range(3)], axis=1) / count[:, None]
            co[free] += 0.5 * strength * (middle[free] - co[free])
            yield (step + 0.5) / steps
            if base.project is not None:
                co[free] = base.project(co[free])
            yield (step + 1.0) / steps
        return Candidate(co, strength=strength, exact=not rough)
