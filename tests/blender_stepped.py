# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Run with: blender --background --factory-startup --python tests/blender_stepped.py

Auto Segments measures a part at sizes made one small step after another, and the result is still all good quads.
"""

import importlib
from pathlib import Path
import sys

import bmesh
import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
addon = importlib.import_module(REPO.name)
addon.register()
autoretopo = importlib.import_module(REPO.name + ".autoretopo")

sphere = bmesh.new()
bmesh.ops.create_icosphere(sphere, subdivisions=5, radius=1.0)
for v in sphere.verts:  # bumps, so small sizes change the view
    v.co *= 1.0 + 0.04 * ((v.co.x * 9.0) % 1.0)
co = [tuple(v.co) for v in sphere.verts]
faces = [tuple(v.index for v in f.verts) for f in sphere.faces]
sphere.free()

job = autoretopo.AutoJob(bpy.context, co, faces, [0] * len(faces), 6.0)
job.wait()
assert not job.warnings, job.warnings
counts = [level.faces for level in job.levels[0]]
assert any(abs(a / b - autoretopo.STEP) < 0.02 for a, b in zip(counts, counts[1:])), counts
assert job.report["tris"] == 0
assert autoretopo._folded(job.output["co"], job.output["faces"]) == 0
print("stepped sizes OK", counts, job.report["faces"])
