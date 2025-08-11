from typing import Dict

import bpy
import numpy as np


def quick_metrics(mesh: bpy.types.Mesh) -> Dict[str, float]:
    """Compute a few quick QA metrics in-Blender: face count and extraordinary vertex ratio."""
    face_count = len(mesh.polygons)
    verts_valence = np.zeros(len(mesh.vertices), dtype=np.int32)
    for e in mesh.edges:
        verts_valence[e.vertices[0]] += 1
        verts_valence[e.vertices[1]] += 1
    extraordinary = int(np.sum(verts_valence != 4))
    ratio = float(extraordinary) / float(max(1, len(mesh.vertices)))
    return {
        "faces": float(face_count),
        "extraordinary": float(extraordinary),
        "extraordinary_ratio": ratio,
    }


