# ruff: noqa: E402

bl_info = {
    "name": "CurioMesh Quad Remesher",
    "author": "CurioCrafter",
    "version": (0, 5, 0),
    "blender": (4, 3, 0),
    "location": "3D View > N-panel > CurioMesh",
    "category": "Mesh",
    "description": "Atlas-Q scan-aware quad retopology with explicit validation",
    "doc_url": "https://github.com/CurioCrafter/CurioMesh",
}

import bpy

from .operators import (
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_analyze,
    CURIOMESH_OT_compute_metrics,
    CURIOMESH_OT_remesh,
)
from . import batch, live, overlay
from .ui import CURIOMESH_PG_settings, PANEL_CLASSES


classes = (
    batch.CURIOMESH_PG_batch_item,
    CURIOMESH_PG_settings,
    batch.CURIOMESH_OT_batch_remesh,
    *overlay.CLASSES,
    CURIOMESH_OT_analyze,
    CURIOMESH_OT_remesh,
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_compute_metrics,
    *PANEL_CLASSES,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_settings = bpy.props.PointerProperty(type=CURIOMESH_PG_settings)
    overlay.register()
    live.register()


def unregister():
    live.unregister()
    overlay.unregister()
    if hasattr(bpy.types.Scene, "curiomesh_settings"):
        del bpy.types.Scene.curiomesh_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
