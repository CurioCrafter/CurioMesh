# ruff: noqa: E402

bl_info = {
    "name": "CurioMesh Quad Remesher",
    "author": "CurioCrafter",
    "version": (0, 2, 0),
    "blender": (4, 3, 0),
    "location": "3D View > N-panel > CurioMesh",
    "category": "Mesh",
    "description": "Production-minded QuadriFlow remeshing with material and UV preservation",
    "doc_url": "https://github.com/CurioCrafter/CurioMesh",
}

import bpy

from .operators import (
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_compute_metrics,
    CURIOMESH_OT_remesh,
)
from .ui import CURIOMESH_PG_settings, CURIOMESH_PT_panel


classes = (
    CURIOMESH_PG_settings,
    CURIOMESH_OT_remesh,
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_compute_metrics,
    CURIOMESH_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_settings = bpy.props.PointerProperty(type=CURIOMESH_PG_settings)


def unregister():
    if hasattr(bpy.types.Scene, "curiomesh_settings"):
        del bpy.types.Scene.curiomesh_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
