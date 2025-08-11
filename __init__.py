bl_info = {
    "name": "CurioMesh Quad Remesher",
    "author": "CurioCrafter",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "3D View > N-panel > CurioMesh",
    "category": "Mesh",
    "description": "Fast field-aligned quad remesher (experimental)",
    "doc_url": "https://github.com/CurioCrafter/AImeshto3Dmesh"
}

import bpy

from .operators import (
    CURIOMESH_OT_remesh,
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_compute_metrics,
    CURIOMESH_OT_preview_cage,
    CURIOMESH_OT_exploded_bake,
)
from .ui import (
    CURIOMESH_PT_panel,
    CURIOMESH_PG_settings,
)


classes = (
    CURIOMESH_PG_settings,
    CURIOMESH_OT_remesh,
    CURIOMESH_OT_apply_preset,
    CURIOMESH_OT_compute_metrics,
    CURIOMESH_OT_preview_cage,
    CURIOMESH_OT_exploded_bake,
    CURIOMESH_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_settings = bpy.props.PointerProperty(type=CURIOMESH_PG_settings)


def unregister():
    del bpy.types.Scene.curiomesh_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()


