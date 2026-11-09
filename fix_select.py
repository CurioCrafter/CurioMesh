# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""Fix Selection: in Edit Mode, find the bad faces inside a rough selection and fix only those, several ways.

Works on any mesh (a result you applied, or one made elsewhere). A Source mesh, when set, says which way is out and
gives the surface to put moved corners back on; without one only the shape checks (folded, crossing, bent, flat...) run.
Every tool is one Undo step and reports the bad faces in the area before and after, then selects what is left.
"""

from __future__ import annotations

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from .triadq import checks, repair
from .translations import tip, tr

KIND_WORDS = {"inward": "inward", "sunk": "sunk", "crossing": "crossing", "folded_over": "folded over", "bent": "bent", "flat": "flat", "duplicate": "duplicate", "overlap": "overlapping", "touching": "touching parts"}


# --------------------------------------------------------------------------- the area and its checks


def _area(bm, grow: int) -> list:
    """Selected faces plus `grow` rings around them; every face when nothing is selected."""
    faces = [f for f in bm.faces if f.select] or list(bm.faces)
    area = set(faces)
    edge = set(faces)
    for _ in range(grow):
        ring = {g for f in edge for v in f.verts for g in v.link_faces} - area
        area |= ring
        edge = ring
    return [f for f in bm.faces if f in area]  # mesh order: the same area gives the same answer


def _source(context, obj):
    """(points, faces) of the Source in this object's space, or None."""
    src = context.scene.curiomesh_fix.source
    if src is None:  # the mesh Auto last worked on
        from . import guided

        src = bpy.data.objects.get(guided.state.get("source") or "")
    if src is None or src.type != "MESH" or src == obj:
        return None
    me = src.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    m = np.asarray(obj.matrix_world.inverted() @ src.matrix_world)
    co = co @ m[:3, :3].T + m[:3, 3]
    return co, [tuple(p.vertices) for p in me.polygons]


def _sub(area):
    """(points, faces, bmesh verts) of the area with its own vertex numbers."""
    verts, index = [], {}
    faces = []
    for f in area:
        loop = []
        for v in f.verts:
            if v not in index:
                index[v] = len(verts)
                verts.append(v)
            loop.append(index[v])
        faces.append(tuple(loop))
    return np.asarray([v.co[:] for v in verts], dtype=np.float64).reshape(-1, 3), faces, verts


def measure(area, source):
    """checks.Defects of the area, and the bad mask (any kind)."""
    points, faces, _verts = _sub(area)
    found = checks.defects(points, faces, *(source or (None, None)))
    bad = np.zeros(len(faces), bool)
    for k in checks.Defects.BROKEN:
        bad |= getattr(found, k)
    return found, bad


def describe(found, bad) -> str:
    if not bad.any():
        return "no bad faces"
    parts = [f"{int(getattr(found, k).sum())} {KIND_WORDS[k]}" for k in checks.Defects.BROKEN if getattr(found, k).any()]
    return f"{int(bad.sum())} bad ({', '.join(parts)})"


def _near(source):
    if source is None:
        return None
    tree = BVHTree.FromPolygons(source[0].tolist(), [list(f) for f in source[1]], epsilon=0.0)

    def near(_part, where):
        hits = [tree.find_nearest(p) for p in np.asarray(where).tolist()]
        return np.asarray([h[0] for h in hits]), np.asarray([h[1] for h in hits])

    return near


def _select_bad(bm, area, bad) -> None:
    for f in bm.faces:
        f.select_set(False)
    for f, b in zip(area, bad.tolist()):
        if b and f.is_valid:
            f.select_set(True)
    bm.select_flush_mode()


# --------------------------------------------------------------------------- the fixes (each changes bm in place)


def fix_flip(bm, area, found, bad, source):
    """Turn faces marked inward around; any still inward afterwards are turned back."""
    turn = [f for f, b in zip(area, found.inward.tolist()) if b]
    bmesh.ops.reverse_faces(bm, faces=turn)
    after, _ = measure(area, source)
    back = [f for f, b, was in zip(area, after.inward.tolist(), found.inward.tolist()) if b and was]
    bmesh.ops.reverse_faces(bm, faces=back)


def fix_untangle(bm, area, found, bad, source):
    """Move the corners of inward and folded faces back onto the source until they face out (the step Auto uses)."""
    near = _near(source)
    if near is None:
        return "needs a Source"
    points, faces, verts = _sub(area)
    marked = found.inward | found.folded_over | found.crossing
    moved, _count = repair.untangle(points, faces, np.zeros(len(faces), np.int64), marked, near)
    for v, p in zip(verts, moved.tolist()):
        v.co = p
    return None


def fix_relax(bm, area, found, bad, source, rounds: int = 4):
    """Smooth the corners of the bad faces (not the area's border), then put them back on the source."""
    inside = set(area)
    border = {v for f in area for e in f.edges if not all(g in inside for g in e.link_faces) for v in e.verts}
    verts = list({v for f, b in zip(area, bad.tolist()) if b for v in f.verts} - border)
    if not verts:
        return None
    for _ in range(rounds):
        bmesh.ops.smooth_vert(bm, verts=verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    near = _near(source)
    if near is not None:
        spots, _normals = near(0, [v.co[:] for v in verts])
        for v, p in zip(verts, spots.tolist()):
            v.co = p
    return None


def fix_clean(bm, area, found, bad, source):
    """Delete duplicate faces and collapse flat ones (no area, or a corner used twice)."""
    dup = [f for f, b in zip(area, found.duplicate.tolist()) if b]
    flat_edges = list({e for f, b in zip(area, found.flat.tolist()) if b for e in f.edges})
    bmesh.ops.delete(bm, geom=dup, context="FACES_ONLY")
    size = np.mean([e.calc_length() for f in area if f.is_valid for e in f.edges] or [1.0])
    bmesh.ops.dissolve_degenerate(bm, edges=[e for e in flat_edges if e.is_valid], dist=1e-4 * size)
    return None


def fix_patch(bm, area, found, bad, source):
    """Delete the bad faces and fill each hole again, in quads where it can, from the corners around it."""
    gone = [f for f, b in zip(area, bad.tolist()) if b]
    if not gone:
        return None
    edges = {e for f in gone for e in f.edges}
    bmesh.ops.delete(bm, geom=gone, context="FACES_ONLY")
    hole = [e for e in edges if e.is_valid and e.is_boundary]
    made = bmesh.ops.holes_fill(bm, edges=hole, sides=0)["faces"]
    if made:
        tris = bmesh.ops.triangulate(bm, faces=made)["faces"]
        bmesh.ops.join_triangles(bm, faces=tris, angle_face_threshold=np.pi, angle_shape_threshold=np.pi)
    loose = [e for e in edges if e.is_valid and not e.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="EDGES")
    return None


FIXES = {
    "CLEAN": (fix_clean, "Clean Up", "Delete duplicate faces and collapse flat ones", "BRUSH_DATA"),
    "FLIP": (fix_flip, "Flip Inward", "Turn inward faces around where that makes them face out (needs a Source)", "NORMALS_FACE"),
    "UNTANGLE": (fix_untangle, "Untangle", "Move corners of folded, inward and crossing faces back onto the source (needs a Source)", "MOD_SMOOTH"),
    "RELAX": (fix_relax, "Relax", "Smooth the bad faces' corners, then put them back on the Source if there is one", "SMOOTHCURVE"),
    "PATCH": (fix_patch, "Patch", "Delete the bad faces and fill the holes again from their border", "MOD_TRIANGULATE"),
}
FIX_ALL = ("CLEAN", "FLIP", "UNTANGLE", "RELAX", "PATCH")  # Patch last: it remakes only what the moves could not fix


# --------------------------------------------------------------------------- operators and panel


def finish(context, obj, bm, area, source, before: str, had=None) -> str:
    """Select the bad faces that are left, remember and return the "Before ... Now ..." line."""
    area = [f for f in area if f.is_valid]
    if had is not None:
        area += [f for f in bm.faces if f not in had]  # faces a patch made
    found, bad = measure(area, source) if area else (None, np.zeros(0, bool))
    _select_bad(bm, area, bad)
    bmesh.update_edit_mesh(obj.data)
    now = describe(found, bad) if found is not None else "no faces left"
    context.scene.curiomesh_fix.last = f"Before: {before}. Now: {now}."
    return context.scene.curiomesh_fix.last


def find_bad(context, obj, grow: int) -> str:
    """Find Bad Faces on `obj` (in Edit Mode): only the bad faces of the area stay selected. Returns the report line."""
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    source = _source(context, obj)
    area = _area(bm, grow)
    found, bad = measure(area, source)
    _select_bad(bm, area, bad)
    bmesh.update_edit_mesh(obj.data)
    context.scene.curiomesh_fix.last = f"In {len(area):,} faces: {describe(found, bad)}."
    return context.scene.curiomesh_fix.last


def run_fix(context, obj, how: str, grow: int) -> tuple[str, str]:
    """Fix the bad faces of the area on `obj` (in Edit Mode) this way. Returns (report line, warning): a warning means
    the fix could not run and nothing changed."""
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    source = _source(context, obj)
    area = _area(bm, grow)
    had = set(bm.faces)
    found, bad = measure(area, source)
    before = describe(found, bad)
    if not bad.any():
        return finish(context, obj, bm, area, source, before), ""
    for key in FIX_ALL if how == "ALL" else (how,):
        area = [f for f in area if f.is_valid]
        found, bad = measure(area, source)
        if not bad.any():
            break
        saved = [v.co.copy() for f in area for v in f.verts]
        note = FIXES[key][0](bm, area, found, bad, source)
        if note and how != "ALL":
            return "", f"{FIXES[key][1]} {note}"
        if key in {"UNTANGLE", "RELAX"} and all(f.is_valid for f in area):  # moves only: undo the move if it made it worse
            _f, worse = measure(area, source)
            if worse.sum() > bad.sum():
                for v, co in zip((v for f in area for v in f.verts), saved):
                    v.co = co
    return finish(context, obj, bm, area, source, before, had), ""


class _FixOp:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH" and context.edit_object is not None and context.edit_object.type == "MESH"


class CURIOMESH_OT_fix_find(_FixOp, bpy.types.Operator):
    bl_idname = "curiomesh.fix_find"
    bl_label = "Find Bad Faces"
    bl_description = "Check the selected faces (and the rings around them) and leave only the bad ones selected. Nothing selected: the whole mesh"

    def execute(self, context):
        self.report({"INFO"}, tr(find_bad(context, context.edit_object, context.scene.curiomesh_fix.grow)))
        return {"FINISHED"}


class CURIOMESH_OT_fix_run(_FixOp, bpy.types.Operator):
    bl_idname = "curiomesh.fix_run"
    bl_label = "Fix"
    bl_description = "Fix only the bad faces in the selected area this way"

    how: bpy.props.EnumProperty(name="Fix", description="Which fix to run on the bad faces", items=[(k, v[1], v[2]) for k, v in FIXES.items()] + [("ALL", "Fix All", "Clean Up, Flip Inward, Untangle and Relax in turn (a move that leaves more bad faces is undone), then Patch what is still bad")])

    @classmethod
    def poll(cls, context):
        from . import overlay

        if context.edit_object is not None and overlay.is_preview(context.edit_object, overlay.GUIDED):  # the preview is rebuilt from the result: a fix there is lost on Apply
            cls.poll_message_set("Apply first, then fix the result. Find Bad Faces works on the preview")
            return False
        return super().poll(context)

    @classmethod
    def description(cls, context, props):
        return tip(FIXES[props.how][2] if props.how in FIXES else "Clean Up, Flip Inward, Untangle and Relax in turn (a move that leaves more bad faces is undone), then Patch what is still bad")

    def execute(self, context):
        text, warning = run_fix(context, context.edit_object, self.how, context.scene.curiomesh_fix.grow)
        if warning:
            self.report({"WARNING"}, tr(warning))
            return {"CANCELLED"}
        self.report({"INFO"}, tr(text))
        return {"FINISHED"}


class CURIOMESH_PG_fix(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(name="Source", type=bpy.types.Object, description="Your original mesh: says which way is out and gives the surface moved corners go back on", poll=lambda self, obj: obj.type == "MESH")
    grow: bpy.props.IntProperty(name="Grow", default=1, min=0, soft_max=20, max=10000, description="Rings of faces around a rough selection that are checked too")
    last: bpy.props.StringProperty()


class CURIOMESH_PT_fix(bpy.types.Panel):
    bl_label = "Fix Selection"
    bl_idname = "CURIOMESH_PT_fix"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_context = "mesh_edit"

    def draw(self, context):
        layout, fs = self.layout, context.scene.curiomesh_fix
        layout.label(text="Select a rough area (or nothing for all)", icon="RESTRICT_SELECT_OFF")
        layout.prop(fs, "source")
        layout.prop(fs, "grow")
        layout.operator("curiomesh.fix_find", icon="VIEWZOOM")
        row = layout.row()
        row.scale_y = 1.4
        row.operator("curiomesh.fix_run", text="Fix All", icon="CHECKMARK").how = "ALL"
        col = layout.column(align=True)
        for key, (_fn, label, _tip, icon) in FIXES.items():
            col.operator("curiomesh.fix_run", text=tr(label), icon=icon).how = key
        if fs.last:
            layout.label(text=tr(fs.last), icon="INFO")


CLASSES = (CURIOMESH_PG_fix, CURIOMESH_OT_fix_find, CURIOMESH_OT_fix_run, CURIOMESH_PT_fix)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_fix = bpy.props.PointerProperty(type=CURIOMESH_PG_fix)


def unregister() -> None:
    if hasattr(bpy.types.Scene, "curiomesh_fix"):
        del bpy.types.Scene.curiomesh_fix
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
