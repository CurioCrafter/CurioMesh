# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Guided Retopo: importance points steer a live, adaptive quad preview you can partly redo by hand.

Points (Detail / Loop / Sharp / Calm) live on the source object. A density field built from them drives
either the whole mesh (adaptive triangle remesh -> TRIAD-Q PairClosure -> fit to surface) or only the union
of the point spheres (the region machinery in triadq/patch.py, seam kept). Faces locked on the preview stay
fixed; Redraw Strip rebuilds a selected strip; Next spot points at the worst area.
"""

from __future__ import annotations

import functools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import guides, overlay, region, segments, symmetry, textures
from . import loops as edgeloops
from . import autoretopo, blockout, flowedit, hard_surface, hardware, jobpack, ladder, multires, organize, parallel, profiles, teach
from .triadq import areas, checks, detail, fit, patch, tasks
from .triadq.budget import density_drop
from .triadq.engine import RemeshOptions, remesh_mesh
from .triadq.shading import sharp_edges
from .triadq.types import FlatFaces, MeshData
from .triadq.uvs import chart_codes

PREVIEW_NAME = "CurioMesh_GuidedPreview"
SPARE_MESHES = (PREVIEW_NAME + "_Size", PREVIEW_NAME + "_Organize")  # preview meshes kept for a quick swap, with no users
REGION_READY = False  # Selected Faces Only: off until it re-stitches its seam (226 of 398 outline vertices joined on a bust). Redo Inside Shapes covers it
POINT_TAG = "curiomesh_point"
DEBOUNCE_SECONDS = 0.35
AREA_DEBOUNCE_SECONDS = 0.12
POLL_SECONDS = 0.2
LIVE_QUAD_CAP = 2_500  # predicted quads above which the live preview waits for "Update"
HARD_QUAD_CAP = 60_000
CHORDS_PROFILES = ("ANIMATION", "SUBDIV")  # uses where Redraw Messy Quads starts on
CHORDS_MOST_PIECES = 40  # more loose pieces than this and it stays off: the compressor (250) took 984 s
LIVE_REGION_CAP = 3_000
LIVE_SOURCE_FACES = 120_000
LIVE_SLOW_MS = 4_000.0  # after an update this slow, wait for "Update" instead of freezing Blender on every edit

KIND_ITEMS = [
    ("DETAIL", "Detail", "More, smaller quads around the point (eyes, fingers, a face)", "SNAP_VERTEX", 0),
    ("LOOP", "Loop", "Edge loops ring around the point (eyes, mouth, elbows): the quads flow around it", "MESH_CIRCLE", 1),
    ("SHARP", "Sharp", "Keep hard edges near the point; the quads are smaller and the creases are not rounded off", "SHARPCURVE", 2),
    ("CALM", "Calm", "Fewer, bigger quads near the point (flat or boring areas). Also cuts a hole in the region of Points Only", "SMOOTHCURVE", 3),
]
KIND_KEYS = ("DETAIL", "LOOP", "SHARP", "CALM")
CURVE_KIND_ITEMS = [
    ("GUIDE", "Guide", "A thin line the quads follow, like a panel line", "CURVE_DATA", 0),
    ("LOOP", "Edge Loop", "A loop or flow line drawn by hand: it becomes real continuous edge loops of regular quads, with the poles outside it", "MESH_CIRCLE", 1),
    ("BEND", "Deformation", "A line across a joint or crease: several evenly spaced loops so the mesh bends without pinching", "MOD_SIMPLEDEFORM", 2),
]
SCOPE_ITEMS = [
    ("WHOLE", "Whole Mesh", "Remesh the whole mesh; the points change the quad size across it", "MESH_DATA", 0),
    ("POINTS", "Points Only", "Remesh only inside the point spheres; the rest of the mesh stays exactly as it is", "SELECT_SET", 1),
    ("AREAS", "Work Areas", "Each point is its own area with its own density and fill style. They are remeshed one at a time", "BRUSH_DATA", 2),
    ("AUTO", "Auto Segments", "Splits the mesh into parts, finds the fewest faces that still look the same (checked by camera), then makes quads. Blender stays usable while it works", "MOD_EXPLODE", 3),
]
AIM_ITEMS = [
    ("KEEP", "Keep Detail", "Keep what you can see: the fewest faces that still look like your mesh, within Detail", "ZOOM_ALL", 0),
    ("REDUCE", "Reduce To", "Land on a face count you pick (a percent of your faces, or a number). It then tells you how much the view changed, so you can judge it", "MOD_DECIM", 1),
]
REDUCE_BY_ITEMS = [
    ("PERCENT", "Percent", "A share of the faces your mesh has now", "FIXED_SIZE", 0),
    ("COUNT", "Face Count", "An exact number of faces to aim for", "LINENUMBERS_ON", 1),
]
SHAPE_OP_ITEMS = [
    ("UNION", "Union", "Redo the faces inside this shape, in addition to the shapes above it", "SELECT_EXTEND", 0),
    ("SUBTRACT", "Subtract", "Take the faces inside this shape out of what the shapes above it cover", "SELECT_SUBTRACT", 1),
    ("INTERSECT", "Intersect", "Keep only the faces that the shapes above it and this shape both cover", "SELECT_INTERSECT", 2),
]
SYMMETRY_ITEMS = [
    ("OFF", "Off", "Solve the whole mesh", "BLANK1", 0),
    ("AUTO", "Auto", "Find the mirror plane (X, Y or Z). It is used only when the mesh is symmetric enough", "MOD_MIRROR", 1),
    ("X", "X", "Mirror across the plane facing X", "BLANK1", 2),
    ("Y", "Y", "Mirror across the plane facing Y", "BLANK1", 3),
    ("Z", "Z", "Mirror across the plane facing Z", "BLANK1", 4),
]
APPLY_ITEMS = [
    ("NEW", "New Object", "Write the result to a new object called <name>_Retopo", "DUPLICATE", 0),
    ("REPLACE", "Replace Mesh", "Put the result into your object (Undo restores it)", "FILE_REFRESH", 1),
]
BAKE_ITEMS = [
    ("OFF", "Off", "Keep the UVs and materials of your mesh. No textures are made", "X", 0),
    ("NORMAL", "Normal Map", "Keep your materials and add a baked normal map, so the fine shape still shows", "NORMALS_FACE", 1),
    ("NORMAL_COLOR", "Normal + Color", "Bake a normal map and the base color into one new material with new UVs. Press Update after choosing it: Auto Segments then ignores the old UV seams and needs fewer faces", "IMAGE_DATA", 2),
]
BAKE_SIZES = [(str(s), f"{s} px", f"Textures of {s} by {s} pixels") for s in (512, 1024, 2048, 4096)]
PRESET_ITEMS = [
    ("CHARACTER", "Character", "Organic bodies and faces: Detail points, flowing quads", "ARMATURE_DATA", 0),
    ("HARD_SURFACE", "Hard-Surface Prop", "Props with hard edges: Sharp points, grid-like quads", "MOD_BEVEL", 1),
    ("SCAN", "Scan / Photogrammetry", "Noisy scans: broad strokes, smoothed curvature, flowing quads", "OUTLINER_OB_POINTCLOUD", 2),
    ("GAME_LOD", "Game LOD", "Few quads, spent only where it matters", "MOD_DECIM", 3),
    ("ANIMATION", "Animation", "Bodies and creatures that bend: like Character, and Auto Segments asks for Deformation Lines across the joints", "BONE_DATA", 4),
]
PRESETS = {
    "CHARACTER": dict(base_faces=3000, kind="DETAIL", strategy="FLOW", feature=40.0, radius=0.08, smooth=1),
    "HARD_SURFACE": dict(base_faces=1500, kind="SHARP", strategy="GRID", feature=25.0, radius=0.06, smooth=0),
    "SCAN": dict(base_faces=4000, kind="DETAIL", strategy="FLOW", feature=50.0, radius=0.10, smooth=3),
    "GAME_LOD": dict(base_faces=800, kind="DETAIL", strategy="FLOW", feature=40.0, radius=0.10, smooth=1),
    "ANIMATION": dict(base_faces=3000, kind="LOOP", strategy="FLOW", feature=40.0, radius=0.08, smooth=1),
}

state = {
    "no_ask": False,  # the user ticked "Don't ask again this session" in a long-job dialog
    "source": "",
    "sig": None,
    "changed_at": 0.0,
    "dirty": False,
    "message": "",
    "ms": 0.0,
    "src": None,
    "res": None,
    "lock": None,
    "spot": None,
    "hover": None,
    "draw": None,
    "draw_key": None,
    "batches": None,
    "L0": 0.1,
    "predicted": 0,
    "work": None,
    "auto_cache": (None, {}),  # (mesh signature, measurements of it): Start again with another amount skips the looking
    "fixes": [],  # remembered fixes, when the user's file can't be written
    "fixes_in_memory": False,
    "fixes_saved": 0,
    "found": [],  # (fix, match) pairs of remembered fixes that fit the result, best first
    "lod": None,  # the LOD set being made: {"made": [(name, faces, view change, bake text)]}
    "lod_report": "",
    "applied": "",  # what the last Apply wrote, shown in step 5 until the next search
    "region": None,  # faces of the source (a boolean each) that Selected Faces Only redoes, taken when Start was pressed
    "curve_sig": None,  # what the drawn curves looked like when the search last started: a change restarts it
    "curve_at": 0.0,
    "auto_base": None,  # the finished Auto Segments result that step 4 (Live Boolean) works on
    "auto_inputs": None,  # what the last search was started with, so the size ladder can run it again at other sizes
    "machine": None,  # what this computer is (hardware.detect), read on demand
    "rays": 0.0,  # camera rays per second measured on it
    "parts": None,  # (mesh signature, loose pieces) counted by Check This Computer
}


def status() -> str:
    return state["message"]


# --------------------------------------------------------------------------- points


@dataclass
class Pt:
    kind: str
    co: np.ndarray
    nrm: np.ndarray
    r: float
    s: float
    fill: str = "FLOW"


def _on_point_edit(self, context):
    overlay.tag_redraw()


class CURIOMESH_PG_guided_point(bpy.types.PropertyGroup):
    co: bpy.props.FloatVectorProperty(name="Location", description="Where the point sits on the surface (object space)", size=3, subtype="XYZ", update=_on_point_edit)
    normal: bpy.props.FloatVectorProperty(name="Normal", description="Surface direction at the point", size=3, default=(0.0, 0.0, 1.0))
    kind: bpy.props.EnumProperty(name="Type", description="What this point asks for", items=KIND_ITEMS, default="DETAIL", update=_on_point_edit)
    radius: bpy.props.FloatProperty(name="Radius", description="How far the point reaches (object-space units)", default=0.1, min=1e-4, soft_max=10.0, subtype="DISTANCE", update=_on_point_edit)
    strength: bpy.props.FloatProperty(name="Strength", description="How strongly the point changes the quad size: 1 = about three times smaller (or bigger for Calm) at the centre", default=1.0, min=0.1, max=4.0, update=_on_point_edit)
    fill: bpy.props.EnumProperty(name="Fill Style", description="Pattern of the new quads in this area (Work Areas only)", items=region.STRATEGY_ITEMS, default="FLOW", update=_on_point_edit)
    auto: bpy.props.BoolProperty(name="Suggested", description="Placed by Auto-suggest (replaced when you suggest again)", default=False)


def _preset_update(self, context):
    p = PRESETS[self.preset]
    self.base_faces = p["base_faces"]
    self.strategy = p["strategy"]
    self.new_kind = p["kind"]
    self.feature_angle = p["feature"]
    if self.preset == "ANIMATION":
        self.bend_use = True
    diag = state["src"]["diag"] if state["src"] else 0.0
    if diag > 0:
        self.new_radius = p["radius"] * diag
    overlay.tag_redraw()


_preview_on: set[int] = set()  # scenes whose Guided Preview runs: runtime only, so nothing is saved into the .blend


def _enabled_get(self) -> bool:
    return self.id_data.as_pointer() in _preview_on


def _enabled_set(self, value) -> None:
    (_preview_on.add if value else _preview_on.discard)(self.id_data.as_pointer())


def _enabled_update(self, context):
    if self.enabled:
        _start(context)
    else:
        _stop()


_applying = False


def _profile_update(self, context):
    """Picking a use sets Detail, Method and Output."""
    global _applying
    values = profiles.preset(self.auto_profile)
    if values is not None:
        _applying = True
        try:
            self.auto_detail, self.auto_method, self.auto_output = values
            self.bend_use = self.auto_profile == "ANIMATION"  # Deformation Lines are for joints: on with Animation, off again with any other use
            self.auto_chords = self.auto_profile in CHORDS_PROFILES  # tidiness first: Animation and Subdivision
            hard = self.auto_profile == "HARD"  # Hard Surface: clean panels, even cylinders, hard edges marked on Apply
            self.hs_panels = self.hs_cylinders = hard
            self.hs_edges = "MARK" if hard else "NONE"
        finally:
            _applying = False
    if self.multires_on != (self.auto_profile == "ANIMATION"):  # Animation's route: a QuadriFlow cage, with the detail in Multires levels
        self.multires_on = self.auto_profile == "ANIMATION"
    _redraw_update(self, context)


def _multires_update(self, context):
    """Multires Detail brings the shape back, so the mesh itself can go for clean rows: QuadriFlow, to a low face count."""
    global _applying
    values = profiles.preset(self.auto_profile)
    _applying = True
    try:
        if self.multires_on:
            obj = _source_mesh(context)
            self.auto_method = "QUADRIFLOW"
            self.auto_output = "QUADS"
            self.auto_aim, self.auto_reduce_by = "REDUCE", "COUNT"
            if obj is not None:
                self.auto_reduce_count = profiles.cage_faces(len(obj.data.polygons))
        elif values is not None:
            self.auto_detail, self.auto_method, self.auto_output = values
            self.auto_aim = "KEEP"
    finally:
        _applying = False
    _redraw_update(self, context)


def _custom_update(self, context):
    """Changing Detail, Method or Output by hand leaves the preset."""
    if not _applying and self.auto_profile != "CUSTOM" and not self.multires_on:
        if profiles.preset(self.auto_profile) != (self.auto_detail, self.auto_method, self.auto_output):
            self.auto_profile = "CUSTOM"
    _redraw_update(self, context)


def _redraw_update(self, context):
    overlay.tag_redraw()


# Stored by number in .blend files: the numbers of these items never change.
STYLE_ITEMS = [
    ("DETAIL", "Fewest Faces", "The fewest faces that hold the look you asked for. Quads follow the shape, so some can look slanted", 0),
    ("BALANCED", "Balanced", "The same look with far fewer messy points, for more faces", 1),
    ("ROWS", "Even Rows", "Even, square quads in long rows. The look changes a little more and small sharp detail softens", 2),
]
# The Detail dropdown: one choice over Aim, Detail and the face count (nothing of its own is saved).
DETAIL_STEPS = (("MOST", "Keep Most", 1.5), ("LOT", "Keep a Lot", 3.0), ("BALANCED", "Balanced", 6.0), ("LIGHTER", "Lighter", 10.0), ("LIGHTEST", "Lightest", 20.0))
DETAIL_PICK_ITEMS = [
    *((key, f"{label} ({value:g}%)", f"Up to {value:g}% of the view may change. " + ("More faces, small bumps kept" if value < 5 else "Fewer faces, small bumps soften"), i) for i, (key, label, value) in enumerate(DETAIL_STEPS)),
    ("CUSTOM", "Custom %", "Type how much of the view may change yourself", 5),
    ("PERCENT", "Share of Faces", "Aim for a share of your faces instead of a look; the result then says how much the look changed", 6),
    ("COUNT", "Exact Face Count", "Aim for a number of faces instead of a look; the result then says how much the look changed", 7),
]
CLEANUP_ITEMS = [
    ("QUICK", "Quick", "No extra tidying after the search. Fastest", 0),
    ("NORMAL", "Normal", "Rebuilds the small spots where faces point inward, fold or cross. Adds up to a minute; recommended", 1),
    ("THOROUGH", "Thorough", "Also redraws messy patches as grids and evens out the quads. Tidiest; adds a few minutes", 2),
]


def _detail_get(self) -> int:
    if self.auto_aim == "REDUCE":
        return 7 if self.auto_reduce_by == "COUNT" else 6
    return next((i for i, (_k, _l, value) in enumerate(DETAIL_STEPS) if abs(self.auto_detail - value) < 1e-4), 5)


def _detail_set(self, value: int) -> None:
    """Picking a Detail keeps the Use (it is not Custom) and turns Auto off: Auto would pick the Detail again."""
    global _applying
    _applying = True
    try:
        if value >= 6:
            self.auto_aim, self.auto_reduce_by = "REDUCE", "COUNT" if value == 7 else "PERCENT"
        else:
            self.auto_aim = "KEEP"
            if value < 5:
                self.auto_detail = DETAIL_STEPS[value][2]
            elif any(abs(self.auto_detail - v) < 1e-4 for _k, _l, v in DETAIL_STEPS):
                self.auto_detail += 0.5  # off the steps, so the field to type it in shows
    finally:
        _applying = False
    self.use_auto = False


def _cleanup_get(self) -> int:
    return 2 if self.auto_tidy else 1 if self.auto_redo_rough else 0


def _cleanup_set(self, value: int) -> None:
    """Quick, Normal, Thorough over the three tidy switches (still in Advanced > Engine one by one)."""
    self.auto_redo_rough = value >= 1
    self.auto_tidy = value == 2
    self.auto_chords = value == 2 or self.auto_profile in CHORDS_PROFILES


def _shape_poll(self, obj) -> bool:
    return obj.type in {"EMPTY", "LATTICE", "MESH"} and obj.name != PREVIEW_NAME and obj.name != state["source"]


class CURIOMESH_PG_guided_shape(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(name="Shape", description="A box, a sphere, a lattice or any closed mesh that marks a part to redo", type=bpy.types.Object, poll=_shape_poll)
    operation: bpy.props.EnumProperty(name="Combine", description="How this shape joins the ones above it. They are worked out from top to bottom, so start with a Union", items=SHAPE_OP_ITEMS, default="UNION", update=_redraw_update)


def _guide_poll(self, obj) -> bool:
    return guides.is_guide_object(obj) and obj.name != PREVIEW_NAME and obj.name != state["source"]


class CURIOMESH_PG_guided_guide(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(name="Guide", description="A curve, a Grease Pencil drawing, or a mesh made of edges only. The quads are made to follow it", type=bpy.types.Object, poll=_guide_poll)
    kind: bpy.props.EnumProperty(name="Type", description="What the quads do with this curve", items=CURVE_KIND_ITEMS, default="GUIDE", update=_redraw_update)


class CURIOMESH_PG_guided(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Guided Preview", description="Classic Guided: shows the point cage and keeps a live preview of the new quads beside your mesh", get=_enabled_get, set=_enabled_set, update=_enabled_update)
    preset: bpy.props.EnumProperty(name="Mesh Type", description="Classic Guided: the starting quad size, point type and fill style for this kind of mesh", items=PRESET_ITEMS, default="CHARACTER", update=_preset_update)
    scope: bpy.props.EnumProperty(name="Remesh", description="Which tool rebuilds the mesh. Auto Segments is the main path; the other three are the older point tools in the Advanced tab", items=SCOPE_ITEMS, default="AUTO")
    strategy: bpy.props.EnumProperty(name="Fill Style", description="Pattern of the new quads in a region: Flow follows the surface, Grid is a regular grid", items=region.STRATEGY_ITEMS, default="FLOW")
    base_faces: bpy.props.IntProperty(name="Base Quads", description="Roughly how many quads the whole mesh would get with no points. The points make some areas denser or coarser than this", default=3000, min=50, soft_max=60000)
    new_kind: bpy.props.EnumProperty(name="New Point Type", description="Type of the next point you place (keys 1-4 while placing)", items=KIND_ITEMS, default="DETAIL", update=_redraw_update)
    new_radius: bpy.props.FloatProperty(name="New Radius", description="Reach of the next point (scroll while placing). 0 picks a size from the mesh", default=0.0, min=0.0, soft_max=10.0, subtype="DISTANCE")
    new_strength: bpy.props.FloatProperty(name="New Strength", description="Strength of the next point (Shift + scroll while placing)", default=1.0, min=0.1, max=4.0)
    auto_update: bpy.props.BoolProperty(name="Auto-Update", description="Rebuild the preview shortly after you change a point or a setting", default=True)
    auto_profile: bpy.props.EnumProperty(name="Use", description="What you will do with the result. Each choice sets the Detail, layout and kind of faces for you", items=profiles.PROFILE_ITEMS, default="GAME", update=_profile_update)
    multires_on: bpy.props.BoolProperty(name="Multires Detail", description="Makes a clean low mesh and, on Apply, Multires levels that bring your shape back. Animate or export level 0; your mesh is only read", default=False, update=_multires_update)
    multires_levels: bpy.props.IntProperty(name="Levels", description="How many Multires levels to add. Each level has four times the faces of the one below it", default=2, min=1, max=5, update=_redraw_update)
    auto_why: bpy.props.StringProperty(name="Why", description="What Auto picked for this mesh and why", default="")
    use_auto: bpy.props.BoolProperty(name="Auto", description="Start looks at your mesh and picks the Use for it, then says what it picked and why. Pick another tile to choose yourself", default=False, update=_redraw_update)
    detail_pick: bpy.props.EnumProperty(name="Detail", description="How close to your mesh the result stays. More detail keeps more faces", items=DETAIL_PICK_ITEMS, get=_detail_get, set=_detail_set, update=_redraw_update)
    cleanup: bpy.props.EnumProperty(name="Cleanup", description="How much tidying runs after the search. More is tidier and takes longer", items=CLEANUP_ITEMS, get=_cleanup_get, set=_cleanup_set, update=_redraw_update)
    ui_setup_open: bpy.props.BoolProperty(name="Settings", description="The settings the result was made with, to change them and start again", default=False)
    auto_detail: bpy.props.FloatProperty(name="Detail", description="How close to your mesh the result must look, as the share of the view that may change. Lower keeps more faces; higher gives a rougher, lighter mesh", default=6.0, min=0.1, max=80.0, soft_max=50.0, precision=1, subtype="PERCENTAGE", update=_custom_update)
    auto_effort: bpy.props.EnumProperty(name="Effort", description="How hard to work this computer. Light keeps Blender smooth while it works; Full finishes soonest; Auto picks from your cores and free memory", items=hardware.EFFORT_ITEMS, default="AUTO", update=_redraw_update)
    auto_style: bpy.props.EnumProperty(name="Style", description="How the quads are laid out. Fewest Faces follows the shape; Even Rows makes tidy rows of square quads but changes the look a little more", items=STYLE_ITEMS, default="DETAIL", update=_redraw_update)
    auto_method: bpy.props.EnumProperty(name="Method", description="How each part's faces are made. Auto suits nearly every mesh; you can give each part its own method after the search", items=profiles.METHOD_ITEMS, default="AUTO", update=_custom_update)
    auto_output: bpy.props.EnumProperty(name="Output", description="Which kind of faces the result has: all triangles, all quads, or quads with the leftover triangles kept", items=profiles.OUTPUT_ITEMS, default="QUADS", update=_custom_update)
    auto_aim: bpy.props.EnumProperty(name="Aim", description="Keep Detail aims for a look; Reduce To aims for a face count and then tells you how much the look changed", items=AIM_ITEMS, default="KEEP", update=_redraw_update)
    auto_reduce_by: bpy.props.EnumProperty(name="Reduce By", description="Give the face count as a share of your faces or as a number", items=REDUCE_BY_ITEMS, default="PERCENT", update=_redraw_update)
    auto_reduce_percent: bpy.props.FloatProperty(name="Faces Left", description="How many faces to end up with, as a share of your mesh's faces now. 25% turns 40,000 faces into about 10,000", default=25.0, min=0.1, max=100.0, precision=1, subtype="PERCENTAGE", update=_redraw_update)
    auto_reduce_count: bpy.props.IntProperty(name="Faces", description="How many faces to end up with. The result lands within about 10% of it unless the parts can't go that low", default=5000, min=12, soft_max=500000, update=_redraw_update)
    seg_colors: bpy.props.BoolProperty(name="Part Colors", description="Tints each part of the result in its own colour, so you can see where the parts are", default=True, update=segments.on_view)
    seg_heat: bpy.props.BoolProperty(name="Where It Changed", description="Colours each part by how much it changed: purple not at all, yellow the most. Readable with colour blindness", default=False, update=segments.on_view)
    seg_bad: bpy.props.BoolProperty(name="Bad Faces", description="Paints in red the faces that point inward, sink into your mesh, cross or fold over others", default=True, update=segments.on_view)
    seg_wire: bpy.props.BoolProperty(name="Wireframe", description="Shows the edges of the result", default=True, update=segments.on_view)
    seg_compare: bpy.props.EnumProperty(name="Original", description="Where your own mesh is shown: next to the result, or under it as a white wireframe", items=[("BESIDE", "Beside", "The result sits next to your mesh"), ("OVER", "Over", "The result sits on your mesh, which is drawn over it as a white wireframe")], default="BESIDE", update=segments.on_view)
    seg_goal: bpy.props.EnumProperty(name="Use", description="What the selected parts are for when you press Redo", items=segments.segment_goals.GOAL_ITEMS, default="ANIMATION")
    seg_goal_detail: bpy.props.FloatProperty(name="Detail", description="How many faces the selected parts get when you press Redo. The middle keeps the quad size they have now", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    seg_method: bpy.props.EnumProperty(name="Method", description="How the selected parts' quads are made when their Use is Reduce or Low Poly", items=segments.METHOD_ITEMS, default="AUTO")
    seg_parts: bpy.props.CollectionProperty(type=segments.CURIOMESH_PG_segpart)
    seg_index: bpy.props.IntProperty(default=-1, update=segments.on_index)
    sym_axis: bpy.props.EnumProperty(name="Mirror", description="Builds one half and mirrors it, so both sides match exactly. Used only when the mesh really is symmetric, unless Symmetrize is on", items=SYMMETRY_ITEMS, default="OFF", update=_redraw_update)
    sym_force: bpy.props.BoolProperty(name="Symmetrize", description="Mirrors even when the two sides differ, so the result is exactly symmetric. Differences on the other side are lost", default=False, update=_redraw_update)
    sym_side: bpy.props.EnumProperty(name="Keep Side", description="Which half is built and then mirrored", items=(("AUTO", "Auto", "The side with more surface"), ("POS", "+", "The side with larger coordinates"), ("NEG", "-", "The side with smaller coordinates")), default="AUTO", update=_redraw_update)
    sym_offset_use: bpy.props.BoolProperty(name="Set Plane", description="Puts the mirror plane where you say, in object space, instead of finding it", default=False, update=_redraw_update)
    sym_offset: bpy.props.FloatProperty(name="Plane Offset", description="Where the mirror plane crosses the axis, in object space", default=0.0, subtype="DISTANCE", update=_redraw_update)
    guide_use: bpy.props.BoolProperty(name="Use Guide Curves", description="Makes the quads follow the curves in the list, such as a line along a panel. The curves are laid onto the mesh", default=False, update=_redraw_update)
    guide_curves: bpy.props.CollectionProperty(type=CURIOMESH_PG_guided_guide)
    guide_index: bpy.props.IntProperty(default=0)
    region_use: bpy.props.BoolProperty(name="Selected Faces Only", description="Redoes only the faces selected in Edit Mode and leaves the rest exactly as it is", default=False, update=_redraw_update)
    loop_use: bpy.props.BoolProperty(name="Edge Loops", description="Lines you draw (round an eye or the mouth) become real edge loops with even rows of quads beside them", default=False, update=_redraw_update)
    loop_count: bpy.props.IntProperty(name="Loops", description="How many parallel loops each drawn line makes: the line itself plus one more on each side is 3", default=3, min=3, max=9, update=_redraw_update)
    bend_use: bpy.props.BoolProperty(name="Deformation Lines", description="Lines drawn across a joint (elbow, knee, mouth corner) become evenly spaced loops, so the mesh bends without pinching", default=False, update=_redraw_update)
    bend_loops: bpy.props.IntProperty(name="Loops", description="How many parallel loops each joint line makes. 3 is the least that bends well; use more across a big joint", default=3, min=3, max=9, update=_redraw_update)
    loop_gap: bpy.props.FloatProperty(name="Loop Gap", description="The distance between two loops of a drawn line, in object units. 0 picks about one quad", default=0.0, min=0.0, soft_max=10.0, subtype="DISTANCE", update=_redraw_update)
    bend_crease: bpy.props.BoolProperty(name="Crease", description="On Apply, marks the middle loop of each joint line with an edge crease, for Subdivision Surface", default=False)
    bend_sharp: bpy.props.BoolProperty(name="Sharp", description="On Apply, marks the middle loop of each joint line as sharp edges", default=False)
    density_use: bpy.props.BoolProperty(name="Detail Markers", description="Keeps more faces where you marked More Detail and fewer where you marked Less Detail. Paint the marks or place marker balls", default=False, update=_redraw_update)
    density_group: bpy.props.StringProperty(name="Group", description="The vertex group of the mesh that holds the More Detail paint", default="CM_Density")
    less_group: bpy.props.StringProperty(name="Less Group", description="The vertex group of the mesh that holds the Less Detail paint", default="CM_LessDetail")
    density_strength: bpy.props.FloatProperty(name="Strength", description="How strongly the marks move faces. 1 keeps about four times as many faces under More Detail as under Less Detail", default=1.0, min=0.25, max=4.0)
    bool_on: bpy.props.BoolProperty(name="Redo Inside Shapes", description="Redoes the result inside the shapes, live; everything outside stays as it is. Move or resize a shape and the quads follow", default=False, update=_redraw_update)
    bool_shapes: bpy.props.CollectionProperty(type=CURIOMESH_PG_guided_shape)
    bool_index: bpy.props.IntProperty(default=0)
    quad_size: bpy.props.FloatProperty(name="Quad Size", description="Left: smaller quads that keep more detail. Right: bigger quads and fewer faces. Each stop is a full result, worked out in short steps while you keep working", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=ladder.on_slider)
    quad_order: bpy.props.FloatProperty(name="Rows", description="Left: fewest faces, quads follow the shape. Right: even rows of square quads, with more faces and a little softer detail", default=0.0, min=0.0, max=1.0, subtype="FACTOR", update=ladder.on_slider)
    organize: bpy.props.FloatProperty(name="Tidy Up", description="Lays out the same faces more tidily: fewer messy spots, straighter rows. Costs a little detail; the line below shows how much", default=0.0, min=0.0, max=1.0, subtype="FACTOR", update=organize.on_slider)
    organize_cap_use: bpy.props.BoolProperty(name="Most Detail Loss", description="Stops Tidy Up before the look changes by more than the amount next to it", default=False, update=organize.on_cap)
    organize_cap: bpy.props.FloatProperty(name="Most Detail Loss", description="How much more of the look Tidy Up may change, in percent, on top of the result as it was", default=1.0, min=0.1, max=20.0, precision=1, subtype="NONE", update=organize.on_cap)
    # Hard surface (hard_surface.py): two search switches and the hard-edge finish on Apply
    hs_panels: hard_surface.prop("hs_panels")
    hs_cylinders: hard_surface.prop("hs_cylinders")
    hs_edges: hard_surface.prop("hs_edges")
    hs_angle: hard_surface.prop("hs_angle")
    hs_hold_unit: hard_surface.prop("hs_hold_unit")
    hs_hold_percent: hard_surface.prop("hs_hold_percent")
    hs_hold_mm: hard_surface.prop("hs_hold_mm")
    organize_flow: bpy.props.BoolProperty(name="Loop Flow", description="Draws each edge loop of the result in its own colour, and its poles (points where 3 or 5+ edges meet) as dots", default=False, update=_redraw_update)
    flow_live: bpy.props.BoolProperty(name="Live", description="Flow Editor: redo the quads around each stroke as soon as you draw, move or erase it. Off: strokes are only kept for the next Start", default=True)
    flow_size: bpy.props.FloatProperty(name="Shape Size", description="Flow Editor: how wide a clicked eye ring, mouth ring or nose-to-mouth shape is. 0 picks a size from your mesh", default=0.0, min=0.0, soft_max=10.0, subtype="DISTANCE")
    flow_pair: bpy.props.BoolProperty(name="Both Sides", description="Flow Editor: a clicked eye ring also goes on the other side, mirrored across the middle of your mesh (always on when the result is mirrored)", default=True)
    bool_detail: bpy.props.FloatProperty(name="Organized / Detailed", description="Inside the shapes: left makes fewer, bigger, tidier quads; the middle keeps the quad size; right makes smaller quads with more detail", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_redraw_update)
    show_cage: bpy.props.BoolProperty(name="Show Cage", description="Classic Guided: draws each point's reach and the outline of the area it covers", default=True, update=_redraw_update)
    blockout_sides: bpy.props.IntProperty(name="Primitive Sides", description="How many sides each cylinder and ball has (small parts get half). More sides look rounder and cost faces", default=12, min=4, max=32, step=2)
    blockout_separate: bpy.props.BoolProperty(name="Each Part Its Own Object", description="On Apply, writes one object per loose part of your mesh instead of one merged mesh", default=False)
    auto_instances: bpy.props.BoolProperty(name="Keep Copies as Instances", description="Repeated parts become linked copies that cost no extra geometry; the copies share UVs. Off makes one merged mesh", default=False)
    apply_mode: bpy.props.EnumProperty(name="Result", description="Where Apply writes the result: a new object, or into your own object (Undo brings yours back)", items=APPLY_ITEMS, default="NEW")
    auto_rough: bpy.props.BoolProperty(name="Rough As Texture", description="For rough scans (rock, ground): judges bumps on average. Far fewer faces; bumps smaller than a few pixels are lost", default=False)
    auto_parallel: bpy.props.BoolProperty(name="Use Several Cores", description="Works on several parts at once in background copies of Blender. Faster on big meshes; uses more memory. Needs Background Blender Processes on in the add-on preferences", default=True)
    auto_drop_hidden: bpy.props.BoolProperty(name="Leave Out Hidden Pieces", description="Leaves out loose pieces that no camera outside the model can see. The result says how many were left out", default=False)
    auto_baked: bpy.props.BoolProperty(name="Judge After Baking (Experimental)", description="Experimental, Low Poly on scans you will bake: judges the look after baking, so far fewer faces remain. Can leave thin slivers; check the result", default=False)
    auto_chords: bpy.props.BoolProperty(name="Redraw Messy Quads", description="Redraws messy patches as plain grids of quads. Much tidier loops; adds a minute or two, and can stall on models of many loose pieces", default=False)
    auto_tidy: bpy.props.BoolProperty(name="Tidy Quads", description="Evens out the quads after the search, closer to hand-made. Slow, and the shape changes a little more", default=False)
    auto_redo_rough: bpy.props.BoolProperty(name="Redo Rough Spots", description="Rebuilds the small spots where faces point inward, fold or cross. Adds up to a minute; recommended", default=False)
    keep_uvs: bpy.props.BoolProperty(name="Keep UV Seams", description="Keeps quads inside each piece of your UV map so textures stay in place. Can double the faces; for game assets, baking is usually better", default=False)
    lod0_percent: bpy.props.FloatProperty(name="LOD0", description="Faces of the most detailed LOD, as a share of your mesh's faces", default=profiles.LOD_PERCENTS[0], min=0.1, max=100.0, precision=1, subtype="PERCENTAGE")
    lod1_percent: bpy.props.FloatProperty(name="LOD1", description="Faces of the middle LOD, as a share of your mesh's faces", default=profiles.LOD_PERCENTS[1], min=0.1, max=100.0, precision=1, subtype="PERCENTAGE")
    lod2_percent: bpy.props.FloatProperty(name="LOD2", description="Faces of the lightest LOD, as a share of your mesh's faces", default=profiles.LOD_PERCENTS[2], min=0.1, max=100.0, precision=1, subtype="PERCENTAGE")
    bake_detail: bpy.props.EnumProperty(name="Bake Detail", description="On Apply, bakes your mesh's fine detail into textures on the result. Can take minutes", items=BAKE_ITEMS, default="OFF")
    bake_size: bpy.props.EnumProperty(name="Bake Size", description="Width and height of the baked textures. Bigger is sharper and slower", items=BAKE_SIZES, default="2048")
    bake_sets: bpy.props.IntProperty(name="Texture Sets", description="Normal + Color: bakes onto this many sets of maps, each its own material, so a big model keeps detail", default=1, min=1, max=16)
    fit_iterations: bpy.props.IntProperty(name="Surface Fit Passes", description="Classic Guided: how many passes settle the new quads onto your surface. More follows it closer, slower", default=2, min=0, max=30)
    tweak_rows: bpy.props.FloatProperty(name="Rows on Round Areas", description="How readily round, smooth areas become rows of quads that follow their curve. Right finds more of them and adds faces. Reruns the search", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_redraw_update)
    tweak_flat: bpy.props.FloatProperty(name="Flat Area Size", description="How big quads inside flat areas may get. Left makes them smaller and more even; right fewer and longer. Reruns the search", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=_redraw_update)
    tweak_poles: bpy.props.BoolProperty(name="Avoid Poles", description="Turns edges so rows of quads keep going past poles (points where 3 or 5+ edges meet). Off: about a fifth more poles, a touch more detail", default=True, update=_redraw_update)
    feature_angle: bpy.props.FloatProperty(name="Hard Edge Angle", description="Classic Guided: edges sharper than this angle count as hard edges and are kept", default=40.0, min=5.0, max=170.0)
    pure_quads: bpy.props.BoolProperty(name="Pure Quads", description="Classic Guided, Whole Mesh: makes every face a quad. Off is faster on big meshes and leaves a few triangles", default=True)
    suggest_count: bpy.props.IntProperty(name="Suggestions", description="Classic Guided: how many points Auto-suggest places", default=8, min=1, max=40)


# --------------------------------------------------------------------------- density maths


def _fac(kind: str, s: float) -> float:
    """Edge-length multiplier at the centre of a point."""
    if kind == "DETAIL":
        return 1.0 / (1.0 + 2.0 * s)
    if kind == "CALM":
        return 1.0 + 2.0 * s
    if kind == "SHARP":
        return 1.0 / (1.0 + s)
    return float(np.sqrt(1.0 / (1.0 + 2.0 * s) / (1.0 + 0.5 * s)))  # LOOP, isotropic mean


def _weight(P: np.ndarray, p: Pt) -> np.ndarray:
    t = np.clip(np.linalg.norm(P - p.co, axis=1) / max(p.r, 1e-9), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)  # 1 at the centre, 0 at the radius (smoothstep)


def multiplier(P: np.ndarray, pts: list[Pt], E: np.ndarray | None = None) -> np.ndarray:
    """Target edge length / base edge length at points P. E: unit edge directions (Loop points go anisotropic)."""
    f = np.ones(len(P))
    for p in pts:
        w = _weight(P, p)
        if p.kind == "LOOP" and E is not None:
            fr = 1.0 + w * (1.0 / (1.0 + 2.0 * p.s) - 1.0)  # across the ring: small steps = many rings
            ft = 1.0 + w * (1.0 / (1.0 + 0.5 * p.s) - 1.0)  # along the ring
            d = P - p.co
            d = d - np.outer(d @ p.nrm, p.nrm)
            d /= np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
            c2 = np.einsum("ij,ij->i", d, E) ** 2
            f *= 1.0 / np.sqrt(c2 / fr**2 + (1.0 - c2) / ft**2)
        else:
            f *= 1.0 + w * (_fac(p.kind, p.s) - 1.0)
    return np.clip(f, 0.125, 8.0)


def region_sd(P: np.ndarray, pts: list[Pt], scope: str) -> np.ndarray:
    """Signed distance of the point-sphere region: union of adding spheres minus the subtracting ones (<0 inside)."""
    add = {"POINTS": ("DETAIL", "LOOP"), "AREAS": KIND_KEYS}.get(scope, ("DETAIL", "LOOP", "SHARP"))
    sub = {"POINTS": ("CALM", "SHARP"), "AREAS": ()}.get(scope, ("CALM",))

    def union(kinds):
        sd = np.full(len(P), np.inf)
        for p in pts:
            if p.kind in kinds:
                sd = np.minimum(sd, np.linalg.norm(P - p.co, axis=1) - p.r)
        return sd

    return np.maximum(union(add), -union(sub))


def stepped_intensity(kind: str, strength: float, up: bool, step: float = 0.25) -> tuple[str, float]:
    """One scroll notch: up is always denser. Detail fades into Calm and back through the base density."""
    was = -strength if kind == "CALM" else strength
    now = was + (step if up else -step)
    if abs(now) < 0.1 or now * was < 0:
        now = 0.1 if up else -0.1
    now = float(np.clip(now, -4.0, 4.0))
    return "CALM" if now < 0 else ("DETAIL" if kind == "CALM" else kind), round(abs(now), 3)


# --------------------------------------------------------------------------- mesh helpers


def _edge_table(tris: np.ndarray):
    """Unique edges of a triangle array: (lo, hi, face0, face1 or -1, count)."""
    n = int(tris.max()) + 1 if len(tris) else 1
    a = tris.ravel()
    b = tris[:, [1, 2, 0]].ravel()
    f = np.repeat(np.arange(len(tris)), 3)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * n + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    count = np.diff(np.r_[start, len(ks)])
    f1 = np.where(count > 1, f[order][np.minimum(start + 1, len(ks) - 1)], -1)
    return lo[order][start], hi[order][start], f[order][start], f1, count


def _tri_normals(co: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]


def _vert_normals(co: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])  # area weighted
    out = np.zeros_like(co)
    for k in range(3):
        for axis in range(3):
            out[:, axis] += np.bincount(tris[:, k], n[:, axis], len(co))
    return out / np.maximum(np.linalg.norm(out, axis=1), 1e-30)[:, None]


def _creases(co: np.ndarray, tris: np.ndarray, degrees: float):
    """(edge table, crease edge mask, crease vertex mask, boundary vertex mask)."""
    lo, hi, f0, f1, count = _edge_table(tris)
    nrm = _tri_normals(co, tris)
    inner = f1 >= 0
    cos = np.ones(len(lo))
    cos[inner] = np.einsum("ij,ij->i", nrm[f0[inner]], nrm[f1[inner]])
    crease = inner & (cos < np.cos(np.radians(degrees)))
    cv = np.zeros(len(co), dtype=bool)
    cv[lo[crease]] = cv[hi[crease]] = True
    bv = np.zeros(len(co), dtype=bool)
    bv[lo[~inner]] = bv[hi[~inner]] = True
    return (lo, hi, f0, f1, count), crease, cv, bv


def _fan(faces) -> np.ndarray:
    return np.asarray([(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)], dtype=np.int64).reshape(-1, 3)


def _source_data(obj) -> dict:
    """Welded copy of the source (glTF/FBX imports split vertices at seams), cached per geometry."""
    mesh = obj.data
    nv = len(mesh.vertices)
    co = np.empty(nv * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    sig = (obj.name, nv, len(mesh.polygons), round(float(co.sum()), 5), mesh.as_pointer())
    cached = state["src"]
    if cached is not None and cached["sig"] == sig:
        return cached
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", sizes)
    corners = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    mat = np.zeros(len(starts), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", mat)
    diag = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0))) or 1.0
    key = np.round(co / (diag * 1e-6)).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    wco = co[first]
    if len(sizes) and (sizes == 3).all():  # a scan: no per-face Python, and the faces stay as arrays
        rows = inv[corners[starts[:, None] + np.arange(3)]]
        keep = np.flatnonzero((rows[:, 0] != rows[:, 1]) & (rows[:, 1] != rows[:, 2]) & (rows[:, 0] != rows[:, 2]))
        edges = np.arange(0, 3 * len(keep) + 1, 3)
        wfaces = FlatFaces(edges, rows[keep].ravel())
        wloops = FlatFaces(edges, (starts[keep, None] + np.arange(3)).ravel())
        tris = wfaces.flat.reshape(-1, 3)
        tri_loops = wloops.flat.reshape(-1, 3)
    else:
        keep, wfaces, wloops = [], [], []
        for i, (start, size) in enumerate(zip(starts.tolist(), sizes.tolist())):
            first: dict[int, int] = {}
            for corner, v in enumerate(corners[start : start + size].tolist()):
                first.setdefault(int(inv[v]), start + corner)
            if len(first) >= 3:
                wfaces.append(tuple(first))
                wloops.append(tuple(first.values()))
                keep.append(i)
        keep = np.asarray(keep, dtype=np.int64)
        tris = _fan(wfaces)
        tri_loops = _fan(wloops)
    area = 0.5 * np.linalg.norm(np.cross(wco[tris[:, 1]] - wco[tris[:, 0]], wco[tris[:, 2]] - wco[tris[:, 0]]), axis=1)
    data = {
        "sig": sig,
        "diag": diag,
        "co": wco,
        "faces": wfaces,
        "tris": tris,
        "tri_area": area,
        "tri_cent": wco[tris].mean(axis=1),
        "tri_face": np.repeat(np.arange(len(wfaces)), np.diff(FlatFaces.of(wfaces).indptr) - 2),
        "tri_loops": tri_loops,
        "loops": wloops,
        "mat": mat[keep],
        "bvh": None,
        "weld": inv,
        "keep": keep,
    }
    state["src"] = data
    return data


def _bvh(src: dict) -> BVHTree:
    if src["bvh"] is None:
        src["bvh"] = BVHTree.FromPolygons(src["co"].tolist(), src["tris"].tolist())
    return src["bvh"]


def _centroids(co: np.ndarray, faces) -> np.ndarray:
    return np.asarray([co[list(f)].mean(axis=0) for f in faces]).reshape(-1, 3)


def _nearest_tris(bvh: BVHTree, points: np.ndarray) -> np.ndarray:
    return np.asarray([bvh.find_nearest(p)[2] for p in points.tolist()], dtype=np.int64)


def _barycentric(co: np.ndarray, tris: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of each point's projection on its own triangle, kept inside the triangle."""
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    v0, v1, v2 = b - a, c - a, points - a
    d00, d01, d11 = (v0 * v0).sum(axis=1), (v0 * v1).sum(axis=1), (v1 * v1).sum(axis=1)
    d20, d21 = (v2 * v0).sum(axis=1), (v2 * v1).sum(axis=1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-30, np.inf, denom)  # degenerate triangle: use its first corner
    v, w = (d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom
    bary = np.clip(np.stack((1.0 - v - w, v, w), axis=1), 0.0, 1.0)
    return bary / bary.sum(axis=1, keepdims=True)


def _uv_islands(src: dict, uv: np.ndarray) -> np.ndarray:
    """UV island of each source face: faces that share an edge with the same UVs on both sides belong together."""
    rows, owner = [], []
    for i, (face, loops) in enumerate(zip(src["faces"], src["loops"])):
        for k in range(len(face)):
            n = (k + 1) % len(face)
            a, b = (k, n) if face[k] < face[n] else (n, k)
            rows.append((face[a], face[b], *uv[loops[a]], *uv[loops[b]]))
            owner.append(i)
    rows = np.asarray(rows)
    rows[:, 2:] = np.round(rows[:, 2:], 5)
    _u, inv = np.unique(rows, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    owner = np.asarray(owner, dtype=np.int64)
    order = np.argsort(inv, kind="stable")
    twin = inv[order][1:] == inv[order][:-1]
    return patch._labels(len(src["faces"]), owner[order][:-1][twin], owner[order][1:][twin])


def _transfer_uvs(mesh, source, src, res) -> None:
    """Copy every UV layer: each corner reads the source at its nearest point on the UV island of its own face."""
    co, faces = res["co"], res["faces"]
    known = res.get("chart")  # Auto Segments remembers each face's island in the active layer
    if known is not None and len(known) != len(faces):
        known = None
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64)
    bvh = _bvh(src)
    corner_tri = _nearest_tris(bvh, co)[flat]
    owner_tri = np.repeat(_nearest_tris(bvh, _centroids(co, faces)), [len(f) for f in faces])
    points = co[flat]
    for layer in source.data.uv_layers:
        uv = np.empty(len(source.data.loops) * 2, dtype=np.float64)
        layer.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        island = _uv_islands(src, uv)[src["tri_face"]]
        tri = corner_tri.copy()
        owner = island[owner_tri]
        if known is not None and layer.active:
            owner = np.repeat(known, [len(f) for f in faces])
        off = np.flatnonzero(island[corner_tri] != owner)  # a corner on a UV seam: stay on this face's side
        for chart in np.unique(owner[off]).tolist():
            members = np.flatnonzero(island == chart)
            used, local = np.unique(src["tris"][members], return_inverse=True)
            chart_bvh = BVHTree.FromPolygons(src["co"][used].tolist(), local.reshape(-1, 3).tolist())
            here = off[owner[off] == chart]
            tri[here] = members[_nearest_tris(chart_bvh, points[here])]
        out = np.einsum("ijk,ij->ik", uv[src["tri_loops"][tri]], _barycentric(src["co"], src["tris"][tri], points))
        target = mesh.uv_layers.new(name=layer.name)
        target.data.foreach_set("uv", out.astype(np.float32).ravel())
    active = source.data.uv_layers.active_index
    if 0 <= active < len(mesh.uv_layers):
        mesh.uv_layers.active_index = active


def _transfer_normals(mesh, src, res) -> None:
    """Shade the result like the source: each face corner gets its normal from the source samples around it."""
    points, faces = res["co"], res["faces"]
    keep = src["tri_area"] > 0.0
    tree = KDTree(int(keep.sum()))
    for index, point in enumerate(src["tri_cent"][keep].tolist()):
        tree.insert(point, index)
    tree.balance()
    normals, area = _tri_normals(src["co"], src["tris"])[keep], src["tri_area"][keep]
    fa, fb = patch.face_adjacency(src["faces"])
    sample_part = np.unique(patch._labels(len(src["faces"]), fa, fb), return_inverse=True)[1][src["tri_face"]][keep]
    part, source_of = fit.vertex_parts(points, faces, sample_part, lambda at: np.fromiter((tree.find(p)[1] for p in at.tolist()), dtype=np.int64, count=len(at)))
    out = fit.corner_normals(points, faces, normals, area, sample_part, source_of[part], tree.find_n)
    mesh.normals_split_custom_set(out.tolist())


# --------------------------------------------------------------------------- whole-mesh path


def _read(me):
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    tris = np.empty(len(me.polygons) * 3, dtype=np.int64)
    me.polygons.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    return co.reshape(-1, 3), tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]


def _write_co(me, co):
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()


def _bm_edges(bm, lo, hi, idx):
    bm.verts.ensure_lookup_table()
    return [e for e in (bm.edges.get((bm.verts[int(lo[i])], bm.verts[int(hi[i])])) for i in idx) if e is not None]


def _targets(co, lo, hi, pts, L0):
    d = co[hi] - co[lo]
    length = np.linalg.norm(d, axis=1)
    return length, L0 * multiplier(0.5 * (co[lo] + co[hi]), pts, d / np.maximum(length, 1e-30)[:, None])


def _split_pass(me, pts, L0) -> int:
    co, tris = _read(me)
    lo, hi, _f0, _f1, _c = _edge_table(tris)
    length, target = _targets(co, lo, hi, pts, L0)
    idx = np.flatnonzero(length > 4.0 / 3.0 * target)
    if not len(idx):
        return 0
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        edges = _bm_edges(bm, lo, hi, idx)
        bmesh.ops.subdivide_edges(bm, edges=edges, cuts=1, use_grid_fill=False)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="SHORT_EDGE")
        bm.to_mesh(me)
    finally:
        bm.free()
    return len(idx)


def _neighbours(count, lo, hi):
    a = np.concatenate((lo, hi))
    b = np.concatenate((hi, lo))
    order = np.argsort(a, kind="stable")
    start = np.searchsorted(a[order], np.arange(count + 1))
    return start, b[order]


def _collapse_pass(me, pts, L0, crease_deg) -> int:
    co, tris = _read(me)
    (lo, hi, _f0, _f1, count), crease, cv, bv = _creases(co, tris, crease_deg)
    length, target = _targets(co, lo, hi, pts, L0)
    ok = (length < 0.8 * target) & (count == 2) & ~bv[lo] & ~bv[hi] & (~(cv[lo] | cv[hi]) | (crease & cv[lo] & cv[hi]))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return 0
    idx = idx[np.argsort(length[idx])]
    start, nbr = _neighbours(len(co), lo, hi)
    locked = np.zeros(len(co), dtype=bool)
    chosen = []
    for i in idx.tolist():
        a, b = int(lo[i]), int(hi[i])
        if locked[a] or locked[b]:
            continue
        na, nb = nbr[start[a] : start[a + 1]], nbr[start[b] : start[b + 1]]
        if len(set(na.tolist()) & set(nb.tolist())) != 2 or len(na) + len(nb) - 4 < 4:
            continue
        ring = np.union1d(na, nb)
        mid = 0.5 * (co[a] + co[b])
        far = np.linalg.norm(co[ring[(ring != a) & (ring != b)]] - mid, axis=1).max()
        if far > 4.0 / 3.0 * target[i]:
            continue
        locked[ring] = locked[a] = locked[b] = True
        chosen.append(i)
    if not chosen:
        return 0
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.collapse(bm, edges=_bm_edges(bm, lo, hi, chosen))
        bm.to_mesh(me)
    finally:
        bm.free()
    return len(chosen)


def _flip_pass(me, crease_deg) -> int:
    co, tris = _read(me)
    (lo, hi, f0, f1, count), crease, _cv, _bv = _creases(co, tris, crease_deg)
    val = np.bincount(np.concatenate((lo, hi)), minlength=len(co))
    inner = (count == 2) & ~crease
    g0, g1 = np.where(inner, f0, 0), np.where(inner, f1, 0)
    n = len(co)
    c = np.where(inner, np.clip(tris[g0].sum(axis=1) - lo - hi, 0, n - 1), 0)
    d = np.where(inner, np.clip(tris[g1].sum(axis=1) - lo - hi, 0, n - 1), 0)
    dev = np.abs
    before = dev(val[lo] - 6) + dev(val[hi] - 6) + dev(val[c] - 6) + dev(val[d] - 6)
    after = dev(val[lo] - 7) + dev(val[hi] - 7) + dev(val[c] - 5) + dev(val[d] - 5)
    new_len = np.linalg.norm(co[c] - co[d], axis=1)
    old_len = np.linalg.norm(co[hi] - co[lo], axis=1)
    nrm = _tri_normals(co, tris)
    flat = np.einsum("ij,ij->i", nrm[g0], nrm[g1]) > 0.85
    good = inner & flat & (after < before) & (new_len < 1.5 * old_len) & (c != d) & (val[lo] > 3) & (val[hi] > 3)
    idx = np.flatnonzero(good)
    if not len(idx):
        return 0
    used = np.zeros(len(tris), dtype=bool)
    existing = set(zip(lo.tolist(), hi.tolist()))
    chosen = []
    for i in idx[np.argsort(before[idx] - after[idx])[::-1]].tolist():
        if used[f0[i]] or used[f1[i]] or (min(c[i], d[i]), max(c[i], d[i])) in existing:
            continue
        used[f0[i]] = used[f1[i]] = True
        existing.add((min(c[i], d[i]), max(c[i], d[i])))
        chosen.append(i)
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        for e in _bm_edges(bm, lo, hi, chosen):
            bmesh.utils.edge_rotate(e, True)
        bm.to_mesh(me)
    finally:
        bm.free()
    return len(chosen)


def _smooth_pass(me, project, crease_deg, strength=0.5) -> None:
    co, tris = _read(me)
    (lo, hi, _f0, _f1, _c), _crease, cv, bv = _creases(co, tris, crease_deg)
    nrm = _vert_normals(co, tris)
    total = np.zeros_like(co)
    for axis in range(3):
        total[:, axis] = np.bincount(lo, co[hi, axis], len(co)) + np.bincount(hi, co[lo, axis], len(co))
    deg = np.bincount(np.concatenate((lo, hi)), minlength=len(co)).astype(np.float64)
    delta = total / np.maximum(deg, 1)[:, None] - co
    delta -= nrm * np.einsum("ij,ij->i", delta, nrm)[:, None]
    free = ~bv & ~cv & (deg > 0)
    moved = co.copy()
    moved[free] += strength * delta[free]
    if project is not None:
        moved[free] = project(moved[free])
    _write_co(me, moved)


def adaptive_triangles(co, tris, pts, L0, project, crease_deg, max_iter=8):
    """Isotropic remeshing (split / collapse / flip / smooth) towards the per-edge target length."""
    me = bpy.data.meshes.new("CM_guided_tmp")
    try:
        me.from_pydata(co.tolist(), [], tris.tolist())
        me.update()
        for it in range(max_iter):
            changed = _split_pass(me, pts, L0) + _collapse_pass(me, pts, L0, crease_deg)
            if it >= 1:
                _flip_pass(me, crease_deg)
            _smooth_pass(me, project, crease_deg)
            if changed < 0.01 * max(len(me.vertices), 1) and it >= 2:
                break
        _flip_pass(me, crease_deg)
        _smooth_pass(me, project, crease_deg)
        out_co, out_tris = _read(me)
    finally:
        bpy.data.meshes.remove(me)
    used = np.zeros(len(out_co), dtype=bool)
    used[out_tris.ravel()] = True
    remap = np.cumsum(used) - 1
    return out_co[used], remap[out_tris]


def _proxy(context, obj, src, max_tris):
    """Source triangles, decimated by Blender (fast) when there are many more than the result needs."""
    tris = src["tris"]
    if len(tris) <= max_tris:
        return src["co"], tris
    welded = bpy.data.meshes.new("CM_guided_proxy")  # the welded copy: seams split by an importer would tear apart
    welded.from_pydata(src["co"].tolist(), [], tris.tolist())
    welded.update()
    tmp = bpy.data.objects.new("CM_guided_proxy", welded)
    context.scene.collection.objects.link(tmp)
    try:
        mod = tmp.modifiers.new("d", "DECIMATE")
        mod.ratio = max(0.01, max_tris / max(len(tris), 1))
        evaluated = tmp.evaluated_get(context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", co)
            mesh.calc_loop_triangles()
            t = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
            mesh.loop_triangles.foreach_get("vertices", t)
        finally:
            evaluated.to_mesh_clear()
    finally:
        data = tmp.data
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.meshes.remove(data)
    co = co.reshape(-1, 3)
    t = t.reshape(-1, 3)
    key = np.round(co / (src["diag"] * 1e-6)).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    t = inv.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return co[first], t


def _whole_mesh(context, obj, src, pts, gs, L0, project):
    qe = float(np.sum(src["tri_area"] / (L0 * multiplier(src["tri_cent"], pts)) ** 2))
    state["predicted"] = int(qe)
    pco, ptris = _proxy(context, obj, src, int(np.clip(8 * qe, 3000, 60000)))
    crease = min(gs.feature_angle, 30.0) if any(p.kind == "SHARP" for p in pts) else gs.feature_angle
    ico, itris = adaptive_triangles(pco, ptris, pts, L0, project, crease)
    mesh = MeshData(vertices=ico, faces=[tuple(t) for t in itris.tolist()], name="guided")
    open_before = int((_edge_table(itris)[4] == 1).sum())
    note = ""
    for angle in dict.fromkeys((crease, max(crease, 45.0))):  # a low angle can make the engine leave holes
        options = RemeshOptions(target_faces=max(1, len(itris) // 2), mode="AUTO", seed_count=3, feature_angle_deg=angle, force_quads=bool(gs.pure_quads))
        out, report = remesh_mesh(mesh, options)
        if report.success and out.faces and int((_edge_table(_fan(out.faces))[4] == 1).sum()) <= open_before:
            note = ""
            break
        note = "PairClosure left open edges" if report.success else f"PairClosure failed: {report.message}"
    if not out.faces:
        return None, note
    co = out.vertices
    faces = [tuple(f) for f in out.faces]
    if project is not None and gs.fit_iterations > 0:
        _t, _c, cv, bv = _creases(co, _fan(faces), gs.feature_angle)
        co = patch.fit_to_surface(co, faces, cv | bv, project, gs.fit_iterations, 0.5)
    mat = src["mat"][src["tri_face"][_nearest_tris(_bvh(src), _centroids(co, faces))]]
    return {"co": co, "faces": faces, "new": np.ones(len(faces), bool), "locked": np.zeros(len(faces), bool), "mat": mat}, note


# --------------------------------------------------------------------------- region path


def _region_pass(base_co, base_faces, mask, pts, L0, gs, project, base_locked, base_mat, fixed_target=0.0):
    """Remesh the faces in `mask` (seam kept). Returns (result dict or None, warnings)."""
    comps, warns = patch.prepare_regions(base_co, base_faces, mask, min_island=4, max_hole=4, smooth=2, split_rings=True, frozen=base_locked)
    results = []
    for comp in comps:
        cent = _centroids(base_co, [base_faces[i] for i in comp])
        target = fixed_target or L0 * float(multiplier(cent, pts).mean())
        res = patch.remesh_region(base_co, base_faces, comp, target_length=target, strategy=gs.strategy, falloff=2.5 * target, max_faces=region.MAX_PATCH_FACES)
        if res.ok and project is not None and gs.fit_iterations > 0:
            region._fit_result(base_co, res, project, gs.fit_iterations)
        if not res.ok:
            warns.append(f"Region ({len(comp)} faces) skipped: {res.message}")
        results.append(res)
    return _merge_results(base_co, base_faces, results, base_locked, base_mat), warns


def _merge_results(base_co, base_faces, results, base_locked, base_mat, no_bigger=False):
    """The base mesh with every finished patch put in. None when no patch worked."""
    good = [r for r in results if r.ok and (not no_bigger or len(r.faces) <= len(r.removed_faces))]  # Live Boolean: a redo never makes a region bigger
    if not good:
        return None
    co, faces, new = region._merged_arrays({"co": base_co, "faces": base_faces, "results": good})
    removed = np.zeros(len(base_faces), dtype=bool)
    for r in good:
        removed[r.removed_faces] = True
    kept = ~removed
    locked = np.concatenate((base_locked[kept], np.zeros(int(new.sum()), bool)))
    mat = None
    if base_mat is not None:
        extra = [base_mat[s] if 0 <= s < len(base_mat) else 0 for r in good for s in (r.face_source or [-1] * len(r.faces))]
        mat = np.concatenate((base_mat[kept], np.asarray(extra, dtype=np.int32)))
    return {"co": co, "faces": faces, "new": new, "locked": locked, "mat": mat, "removed": removed}


# --------------------------------------------------------------------------- gather / suggest / spot


def gather(obj, gs=None) -> list[Pt]:
    """Points on the object plus tagged empties."""
    pts = [Pt(p.kind, np.asarray(p.co, float), np.asarray(p.normal, float), float(p.radius), float(p.strength), p.fill) for p in obj.curiomesh_points]
    inv = obj.matrix_world.inverted()
    scale = max(float(np.mean(obj.matrix_world.to_scale())), 1e-9)
    default = gs.new_kind if gs is not None else "DETAIL"
    for o in bpy.data.objects:
        if o.type != "EMPTY" or o.parent not in (None, obj) or not (o.get(POINT_TAG) is not None or o.name.startswith("CM_Point")):
            continue
        local = inv @ o.matrix_world.translation
        kind = str(o.get("curiomesh_kind", default)).upper()
        ok, _loc, normal, _i = obj.closest_point_on_mesh(local)
        pts.append(
            Pt(
                kind if kind in KIND_KEYS else default,
                np.asarray(local, float),
                np.asarray(normal if ok else (0, 0, 1), float),
                max(float(o.empty_display_size) * float(np.mean(o.matrix_world.to_scale())) / scale, 1e-4),
                float(o.get("curiomesh_strength", 1.0)),
                str(o.get("curiomesh_fill", "FLOW")),
            )
        )
    return pts


def add_point(obj, co, normal, kind, radius, strength, auto=False, fill="FLOW"):
    p = obj.curiomesh_points.add()
    p.co, p.normal, p.kind, p.radius, p.strength, p.auto, p.fill = tuple(co), tuple(normal), kind, radius, strength, auto, fill
    obj.curiomesh_point_index = len(obj.curiomesh_points) - 1
    return p


def suggest(obj, gs) -> list[dict]:
    """High-curvature spots, hard-edge spots and small holes, as point descriptions (nothing is written)."""
    src = _source_data(obj)
    co, tris, diag = src["co"], src["tris"], src["diag"]
    preset = PRESETS[gs.preset]
    (lo, hi, _f0, _f1, count), _crease, cv, bv = _creases(co, tris, gs.feature_angle)
    vn = _vert_normals(co, tris)
    length = np.linalg.norm(co[hi] - co[lo], axis=1)
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", vn[lo], vn[hi]), -1, 1))
    k = ang / np.maximum(length, 1e-12)
    deg = np.maximum(np.bincount(lo, minlength=len(co)) + np.bincount(hi, minlength=len(co)), 1)
    curv = (np.bincount(lo, k, len(co)) + np.bincount(hi, k, len(co))) / deg
    for _ in range(preset["smooth"]):
        curv = 0.5 * curv + 0.5 * (np.bincount(lo, curv[hi], len(co)) + np.bincount(hi, curv[lo], len(co))) / deg
    curv = curv * (~bv)
    top = float(np.percentile(curv[curv > 0], 97)) if (curv > 0).any() else 1.0
    score = curv / max(top, 1e-12) + cv * 0.8
    radius = gs.new_radius or preset["radius"] * diag
    out = []
    be = np.flatnonzero(count == 1)  # small holes (eyes, mouths) become Loop points
    if len(be):
        parent: dict[int, int] = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in be.tolist():
            parent[find(int(lo[i]))] = find(int(hi[i]))
        groups: dict[int, list[int]] = {}
        for v in {int(x) for x in np.concatenate((lo[be], hi[be]))}:
            groups.setdefault(find(v), []).append(v)
        for verts in groups.values():
            vp = co[verts]
            perimeter = float(length[be][np.isin(lo[be], verts)].sum())
            if len(verts) >= 4 and perimeter < 0.25 * diag:
                c = vp.mean(axis=0)
                out.append({"kind": "LOOP", "co": c, "r": max(1.8 * float(np.linalg.norm(vp - c, axis=1).max()), 0.02 * diag)})
    taken = [np.asarray(o["co"]) for o in out]
    for v in np.argsort(-score).tolist():
        if len(out) >= gs.suggest_count or score[v] < 0.35:
            break
        if all(np.linalg.norm(co[v] - t) > radius for t in taken):
            kind = "SHARP" if cv[v] and preset["kind"] == "SHARP" else "DETAIL"
            out.append({"kind": kind, "co": co[v], "r": radius})
            taken.append(co[v])
    for o in out:
        ok, loc, normal, _i = obj.closest_point_on_mesh(Vector(o["co"]))
        o["co"] = np.asarray(loc if ok else o["co"], float)
        o["normal"] = np.asarray(normal if ok else (0, 0, 1), float)
    return out[: gs.suggest_count]


def next_spot(res, project, L0):
    """The worst area of a result: leftover triangles, n-gons, poles and surface error. (position, text) or None."""
    co, faces = res["co"], res["faces"]
    size = np.asarray([len(f) for f in faces])
    flat = np.concatenate([np.asarray(f) for f in faces])
    owner = np.repeat(np.arange(len(faces)), size)
    valence = np.bincount(flat, minlength=len(co))
    _t, _c, _cv, bv = _creases(co, _fan(faces), 180.0)
    pole = (valence != 4) & ~bv
    pole_w = np.where(valence >= 6, 1.5, 1.0) * pole
    score = np.where(size == 3, 2.0, 0.0) + np.where(size > 4, 3.0, 0.0) + np.bincount(owner, pole_w[flat], len(faces)) / np.maximum(size, 1) * 2.0
    cent = _centroids(co, faces)
    if project is not None:
        score = score + np.clip(np.linalg.norm(project(cent) - cent, axis=1) / max(0.25 * L0, 1e-9), 0.0, 3.0)
    vs = np.bincount(flat, score[owner], len(co))
    smooth = np.bincount(owner, vs[flat], len(faces)) / np.maximum(size, 1)
    i = int(np.argmax(smooth))
    if smooth[i] < 1.5:
        return None
    near = np.linalg.norm(cent - cent[i], axis=1) < 3 * L0
    tris = int(((size == 3) & near).sum())
    poles = int(pole[np.unique(flat[np.isin(owner, np.flatnonzero(near))])].sum())
    return cent[i], f"{tris} triangles and {poles} poles around here"


# --------------------------------------------------------------------------- update


def state_source():
    obj = bpy.data.objects.get(state["source"]) if state["source"] else None
    return obj if obj is not None and obj.type == "MESH" else None


def _source_mesh(context):
    o = context.object
    if o is not None and o.type == "MESH" and o.name != PREVIEW_NAME:
        return o
    return state_source()


def _signature(context, src_obj) -> tuple:
    gs = context.scene.curiomesh_guided
    parts = [
        src_obj.name,
        len(src_obj.data.vertices),
        len(src_obj.data.polygons),
        gs.scope,
        gs.strategy,
        gs.base_faces,
        gs.fit_iterations,
        gs.pure_quads,
        round(gs.feature_angle, 2),
        tuple(round(x, 5) for row in src_obj.matrix_world for x in row),
        state["lock"]["id"] if state["lock"] else 0,
    ]
    if gs.scope == "AUTO":
        parts += [gs.bool_on, round(gs.bool_detail, 3), state["auto_base"] is not None]
        if gs.bool_on:
            for shape in gs.bool_shapes:
                if shape.obj is not None:
                    data_size = len(shape.obj.data.vertices) if shape.obj.type == "MESH" else 0
                    parts.append((shape.operation, shape.obj.name, data_size, tuple(round(x, 5) for row in shape.obj.matrix_world for x in row)))
    for p in src_obj.curiomesh_points:
        parts.append((p.kind, p.fill, round(p.radius, 5), round(p.strength, 4), tuple(round(x, 5) for x in p.co)))
    for o in bpy.data.objects:
        if o.type == "EMPTY" and (o.get(POINT_TAG) is not None or o.name.startswith("CM_Point")):
            parts.append((o.name, o.parent, round(o.empty_display_size, 4), tuple(round(x, 4) for x in o.matrix_world.translation), str(o.get("curiomesh_kind", "")), o.get("curiomesh_strength", 1.0)))
    return tuple(parts)


def remove_preview() -> None:
    obj = bpy.data.objects.get(PREVIEW_NAME)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    state.update(res=None, spot=None, draw=None, draw_key=None, batches=None)


# --------------------------------------------------------------------------- teach by example


class _PreviewOp:
    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(PREVIEW_NAME) is not None and state["res"] is not None


def library_path(create: bool = True) -> str | None:
    """The user's own file of remembered fixes, in the add-on's user folder (not in the add-on). None if that can't be
    written. With create False (reading), the folder is not made."""
    folder = hardware.user_folder(create=create)
    if folder is None:
        return None
    return os.path.join(folder, "fixes.json")


def _library() -> list[dict]:
    """The remembered fixes: from the user's file, or this session's memory when it can't be used."""
    path = None if state["fixes_in_memory"] else library_path(create=False)
    if path is None:
        return state["fixes"]
    return teach.load(path)


def fix_count() -> int | None:
    """How many fixes are remembered, for the panel: read from the file in a one-shot timer, never in the panel's draw.
    None until that has run."""
    return overlay.drawn_arrays("teach.count", (state["fixes_saved"], state["fixes_in_memory"]), lambda: len(_library()))


def _keep_library(fixes: list[dict]) -> None:
    """Save the fixes to the user's file. When that can't be written they stay in memory for this session, and the panel says so."""
    path = library_path()
    try:
        if path is None:
            raise OSError("no config folder")
        teach.save(path, fixes)
        state["fixes_in_memory"] = False
    except OSError:
        state["fixes"] = fixes
        state["fixes_in_memory"] = True
    state["fixes_saved"] += 1  # the panel's count is read again


def _drain(generator):
    """Run a generator to its end and give back what it returns."""
    try:
        while True:
            next(generator)
    except StopIteration as done:
        return done.value


def _camera_accepts(src, before, after, centre, half) -> bool:
    """A camera looks at the spot from six sides: the new faces must not look further from the original than the old ones."""
    pixels = 64
    pixel = 2.0 * half / pixels
    reference = _drain(autoretopo.pictures(src["co"], src["faces"], centre, half, pixels, _bvh(src)))
    wrong = []
    for co, faces in (before, after):
        cent = np.asarray([np.mean([co[v] for v in f], axis=0) for f in faces])
        near = np.abs(cent - centre).max(axis=1) < 2.0 * half
        (sub_co, sub_faces, _m), _rest = edgeloops.split_region(co, faces, np.zeros(len(faces), dtype=int), near)
        view = _drain(autoretopo.pictures(sub_co, sub_faces, centre, half, pixels))
        wrong.append(autoretopo.picture_counts(reference, view, pixel))
    return wrong[1][0] <= wrong[0][0] * 1.15 + 0.01 * max(wrong[0][1], 1)


def _adopt(context, source, co, faces, mat) -> None:
    """Make an edited mesh the result: the preview shows it and everything after it (more fixes, Apply) works on it."""
    fa, fb = patch.face_adjacency(faces)
    parts = np.unique(patch._labels(len(faces), fa, fb), return_inverse=True)[1]
    count = len(faces)
    res = {"co": np.asarray(co, dtype=np.float64), "faces": [tuple(f) for f in faces], "new": np.ones(count, bool), "locked": np.zeros(count, bool), "mat": np.asarray(mat, dtype=np.int32), "parts": parts}
    state["res"] = res
    if state["auto_base"] is not None:
        state["auto_base"]["out"] = dict(res)
    _build_preview(context, source, res, uvs=False)
    state["draw_key"] = None
    overlay.tag_redraw()


def _kept(source) -> list[dict]:
    """The fixes kept on this object. A .blend file can come from anybody, so only valid ones count."""
    try:
        text = source.get("curiomesh_kept", "[]")
        if not isinstance(text, str) or len(text) > teach.MOST_BYTES:  # a huge string would freeze Blender while parsing
            return []
        kept = json.loads(text)[: teach.MOST_FIXES]
    except (ValueError, TypeError, RecursionError):
        return []
    try:
        return [e for e in kept if isinstance(e, dict) and teach.valid_fix(e.get("fix")) and type(e.get("reach")) in (int, float) and teach._points([e.get("at")])] if isinstance(kept, list) else []
    except (TypeError, ValueError):
        return []


def _scan(res) -> None:
    """Which remembered fixes fit the result now."""
    found = []
    for fix in _library():
        found += [(fix, m) for m in teach.find(fix, res["co"], res["faces"])[:5]]
    state["found"] = sorted(found, key=lambda pair: pair[1]["score"])


def _restore_kept(context, source) -> None:
    """After a new search: put each fix you made by hand last time back where it was, if the quads there came out the same."""
    res = state["res"]
    if res is None:
        return
    co, faces, mat = res["co"], list(res["faces"]), np.asarray(res["mat"])
    done = lost = 0
    for entry in _kept(source):
        hits = teach.find(entry["fix"], co, faces, near=entry["at"], reach=entry["reach"])
        if not hits:
            lost += 1
            continue
        style = int(mat[hits[0]["faces"][0]]) if len(mat) else 0
        co, faces, kept = teach.put_in(co, faces, entry["fix"], hits[0], None)
        mat = np.concatenate((mat[kept], np.full(len(faces) - int(kept.sum()), style, dtype=np.int32)))
        done += 1
    if done:
        _adopt(context, source, co, faces, mat)
    if done or lost:
        state["message"] = f"{state['message']} | Put back {done} fix(es) you made by hand" + (f"; {lost} did not fit the new quads" if lost else "")


class CURIOMESH_OT_teach_remember(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.teach_remember"
    bl_label = "Remember This Fix"
    bl_description = "Remember the area you fixed by hand on the preview, and offer the same fix where the pattern shows up again. Kept on this PC only; nothing is uploaded"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source, res = state_source(), state["res"]
        preview = bpy.data.objects[PREVIEW_NAME]
        if preview.mode == "EDIT":
            preview.update_from_editmode()
        mesh = preview.data
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        faces = [tuple(p.vertices) for p in mesh.polygons]
        mat = np.empty(len(faces), dtype=np.int32)
        mesh.polygons.foreach_get("material_index", mat)
        size = float(np.linalg.norm(res["co"].max(axis=0) - res["co"].min(axis=0))) or 1.0
        gone, new = teach.changed(res["co"], res["faces"], co, faces, size)
        if not gone and not new:
            self.report({"WARNING"}, "Nothing changed on the preview: fix an area in Edit Mode first")
            return {"CANCELLED"}
        fix = teach.make_fix(res["co"], res["faces"], gone, co, faces, new, size)
        if isinstance(fix, str):
            self.report({"WARNING"}, fix)
            return {"CANCELLED"}
        ring = np.asarray(res["co"])[teach.outline(res["faces"], gone)]
        edge = float(np.mean(np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)))
        _leave_preview(context)
        _keep_library(teach.merge(_library(), [fix]))
        entry = {"fix": fix, "at": ring.mean(axis=0).tolist(), "reach": 3.0 * edge}
        source["curiomesh_kept"] = json.dumps(_kept(source) + [entry])
        _adopt(context, source, co, faces, mat)
        _scan(state["res"])
        state["message"] = f"Remembered: {len(fix['before'])} faces became {len(fix['after'])}. It is kept on this mesh, and offered on others that look like it"
        self.report({"INFO"}, state["message"])
        return {"FINISHED"}


class CURIOMESH_OT_teach_find(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.teach_find"
    bl_label = "Look for Known Fixes"
    bl_description = "Look in this result for patches that match a fix you remembered"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        _scan(state["res"])
        state["message"] = f"{len(state['found'])} place(s) look like something you fixed before" if state["found"] else "Nothing here matches a fix you remembered"
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_teach_apply(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.teach_apply"
    bl_label = "Apply Known Fix"
    bl_description = "Put the closest remembered fix in. A camera looks at the spot from six sides first and the fix is dropped if it makes the spot look further from your original"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(state["found"]) and super().poll(context)

    def execute(self, context):
        source, res = state_source(), state["res"]
        src = _source_data(source)
        fix, match = state["found"][0]
        co, faces = res["co"], list(res["faces"])
        project = region._projector(_bvh(src))
        new_co, new_faces, kept = teach.put_in(co, faces, fix, match, project)
        centre = np.asarray(match["centre"])
        ring = co[[match["vmap"][i] for i in range(int(fix["ring"]))]]
        half = 1.5 * float(np.linalg.norm(ring - centre, axis=1).max())
        if not _camera_accepts(src, (co, faces), (new_co, new_faces), centre, half):
            state["found"] = state["found"][1:]
            state["message"] = "The camera says that fix would make the spot look worse: left alone"
            self.report({"WARNING"}, state["message"])
            return {"CANCELLED"}
        mat = np.asarray(res["mat"])
        style = int(mat[match["faces"][0]]) if len(mat) else 0
        mat = np.concatenate((mat[kept], np.full(len(new_faces) - int(kept.sum()), style, dtype=np.int32)))
        _adopt(context, source, new_co, new_faces, mat)
        _scan(state["res"])
        state["message"] = f"Put in a fix you made before. {len(state['found'])} more place(s) look like it"
        self.report({"INFO"}, state["message"])
        return {"FINISHED"}


class CURIOMESH_OT_teach_export(bpy.types.Operator, ExportHelper):
    bl_idname = "curiomesh.teach_export"
    bl_label = "Export Fixes"
    bl_description = "Save a copy of your remembered fixes to share. They hold quad patterns and shape numbers only, no mesh"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        teach.save(self.filepath, _library())
        return {"FINISHED"}


class CURIOMESH_OT_teach_import(bpy.types.Operator, ImportHelper):
    bl_idname = "curiomesh.teach_import"
    bl_label = "Import Fixes"
    bl_description = "Add the fixes in a file somebody shared to your own. Ones you already have are skipped"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        mine = _library()
        both = teach.merge(mine, teach.load(self.filepath))
        _keep_library(both)
        self.report({"INFO"}, f"{len(both) - len(mine)} fix(es) added")
        return {"FINISHED"}


class CURIOMESH_OT_teach_forget_kept(bpy.types.Operator):
    bl_idname = "curiomesh.teach_forget_kept"
    bl_label = "Forget Kept Fixes"
    bl_description = "Stop putting the fixes you made by hand back on this mesh when you press Start again. Your remembered library is not touched"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        source = _source_mesh(context)
        return source is not None and bool(_kept(source))

    def execute(self, context):
        del _source_mesh(context)["curiomesh_kept"]
        return {"FINISHED"}


def _bend_edges(base, res) -> list[tuple[int, int]]:
    """Edges of the finished result that lie on the middle loop of a deformation line (and its mirror image)."""
    job = base["job"] if base else None
    groups = getattr(job, "bend_groups", None)
    if not groups:
        return []
    centres = [loops[len(loops) // 2] for tag, loops in groups if tag == "BEND"]
    if not centres:
        return []
    if job.mirror is not None:
        axis, offset = job.mirror[:2]
        for line in list(centres):
            flipped = line.copy()
            flipped[:, axis] = 2.0 * offset - flipped[:, axis]
            centres.append(flipped)
    return guides.loop_edges(centres, res["co"], res["faces"], 0.3 * job.gap)


def _mark_edges(mesh, edges, crease: bool, sharp: bool) -> None:
    """Crease and sharp marks on the given (lo, hi) edges of a new mesh."""
    if not edges or not (crease or sharp):
        return
    pairs = np.empty(len(mesh.edges) * 2, dtype=np.int64)
    mesh.edges.foreach_get("vertices", pairs)
    index = {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(pairs.reshape(-1, 2).tolist())}
    ids = [index[e] for e in edges if e in index]
    if crease:
        layer = mesh.attributes.get("crease_edge") or mesh.attributes.new("crease_edge", "FLOAT", "EDGE")
        values = np.zeros(len(mesh.edges), dtype=np.float32)
        values[ids] = 1.0
        layer.data.foreach_set("value", values)
    if sharp:
        flags = np.zeros(len(mesh.edges), dtype=bool)
        flags[ids] = True
        layer = mesh.attributes.get("sharp_edge") or mesh.attributes.new("sharp_edge", "BOOLEAN", "EDGE")
        layer.data.foreach_set("value", flags)


def _make_mesh(name, source, res, uvs=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(res["co"].tolist(), [], res["faces"])
    mesh.update()
    for m in source.data.materials:
        mesh.materials.append(m)
    if res.get("mat") is not None and len(mesh.polygons) == len(res["mat"]):
        mesh.polygons.foreach_set("material_index", np.asarray(res["mat"], dtype=np.int32))
    if uvs:
        _transfer_uvs(mesh, source, _source_data(source), res)
    return mesh


def _shade_by_angle(mesh) -> None:
    """Smooth shading, hard edges past SMOOTH_DEG, marked as sharp edges (no modifier needed)."""
    faces = len(mesh.polygons)
    mesh.polygons.foreach_set("use_smooth", np.ones(faces, dtype=bool))
    edge_of_corner = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", edge_of_corner)
    starts, sizes, normals = np.empty(faces, dtype=np.int64), np.empty(faces, dtype=np.int64), np.empty(faces * 3)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", sizes)
    mesh.polygons.foreach_get("normal", normals)
    order = np.argsort(starts)
    face_of_corner = np.repeat(order, sizes[order])
    mesh.edges.foreach_set("use_edge_sharp", sharp_edges(edge_of_corner, face_of_corner, normals.reshape(-1, 3), len(mesh.edges)))
    mesh.update()


def _build_preview(context, source, res, uvs=True, mesh=None) -> None:
    """Show `res` as the preview object. `mesh`: one already made for it (the size ladder keeps one per size)."""
    obj = bpy.data.objects.get(PREVIEW_NAME)
    mesh = _make_mesh(PREVIEW_NAME + "_Mesh", source, res, uvs) if mesh is None else mesh
    if obj is None:
        obj = bpy.data.objects.new(PREVIEW_NAME, mesh)
        (source.users_collection[0] if source.users_collection else context.scene.collection).objects.link(obj)
    else:
        old = obj.data
        obj.data = mesh
        if old.users == 0 and not old.name.startswith(SPARE_MESHES):  # spares go when their size ladder or Organize forgets them
            bpy.data.meshes.remove(old)
    gs = context.scene.curiomesh_guided
    segments.place(obj, source, gs)
    obj.show_wire = obj.show_all_edges = gs.seg_wire or gs.scope != "AUTO"
    obj.hide_select = False


# --------------------------------------------------------------------------- work areas


def stop_work() -> None:
    if state["work"] is not None:
        state["work"]["runner"].stop()
        state["work"] = None


def _area_target(p: Pt, L0: float) -> float:
    return L0 * _fac(p.kind, p.s)


def _find_spot(res, project, L0) -> None:
    if res is not None:
        try:
            spot = next_spot(res, project, L0)
            res["spot"] = {"pos": spot[0], "text": spot[1]} if spot else None
        except Exception:  # the hint must never break an update
            res["spot"] = None


def _start_areas(source, gs, pts, src, L0) -> bool:
    """Remesh the areas (one synchronous call per plan); _collect_work picks up each finished piece."""
    lock = state["lock"]
    co, faces = (lock["co"], lock["faces"]) if lock else (src["co"], src["faces"])
    frozen = lock["locked"] if lock else np.zeros(len(faces), dtype=bool)
    mat = lock.get("mat") if lock else src["mat"]
    key = (src["sig"], lock["id"] if lock else 0, gs.fit_iterations, L0)
    if state["work"] is None or state["work"]["key"] != key:
        stop_work()
        project = region._projector(_bvh(src))
        iterations = gs.fit_iterations
        runner = areas.AreaRunner(
            co, faces, frozen,
            fit=(lambda r: region._fit_result(co, r, project, iterations)) if iterations > 0 else None,
            merge=lambda results: _merge_results(co, faces, results, frozen, mat),
            finish=lambda res: _find_spot(res, project, L0),
            max_faces=region.MAX_PATCH_FACES,
        )
        state["work"] = {"key": key, "runner": runner, "cent": _centroids(co, faces), "seen": 0, "areas": []}
    work = state["work"]
    items = [areas.Area(p.co, p.r, _area_target(p, L0), p.fill) for p in pts]
    groups = areas.group_areas(items)
    jobs = [areas.Job(m & ~frozen, items[g[0]].target, items[g[0]].strategy, g) for g, m in zip(groups, areas.area_masks(work["cent"], items, groups)) if (m & ~frozen).any()]
    active = source.curiomesh_point_index
    jobs.sort(key=lambda j: (active not in j.tag, -j.tag[-1]))  # the area you just touched first, then newest first
    work["areas"] = pts
    state.update(L0=L0, ms=0.0, spot=None)
    state["message"] = "Working..."
    work["runner"].start(jobs)
    return True


def auto_target(gs, faces: int) -> int:
    """Faces to land on for Reduce To, or 0 when the aim is to keep detail."""
    if gs.auto_aim != "REDUCE":
        return 0
    return max(12, int(round(gs.auto_reduce_percent / 100.0 * faces)) if gs.auto_reduce_by == "PERCENT" else int(gs.auto_reduce_count))


def _mirror_half(gs, src, codes):
    """(half, note): the half to solve as (co, faces, mat, mirror) or None, and what to tell the user about the symmetry."""
    if gs.sym_axis == "OFF":
        state["sym_note"] = ""
        return None, ""
    key = (src["sig"], hash(np.asarray(codes).tobytes()), gs.sym_axis, gs.sym_force, gs.sym_side, gs.sym_offset_use and round(gs.sym_offset, 6))
    if state.get("sym_half", (None,))[0] == key:  # the ladder asks again for every size: same mesh, same half
        return state["sym_half"][1]
    found = symmetry.find_plane(src["co"], _bvh(src), "XYZ" if gs.sym_axis == "AUTO" else gs.sym_axis, gs.sym_offset if gs.sym_offset_use else None)[0]
    state["sym_note"] = f"{symmetry.AXES[found.axis]} plane: {found.differ()}"
    if not gs.sym_force and not found.ok(gs.sym_axis != "AUTO"):
        out = None, f"Not symmetric enough to mirror ({found.text()}): solved the whole mesh. Symmetrize mirrors it anyway"
    else:
        side = src["tri_cent"][:, found.axis] >= found.offset
        keep = {"POS": 1, "NEG": -1}.get(gs.sym_side) or (1 if src["tri_area"][side].sum() >= src["tri_area"][~side].sum() else -1)  # Auto: the side with more surface
        co, faces, source, origin = symmetry.cut_half(src["co"], src["faces"], found.axis, found.offset, keep)
        if not faces:
            out = None, "Mirror found nothing on one side of the plane: solved the whole mesh"
        else:
            note = f"{'Symmetrized' if gs.sym_force else 'Mirrored'} from the {'+' if keep > 0 else '-'}{symmetry.AXES[found.axis]} half; {found.differ()}"
            out = (co, faces, codes[source], (found.axis, found.offset, src["diag"], len(src["faces"]), keep), origin), note
    state["sym_half"] = (key, out)
    return out


def _half_paint(paint, src, half):
    """The marker paint on the half that is solved. Paint on the other side counts too, mirrored over: whichever side
    is marked more strongly wins, so you can mark either side."""
    co, mirror, origin = half[0], half[3], half[4]
    other = co.copy()
    other[:, mirror[0]] = 2.0 * mirror[1] - other[:, mirror[0]]
    tree = KDTree(len(src["co"]))
    for i, p in enumerate(src["co"].tolist()):
        tree.insert(p, i)
    tree.balance()
    across = paint[np.fromiter((tree.find(p)[1] for p in other.tolist()), dtype=np.int64, count=len(other))]
    here = paint[origin]
    return np.where(np.abs(across) > np.abs(here), across, here)


def _guide_lines(context, gs, src, kind: str = "GUIDE") -> list[np.ndarray]:
    """The curves of one type from the list (guides, edge loops or deformation lines), laid on the surface of the source mesh."""
    source = state_source()
    on = {"GUIDE": gs.guide_use, "LOOP": gs.loop_use, "BEND": gs.bend_use}[kind]
    objects = [g.obj for g in gs.guide_curves if g.obj is not None and g.kind == kind]
    if not on or not objects or source is None:
        return []
    bvh = _bvh(src)
    raw = guides.gather(objects, source, context.evaluated_depsgraph_get())
    return guides.on_surface(raw, lambda p: bvh.find_nearest(Vector(p))[0], src["diag"] / 400.0)


def _curve_sig(gs) -> tuple:
    """A cheap fingerprint of the curves in the list: moving, redrawing, adding or deleting one changes it."""
    out = [(gs.loop_use, gs.guide_use, gs.bend_use, gs.loop_count, gs.bend_loops, round(gs.loop_gap, 6)), tuple(tweaks(gs).values())]  # a Tweak slider reruns the search like a curve
    if gs.density_use:
        out.append(_marker_sig(bpy.context.scene))  # so does moving a marker ball
    for g in gs.guide_curves:
        o = g.obj
        if o is None:
            continue
        data = ()
        if o.type == "CURVE":
            data = tuple(round(c, 5) for sp in o.data.splines for pt in (sp.points if sp.type == "POLY" else sp.bezier_points) for c in pt.co[:3])
        elif o.type == "MESH":
            flat = np.empty(len(o.data.vertices) * 3)
            o.data.vertices.foreach_get("co", flat)
            data = (len(flat), round(float(flat.sum()), 5))
        out.append((o.name, g.kind, tuple(round(x, 5) for row in o.matrix_world for x in row), data))
    return tuple(out)


MARKER_KEY = "curiomesh_marker"  # custom property of a marker ball: +1 More Detail, -1 Less Detail


def markers(scene) -> list:
    return [o for o in scene.objects if o.type == "EMPTY" and MARKER_KEY in o]


def _marker_sig(scene) -> tuple:
    return tuple((o.name, float(o[MARKER_KEY]), o.empty_display_size, tuple(round(x, 5) for row in o.matrix_world for x in row)) for o in markers(scene))


def _group_paint(source, name: str) -> np.ndarray | None:
    """Weights 0..1 per vertex of the mesh from one vertex group, or None."""
    group = source.vertex_groups.get(name)
    if group is None:
        return None
    bm = bmesh.new()
    try:
        bm.from_mesh(source.data)
        layer = bm.verts.layers.deform.active
        if layer is None:
            return None
        return np.asarray([v[layer].get(group.index, 0.0) for v in bm.verts], dtype=np.float64)
    finally:
        bm.free()


def _painted(source, gs, src) -> np.ndarray | None:
    """Detail markers -1..1 per vertex of the welded source (positive: More Detail, negative: Less Detail), from the two
    vertex groups and the marker balls, or None when nothing is marked."""
    if not gs.density_use:
        return None
    more, less = np.zeros(len(src["co"])), np.zeros(len(src["co"]))
    for name, into in ((gs.density_group, more), (gs.less_group, less)):
        paint = _group_paint(source, name)
        if paint is not None:
            np.maximum.at(into, src["weld"], paint)
    inverse = source.matrix_world.inverted()
    scale = max(sum(source.matrix_world.to_scale()) / 3.0, 1e-12)
    for marker in markers(bpy.context.scene):
        centre = np.asarray(inverse @ marker.matrix_world.translation)
        radius = marker.empty_display_size * max(marker.matrix_world.to_scale()) / scale
        np.maximum(more if marker[MARKER_KEY] > 0 else less, marker_weights(src["co"], centre, radius), out=more if marker[MARKER_KEY] > 0 else less)
    marked = more - less
    return marked if np.abs(marked).max(initial=0.0) > 0.0 else None


def marker_weights(co: np.ndarray, centre, radius: float) -> np.ndarray:
    """1 at a marker ball's centre, falling smoothly to 0 at its radius."""
    d2 = ((co - np.asarray(centre)) ** 2).sum(axis=1) / max(radius, 1e-12) ** 2
    return np.clip(1.0 - d2, 0.0, 1.0) ** 2


def _real_materials(out: dict, charts) -> dict:
    """A copy of a result with its face kinds turned back into real materials, remembering each face's UV island."""
    return dict(out) if charts is None else dict(out, mat=charts[1][out["mat"]], chart=charts[2][out["mat"]])


def _uv_charts(source, src):
    """(code per face, material per code, UV island per code) for the active UV layer, or None if there's nothing to keep."""
    layer = source.data.uv_layers.active if source is not None else None
    if layer is None:
        return None
    uv = np.empty(len(source.data.loops) * 2, dtype=np.float64)
    layer.data.foreach_get("uv", uv)
    return chart_codes(src["mat"], _uv_islands(src, uv.reshape(-1, 2)))


def machine_info(refresh: bool = False) -> dict:
    """This computer, read once (cores, memory and a 0.1 s ray test) and kept."""
    if state["machine"] is None or refresh:
        state["machine"] = hardware.detect()
        state["rays"] = hardware.benchmark()
    return state["machine"]


def effort_settings(gs) -> hardware.Settings:
    return hardware.settings_for(gs.auto_effort, machine_info())


def job_cost(gs, faces: int, sig=None, bakes: int = 1) -> tuple[str, str, hardware.Settings, float, float]:
    """(level, sentence, settings, search seconds, bake seconds) for a mesh of this many faces on this computer. The sentence includes the bake."""
    machine = state["machine"]
    settings = hardware.settings_for(gs.auto_effort, machine)
    parts = state["parts"][1] if state["parts"] is not None and state["parts"][0] == sig else None
    factor = hardware.style_factor(gs.auto_style, gs.auto_method)
    est = hardware.estimate(faces, parts, settings, state["rays"])
    est["seconds"] *= factor
    level, text = hardware.rate(machine, est, settings, factor > 1 or gs.multires_on)
    bake_s = bakes * hardware.bake_seconds(int(gs.bake_size), gs.bake_detail, gs.bake_sets, machine)
    return level, text if level == "TOO_BIG" else hardware.with_bake(text, bake_s), settings, est["seconds"], bake_s


def job_rating(gs, faces: int, sig=None) -> tuple[str, str, hardware.Settings]:
    """(level, sentence, settings) for a mesh of this many faces on this computer."""
    return job_cost(gs, faces, sig)[:3]


def _solve_inputs(gs, src, lines, paint):
    """What the search runs on: (co, faces, mat, mirror, lines, paint, note). With Mirror it is one half, and the strokes and paint follow it."""
    half, note = _mirror_half(gs, src, src["mat"])
    co, faces, mat, mirror = half[:4] if half is not None else (src["co"], src["faces"], src["mat"], None)
    if paint is not None:
        paint = _half_paint(paint, src, half) if half is not None else paint
    if lines and mirror:
        lines = guides.reflect_to_side(lines, mirror[0], mirror[1], mirror[4])
    return co, faces, mat, mirror, lines, paint, note


def loose_pieces(src) -> int:
    """How many loose pieces the source has, counted once per mesh state."""
    if state["parts"] is None or state["parts"][0] != src["sig"]:
        fa, fb = patch.face_adjacency(src["faces"])
        state["parts"] = (src["sig"], int(len(np.unique(patch._labels(len(src["faces"]), fa, fb)))))
    return state["parts"][1]


def _start_segments(context, gs, src, target: int | None = None) -> bool:
    """Start the search, to land on `target` faces if given. It runs in slices from a timer on the main thread."""
    if gs.auto_profile in profiles.BLOCKOUT_USES:
        return _start_blockout(gs, src)
    target = auto_target(gs, len(src["faces"])) if target is None else target
    state["applied"] = ""
    lines = _guide_lines(context, gs, src)
    drawn = {kind: _guide_lines(context, gs, src, kind) for kind in ("LOOP", "BEND")}
    paint = _painted(state_source(), gs, src)
    # Each kind of face is one material and UV island: quads don't cross between kinds, and borders are reduced last.
    charts = _uv_charts(state_source(), src) if gs.keep_uvs and gs.bake_detail != "NORMAL_COLOR" else None
    codes = src["mat"] if charts is None else charts[0]
    settings = effort_settings(gs)
    baked = gs.auto_baked and gs.auto_profile == "LOW_POLY"
    chords = gs.auto_chords and not baked and loose_pieces(src) <= CHORDS_MOST_PIECES
    key = (src["sig"], "AUTO", settings.name, round(gs.auto_detail, 3), target, gs.sym_axis, gs.sym_force, gs.sym_side, gs.sym_offset_use and round(gs.sym_offset, 6), tuple(tweaks(gs).values()), hash(b"".join(line.tobytes() for line in lines)), None if paint is None else (hash(paint.tobytes()), gs.density_strength), charts is None, gs.auto_tidy, gs.auto_redo_rough, chords, gs.auto_method, gs.auto_output, gs.auto_style, gs.auto_rough, gs.auto_drop_hidden, baked, gs.region_use and state["region"] is not None and hash(state["region"].tobytes()), hash(b"".join(line.tobytes() for found in drawn.values() for line in found)), gs.loop_count, gs.bend_loops, gs.loop_gap, tuple(hard_surface.job_flags(gs).values()))
    state["curve_sig"], state["curve_at"] = _curve_sig(gs), 0.0
    if state["work"] is not None and state["work"]["key"] == key:
        return True
    stop_work()
    ladder.cancel()
    state["auto_base"] = None
    inputs = (target, lines, drawn, paint, charts, codes, settings, baked, chords)
    state["auto_inputs"] = inputs
    job = make_job(context, gs, src, inputs)
    state["work"] = {"key": key, "runner": job, "cent": None, "seen": 0, "areas": [], "charts": charts}
    state.update(ms=0.0, spot=None)
    state["message"] = job.status()
    if not bpy.app.timers.is_registered(_auto_slice):
        bpy.app.timers.register(_auto_slice, first_interval=0.0)
    return True


def tweaks(gs) -> dict:
    """The Tweak sliders, as the search takes them."""
    return {"rows": round(gs.tweak_rows, 3), "flat": round(gs.tweak_flat, 3), "poles": int(gs.tweak_poles)}


def make_job(context, gs, src, inputs, detail: float | None = None, style: str | None = None, quiet: bool = False):
    """The search job for these inputs (what `_start_segments` gathered). The size ladder asks for others at its own
    Detail and Style, `quiet`: no helper Blenders and no saved progress, so it never competes with what you do."""
    target, lines, drawn, paint, charts, codes, settings, baked, chords = inputs
    detail = (3.0 if baked else gs.auto_detail) if detail is None else detail
    half, note = _mirror_half(gs, src, codes)
    co, faces, mat, mirror = half[:4] if half is not None else (src["co"], src["faces"], codes, None)
    outside = None
    region = state["region"] if REGION_READY and gs.region_use and state["region"] is not None and len(state["region"]) == len(src["faces"]) else None
    if region is not None and region.any() and not region.all():
        half, mirror, paint = None, None, None
        (co, faces, mat), outside = edgeloops.split_region(src["co"], src["faces"], codes, region)
        note = f"Only the {int(region.sum()):,} selected faces are redone"
    if paint is not None and half is not None:
        if state.get("half_paint", (None,))[0] != (id(paint), id(half)):  # every size of the ladder asks again
            state["half_paint"] = ((id(paint), id(half)), _half_paint(paint, src, half), paint, half)  # kept: the ids stay theirs
        paint = state["half_paint"][1]
    if lines and mirror:
        lines = guides.reflect_to_side(lines, mirror[0], mirror[1], mirror[4])
    if mirror:
        drawn = {kind: guides.reflect_to_side(found, mirror[0], mirror[1], mirror[4]) for kind, found in drawn.items()}
    bvh = _bvh(src)
    items = [(line, gs.loop_count if kind == "LOOP" else gs.bend_loops, kind) for kind, found in drawn.items() for line in found]
    strips = (items, gs.loop_gap, lambda p: bvh.find_nearest(Vector(p))[:2]) if items else None
    cache_key = (src["sig"], (mirror[0], mirror[1], mirror[4]) if mirror else None, charts is None)
    if state["auto_cache"][0] != cache_key:
        state["auto_cache"] = (cache_key, {})  # a different mesh: what was measured on the old one is no use
    saved = None if quiet else hardware.checkpoint_dir(co, len(faces))
    resume = len(list(saved.glob("*.npz"))) if saved is not None else 0
    job = autoretopo.AutoJob(context, co, faces, mat, detail, target // 2 if mirror and target else target, state["auto_cache"][1], mirror, note, lines, lambda p: bvh.find_nearest(Vector(p))[0], None if paint is None else density_drop(paint, gs.density_strength), tidy=gs.auto_tidy, chords=chords, method="AUTO" if baked else gs.auto_method, output="QUADS" if baked else gs.auto_output, strips=strips, outside=outside, style=style or gs.auto_style, settings=settings, checkpoint=saved, rough=gs.auto_rough, drop_hidden=gs.auto_drop_hidden, baked=baked, parallel=gs.auto_parallel and not quiet, helpers=parallel.allowed(), redo_rough=gs.auto_redo_rough, tweak=tweaks(gs), hard=hard_surface.job_flags(gs))
    if gs.auto_chords and not chords and not baked:
        job.note = (job.note + " | " if job.note else "") + f"Redraw Messy Quads was left off: {loose_pieces(src):,} loose pieces, where it stalls"
    if resume:
        job.note = (job.note + " | " if job.note else "") + f"{resume} finished parts from an earlier run will be reused"
    return job


def _start_blockout(gs, src) -> bool:
    """Blockout or Hulls: each loose part becomes low-poly solids. In a LOD set each level halves the sides."""
    level = len(state["lod"]["made"]) if state["lod"] is not None else 0
    sides = max(4, gs.blockout_sides >> level)
    key = (src["sig"], "AUTO", gs.auto_profile, sides)
    state["applied"] = ""
    if state["work"] is not None and state["work"]["key"] == key:
        return True
    stop_work()
    state["auto_base"] = None
    job = blockout.BlockoutJob(src["co"], src["faces"], src["mat"], hull=gs.auto_profile == "HULL", sides=sides, settings=effort_settings(gs))
    state["work"] = {"key": key, "runner": job, "cent": None, "seen": 0, "areas": [], "charts": None}
    state.update(ms=0.0, spot=None, message=job.status())
    if not bpy.app.timers.is_registered(_auto_slice):
        bpy.app.timers.register(_auto_slice, first_interval=0.0)
    return True


def _face_areas(co: np.ndarray, faces) -> np.ndarray:
    tris = _fan(faces)
    area = 0.5 * np.linalg.norm(np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]]), axis=1)
    return np.bincount(np.repeat(np.arange(len(faces)), [len(f) - 2 for f in faces]), area, len(faces))


def _show_base(context, source) -> None:
    """Show the finished Auto Segments result again, with nothing redone."""
    stop_work()
    out = state["auto_base"]["out"]
    state["res"] = out
    _build_preview(context, source, out, uvs=False)
    state["draw_key"] = None


def _shape_mask(context, source, base, shapes) -> np.ndarray:
    """Faces of the base result inside the shapes. A shape works over the result and over your original mesh alike."""
    depsgraph = context.evaluated_depsgraph_get()
    frames = [source.matrix_world]
    preview = bpy.data.objects.get(PREVIEW_NAME)
    if preview is not None:
        frames.append(preview.matrix_world)
    local = _centroids(base["out"]["co"], base["out"]["faces"])
    entries = []
    for operation, shape in shapes:
        inside = np.zeros(len(local), dtype=bool)
        for matrix in frames:
            inside |= region.volume_mask(shape, _world(matrix, local), depsgraph)
        entries.append((operation, inside))
    return patch.combine_masks(len(local), entries)


def _start_boolean(context, gs, source, src, coarser: float = 1.0) -> bool:
    """Step 4: redo the faces inside the shapes, from the finished Auto Segments result."""
    base = state["auto_base"]
    if base is None or base["sig"] != src["sig"]:
        state["auto_base"] = None
        stop_work()
        state["message"] = "Your mesh changed: press Start again"
        return False
    shapes = [(s.operation, s.obj) for s in gs.bool_shapes if s.obj is not None and s.obj != source]
    work = state["work"]
    if not (gs.bool_on and shapes):
        if work is None or work["key"][1] == "BOOL":
            _show_base(context, source)
        state["message"] = "Add a shape over a part to redo it" if gs.bool_on else base["summary"]
        return True
    out = base["out"]
    co, faces, mat = out["co"], out["faces"], out["mat"]
    frozen = np.zeros(len(faces), dtype=bool)
    key = (src["sig"], "BOOL", base["id"], gs.fit_iterations)
    if work is None or work["key"] != key:
        stop_work()
        project = region._projector(_bvh(src))
        iterations = gs.fit_iterations
        runner = areas.AreaRunner(
            co, faces, frozen,
            fit=(lambda r: region._fit_result(co, r, project, iterations)) if iterations > 0 else None,
            merge=lambda results: _merge_results(co, faces, results, frozen, mat, no_bigger=True) or dict(out, new=np.zeros(len(faces), dtype=bool)),
            max_faces=region.MAX_PATCH_FACES,
        )
        work = state["work"] = {"key": key, "runner": runner, "cent": None, "seen": 0, "areas": []}
    mask = _shape_mask(context, source, base, shapes)
    work["note"] = "" if mask.any() else "The shapes cover no faces: move one over a part"
    if base.get("area") is None:
        base["area"] = _face_areas(co, faces)
    jobs = []
    if mask.any():
        edge = float(np.sqrt(base["area"][mask].mean()))
        # Never finer than the quads already there, so a redo does not add faces; Reduce To scales it by the share asked for.
        report = base["report"]
        share = min(1.0, report["asked"] / max(report["before"], 1)) if report["asked"] else 1.0
        edge = coarser * max(edge * 1.2 * 2.5 ** (1.0 - 2.0 * gs.bool_detail), edge) / math.sqrt(max(share, 0.01))
        work["coarser"] = coarser
        jobs = [areas.Job(mask, edge, gs.strategy, (0,))]
    state["message"] = work["note"] or "Redoing the part inside the shapes..."
    work["runner"].start(jobs)
    return True


def _auto_slice():
    """Run the search in short slices with gaps between them, so Blender stays usable. Ends when the job does."""
    work = state["work"]
    if work is None or not isinstance(work["runner"], (autoretopo.AutoJob, blockout.BlockoutJob, segments.PartJob)) or not work["runner"].busy:
        return None
    work["runner"].step()
    cfg = getattr(work["runner"], "cfg", None)  # a part redo has no effort settings of its own: it ran once and the timer died
    return cfg.gap_s if cfg is not None else 0.0


def _collect_work(context) -> None:
    """Show what the job finished since the last look."""
    segments.ensure()
    work = state["work"]
    source = state_source()
    if work is None or source is None:
        return
    runner = work["runner"]
    done, total = runner.progress
    if runner.output_id != work["seen"]:
        work["seen"] = runner.output_id
        if runner.output is None:
            remove_preview()
        else:
            res = _real_materials(runner.output, work.get("charts"))
            state["res"] = res
            _build_preview(context, source, res, uvs=False)  # the UVs are copied when you Apply
        state["draw_key"] = None
    res = state["res"]
    if work["key"][1] == "BOOL" and not runner.busy and res is not None and not res["new"].any() and not work.get("note") and work.get("coarser", 4.0) < 4.0:
        _start_boolean(context, _gs(context), source, _source_data(source), work["coarser"] * 1.6)  # too many faces for the room: try bigger quads
        return
    if isinstance(runner, segments.PartJob):
        state["message"] = runner.status()
        if not runner.busy and not runner.committed:
            segments.commit(runner)
        overlay.tag_redraw()
        return
    if isinstance(runner, (autoretopo.AutoJob, blockout.BlockoutJob)):
        state["message"] = runner.warnings[0] if runner.warnings else runner.status()
        base = state["auto_base"]
        if not runner.busy and runner.report is not None and runner.output is not None and (base is None or base["job"] is not runner):
            state["auto_base"] = {"id": time.perf_counter_ns(), "sig": work["key"][0], "key": work["key"], "out": _real_materials(runner.output, work.get("charts")), "charts": work.get("charts"), "report": runner.report, "job": runner, "summary": state["message"]}
            if state["lod"] is not None:
                _lod_finished(context, source, runner.report)
                overlay.tag_redraw()
                return
            if isinstance(runner, blockout.BlockoutJob):  # solids: no hand fixes to put back, no segments to find on them
                overlay.tag_redraw()
                return
            segments.ensure()
            _restore_kept(context, source)
            if state["res"] is not None and len(_library()) * len(state["res"]["faces"]) < 300_000:
                _scan(state["res"])
            segments.ensure()
            segments.segment_goals.start_find(context)
            ladder.start(context)
        elif not runner.busy and runner.report is None and state["lod"] is not None:  # the search failed: nothing more to come
            state["lod"] = None
        overlay.tag_redraw()
        return
    res = state["res"]
    made = f"{int(res['new'].sum()):,} new quads in {len(res['faces']):,} faces" if res is not None else "No quads yet"
    if res is not None and work["key"][1] == "BOOL" and not res["new"].any():
        made = "Nothing changed: a redo there would make more faces than it replaces. Slide towards Organized or use a bigger shape"
    if runner.busy:
        state["message"] = f"Working on piece {min(done + 1, max(total, 1))} of {max(total, 1)}. {made}"
    elif not work.get("settled") == runner.output_id:
        work["settled"] = runner.output_id
        state["message"] = work.get("note") or made + (f" | {runner.warnings[0]}" if runner.warnings else "")
        state["spot"] = res.get("spot") if res is not None else None
    overlay.tag_redraw()


def job_running() -> bool:
    """True while a job is still running in timer slices."""
    return state["work"] is not None and state["work"]["runner"].busy


def finish_work(context) -> bool:
    """Take in a finished job's last result (Apply). Never waits: False while a job is still running."""
    while state["work"] is not None:  # taking in one job's result can start the next (the search, then its segments)
        work = state["work"]
        if work["runner"].busy:
            return False
        _collect_work(context)
        if state["work"] is work:
            break
    return True


def run_update(context, force: bool = False) -> bool:
    """Recompute the preview now. Returns True when a preview was produced."""
    gs = context.scene.curiomesh_guided
    source = _source_mesh(context)
    if source is None:
        state["message"] = "Select a mesh first"
        return False
    state["source"] = source.name
    if source.mode == "EDIT":
        source.update_from_editmode()
    if gs.scope == "AUTO":
        if state["auto_base"] is not None:  # the search is done: this update is step 4, redoing parts inside the shapes
            return _start_boolean(context, gs, source, _source_data(source))
        return _start_segments(context, gs, _source_data(source))
    pts = gather(source, gs)
    lock = state["lock"]
    if not pts and lock is None:
        stop_work()
        remove_preview()
        state["message"] = "Place a point or press Auto-suggest"
        return False
    started = time.perf_counter()
    src = _source_data(source)
    L0 = float(np.sqrt(src["tri_area"].sum() / max(gs.base_faces, 1)))
    if gs.scope == "AREAS":
        return _start_areas(source, gs, pts, src, L0)
    stop_work()
    project = region._projector(_bvh(src))
    notes = ""
    res = None
    if lock is not None:
        mask = ~lock["locked"]
        if gs.scope == "POINTS" and not lock["free_all"]:
            mask &= region_sd(_centroids(lock["co"], lock["faces"]), pts, "POINTS") < 0
        if not mask.any():
            state["message"] = "Nothing left to remesh: all faces here are locked"
            return False
        res, warns = _region_pass(lock["co"], lock["faces"], mask, pts, L0, gs, project, lock["locked"], lock.get("mat"))
        notes = warns[0] if warns else ""
    else:
        if gs.scope == "POINTS":
            mask = region_sd(_centroids(src["co"], src["faces"]), pts, "POINTS") < 0
            if not mask.any():
                state["message"] = "Points Only needs a Detail or Loop point on the mesh"
                return False
            if not force and int(mask.sum()) > LIVE_REGION_CAP:
                state["message"] = f"Region has {int(mask.sum()):,} faces: press Update"
                return False
            if mask.mean() > 0.92:
                notes = "Region covers the mesh: used Whole Mesh"
            else:
                res, warns = _region_pass(src["co"], src["faces"], mask, pts, L0, gs, project, np.zeros(len(src["faces"]), bool), src["mat"])
                notes = warns[0] if warns else ""
        if res is None:
            qe = float(np.sum(src["tri_area"] / (L0 * multiplier(src["tri_cent"], pts)) ** 2))
            if qe > HARD_QUAD_CAP:
                state["message"] = f"Would make {int(qe):,} quads: lower Base Quads or the Detail strength"
                return False
            if not force and (qe > LIVE_QUAD_CAP or len(src["faces"]) > LIVE_SOURCE_FACES):
                state["message"] = f"About {int(qe):,} quads: press Update"
                return False
            res, extra = _whole_mesh(context, source, src, pts, gs, L0, project)
            notes = notes or extra
    if res is None:
        state["message"] = notes or "Nothing could be remeshed"
        return False
    state["res"] = res
    state["L0"] = L0
    _build_preview(context, source, res)
    sizes = [len(f) for f in res["faces"]]
    state["ms"] = (time.perf_counter() - started) * 1000.0
    state["spot"] = None
    try:
        spot = next_spot(res, project, L0)
        state["spot"] = {"pos": spot[0], "text": spot[1]} if spot else None
    except Exception as exc:  # the hint must never break an update
        notes = notes or f"hint: {exc}"
    summary = f"{len(sizes):,} faces, {sizes.count(4) / max(len(sizes), 1) * 100:.0f}% quads"
    if not res["new"].all():  # the kept faces are whatever the source had: count only the new ones
        summary = f"{int(res['new'].sum()):,} new quads in {len(sizes):,} faces"
    state["message"] = f"Preview: {summary}, {state['ms'] / 1000.0:.1f} s" + (f" | {notes}" if notes else "")
    state["draw_key"] = None
    overlay.tag_redraw()
    return True


def _start(context) -> None:
    obj = context.object
    if obj is not None and obj.type == "MESH" and obj.name != PREVIEW_NAME:
        state["source"] = obj.name
    state.update(sig=None, dirty=True, changed_at=0.0)
    gs = context.scene.curiomesh_guided
    src_obj = state_source()
    if src_obj is not None and gs.new_radius <= 0:
        gs.new_radius = PRESETS[gs.preset]["radius"] * _source_data(src_obj)["diag"]
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_SECONDS)
    overlay.tag_redraw()


def _stop() -> None:
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    stop_work()
    ladder.cancel()
    remove_preview()
    state.update(sig=None, message="", lock=None, hover=None, auto_base=None, auto_cache=(None, {}), lod=None)
    segments.reset()
    overlay.tag_redraw()


def _poll():
    try:
        context = bpy.context
        gs = getattr(context.scene, "curiomesh_guided", None)
        if gs is None or not gs.enabled:
            return None
        active = context.object
        if active is not None and active.type == "MESH" and active.name not in (PREVIEW_NAME, state["source"]) and active.mode == "OBJECT":
            state.update(source=active.name, lock=None, sig=None, auto_base=None)
            stop_work()
            ladder.cancel()
            remove_preview()
        source = state_source()
        preview = bpy.data.objects.get(PREVIEW_NAME)
        if source is None or source.mode != "OBJECT" or (preview is not None and preview.mode != "OBJECT"):
            return POLL_SECONDS
        _collect_work(context)
        if gs.scope != "AUTO":
            state["auto_base"] = None
        elif gs.auto_update and state["curve_sig"] is not None:  # a curve was moved, redrawn or deleted: the search starts again, quickly (what was measured is kept)
            now = time.perf_counter()
            if _curve_sig(gs) != state["curve_sig"]:
                if state["curve_at"] == 0.0:
                    state["curve_at"] = now
                elif now - state["curve_at"] >= 0.8:
                    state["curve_sig"], state["curve_at"] = None, 0.0
                    state["auto_base"] = None
                    run_update(context, force=True)
            else:
                state["curve_at"] = 0.0
        sig = _signature(context, source)
        now = time.perf_counter()
        if sig != state["sig"]:
            state.update(sig=sig, changed_at=now, dirty=True)
            overlay.tag_redraw()
        elif state["dirty"] and now - state["changed_at"] >= (AREA_DEBOUNCE_SECONDS if gs.scope in {"AREAS", "AUTO"} else DEBOUNCE_SECONDS):
            state["dirty"] = False
            if gs.scope == "AUTO":  # Start is its own button; only step 4 follows your edits
                if state["auto_base"] is not None:
                    run_update(context)
            elif gs.auto_update and (gs.scope == "AREAS" or state["ms"] <= LIVE_SLOW_MS):
                run_update(context)
            elif gs.auto_update:
                state["message"] = f"The last update took {state['ms'] / 1000.0:.0f} s: press Update"
            overlay.tag_redraw()
    except Exception as exc:
        state["message"] = f"Guided error: {exc}"[:120]
        state["dirty"] = False
    return POLL_SECONDS


# --------------------------------------------------------------------------- drawing


def _unit_sphere(lat=10, lon=16) -> np.ndarray:
    out = []
    for i in range(lat):
        t0, t1 = np.pi * i / lat, np.pi * (i + 1) / lat
        for j in range(lon):
            p0, p1 = 2 * np.pi * j / lon, 2 * np.pi * (j + 1) / lon
            q = [np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]) for t, p in ((t0, p0), (t0, p1), (t1, p1), (t1, p0))]
            out += [q[0], q[1], q[2], q[0], q[2], q[3]]
    return np.asarray(out)


_SPHERE = None


def _ring(center, normal, r, segments=48) -> np.ndarray:
    n = np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    a = np.cross(n, (1, 0, 0) if abs(n[0]) < 0.9 else (0, 1, 0))
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.asarray(center, float) + r * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))
    return np.stack((ring, np.roll(ring, -1, axis=0)), axis=1).reshape(-1, 3)


def _outline(src, pts, scope) -> np.ndarray:
    """Segments where the region boundary crosses the source surface."""
    co, tris = src["co"], src["tris"]
    sd = region_sd(co, pts, scope)
    inside = sd < 0
    mixed = inside[tris].any(axis=1) & ~inside[tris].all(axis=1)
    t = tris[mixed]
    if not len(t):
        return np.zeros((0, 3))
    points, flags = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = t[:, a], t[:, b]
        cross = inside[va] != inside[vb]
        da, db = sd[va], sd[vb]
        den = np.where(cross, da - db, 1.0)
        den = np.where(np.isfinite(den), den, 1.0)
        u = np.where(cross, np.clip(np.nan_to_num(da, posinf=0.0, neginf=0.0) / den, 0, 1), 0.0)
        points.append(co[va] + (co[vb] - co[va]) * u[:, None])
        flags.append(cross)
    segs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        pick = flags[i] & flags[j]
        segs.append(np.stack((points[i][pick], points[j][pick]), axis=1).reshape(-1, 3))
    return np.concatenate(segs)


def _colors():
    prefs = overlay.get_prefs()
    kinds = {k: overlay.pref_color(prefs, f"guided_{k.lower()}_color") for k in KIND_KEYS}
    return kinds, overlay.pref_color(prefs, "guided_outline_color"), overlay.pref_color(prefs, "guided_lock_color"), overlay.pref_color(prefs, "region_seam_color")


def _world(matrix, arr):
    m = np.asarray(matrix, dtype=np.float64)
    return np.asarray(arr, float) @ m[:3, :3].T + m[:3, 3]


def _draw_data(source, gs):
    """The arrays `_draw` shows. Built by overlay.drawn_arrays' one-shot timer when something changed, never in the draw."""
    global _SPHERE
    pts = gather(source, gs)
    preview = bpy.data.objects.get(PREVIEW_NAME)
    if _SPHERE is None:
        _SPHERE = _unit_sphere()
    m = source.matrix_world
    scale = float(np.mean(m.to_scale()))
    data = {"spheres": {}, "rings": {}, "dots": {}, "outline": np.zeros((0, 3), np.float32), "lock": np.zeros((0, 3), np.float32), "spot": None}
    for kind in KIND_KEYS:
        mine = [p for p in pts if p.kind == kind]
        if mine:
            data["spheres"][kind] = np.concatenate([_world(m, _SPHERE * p.r + p.co) for p in mine]).astype(np.float32)
            rings = [_world(m, _ring(p.co, p.nrm, p.r)) for p in mine]
            rings += [_world(m, _ring(p.co, p.nrm, 0.5 * p.r)) for p in mine if p.kind == "LOOP"]
            if gs.scope == "AREAS" and preview is not None:  # the same areas on the result, side by side
                rings += [_world(preview.matrix_world, _ring(p.co, p.nrm, p.r)) for p in mine]
            data["rings"][kind] = np.concatenate(rings).astype(np.float32)
            data["dots"][kind] = _world(m, [p.co for p in mine]).astype(np.float32)
    if pts and gs.show_cage:
        seg = _outline(_source_data(source), pts, gs.scope)
        data["outline"] = _world(m, seg).astype(np.float32) if len(seg) else seg.astype(np.float32)
    res = state["res"]
    if res is not None and preview is not None and res["locked"].any():
        tri = [(f[0], f[k], f[k + 1]) for f, lk in zip(res["faces"], res["locked"]) if lk for k in range(1, len(f) - 1)]
        data["lock"] = _world(preview.matrix_world, res["co"][np.asarray(tri).reshape(-1)]).astype(np.float32)
    if state["spot"] is not None and preview is not None:
        c = _world(preview.matrix_world, [state["spot"]["pos"]])[0]
        r = 1.6 * state["L0"] * scale
        up = np.array([0.0, 0.0, 1.0])
        side = np.array([1.0, 0.0, 0.0])
        data["spot"] = np.concatenate((_ring(c, up, r), _ring(c, side, r))).astype(np.float32)
    return data


def _draw() -> None:
    try:
        gs = getattr(bpy.context.scene, "curiomesh_guided", None)
        source = state_source()
        if gs is None or not gs.enabled or source is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        if state["draw_key"] is None:  # something set it to None: the arrays are built again
            state["draw_key"] = time.perf_counter_ns()
        key = (state["sig"], id(state["res"]), id(state["spot"]), gs.scope, gs.show_cage, state["draw_key"], source.name)
        data = overlay.drawn_arrays("guided", key, lambda: _draw_data(source, gs))
        if data is None:
            return
        if state["draw"] is not data:
            state["draw"], state["batches"] = data, None
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        colors, outline, lock_color, spot_color = _colors()

        def batch(kind, arr):
            return batch_for_shader(shader, kind, {"pos": arr}) if arr is not None and len(arr) else None

        if state["batches"] is None:
            state["batches"] = {
                "spheres": {k: batch("TRIS", v) for k, v in data["spheres"].items()},
                "rings": {k: batch("LINES", v) for k, v in data["rings"].items()},
                "dots": {k: batch("POINTS", v) for k, v in data["dots"].items()},
                "outline": batch("LINES", data["outline"]),
                "lock": batch("TRIS", data["lock"]),
                "spot": batch("LINES", data["spot"]),
            }
        b = state["batches"]
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_mask_set(False)
        shader.bind()
        if gs.show_cage:
            gpu.state.depth_test_set("LESS_EQUAL")
            for kind, bt in b["spheres"].items():
                if bt is not None:
                    shader.uniform_float("color", (*colors[kind][:3], 0.16))
                    bt.draw(shader)
            gpu.state.line_width_set(2.0)
            for kind, bt in b["rings"].items():
                if bt is not None:
                    shader.uniform_float("color", colors[kind])
                    bt.draw(shader)
            if b["outline"] is not None:
                gpu.state.depth_test_set("ALWAYS")
                gpu.state.line_width_set(3.5)
                shader.uniform_float("color", outline)
                b["outline"].draw(shader)
        gpu.state.depth_test_set("ALWAYS")
        gpu.state.point_size_set(9.0)
        for kind, bt in b["dots"].items():
            if bt is not None:
                shader.uniform_float("color", colors[kind])
                bt.draw(shader)
        hv = state["hover"]
        if hv is not None:
            gpu.state.line_width_set(2.5)
            shader.uniform_float("color", colors[hv["kind"]])
            batch("LINES", _world(source.matrix_world, _ring(hv["co"], hv["nrm"], hv["r"])).astype(np.float32)).draw(shader)
            preview = bpy.data.objects.get(PREVIEW_NAME)
            if preview is not None:
                batch("LINES", _world(preview.matrix_world, _ring(hv["co"], hv["nrm"], hv["r"])).astype(np.float32)).draw(shader)
        work = state["work"]
        running = work["runner"].current if work is not None and work["runner"].busy else None
        if running:  # the area the algorithm is on right now: a pulsing ring on both meshes
            pulse = 0.55 + 0.45 * float(np.sin(time.perf_counter() * 6.0))
            gpu.state.depth_test_set("ALWAYS")
            gpu.state.line_width_set(4.0)
            shader.uniform_float("color", (*spot_color[:3], pulse))
            preview = bpy.data.objects.get(PREVIEW_NAME)
            for i in running:
                if i < len(work["areas"]):
                    p = work["areas"][i]
                    for owner in (source, preview):
                        if owner is not None:
                            batch("LINES", _world(owner.matrix_world, _ring(p.co, p.nrm, p.r)).astype(np.float32)).draw(shader)
        if b["lock"] is not None:
            gpu.state.depth_test_set("LESS_EQUAL")
            shader.uniform_float("color", lock_color)
            b["lock"].draw(shader)
        if b["spot"] is not None:
            gpu.state.depth_test_set("ALWAYS")
            gpu.state.line_width_set(3.0)
            shader.uniform_float("color", spot_color)
            b["spot"].draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
    except Exception:
        overlay.draw_failed("guided._draw")


# --------------------------------------------------------------------------- operators


def _gs(context):
    return context.scene.curiomesh_guided


def surface_hit(obj, origin: Vector, direction: Vector):
    """Raycast in world space. (local location, local normal) or None."""
    inv = obj.matrix_world.inverted()
    ok, loc, normal, _i = obj.ray_cast(inv @ origin, (inv.to_3x3() @ direction).normalized())
    return (np.asarray(loc, float), np.asarray(normal, float)) if ok else None


class _SurfaceTool:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.object
        return o is not None and o.type == "MESH" and o.name != PREVIEW_NAME and o.mode == "OBJECT" and context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        gs = _gs(context)
        self.obj = context.object
        state["source"] = self.obj.name
        if gs.new_radius <= 0:
            gs.new_radius = PRESETS[gs.preset]["radius"] * _source_data(self.obj)["diag"]
        if not gs.enabled:
            gs.enabled = True
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _status(self, context):
        gs = _gs(context)
        context.workspace.status_text_set(
            f"Place Points [{gs.new_kind.title()}]  Click: add  Ctrl+Click: remove  Scroll: radius {gs.new_radius:.3g}  "
            f"Shift+Scroll: strength {gs.new_strength:.1f}  1-4: type  Enter/Esc: done"
        )

    def _view_region(self, context, event):
        """The 3D view under the mouse. None over the sidebar, a header or another area, which keep working."""

        def under_mouse(r):
            return r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height

        area = context.area
        if area is None or area.type != "VIEW_3D" or not under_mouse(area):
            return None
        if any(r.type != "WINDOW" and r.width > 1 and r.height > 1 and under_mouse(r) for r in area.regions):
            return None
        return next((r for r in area.regions if r.type == "WINDOW" and under_mouse(r)), None)

    def _ray(self, region_, event):
        from bpy_extras import view3d_utils

        coord = (event.mouse_x - region_.x, event.mouse_y - region_.y)
        return surface_hit(self.obj, view3d_utils.region_2d_to_origin_3d(region_, region_.data, coord), view3d_utils.region_2d_to_vector_3d(region_, region_.data, coord))

    def _finish(self, context):
        context.workspace.status_text_set(None)
        state["hover"] = None
        overlay.tag_redraw()

    def modal(self, context, event):
        gs = _gs(context)
        if context.area is not None:
            context.area.tag_redraw()
        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        view = self._view_region(context, event)
        if view is None:
            state["hover"] = None
            return {"PASS_THROUGH"}
        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        if event.type == "MOUSEMOVE":
            hit = self._ray(view, event)
            state["hover"] = {"co": hit[0], "nrm": hit[1], "r": gs.new_radius, "kind": gs.new_kind} if hit else None
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and not event.alt and not event.ctrl:
            up = event.type == "WHEELUPMOUSE"
            if event.shift:
                gs.new_strength = float(np.clip(gs.new_strength + (0.1 if up else -0.1), 0.1, 4.0))
            else:
                gs.new_radius = gs.new_radius * (1.1 if up else 1 / 1.1)
            if state["hover"]:
                state["hover"]["r"] = gs.new_radius
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type in {"ONE", "TWO", "THREE", "FOUR"} and event.value == "PRESS":
            gs.new_kind = KIND_KEYS[("ONE", "TWO", "THREE", "FOUR").index(event.type)]
            if state["hover"]:
                state["hover"]["kind"] = gs.new_kind
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = self._ray(view, event)
            if hit is None:
                return {"RUNNING_MODAL"}
            if event.ctrl:
                pts = self.obj.curiomesh_points
                near = [(float(np.linalg.norm(np.asarray(p.co) - hit[0])), i) for i, p in enumerate(pts) if np.linalg.norm(np.asarray(p.co) - hit[0]) <= p.radius]
                if near:
                    pts.remove(min(near)[1])
                    self.obj.curiomesh_point_index = max(0, min(self.obj.curiomesh_point_index, len(pts) - 1))
            else:
                add_point(self.obj, hit[0], hit[1], gs.new_kind, gs.new_radius, gs.new_strength)
            return {"RUNNING_MODAL"}
        if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._finish(context)


class CURIOMESH_OT_guided_place(_SurfaceTool, bpy.types.Operator):
    bl_idname = "curiomesh.guided_place"
    bl_label = "Place Points"
    bl_description = (
        "Click the surface to add a point. Ctrl+click removes one, scroll sets the radius, Shift+scroll the strength, "
        "keys 1-4 pick Detail / Loop / Sharp / Calm, Enter or Esc finishes"
    )


class CURIOMESH_OT_guided_work(_SurfaceTool, bpy.types.Operator):
    bl_idname = "curiomesh.guided_work"
    bl_label = "Work Live"
    bl_description = (
        "Click or drag on the mesh to add an area; it is remeshed right away. Over an area: scroll for size, Shift+scroll for density, F for fill style"
    )

    def invoke(self, context, event):
        _gs(context).scope = "AREAS"
        self.dab = None
        self.under = None
        return super().invoke(context, event)

    def _status(self, context):
        gs = _gs(context)
        p = self._area()
        kind, strength, fill, radius = (p.kind, p.strength, p.fill, p.radius) if p else (gs.new_kind, gs.new_strength, gs.strategy, gs.new_radius)
        head = f"Area {self.under + 1}" if p else "New area"
        context.workspace.status_text_set(
            f"{head}: {kind.title()} {strength:.2f}, {fill.title()}, size {radius:.3g}  |  Click/Drag: add  Ctrl+Click: remove  "
            "Scroll: size  Shift+Scroll: density  F: fill style  Enter/Esc: done"
        )

    def _area(self):
        pts = self.obj.curiomesh_points
        return pts[self.under] if self.under is not None and self.under < len(pts) else None

    def _ray(self, region_, event):
        hit = super()._ray(region_, event)
        preview = bpy.data.objects.get(PREVIEW_NAME)
        if hit is not None or preview is None:
            return hit
        from bpy_extras import view3d_utils

        coord = (event.mouse_x - region_.x, event.mouse_y - region_.y)
        try:  # same object-space shape as your mesh, only moved aside
            return surface_hit(preview, view3d_utils.region_2d_to_origin_3d(region_, region_.data, coord), view3d_utils.region_2d_to_vector_3d(region_, region_.data, coord))
        except RuntimeError:  # the result was swapped this very moment
            return None

    def _hover(self, context, hit):
        gs = _gs(context)
        self.under = None
        if hit is not None:
            near = [(float(np.linalg.norm(np.asarray(p.co) - hit[0])) / max(p.radius, 1e-9), i) for i, p in enumerate(self.obj.curiomesh_points)]
            near = [n for n in near if n[0] <= 1.0]
            self.under = min(near)[1] if near else None
        p = self._area()
        if self.under is not None:
            self.obj.curiomesh_point_index = self.under
        state["hover"] = {"co": hit[0], "nrm": hit[1], "r": p.radius if p else gs.new_radius, "kind": p.kind if p else gs.new_kind} if hit else None
        self._status(context)

    def _add(self, gs, hit):
        add_point(self.obj, hit[0], hit[1], gs.new_kind, gs.new_radius, gs.new_strength, fill=gs.strategy)
        self.dab = hit[0]

    def _brush_follows(self, gs, p):
        gs.new_kind, gs.new_strength, gs.new_radius, gs.strategy = p.kind, p.strength, p.radius, p.fill

    def modal(self, context, event):
        gs = _gs(context)
        if context.area is not None:
            context.area.tag_redraw()
        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self.dab = None
        view = self._view_region(context, event)
        if view is None:
            state["hover"] = None
            self.dab = None
            return {"PASS_THROUGH"}
        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        if event.type == "MOUSEMOVE":
            hit = self._ray(view, event)
            if self.dab is not None and hit is not None and np.linalg.norm(hit[0] - self.dab) > 0.75 * gs.new_radius:
                self._add(gs, hit)
            self._hover(context, hit)
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and not event.alt and not event.ctrl:
            up = event.type == "WHEELUPMOUSE"
            p = self._area()
            kind, strength, radius = (p.kind, p.strength, p.radius) if p else (gs.new_kind, gs.new_strength, gs.new_radius)
            if event.shift:
                kind, strength = stepped_intensity(kind, strength, up)
            else:
                radius *= 1.1 if up else 1 / 1.1
            if p:
                p.kind, p.strength, p.radius = kind, strength, radius
                self._brush_follows(gs, p)
            else:
                gs.new_kind, gs.new_strength, gs.new_radius = kind, strength, radius
            if state["hover"]:
                state["hover"].update(r=radius, kind=kind)
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type == "F" and event.value == "PRESS":
            p = self._area()
            fill = "GRID" if (p.fill if p else gs.strategy) == "FLOW" else "FLOW"
            if p:
                p.fill = fill
            gs.strategy = fill
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = self._ray(view, event)
            if hit is None:
                return {"RUNNING_MODAL"}
            self._hover(context, hit)
            if event.ctrl:
                if self.under is not None:
                    self.obj.curiomesh_points.remove(self.under)
                    self.obj.curiomesh_point_index = max(0, min(self.under, len(self.obj.curiomesh_points) - 1))
                    self.under = None
            else:
                self._add(gs, hit)
            return {"RUNNING_MODAL"}
        if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


class CURIOMESH_OT_guided_add_point(bpy.types.Operator):
    bl_idname = "curiomesh.guided_add_point"
    bl_label = "Add Point"
    bl_description = "Add an importance point at the 3D cursor, snapped to the surface of the active mesh"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Type", description="Which kind of point to add: Detail, Loop, Sharp or Calm", items=KIND_ITEMS, default="DETAIL")
    radius: bpy.props.FloatProperty(name="Radius", description="How far the point reaches, in scene units. 0 picks a size from the mesh type", default=0.0, min=0.0, subtype="DISTANCE")
    location: bpy.props.FloatVectorProperty(name="Location", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), description="World-space spot (0,0,0 uses the 3D cursor)")

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        obj = _source_mesh(context)
        gs = _gs(context)
        world = Vector(self.location) if any(self.location) else context.scene.cursor.location
        ok, loc, normal, _i = obj.closest_point_on_mesh(obj.matrix_world.inverted() @ world)
        if not ok:
            self.report({"ERROR"}, "No surface found")
            return {"CANCELLED"}
        radius = self.radius or gs.new_radius or PRESETS[gs.preset]["radius"] * _source_data(obj)["diag"]
        add_point(obj, loc, normal, self.kind, radius, gs.new_strength)
        return {"FINISHED"}


class _PointsOp:
    @classmethod
    def poll(cls, context):
        o = _source_mesh(context)
        return o is not None and len(o.curiomesh_points) > 0


class CURIOMESH_OT_guided_remove_point(_PointsOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_remove_point"
    bl_label = "Remove Point"
    bl_description = "Remove the point selected in the list"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        o = _source_mesh(context)
        o.curiomesh_points.remove(o.curiomesh_point_index)
        o.curiomesh_point_index = max(0, min(o.curiomesh_point_index, len(o.curiomesh_points) - 1))
        return {"FINISHED"}


class CURIOMESH_OT_guided_clear_points(_PointsOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_clear_points"
    bl_label = "Clear Points"
    bl_description = "Remove every point from the mesh (tagged empties stay)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _source_mesh(context).curiomesh_points.clear()
        return {"FINISHED"}


class CURIOMESH_OT_guided_add_empty(bpy.types.Operator):
    bl_idname = "curiomesh.guided_add_empty"
    bl_label = "Add Empty Point"
    bl_description = (
        "Add a sphere Empty at the 3D cursor as a point. Move and scale it like any object; its size is the radius. "
        "Custom properties curiomesh_kind and curiomesh_strength set the type and strength"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        obj = _source_mesh(context)
        gs = _gs(context)
        empty = bpy.data.objects.new("CM_Point", None)
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = gs.new_radius or PRESETS[gs.preset]["radius"] * _source_data(obj)["diag"]
        empty.location = context.scene.cursor.location
        empty[POINT_TAG] = 1
        empty["curiomesh_kind"] = gs.new_kind
        empty["curiomesh_strength"] = gs.new_strength
        (obj.users_collection[0] if obj.users_collection else context.scene.collection).objects.link(empty)
        return {"FINISHED"}


class CURIOMESH_OT_guided_suggest(bpy.types.Operator):
    bl_idname = "curiomesh.guided_suggest"
    bl_label = "Auto-suggest Points"
    bl_description = "Place points at high-curvature spots, hard edges and small holes (eyes, mouths). Edit them afterwards; suggesting again replaces the earlier suggestions"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        obj = _source_mesh(context)
        gs = _gs(context)
        for i in reversed(range(len(obj.curiomesh_points))):
            if obj.curiomesh_points[i].auto:
                obj.curiomesh_points.remove(i)
        found = suggest(obj, gs)
        for f in found:
            add_point(obj, f["co"], f["normal"], f["kind"], f["r"], gs.new_strength, auto=True)
        state["source"] = obj.name
        self.report({"INFO"}, f"Placed {len(found)} point(s). Edit them in the list or with Place Points")
        return {"FINISHED"}


class CURIOMESH_OT_guided_update(bpy.types.Operator):
    bl_idname = "curiomesh.guided_update"
    bl_label = "Update"
    bl_description = "Rebuild the preview now, even when it is too big for automatic updates"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        gs = _gs(context)
        obj = _source_mesh(context)
        state["source"] = obj.name
        if not gs.enabled:
            gs.enabled = True
        ok = run_update(context, force=True)
        state["sig"] = _signature(context, obj)
        state["dirty"] = False
        if not ok:
            self.report({"WARNING"}, state["message"])
        return {"FINISHED"} if ok else {"CANCELLED"}


class CURIOMESH_OT_guided_edit_preview(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_edit_preview"
    bl_label = "Select Faces on Preview"
    bl_description = "Enter Edit Mode on the preview so you can select faces to lock or a strip to redraw"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        preview = bpy.data.objects[PREVIEW_NAME]
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        preview.hide_select = False
        preview.select_set(True)
        context.view_layer.objects.active = preview
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        return {"FINISHED"}


def _preview_selection():
    preview = bpy.data.objects[PREVIEW_NAME]
    if preview.mode == "EDIT":
        preview.update_from_editmode()
    sel = np.zeros(len(preview.data.polygons), dtype=bool)
    preview.data.polygons.foreach_get("select", sel)
    return sel


def _leave_preview(context):
    if context.object is not None and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    source = state_source()
    if source is not None:
        bpy.ops.object.select_all(action="DESELECT")
        source.select_set(True)
        context.view_layer.objects.active = source


def _set_lock(res, locked, free_all=False):
    stop_work()
    state["lock"] = {"co": res["co"], "faces": res["faces"], "locked": locked, "mat": res.get("mat"), "free_all": free_all, "id": time.perf_counter_ns()}


class CURIOMESH_OT_guided_lock(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_lock"
    bl_label = "Lock Faces"
    bl_description = "Keep the faces selected on the preview exactly as they are; later updates rebuild only the rest, around them"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        sel = _preview_selection()
        res = state["res"]
        if len(sel) != len(res["faces"]) or not sel.any():
            self.report({"WARNING"}, "Select some preview faces in Edit Mode first")
            return {"CANCELLED"}
        _leave_preview(context)
        _set_lock(res, res["locked"] | sel, state["lock"]["free_all"] if state["lock"] else False)
        ok = run_update(context, force=True)
        state["sig"] = _signature(context, state_source())
        state["dirty"] = False
        self.report({"INFO"} if ok else {"WARNING"}, f"{int(state['lock']['locked'].sum()):,} faces locked. {state['message']}")
        return {"FINISHED"}


class CURIOMESH_OT_guided_unlock(bpy.types.Operator):
    bl_idname = "curiomesh.guided_unlock"
    bl_label = "Unlock All"
    bl_description = "Forget all locked faces and rebuild everything from your mesh"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return state["lock"] is not None

    def execute(self, context):
        state["lock"] = None
        if state_source() is not None and _gs(context).enabled:
            run_update(context, force=True)
        return {"FINISHED"}


class CURIOMESH_OT_guided_redraw_strip(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_redraw_strip"
    bl_label = "Redraw Strip"
    bl_description = "Rebuild only the faces selected on the preview (a strip or loop of faces) with the region tool. The rest of the preview is frozen afterwards"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        sel = _preview_selection()
        res = state["res"]
        gs = _gs(context)
        if len(sel) != len(res["faces"]) or not sel.any():
            self.report({"WARNING"}, "Select the strip of preview faces in Edit Mode first")
            return {"CANCELLED"}
        _leave_preview(context)
        source = state_source()
        co, faces = res["co"], res["faces"]
        edge = float(np.mean([np.linalg.norm(co[f[k]] - co[f[(k + 1) % len(f)]]) for i, f in enumerate(faces) if sel[i] for k in range(len(f))]))
        new, warns = _region_pass(co, faces, sel, [], state["L0"], gs, region._projector(_bvh(_source_data(source))), res["locked"], res.get("mat"), fixed_target=edge)
        if new is None:
            self.report({"WARNING"}, warns[0] if warns else "Could not redraw that strip")
            return {"CANCELLED"}
        _set_lock(new, ~new["new"], free_all=True)
        new["locked"] = state["lock"]["locked"]
        state["res"] = new
        _build_preview(context, source, new)
        state["draw_key"] = None
        state["sig"] = _signature(context, source)
        state["message"] = f"Strip rebuilt: {int(new['new'].sum())} new faces. The rest is frozen (Unlock All to start over)"
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_guided_frame(bpy.types.Operator):
    bl_idname = "curiomesh.guided_frame"
    bl_label = "Frame"
    bl_description = "Move the 3D view to the next spot worth fixing"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return state["spot"] is not None and bpy.data.objects.get(PREVIEW_NAME) is not None

    def execute(self, context):
        rv3d = getattr(context.space_data, "region_3d", None) or context.region_data
        if rv3d is None:
            return {"CANCELLED"}
        preview = bpy.data.objects[PREVIEW_NAME]
        rv3d.view_location = preview.matrix_world @ Vector(state["spot"]["pos"])
        rv3d.view_distance = max(12.0 * state["L0"] * float(np.mean(preview.matrix_world.to_scale())), 1e-3)
        return {"FINISHED"}


def _bake_result(source, out, gs, new_uvs: bool) -> str:
    """Bake on Apply. A failed bake leaves the result as it was: the quads are still good. Returns text for the report."""
    src = _source_data(source)
    co = np.empty(len(out.data.vertices) * 3, dtype=np.float64)
    out.data.vertices.foreach_get("co", co)
    bvh = _bvh(src)
    gaps = np.asarray([bvh.find_nearest(p)[3] for p in co.reshape(-1, 3).tolist()])
    # The rays must start outside both surfaces: past the widest gap between them, measured on this mesh.
    cage = (1.5 * float(np.percentile(gaps, 99)) + 0.002 * src["diag"]) * max(source.matrix_world.to_scale())
    for slot in out.material_slots:  # baking works on the mesh's own material list
        if slot.link == "OBJECT":
            material = slot.material
            slot.link = "DATA"
            slot.material = material
    color = gs.bake_detail == "NORMAL_COLOR"
    try:
        if color and gs.bake_sets > 1:
            piece = segments.seg["piece"]
            labels = piece if piece is not None and len(piece) == len(out.data.polygons) else None
            images = textures.bake_texture_sets(source, out, count=gs.bake_sets, labels=labels, resolution=int(gs.bake_size), cage=cage)
        else:
            images = textures.bake_detail(source, out, with_color=color, resolution=int(gs.bake_size), cage=cage, new_uvs=new_uvs)
    except Exception as exc:  # the bake has its own rollback, and the quads are worth keeping
        return f". Bake failed: {exc}"
    sets = f" in {len(images) // 2} texture sets" if color and len(images) > 2 else ""
    return f", {len(images)} textures baked at {gs.bake_size}{sets} (a Triangulate modifier keeps the bake's triangles on export)"


def _write_instances(out, source, instances) -> int:
    """Under `out`, link one object per placed copy of each repeated part. Returns how many objects that made."""
    made = 0
    for number, group in enumerate(instances["groups"]):
        mesh = _make_mesh(f"{source.name}_Part_{number}", source, group)
        _shade_by_angle(mesh)
        mesh.transform(Matrix.Translation(-Vector(group["centre"].tolist())))  # UVs come from the source, so read them in place first
        for k, matrix in enumerate(group["matrices"]):
            child = bpy.data.objects.new(f"{mesh.name}.{k}", mesh)
            for collection in out.users_collection:
                collection.objects.link(child)
            child.parent = out
            child.matrix_local = Matrix(matrix.tolist())
            for slot, copy in zip(source.material_slots, child.material_slots):
                if slot.link == "OBJECT":
                    copy.link = "OBJECT"
                    copy.material = slot.material
            made += 1
    return made


def _blockout_result() -> bool:
    """The result on show is solids from Blockout or Hulls: they shade as themselves, not like your mesh."""
    base = state["auto_base"]
    return base is not None and isinstance(base["job"], blockout.BlockoutJob)


def _write_result(context, source, res, gs, name: str):
    """A new object from a result, with UVs, normals, shading and materials as your mesh has them. Returns (object, bake text)."""
    mesh = _make_mesh(name, source, res)
    _shade_by_angle(mesh)
    if not _blockout_result():
        _transfer_normals(mesh, _source_data(source), res)
    if gs.bend_crease or gs.bend_sharp:
        _mark_edges(mesh, _bend_edges(state["auto_base"], res), gs.bend_crease, gs.bend_sharp)
    hard_surface.finish(mesh, gs, context.scene.unit_settings.scale_length)
    out = bpy.data.objects.new(name, mesh)
    (source.users_collection[0] if source.users_collection else context.scene.collection).objects.link(out)
    out.matrix_world = source.matrix_world.copy()
    for slot, copy in zip(source.material_slots, out.material_slots):
        if slot.link == "OBJECT":  # materials kept on the object, not in the mesh
            copy.link = "OBJECT"
            copy.material = slot.material
    baked = ""
    if gs.bake_detail != "OFF":
        baked = _bake_result(source, out, gs, res.get("chart") is None)  # without seams kept, the old UVs can stretch
    return out, baked


def _lod_next(context, source) -> None:
    """Start the next size of the LOD set. What the first one measured is reused, so only it does the full search."""
    gs = context.scene.curiomesh_guided
    src = _source_data(source)
    targets = profiles.lod_targets((gs.lod0_percent, gs.lod1_percent, gs.lod2_percent), len(src["faces"]))
    stop_work()
    state["auto_base"] = None
    _start_segments(context, gs, src, targets[len(state["lod"]["made"])])


def _lod_finished(context, source, report) -> None:
    """One size is done: write its object, and go on to the next or end the set."""
    gs = context.scene.curiomesh_guided
    lod = state["lod"]
    level = len(lod["made"])
    name = profiles.lod_name(source.name, level)
    old = bpy.data.objects.get(name)
    if old is not None and "curiomesh_lod" in old:  # made by this button before: the name must stay free for the engine's import
        bpy.data.objects.remove(old)
    out, baked = _write_result(context, source, state["res"], gs, name)
    out["curiomesh_lod"] = level
    lod["made"].append((out.name, report["faces"], report["loss"], baked))
    if level < profiles.LOD_LEVELS - 1:
        _lod_next(context, source)
        return
    made = lod["made"]
    gs.enabled = False  # removes the preview, stops the timer
    state["lod_report"] = " | ".join(f"{name}: {faces:,} faces, view changed by {loss:.1f}%{bake}" for name, faces, loss, bake in made)
    bpy.ops.object.select_all(action="DESELECT")
    first = bpy.data.objects[made[0][0]]
    first.select_set(True)
    context.view_layer.objects.active = first


class _AskFirst:
    """Mixin: a long job asks first, in a small dialog with the cheaper ways, unless this session was told not to ask."""

    dont_ask: bpy.props.BoolProperty(name="Don't ask again this session", default=False, options={"SKIP_SAVE"})

    def _question(self, context) -> list[str]:
        """Lines for the dialog, or an empty list when the job is short enough to just run."""
        return []

    def invoke(self, context, event):
        machine_info()
        lines = [] if state["no_ask"] or bpy.app.background else self._question(context)
        if not lines:
            return self.execute(context)
        self._lines = lines
        return context.window_manager.invoke_props_dialog(self, width=440)

    def draw(self, context):
        for i, line in enumerate(self._lines):
            overlay.wrap_label(self.layout, line, icon="TIME" if i == 0 else "BLANK1", width=60, limit=3)
        self.layout.prop(self, "dont_ask")

    def _remember(self) -> None:
        state["no_ask"] = state["no_ask"] or self.dont_ask


def _ask_lines(context, bakes: int = 1, applying: bool = False) -> list[str]:
    """What the long-job dialog says, or [] when the job is short: the time here, then what would make it cheaper."""
    gs = _gs(context)
    obj = _source_mesh(context)
    faces = len(obj.data.polygons) if obj is not None else 0
    level, text, _settings, seconds, bake_s = job_cost(gs, faces, state["src"]["sig"] if state["src"] is not None else None, bakes)
    if not hardware.needs_confirm(level, seconds, bake_s, applying):
        return []
    head = f"The bake takes {hardware.clock(bake_s)} on this PC and Blender is frozen until it ends." if applying else text
    return [head, "To make it cheaper first:"] + ["- " + way for way in hardware.cheaper_ways(gs.auto_style, gs.bake_detail, int(gs.bake_size), gs.auto_effort)]


class CURIOMESH_OT_guided_make_lods(_AskFirst, bpy.types.Operator):
    bl_idname = "curiomesh.guided_make_lods"
    bl_label = "Make LOD Set"
    bl_description = "Make three new objects named for game engines (_LOD0, _LOD1, _LOD2), each a Reduce To of your mesh at the percents here. Blender stays usable, and only the first needs the full search. Uses your UV and bake settings"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def _question(self, context):
        return _ask_lines(context, bakes=profiles.LOD_LEVELS)

    def execute(self, context):
        self._remember()
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        gs = _gs(context)
        source = _source_mesh(context)
        state["source"] = source.name
        stop_work()
        remove_preview()
        state.update(region=None, lod_report="", lod=None)
        gs["bool_on"] = False
        gs.scope = "AUTO"
        if not gs.enabled:
            gs.enabled = True
        state["lod"] = {"made": []}
        _lod_next(context, source)
        state["sig"] = _signature(context, source)
        state["dirty"] = False
        return {"FINISHED"}


class CURIOMESH_OT_guided_apply(_AskFirst, bpy.types.Operator):
    bl_idname = "curiomesh.guided_apply"
    bl_label = "Apply"
    bl_description = "Write the preview as <name>_Retopo (or into your object, see Result). Undo restores"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if job_running():
            cls.poll_message_set("Still working: wait for the job to finish or press Stop")
            return False
        return state["res"] is not None and state_source() is not None

    def _question(self, context):
        return _ask_lines(context, applying=True)

    def execute(self, context):
        with tasks.busy_cursor():  # writing the mesh, its UVs and hard edges takes a few seconds on a big result
            return self._apply(context)

    def _apply(self, context):
        self._remember()
        gs = _gs(context)
        source = state_source()
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if not finish_work(context):
            self.report({"WARNING"}, "Still working: wait for the job to finish or press Stop")
            return {"CANCELLED"}
        res = state["res"]
        if res is None:
            self.report({"WARNING"}, "Nothing to apply")
            return {"CANCELLED"}
        if gs.scope == "AUTO" and gs.blockout_separate and _blockout_result():
            made = [_write_result(context, source, part, gs, f"{source.name}_Part")[0] for part in blockout.split_result(res)]
            gs.enabled = False  # removes the preview, stops the timer
            bpy.ops.object.select_all(action="DESELECT")
            for obj in made:
                obj.select_set(True)
            context.view_layer.objects.active = made[0]
            state["applied"] = f"{len(res['faces']):,} faces written to {len(made):,} objects named {source.name}_Part"
            self.report({"INFO"}, state["applied"])
            return {"FINISHED"}
        # Keep Copies as Instances: one mesh per repeated part. Not with a bake or kept UV seams, which work on one mesh.
        instances = res.get("instances") if gs.scope == "AUTO" and gs.auto_instances and gs.bake_detail == "OFF" and res.get("chart") is None and not gs.multires_on else None
        if gs.auto_instances and gs.scope == "AUTO" and instances is None:
            self.report({"INFO"}, "Keep Copies as Instances was not used: this result has no exact copies, or it is baked, mirrored, tidied or repaired")
        written = instances["rest"] if instances else res
        mesh = _make_mesh(source.name + "_Retopo", source, written)
        _shade_by_angle(mesh)
        if not _blockout_result():
            _transfer_normals(mesh, _source_data(source), written)
        if gs.bend_crease or gs.bend_sharp:
            _mark_edges(mesh, _bend_edges(state["auto_base"], res), gs.bend_crease, gs.bend_sharp)
        if not (gs.multires_on and gs.scope == "AUTO"):  # Multires creases its own cage
            hard_surface.finish(mesh, gs, context.scene.unit_settings.scale_length)
        out = bpy.data.objects.new(source.name + "_Retopo", mesh)
        (source.users_collection[0] if source.users_collection else context.scene.collection).objects.link(out)
        out.matrix_world = source.matrix_world.copy()
        for slot, copy in zip(source.material_slots, out.material_slots):
            if slot.link == "OBJECT":  # materials kept on the object, not in the mesh
                copy.link = "OBJECT"
                copy.material = slot.material
        baked = ""
        multires_on = gs.multires_on and (gs.scope == "AUTO")
        flat_faces = None
        tree = multires.source_tree(source) if multires_on else None  # before Replace takes the source's mesh away
        if multires_on:
            _mark_edges(mesh, multires.hard_edges(mesh), True, False)
            flat_faces = segments.segment_goals.faces_without_detail(res)
        if gs.bake_detail != "OFF":  # first, while level 0 is all there is: a bake reads the levels too
            baked = _bake_result(source, out, gs, res.get("chart") is None)  # without seams kept, the old UVs can stretch
        flag = res.get("flag")
        if flag is not None and len(flag) == len(mesh.polygons) and flag.any():
            mesh.polygons.foreach_set("select", flag)  # the faces that could not be repaired, ready in Edit Mode
        if gs.apply_mode == "REPLACE":
            bpy.data.objects.remove(out)
            source.data = mesh
            out = source
            if baked:  # the baked materials live on the mesh
                for slot in out.material_slots:
                    slot.link = "DATA"
        count = len(mesh.polygons)
        levels = ""
        if multires_on:
            levels = multires.add_to_result(context, out, tree, gs.multires_levels, flat_faces)
        if segments.seg.get("reviewing"):  # Review Segments ends with the preview: its hover card and popup have nothing left to show
            segments.seg.update(reviewing=False, hover=None)
            context.workspace.status_text_set(None)
        gs.enabled = False  # removes the preview, stops the timer
        bpy.ops.object.select_all(action="DESELECT")
        out.select_set(True)
        context.view_layer.objects.active = out
        placed = _write_instances(out, source, instances) if instances else 0
        if instances:
            count += sum(len(group["faces"]) for group in instances["groups"])
            baked = f", in {placed + 1} objects sharing {len(instances['groups']) + 1} meshes"
        state["applied"] = f"{count:,} faces written to {out.name}{baked}{levels}"
        self.report({"WARNING"} if baked.startswith("Bake failed") else {"INFO"}, state["applied"])
        return {"FINISHED"}


class CURIOMESH_OT_guided_stop(bpy.types.Operator):
    bl_idname = "curiomesh.guided_stop"
    bl_label = "Cancel"
    bl_description = "Remove the preview and keep your mesh as it was. Your points and guides stay"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        _leave_preview(context)
        _gs(context).enabled = False
        return {"FINISHED"}


class CURIOMESH_OT_guided_auto_start(_AskFirst, bpy.types.Operator):
    bl_idname = "curiomesh.guided_auto_start"
    bl_label = "Start"
    bl_description = "Start Auto Segments on the selected mesh. A bar shows the progress and Blender stays usable. Starting again throws away the areas you fixed"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def _question(self, context):
        return _ask_lines(context)

    def execute(self, context):
        self._remember()
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        gs = _gs(context)
        obj = _source_mesh(context)
        state["source"] = obj.name
        stop_work()
        remove_preview()
        state["auto_base"] = None
        state["region"] = None
        if gs.region_use:
            picked = np.zeros(len(obj.data.polygons), dtype=bool)
            obj.data.polygons.foreach_get("select", picked)
            state["region"] = picked[_source_data(obj)["keep"]] if picked.any() else None
        gs["bool_on"] = False
        gs.scope = "AUTO"
        if not gs.enabled:
            gs.enabled = True
        ok = run_update(context, force=True)
        state["sig"] = _signature(context, obj)
        state["dirty"] = False
        return {"FINISHED"} if ok else {"CANCELLED"}


class CURIOMESH_OT_guided_check_pc(bpy.types.Operator):
    bl_idname = "curiomesh.guided_check_pc"
    bl_label = "Check This Computer"
    bl_description = "Read your cores and memory, time a short test, and count the loose pieces of the selected mesh, so the panel can say how big this job is for this PC"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        machine_info(refresh=True)
        obj = _source_mesh(context)
        if obj is not None:
            loose_pieces(_source_data(obj))
        return {"FINISHED"}


class CURIOMESH_OT_guided_export_job(bpy.types.Operator):
    bl_idname = "curiomesh.guided_export_job"
    bl_label = "Export Job For Another Computer"
    bl_description = "Write a folder with this mesh, your settings and a script, to run the same search on a stronger computer. You copy it there; nothing is uploaded"
    bl_options = {"INTERNAL"}

    directory: bpy.props.StringProperty(name="Folder", description="The folder the job is written into", subtype="DIR_PATH")

    @classmethod
    def poll(cls, context):
        if _gs(context).auto_profile in profiles.BLOCKOUT_USES:  # the job script runs the reducing search only
            cls.poll_message_set("Blockout and Hulls are quick: press Start here")
            return False
        return _source_mesh(context) is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        gs = _gs(context)
        obj = _source_mesh(context)
        src = _source_data(obj)
        try:
            folder = jobpack.job_folder(self.directory, obj.name)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        target = auto_target(gs, len(src["faces"]))
        co, faces, mat, mirror, lines, paint, _note = _solve_inputs(gs, src, _guide_lines(context, gs, src), _painted(obj, gs, src))
        jobpack.export_job(
            folder, co, faces, mat, gs.auto_detail, target // 2 if mirror and target else target, "FULL",
            mirror=mirror, lines=lines, drop=None if paint is None else density_drop(paint, gs.density_strength), source=(len(src["faces"]), len(src["co"])),
            helpers=parallel.allowed(),
        )
        state["message"] = f"Job written to {folder}. Copy the folder to the other computer and follow README.txt in it"
        self.report({"INFO"}, state["message"])
        return {"FINISHED"}


class CURIOMESH_OT_guided_import_result(bpy.types.Operator):
    bl_idname = "curiomesh.guided_import_result"
    bl_label = "Import Job Result"
    bl_description = "Bring the result.npz of an exported job back as the preview of the selected mesh, then press Apply"
    bl_options = {"INTERNAL"}

    filepath: bpy.props.StringProperty(name="Result File", description="The result.npz file the other computer made", subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.npz", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        gs = _gs(context)
        obj = _source_mesh(context)
        src = _source_data(obj)
        try:
            res, report = jobpack.load_result(self.filepath, len(src["faces"]), len(src["co"]))
        except (ValueError, OSError, KeyError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        stop_work()
        state["source"] = obj.name
        gs.scope = "AUTO"
        state["auto_base"] = {"id": time.perf_counter_ns(), "sig": src["sig"], "out": dict(res), "report": report, "job": None, "summary": jobpack.summary(report)}
        state["res"] = res
        _build_preview(context, obj, res, uvs=False)
        state["message"] = "Imported: " + state["auto_base"]["summary"] + ". Press Apply"
        if not gs.enabled:
            gs.enabled = True
        state["draw_key"] = None
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_guided_stop_work(bpy.types.Operator):
    bl_idname = "curiomesh.guided_stop_work"
    bl_label = "Stop"
    bl_description = "Stop the work that is running now. Stopping the search throws its partial result away; stopping a redo inside the shapes brings back the result from before it"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        gs = _gs(context)
        work = state["work"]
        if work is not None and isinstance(work["runner"], segments.PartJob) and state["auto_base"] is not None:
            _show_base(context, state_source())
            state["message"] = "Stopped. The parts are as they were"
        elif work is not None and work["key"][1] == "BOOL" and state["auto_base"] is not None:
            gs["bool_on"] = False
            _show_base(context, state_source())
            state["message"] = "Stopped. The shapes are kept: turn Redo Inside Shapes back on to go on"
        else:
            stop_work()
            remove_preview()
            state["auto_base"] = None
            state["lod"] = None  # the sizes already made stay as objects
            state["message"] = "Stopped. Press Start to try again"
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_guided_add_shape(bpy.types.Operator):
    bl_idname = "curiomesh.guided_add_shape"
    bl_label = "Add Shape"
    bl_description = "Add a shape that marks a part of the result to redo. Move and scale it over the part; shapes over your original mesh work too"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Shape", description="A box, a sphere, or the shapes you have selected", items=[("BOX", "Box", "An Empty cube you can move and scale"), ("SPHERE", "Sphere", "A round shape you can move and scale"), ("SELECTED", "Selected", "Use the other objects you have selected (boxes, lattices, closed meshes)")], default="BOX")

    def execute(self, context):
        gs = _gs(context)
        source = state_source()
        preview = bpy.data.objects.get(PREVIEW_NAME)
        if self.kind == "SELECTED":
            picked = [o for o in context.selected_objects if o != source and _shape_poll(None, o)]
            if not picked:
                self.report({"WARNING"}, "Select a box, lattice or closed mesh first (not the source mesh)")
                return {"CANCELLED"}
            for obj in picked:
                item = gs.bool_shapes.add()
                item.obj = obj
        else:
            anchor = preview or source
            size = 0.25 * max(max(anchor.dimensions), 1e-3) if anchor is not None else 0.5
            collection = context.collection or context.scene.collection
            if self.kind == "BOX":
                shape = bpy.data.objects.new("CurioBool_Box", None)
                shape.empty_display_type = "CUBE"
                shape.empty_display_size = 1.0
            else:
                bm = bmesh.new()
                bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
                mesh = bpy.data.meshes.new("CurioBool_Sphere")
                bm.to_mesh(mesh)
                bm.free()
                shape = bpy.data.objects.new("CurioBool_Sphere", mesh)
                shape.display_type = "WIRE"
            shape.scale = (size, size, size)
            if anchor is not None:
                shape.location = anchor.matrix_world.translation
            shape.show_in_front = True
            collection.objects.link(shape)
            item = gs.bool_shapes.add()
            item.obj = shape
        gs.bool_index = len(gs.bool_shapes) - 1
        gs.bool_on = True
        return {"FINISHED"}


class CURIOMESH_OT_guided_remove_shape(bpy.types.Operator):
    bl_idname = "curiomesh.guided_remove_shape"
    bl_label = "Remove Shape"
    bl_description = "Take the chosen shape out of the list. A box or sphere made here is deleted with it; your own meshes stay"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        gs = _gs(context)
        return 0 <= gs.bool_index < len(gs.bool_shapes)

    def execute(self, context):
        gs = _gs(context)
        obj = gs.bool_shapes[gs.bool_index].obj
        gs.bool_shapes.remove(gs.bool_index)
        gs.bool_index = max(0, min(gs.bool_index, len(gs.bool_shapes) - 1))
        if obj is not None and obj.name.startswith("CurioBool_"):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
        return {"FINISHED"}


class CURIOMESH_OT_guided_add_guide(bpy.types.Operator):
    bl_idname = "curiomesh.guided_add_guide"
    bl_label = "Add Selected"
    bl_description = "Use the selected curves, Grease Pencil drawings or edge-only meshes as guide curves (or as Deformation Lines)"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Type", description="What the quads do with the line: follow it, build edge loops along it, or bend evenly across it", items=CURVE_KIND_ITEMS, default="GUIDE", options={"SKIP_SAVE"})

    def execute(self, context):
        gs = _gs(context)
        have = {g.obj for g in gs.guide_curves}
        picked = [o for o in context.selected_objects if _guide_poll(None, o) and o not in have]
        if not picked:
            self.report({"WARNING"}, "Select a curve, a Grease Pencil drawing or an edge-only mesh first")
            return {"CANCELLED"}
        for obj in picked:
            item = gs.guide_curves.add()
            item.obj, item.kind = obj, self.kind
        gs.guide_index = len(gs.guide_curves) - 1
        return {"FINISHED"}


class CURIOMESH_OT_guided_remove_guide(bpy.types.Operator):
    bl_idname = "curiomesh.guided_remove_guide"
    bl_label = "Remove Guide"
    bl_description = "Take the chosen guide out of the list. A curve drawn with Draw Guide is deleted with it; your own curves stay"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        gs = _gs(context)
        return 0 <= gs.guide_index < len(gs.guide_curves)

    def execute(self, context):
        gs = _gs(context)
        obj = gs.guide_curves[gs.guide_index].obj
        gs.guide_curves.remove(gs.guide_index)
        gs.guide_index = max(0, min(gs.guide_index, len(gs.guide_curves) - 1))
        if obj is not None and obj.name.startswith(("CM_Guide", "CM_Loop", "CM_Bend")):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0 and isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)
        return {"FINISHED"}


def new_curve(context, source, kind: str, points, normals=None, name: str | None = None):
    """A curve object of a stroke laid on `source` (points in its own space, lifted a hair along `normals` so it shows)."""
    matrix = source.matrix_world
    lift = 0.002 * max(max(source.dimensions), 1e-6)
    name = name or {"GUIDE": "CM_Guide", "LOOP": "CM_Loop", "BEND": "CM_Bend"}[kind]
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for k, p in enumerate(spline.points):
        co = np.asarray(points[k], dtype=np.float64)
        if normals is not None:
            co = co + np.asarray(normals[k], dtype=np.float64) * lift
        w = matrix @ Vector(co)
        p.co = (w.x, w.y, w.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.show_in_front = True
    (context.collection or context.scene.collection).objects.link(obj)
    return obj


class CURIOMESH_OT_guided_draw_guide(_SurfaceTool, bpy.types.Operator):
    bl_idname = "curiomesh.guided_draw_guide"
    bl_label = "Draw Guide"
    bl_description = "Draw a stroke on the mesh with the left mouse button held; each stroke becomes a curve in the list. Right click, Enter or Esc finishes"

    kind: bpy.props.EnumProperty(name="Type", description="What the quads do with the line: follow it, build edge loops along it, or bend evenly across it", items=CURVE_KIND_ITEMS, default="GUIDE", options={"SKIP_SAVE"})
    replace: bpy.props.BoolProperty(name="Redraw", description="Replace the chosen curve of the list with the next stroke", default=False, options={"SKIP_SAVE"})

    def invoke(self, context, event):
        self.obj = context.object
        state["source"] = self.obj.name
        self.stroke = None
        self.last = None
        self.done = False
        gs = _gs(context)
        if self.replace and 0 <= gs.guide_index < len(gs.guide_curves):
            self.kind = gs.guide_curves[gs.guide_index].kind
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _status(self, context):
        context.workspace.status_text_set("Draw " + {"GUIDE": "Guide", "LOOP": "Edge Loop", "BEND": "Deformation Line"}[self.kind] + "  Hold left mouse and drag over the mesh: one curve per stroke  Right click/Enter/Esc: done")

    def _make_curve(self, context):
        points = self.stroke
        self.stroke = None
        if points is None or len(points) < 2:
            return
        gs = _gs(context)
        obj = new_curve(context, self.obj, self.kind, [co for co, _n in points], [n for _c, n in points])
        if self.replace and 0 <= gs.guide_index < len(gs.guide_curves):
            item = gs.guide_curves[gs.guide_index]
            old = item.obj
            if old is not None and old.name.startswith(("CM_Guide", "CM_Loop", "CM_Bend")):
                data = old.data
                bpy.data.objects.remove(old, do_unlink=True)
                if data is not None and data.users == 0 and isinstance(data, bpy.types.Curve):
                    bpy.data.curves.remove(data)
            item.obj = obj
            self.done = True
            return
        item = gs.guide_curves.add()
        item.obj, item.kind = obj, self.kind
        gs.guide_index = len(gs.guide_curves) - 1

    def modal(self, context, event):
        if context.area is not None:
            context.area.tag_redraw()
        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            self._make_curve(context)
            self._finish(context)
            return {"FINISHED"}
        view = self._view_region(context, event)
        if view is None:
            return {"PASS_THROUGH"}
        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            self._make_curve(context)
            self._finish(context)
            return {"FINISHED"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            self.stroke, self.last = [], None
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._make_curve(context)
            if self.done:
                self._finish(context)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}
        if event.type in {"LEFTMOUSE", "MOUSEMOVE"} and self.stroke is not None:
            at = (event.mouse_x, event.mouse_y)
            if self.last is None or (at[0] - self.last[0]) ** 2 + (at[1] - self.last[1]) ** 2 >= 36:  # one point per six pixels
                hit = self._ray(view, event)
                if hit is not None:
                    self.stroke.append(hit)
                    self.last = at
            return {"RUNNING_MODAL"}
        if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


class CURIOMESH_OT_guided_edit_guide(bpy.types.Operator):
    bl_idname = "curiomesh.guided_edit_guide"
    bl_label = "Edit Curve"
    bl_description = "Select the chosen curve and open it in Edit Mode: move its points, or press G to move all of it. The preview follows. Tab back to the mesh when done"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        gs = _gs(context)
        return 0 <= gs.guide_index < len(gs.guide_curves) and gs.guide_curves[gs.guide_index].obj is not None

    def execute(self, context):
        obj = _gs(context).guide_curves[_gs(context).guide_index].obj
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if obj.type in {"CURVE", "MESH"}:
            bpy.ops.object.mode_set(mode="EDIT")
        return {"FINISHED"}


class CURIOMESH_OT_guided_paint_density(bpy.types.Operator):
    bl_idname = "curiomesh.guided_paint_density"
    bl_label = "Paint"
    bl_description = "Go to Weight Paint on the mesh with a vertex group for the markers. Paint red where you want more (or fewer) quads, then leave Weight Paint (Tab) and press Start"
    bl_options = {"REGISTER", "UNDO"}

    less: bpy.props.BoolProperty(name="Less Detail", description="On: bigger quads there (Less Detail). Off: smaller quads there (More Detail)", default=False, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        if properties.less:
            return "Weight Paint the Less Detail group: red areas get bigger quads, so the faces go elsewhere. Leave Weight Paint (Tab) and press Start"
        return "Weight Paint the More Detail group: red areas keep more, smaller quads at every Quad Size. Leave Weight Paint (Tab) and press Start"

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        gs = _gs(context)
        obj = _source_mesh(context)
        state["source"] = obj.name
        if self.less:
            group = obj.vertex_groups.get(gs.less_group) or obj.vertex_groups.new(name=gs.less_group or "CM_LessDetail")
            gs.less_group = group.name
        else:
            group = obj.vertex_groups.get(gs.density_group) or obj.vertex_groups.new(name=gs.density_group or "CM_Density")
            gs.density_group = group.name
        obj.vertex_groups.active_index = group.index
        gs.density_use = True
        context.view_layer.objects.active = obj
        if obj.mode != "WEIGHT_PAINT":
            bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
        try:  # a full red brush; Blender 5.0 keeps unified settings per paint mode, without this one
            paint = context.tool_settings.unified_paint_settings
            paint.use_unified_weight = True
            paint.weight = 1.0
        except AttributeError:
            pass
        return {"FINISHED"}


class CURIOMESH_OT_guided_clear_density(bpy.types.Operator):
    bl_idname = "curiomesh.guided_clear_density"
    bl_label = "Clear"
    bl_description = "Delete all Detail Markers: both paint groups and the marker balls"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj, gs = _source_mesh(context), _gs(context)
        return obj is not None and (any(obj.vertex_groups.get(n) is not None for n in (gs.density_group, gs.less_group)) or bool(markers(context.scene)))

    def execute(self, context):
        obj, gs = _source_mesh(context), _gs(context)
        for name in (gs.density_group, gs.less_group):
            if obj.vertex_groups.get(name) is not None:
                obj.vertex_groups.remove(obj.vertex_groups[name])
        for ball in markers(context.scene):
            bpy.data.objects.remove(ball, do_unlink=True)
        return {"FINISHED"}


class CURIOMESH_OT_guided_add_marker(bpy.types.Operator):
    bl_idname = "curiomesh.guided_add_marker"
    bl_label = "Add Detail Marker"
    bl_description = "A marker ball at the 3D cursor. Scale it to change its reach"
    bl_options = {"REGISTER", "UNDO"}

    less: bpy.props.BoolProperty(name="Less Detail", description="On: bigger quads there (Less Detail). Off: smaller quads there (More Detail)", default=False, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        if properties.less:
            return "A Less Detail ball at the 3D cursor: bigger quads inside it, at every Quad Size. Scale it to change its reach"
        return "A More Detail ball at the 3D cursor: more, smaller quads inside it, kept even at low face counts. Scale it to change its reach"

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        gs, obj = _gs(context), _source_mesh(context)
        ball = bpy.data.objects.new("CM_LessDetail" if self.less else "CM_MoreDetail", None)
        ball.empty_display_type = "SPHERE"
        ball.empty_display_size = 0.1 * max(obj.dimensions)
        ball.location = context.scene.cursor.location
        ball[MARKER_KEY] = -1.0 if self.less else 1.0
        ball.color = (0.2, 0.4, 1.0, 1.0) if self.less else (1.0, 0.2, 0.1, 1.0)
        context.scene.collection.objects.link(ball)
        gs.density_use = True
        return {"FINISHED"}


class CURIOMESH_UL_guided_guides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.obj.name if item.obj is not None else "(removed)", icon={"GUIDE": "CURVE_DATA", "LOOP": "MESH_CIRCLE", "BEND": "MOD_SIMPLEDEFORM"}[item.kind] if item.obj is not None else "ERROR")


class CURIOMESH_UL_guided_shapes(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "operation", text="")
        row.label(text=item.obj.name if item.obj is not None else "(removed)", icon="MESH_CUBE" if item.obj is not None else "ERROR")


# --------------------------------------------------------------------------- progress


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{max(seconds, 1)} s"


def progress_info(kind: str | None = None):
    """(fraction 0 to 1, words) for the job running now, or None. kind: "AUTO", "BOOL", or None for either."""
    work = state["work"]
    if work is None or not work["runner"].busy:
        return None
    runner = work["runner"]
    if isinstance(runner, segments.PartJob):
        if kind == "AUTO":
            return None
        left = runner.seconds_left()
        return runner.fraction, runner.stage + (f", about {_clock(left)} left" if left is not None else "")
    if isinstance(runner, (autoretopo.AutoJob, blockout.BlockoutJob)):
        if kind == "BOOL":
            return None
        left = runner.seconds_left()
        return runner.fraction, runner.stage + (f", about {_clock(left)} left" if left is not None else "")
    if kind == "AUTO":
        return None
    done, total = runner.progress
    if not total:
        return 0.0, "Getting ready"
    return done / total, f"Redoing piece {min(done + 1, total)} of {total}"


def fold(layout, data, prop: str, text: str):
    """A fold-out section (Blender's layout.panel). Returns the body, or None when closed."""
    header, body = layout.panel_prop(data, prop)
    header.label(text=text)
    return body


def draw_progress(layout, fraction: float, text: str) -> None:
    """A progress bar, with a longer stage name in full under it."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    short = len(text) <= 30  # a bar can't wrap: a longer stage goes in full on the lines under it, not cut to "Laying rows of q..."
    layout.progress(factor=fraction, type="BAR", text=text if short else f"{fraction * 100:.0f}%")
    if not short:
        overlay.wrap_label(layout, text, width=36, limit=3)


# --------------------------------------------------------------------------- panel


class CURIOMESH_UL_guided_points(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "kind", text="", icon_only=True)
        row.prop(item, "fill", text="", icon_only=True)
        row.prop(item, "radius", text="")
        row.prop(item, "strength", text="")


def _is_problem(message: str) -> bool:
    return any(w in message for w in ("skipped", "error", "Error", "failed", "Would make")) or message.startswith(("Select", "Place", "Nothing", "Points Only"))


class CURIOMESH_OT_multires_detail(bpy.types.Operator):
    bl_idname = "curiomesh.multires_detail"
    bl_label = "Add Multires Detail"
    bl_description = "Give the active mesh (your cage, from any tool or by hand) Multires levels that follow the other selected mesh. The selected mesh is only read"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        active = context.active_object
        return active is not None and active.type == "MESH" and active.mode == "OBJECT" and any(o is not active and o.type == "MESH" for o in context.selected_objects)

    def execute(self, context):
        cage = context.active_object
        source = next(o for o in context.selected_objects if o is not cage and o.type == "MESH")
        tree = multires.source_tree(source, cage.matrix_world.inverted() @ source.matrix_world)
        words = multires.add_to_result(context, cage, tree, _gs(context).multires_levels)
        self.report({"WARNING"} if "failed" in words else {"INFO"}, f"{cage.name}{words}, read from {source.name}")
        return {"FINISHED"}


class CURIOMESH_OT_guided_auto_all(bpy.types.Operator):
    bl_idname = "curiomesh.guided_auto_all"
    bl_label = "Start with Auto"
    bl_description = "Looks at your mesh and this computer, picks the Use, then starts. It says what it picked and why; nothing changes until you press Apply"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        gs = _gs(context)
        src = _source_data(_source_mesh(context))
        facts = detail.mesh_facts(src["co"], src["tris"])
        choice = profiles.choose(facts, profiles.free_memory_gb())
        gs.multires_on = False
        gs.auto_profile = choice["use"]
        gs.region_use = False  # Mirror stays as you set it: it sits next to the Use tiles
        if choice["levels"]:
            gs.multires_levels = choice["levels"]
            gs.multires_on = True
        gs.bake_detail, gs.bake_size = choice["bake"], choice["bake_size"]
        gs.auto_why = choice["why"]
        gs.use_auto = True
        return {"CANCELLED"} if bpy.ops.curiomesh.guided_auto_start("INVOKE_DEFAULT") == {"CANCELLED"} else {"FINISHED"}  # may ask first when the job is long


def _tag(layout, line: tuple[str, bool], suffix: str = "") -> None:
    """A one-line cost tag under an option: nothing when the option is cheap."""
    if line[0]:
        layout.label(text=line[0] + suffix, icon="ERROR" if line[1] else "TIME")


def _bake_tag(layout, gs) -> None:
    _tag(layout, hardware.cost_tag(hardware.bake_seconds(int(gs.bake_size), gs.bake_detail, gs.bake_sets, state["machine"])), ", blocks Blender")


REPO_URL = "https://github.com/CurioCrafter/CurioMesh"


@functools.cache
def problem_url() -> str:
    """New-issue link with the Blender and CurioMesh versions filled in (nothing else about the computer). Built once:
    the Issue Reports panel draws it on every redraw."""
    import tomllib
    from urllib.parse import urlencode  # only builds the link's query string; nothing is sent from Blender

    try:
        version = tomllib.loads((Path(__file__).parent / "blender_manifest.toml").read_text(encoding="utf-8")).get("version", "?")
    except (OSError, ValueError):
        version = "?"
    body = f"CurioMesh {version}, Blender {bpy.app.version_string}\nOperating system: \n\nWhat I did:\n\nWhat happened:\n\nWhat I expected:\n"
    return f"{REPO_URL}/issues/new?" + urlencode({"title": "Problem: ", "body": body})


CLASSES = (
    CURIOMESH_PG_guided_point,
    CURIOMESH_PG_guided_shape,
    CURIOMESH_PG_guided_guide,
    CURIOMESH_PG_guided,
    CURIOMESH_OT_guided_place,
    CURIOMESH_OT_guided_work,
    CURIOMESH_OT_guided_add_point,
    CURIOMESH_OT_guided_remove_point,
    CURIOMESH_OT_guided_clear_points,
    CURIOMESH_OT_guided_add_empty,
    CURIOMESH_OT_guided_suggest,
    CURIOMESH_OT_guided_update,
    CURIOMESH_OT_guided_edit_preview,
    CURIOMESH_OT_guided_lock,
    CURIOMESH_OT_guided_unlock,
    CURIOMESH_OT_guided_redraw_strip,
    CURIOMESH_OT_guided_frame,
    CURIOMESH_OT_guided_apply,
    CURIOMESH_OT_guided_make_lods,
    CURIOMESH_OT_guided_stop,
    CURIOMESH_OT_guided_auto_start,
    CURIOMESH_OT_guided_check_pc,
    CURIOMESH_OT_guided_export_job,
    CURIOMESH_OT_guided_import_result,
    CURIOMESH_OT_guided_auto_all,
    CURIOMESH_OT_multires_detail,
    CURIOMESH_OT_guided_stop_work,
    CURIOMESH_OT_guided_add_shape,
    CURIOMESH_OT_guided_remove_shape,
    CURIOMESH_OT_guided_add_guide,
    CURIOMESH_OT_guided_remove_guide,
    CURIOMESH_OT_guided_draw_guide,
    CURIOMESH_OT_guided_edit_guide,
    CURIOMESH_OT_teach_remember,
    CURIOMESH_OT_teach_find,
    CURIOMESH_OT_teach_apply,
    CURIOMESH_OT_teach_export,
    CURIOMESH_OT_teach_import,
    CURIOMESH_OT_teach_forget_kept,
    CURIOMESH_OT_guided_paint_density,
    CURIOMESH_OT_guided_clear_density,
    CURIOMESH_OT_guided_add_marker,
    CURIOMESH_UL_guided_guides,
    CURIOMESH_UL_guided_points,
    CURIOMESH_UL_guided_shapes,
)

_draw_handles: list = []


@bpy.app.handlers.persistent
def _on_load_post(*_args):
    stop_work()
    remove_preview()  # a preview saved with the file: nothing can apply it after a load
    ladder.cancel()
    segments.reset()
    flowedit.cancel()
    state.update(source="", sig=None, src=None, res=None, lock=None, spot=None, draw=None, draw_key=None, batches=None, message="", auto_base=None, curve_sig=None, region=None, found=[])
    _preview_on.clear()  # its timer is gone after a load; the user turns it on again
    checks.forget()
    overlay.ensure_watch()  # starts only if the loaded scene has the HUD, Heatmap or Live turned on


def register() -> None:
    for cls in (*segments.PRE_CLASSES, *CLASSES, *segments.CLASSES, *ladder.CLASSES, *flowedit.classes()):
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_guided = bpy.props.PointerProperty(type=CURIOMESH_PG_guided)
    bpy.types.Object.curiomesh_points = bpy.props.CollectionProperty(type=CURIOMESH_PG_guided_point)
    bpy.types.Object.curiomesh_point_index = bpy.props.IntProperty(default=0)
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW"))
    segments.register()
    organize.register()
    flowedit.register()
    bpy.app.handlers.load_post.append(_on_load_post)


def unregister() -> None:
    while _draw_handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handles.pop(), "WINDOW")
        except Exception:
            pass
    segments.unregister()
    organize.unregister()
    flowedit.unregister()
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    for timer in (_poll, _auto_slice):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    stop_work()
    ladder.cancel()
    ladder.unregister()
    try:
        remove_preview()
    except Exception:
        pass
    state.update(source="", sig=None, src=None, res=None, lock=None, spot=None, hover=None, message="")
    _preview_on.clear()
    checks.forget()
    for owner, name in ((bpy.types.Scene, "curiomesh_guided"), (bpy.types.Object, "curiomesh_points"), (bpy.types.Object, "curiomesh_point_index")):
        if hasattr(owner, name):
            delattr(owner, name)
    for cls in reversed((*segments.PRE_CLASSES, *CLASSES, *segments.CLASSES, *ladder.CLASSES, *flowedit.classes())):
        bpy.utils.unregister_class(cls)
