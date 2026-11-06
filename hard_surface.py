# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""Hard-surface tools for Auto Segments: what the switches mean, what goes to the search, the hard-edge finish on Apply
and the controls (one small draw function, to be placed wherever the panel wants it).

Clean Panels and Even Cylinders add rebuilt versions of a part to the search (`triadq.hardsurf`); the camera picks them
only where they hold the view. They are off by default and on with the Hard Surface use: measured on six free
hard-surface models they helped the crane and hurt the compressor (see the changelog). Hard Edges finishes the written mesh: Mark Sharp + Crease
flags every hard edge (a game engine shades it hard, a Subdivision Surface keeps it sharp, no extra faces), Holding
Edges adds a loop on both sides of every hard edge at a small distance (Blender's own bevel, two segments with a square
profile, so the shape does not move), which is how a modeller makes a mesh ready for subdivision.
"""

from __future__ import annotations

import math

import bmesh
import bpy

EDGE_ITEMS = [
    ("NONE", "Leave", "Hard edges are only shaded hard, as before", "BLANK1", 0),
    ("MARK", "Mark Sharp + Crease", "Every hard edge is marked Sharp and given full Crease on Apply. Good for game assets: no extra faces, and a Subdivision Surface still keeps the edges sharp", "EDGESEL", 1),
    ("HOLD", "Holding Edges", "On Apply, a loop is added on both sides of every hard edge, close to it, so a Subdivision Surface keeps the edge sharp without creases. Adds faces along every hard edge; use it for a mesh you will subdivide", "MOD_BEVEL", 2),
]
HOLD_UNITS = [
    ("PERCENT", "% of Face", "The loops sit this share of the next face's width from the edge, so small parts get small gaps", 0),
    ("MM", "Millimetres", "The loops sit this far from the edge everywhere, in millimetres of the scene's units", 1),
]

TIPS = {
    "hs_panels": "Auto Segments: flat parts (boxes, planks, plates, brackets, bevel strips) are also laid out as clean panels: each flat area becomes one grid along its own edges, holes get rings of quads, and every panel border stays an exact edge loop. The camera keeps it only where the view holds, and a clean layout counts as half its faces when the search weighs faces against change, so it can win with up to about twice the faces. Costs some search time; on with the Hard Surface use",
    "hs_cylinders": "Auto Segments: round parts (pipes, rods, barrels, rings, also scanned ones) are also rebuilt as even cylinders: the same number of segments all the way round (a multiple of four, at least 16 unless the part is a thin rod), straight rows along it and grid caps with no fans. A visibly round part (a pipe, a barrel) also never drops below about 16 sides. The camera keeps the rebuild only where the view holds; dents and bumps of a scan are smoothed away, so scans rarely take it. On with the Hard Surface use",
    "hs_edges": "What happens to hard edges when you Apply",
    "hs_angle": "Faces meeting at more than this angle make a hard edge. Lower finds softer edges too",
    "hs_hold_unit": "How the distance of the holding loops is given",
    "hs_hold_percent": "How far the holding loops sit from the hard edge, as a share of the next face's width. Smaller keeps the edge sharper under subdivision; too small and the shading pinches",
    "hs_hold_mm": "How far the holding loops sit from the hard edge, in millimetres. Smaller keeps the edge sharper under subdivision",
}


def job_flags(gs) -> dict:
    """What the search needs to know (also part of the search's cache key)."""
    return {"panels": bool(gs.hs_panels), "cylinders": bool(gs.hs_cylinders)}


def finish(mesh, gs, unit_scale: float = 1.0) -> int:
    """Hard Edges on a written mesh. Returns how many hard edges were treated."""
    if gs.hs_edges == "NONE":
        return 0
    limit = math.radians(gs.hs_angle)
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        ids = [e.index for e in bm.edges if e.is_manifold and e.calc_face_angle(0.0) > limit]
        if ids and gs.hs_edges == "HOLD":
            if gs.hs_hold_unit == "PERCENT":
                offset, kind = gs.hs_hold_percent, "PERCENT"
            else:
                offset, kind = gs.hs_hold_mm / 1000.0 / max(unit_scale, 1e-9), "OFFSET"
            hard = [bm.edges[i] for i in ids]
            verts = list({v for e in hard for v in e.verts})
            # Two segments with a square profile: the middle row stays on the old edge, so the shape keeps its corners.
            made = bmesh.ops.bevel(bm, geom=hard + verts, offset=offset, offset_type=kind, segments=2, profile=1.0, affect="EDGES", clamp_overlap=True, loop_slide=True)
            for f in made.get("faces", ()):
                f.smooth = True
            bm.to_mesh(mesh)
    finally:
        bm.free()
    if not ids:
        return 0
    if gs.hs_edges == "MARK":  # straight on the mesh's own edge attributes
        sharp = mesh.attributes.get("sharp_edge") or mesh.attributes.new("sharp_edge", "BOOLEAN", "EDGE")
        crease = mesh.attributes.get("crease_edge") or mesh.attributes.new("crease_edge", "FLOAT", "EDGE")
        flags = [False] * len(mesh.edges)
        values = [0.0] * len(mesh.edges)
        sharp.data.foreach_get("value", flags)
        crease.data.foreach_get("value", values)
        for i in ids:
            flags[i], values[i] = True, 1.0
        sharp.data.foreach_set("value", flags)
        crease.data.foreach_set("value", values)
    else:
        custom = mesh.attributes.get("custom_normal")  # custom normals no longer match the new faces
        if custom is not None:
            mesh.attributes.remove(custom)
        elif mesh.has_custom_normals:  # where they are not an attribute, zero vectors reset them to automatic
            mesh.normals_split_custom_set([(0.0, 0.0, 0.0)] * len(mesh.loops))
    mesh.update()
    return len(ids)


def draw(layout, gs) -> None:
    """The hard-surface controls: two search switches and the Apply finish."""
    col = layout.column(align=True)
    row = col.row(align=True)
    row.prop(gs, "hs_panels", toggle=True, icon="MESH_PLANE")
    row.prop(gs, "hs_cylinders", toggle=True, icon="MESH_CYLINDER")
    col.prop(gs, "hs_edges", text="Hard Edges")
    if gs.hs_edges != "NONE":
        col.prop(gs, "hs_angle")
    if gs.hs_edges == "HOLD":
        row = col.row(align=True)
        row.prop(gs, "hs_hold_unit", text="")
        row.prop(gs, "hs_hold_percent" if gs.hs_hold_unit == "PERCENT" else "hs_hold_mm", text="Distance")


def prop(name: str):
    """One of the properties, for the Auto Segments settings (an annotation of its property group)."""
    tip = TIPS[name]
    return {
        "hs_panels": lambda: bpy.props.BoolProperty(name="Clean Panels", description=tip, default=False),
        "hs_cylinders": lambda: bpy.props.BoolProperty(name="Even Cylinders", description=tip, default=False),
        "hs_edges": lambda: bpy.props.EnumProperty(name="Hard Edges", description=tip, items=EDGE_ITEMS, default="NONE"),
        "hs_angle": lambda: bpy.props.FloatProperty(name="Hard Edge Angle", description=tip, default=40.0, min=5.0, max=150.0, precision=0),
        "hs_hold_unit": lambda: bpy.props.EnumProperty(name="Distance In", description=tip, items=HOLD_UNITS, default="PERCENT"),
        "hs_hold_percent": lambda: bpy.props.FloatProperty(name="Distance", description=tip, default=10.0, min=1.0, max=45.0, subtype="PERCENTAGE", precision=0),
        "hs_hold_mm": lambda: bpy.props.FloatProperty(name="Distance", description=tip, default=2.0, min=0.01, max=1000.0, precision=2),
    }[name]()
