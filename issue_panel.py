# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The Issues box of the Fix Parts panel: the worst spots of a finished result (faces turned inward, folded, crossing,
slivers, pinched faces, poles...), a way to jump to each, a saved record, and a comparison with an older record.

The spots come from `triadq.issues`, made at the end of every Auto Segments run (`res["issue_spots"]`)."""

from __future__ import annotations

import json
import os

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Vector

from . import overlay
from .triadq import issues

LISTED = 8  # spots shown in the panel; Next Issue walks through all of them
SPOT_COLOUR = (1.0, 0.45, 0.0, 0.6)
CURRENT_COLOUR = (1.0, 1.0, 0.0, 0.95)
KIND_WORDS = {
    "inward": "turned inward", "sunk": "sunk into your mesh", "crossing": "crossing", "folded_over": "folded over",
    "bent": "bent", "flat": "flat", "duplicate": "duplicate", "overlap": "overlapping",
    "sliver": "slivers", "pinched": "pinched", "pole": "poles", "concave": "concave",
}

ui = {"res": None, "cur": -1, "diff": None, "batch": None, "batch_key": None}


def _g():
    from . import guided

    return guided


def spots() -> list:
    res = _g().state["res"]
    if res is None:
        return []
    if ui["res"] is not res:  # a new result: start the walk over
        ui.update(res=res, cur=-1, diff=None, batch=None, batch_key=None)
    return res.get("issue_spots") or []


def summary(kinds: dict) -> str:
    return ", ".join(f"{c:,} {KIND_WORDS.get(k, k)}" for k, c in sorted(kinds.items(), key=lambda kv: -issues.WEIGHTS.get(kv[0], 0) * kv[1])[:3])


def _frame(context, spot) -> bool:
    guided = _g()
    rv3d = getattr(context.space_data, "region_3d", None) or context.region_data
    preview = bpy.data.objects.get(guided.PREVIEW_NAME)
    if rv3d is None or preview is None:
        return False
    centre = guided._world(preview.matrix_world, np.asarray([spot["center"]]))[0]
    scale = float(np.linalg.norm(np.asarray(preview.matrix_world.to_scale())) / 3 ** 0.5)
    rv3d.view_location = Vector(centre.tolist())
    rv3d.view_distance = max(spot["radius"] * scale * 4.0, 1e-3)
    return True


def _select(context, spot) -> bool:
    import bmesh

    preview = bpy.data.objects.get(_g().PREVIEW_NAME)
    if preview is None:
        return False
    for o in context.view_layer.objects:
        o.select_set(False)
    context.view_layer.objects.active = preview
    preview.select_set(True)
    if preview.mode != "EDIT":
        bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(preview.data)
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        f.select_set(False)
    for i in spot["faces"]:
        if i < len(bm.faces):
            bm.faces[i].select_set(True)
    bm.select_flush_mode()
    bmesh.update_edit_mesh(preview.data)
    return True


def _go(context, index: int) -> None:
    found = spots()
    if not found:
        return
    ui["cur"] = index % len(found)
    _frame(context, found[ui["cur"]])
    overlay.tag_redraw()


class _IssueOp:
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(spots())


class CURIOMESH_OT_issue_view(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_view"
    bl_label = "View"
    bl_description = "Move the 3D view to this problem spot and highlight it"

    index: bpy.props.IntProperty()

    def execute(self, context):
        _go(context, self.index)
        return {"FINISHED"}


class CURIOMESH_OT_issue_select(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_select"
    bl_label = "Select"
    bl_description = "Select the faces of this problem spot in Edit Mode"

    index: bpy.props.IntProperty()

    def execute(self, context):
        found = spots()
        if not 0 <= self.index < len(found):
            return {"CANCELLED"}
        ui["cur"] = self.index
        _select(context, found[self.index])
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_issue_step(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_step"
    bl_label = "Next Issue"
    bl_description = "Go to the next (or previous) problem spot, worst first, and move the view to it"

    step: bpy.props.IntProperty(default=1)

    def execute(self, context):
        _go(context, ui["cur"] + self.step if ui["cur"] >= 0 else (0 if self.step > 0 else -1))
        return {"FINISHED"}


def _record(context) -> dict:
    res = _g().state["res"]
    rec = dict(res["issues"])
    rec["meta"] = dict(rec.get("meta", {}), source=_g().state["source"], blend=os.path.basename(bpy.data.filepath))
    return rec


class CURIOMESH_OT_issue_save(_IssueOp, bpy.types.Operator, ExportHelper):
    bl_idname = "curiomesh.issue_save"
    bl_label = "Save Issue Report"
    bl_description = "Save the list of problem spots as a JSON file (next to your .blend file if it is saved), so a later run can be compared with it"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        name = f"curiomesh_issues_{_g().state['source']}.json"
        if bpy.data.filepath:  # next to the .blend: no question asked
            self.filepath = os.path.join(os.path.dirname(bpy.data.filepath), name)
            return self.execute(context)
        self.filepath = name
        return super().invoke(context, event)

    def execute(self, context):
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(_record(context), fh, indent=1)
        self.report({"INFO"}, f"Issue report saved: {self.filepath}")
        return {"FINISHED"}


class CURIOMESH_OT_issue_compare(_IssueOp, bpy.types.Operator, ImportHelper):
    bl_idname = "curiomesh.issue_compare"
    bl_label = "Compare With Report..."
    bl_description = "Load an older issue report and show which of its problem spots are fixed, still there, or new"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                old = json.load(fh)
            ui["diff"] = issues.compare(old, _record(context))
        except (OSError, ValueError, KeyError) as exc:
            self.report({"ERROR"}, f"Could not read that report: {exc}")
            return {"CANCELLED"}
        overlay.tag_redraw()
        return {"FINISHED"}


def draw(layout) -> None:
    """The Issues box. Nothing before a result exists."""
    found = spots()
    res = _g().state["res"]
    if res is None or res.get("issues") is None:
        return
    box = layout.box()
    totals = res["issues"]["totals"]
    box.label(text=f"Issues: {len(found)} spot{'s' if len(found) != 1 else ''}" if found else "Issues: none found", icon="ERROR" if found else "CHECKMARK")
    if not found:
        return
    for kind, count in sorted(totals.items(), key=lambda kv: -issues.WEIGHTS.get(kv[0], 0)):
        box.label(text=f"{count:,} {KIND_WORDS.get(kind, kind)}")
    row = box.row(align=True)
    row.operator("curiomesh.issue_step", text="Previous", icon="TRIA_LEFT").step = -1
    row.operator("curiomesh.issue_step", text="Next Issue", icon="TRIA_RIGHT").step = 1
    for i, spot in enumerate(found[:LISTED]):
        line = box.row(align=True)
        line.alignment = "LEFT"
        line.label(text=f"{'>' if i == ui['cur'] else ' '} #{i + 1}: {summary(spot['kinds'])}", icon="DOT")
        sub = box.row(align=True)
        part = f"part {spot['part'] + 1}" if spot["part"] >= 0 else "no part"
        sub.label(text=f"{len(spot['faces']):,} faces, {part}")
        sub.operator("curiomesh.issue_view", text="View").index = i
        sub.operator("curiomesh.issue_select", text="Select").index = i
    if len(found) > LISTED:
        box.label(text=f"... and {len(found) - LISTED} more (use Next Issue)")
    row = box.row(align=True)
    row.operator("curiomesh.issue_save", icon="FILE_TICK")
    row.operator("curiomesh.issue_compare", icon="FILE_REFRESH")
    diff = ui["diff"]
    if diff is not None:
        box.label(text=f"Since that report: {len(diff['fixed'])} fixed, {len(diff['still'])} still there, {len(diff['new'])} new", icon="INFO")
        for kind, t in diff["totals"].items():
            box.label(text=f"   {KIND_WORDS.get(kind, kind)}: {t['old']:,} before, {t['new']:,} now")


def draw_overlay(guided, preview, res, flat, batch_for_shader) -> None:
    """All spots in orange, the current one in yellow, when Show Bad Faces is on. Called from the segment view's draw."""
    found = res.get("issue_spots")
    if not found or ui["res"] is not res:
        return
    matrix = tuple(round(x, 5) for row in preview.matrix_world for x in row)
    key = (id(res), matrix, ui["cur"])
    if ui["batch_key"] != key:
        ui["batch"] = []
        tris = guided._fan(res["faces"]).reshape(-1, 3)
        owner = np.repeat(np.arange(len(res["faces"])), [len(f) - 2 for f in res["faces"]])
        for colour, picked in ((SPOT_COLOUR, [s for i, s in enumerate(found) if i != ui["cur"]]), (CURRENT_COLOUR, [found[ui["cur"]]] if 0 <= ui["cur"] < len(found) else [])):
            mask = np.zeros(len(res["faces"]), bool)
            for s in picked:
                mask[s["faces"]] = True
            corners = tris[mask[owner]].reshape(-1)
            if len(corners):
                pos = guided._world(preview.matrix_world, res["co"][corners]).astype(np.float32)
                ui["batch"].append(batch_for_shader(flat, "TRIS", {"pos": pos, "color": np.tile(np.float32(colour), (len(corners), 1))}))
        ui["batch_key"] = key
    for batch in ui["batch"]:
        flat.bind()
        batch.draw(flat)


CLASSES = (CURIOMESH_OT_issue_view, CURIOMESH_OT_issue_select, CURIOMESH_OT_issue_step, CURIOMESH_OT_issue_save, CURIOMESH_OT_issue_compare)
