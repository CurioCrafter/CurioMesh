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

from . import overlay, tidytools
from .triadq import issues

LISTED = 8  # spots shown in the panel; Next Issue walks through all of them
SPOT_COLOUR = (1.0, 0.45, 0.0, 0.6)
CURRENT_COLOUR = (1.0, 1.0, 0.0, 0.95)
KIND_WORDS = {
    "inward": "turned inward", "sunk": "sunk into your mesh", "crossing": "crossing", "folded_over": "folded over",
    "bent": "bent", "flat": "flat", "duplicate": "duplicate", "overlap": "overlapping",
    "sliver": "slivers", "pinched": "pinched", "pole": "poles", "concave": "concave",
    "touching": "touching parts",
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
    preview = guided.preview_object()
    if rv3d is None or preview is None:
        return False
    centre = guided._world(preview.matrix_world, np.asarray([spot["center"]]))[0]
    scale = float(np.linalg.norm(np.asarray(preview.matrix_world.to_scale())) / 3 ** 0.5)
    rv3d.view_location = Vector(centre.tolist())
    rv3d.view_distance = max(spot["radius"] * scale * 4.0, 1e-3)
    return True


def _select(context, spot) -> bool:
    import bmesh

    preview = _g().preview_object()
    if preview is None:
        return False
    if preview.mode != "EDIT":  # leaves any other mode first, and makes the preview (unselectable otherwise) active, faces mode
        bpy.ops.curiomesh.guided_edit_preview()
    if preview.mode != "EDIT":
        return False
    bm = bmesh.from_edit_mesh(preview.data)
    bm.select_mode = {"FACE"}
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
    bl_description = "Moves the 3D view to this bad spot and highlights it"

    index: bpy.props.IntProperty(name="Spot", description="Which bad spot, 0 is the worst")

    def execute(self, context):
        _go(context, self.index)
        return {"FINISHED"}


class CURIOMESH_OT_issue_select(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_select"
    bl_label = "Select"
    bl_description = "Select the faces of this problem spot in Edit Mode"

    index: bpy.props.IntProperty(name="Spot", description="Which bad spot, 0 is the worst")

    def execute(self, context):
        found = spots()
        if not 0 <= self.index < len(found):
            return {"CANCELLED"}
        ui["cur"] = self.index
        if not _select(context, found[self.index]):
            self.report({"WARNING"}, "The preview is gone: run Auto again")
            return {"CANCELLED"}
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_issue_step(_IssueOp, bpy.types.Operator):
    bl_idname = "curiomesh.issue_step"
    bl_label = "Next Issue"
    bl_description = "Goes to the next (or previous) bad spot, worst first, and moves the view to it"

    step: bpy.props.IntProperty(name="Step", description="1 goes to the next bad spot, -1 to the one before", default=1)

    def execute(self, context):
        _go(context, ui["cur"] + self.step if ui["cur"] >= 0 else (0 if self.step > 0 else -1))
        return {"FINISHED"}


def spot_part(index: int) -> int:
    """The part a spot sits on, or -1."""
    found = spots()
    return int(found[index].get("part", -1)) if 0 <= index < len(found) else -1


class _SpotPartOp(_IssueOp):
    index: bpy.props.IntProperty(name="Spot", description="Which bad spot, 0 is the worst")

    @classmethod
    def poll(cls, context):
        from . import segments

        segments.ensure()
        return bool(spots()) and segments.ready() and not segments.mirrored() and _g().progress_info() is None

    def _part(self):
        from . import segments

        p = spot_part(self.index)
        return p if 0 <= p < len(segments.seg["parts"]) else None


class CURIOMESH_OT_issue_redo(_SpotPartOp, tidytools.PartSwap, bpy.types.Operator):
    bl_idname = "curiomesh.issue_redo"
    bl_label = "Redo This Spot"
    bl_description = "Shows the part under this spot remade with smaller quads, which usually clears folded or crossing faces, then Enter keeps it. The old version stays in the part's tries"

    @classmethod
    def description(cls, context, properties):
        from . import segments

        p = spot_part(properties.index)
        if not 0 <= p < len(segments.seg["parts"]):
            return cls.bl_description
        return f"Shows {segments.name_of(p)} ({segments.seg['parts'][p]['after']:,} faces now) remade with smaller quads, which usually clears folded or crossing faces. Enter keeps it; the old version stays in its tries"

    def plan(self, context) -> dict:
        return redo_plan(self._part(), self.index)

    def keep(self, context, plan: dict) -> bool:
        from . import ladder

        changed = ladder.apply_sizes(context, plan["sizes"], plan["todo"], False, plan["name"]) > 0
        overlay.tag_redraw()
        return changed

    def again(self, plan: dict):
        return "curiomesh.part_size", {"bigger": False}  # Again: the same part, smaller again, from the result as it is now


class CURIOMESH_OT_issue_keep_original(_SpotPartOp, tidytools.PartSwap, bpy.types.Operator):
    bl_idname = "curiomesh.issue_keep_original"
    bl_label = "Keep Original Here"
    bl_description = "Shows your own faces back on the part under this spot, then Enter keeps them. Put Reduced Back or Ctrl+Z undoes it"

    @classmethod
    def description(cls, context, properties):
        from . import segments

        p = spot_part(properties.index)
        if not 0 <= p < len(segments.seg["parts"]):
            return cls.bl_description
        return f"Shows your own faces back on {segments.name_of(p)}, about {segments.original_cost([p]):,} more faces, then Enter keeps them. Put Reduced Back or Ctrl+Z undoes it"

    def plan(self, context) -> dict:
        return original_plan(self._part(), self.index)

    def keep(self, context, plan: dict) -> bool:
        from . import segments

        return bool(plan["tries"]) and segments.use_tries(context, plan["tries"], plan["name"])


def original_parts() -> list[int]:
    """Parts that show your own faces now (Keep Original), which Put Reduced Back can switch back."""
    from . import segments

    return [p for p, r in enumerate(segments.seg["parts"]) if r["method"] == segments.ORIGINAL]


class CURIOMESH_OT_issue_put_back(tidytools.PartSwap, bpy.types.Operator):
    bl_idname = "curiomesh.issue_put_back"
    bl_label = "Put Reduced Back"
    bl_description = "Shows the parts that got your own faces back with the reduced faces they had before Keep Original, then Enter keeps them"
    bl_options = {"INTERNAL"}

    part: bpy.props.IntProperty(name="Part", description="Which part to switch back; -1 for every part showing your own faces", default=-1)

    @classmethod
    def poll(cls, context):
        from . import segments

        segments.ensure()
        return segments.ready() and _g().progress_info() is None and bool(original_parts())

    def plan(self, context) -> dict:
        return put_back_plan(self.part)

    def keep(self, context, plan: dict) -> bool:
        from . import segments

        return bool(plan["tries"]) and segments.use_tries(context, plan["tries"], plan["name"])


def _record(context) -> dict:
    res = _g().state["res"]
    rec = dict(res["issues"])
    rec["meta"] = dict(rec.get("meta", {}), source=_g().state["source"], blend=os.path.basename(bpy.data.filepath))
    return rec


def redo_plan(p, index: int) -> dict:
    """The plan of Redo This Spot for part `p` (None: the spot is on no part that can be redone). Nothing changes yet,
    except that a ready size is kept in the part's tries."""
    from . import ladder, segments

    if p is None:
        return {"name": "Redo This Spot", "tries": {}, "todo": [], "sizes": {}, "why": "This spot is not on a part that can be redone"}
    ui["cur"] = index
    segments.select([p])
    wanted, todo, why = ladder.size_plan([p], bigger=False)
    return {"name": "Redo This Spot", "tries": {q: i for q, (i, _r) in wanted.items()}, "sizes": wanted, "todo": todo, "why": why[0] if why else ""}


def original_plan(p, index: int) -> dict:
    """The plan of Keep Original Here for part `p` (None: no part)."""
    from . import segments

    tries = segments.original_tries([p]) if p is not None else {}
    if p is not None:
        ui["cur"] = index
    return {"name": "Keep Original Here", "tries": tries, "todo": [], "why": "" if tries else "This part could not be put back"}


def put_back_plan(part: int = -1) -> dict:
    """The plan of Put Reduced Back for one part, or -1 for every part showing your own faces."""
    from . import segments

    wanted = {}
    for p in original_parts() if part < 0 else [part]:
        if not 0 <= p < len(segments.seg["parts"]):
            continue
        tries = segments.seg["parts"][p]["tries"]
        index = next((i for i in range(len(tries) - 1, -1, -1) if tries[i]["method"] != segments.ORIGINAL), None)
        if index is not None:
            wanted[p] = index
    return {"name": "Put Reduced Back", "tries": wanted, "todo": [], "why": "" if wanted else "These parts have no reduced faces to go back to"}


def save_report(path: str) -> dict:
    """Write the issue record to `path` (what Save Issue Report does). Raises OSError."""
    rec = _record(bpy.context)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def compare_report(path: str) -> dict:
    """Compare the result's spots with the report at `path` and keep the answer for the panel (what Compare does).
    Raises OSError, or ValueError and friends for a file that is not one of our reports."""
    with open(path, encoding="utf-8") as fh:
        old = json.load(fh)
    ui["diff"] = issues.compare(old, _record(bpy.context))
    return ui["diff"]


class CURIOMESH_OT_issue_save(_IssueOp, bpy.types.Operator, ExportHelper):
    bl_idname = "curiomesh.issue_save"
    bl_label = "Save Issue Report"
    bl_description = "Saves the list of bad spots as a file next to your .blend, so a later run can be compared with it"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        name = f"curiomesh_issues_{bpy.path.clean_name(_g().state['source'] or 'mesh')}.json"  # an object name can hold : or /
        if bpy.data.filepath:  # next to the .blend: no question asked
            self.filepath = os.path.join(os.path.dirname(bpy.data.filepath), name)
            return self.execute(context)
        self.filepath = name
        return super().invoke(context, event)

    def execute(self, context):
        try:
            save_report(self.filepath)
        except OSError as exc:
            self.report({"ERROR"}, f"Could not save the issue report: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Issue report saved: {self.filepath}")
        return {"FINISHED"}


class CURIOMESH_OT_issue_compare(_IssueOp, bpy.types.Operator, ImportHelper):
    bl_idname = "curiomesh.issue_compare"
    bl_label = "Compare With Report..."
    bl_description = "Loads an older report and shows which of its bad spots are fixed, still there, or new"
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        try:
            compare_report(self.filepath)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:  # not a report, or not one of ours
            self.report({"ERROR"}, f"Could not read that report: {exc}")
            return {"CANCELLED"}
        overlay.tag_redraw()
        return {"FINISHED"}


def draw_reports(layout) -> None:
    """Save the spot list and compare a later run with it (Pro > Issue Reports)."""
    found = spots()
    res = _g().state["res"]
    if res is None or res.get("issues") is None:
        layout.label(text="Reports need a finished result", icon="INFO")
        return
    layout.label(text=f"{len(found)} spot{'s' if len(found) != 1 else ''} with bad faces" if found else "No bad spots found", icon="ERROR" if found else "CHECKMARK")
    for kind, count in sorted(res["issues"]["totals"].items(), key=lambda kv: -issues.WEIGHTS.get(kv[0], 0)):
        layout.label(text=f"{count:,} {KIND_WORDS.get(kind, kind)}", icon="BLANK1")
    row = layout.row(align=True)
    row.operator("curiomesh.issue_save", icon="FILE_TICK")
    row.operator("curiomesh.issue_compare", icon="FILE_REFRESH")
    diff = ui["diff"]
    if diff is not None:
        layout.label(text=f"Since that report: {len(diff['fixed'])} fixed, {len(diff['still'])} still there, {len(diff['new'])} new", icon="INFO")
        for kind, t in diff["totals"].items():
            layout.label(text=f"   {KIND_WORDS.get(kind, kind)}: {t['old']:,} before, {t['new']:,} now")


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


CLASSES = (
    CURIOMESH_OT_issue_view,
    CURIOMESH_OT_issue_select,
    CURIOMESH_OT_issue_step,
    CURIOMESH_OT_issue_redo,
    CURIOMESH_OT_issue_keep_original,
    CURIOMESH_OT_issue_put_back,
    CURIOMESH_OT_issue_save,
    CURIOMESH_OT_issue_compare,
)
