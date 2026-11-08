# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Send to Agent and Save for a Report (bench/tools_design/AGENT_TOOLS.md, sections 3 and 4).

Both write the marked parts of the result on screen as files (format in `partpack.py`). Neither changes the mesh or the
History, and neither uses the network. `result_key` is `partpack.result_key(result faces, border points of the exported
parts, in the order `parts.json` lists them)`: the importer recomputes it from the same parts of its current result.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import tempfile
import urllib.parse

import bpy
import numpy as np

from . import guided, partpack, segments
from .triadq import issues
from .triadq import parts as P
from .triadq import segment as S
from .agentcmd import AgentError as PartsError

WARNING = "This zip holds the mesh pieces of your model named below. Check them before you share it."



def _faces_of(polys) -> list:
    return [tuple(int(v) for v in f) for f in polys]


def _local(co, faces, ids):
    """(points, faces) of the result faces `ids` with their own vertex numbers."""
    if not len(ids):
        return np.zeros((0, 3)), []
    used, flat = np.unique(np.fromiter((v for i in ids for v in faces[i]), dtype=np.int64), return_inverse=True)
    out, at = [], 0
    for i in ids:
        out.append(tuple(int(v) for v in flat[at : at + len(faces[i])]))
        at += len(faces[i])
    return np.asarray(co)[used], out


def _marked(ids):
    ids = sorted(segments.seg["sel"]) if ids is None else [int(p) for p in ids]
    if not ids:
        raise PartsError("BAD_ARGS", "No parts are marked", "Mark parts in the viewport, or pass parts: [ids]")
    n = len(segments.seg["parts"])
    bad = [p for p in ids if not 0 <= p < n]
    if bad:
        raise PartsError("NOT_FOUND", f"No part {bad[0]}", f"Parts run from 0 to {n - 1}", {"parts": bad})
    return ids


def _need_result():
    segments.ensure()
    res = guided.state["res"]
    if res is None or not segments.seg["parts"] or segments.seg["piece"] is None or len(segments.seg["piece"]) != len(res["faces"]):
        raise PartsError("NO_RESULT", "There is no result with parts on screen", "Run auto.start and wait for it to finish")
    return res


def collect(ids=None):
    """(parts, skipped, result_info, object name) for the parts `ids` (default: the marked ones)."""
    res = _need_result()
    ids = _marked(ids)
    seg = segments.seg
    co, faces, mat, piece = np.asarray(res["co"]), res["faces"], np.asarray(res["mat"]), np.asarray(seg["piece"])
    flat_v = np.fromiter((v for f in faces for v in f), dtype=np.int64)
    face_of = np.repeat(np.arange(len(faces)), np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces)))
    flags = issues.topology_flags(co, faces)
    base = guided.state["auto_base"]
    job = base["job"] if base else None
    parts, skipped = [], []
    for pid in ids:
        mine = np.flatnonzero(piece == pid)
        if not len(mine):
            skipped.append({"id": pid, "why": "it has no faces in the result"})
            continue
        try:
            S.seam_loops(co, faces, mine.tolist())
            loops, _ = S.boundary_loops(faces, mine.tolist())
        except ValueError:
            skipped.append({"id": pid, "why": "its border touches itself at a point, so it cannot be swapped on its own"})
            continue
        points, pfaces, pmat = P.cut(co, faces, mat, piece, pid)
        used = np.unique(np.fromiter((v for i in mine.tolist() for v in faces[i]), dtype=np.int64))
        border = [[int(np.searchsorted(used, v)) for v in loop] + [int(np.searchsorted(used, loop[0]))] for loop in loops]
        near = np.zeros(len(co), bool)
        near[used] = True
        touching = np.unique(face_of[near[flat_v]])
        around = _local(co, faces, touching[piece[touching] != pid].tolist())
        source = ([], [])
        if job is not None:
            sp, sf, _kinds = job._piece_arrays(np.asarray(seg["parts"][pid]["source"]))
            source = (np.asarray(sp), _faces_of(sf))
        rec = seg["parts"][pid]
        parts.append({
            "id": pid, "name": segments.name_of(pid), "kind": rec["kind"],
            "numbers": {"faces": len(pfaces), "view_change_pct": None if rec["change"] is None else round(float(rec["change"]), 2), "tidiness": None if rec.get("topology") is None else round(float(rec["topology"]), 1)},
            "issues": sorted(k for k, v in flags.items() if v[mine].any()),
            "points": points, "faces": pfaces, "mats": [int(m) for m in pmat], "border": border, "source": source, "around": around,
        })
    obj = bpy.data.objects.get(guided.state["source"] or "")
    border_pts = [pt for p in parts for loop in p["border"] for pt in np.asarray(p["points"], dtype=np.float32)[loop[:-1]].tolist()]
    info = {
        "made_by": f"CurioMesh {'.'.join(str(x) for x in _version())}", "skipped": skipped,
        "source_faces": len(obj.data.polygons) if obj is not None else 0, "source_verts": len(obj.data.vertices) if obj is not None else 0,
        "result_faces": len(faces), "result_key": partpack.result_key(len(faces), border_pts),
    }
    return parts, skipped, info, (obj.name if obj is not None else "Mesh")


def _version():
    import importlib

    return importlib.import_module(__package__).bl_info["version"]


def _stamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _clean(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "Mesh"


def _parent(folder):
    if folder:
        return bpy.path.abspath(folder)
    return os.path.dirname(bpy.data.filepath) if bpy.data.filepath else tempfile.gettempdir()


def export_parts(ids=None, folder=None, note=""):
    """Write the part folder inside `folder` (default: next to the .blend). Returns {folder, parts, skipped}."""
    parts, skipped, info, name = collect(ids)
    if not parts:
        raise PartsError("NOT_READY", "None of those parts can be exported: " + "; ".join(f"part {s['id']} {s['why']}" for s in skipped), "", {"skipped": skipped})
    target = os.path.join(_parent(folder), f"{_clean(name)}-parts-{_stamp()}")
    try:
        partpack.write_parts(target, name, parts, info, note)
    except OSError as exc:
        raise PartsError("FILE", f"Could not write {target}: {exc}", "Pick a folder you can write to") from exc
    return {"folder": target, "parts": [p["id"] for p in parts], "skipped": skipped}


def _changed_settings(context):
    gs = context.scene.curiomesh_guided
    out = {}
    for prop in gs.bl_rna.properties:
        if prop.identifier == "rna_type" or prop.type in {"POINTER", "COLLECTION"}:
            continue
        arr = getattr(prop, "is_array", False)
        value = list(getattr(gs, prop.identifier)) if arr else getattr(gs, prop.identifier)
        default = list(prop.default_array) if arr else prop.default
        if value != default:
            out[prop.identifier] = value
    return out


def report_md(context, parts, skipped, info, name, note):
    gs = context.scene.curiomesh_guided
    changed = _changed_settings(context)
    lines = [
        f"# CurioMesh parts report: {name}", "", f"> {WARNING}", "", "## Versions",
        f"- {info['made_by']}", f"- Blender {bpy.app.version_string}", f"- {platform.system()} {platform.release()}", "",
        "## Use and settings", f"- Use: {gs.auto_profile}",
        f"- Changed from the defaults: {', '.join(f'{k} = {v}' for k, v in sorted(changed.items())) or 'none'}", "",
        "## Parts", f"Result key {info['result_key']}, {info['result_faces']} result faces.", "",
        "| Part | Name | Kind | Faces | View change % | Tidiness | Issues |", "|---|---|---|---|---|---|---|",
    ]
    for p in parts:
        n = p["numbers"]
        lines.append(f"| {p['id']} | {p['name']} | {p['kind']} | {n['faces']} | {n['view_change_pct']} | {n['tidiness']} | {', '.join(p['issues']) or 'none'} |")
    if skipped:
        lines += ["", "Left out: " + "; ".join(f"part {s['id']}: {s['why']}" for s in skipped)]
    lines += ["", "## Note", note or "(none)", ""]
    return "\n".join(lines), {"use": gs.auto_profile, "changed": changed, "version": info["made_by"], "blender": bpy.app.version_string}


def report_parts(ids=None, path=None, note=""):
    """Write one zip: the part folder, report.md and settings.json. Returns {path, parts, skipped}."""
    parts, skipped, info, name = collect(ids)
    if not parts:
        raise PartsError("NOT_READY", "None of those parts can be exported", "", {"skipped": skipped})
    path = bpy.path.abspath(path) if path else os.path.join(_parent(None), f"{_clean(name)}-parts-report-{_stamp()}.zip")
    md, settings = report_md(bpy.context, parts, skipped, info, name, note)
    tmp = tempfile.mkdtemp(prefix="curiomesh-report-")
    try:
        folder = os.path.join(tmp, f"{_clean(name)}-parts-{_stamp()}")
        partpack.write_parts(folder, name, parts, info, note)
        partpack.write_report_zip(path, folder, md, json.dumps(settings, indent=1, default=str))
    except OSError as exc:
        raise PartsError("FILE", f"Could not write {path}: {exc}", "Pick a place you can write to") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"path": path, "parts": [p["id"] for p in parts], "skipped": skipped}


# --------------------------------------------------------------------------- Send as a Problem

ISSUE_URL = "https://github.com/CurioCrafter/CurioMesh/issues/new?template=bad-spot.yml"
# The only things the link carries: form field id (.github/ISSUE_TEMPLATE/bad-spot.yml) for each.
FORM_FIELDS = {"version": "version", "blender": "blender", "use": "purpose"}
# Use id -> the form's "What the mesh is for" option; a Use with no match leaves the dropdown empty.
_FORM_PURPOSE = {
    "GAME": "Game asset", "LOW_POLY": "Game asset", "ANIMATION": "Animation or character", "SUBDIV": "Subdivision or sculpt base",
    "SCULPT": "Subdivision or sculpt base", "SCAN": "Scan or photogrammetry clean-up", "HARD": "Hard surface or machine part",
}


def issue_url(version: str, blender: str, use: str) -> str:
    """The issue-form link. Only version numbers and the Use: no mesh, names, paths or note text."""
    q = {FORM_FIELDS["version"]: version, FORM_FIELDS["blender"]: blender}
    if use in _FORM_PURPOSE:
        q[FORM_FIELDS["use"]] = _FORM_PURPOSE[use]
    return ISSUE_URL + "&" + urllib.parse.urlencode(q)


# --------------------------------------------------------------------------- operators


def _poll(cls) -> bool:
    if not (segments.ready() and segments.seg["sel"]):
        cls.poll_message_set("Mark one or more parts first (click them in the viewport)")
        return False
    return True


class CURIOMESH_OT_parts_export(bpy.types.Operator):
    bl_idname = "curiomesh.parts_export"
    bl_label = "Send to Agent"
    bl_description = "Write the marked parts, their borders and your original surface as files, for an agent or another program to make better. Nothing is uploaded and nothing changes here"
    bl_options = {"INTERNAL"}

    directory: bpy.props.StringProperty(name="Folder", description="The folder the parts are written into (a new dated folder is made inside it)", subtype="DIR_PATH")
    note: bpy.props.StringProperty(name="Note", description="What is wrong with these parts, in your words (saved in parts.json)")

    @classmethod
    def poll(cls, context):
        return _poll(cls)

    def invoke(self, context, event):
        if not self.directory:
            self.directory = _parent(None) + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            out = export_parts(None, self.directory or None, self.note)
        except PartsError as exc:
            self.report({"ERROR"}, f"{exc} {exc.hint}".strip())
            return {"CANCELLED"}
        left = f" Left out: {len(out['skipped'])} (see parts.json)." if out["skipped"] else ""
        guided.state["message"] = f"Wrote {len(out['parts'])} part(s) to {out['folder']}.{left}"
        self.report({"INFO"}, guided.state["message"])
        return {"FINISHED"}


class CURIOMESH_OT_parts_report(bpy.types.Operator):
    bl_idname = "curiomesh.parts_report"
    bl_label = "Save for a Report"
    bl_description = "Save the marked parts and a short report in one zip you can attach to an issue yourself. " + WARNING
    bl_options = {"INTERNAL"}

    filepath: bpy.props.StringProperty(name="Zip", description="Where the zip is written", subtype="FILE_PATH")
    note: bpy.props.StringProperty(name="Note", description="What is wrong with these parts, in your words (saved in the report)")

    @classmethod
    def poll(cls, context):
        return _poll(cls)

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = os.path.join(_parent(None), f"parts-report-{_stamp()}.zip")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        path = self.filepath if self.filepath.lower().endswith(".zip") else self.filepath + ".zip"
        try:
            out = report_parts(None, path, self.note)
        except PartsError as exc:
            self.report({"ERROR"}, f"{exc} {exc.hint}".strip())
            return {"CANCELLED"}
        guided.state["message"] = f"Report saved to {out['path']}. {WARNING}"
        self.report({"INFO"}, guided.state["message"])
        return {"FINISHED"}


class CURIOMESH_OT_send_problem(bpy.types.Operator):
    bl_idname = "curiomesh.send_problem"
    bl_label = "Send as a Problem"
    bl_description = "Save the marked parts in a zip and open the issue form in your browser. CurioMesh sends nothing: you attach the zip yourself"
    bl_options = {"INTERNAL"}

    note: bpy.props.StringProperty(name="Note", description="What is wrong with these parts. It is saved in the zip, not put in the link")

    @classmethod
    def poll(cls, context):
        return _poll(cls)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="CurioMesh sends nothing. It saves a zip of the marked")
        col.label(text="parts and opens the issue form. You attach the zip yourself.")
        col.prop(self, "note")

    def execute(self, context):
        try:
            out = report_parts(None, None, self.note)
        except PartsError as exc:
            self.report({"ERROR"}, f"{exc} {exc.hint}".strip())
            return {"CANCELLED"}
        try:
            version = ".".join(str(i) for i in _version())
            bpy.ops.wm.url_open(url=issue_url(version, bpy.app.version_string, context.scene.curiomesh_guided.auto_profile))
        except Exception:  # no browser: the zip is still saved
            pass
        guided.state["message"] = f"Report saved to {out['path']}. Attach it to the issue form. {WARNING}"
        self.report({"INFO"}, guided.state["message"])
        bpy.ops.curiomesh.problem_saved("INVOKE_DEFAULT", path=out["path"])
        return {"FINISHED"}


class CURIOMESH_OT_problem_saved(bpy.types.Operator):
    bl_idname = "curiomesh.problem_saved"
    bl_label = "Zip saved"
    bl_description = "Shows where the zip for the issue form was saved"
    bl_options = {"INTERNAL"}

    path: bpy.props.StringProperty(options={"SKIP_SAVE"})

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=460)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="Zip saved. Attach it to the issue form yourself:")
        col.label(text=self.path)
        col.operator("wm.path_open", text="Open Folder").filepath = os.path.dirname(self.path)

    def execute(self, context):
        return {"FINISHED"}


# --------------------------------------------------------------------------- agent commands (read by agentcmd)

_IDS = {"type": "array", "items": {"type": "integer"}, "description": "Part numbers; without it the marked parts"}


def _cmd_export(context, parts=None, folder=None, note=""):
    return export_parts(parts, folder, note)


def _cmd_report(context, parts=None, path=None, note=""):
    return report_parts(parts, path, note)


COMMANDS = {
    "parts.export": {
        "fn": _cmd_export, "edits": False,
        "help": "Write the marked parts (or the given part numbers) with their borders, your original surface and the faces around them, as OBJ files and parts.json, for an agent to make better. Changes nothing.",
        "args": {"type": "object", "properties": {"parts": _IDS, "folder": {"type": "string", "description": "Parent folder; a dated folder is made inside it. Default: next to the .blend"}, "note": {"type": "string"}}},
    },
    "parts.report": {
        "fn": _cmd_report, "edits": False,
        "help": "Save the same parts plus report.md and settings.json in one zip for an issue report. The zip holds your mesh pieces: check it before sharing. No network.",
        "args": {"type": "object", "properties": {"parts": _IDS, "path": {"type": "string", "description": "The .zip to write. Default: next to the .blend"}, "note": {"type": "string"}}},
    },
}

CLASSES = (CURIOMESH_OT_parts_export, CURIOMESH_OT_parts_report, CURIOMESH_OT_send_problem, CURIOMESH_OT_problem_saved)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
