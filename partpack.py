# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
# main-only: agent tools (not in the extensions.blender.org build)
"""Writes the part exchange files (bench/tools_design/AGENT_TOOLS.md, section 3, format version 1.0). No bpy.

A part is a dict: id, name, kind, numbers ({faces, view_change_pct, tidiness}), issues (kind words), points, faces,
mats (one per face), border (closed loops of local vertex ids, each ending on its first id), source = (points, faces),
around = (points, faces). `border_points` in parts.json is one flat list: the loops' points in loop order, each loop
without its closing repeat.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile

import numpy as np

FORMAT = "curiomesh-parts"
VERSION = [1, 0]
RULES = "Keep every border vertex exactly where it is. Faces 3 to 64 corners. Quads preferred."
README = """CurioMesh part exchange, format 1.0
1. parts.json lists the parts. part_<id>.obj is a part's current mesh; part_<id>_source.obj is your original surface to follow.
2. part_<id>_around.obj is one ring of neighbouring faces for flow at the border. Read it, do not change it.
3. Make a better mesh for each part. Keep every border vertex exactly where parts.json puts it (see "border" and "border_points").
4. Faces may have 3 to 64 corners, quads preferred. Write material changes as usemtl m<index> lines.
5. Save each part as returned/part_<id>.obj and use Bring Back in CurioMesh. A part that fails a check is left out; the rest come in.
"""


def result_key(faces_count: int, border_points) -> str:
    """A short stable hash of the result: its face count and its border points (in any order)."""
    pts = np.asarray(border_points, dtype=np.float32).reshape(-1, 3)
    if len(pts):
        pts = pts[np.lexsort(pts.T[::-1])]
    return hashlib.sha1(str(int(faces_count)).encode() + pts.tobytes()).hexdigest()[:8]


def obj_text(points, faces, mats=None) -> str:
    """OBJ text. Coordinates go out as float32 with 9 significant digits, which reads back exactly."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in pts.astype(np.float64).tolist()]
    last = None
    for i, f in enumerate(faces):
        if mats is not None and len(mats) and int(mats[i]) != last:
            last = int(mats[i])
            lines.append(f"usemtl m{last}")
        lines.append("f " + " ".join(str(int(v) + 1) for v in f))
    return "\n".join(lines) + "\n"


def _write(path, text) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_parts(folder, object_name, parts, result_info, note="") -> str:
    """Write the folder. result_info: source_faces, source_verts, result_faces, result_key; optional made_by and skipped."""
    os.makedirs(os.path.join(folder, "returned"), exist_ok=True)
    entries = []
    for p in parts:
        i = int(p["id"])
        _write(os.path.join(folder, f"part_{i}.obj"), obj_text(p["points"], p["faces"], p.get("mats")))
        sp, sf = p.get("source") or ([], [])
        ap, af = p.get("around") or ([], [])
        _write(os.path.join(folder, f"part_{i}_source.obj"), obj_text(sp, sf))
        _write(os.path.join(folder, f"part_{i}_around.obj"), obj_text(ap, af))
        pts = np.asarray(p["points"], dtype=np.float32).reshape(-1, 3)
        loops = [[int(v) for v in loop] for loop in p["border"]]
        flat = [pts[v].astype(np.float64).tolist() for loop in loops for v in loop[:-1]]
        n = p.get("numbers", {})
        entries.append({
            "id": i, "name": p.get("name", f"Part {i + 1}"), "kind": p.get("kind", ""),
            "faces": int(n.get("faces", len(p["faces"]))), "view_change_pct": n.get("view_change_pct"), "tidiness": n.get("tidiness"),
            "issues": list(p.get("issues", [])), "border": loops, "border_points": flat,
            "materials": sorted({int(m) for m in p.get("mats", [])}),
        })
    manifest = {
        "format": FORMAT, "version": VERSION, "made_by": result_info.get("made_by", "CurioMesh"), "object": object_name,
        "result": {k: result_info.get(k) for k in ("source_faces", "source_verts", "result_faces", "result_key")},
        "units": "metres, object space, Z up", "rules": RULES, "parts": entries, "note": note or "",
    }
    if result_info.get("skipped"):
        manifest["skipped"] = list(result_info["skipped"])
    _write(os.path.join(folder, "parts.json"), json.dumps(manifest, indent=1))
    _write(os.path.join(folder, "README.txt"), README)
    return folder


def write_report_zip(path, folder, report_md, settings_json) -> str:
    """One zip: the part folder, report.md and settings.json."""
    if not isinstance(settings_json, str):
        settings_json = json.dumps(settings_json, indent=1, default=str)
    top = os.path.basename(os.path.normpath(folder))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(folder):
            rel = os.path.relpath(root, folder)
            base = top if rel == "." else f"{top}/{rel.replace(os.sep, '/')}"
            if not files and not dirs:
                z.writestr(base + "/", "")
            for name in files:
                z.write(os.path.join(root, name), f"{base}/{name}")
        z.writestr("report.md", report_md)
        z.writestr("settings.json", settings_json)
    return path
