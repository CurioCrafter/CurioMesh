# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helpers for the CurioMesh agent scripts (curiomesh_run.py, curiomesh_fix.py).

The pure functions at the top need no Blender and are unit tested. Everything that touches bpy is below and imports it lazily.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

USES = ("AUTO", "LOW_POLY", "GAME", "ANIMATION", "SUBDIV", "SCAN", "SCULPT")
ALIASES = {"LOWPOLY": "LOW_POLY", "LOW": "LOW_POLY", "SUBDIVISION": "SUBDIV", "SCANCLEANUP": "SCAN", "SCAN_CLEANUP": "SCAN", "GAMEASSET": "GAME", "GAME_ASSET": "GAME", "ANIM": "ANIMATION", "SCULPT_BASE": "SCULPT"}
FIXES = ("CLEAN", "FLIP", "UNTANGLE", "RELAX", "PATCH", "ALL")
FIX_ALIASES = {"CLEAN_UP": "CLEAN", "CLEANUP": "CLEAN", "FLIP_INWARD": "FLIP", "FIX_ALL": "ALL"}
DEFECT_KINDS = ("inward", "sunk", "crossing", "folded_over", "bent", "flat", "duplicate", "overlap")


# --------------------------------------------------------------------------- pure helpers


def parse_use(text: str) -> str:
    key = text.strip().upper().replace("-", "_").replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in USES:
        raise ValueError(f"unknown --use {text!r}; choose one of {', '.join(USES)}")
    return key


def parse_fixes(text: str) -> list[str]:
    out = []
    for part in text.split(","):
        key = part.strip().upper().replace("-", "_").replace(" ", "_")
        key = FIX_ALIASES.get(key, key)
        if key not in FIXES:
            raise ValueError(f"unknown fix {part!r}; choose from {', '.join(FIXES)}")
        out.append(key)
    return out


def parse_reduce(text: str | None):
    """'5000' -> ('COUNT', 5000); '25%' -> ('PERCENT', 25.0); None -> None."""
    if not text:
        return None
    text = text.strip()
    if text.endswith("%"):
        return "PERCENT", float(text[:-1])
    return "COUNT", int(float(text))


def parse_faces(text: str) -> list[int]:
    """'1,5,10-12' -> [1, 5, 10, 11, 12]."""
    ids: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def issues_summary(record: dict | None, top: int = 10) -> dict:
    """Totals per kind, the spot count, bad kinds per part, and the worst spots (no face lists)."""
    record = record or {}
    spots = record.get("spots", [])
    per_part: dict[str, dict[str, int]] = {}
    for s in spots:
        bucket = per_part.setdefault(str(s.get("part", -1)), {})
        for kind, count in s.get("kinds", {}).items():
            bucket[kind] = bucket.get(kind, 0) + int(count)
    worst = [{k: s.get(k) for k in ("part", "kinds", "face_count", "score", "center", "radius")} for s in spots[:top]]
    return {"totals": dict(record.get("totals", {})), "spots": len(spots), "per_part": per_part, "worst": worst}


def defect_faces(summary: dict) -> int:
    """Sum of the defect kinds (not untidy topology such as poles). A face with two kinds counts twice, so this is an upper bound."""
    return sum(int(v) for k, v in summary.get("totals", {}).items() if k in DEFECT_KINDS)


def verdict(report: dict) -> dict:
    """Yes/no on the numbers the README calls good. A guide, not a rule."""
    faces = max(int(report.get("faces", 0)), 1)
    asked = float(report.get("detail", 0) or 0)
    changed = report.get("view_changed_pct")
    notes = []
    if changed is not None and asked and changed > 1.5 * asked + 1:
        notes.append(f"view changed {changed:.1f}% is well above the Detail asked ({asked:g}%)")
    bad = defect_faces(report.get("issues", {}))
    if bad > 0.005 * faces:
        notes.append(f"{bad} defect faces is more than 0.5% of {faces}")
    if report.get("tris", 0) > 0.02 * faces and report.get("output") != "TRIS":
        notes.append("more than 2% triangles left")
    return {"ok": not notes, "problems": notes}


def better(before: dict, after: dict) -> bool:
    """True when `after` has strictly fewer bad faces than `before` (dicts from count_bad)."""
    return after["bad"] < before["bad"]


def write_json(path: str | None, data: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print("CURIOMESH_REPORT " + json.dumps(data, default=str), flush=True)


def script_args() -> list[str]:
    """Arguments after the first `--` on Blender's command line."""
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


# --------------------------------------------------------------------------- Blender side


def load_addon(path: str | None = None) -> str:
    """Enable CurioMesh from a repo folder (--addon or $CURIOMESH_REPO) or the installed extension. Returns the package name."""
    import bpy

    path = path or os.environ.get("CURIOMESH_REPO")
    if path:
        path = os.path.abspath(path)
        sys.path.insert(0, os.path.dirname(path))
        name = os.path.basename(path)
        importlib.import_module(name).register()
        return name
    tries = ["curiomesh"]
    ext_root = bpy.utils.user_resource("EXTENSIONS")
    if ext_root and os.path.isdir(ext_root):
        tries += [f"bl_ext.{d}.curiomesh" for d in sorted(os.listdir(ext_root)) if os.path.isdir(os.path.join(ext_root, d, "curiomesh"))]
    tries += ["bl_ext.user_default.curiomesh", "bl_ext.blender_org.curiomesh"]
    for module in dict.fromkeys(tries):
        try:
            bpy.ops.preferences.addon_enable(module=module)
        except Exception:
            continue
        if module in sys.modules and hasattr(bpy.context.scene, "curiomesh_guided"):
            return module
    raise RuntimeError("CurioMesh is not installed in this Blender. Install the add-on, or pass --addon <repo folder> / set CURIOMESH_REPO.")


def sub(package: str, module: str):
    return importlib.import_module(f"{package}.{module}")


def _call(new: str, old: str, path: str, **kw):
    import bpy

    namespace, name = new.split(".")
    op = getattr(getattr(bpy.ops, namespace), name, None) if hasattr(getattr(bpy.ops, namespace), name) else None
    if op is None:
        namespace, name = old.split(".")
        op = getattr(getattr(bpy.ops, namespace), name)
    op(filepath=path, **kw)


def _blend(path):
    import bpy

    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)
    for obj in dst.objects:
        if obj is not None and obj.type == "MESH":
            bpy.context.scene.collection.objects.link(obj)


def _npz(path):
    """Arrays co/loops/start/total, the layout of the repo's bench meshes."""
    import bpy
    import numpy as np

    d = np.load(path)
    loops, starts, totals = d["loops"].tolist(), d["start"].tolist(), d["total"].tolist()
    mesh = bpy.data.meshes.new(Path(path).stem)
    mesh.from_pydata(d["co"].tolist(), [], [tuple(loops[a : a + b]) for a, b in zip(starts, totals)])
    mesh.update()
    bpy.context.scene.collection.objects.link(bpy.data.objects.new(Path(path).stem, mesh))


IMPORTERS = {
    ".glb": lambda p: _call("import_scene.gltf", "import_scene.gltf", p),
    ".gltf": lambda p: _call("import_scene.gltf", "import_scene.gltf", p),
    ".obj": lambda p: _call("wm.obj_import", "import_scene.obj", p),
    ".fbx": lambda p: _call("import_scene.fbx", "import_scene.fbx", p),
    ".ply": lambda p: _call("wm.ply_import", "import_mesh.ply", p),
    ".stl": lambda p: _call("wm.stl_import", "import_mesh.stl", p),
    ".blend": _blend,
    ".npz": _npz,
}


def load_mesh(path: str, name: str):
    """Import a file and return ONE mesh object called `name` (all meshes in the file joined, transforms baked in)."""
    import bpy

    ext = Path(path).suffix.lower()
    if ext not in IMPORTERS:
        raise ValueError(f"unsupported input {ext}; use one of {', '.join(sorted(IMPORTERS))}")
    before = set(bpy.context.scene.objects)
    IMPORTERS[ext](os.path.abspath(path))
    new = [o for o in bpy.context.scene.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh found in {path}")
    for obj in meshes:
        for mod in list(obj.modifiers):  # a joined static mesh: drop rigs and everything else that would move it
            obj.modifiers.remove(mod)
        world = obj.matrix_world.copy()
        obj.parent = None
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        obj.data.transform(world)
        obj.matrix_world.identity()
    for obj in bpy.context.scene.objects:
        obj.select_set(obj in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    for other in [o for o in new if o is not obj and o.name in bpy.data.objects]:
        bpy.data.objects.remove(other, do_unlink=True)  # empties and cameras that came with the file
    obj.name = name
    obj.data.name = name
    return obj


def select_only(obj) -> None:
    import bpy

    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.context.scene.objects:
        o.select_set(o is obj)
    bpy.context.view_layer.objects.active = obj


def export_mesh(obj, path: str) -> str:
    import bpy

    select_only(obj)
    path = os.path.abspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(path).suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB" if ext == ".glb" else "GLTF_SEPARATE", use_selection=True)
    elif ext == ".fbx":
        bpy.ops.export_scene.fbx(filepath=path, use_selection=True)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True)
        else:
            bpy.ops.export_scene.obj(filepath=path, use_selection=True)
    elif ext == ".blend":
        bpy.ops.wm.save_as_mainfile(filepath=path)
    else:
        raise ValueError(f"unsupported output {ext}; use .glb, .fbx, .obj or .blend")
    return path


def mesh_stats(obj) -> dict:
    sizes = [len(p.vertices) for p in obj.data.polygons]
    return {"faces": len(sizes), "quads": sizes.count(4), "tris": sizes.count(3), "ngons": sum(1 for s in sizes if s > 4)}


def finish(guided, context, limit: int = 50) -> None:
    """Run the Auto job (and what it starts next) to the end. Never loop on state['work']: it does not clear headless."""
    for _ in range(limit):
        work = guided.state["work"]
        if work is None:
            return
        runner = work["runner"]
        while runner.busy:
            runner.step(0.04)
        guided._collect_work(context)
        after = guided.state["work"]
        if after is work or after is None or not after["runner"].busy:
            return
