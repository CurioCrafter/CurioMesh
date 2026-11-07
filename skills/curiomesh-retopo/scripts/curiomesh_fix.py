# SPDX-License-Identifier: GPL-3.0-or-later
"""Find and fix bad faces in a mesh with CurioMesh's Fix Selection tools, headless. Never leaves the mesh worse.

blender -b --factory-startup --python curiomesh_fix.py -- RESULT --original ORIGINAL --fix ALL --out FIXED.glb --report FIX.json

--fix: ALL, or a comma list in the order to try: CLEAN,FLIP,UNTANGLE,RELAX,PATCH (omit to only measure).
--faces 3,10-20: fix only these face numbers (default: every bad face).  --grow N: rings of faces checked around them (0 with --faces).
--original: your source mesh. Says which way is out; Flip Inward and Untangle need it.
Each step is kept only if it leaves fewer bad faces than before; otherwise it is undone. The list is run again while a round still
gains something, and stops when a round gains nothing (no round cap). bad_face_ids in the report feed --faces.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_agent as cm  # noqa: E402


def count_bad(obj, fix_select, context, limit=2000):
    """Defects of the whole mesh, by kind, using the same checks the Fix Selection panel uses."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    area = list(bm.faces)
    found, bad = fix_select.measure(area, fix_select._source(context, obj))
    out = {k: int(getattr(found, k).sum()) for k in cm.DEFECT_KINDS}
    out["bad"] = int(bad.sum())
    out["faces"] = len(area)
    out["bad_face_ids"] = [int(i) for i in bad.nonzero()[0][:limit]]
    bm.free()
    return out


def select_faces(obj, ids):
    """Select exactly these faces (none selected = the whole mesh) and go into Edit Mode."""
    import bpy
    import numpy as np

    me = obj.data
    n = len(me.polygons)
    face = np.zeros(n, bool)
    vert = np.zeros(len(me.vertices), bool)
    if ids:
        face[[i for i in ids if 0 <= i < n]] = True
        verts = np.array([v for i in np.nonzero(face)[0] for v in me.polygons[int(i)].vertices], dtype=int)
        vert[verts] = True
    me.vertices.foreach_set("select", vert)
    me.polygons.foreach_set("select", face)
    me.edges.foreach_set("select", np.zeros(len(me.edges), bool))
    me.update()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="VERT")
    if not ids:
        bpy.ops.mesh.select_all(action="DESELECT")


def fix_rounds(fixes, first, do_step):
    """Keep going while the bad-face count drops: run the whole fix list again after every round that gained something,
    and stop when a round gains nothing or no bad face is left. No round cap. do_step(key, current, fresh) returns
    (record, new_count_or_None); None means the step was undone."""
    current, steps, rounds = first, [], 0
    while fixes and current["bad"] > 0:
        rounds += 1
        gained = False
        for key in fixes:
            record, after = do_step(key, current, current is first)
            record["round"] = rounds
            steps.append(record)
            if after is not None:
                current, gained = after, True
                if current["bad"] == 0:
                    break
        if not gained:
            break
    return current, steps, rounds


def main() -> int:
    ap = argparse.ArgumentParser(prog="curiomesh_fix.py")
    ap.add_argument("input")
    ap.add_argument("--original", default=None)
    ap.add_argument("--fix", default=None)
    ap.add_argument("--faces", default=None)
    ap.add_argument("--grow", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--addon", default=None)
    args = ap.parse_args(cm.script_args())
    fixes = cm.parse_fixes(args.fix) if args.fix else []
    ids = cm.parse_faces(args.faces) if args.faces else []

    import bpy

    package = cm.load_addon(args.addon)
    fix_select = cm.sub(package, "fix_select")
    ctx = bpy.context
    obj = cm.load_mesh(args.input, "Result")
    if args.original:
        original = cm.load_mesh(args.original, "Original")
        original.hide_set(True)
        ctx.scene.curiomesh_fix.source = original
    ctx.scene.curiomesh_fix.grow = args.grow if args.grow is not None else (0 if ids else 1)

    first = count_bad(obj, fix_select, ctx)

    def do_step(key, current, fresh):
        """One fix on the real mesh. Returns (record, count if kept else None)."""
        cm.select_only(obj)
        backup = obj.data.copy()
        select_faces(obj, ids if fresh else [])  # the face numbers only mean something before the first change
        try:
            bpy.ops.curiomesh.fix_run(how=key)
        except Exception as exc:
            bpy.ops.object.mode_set(mode="OBJECT")
            obj.data = backup
            return {"fix": key, "kept": False, "error": str(exc).strip()[:300]}, None
        bpy.ops.object.mode_set(mode="OBJECT")
        after = count_bad(obj, fix_select, ctx)
        keep = cm.better(current, after)
        record = {"fix": key, "kept": keep, "bad_before": current["bad"], "bad_after": after["bad"], "faces_after": after["faces"], "by_kind": {k: after[k] for k in cm.DEFECT_KINDS}}
        if keep:
            bpy.data.meshes.remove(backup)
            return record, after
        old = obj.data
        obj.data = backup
        bpy.data.meshes.remove(old)
        return record, None

    current, steps, rounds = fix_rounds(fixes, first, do_step)
    report = {
        "input": args.input,
        "original": args.original,
        "had_original": bool(args.original),
        "before": {k: v for k, v in first.items() if k != "bad_face_ids"},
        "after": {k: v for k, v in current.items() if k != "bad_face_ids"},
        "bad_face_ids": current["bad_face_ids"],
        "steps": steps,
        "rounds": rounds,
        "improved": cm.better(first, current),
        "ok": True,
    }
    if args.out:
        report["out_file"] = cm.export_mesh(obj, args.out)
    if Path(args.input).suffix.lower() in (".glb", ".gltf"):
        report["warning"] = "glTF stores triangles only: quads were split in two, so counts are on triangles. Use .obj or .blend between steps"
    if fixes and not args.original:
        report["note"] = "no --original given: Flip Inward and Untangle need it, so they could not run"
    cm.write_json(args.report, report)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        args = cm.script_args()
        path = args[args.index("--report") + 1] if "--report" in args else None
        cm.write_json(path, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        code = 1
    sys.exit(code)
