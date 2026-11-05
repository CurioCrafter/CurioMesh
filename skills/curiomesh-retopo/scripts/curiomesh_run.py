# SPDX-License-Identifier: GPL-3.0-or-later
"""Retopologise a model with CurioMesh, headless, exactly as the Auto Segments panel does.

blender -b --factory-startup --python curiomesh_run.py -- IN --use GAME --detail 6 --out OUT.glb --report REPORT.json

IN: .obj .fbx .glb .gltf .blend .ply .stl .npz.  OUT: .glb .fbx .obj .blend.
--use AUTO lets CurioMesh pick the Use from the mesh (the Auto button); then --detail and --reduce-to are ignored.
Options: --detail PCT, --reduce-to N|PCT%, --keep-original-bad, --lods 50,25,12.5, --bake normal|normal_color, --bake-size 2048,
--addon REPO_FOLDER (else $CURIOMESH_REPO, else the installed add-on).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cm_agent as cm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(prog="curiomesh_run.py")
    ap.add_argument("input")
    ap.add_argument("--use", default="GAME")
    ap.add_argument("--detail", type=float, default=None, help="percent of the view that may change; lower keeps more faces")
    ap.add_argument("--reduce-to", default=None, help="face count (5000) or percent of the source faces (25%%)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--addon", default=None)
    ap.add_argument("--keep-original-bad", action="store_true", help="after Auto, give parts with problem spots their original faces back")
    ap.add_argument("--lods", default=None, help="Make LOD Set instead: three percents, e.g. 50,25,12.5. --out becomes the stem of NAME_LOD0..2")
    ap.add_argument("--bake", default="off", choices=["off", "normal", "normal_color"])
    ap.add_argument("--bake-size", default="2048")
    args = ap.parse_args(cm.script_args())
    started = time.time()
    use = cm.parse_use(args.use)
    reduce_to = cm.parse_reduce(args.reduce_to)

    import bpy

    package = cm.load_addon(args.addon)
    guided, segments = cm.sub(package, "guided"), cm.sub(package, "segments")
    cm.sub(package, "hardware").checkpoint_dir = lambda *a, **k: None  # never resume parts measured by another run
    name = Path(args.input).stem
    src = cm.load_mesh(args.input, name)
    ctx = bpy.context
    cm.select_only(src)
    gs = ctx.scene.curiomesh_guided
    gs.scope, gs.auto_update = "AUTO", False
    gs.bake_detail, gs.bake_size = args.bake.upper(), str(args.bake_size)
    source_faces = len(src.data.polygons)
    report = {"input": args.input, "use": use, "detail_asked": args.detail, "reduce_to": args.reduce_to, "source_faces": source_faces, "blender": bpy.app.version_string, "addon": package}

    if args.lods:
        gs.lod0_percent, gs.lod1_percent, gs.lod2_percent = [float(x) for x in args.lods.split(",")]
        if use != "AUTO":
            gs.auto_profile = use
        if args.detail is not None:
            gs.auto_detail = args.detail
        bpy.ops.curiomesh.guided_make_lods()
        for _ in range(3 * 50):
            if guided.state["lod"] is None:
                break
            cm.finish(guided, ctx)
        outs = []
        for level in range(3):
            obj = bpy.data.objects.get(f"{name}_LOD{level}")
            if obj is None:
                raise RuntimeError(f"LOD{level} was not made: {guided.state.get('message')}")
            path = cm.export_mesh(obj, str(Path(args.out).with_name(f"{Path(args.out).stem}_LOD{level}{Path(args.out).suffix}")))
            outs.append({"level": level, "file": path, "view_changed_pct": obj.get("curiomesh_view_change"), **cm.mesh_stats(obj)})
        report.update(lods=outs, lod_text=guided.state["lod_report"], seconds=round(time.time() - started, 1))
        cm.write_json(args.report, report)
        return 0

    if use == "AUTO":
        bpy.ops.curiomesh.guided_auto_all()  # picks the Use and settings from the mesh, then starts
        use = gs.auto_profile
        report["auto_why"] = gs.auto_why
    else:
        gs.auto_profile = use
        if args.detail is not None:
            gs.auto_detail = args.detail
        if reduce_to:
            gs.auto_aim = "REDUCE"
            gs.auto_reduce_by = reduce_to[0]
            if reduce_to[0] == "COUNT":
                gs.auto_reduce_count = reduce_to[1]
            else:
                gs.auto_reduce_percent = reduce_to[1]
        bpy.ops.curiomesh.guided_auto_start()
    cm.finish(guided, ctx)
    segments.ensure()
    base = guided.state["auto_base"]
    if base is None or guided.state["res"] is None:
        report.update(ok=False, error=guided.state.get("message") or "Auto produced no result")
        cm.write_json(args.report, report)
        return 1
    job_report = dict(base["report"])
    kept = 0
    before_keep = None
    if args.keep_original_bad:
        before_keep = cm.issues_summary(guided.state["res"].get("issues"))
        bad = segments.bad_parts()
        kept = segments.keep_original(ctx, bad) if bad else 0
        cm.finish(guided, ctx)
    res = guided.state["res"]
    bpy.ops.curiomesh.guided_apply()
    out_obj = ctx.view_layer.objects.active
    path = cm.export_mesh(out_obj, args.out)
    stats = cm.mesh_stats(out_obj)
    report.update(
        ok=True,
        use=use,
        detail=gs.auto_detail,
        output=gs.auto_output,
        out_file=path,
        out_object=out_obj.name,
        view_changed_pct=round(float(job_report["loss"]), 2),
        parts=job_report.get("pieces"),
        notes=job_report.get("notes"),
        seconds=round(time.time() - started, 1),
        issues=cm.issues_summary(res.get("issues")),
        **stats,
    )
    if args.keep_original_bad:
        report.update(kept_original_parts=kept, issues_before_keep=before_keep, view_changed_note="view_changed_pct is the reduced result's; kept parts are your own faces, unchanged")
    report["quad_pct"] = round(100.0 * stats["quads"] / max(stats["faces"], 1), 1)
    report["verdict"] = cm.verdict(report)
    cm.write_json(args.report, report)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:  # a failing run must still leave a report the agent can read
        import traceback

        traceback.print_exc()
        args = cm.script_args()
        path = args[args.index("--report") + 1] if "--report" in args else None
        cm.write_json(path, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        code = 1
    sys.exit(code)
