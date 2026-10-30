# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Headless real-mesh benchmark for the CurioMesh add-on.

Usage (one Blender instance, headless, no renders):

  blender --background --factory-startup --python scripts/real_mesh_benchmark.py -- \
      --manifest meshes.json --out results/ [--task mesh|batch|live|survey|aggregate] [--ids a,b,c]

Manifest: JSON list of {"id", "path", "type", "why"}; a .blend path may add "object" (mesh object name).
Source files are only read; nothing is saved over them and nothing is written next to them.
Every task writes <out>/<task>_<id>.json incrementally, so a timeout/crash keeps earlier results.
`--task aggregate` merges them into results.json + results.md.
"""

from __future__ import annotations

import faulthandler
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

import bpy

ADDON_ROOT = Path(__file__).resolve().parents[1]
SLOW_SECONDS = 300.0


def load_addon():
    for name in list(sys.modules):
        if name == "curiomesh" or name.startswith("curiomesh."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "curiomesh", ADDON_ROOT / "__init__.py", submodule_search_locations=[str(ADDON_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["curiomesh"] = module
    spec.loader.exec_module(module)
    module.register()
    return module


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = {"manifest": None, "out": ".", "task": "mesh", "ids": None, "extra": ""}
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-")
        args[key] = argv[i + 1]
        i += 2
    args["ids"] = args["ids"].split(",") if args["ids"] else None
    return args


# ----------------------------------------------------------------------------- scene / import


def reset_scene():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures, bpy.data.actions):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def import_mesh(entry: dict, name: str = "SRC"):
    """Import `entry['path']` read-only and return one joined mesh object called `name`."""
    path = entry["path"]
    ext = Path(path).suffix.lower()
    before = set(bpy.context.scene.objects)
    t0 = time.perf_counter()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            wanted = [entry["object"]] if entry.get("object") else list(src.objects)
            dst.objects = [n for n in wanted if n in src.objects]
        for obj in dst.objects:
            if obj is not None and obj.type == "MESH":
                bpy.context.scene.collection.objects.link(obj)
    else:
        raise ValueError(f"unsupported extension {ext}")
    import_s = time.perf_counter() - t0
    new = [o for o in bpy.context.scene.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh objects after import")
    view_layer = bpy.context.view_layer
    for obj in meshes:  # drop rigs so the joined mesh is a plain static one
        world = obj.matrix_world.copy()
        for mod in list(obj.modifiers):
            if mod.type == "ARMATURE":
                obj.modifiers.remove(mod)
        obj.parent = None
        obj.matrix_world = world
    view_layer.update()
    meshes.sort(key=lambda o: len(o.data.polygons), reverse=True)
    for obj in new:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    note = ""
    if len(meshes) > 1:
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in meshes:
                obj.select_set(True)
            view_layer.objects.active = meshes[0]
            bpy.ops.object.join()
        except Exception as exc:  # shape keys etc.: fall back to the biggest piece
            note = f"join failed ({exc}); using largest of {len(meshes)} meshes"
            for obj in meshes[1:]:
                bpy.data.objects.remove(obj, do_unlink=True)
    src = meshes[0]
    src.name = name
    src.data.name = name + "_Mesh"
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    view_layer.objects.active = src
    return src, import_s, note


def describe(src) -> dict:
    mesh = src.data
    import numpy as np

    nf = len(mesh.polygons)
    sizes = np.zeros(nf, dtype=np.int32)
    starts = np.zeros(nf, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    sizes = np.diff(np.append(starts, len(mesh.loops))) if nf else sizes
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    dims = (co.max(0) - co.min(0)).tolist() if len(co) else [0, 0, 0]
    return {
        "faces": nf,
        "verts": len(mesh.vertices),
        "tris": int((sizes == 3).sum()),
        "quads": int((sizes == 4).sum()),
        "ngons": int((sizes > 4).sum()),
        "materials": len([m for m in mesh.materials if m]),
        "uv_layers": len(mesh.uv_layers),
        "dims": [round(d, 4) for d in dims],
        "diag": round(float(math.sqrt(sum(d * d for d in dims))), 5),
        "scale": [round(s, 4) for s in src.scale],
        "rotation_euler": [round(r, 4) for r in src.rotation_euler],
    }


# ----------------------------------------------------------------------------- independent checks


def surface_check(src, out, samples=4000):
    """Independent distance check: out-vertices -> source surface and source-vertices -> out surface."""
    import numpy as np
    from mathutils.bvhtree import BVHTree

    if len(src.data.polygons) > 1_500_000:
        return {"skipped": "source > 1.5M faces"}
    dg = bpy.context.evaluated_depsgraph_get()
    t0 = time.perf_counter()
    bvh_src = BVHTree.FromObject(src, dg)
    bvh_out = BVHTree.FromObject(out, dg)
    build_s = time.perf_counter() - t0

    def sample(obj, n, into):
        """World-space vertex samples expressed in the local space of `into` (BVHTree.FromObject is local)."""
        verts = obj.data.vertices
        idx = np.random.default_rng(1).choice(len(verts), size=min(n, len(verts)), replace=False)
        to_local = into.matrix_world.inverted() @ obj.matrix_world
        return [to_local @ verts[int(i)].co for i in idx]

    def dists(points, bvh):
        out_d = []
        for p in points:
            hit = bvh.find_nearest(p)
            out_d.append(hit[3] if hit[0] is not None else float("inf"))
        return np.array(out_d)

    d_out = dists(sample(out, samples, src), bvh_src)
    d_src = dists(sample(src, samples, out), bvh_out)
    diag = max(describe(src)["diag"], 1e-9)
    # src->out is scaled by world matrix; describe() diag is object-space, fine for unit scale; report both raw and /diag.
    return {
        "bvh_build_s": round(build_s, 2),
        "out_to_src_p95_pct_diag": round(float(np.percentile(d_out, 95)) / diag * 100, 4),
        "out_to_src_max_pct_diag": round(float(d_out.max()) / diag * 100, 4),
        "src_to_out_p95_pct_diag": round(float(np.percentile(d_src, 95)) / diag * 100, 4),
        "src_to_out_max_pct_diag": round(float(d_src.max()) / diag * 100, 4),
    }


def output_checks(src, out, src_info):
    from curiomesh import overlay
    import numpy as np

    mesh = out.data
    st = overlay.compute_stats(mesh)
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return {
        "faces": st.faces,
        "quad_ratio": round(st.quad_ratio, 4),
        "tris": st.tris,
        "ngons": st.ngons,
        "poles": st.poles,
        "nan_verts": bool(not np.isfinite(co).all()),
        "materials": len([m for m in mesh.materials if m]),
        "uv_layers": len(mesh.uv_layers),
        "dims": [round(d, 4) for d in out.dimensions],
        "src_still_present": bpy.data.objects.get(src.name) is src,
        "src_faces_unchanged": len(src.data.polygons) == src_info["faces"],
        "out_is_src": out is src,
    }


# ----------------------------------------------------------------------------- tasks


def target_for(faces: int) -> int:
    return max(500, min(20000, int(round(faces * 0.07))))


def configure(settings, **kw):
    defaults = dict(
        quality="BALANCED",
        atlas_strategy="AUTO",
        output_mode="NEW",
        texture_mode="PROJECT",
        require_quads=True,
        seed=0,
        topology_policy="PRESERVE",
    )
    defaults.update(kw)
    for key, value in defaults.items():
        setattr(settings, key, value)


def run_engine(src, src_info, label, target, **cfg):
    from curiomesh.bridge import run_curiomesh
    from curiomesh.operators import config_from_owner

    settings = bpy.context.scene.curiomesh_settings
    configure(settings, target_faces=target, **cfg)
    rec = {"label": label, "target": target, "engine_requested": cfg.get("engine")}
    t0 = time.perf_counter()
    try:
        res = run_curiomesh(src, config_from_owner(settings))
    except Exception:
        rec.update(success=False, exception=traceback.format_exc(), time_s=round(time.perf_counter() - t0, 2))
        return rec
    wall = time.perf_counter() - t0
    d = res.diagnostics
    rec.update(
        success=bool(res.success),
        message=res.message,
        time_s=round(wall, 2),
        reported_ms=round(res.elapsed_ms, 0),
        engine=res.engine,
        strategy=res.strategy,
        profile=res.profile,
        candidates=res.candidate_count,
        faces=int(d.faces),
        faces_vs_target=round(d.faces / target, 3) if target else None,
        quad_ratio=round(float(d.quad_ratio), 4),
        tris=int(d.tris),
        ngons=int(d.ngons),
        poles=int(d.extraordinary),
        pole_ratio=round(float(d.extraordinary_ratio), 4),
        invalid_quads=int(d.invalid_quads),
        min_jacobian=round(float(d.min_scaled_jacobian), 3),
        p95_aspect=round(float(d.p95_aspect_ratio), 2),
        non_manifold_edges=int(d.non_manifold_edges),
        boundary_loops=int(d.boundary_loops),
        components=int(d.connected_components),
        geometry_rms=round(res.geometry_rms, 6),
        geometry_p95=round(res.geometry_p95, 6),
        geometry_max=round(res.geometry_max, 6),
        sampling_status=res.sampling_status,
        fidelity_status=res.fidelity_status,
        uv_status=res.uv_status,
        uv_valid=bool(d.uv_valid),
        materials_preserved=bool(res.materials_preserved),
        topology_status=res.topology_status,
        quad_only_verified=bool(res.quad_only_verified),
        slow=wall > SLOW_SECONDS,
        candidate_table=[
            {k: c.get(k) for k in ("strategy", "label", "valid", "reason", "faces", "quad_ratio", "target_error",
                                  "geometry_p95", "min_scaled_jacobian", "p95_aspect_ratio", "elapsed_ms")}
            for c in (res.candidates or [])
        ],
        out_location_offset=(
            [round(v, 3) for v in (res.output_object.location - src.location)] if res.output_object is not None and res.output_object is not src else None
        ),
    )
    out = res.output_object
    if out is not None:
        try:
            rec["independent"] = output_checks(src, out, src_info)
            # add-on geometry_* values are already fractions of the source diagonal
            rec["reported_p95_pct_diag"] = round(res.geometry_p95 * 100, 4)
            rec["reported_max_pct_diag"] = round(res.geometry_max * 100, 4)
            if out is not src:
                rec["surface_check"] = surface_check(src, out)
            rec["heat"] = heat_checks(out)
        except Exception:
            rec["check_exception"] = traceback.format_exc()
        if out is not src:
            mesh = out.data
            bpy.data.objects.remove(out, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        else:
            rec["output_is_source"] = True
    bpy.context.view_layer.objects.active = src
    src.select_set(True)
    return rec


def heat_checks(obj) -> dict:
    """overlay stats / heatmap arrays / HUD model on `obj`, plus the >300k guard decision."""
    from curiomesh import overlay

    rec = {}
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    overlay.invalidate()
    t0 = time.perf_counter()
    stats = overlay.get_stats(obj)
    rec["stats_s"] = round(time.perf_counter() - t0, 3)
    if stats is None:
        rec["error"] = "get_stats returned None"
        return rec
    nf = len(obj.data.polygons)
    rec.update(faces=stats.faces, tris=stats.tris, quads=stats.quads, ngons=stats.ngons, poles=stats.poles)
    rec["counts_consistent"] = stats.faces == nf and stats.tris + stats.quads + stats.ngons == nf
    rec["pole_plausible"] = 0 <= stats.poles <= stats.verts
    rec["draw_guard_skips"] = stats.faces > overlay.HEATMAP_MAX_FACES
    t0 = time.perf_counter()
    arrays = overlay.build_heat_arrays(obj, stats)
    rec["heat_arrays_s"] = round(time.perf_counter() - t0, 3)
    tri_pos, tri_col, pt_pos, pt_col = arrays
    rec["heat_tri_verts"] = len(tri_pos)
    rec["heat_points"] = len(pt_pos)
    rec["heat_sizes_ok"] = len(tri_pos) == len(tri_col) and len(pt_pos) == len(pt_col)
    rec["heat_tri_expected"] = (stats.tris + 0) * 3  # lower bound (n-gons add more)
    model = overlay.hud_model(bpy.context)
    rec["hud"] = {k: model[k] for k in ("name", "ratio", "faces", "tris", "poles", "info", "hint")}
    return rec


EXTRA = ""  # "atlas": only Atlas-Q Auto; "repair": only run Atlas-Q with topology_policy=REPAIR (what Analyze recommends for dirty scans)


def task_mesh(entry, out_dir):
    from curiomesh import atlasq

    rec = {"id": entry["id"], "path": entry["path"], "type": entry.get("type"), "why": entry.get("why")}
    reset_scene()
    try:
        src, import_s, note = import_mesh(entry)
    except Exception:
        rec["import_exception"] = traceback.format_exc()
        return rec
    rec["import_s"] = round(import_s, 2)
    rec["import_note"] = note
    info = describe(src)
    rec["source"] = info
    rec["target"] = target = target_for(info["faces"])
    settings = bpy.context.scene.curiomesh_settings

    # Analyze (operator + direct profile)
    print(f"[{entry['id']}] imported {info['faces']} faces; analyze...", flush=True)
    t0 = time.perf_counter()
    try:
        configure(settings, target_faces=target)
        ret = bpy.ops.curiomesh.analyze()
        prof = atlasq.analyze_object(src, target, max_sample_faces=settings.analysis_sample_faces)
        rec["analyze"] = {
            "operator": sorted(ret),
            "time_s": round(time.perf_counter() - t0, 2),
            "category": prof.category,
            "strategy": prof.recommended_strategy,
            "exact": prof.exact_topology,
            "boundary_loops": prof.boundary_loops,
            "components": prof.connected_components,
            "non_manifold_edges": prof.non_manifold_edges,
            "duplicate_faces": prof.duplicate_faces,
            "degenerate_ratio": round(prof.degenerate_ratio, 5),
            "sharp_ratio": round(prof.sharp_ratio, 4),
            "triangle_ratio": round(prof.triangle_ratio, 4),
            "texture_critical": prof.texture_critical,
            "warnings": list(prof.warnings),
            "target_edge": round(prof.target_edge_length, 6),
        }
    except Exception:
        rec["analyze"] = {"exception": traceback.format_exc()}

    # Heatmap / HUD on the source too (shows the >300k guard and cost on big meshes)
    try:
        rec["source_heat"] = heat_checks(src)
    except Exception:
        rec["source_heat"] = {"exception": traceback.format_exc()}

    rec["runs"] = []
    slow = False
    pc = ("pairclosure", dict(engine="TRIAD_Q_LITE", quality="BALANCED", require_quads=True))
    plan = (
        (pc,)
        if EXTRA == "pc"
        else (("atlas_q_balanced", dict(engine="ATLAS_Q", quality="BALANCED")),)
        if EXTRA == "atlas"
        else (("pairclosure_notex", dict(pc[1], texture_mode="NONE")),)
        if EXTRA == "pcnotex"
        else (("atlas_q_noreq", dict(engine="ATLAS_Q", quality="BALANCED", require_quads=False)),)
        if EXTRA == "norq"
        else (("atlas_q_repair", dict(engine="ATLAS_Q", quality="BALANCED", topology_policy="REPAIR")),)
        if EXTRA == "repair"
        else (
            ("atlas_q_balanced", dict(engine="ATLAS_Q", quality="BALANCED")),
            pc,
        )
    )
    for label, cfg in plan:
        if slow:
            rec["runs"].append({"label": label, "skipped": "previous run on this mesh was slow (>5 min)"})
            continue
        print(f"[{entry['id']}] {label} start", flush=True)
        run = run_engine(src, info, label, target, **cfg)
        print(f"[{entry['id']}] {label} done {run.get('time_s')}s success={run.get('success')}", flush=True)
        rec["runs"].append(run)
        slow = bool(run.get("slow"))
        (Path(out_dir) / f"mesh_{entry['id']}{'_' + EXTRA if EXTRA else ''}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def task_batch(entries, out_dir):
    rec = {"ids": [e["id"] for e in entries]}
    reset_scene()
    names = []
    x = 0.0
    for i, entry in enumerate(entries):
        # import one at a time; rename; keep earlier ones
        src, _, _ = import_mesh(entry, f"BATCH_{i}")
        src.location.x = x
        x += max(src.dimensions.x, 1.0) * 1.5
        names.append(src.name)
    settings = bpy.context.scene.curiomesh_settings
    configure(settings, target_faces=1000, engine="TRIAD_Q_LITE", quality="BALANCED", require_quads=True)
    bpy.ops.object.select_all(action="DESELECT")
    for n in names:
        bpy.data.objects[n].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[names[0]]
    rec["source_faces"] = {n: len(bpy.data.objects[n].data.polygons) for n in names}
    t0 = time.perf_counter()
    try:
        ret = bpy.ops.curiomesh.batch_remesh()
        rec["operator"] = sorted(ret)
    except Exception:
        rec["exception"] = traceback.format_exc()
        return rec
    rec["time_s"] = round(time.perf_counter() - t0, 2)
    rec["status"] = settings.batch_status
    path = Path(settings.batch_report_path) if settings.batch_report_path else None
    rec["report_path"] = str(path) if path else ""
    if path and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        rec["report"] = data
        rec["report_rows_ok"] = [r["success"] for r in data["objects"]]
        rec["report_object_count_matches"] = len(data["objects"]) == len(names)
        rec["report_total_vs_sum_ms"] = [data["total_seconds"], round(sum(r["time_ms"] for r in data["objects"]) / 1000, 2)]
        path.unlink()
    else:
        rec["report_missing"] = True
    rec["ui_rows"] = [(i.name, i.output, i.faces, round(i.quad_ratio, 3), i.ok, i.error) for i in settings.batch_results]
    rec["output_objects_exist"] = [bool(bpy.data.objects.get(r.output)) for r in settings.batch_results]
    # Overlay must not treat the preview/object names badly; HUD on each output
    for row in settings.batch_results:
        out = bpy.data.objects.get(row.output)
        if out is not None:
            try:
                rec.setdefault("heat", {})[row.output] = heat_checks(out)
            except Exception:
                rec.setdefault("heat", {})[row.output] = {"exception": traceback.format_exc()}
    # re-run with the preview object selected: it must be excluded
    return rec


def task_live(entries, out_dir):
    from curiomesh import live, overlay

    rec = {}
    settings = bpy.context.scene.curiomesh_settings
    # small mesh
    small, big = entries[0], entries[1]
    reset_scene()
    src, _, _ = import_mesh(small)
    n = len(src.data.polygons)
    rec["small"] = {"id": small["id"], "faces": n}
    configure(settings, target_faces=max(200, min(2000, n)), engine="TRIAD_Q_LITE", require_quads=True)
    try:
        t0 = time.perf_counter()
        settings.live_enabled = True  # toggles timer; background never ticks it
        ok = live.run_preview(bpy.context)
        rec["small"].update(ok=ok, status=live.status(), time_s=round(time.perf_counter() - t0, 2))
        prev = bpy.data.objects.get(overlay.PREVIEW_NAME)
        rec["small"]["preview_exists"] = prev is not None
        if prev is not None:
            rec["small"]["preview_faces"] = len(prev.data.polygons)
            rec["small"]["preview_materials"] = len([m for m in prev.data.materials if m])
            rec["small"]["preview_quad_ratio"] = overlay.get_stats(prev).quad_ratio
            rec["small"]["preview_x_offset_ok"] = prev.location.x > src.location.x
            rec["small"]["selection_restored"] = bpy.context.object is src and src.select_get() and not prev.select_get()
        t0 = time.perf_counter()
        live.run_preview(bpy.context)
        rec["small"]["second_run_s"] = round(time.perf_counter() - t0, 2)
        rec["small"]["previews_after_second"] = len([o for o in bpy.data.objects if o.name.startswith(overlay.PREVIEW_NAME)])
        # signature cost (runs on every geometry update while Live is on)
        t0 = time.perf_counter()
        live._signature(src)
        rec["small"]["signature_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        # heatmap target selection with live preview
        settings.heatmap_enabled = True
        rec["small"]["heatmap_targets"] = [o.name for o in overlay.heatmap_targets(bpy.context, settings)]
        settings.heatmap_enabled = False
    except Exception:
        rec["small"]["exception"] = traceback.format_exc()
    finally:
        try:
            settings.live_enabled = False
        except Exception:
            pass
    rec["small"]["preview_removed_after_off"] = bpy.data.objects.get(overlay.PREVIEW_NAME) is None
    rec["small"]["timer_left_registered"] = bpy.app.timers.is_registered(live._tick)

    # big mesh: cap message
    reset_scene()
    src, _, _ = import_mesh(big)
    n = len(src.data.polygons)
    rec["big"] = {"id": big["id"], "faces": n}
    try:
        settings.live_enabled = True
        t0 = time.perf_counter()
        ok = live.run_preview(bpy.context)
        rec["big"].update(ok=ok, status=live.status(), time_s=round(time.perf_counter() - t0, 3))
        t0 = time.perf_counter()
        live._signature(src)
        rec["big"]["signature_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rec["big"]["preview_exists"] = bpy.data.objects.get(overlay.PREVIEW_NAME) is not None
    except Exception:
        rec["big"]["exception"] = traceback.format_exc()
    finally:
        settings.live_enabled = False
    # edit-mode and non-mesh messages
    reset_scene()
    live.state["source"] = ""
    rec["no_mesh_status"] = (live.run_preview(bpy.context), live.status())
    return rec


def task_survey(entry, out_dir):
    reset_scene()
    try:
        src, imp, note = import_mesh(entry)
        info = describe(src)
        info.update(import_s=round(imp, 1), note=note)
        from curiomesh import atlasq

        prof = atlasq.analyze_object(src, target_for(info["faces"]))
        info.update(category=prof.category, boundary_loops=prof.boundary_loops, components=prof.connected_components,
                    nm=prof.non_manifold_edges)
        return {"id": entry["id"], **info}
    except Exception as exc:
        return {"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------------------- aggregate


def aggregate(out_dir: Path):
    meshes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("mesh_*.json"))]
    extra = {}
    for task in ("batch", "live"):
        p = out_dir / f"{task}.json"
        if p.is_file():
            extra[task] = json.loads(p.read_text(encoding="utf-8"))
    (out_dir / "results.json").write_text(json.dumps({"meshes": meshes, **extra}, indent=2), encoding="utf-8")
    lines = [
        "| mesh | src faces | engine | ok | time s | faces/target | quad | tris | poles | p95 err %diag (rep/indep) | max %diag | UV | mats | topology |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in meshes:
        src = m.get("source", {})
        for r in m.get("runs", []):
            if "skipped" in r:
                lines.append(f"| {m['id']} | {src.get('faces')} | {r['label']} | skipped | | | | | | | | | | |")
                continue
            ind = r.get("surface_check", {})
            lines.append(
                f"| {m['id']} | {src.get('faces')} | {r['label']} | {'PASS' if r.get('success') else 'FAIL'} | {r.get('time_s')} | "
                f"{r.get('faces')}/{r.get('target')} ({r.get('faces_vs_target')}) | {r.get('quad_ratio')} | {r.get('tris')} | {r.get('poles')} | "
                f"{r.get('reported_p95_pct_diag')}/{ind.get('out_to_src_p95_pct_diag')} | {r.get('reported_max_pct_diag')}/{ind.get('out_to_src_max_pct_diag')} | "
                f"{r.get('uv_status')} | {r.get('materials_preserved')} | {r.get('topology_status')} |"
            )
            if not r.get("success"):
                lines.append(f"|  | | | | | msg: {str(r.get('message') or r.get('exception'))[:160]} | | | | | | | | |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main


def main():
    args = parse_args()
    faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)  # shows where a slow run is stuck
    out_dir = Path(args["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args["task"] == "aggregate":
        aggregate(out_dir)
        return
    manifest = json.loads(Path(args["manifest"]).read_text(encoding="utf-8"))
    if args["ids"]:
        by_id = {e["id"]: e for e in manifest}
        manifest = [by_id[i] for i in args["ids"]]
    global EXTRA
    EXTRA = args["extra"]
    load_addon()
    task = args["task"]
    try:
        if task == "mesh":
            for entry in manifest:
                print(f"=== mesh {entry['id']}", flush=True)
                try:
                    rec = task_mesh(entry, out_dir)
                except Exception:
                    rec = {"id": entry["id"], "task_exception": traceback.format_exc()}
                (out_dir / f"mesh_{entry['id']}{'_' + EXTRA if EXTRA else ''}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        elif task == "batch":
            rec = task_batch(manifest, out_dir)
            (out_dir / "batch.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        elif task == "live":
            rec = task_live(manifest, out_dir)
            (out_dir / "live.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        elif task == "survey":
            rows = []
            for entry in manifest:
                row = task_survey(entry, out_dir)
                print("SURVEY", json.dumps(row), flush=True)
                rows.append(row)
            (out_dir / "survey.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        (out_dir / f"{task}_FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")


main()
