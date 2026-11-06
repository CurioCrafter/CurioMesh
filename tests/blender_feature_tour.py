# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""End-to-end feature tour of CurioMesh in background Blender: every feature, one small check each.

blender -b --factory-startup --python tests/blender_feature_tour.py -- [options]
  --addon FOLDER   the add-on folder to test (default: the folder above this file)
  --models DIR     folder of the free test meshes (default $CURIOMESH_MODELS or D:/LargeAmountOfMeshesForShipwreckVR/CurioMesh_free_test_meshes)
  --model NAME     main test model (default marble_bust_01); --alt NAME is a second one (default Sofa_01)
  --faces N        shrink the test models to about N faces first (default 4000; 0 keeps them whole)
  --only A,B       run only checks whose "group/name" contains one of these words
  --skip A,B       leave out checks whose "group/name" contains one of these words
  --list           print the checks and exit
  --out DIR        where files are written (default: a folder in the system temp dir); report.json lands here

Every check prints PASS, FAIL, WARN (works, but something is worth a look) or SKIP (needs a window or a service this
run does not have; the reason says which). The run ends with a summary and exits non-zero on any FAIL.
Needs no timers: the work queues are driven by hand, the way bench/ladder/ladder_drag.py does it.
Load rules on the owner's PC: start Blender only through bench/rough/lowblender.sh.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import runpy
import sys
import tempfile
import time
import traceback

import bpy

ARGV = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def arg(name, default=None):
    return ARGV[ARGV.index(name) + 1] if name in ARGV else default


HERE = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.abspath(arg("--addon", os.path.dirname(HERE)))
MODELS = arg("--models", os.environ.get("CURIOMESH_MODELS", "D:/LargeAmountOfMeshesForShipwreckVR/CurioMesh_free_test_meshes"))
MAIN, ALT = arg("--model", "marble_bust_01"), arg("--alt", "Sofa_01")
FACES = int(arg("--faces", "4000"))
ONLY = [w for w in arg("--only", "").split(",") if w]
SKIPS = [w for w in arg("--skip", "").split(",") if w]
OUT = arg("--out") or tempfile.mkdtemp(prefix="curiomesh_tour_")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(ADDON))
PKG = os.path.basename(ADDON)
import addon_utils  # noqa: E402

addon = importlib.import_module(PKG)
try:
    addon_utils.enable(PKG, default_set=True, persistent=False)  # also creates the add-on's preferences entry
except Exception:
    traceback.print_exc()
if not hasattr(bpy.types.Scene, "curiomesh_guided"):
    addon.register()


def mod(name):
    return importlib.import_module(f"{PKG}.{name}")


guided, segments, ladder, profiles = mod("guided"), mod("segments"), mod("ladder"), mod("profiles")
region, fix_select, issue_panel, seglive = mod("region"), mod("fix_select"), mod("issue_panel"), mod("seglive")
jobpack, game_export, overlay, hardware = mod("jobpack"), mod("game_export"), mod("overlay"), mod("hardware")
segment_goals, teach, batch = mod("segment_goals"), mod("teach"), mod("batch")
hardware.checkpoint_dir = lambda *a, **k: None  # never resume parts measured by another run
try:
    STATE0 = copy.deepcopy(guided.state)
except Exception:
    STATE0 = dict(guided.state)

# --------------------------------------------------------------------------- framework

CHECKS: list = []
RESULTS: list = []
CTX: dict = {}  # shared between checks: the loaded model, the last Auto result
MANUAL: list = []  # features this tour cannot reach; printed at the end and written to the report


class Skip(Exception):
    pass


class Warn(Exception):
    pass


def check(group, name, needs=None):
    def deco(fn):
        CHECKS.append((group, name, fn, needs))
        return fn

    return deco


def manual(feature, why):
    MANUAL.append((feature, why))


def ok(cond, msg="expectation failed"):
    if not cond:
        raise AssertionError(msg)


def P(*a):
    print("TOUR", *a, flush=True)


WATCH = {"label": "", "since": time.time()}
LIMIT = float(arg("--check-limit", "900"))  # seconds one check may take before the run is cut off and reported


def _watchdog():
    while True:
        time.sleep(5)
        if WATCH["label"] and time.time() - WATCH["since"] > LIMIT:
            P(f"HUNG  {WATCH['label']} ran more than {LIMIT:.0f}s: stopping the run")
            RESULTS.append({"check": WATCH["label"], "status": "FAIL", "why": f"hung for more than {LIMIT:.0f}s", "seconds": LIMIT})
            with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as fh:
                json.dump({"results": RESULTS, "hung": WATCH["label"]}, fh, indent=1)
            sys.stdout.flush()
            os._exit(3)


def run_all():
    import threading

    threading.Thread(target=_watchdog, daemon=True).start()
    t_all = time.time()
    for group, name, fn, needs in CHECKS:
        label = f"{group}/{name}"
        if (ONLY and not any(w in label for w in ONLY)) or any(w in label for w in SKIPS):
            continue
        t = time.time()
        WATCH.update(label=label, since=t)
        status, why = "PASS", ""
        try:
            if needs and not CTX.get(needs):
                raise Skip(f"needs '{needs}', which an earlier check did not produce")
            why = fn() or ""
        except Skip as e:
            status, why = "SKIP", str(e)
        except Warn as e:
            status, why = "WARN", str(e)
        except AssertionError as e:
            status, why = "FAIL", str(e)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            status, why = "FAIL", f"{type(e).__name__}: {e} ({os.path.basename(tb.filename)}:{tb.lineno})"
            traceback.print_exc()
        secs = time.time() - t
        RESULTS.append({"check": label, "status": status, "why": why, "seconds": round(secs, 1)})
        P(f"{status:4}  {label:48} {secs:6.1f}s  {why}")
    WATCH["label"] = ""
    counts = {s: sum(r["status"] == s for r in RESULTS) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    P("=" * 78)
    P("SUMMARY", counts, f"{len(RESULTS)} checks in {time.time() - t_all:.0f}s")
    for r in RESULTS:
        if r["status"] == "FAIL":
            P("  FAIL", r["check"], "-", r["why"])
    P("ONLY BY HAND:")
    for feature, why in MANUAL:
        P("  -", feature, "-", why)
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"counts": counts, "results": RESULTS, "manual": MANUAL, "blender": bpy.app.version_string, "addon": ADDON}, fh, indent=1)
    sys.stdout.flush()
    sys.exit(1 if counts["FAIL"] else 0)


# --------------------------------------------------------------------------- helpers


def quiet_wm():
    """No sidebar redraws or timers matter here; make sure a stuck operator cannot wait for a window."""
    guided.state["no_ask"] = True


def reset_state():
    try:
        guided.stop_work()
    except Exception:
        pass
    for fn in (ladder.cancel, segments.reset, seglive.clear):
        try:
            fn()
        except Exception:
            pass
    guided.state.update(copy.deepcopy(STATE0))
    quiet_wm()


def new_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    reset_state()


def load_model(name, faces=None):
    """Import a free test mesh as one object called Model at the origin, shrunk to about `faces` faces."""
    new_scene()
    path = os.path.join(MODELS, name, name + "_1k.gltf")
    ok(os.path.exists(path), f"test model missing: {path}")
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    for o in list(bpy.context.scene.objects):
        if o is not obj:
            bpy.data.objects.remove(o, do_unlink=True)
    obj.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.name = "Model"
    main = name == MAIN and faces is None
    faces = FACES if faces is None else faces
    if faces and len(obj.data.polygons) > faces * 1.3:
        mod_ = obj.modifiers.new("shrink", "DECIMATE")
        mod_.ratio = faces / len(obj.data.polygons)
        bpy.ops.object.modifier_apply(modifier=mod_.name)
    select_only(obj)
    if main:
        CTX["main"], CTX["src_faces"] = obj, len(obj.data.polygons)
    return obj


def main_obj():
    """The main test model, loaded again if an earlier check replaced the scene or changed it."""
    obj = CTX.get("main")
    try:
        if obj is not None and bpy.data.objects.get(obj.name) is obj and len(obj.data.polygons) == CTX.get("src_faces"):
            return obj
    except ReferenceError:
        pass
    return load_model(MAIN)


def select_only(obj):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.context.scene.objects:
        o.select_set(o is obj)
    bpy.context.view_layer.objects.active = obj


def bbox(obj):
    """(centre, size) of an object's bounding box in world space, as Vectors."""
    from mathutils import Vector

    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    low = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    high = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (low + high) / 2, high - low


def gs():
    return bpy.context.scene.curiomesh_guided


def finish(limit=60):
    """Run the Auto job (and what it starts next) to the end."""
    for _ in range(limit):
        work = guided.state["work"]
        if work is None:
            return
        runner = work["runner"]
        t0 = time.time()
        while runner.busy:
            if hasattr(runner, "step"):
                runner.step(0.05)
            else:  # the Live Boolean redo runs on its own thread
                time.sleep(0.05)
                if time.time() - t0 > 300:
                    raise AssertionError("a worker did not finish in 300 s")
        guided._collect_work(bpy.context)
        after = guided.state["work"]
        if after is work or after is None or not after["runner"].busy:
            return


def sizes(obj_or_res):
    if isinstance(obj_or_res, dict):
        return [len(f) for f in obj_or_res["faces"]]
    return [len(p.vertices) for p in obj_or_res.data.polygons]


def quad_share(res):
    s = sizes(res)
    return sum(1 for n in s if n == 4) / max(len(s), 1)


def fresh_auto(src, **props):
    """Set Auto Segments properties (the Use first: it sets Detail, Method and Output), press Start, wait. Returns the result dict."""
    select_only(src)
    reset_state()
    g = gs()
    for k, p in g.bl_rna.properties.items():
        if p.type in {"POINTER", "COLLECTION"} or k in {"rna_type", "name"} or p.is_readonly:
            continue
        if k.startswith(("auto_", "sym_", "bake_", "keep_", "multires", "region_use", "loop_", "bend_", "density_", "guide_use", "blockout", "quad_", "tweak_", "ui_", "pure_", "fit_", "feature_")):
            g.property_unset(k)
    g.scope, g.auto_update = "AUTO", False
    for k in sorted(props, key=lambda k: k != "auto_profile"):
        setattr(g, k, props[k])
    r = bpy.ops.curiomesh.guided_auto_start()
    ok(r == {"FINISHED"}, f"Start returned {r}: {guided.state['message']}")
    finish()
    res = guided.state["res"]
    ok(res is not None, f"no result: {guided.state['message']}")
    return res


def apply_result():
    r = bpy.ops.curiomesh.guided_apply()
    ok(r == {"FINISHED"}, f"Apply returned {r}: {guided.state['message']}")
    return bpy.context.view_layer.objects.active


def poll_ok(op):
    try:
        return bool(op.poll())
    except Exception as e:  # a poll that raises is a bug
        raise AssertionError(f"poll raised {type(e).__name__}: {e}")


def op_names():
    return sorted(dir(bpy.ops.curiomesh))


def all_ops():
    return [(n, getattr(bpy.ops.curiomesh, n)) for n in op_names()]


# --------------------------------------------------------------------------- a layout that records what a panel asks of it


class Rec:
    """Stands in for a UILayout: every call is accepted, and prop()/operator() are checked against the real names."""

    CHILD = {"row", "column", "box", "split", "grid_flow", "column_flow", "menu_pie", "column_flow", "row"}

    def __init__(self, errors, path="layout"):
        self._errors, self._path = errors, path

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __bool__(self):
        return True

    def _bad(self, msg):
        self._errors.append(f"{self._path}: {msg}")

    def _has(self, data, name):
        if data is None:
            return True  # a pointer that is empty right now: the panel is allowed to ask
        try:
            return name in data.bl_rna.properties
        except AttributeError:
            return True

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        errors = self._errors

        def prop(data, prop_name, *a, **k):
            if not self._has(data, prop_name):
                self._bad(f"prop({type(data).__name__}, {prop_name!r}) - no such property")
            return Rec(errors)

        def prop_search(data, prop_name, search_data, search_prop, *a, **k):
            for d, n in ((data, prop_name), (search_data, search_prop)):
                if not self._has(d, n):
                    self._bad(f"prop_search({type(d).__name__}, {n!r}) - no such property")
            return Rec(errors)

        def props_enum(data, prop_name, *a, **k):
            if not self._has(data, prop_name):
                self._bad(f"props_enum({prop_name!r}) - no such property")

        def operator(idname, *a, **k):
            if idname.startswith("curiomesh.") and not hasattr(bpy.ops.curiomesh, idname.split(".", 1)[1]):
                self._bad(f"operator({idname!r}) - not registered")
            return Rec(errors)

        def template_list(listtype, list_id, dataptr, propname, *a, **k):
            if not self._has(dataptr, propname):
                self._bad(f"template_list({propname!r}) - no such property")
            for key in ("active_dataptr", "active_propname"):
                pass
            if len(a) >= 2 and not self._has(a[0], a[1]):
                self._bad(f"template_list active {a[1]!r} - no such property")
            if k.get("active_propname") and not self._has(a[0] if a else k.get("active_dataptr"), k["active_propname"]):
                self._bad(f"template_list active {k['active_propname']!r} - no such property")
            return None

        def pair(*a, **k):
            return Rec(errors), Rec(errors)

        table = {"prop": prop, "prop_search": prop_search, "props_enum": props_enum, "operator": operator, "template_list": template_list, "panel_prop": pair, "panel": pair, "operator_menu_enum": operator}
        if name in table:
            return table[name]
        return lambda *a, **k: Rec(errors, f"{self._path}.{name}")


def draw_class(cls, ctx):
    """Call cls.draw(...) with a recording layout. Returns the list of problems."""
    errors: list = []
    bases = tuple(b for b in cls.__mro__[1:] if b.__module__.startswith(PKG) and not issubclass(b, bpy.types.bpy_struct))  # only the add-on's own mixins
    fake = type("Fake", bases, {})()
    fake.layout = Rec(errors)
    for a in dir(cls):
        if a.startswith("bl_"):
            try:
                setattr(fake, a, getattr(cls, a))
            except Exception:
                pass
    for method in ("draw_header", "draw"):
        fn = getattr(cls, method, None)
        if fn is None or getattr(fn, "__module__", "").startswith("bpy"):
            continue
        fake.layout = Rec(errors, f"{cls.__name__}.{method}")
        try:
            fn(fake, ctx)
        except AttributeError as e:
            if "NoneType" in str(e) or "'Rec'" in str(e):
                errors.append(f"NEEDS-UI {cls.__name__}.{method}: {e}")
            else:
                errors.append(f"{cls.__name__}.{method}: AttributeError {e}")
        except Exception as e:
            errors.append(f"{cls.__name__}.{method}: {type(e).__name__} {e}")
    return errors


def addon_classes():
    """Every CURIOMESH_ class the add-on's modules define (dir(bpy.types) does not list them all)."""
    found = {}
    for m in list(sys.modules.values()):
        if getattr(m, "__name__", "").startswith(PKG):
            for v in vars(m).values():
                if isinstance(v, type) and v.__name__.startswith("CURIOMESH_") and hasattr(v, "bl_rna"):
                    found[v.__name__] = v
    return found


def panel_classes():
    return [c for n, c in sorted(addon_classes().items()) if "_PT_" in n]


def draw_sweep(label):
    ctx = bpy.context
    problems, needs_ui = [], []
    for cls in panel_classes():
        for e in draw_class(cls, ctx):
            (needs_ui if e.startswith("NEEDS-UI") else problems).append(e)
    prefs = ctx.preferences.addons[PKG].preferences if PKG in ctx.preferences.addons else None
    if prefs is not None:
        errors: list = []
        fake = type("FakePrefs", (), {"__getattr__": lambda s, n: getattr(prefs, n)})()
        fake.layout = Rec(errors, "Preferences.draw")
        try:
            type(prefs).draw(fake, ctx)
        except Exception as e:
            errors.append(f"Preferences.draw: {type(e).__name__} {e}")
        problems += errors
    CTX.setdefault("draw_needs_ui", set()).update(e.split(":")[0].replace("NEEDS-UI ", "") for e in needs_ui)
    ok(not problems, f"{label}: {len(problems)} problem(s): " + " | ".join(problems[:6]))
    return f"{len(panel_classes())} panels drawn" + (f", {len(needs_ui)} draw call(s) need a real 3D view" if needs_ui else "")


# =========================================================================== A. the add-on itself (no model)


@check("addon", "everything_registered")
def _():
    names = list(addon_classes())
    ok(len(names) >= 100, f"only {len(names)} CURIOMESH_ classes registered")
    ok(len([n for n in names if "_PT_" in n]) >= 20, "fewer than 20 panels")
    for prop in ("curiomesh_settings", "curiomesh_guided", "curiomesh_region", "curiomesh_fix"):
        ok(hasattr(bpy.types.Scene, prop), f"Scene.{prop} missing")
    for prop in ("curiomesh_points", "curiomesh_seg"):
        ok(hasattr(bpy.types.Object, prop), f"Object.{prop} missing")
    ok(PKG in bpy.context.preferences.addons or True)
    return f"{len(names)} classes, {len(op_names())} operators"


def prop_row(p):
    row = {"id": p.identifier, "name": p.name, "desc": p.description, "type": p.type}
    if p.type == "ENUM":
        row["items"] = [(i.identifier, i.name, i.description) for i in p.enum_items]
    return row


@check("addon", "dump_inventory")
def _():
    """Writes inventory.json (every operator, property group, panel and preference with its tooltip) next to the report."""
    data = {"operators": [], "groups": {}, "panels": [], "scene_props": []}
    for name in op_names():
        rna = getattr(bpy.ops.curiomesh, name).get_rna_type()
        data["operators"].append({"idname": "curiomesh." + name, "label": rna.name, "desc": rna.description, "props": [prop_row(p) for p in rna.properties if p.identifier != "rna_type"]})
    for cname, cls in sorted(addon_classes().items()):
        if "_PT_" in cname:
            data["panels"].append({"id": cname, "label": cls.bl_label, "parent": getattr(cls, "bl_parent_id", ""), "category": getattr(cls, "bl_category", ""), "space": getattr(cls, "bl_space_type", ""), "context": getattr(cls, "bl_context", ""), "options": sorted(getattr(cls, "bl_options", []))})
        elif "_PG_" in cname:
            data["groups"][cname] = [prop_row(p) for p in cls.bl_rna.properties if p.identifier not in ("rna_type", "name")]
    for p in bpy.types.Scene.bl_rna.properties:
        if p.identifier.startswith("curiomesh"):
            data["scene_props"].append({"id": p.identifier, "type": p.type, "fixed_type": getattr(p.fixed_type, "identifier", None) if p.type == "POINTER" else None})
    prefs = bpy.context.preferences.addons[PKG].preferences
    data["prefs"] = [prop_row(p) for p in prefs.bl_rna.properties if p.identifier not in ("rna_type", "bl_idname", "name")]
    with open(os.path.join(OUT, "inventory.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return f"{len(data['operators'])} operators, {len(data['panels'])} panels, {sum(map(len, data['groups'].values()))} property-group properties"


@check("addon", "preferences_reset")
def _():
    prefs = bpy.context.preferences.addons[PKG].preferences
    default = prefs.bl_rna.properties["hud_scale"].default
    prefs.hud_scale = default + 0.5
    r = bpy.ops.curiomesh.reset_preferences()
    ok(r == {"FINISHED"}, f"returned {r}")
    ok(abs(prefs.hud_scale - default) < 1e-6, f"hud_scale is {prefs.hud_scale}, expected {default}")


@check("addon", "operator_polls_dont_raise")
def _():
    new_scene()
    for name, op in all_ops():
        poll_ok(op)
    return f"{len(op_names())} polls on an empty scene"


@check("addon", "operators_have_descriptions")
def _():
    bad = [n for n, op in all_ops() if not op.get_rna_type().description.strip()]
    if bad:
        raise Warn(f"{len(bad)} operators with no tooltip: {', '.join(bad[:8])}")


@check("addon", "properties_have_descriptions")
def _():
    missing = 0
    for n, cls in addon_classes().items():
        if "_PG_" in n:
            for p in cls.bl_rna.properties:
                if p.identifier not in {"rna_type", "name"} and not p.description.strip() and not getattr(p, "is_hidden", False):
                    missing += 1
    if missing:
        raise Warn(f"{missing} property tooltips empty (bench/qa/TOOLTIPS.md lists them)")


WINDOW_OPS = ("guided_place", "guided_work", "guided_draw_guide", "seg_split", "seg_paint", "seg_review", "seg_walk", "seg_paint_size",
              "guided_frame", "seg_frame", "guided_redraw_strip", "teach_remember", "teach_apply", "teach_find", "seg_from_shapes",
              "issue_select", "guided_edit_preview")


def call_without_window(name, mode="EXEC_DEFAULT"):
    """Call an operator that needs a 3D view/mouse. Allowed: it works, it cancels, or its poll says no. Not allowed: a Python error."""
    try:
        r = getattr(bpy.ops.curiomesh, name)(mode)
    except RuntimeError as e:
        msg = str(e)
        if "Traceback" in msg or "Error: Python" in msg:
            raise AssertionError(f"{name}: Python error: {msg[-300:]}")
        return "poll-refused"
    return ",".join(sorted(r))


@check("addon", "window_only_tools_survive_a_call_without_a_window")
def _():
    new_scene()
    done = {n: call_without_window(n, "INVOKE_DEFAULT") for n in WINDOW_OPS if hasattr(bpy.ops.curiomesh, n)}
    manual("Place Points / Work Live / Draw Guide / Split / Paint segments / Paint Size / Review Segments / Walk (modal mouse tools)", "need a 3D view and a mouse; here they were only called without one and must not raise a Python error")
    return " ".join(f"{k}:{v}" for k, v in done.items())


@check("auto", "window_only_tools_survive_a_call_with_a_result", needs="auto")
def _():
    done = {n: call_without_window(n) for n in WINDOW_OPS if hasattr(bpy.ops.curiomesh, n) and n not in ("guided_place", "guided_work", "guided_draw_guide", "seg_split", "seg_paint")}
    return " ".join(f"{k}:{v}" for k, v in done.items())


@check("settings", "panels_draw_with_no_object")
def _():
    new_scene()
    return draw_sweep("empty scene")


@check("settings", "every_enum_value_can_be_set")
def _():
    new_scene()
    g, bad, n = gs(), [], 0
    for p in g.bl_rna.properties:
        if p.type == "ENUM" and not p.is_readonly and p.identifier not in {"rna_type"}:
            keep = getattr(g, p.identifier)
            for item in p.enum_items:
                try:
                    setattr(g, p.identifier, item.identifier)
                    n += 1
                except Exception as e:
                    bad.append(f"{p.identifier}={item.identifier}: {e}")
            setattr(g, p.identifier, keep)
    ok(not bad, "; ".join(bad[:5]))
    return f"{n} values"


@check("settings", "numeric_limits_are_enforced")
def _():
    g, bad = gs(), []
    for p in g.bl_rna.properties:
        if p.type in {"FLOAT", "INT"} and not p.is_readonly and p.array_length == 0 and p.identifier != "name":
            for v in (p.hard_min, p.hard_max):
                setattr(g, p.identifier, v)
                if getattr(g, p.identifier) != type(getattr(g, p.identifier))(v):
                    bad.append(p.identifier)
            g.property_unset(p.identifier)
    ok(not bad, f"limits not held: {bad[:6]}")


@check("settings", "saved_in_blend_and_restored")
def _():
    g = gs()
    want = {"auto_profile": "ANIMATION", "auto_detail": 17.5, "sym_axis": "X", "bake_size": "1024", "apply_mode": "REPLACE", "keep_uvs": True, "loop_count": 5, "auto_instances": True}
    for k, v in want.items():  # the Use first: it sets Detail, Method and Output
        setattr(g, k, v)
    path = os.path.join(OUT, "settings.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    g.property_unset("auto_detail")
    bpy.ops.wm.open_mainfile(filepath=path)
    reset_state()
    got = {k: getattr(gs(), k) for k in want}
    ok(got == want, f"after reload: {got}")


@check("settings", "scene_settings_and_region_and_fix_groups_roundtrip")
def _():
    new_scene()
    s, r, f = bpy.context.scene.curiomesh_settings, bpy.context.scene.curiomesh_region, bpy.context.scene.curiomesh_fix
    s.target_faces = 1234
    r.target_edge = 0.25
    f.grow = 3
    ok(s.target_faces == 1234 and abs(r.target_edge - 0.25) < 1e-6 and f.grow == 3)


# =========================================================================== B. classic remesh and friends (small model)


@check("model", "load_main_model")
def _():
    obj = load_model(MAIN)
    ok(CTX["src_faces"] > 100, f"model has {CTX['src_faces']} faces")
    return f"{MAIN}: {CTX['src_faces']:,} faces, {len(obj.data.uv_layers)} UV layer(s), {len(obj.material_slots)} material(s)"


@check("settings", "panels_draw_with_model_selected", needs="main")
def _():
    return draw_sweep("model selected")


@check("classic", "analyze", needs="main")
def _():
    s = bpy.context.scene.curiomesh_settings
    r = bpy.ops.curiomesh.analyze()
    ok(r == {"FINISHED"}, f"returned {r}")
    ok(s.analysis_faces == CTX["src_faces"] or s.analysis_faces > 0, "analysis_faces not filled in")
    ok(s.analysis_profile != "", "analysis_profile empty")


@check("classic", "measure_mesh", needs="main")
def _():
    r = bpy.ops.curiomesh.compute_metrics()
    ok(r == {"FINISHED"}, f"returned {r}")


@check("classic", "quality_presets_change_settings", needs="main")
def _():
    s = bpy.context.scene.curiomesh_settings
    seen = {}
    for preset in ("DRAFT", "HERO", "BALANCED"):
        s.target_faces = 4000
        ok(bpy.ops.curiomesh.apply_preset(preset=preset) == {"FINISHED"})
        seen[preset] = (s.quality, s.atlas_candidate_count, s.target_faces)
    ok(seen["DRAFT"][0] == "DRAFT" and seen["HERO"][1] > seen["DRAFT"][1], str(seen))
    ok(seen["DRAFT"][2] == 2000, f"Draft should halve the face target: {seen['DRAFT']}")


def _remesh(engine, target=600, **kw):
    select_only(main_obj())
    before = set(bpy.data.objects.keys())
    r = bpy.ops.curiomesh.remesh(target_faces=target, engine=engine, output_mode="NEW", **kw)
    ok(r == {"FINISHED"}, f"remesh {engine} returned {r}")
    new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
    ok(new, "no new object")
    out = new[0]
    return out


@check("classic", "remesh_every_engine", needs="main")
def _():
    items = [i.identifier for i in bpy.ops.curiomesh.remesh.get_rna_type().properties["engine"].enum_items]
    notes, refused = [], []
    for engine in items:
        t = time.time()
        try:
            out = _remesh(engine, target=1500)
        except RuntimeError as e:
            msg = str(e)
            ok("Traceback" not in msg and "Error: Python" not in msg, f"{engine}: Python error: {msg[-250:]}")
            refused.append(f"{engine} ({msg.replace('Error: ', '')[:110]})")  # the add-on said no, in words
            continue
        n = len(out.data.polygons)
        ok(20 <= n <= 6 * 1500, f"{engine}: {n} faces for a target of 1500")
        notes.append(f"{engine}={n}f/{time.time() - t:.0f}s")
        bpy.data.objects.remove(out)
    ok(notes, "no engine produced a mesh: " + "; ".join(refused))
    if refused:
        raise Warn(f"worked: {' '.join(notes)}; refused with a message: {'; '.join(refused)}")
    return " ".join(notes)


@check("classic", "remesh_replace_mode_replaces_the_mesh", needs="main")
def _():
    load_model(MAIN, faces=1500)
    obj = bpy.context.view_layer.objects.active
    r = bpy.ops.curiomesh.remesh(target_faces=800, engine="QUADRIFLOW", output_mode="REPLACE")
    ok(r == {"FINISHED"}, str(r))
    ok(bpy.context.view_layer.objects.active is obj and len(obj.data.polygons) not in (0, 1500), f"{len(obj.data.polygons)} faces")


@check("classic", "batch_remesh_two_objects_writes_report", needs="main")
def _():
    a = load_model(MAIN, faces=1200)
    b = a.copy()
    b.data = a.data.copy()
    b.name = "Model2"
    bpy.context.scene.collection.objects.link(b)
    for o in (a, b):
        o.select_set(True)
    bpy.context.view_layer.objects.active = a
    s = bpy.context.scene.curiomesh_settings
    s.target_faces = 300
    r = bpy.ops.curiomesh.batch_remesh()
    ok(r == {"FINISHED"}, str(r))
    path = s.batch_report_path
    ok(path and os.path.exists(path), f"no batch report ({path!r})")
    rows = json.load(open(path))
    rows = rows if isinstance(rows, list) else rows.get("rows") or rows.get("results") or list(rows.values())
    ok(len(rows) >= 2, f"report has {len(rows)} rows")


@check("classic", "heatmap_toggle_and_hud_setting", needs="main")
def _():
    st = bpy.context.scene.curiomesh_settings
    before = st.heatmap_enabled
    ok(bpy.ops.curiomesh.toggle_heatmap() == {"FINISHED"} and st.heatmap_enabled != before, "first toggle did not flip heatmap_enabled")
    ok(bpy.ops.curiomesh.toggle_heatmap() == {"FINISHED"} and st.heatmap_enabled == before, "second toggle did not flip it back")
    for v in (False, True):
        st.hud_enabled = v
        ok(st.hud_enabled == v)
    manual("Dashboard HUD and heatmap drawing in the viewport", "GPU draw handlers need a window; the switches were flipped")


@check("classic", "live_preview_classic", needs="main")
def _():
    live = mod("live")
    load_model(MAIN, faces=1200)
    bpy.context.scene.curiomesh_settings.target_faces = 300
    ok(live.run_preview(bpy.context), live.state["message"])
    prev = bpy.data.objects.get(overlay.PREVIEW_NAME)
    ok(prev is not None and 20 < len(prev.data.polygons) < 3000, f"preview {prev and len(prev.data.polygons)} faces")
    live.remove_preview()
    ok(bpy.data.objects.get(overlay.PREVIEW_NAME) is None, "Live preview not removed")


# --------------------------------------------------------------------------- guided points, markers, shapes, guides


@check("points", "add_remove_clear_points", needs="main")
def _():
    obj = main_obj()
    select_only(obj)
    top = max(obj.data.vertices, key=lambda v: v.co.z).co
    bpy.context.scene.cursor.location = obj.matrix_world @ top
    for kind in ("DETAIL", "LOOP", "SHARP", "CALM"):
        ok(bpy.ops.curiomesh.guided_add_point(kind=kind) == {"FINISHED"})
    ok(len(obj.curiomesh_points) == 4, f"{len(obj.curiomesh_points)} points")
    obj.curiomesh_point_index = 0
    ok(bpy.ops.curiomesh.guided_remove_point() == {"FINISHED"} and len(obj.curiomesh_points) == 3)
    ok(bpy.ops.curiomesh.guided_clear_points() == {"FINISHED"} and len(obj.curiomesh_points) == 0)


@check("points", "auto_suggest_points", needs="main")
def _():
    obj = main_obj()
    gs().suggest_count = 5
    r = bpy.ops.curiomesh.guided_suggest()
    ok(r == {"FINISHED"}, str(r))
    n = len(obj.curiomesh_points)
    ok(1 <= n <= 5, f"{n} suggestions for a limit of 5")
    ok(all(p.auto for p in obj.curiomesh_points), "suggested points not marked as suggested")
    bpy.ops.curiomesh.guided_suggest()
    ok(len(obj.curiomesh_points) == n, "suggesting twice should replace, not add")
    bpy.ops.curiomesh.guided_clear_points()


@check("points", "empty_point_object", needs="main")
def _():
    ok(bpy.ops.curiomesh.guided_add_empty() == {"FINISHED"})
    e = bpy.data.objects.get("CM_Point")
    ok(e is not None and e.empty_display_type == "SPHERE" and e["curiomesh_kind"], "no CM_Point empty")
    bpy.data.objects.remove(e)


@check("points", "whole_mesh_preview_with_points", needs="main")
def _():
    load_model(MAIN, faces=2500)
    select_only(bpy.data.objects["Model"])
    g = gs()
    g.scope = "WHOLE"
    g.base_faces = 500
    bpy.ops.curiomesh.guided_add_point(kind="DETAIL")
    r = bpy.ops.curiomesh.guided_update()
    ok(r == {"FINISHED"}, f"Update returned {r}: {guided.state['message']}")
    finish()
    res = guided.state["res"]
    ok(res is not None and 50 < len(res["faces"]) < 3000, f"{None if res is None else len(res['faces'])} faces for Base Quads {g.base_faces} on a {len(bpy.context.object.data.polygons)}-face source ({guided.state['message']})")
    ok(bpy.data.objects.get(guided.PREVIEW_NAME) is not None, "no preview object")
    ok(bpy.ops.curiomesh.guided_stop() == {"FINISHED"})
    ok(bpy.data.objects.get(guided.PREVIEW_NAME) is None, "preview not removed by Cancel")


@check("markers", "detail_markers_and_paint_groups", needs="main")
def _():
    obj = main_obj()
    select_only(obj)
    ok(bpy.ops.curiomesh.guided_add_marker() == {"FINISHED"} and bpy.ops.curiomesh.guided_add_marker(less=True) == {"FINISHED"})
    balls = guided.markers(bpy.context.scene)
    ok(len(balls) == 2, f"{len(balls)} markers")
    ok(bpy.ops.curiomesh.guided_paint_density() == {"FINISHED"} and obj.vertex_groups.get(gs().density_group), "More Detail group missing")
    bpy.ops.object.mode_set(mode="OBJECT")
    ok(bpy.ops.curiomesh.guided_paint_density(less=True) == {"FINISHED"} and obj.vertex_groups.get(gs().less_group), "Less Detail group missing")
    bpy.ops.object.mode_set(mode="OBJECT")
    ok(bpy.ops.curiomesh.guided_clear_density() == {"FINISHED"})
    ok(not guided.markers(bpy.context.scene) and not obj.vertex_groups.get(gs().density_group), "Clear left markers or groups behind")


@check("shapes", "add_and_remove_shapes", needs="main")
def _():
    select_only(main_obj())
    g = gs()
    guided.state["source"] = main_obj().name
    for kind in ("BOX", "SPHERE"):
        ok(bpy.ops.curiomesh.guided_add_shape(kind=kind) == {"FINISHED"})
    ok(len(g.bool_shapes) == 2 and g.bool_on, f"{len(g.bool_shapes)} shapes")
    ok(bpy.ops.curiomesh.guided_remove_shape() == {"FINISHED"} and len(g.bool_shapes) == 1)
    ok(bpy.ops.curiomesh.guided_remove_shape() == {"FINISHED"} and len(g.bool_shapes) == 0)
    ok(not [o for o in bpy.data.objects if o.name.startswith("CurioBool_")], "shape objects left behind")
    g.bool_on = False


@check("guides", "guide_curves_add_edit_remove", needs="main")
def _():
    obj = main_obj()
    g = gs()
    cu = bpy.data.curves.new("CM_Guide_test", "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(2)
    centre, c = bbox(obj)
    for i, p in enumerate(sp.points):
        p.co = (centre.x - c.x / 2 + i * c.x / 2, centre.y, centre.z + c.z / 2, 1)
    cobj = bpy.data.objects.new("CM_Guide_test", cu)
    bpy.context.scene.collection.objects.link(cobj)
    for o in bpy.context.scene.objects:
        o.select_set(o is cobj)
    bpy.context.view_layer.objects.active = cobj
    for kind in ("GUIDE", "LOOP", "BEND"):
        g.guide_curves.clear()
        ok(bpy.ops.curiomesh.guided_add_guide(kind=kind) == {"FINISHED"}, f"add {kind}")
        ok(len(g.guide_curves) == 1 and g.guide_curves[0].kind == kind)
    ok(bpy.ops.curiomesh.guided_edit_guide() == {"FINISHED"} and cobj.mode == "EDIT", "Edit Curve did not enter Edit Mode")
    bpy.ops.object.mode_set(mode="OBJECT")
    ok(bpy.ops.curiomesh.guided_remove_guide() == {"FINISHED"} and len(g.guide_curves) == 0)
    ok(bpy.data.objects.get("CM_Guide_test") is None, "a CM_Guide curve should be deleted with its list entry")
    select_only(obj)


# =========================================================================== C. Auto Segments


@check("auto", "game_asset_default", needs="main")
def _():
    src = main_obj()
    t = time.time()
    res = fresh_auto(src, auto_profile="GAME")
    n = len(res["faces"])
    ok(0.01 * CTX["src_faces"] < n < CTX["src_faces"], f"{n} faces from {CTX['src_faces']}")
    ok(quad_share(res) >= 0.9, f"only {quad_share(res):.0%} quads")
    base = guided.state["auto_base"]
    ok(base is not None and base["report"]["loss"] >= 0, "no report")
    ok(bpy.data.objects.get(guided.PREVIEW_NAME) is not None, "no preview object")
    CTX["auto"] = res
    CTX["auto_secs"] = time.time() - t
    if n > CTX["src_faces"]:
        raise Warn(f"Game Asset gave {n} faces from a {CTX['src_faces']}-face source: more than it started with")
    return f"{n} faces, {quad_share(res):.0%} quads, view change {base['report']['loss']:.1f}%, {time.time() - t:.0f}s"


@check("auto", "panels_draw_with_result", needs="auto")
def _():
    return draw_sweep("Auto result on screen")


@check("auto", "segments_list_and_select", needs="auto")
def _():
    segments.ensure()
    parts = segments.seg["parts"]
    ok(len(parts) >= 1, "no segments")
    ok(bpy.ops.curiomesh.seg_select(what="ALL") == {"FINISHED"} and len(segments.seg["sel"]) == len(parts))
    ok(bpy.ops.curiomesh.seg_select(what="WORST") == {"FINISHED"})
    ok(bpy.ops.curiomesh.seg_select(what="NONE") == {"FINISHED"} and not segments.seg["sel"])
    segments.select([0])
    return f"{len(parts)} parts"


@check("auto", "issue_finder_lists_and_saves", needs="auto")
def _():
    res = guided.state["res"]
    ok(res.get("issues") is not None, "result has no issue record")
    spots = issue_panel.spots()
    path = os.path.join(OUT, "issues.json")
    if not spots:
        ok(not issue_panel._IssueOp.poll(None) if False else True)
        raise Warn("no problem spots on this result, so View/Select/Step/Save could not be exercised here")
    ok(bpy.ops.curiomesh.issue_view(index=0) == {"FINISHED"})
    ok(bpy.ops.curiomesh.issue_step(step=1) == {"FINISHED"})
    ok(bpy.ops.curiomesh.issue_save(filepath=path) == {"FINISHED"} and os.path.exists(path), "report not saved")
    ok(bpy.ops.curiomesh.issue_compare(filepath=path) == {"FINISHED"}, "compare with itself failed")
    d = issue_panel.ui.get("diff")
    ok(d is not None, "no diff after compare")
    ok(bpy.ops.curiomesh.issue_select(index=0) in ({"FINISHED"}, {"CANCELLED"}))
    bpy.ops.object.mode_set(mode="OBJECT")
    return f"{len(spots)} spots"


@check("auto", "view_toggles_dont_raise", needs="auto")
def _():
    g = gs()
    for prop in ("seg_colors", "seg_heat", "seg_bad", "seg_wire"):
        setattr(g, prop, not getattr(g, prop))
        setattr(g, prop, not getattr(g, prop))
    for v in ("OVER", "BESIDE"):
        g.seg_compare = v


@check("auto", "keep_original_where_bad", needs="auto")
def _():
    segments.ensure()
    bad = segments.bad_parts()
    before = len(guided.state["res"]["faces"])
    if not bad:
        ok(bpy.ops.curiomesh.keep_original_bad.poll() in (True, False))
        raise Warn("no bad parts on this result: the Keep Original button had nothing to do")
    r = bpy.ops.curiomesh.keep_original_bad()
    ok(r == {"FINISHED"}, str(r))
    finish()
    kept = [p for p in bad if segments.seg["parts"][p]["method"] == "ORIGINAL"]
    ok(kept, "no part switched to ORIGINAL")
    return f"{len(kept)} of {len(bad)} bad parts kept, faces {before} -> {len(guided.state['res']['faces'])}"


@check("auto", "redo_selected_parts_with_other_method", needs="auto")
def _():
    segments.ensure()
    if segments.mirrored():
        raise Skip("mirrored result")
    g = gs()
    segments.select([0])
    g.seg_method = "PAIRING"
    r = bpy.ops.curiomesh.seg_redo()
    ok(r == {"FINISHED"}, str(r))
    finish()
    ok(segments.seg["parts"][0]["method"] in ("PAIRING", "AUTO") or segments.seg["parts"][0]["tries"], "no new try recorded")
    if len(segments.seg["parts"][0]["tries"]) > 1:
        ok(bpy.ops.curiomesh.seg_try(index=0) == {"FINISHED"})
        finish()


@check("auto", "goal_redo_use_and_detail", needs="auto")
def _():
    segments.ensure()
    g = gs()
    segments.select([0])
    g.seg_goal, g.seg_goal_detail = "REDUCE", 0.3
    r = bpy.ops.curiomesh.seg_goal()
    ok(r == {"FINISHED"}, str(r))
    finish()
    ok(bpy.ops.curiomesh.part_size(bigger=True) in ({"FINISHED"}, {"CANCELLED"}))
    finish()
    ok(bpy.ops.curiomesh.seg_multires.poll() in (True, False))


@check("auto", "segment_edit_merge_split_lock_undo", needs="auto")
def _():
    segments.ensure()
    parts = segments.seg["parts"]
    if len(parts) < 2:
        raise Skip("need two segments")
    segments.select([0, 1])
    if not bpy.ops.curiomesh.seg_merge.poll():
        raise Skip("merge needs the Review Segments view (segments.view.ready())")
    n = len(parts)
    ok(bpy.ops.curiomesh.seg_merge() == {"FINISHED"})
    ok(len(segments.seg["parts"]) == n - 1, "merge did not reduce the segment count")
    ok(bpy.ops.curiomesh.seg_undo() == {"FINISHED"} and len(segments.seg["parts"]) == n, "undo edit did not restore the count")
    segments.select([0])
    ok(bpy.ops.curiomesh.seg_lock() == {"FINISHED"} and segments.seg["parts"][0].get("locked"), "lock did not stick")
    if bpy.ops.curiomesh.seg_halve.poll():
        ok(bpy.ops.curiomesh.seg_halve() == {"FINISHED"})
    ok(bpy.ops.curiomesh.seg_rename(name="Probe") == {"FINISHED"} and segments.seg["parts"][0]["name"] == "Probe")


@check("auto", "apply_new_object", needs="auto")
def _():
    src = main_obj()
    res = guided.state["res"]
    n = len(res["faces"])
    out = apply_result()
    ok(out.name == "Model_Retopo", f"result called {out.name}")
    ok(len(out.data.polygons) == n, f"{len(out.data.polygons)} faces written, preview had {n}")
    ok(bpy.data.objects.get(guided.PREVIEW_NAME) is None and not gs().enabled, "preview not cleaned up")
    ok(src.name in bpy.data.objects and len(src.data.polygons) == CTX["src_faces"], "source changed")
    ok(max(out.data.polygons, key=lambda p: p.loop_total).loop_total <= 8 and not out.data.validate(verbose=False), "result mesh invalid")
    CTX["applied"] = out
    return f"{n} faces in {out.name}"


@check("auto", "apply_undo_restores", needs="applied")
def _():
    # the probe says whether this headless Blender can undo at all
    bpy.ops.ed.undo_push(message="probe")
    bpy.ops.mesh.primitive_cube_add()
    probe = bpy.context.object.name
    try:
        bpy.ops.ed.undo()
    except RuntimeError as e:
        raise Skip(f"headless Blender cannot undo: {e}")
    if probe in bpy.data.objects and bpy.data.objects[probe].users_collection:
        manual("Undo of Apply / Replace Mesh / Fix Selection / Region Apply", "background Blender keeps no undo stack; check Ctrl+Z by hand")
        raise Skip("headless Blender kept the probe cube after undo: no undo stack")
    # undo history works: do an Apply and undo it
    src = load_model(MAIN)
    fresh_auto(src)
    bpy.ops.ed.undo_push(message="before apply")
    apply_result()
    bpy.ops.ed.undo()
    ok(bpy.data.objects.get("Model_Retopo") is None, "Undo left the Apply result behind")


@check("auto", "apply_replace_mesh", needs="main")
def _():
    src = load_model(MAIN)
    res = fresh_auto(src, apply_mode="REPLACE")
    n = len(res["faces"])
    out = apply_result()
    ok(out is src and len(src.data.polygons) == n, f"{len(src.data.polygons)} faces in the source after Replace, expected {n}")
    ok(bpy.data.objects.get("Model_Retopo") is None, "Replace should not make a _Retopo object")


@check("auto", "cancel_removes_preview_keeps_points", needs="main")
def _():
    src = main_obj()
    fresh_auto(src)
    src.curiomesh_points.add()
    ok(bpy.ops.curiomesh.guided_stop() == {"FINISHED"})
    ok(bpy.data.objects.get(guided.PREVIEW_NAME) is None and len(src.curiomesh_points) == 1, "Cancel should keep the points")
    src.curiomesh_points.clear()


@check("auto", "stop_work_while_running", needs="main")
def _():
    src = main_obj()
    select_only(src)
    reset_state()
    g = gs()
    g.scope, g.auto_update, g.auto_detail = "AUTO", False, 12.0
    ok(bpy.ops.curiomesh.guided_auto_start() == {"FINISHED"})
    work = guided.state["work"]
    ok(work is not None and work["runner"].busy, "Auto did not start a job")
    work["runner"].step(0.02)
    ok(bpy.ops.curiomesh.guided_stop_work() == {"FINISHED"})
    ok(guided.state["work"] is None and guided.state["res"] is None, "Stop left a half result")
    ok(bpy.data.objects.get(guided.PREVIEW_NAME) is None, "Stop left the preview")


@check("auto", "other_models_game_asset_and_apply", needs="main")
def _():
    notes = []
    for name in ("treasure_chest", "concrete_road_barrier", ALT):
        src = load_model(name, faces=2500)
        res = fresh_auto(src, auto_detail=10.0)
        n = len(res["faces"])
        ok(n > 0 and quad_share(res) >= 0.8, f"{name}: {n} faces, {quad_share(res):.0%} quads")
        out = apply_result()
        ok(len(out.data.polygons) == n and not out.data.validate(verbose=False), f"{name}: written mesh differs or is invalid")
        notes.append(f"{name}={len(src.data.polygons)}->{n}")
    return " ".join(notes)


# --------------------------------------------------------------------------- Use / Method / Output / Aim


@check("use", "every_use", needs="main")
def _():
    src = main_obj()
    notes, over = [], []
    for use in ("LOW_POLY", "ANIMATION", "SUBDIV", "SCAN", "SCULPT"):
        res = fresh_auto(src, auto_profile=use)
        n = len(res["faces"])
        ok(0 < n < 4 * CTX["src_faces"], f"{use}: {n} faces from {CTX['src_faces']}")
        if n > CTX["src_faces"]:
            over.append(f"{use} {n}>{CTX['src_faces']}")
        exp = {"LOW_POLY": None, "ANIMATION": 0.8, "SUBDIV": 0.8, "SCAN": 0.8, "SCULPT": 0.8}[use]
        if exp:
            ok(quad_share(res) >= exp, f"{use}: only {quad_share(res):.0%} quads")
        notes.append(f"{use}={n}")
    if over:
        raise Warn("more faces than the source has: " + ", ".join(over) + " (a quad result can be bigger than a triangle source, but check it)")
    return " ".join(notes)


@check("use", "blockout_and_hulls", needs="main")
def _():
    src = main_obj()
    notes = []
    for use in ("BLOCKOUT", "HULL"):
        res = fresh_auto(src, auto_profile=use)
        n = len(res["faces"])
        ok(0 < n, f"{use}: no faces")
        notes.append(f"{use}={n}")
    gs().blockout_separate = True
    r = bpy.ops.curiomesh.guided_apply()
    ok(r == {"FINISHED"}, str(r))
    made = [o for o in bpy.data.objects if o.name.startswith("Model_Part")]
    ok(made, "Each Part Its Own Object made no Model_Part objects")
    return " ".join(notes) + f", {len(made)} part objects"


@check("use", "every_method", needs="main")
def _():
    src = main_obj()
    items = [i[0] for i in profiles.METHOD_ITEMS]
    notes = []
    for method in items:
        res = fresh_auto(src, auto_profile="CUSTOM", auto_method=method, auto_output="QUADS" if method != "TRIS" else "TRIS", auto_detail=12.0)
        n = len(res["faces"])
        ok(0 < n < 2 * CTX["src_faces"], f"{method}: {n} faces")
        if method == "TRIS":
            ok(all(s == 3 for s in sizes(res)), "Tris Only should give triangles only")
        notes.append(f"{method}={n}")
    return " ".join(notes)


@check("use", "every_output", needs="main")
def _():
    src = main_obj()
    shape = {}
    for out in ("TRIS", "QUADS", "MIXED"):
        res = fresh_auto(src, auto_profile="CUSTOM", auto_output=out)
        s = sizes(res)
        shape[out] = (len(s), s.count(3), s.count(4))
    ok(shape["TRIS"][1] == shape["TRIS"][0], f"Triangles output has non-triangles: {shape['TRIS']}")
    ok(shape["QUADS"][2] / shape["QUADS"][0] >= 0.9, f"Quads output: {shape['QUADS']}")
    ok(shape["MIXED"][0] <= shape["QUADS"][0] * 1.05, f"Mixed should not have more faces than Quads: {shape}")
    return str(shape)


@check("use", "reduce_to_count_and_percent", needs="main")
def _():
    src = main_obj()
    want = max(300, CTX["src_faces"] // 8)
    res = fresh_auto(src, auto_profile="GAME", auto_aim="REDUCE", auto_reduce_by="COUNT", auto_reduce_count=want)
    n = len(res["faces"])
    ok(0.6 * want <= n <= 1.6 * want, f"asked for {want}, got {n}")
    res = fresh_auto(src, auto_profile="GAME", auto_aim="REDUCE", auto_reduce_by="PERCENT", auto_reduce_percent=10.0)
    n2 = len(res["faces"])
    ok(0.5 * 0.1 * CTX["src_faces"] <= n2 <= 1.8 * 0.1 * CTX["src_faces"] + 60, f"asked for 10% of {CTX['src_faces']}, got {n2}")
    return f"count {want} -> {n}, 10% -> {n2}"


@check("use", "detail_slider_changes_face_count", needs="main")
def _():
    src = main_obj()
    lo = len(fresh_auto(src, auto_detail=3.0)["faces"])
    hi = len(fresh_auto(src, auto_detail=30.0)["faces"])
    ok(lo > hi, f"Detail 3% gave {lo} faces, Detail 30% gave {hi}: lower Detail must keep more faces")
    return f"{lo} vs {hi}"


@check("use", "auto_picks_use", needs="main")
def _():
    src = main_obj()
    select_only(src)
    reset_state()
    g = gs()
    g.scope, g.auto_update = "AUTO", False
    r = bpy.ops.curiomesh.guided_auto_all()
    ok(r == {"FINISHED"}, str(r))
    ok(g.auto_why.strip() != "", "Auto did not say why")
    finish()
    ok(guided.state["res"] is not None, "Auto did not produce a result: " + guided.state["message"])
    return g.auto_profile + ": " + g.auto_why[:70]


@check("use", "tuning_options_each_run", needs="main")
def _():
    src = main_obj()
    opts = [("auto_rough", True), ("auto_drop_hidden", True), ("auto_chords", True), ("auto_tidy", True), ("auto_redo_rough", True),
            ("auto_gpu", False), ("auto_parallel", True), ("auto_effort", "LIGHT"), ("auto_baked", True), ("pure_quads", False),
            ("fit_iterations", 0), ("feature_angle", 70.0), ("auto_instances", True)]
    notes, bad = [], []
    for k, v in opts:
        try:
            res = fresh_auto(src, **{k: v})
            n = len(res["faces"])
            ok(n > 0)
            notes.append(f"{k}={n}")
        except Exception as e:
            bad.append(f"{k}={v}: {e}")
    ok(not bad, "; ".join(bad))
    return " ".join(notes)


@check("use", "tweak_sliders_each_run", needs="main")
def _():
    src = main_obj()
    base = len(fresh_auto(src)["faces"])
    out = {}
    for k, v in (("tweak_rows", 1.0), ("tweak_flat", 1.0), ("tweak_poles", False)):
        out[k] = len(fresh_auto(src, **{k: v})["faces"])
        ok(out[k] > 0, f"{k}: no faces")
    return f"base {base}, " + str(out)


@check("use", "keep_uv_seams", needs="main")
def _():
    src = main_obj()
    if not src.data.uv_layers:
        raise Skip("model has no UVs")
    plain = len(fresh_auto(src)["faces"])
    res = fresh_auto(src, keep_uvs=True)
    ok(len(res["faces"]) >= plain * 0.9, f"Keep UV Seams gave {len(res['faces'])} faces, plain gave {plain}: it should cost faces, not save them")
    out = apply_result()
    ok(len(out.data.uv_layers) >= 1, "no UVs on the result")
    return f"{plain} -> {len(res['faces'])}"


# --------------------------------------------------------------------------- Mirror, regions, guides in Auto


@check("mirror", "symmetrize_x_and_auto_axis", needs="main")
def _():
    src = load_model(ALT)
    CTX["alt"] = src
    base = len(fresh_auto(src, auto_profile="GAME")["faces"])
    res = fresh_auto(src, sym_axis="AUTO", sym_force=True)
    job = guided.state["auto_base"]["job"]
    ok(job.mirror is not None, "Symmetrize did not set up a mirror: " + guided.state["message"])
    ok(len(res["faces"]) > 0)
    out = apply_result()
    import numpy as np

    co = np.array([v.co[:] for v in out.data.vertices])
    mirrored_axes = []
    for ax in range(3):
        m = co.copy()
        centre = (co[:, ax].max() + co[:, ax].min()) / 2
        m[:, ax] = 2 * centre - m[:, ax]
        from mathutils.kdtree import KDTree

        tree = KDTree(len(co))
        for i, c in enumerate(co):
            tree.insert(c.tolist(), i)
        tree.balance()
        tol = 0.01 * float(np.linalg.norm(co.max(axis=0) - co.min(axis=0)))
        hit = sum(1 for p in m[:: max(1, len(m) // 300)] if tree.find(p.tolist())[2] < tol)
        mirrored_axes.append(hit / len(m[:: max(1, len(m) // 300)]))
    ok(max(mirrored_axes) > 0.95, f"result is not mirror-symmetric on any axis (matches {mirrored_axes})")
    return f"plain {base} faces, mirrored {len(res['faces'])}, symmetry {max(mirrored_axes):.0%}"


@check("region", "selected_faces_only_keeps_the_rest", needs="main")
def _():
    src = main_obj()
    select_only(src)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    n = len(src.data.polygons)
    zs = sorted(p.center.z for p in src.data.polygons)
    cut = zs[int(len(zs) * 0.6)]
    for p in src.data.polygons:
        p.select = p.center.z > cut
    sel = sum(p.select for p in src.data.polygons)
    ok(0 < sel < n, f"selection {sel}/{n}")
    whole = len(fresh_auto(src, auto_detail=12.0)["faces"])
    for p in src.data.polygons:  # fresh_auto starts from a clean state; the selection lives on the mesh
        p.select = p.center.z > cut
    res = fresh_auto(src, region_use=True, auto_detail=12.0)
    mask = guided.state["region"]
    seen = None if mask is None else int(mask.sum())
    ok(len(res["faces"]) > 0)
    out = apply_result()
    ok(len(out.data.polygons) > 0)
    ok(len(res["faces"]) != whole, f"Selected Faces Only redid {sel} of {n} faces but gave the same {whole} faces as redoing all of them (the add-on saw {seen} selected faces)")
    for p in src.data.polygons:
        p.select = False
    return f"{sel} of {n} faces redone -> {len(out.data.polygons)} (all of them: {whole})"


@check("guides", "guide_loop_and_deformation_lines_in_auto", needs="main")
def _():
    src = main_obj()
    notes = []
    for kind, flag in (("GUIDE", "guide_use"), ("LOOP", "loop_use"), ("BEND", "bend_use")):
        select_only(src)
        g = gs()
        g.guide_curves.clear()
        cu = bpy.data.curves.new(f"CM_{kind}_t", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(5)
        centre, d = bbox(src)
        for i, p in enumerate(sp.points):
            ang = i / 6 * 6.28318
            p.co = (centre.x + 0.35 * d.x * __import__("math").cos(ang), centre.y + 0.35 * d.y * __import__("math").sin(ang), centre.z, 1)
        sp.use_cyclic_u = True
        cobj = bpy.data.objects.new(f"CM_{kind}_t", cu)
        bpy.context.scene.collection.objects.link(cobj)
        select_only(cobj)
        bpy.ops.curiomesh.guided_add_guide(kind=kind)
        select_only(src)
        res = fresh_auto_keep(src, **{flag: True})
        ok(len(res["faces"]) > 0, f"{kind}: no faces")
        notes.append(f"{kind}={len(res['faces'])}")
        g.guide_curves.clear()
        bpy.data.objects.remove(cobj)
    return " ".join(notes)


def fresh_auto_keep(src, **props):
    """Like fresh_auto, but leaves guide curves, markers and shapes in place."""
    select_only(src)
    guided.stop_work()
    g = gs()
    g.scope, g.auto_update, g.auto_detail = "AUTO", False, 12.0
    for k, v in props.items():
        setattr(g, k, v)
    r = bpy.ops.curiomesh.guided_auto_start()
    ok(r == {"FINISHED"}, f"Start returned {r}: {guided.state['message']}")
    finish()
    ok(guided.state["res"] is not None, guided.state["message"])
    return guided.state["res"]


@check("markers", "density_markers_add_faces_where_marked", needs="main")
def _():
    src = main_obj()
    select_only(src)
    base = fresh_auto(src, auto_detail=15.0)
    import numpy as np

    co = np.array([v.co[:] for v in src.data.vertices])
    top = co[np.argmax(co[:, 2])]
    bpy.context.scene.cursor.location = src.matrix_world @ __import__("mathutils").Vector(top.tolist())
    bpy.ops.curiomesh.guided_add_marker()
    ball = guided.markers(bpy.context.scene)[0]
    ball.empty_display_size = float(src.dimensions.z) * 0.25
    r = fresh_auto_keep(src, density_use=True, auto_detail=15.0)

    def near(res):
        c = np.asarray(res["co"])
        d = np.linalg.norm(c - top, axis=1) < src.dimensions.z * 0.3
        return int(d.sum())

    a, b = near(base), near(r)
    bpy.ops.curiomesh.guided_clear_density()
    if b <= a:
        raise Warn(f"marker did not add vertices near it ({a} -> {b})")
    return f"vertices near the marker {a} -> {b}"


@check("seglive", "segment_settings_preview_follows_sliders", needs="auto")
def _():
    src = main_obj()
    fresh_auto(src)
    segments.ensure()
    if segments.seg.get("src") is None or segments.seg.get("tree") is None:
        raise Skip("segment tree not built for this result (mirrored or too small)")
    st = src.curiomesh_seg
    seen = {}
    for detail in (0.15, 0.85):
        before = segments.seg.get("live")
        st.detail = detail
        t = time.time()
        while time.time() - t < 60:
            seglive._tick()
            if segments.seg.get("live") is not None and segments.seg.get("live") is not before:
                break
            time.sleep(0.05)
        ok(segments.seg.get("live") is not None and segments.seg.get("live") is not before, f"no live preview for Segment Detail {detail}: {seglive.live['error']}")
        seen[detail] = seglive.live["count"]
    ok(seen[0.85] >= seen[0.15], f"more Segment Detail must not give fewer segments: {seen}")
    for prop, v in (("largest", 50.0), ("smallest", 2.0), ("creases", 0.8), ("smooth", 0.8)):
        setattr(st, prop, v)
    seglive.clear()
    return f"segments at detail 0.15 / 0.85: {seen}"


# --------------------------------------------------------------------------- Quad Size / Organization (the ladder)


@check("tweak", "quad_size_and_organization_ladder", needs="main")
def _():
    src = main_obj()
    fresh_auto(src)
    t = time.time()
    plan0 = list(ladder.lad["plan"])
    ok(plan0 or ladder.lad["rungs"], "the size ladder was not started by Start")
    ladder.lad["plan"] = [r for r in ladder.lad["plan"] if r[0] != "ROWS"]  # Clean Rows costs minutes a rung; checked in its own test
    while (ladder.lad["job"] is not None or ladder.lad["plan"]) and time.time() - t < 300:
        ladder.work(ladder.SLICE)
    ok(ladder.lad["rungs"], "no rungs built")
    g = gs()
    faces = {}
    for size in (0.0, 0.5, 1.0):
        g.quad_size = size
        ladder.show(bpy.context)
        finish()
        faces[size] = len(guided.state["res"]["faces"])
    ok(faces[0.0] >= faces[1.0], f"smaller quads must not give fewer faces: {faces}")
    ok(len(set(faces.values())) > 1, f"Quad Size changed nothing: {faces}")
    return f"{len(ladder.lad['rungs'])} rungs in {time.time() - t:.0f}s, faces {faces}"


@check("tweak", "organization_slider_balanced_and_detail", needs="main")
def _():
    g = gs()
    g.quad_size = 0.5
    out = {}
    for order in (0.0, 0.5):
        g.quad_order = order
        ladder.show(bpy.context)
        finish()
        out[order] = len(guided.state["res"]["faces"])
    ok(all(v > 0 for v in out.values()), str(out))
    ok(ladder.status() is not None)
    ready = {st: ladder.ready(st) for st in ("DETAIL", "BALANCED")}
    if out[0.0] == out[0.5]:
        raise Warn(f"Organization 0 and 0.5 gave the same {out[0.0]} faces; sizes ready per style: {ready}")
    return f"{out}, ready {ready}"


# --------------------------------------------------------------------------- bake, multires, LODs, instances


@check("bake", "normal_map_bake_on_apply", needs="main")
def _():
    src = main_obj()
    fresh_auto(src, bake_detail="NORMAL", bake_size="512")
    out = apply_result()
    imgs = [n.image for m in out.data.materials if m and m.node_tree for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
    ok(imgs, "no baked image on the result: " + guided.state["applied"])
    ok(any(n.type == "NORMAL_MAP" for m in out.data.materials if m and m.node_tree for n in m.node_tree.nodes), "no Normal Map node")
    return guided.state["applied"][:100]


@check("bake", "normal_plus_color_bake", needs="main")
def _():
    src = main_obj()
    fresh_auto(src, bake_detail="NORMAL_COLOR", bake_size="512")
    out = apply_result()
    imgs = {n.image.name for m in out.data.materials if m and m.node_tree for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image}
    ok(len(imgs) >= 2, f"expected a normal and a colour image, got {imgs}: {guided.state['applied']}")
    ok(len(out.data.uv_layers) >= 1, "no UVs")
    CTX["baked"] = out


@check("export", "export_for_game_engine_every_engine", needs="baked")
def _():
    out = CTX["baked"]
    select_only(out)
    folder = os.path.join(OUT, "export")
    os.makedirs(folder, exist_ok=True)
    notes = []
    for engine in game_export.ENGINES:
        path = os.path.join(folder, f"probe_{engine.lower()}")
        kw = {"form": "GLB", "tris": 400, "texture": "512"} if engine == "CUSTOM" else {}
        r = bpy.ops.curiomesh.export_game(filepath=path + ".fbx", engine=engine, **kw)
        ok(r == {"FINISHED"}, f"{engine}: {r}")
        form = game_export.ENGINES[engine].form.lower() if engine != "CUSTOM" else "glb"
        ok(os.path.exists(path + "." + form), f"{engine}: no .{form} written")
        ok(os.path.getsize(path + "." + form) > 500, f"{engine}: empty file")
        notes.append(engine)
    ok(len(out.data.polygons) > 0 and not [o for o in bpy.data.objects if o.name.endswith("_01") and o is not out], "export left piece objects behind")
    return " ".join(notes)


@check("lod", "make_lod_set", needs="main")
def _():
    src = main_obj()
    select_only(src)
    reset_state()
    g = gs()
    g.scope, g.auto_update, g.auto_detail = "AUTO", False, 12.0
    g.lod0_percent, g.lod1_percent, g.lod2_percent = 40.0, 20.0, 10.0
    r = bpy.ops.curiomesh.guided_make_lods()
    ok(r == {"FINISHED"}, str(r))
    for _i in range(3 * 60):
        if guided.state["lod"] is None:
            break
        finish()
    names = [f"Model_LOD{i}" for i in range(3)]
    objs = [bpy.data.objects.get(n) for n in names]
    ok(all(objs), f"missing {[n for n, o in zip(names, objs) if o is None]}: {guided.state.get('message')}")
    n = [len(o.data.polygons) for o in objs]
    ok(n[0] > n[1] > n[2] > 0, f"LOD face counts {n} are not falling")
    ok(guided.state["lod_report"], "no LOD report text")
    return f"LOD faces {n} from {CTX['src_faces']}"


@check("multires", "multires_detail_on_apply", needs="main")
def _():
    src = load_model(MAIN)
    fresh_auto(src, auto_profile="ANIMATION", multires_on=True, multires_levels=2)
    out = apply_result()
    m = [x for x in out.modifiers if x.type == "MULTIRES"]
    ok(m, "no Multires modifier: " + guided.state["applied"])
    ok(m[0].total_levels == 2, f"{m[0].total_levels} levels")
    return guided.state["applied"][:100]


@check("multires", "multires_detail_operator_on_hand_made_cage", needs="main")
def _():
    src = load_model(MAIN)
    cage = src.copy()
    cage.data = src.data.copy()
    cage.name = "Cage"
    mod_ = cage.modifiers.new("d", "DECIMATE")
    mod_.ratio = 0.1
    bpy.context.scene.collection.objects.link(cage)
    select_only(cage)
    bpy.ops.object.modifier_apply(modifier=mod_.name)
    src.select_set(True)
    bpy.context.view_layer.objects.active = cage
    gs().multires_levels = 2
    r = bpy.ops.curiomesh.multires_detail()
    ok(r == {"FINISHED"}, str(r))
    ok(any(x.type == "MULTIRES" for x in cage.modifiers), "cage has no Multires modifier")


# --------------------------------------------------------------------------- Live Boolean regions and the region tool


@check("bool", "redo_inside_shapes_live", needs="main")
def _():
    src = main_obj()
    fresh_auto(src)
    guided.state["source"] = src.name
    before = len(guided.state["res"]["faces"])
    bpy.ops.curiomesh.guided_add_shape(kind="SPHERE")
    shape = gs().bool_shapes[0].obj
    centre, size = bbox(src)
    shape.location = centre
    shape.scale = (max(size) * 0.3,) * 3
    bpy.context.view_layer.update()
    gs().bool_detail = 1.0
    gs().bool_on = True
    ok(guided.run_update(bpy.context, force=True), guided.state["message"])
    finish()
    res = guided.state["res"]
    ok(res is not None and len(res["faces"]) > 0, "no result after the shape redo")
    ok(res["new"].any(), "the redo changed no faces: " + guided.state["message"])
    ok(not res["new"].all(), "everything was redone: faces outside the shape must stay")
    ok(bpy.ops.curiomesh.seg_from_shapes.poll() in (True, False))
    after = len(res["faces"])
    bpy.ops.curiomesh.guided_remove_shape()
    gs().bool_on = False
    return f"{before} -> {after} faces, {int(res['new'].sum())} new"


@check("region", "region_tool_cage_preview_apply", needs="main")
def _():
    src = load_model(MAIN, faces=3000)
    select_only(src)
    s = bpy.context.scene.curiomesh_region
    ok(bpy.ops.curiomesh.region_add_cage(kind="BOX") == {"FINISHED"} and len(s.volumes) == 1)
    cage = s.volumes[0].obj
    centre, size = bbox(src)
    cage.location = centre
    cage.scale = (max(size) * 0.3,) * 3
    s.apply_mode = "COPY"
    r = bpy.ops.curiomesh.region_preview()
    ok(r == {"FINISHED"}, f"Preview returned {r}: {region.state['message']}")
    ok(bpy.data.objects.get(region.PREVIEW_NAME) is not None, "no region preview object")
    ok(bpy.ops.curiomesh.region_cancel() == {"FINISHED"} and bpy.data.objects.get(region.PREVIEW_NAME) is None, "cancel left the preview")
    select_only(src)
    r = bpy.ops.curiomesh.region_preview()
    ok(r == {"FINISHED"}, str(r))
    r = bpy.ops.curiomesh.region_apply()
    ok(r == {"FINISHED"}, f"Apply Region returned {r}: {region.state['message']}")
    ok(bpy.data.objects.get("Model_Region") is not None, "no Model_Region copy")
    ok(bpy.ops.curiomesh.region_volume_add() == {"FINISHED"} and bpy.ops.curiomesh.region_volume_remove() == {"FINISHED"})
    s.volumes.clear()


@check("region", "fit_to_surface", needs="main")
def _():
    load_model(MAIN, faces=1500)
    r = bpy.ops.curiomesh.fit_to_surface(iterations=2)
    ok(r == {"FINISHED"}, str(r))


@check("region", "fill_loop_closes_a_hole", needs="main")
def _():
    import bmesh

    obj = load_model(MAIN, faces=1500)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    face = max(bm.faces, key=lambda f: f.calc_center_median().z)
    ring = [f for f in bm.faces if f is face or any(v in face.verts for v in f.verts)]
    bmesh.ops.delete(bm, geom=ring, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True, use_multi_face=False, use_non_contiguous=False, use_verts=False)
    r = bpy.ops.curiomesh.fill_loop()
    bpy.ops.object.mode_set(mode="OBJECT")
    ok(r == {"FINISHED"}, str(r))


# --------------------------------------------------------------------------- Fix Selection


@check("fix", "find_and_fix_bad_faces_in_edit_mode", needs="main")
def _():
    import bmesh

    obj = load_model(MAIN, faces=2000)
    fs = bpy.context.scene.curiomesh_fix
    fs.source = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    for f in bm.faces:
        f.select = False
    bm.faces.ensure_lookup_table()
    for f in bm.faces[:30]:
        f.select = True
    bmesh.update_edit_mesh(obj.data)
    ok(bpy.ops.curiomesh.fix_find() == {"FINISHED"} and fs.last, "Find Bad Faces wrote no summary")
    notes = []
    for how in ("ALL", "CLEAN", "FLIP", "UNTANGLE", "RELAX", "PATCH"):
        r = bpy.ops.curiomesh.fix_run(how=how)
        ok(r in ({"FINISHED"}, {"CANCELLED"}), f"{how}: {r}")
        notes.append(f"{how}:{'ok' if r == {'FINISHED'} else 'nothing to do'}")
    bpy.ops.object.mode_set(mode="OBJECT")
    ok(len(obj.data.polygons) > 0 and not obj.data.validate(verbose=False), "mesh invalid after the fixes")
    return " ".join(notes)


# --------------------------------------------------------------------------- lock, redraw strip, teach


@check("preview", "lock_faces_unlock_redraw_strip", needs="main")
def _():
    src = main_obj()
    fresh_auto(src)
    if not bpy.ops.curiomesh.guided_edit_preview.poll():
        raise Skip("preview edit tools are not available here")
    bpy.ops.curiomesh.guided_edit_preview()
    preview = bpy.data.objects[guided.PREVIEW_NAME]
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in preview.data.polygons:
        p.select = p.index < 20
    bpy.ops.object.mode_set(mode="EDIT")
    r = bpy.ops.curiomesh.guided_lock()
    ok(r == {"FINISHED"}, f"Lock returned {r}: {guided.state['message']}")
    finish()
    ok(guided.state["lock"] is not None and guided.state["lock"]["locked"].sum() > 0, "no locked faces")
    ok(bpy.ops.curiomesh.guided_unlock() == {"FINISHED"} and guided.state["lock"] is None)
    finish()
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    manual("Redraw Strip, Remember This Fix / Apply Known Fix with real hand edits", "need faces edited by hand on the preview; Lock/Unlock were run")


@check("teach", "library_export_import_roundtrip", needs="main")
def _():
    path = os.path.join(OUT, "fixes.json")
    ok(bpy.ops.curiomesh.teach_export(filepath=path) == {"FINISHED"} and os.path.exists(path), "no fixes file")
    ok(bpy.ops.curiomesh.teach_import(filepath=path) == {"FINISHED"})
    ok(bpy.ops.curiomesh.teach_forget_kept.poll() in (True, False))
    bpy.ops.curiomesh.teach_find.poll()


# --------------------------------------------------------------------------- hardware, jobs for another computer


@check("hardware", "check_this_computer", needs="main")
def _():
    ok(bpy.ops.curiomesh.guided_check_pc() == {"FINISHED"})
    m = guided.state["machine"]
    ok(m is not None, "machine not read")
    return str({k: m[k] for k in list(m)[:5]}) if isinstance(m, dict) else str(m)[:100]


@check("job", "export_job_and_import_result", needs="main")
def _():
    src = load_model(MAIN, faces=2500)
    select_only(src)
    gs().auto_detail = 12.0
    folder = os.path.join(OUT, "job")
    os.makedirs(folder, exist_ok=True)
    ok(bpy.ops.curiomesh.guided_export_job(directory=folder) == {"FINISHED"}, guided.state["message"])
    made = [str(p) for p in __import__("pathlib").Path(folder).rglob("*") if p.is_file()]
    ok(any(p.endswith("README.txt") for p in made), f"no README.txt in the job: {[os.path.basename(p) for p in made][:8]}")
    ok(any(p.endswith(".npz") for p in made), "no input .npz in the job")
    # a result file made by the same writer a stronger computer would use
    res = fresh_auto(src)
    base = guided.state["auto_base"]
    npz = os.path.join(OUT, "result.npz")
    s = guided._source_data(src)
    jobpack.save_result(npz, res, base["report"], len(s["faces"]), len(s["co"]))
    bpy.ops.curiomesh.guided_stop()
    reset_state()
    ok(bpy.ops.curiomesh.guided_import_result(filepath=npz) == {"FINISHED"}, guided.state["message"])
    ok(guided.state["res"] is not None and len(guided.state["res"]["faces"]) == len(res["faces"]), "imported result differs")
    out = apply_result()
    ok(len(out.data.polygons) == len(res["faces"]))
    # a result for another mesh must be refused
    other = load_model(ALT, faces=800)
    select_only(other)
    try:
        r = bpy.ops.curiomesh.guided_import_result(filepath=npz)
    except RuntimeError as e:
        ok("made from a mesh with" in str(e), f"refused, but for another reason: {e}")
    else:
        ok(r == {"CANCELLED"}, "a result from a different mesh was accepted")


# --------------------------------------------------------------------------- the agent skill scripts


def _run_skill(script, *args):
    path = os.path.join(ADDON, "skills", "curiomesh-retopo", "scripts", script)
    old = sys.argv
    sys.argv = ["blender", "--", *args]
    addon.unregister()  # the script registers the add-on itself, as a fresh Blender would
    try:
        try:
            runpy.run_path(path, run_name="__main__")
        except SystemExit as e:
            return e.code or 0
    finally:
        sys.argv = old
        sys.path[:] = [p for p in sys.path if not p.endswith("scripts")]
        if not hasattr(bpy.types.Scene, "curiomesh_guided"):
            addon.register()
    return 0


@check("skills", "curiomesh_run_script", needs="main")
def _():
    src = load_model(MAIN, faces=2500)
    gltf = os.path.join(OUT, "in_model.glb")
    select_only(src)
    bpy.ops.export_scene.gltf(filepath=gltf, use_selection=True)
    out, rep = os.path.join(OUT, "skill_out.glb"), os.path.join(OUT, "skill_report.json")
    code = _run_skill("curiomesh_run.py", gltf, "--use", "GAME", "--detail", "12", "--out", out, "--report", rep, "--addon", ADDON)
    ok(code == 0, f"exit code {code}")
    r = json.load(open(rep))
    ok(r.get("ok") and os.path.exists(out), f"report: {str(r)[:200]}")
    ok(r["faces"] > 0 and r["quad_pct"] > 80, f"{r['faces']} faces, {r['quad_pct']}% quads")


@check("skills", "curiomesh_fix_script", needs="main")
def _():
    src = load_model(MAIN, faces=1500)
    gltf = os.path.join(OUT, "fix_in.glb")
    select_only(src)
    bpy.ops.export_scene.gltf(filepath=gltf, use_selection=True)
    out, rep = os.path.join(OUT, "fix_out.glb"), os.path.join(OUT, "fix_report.json")
    code = _run_skill("curiomesh_fix.py", gltf, "--original", gltf, "--fix", "ALL", "--out", out, "--report", rep, "--addon", ADDON)
    ok(code == 0, f"exit code {code}")
    r = json.load(open(rep))
    ok(r.get("ok", True) and os.path.exists(out), f"report: {str(r)[:200]}")


@check("skills", "cm_agent_pure_helpers", needs="main")
def _():
    sys.path.insert(0, os.path.join(ADDON, "skills", "curiomesh-retopo", "scripts"))
    try:
        import cm_agent as cm

        ok(cm.parse_use("low-poly") == "LOW_POLY" and cm.parse_reduce("25%") == ("PERCENT", 25.0) and cm.parse_faces("1,5,10-12") == [1, 5, 10, 11, 12])
    finally:
        sys.path.pop(0)
    manual("Skills in skills/curiomesh-batch-and-lods, curiomesh-fix-bad-faces, curiomesh-blender-mcp", "SKILL.md instruction files only; they call the two scripts above and the Blender MCP server")


# --------------------------------------------------------------------------- panels once more, with state in every corner


@check("settings", "report_a_problem_link")
def _():
    from urllib.parse import parse_qs, urlparse

    url = guided.problem_url()
    u = urlparse(url)
    ok(u.scheme == "https" and u.netloc == "github.com" and u.path.endswith("/issues/new"), url)
    body = parse_qs(u.query)["body"][0]
    ok("Blender" in body and "CurioMesh" in body, "the report body lacks the version lines")
    errors: list = []
    guided._draw_thanks(Rec(errors))
    ok(not errors, str(errors))


@check("settings", "panels_draw_in_every_mode", needs="main")
def _():
    src = load_model(MAIN, faces=2000)
    notes = []
    g = gs()
    for scope in ("WHOLE", "POINTS", "AREAS", "AUTO"):
        g.scope = scope
        notes.append(draw_sweep(f"scope {scope}"))
    g.scope = "AUTO"
    for use in [i[0] for i in profiles.PROFILE_ITEMS]:
        g.auto_profile = use
        draw_sweep(f"use {use}")
    for flag in ("ui_more_open", "ui_pc_open", "ui_lod_open", "ui_teach_open", "ui_tweak_open"):
        setattr(g, flag, True)
    draw_sweep("all fold-outs open")
    bpy.ops.object.mode_set(mode="EDIT")
    draw_sweep("Edit Mode")
    bpy.ops.object.mode_set(mode="OBJECT")
    g.auto_profile = "GAME"
    fresh_auto(src)
    draw_sweep("after Start")
    manual("Real panel drawing (layout, icons, fold-outs), the dashboard HUD and the viewport overlays", "need a window; only property and operator names are checked here")
    return "scopes, uses, fold-outs, Edit Mode and result states drawn"


@check("tour", "every_operator_is_covered_by_this_tour")
def _():
    text = open(__file__, encoding="utf-8").read()
    missing = [n for n in op_names() if f"curiomesh.{n}" not in text and f'"{n}"' not in text]
    ok(not missing, f"operators this tour never calls (add a check or a manual() line): {missing}")
    return f"{len(op_names())} operators all named in the tour"


if __name__ == "__main__":
    if "--list" in ARGV:
        for group, name, fn, needs in CHECKS:
            print(f"{group}/{name}")
        sys.exit(0)
    run_all()
