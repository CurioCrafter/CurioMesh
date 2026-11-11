# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Draw every CurioMesh panel and popover in each state of the main path against a recording layout, headless.

blender -b --factory-startup --python tests/blender_panel_states.py -- [OUT.md]

States: no mesh, ready, working, result, fixing. Fails on any draw error, unknown icon, property or operator, and when
the main tab shows more controls than the design budget. OUT.md gets the recorded layout of each state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
RECORDED: dict = {}  # state -> [(panel label, closed, [(depth, kind, words)])], saved as JSON beside OUT.md
BUDGET = {"none": 3, "ready": 27, "working": 13, "result": 54, "fixing": 80}  # main tab at default open state, ? buttons not counted. Set to the 0.9.9 panel (2026-11-10): a guard against further growth; lower it when the panel is condensed


def load_addon():
    for name in list(sys.modules):
        if name == "curiomesh" or name.startswith("curiomesh."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("curiomesh", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["curiomesh"] = module
    spec.loader.exec_module(module)
    module.register()
    return module


ICONS = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())


class Op:
    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


class Rec:
    """A UILayout stand-in that checks names and records what a panel draws: (depth, kind, words)."""

    def __init__(self, log, depth=0):
        self._log, self._depth = log, depth

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def _sub(self, *args, **kwargs):
        return Rec(self._log, self._depth + 1)

    row = column = split = grid_flow = column_flow = _sub

    def box(self, *args, **kwargs):
        self._log.append((self._depth, "box", ""))
        return Rec(self._log, self._depth + 1)

    def _icon(self, icon):
        if icon not in ICONS:
            raise AssertionError(f"unknown icon {icon!r}")

    def label(self, text="", icon="NONE", icon_value=0, **kwargs):
        self._icon(icon)
        self._log.append((self._depth, "label", text))

    def separator(self, *args, **kwargs):
        pass

    def progress(self, factor=0.0, type="BAR", text=""):
        self._log.append((self._depth, "label", f"[progress] {text}"))

    def template_icon(self, icon_value=0, scale=1.0):
        self._log.append((self._depth, "picture", str(icon_value)))

    def prop(self, data, name, text=None, icon="NONE", expand=False, **kwargs):
        self._icon(icon)
        rna = data.bl_rna.properties.get(name)
        if rna is None:
            raise AssertionError(f"{data.bl_rna.identifier} has no property {name!r}")
        count = len(rna.enum_items) if expand and rna.type == "ENUM" else 1
        self._log.append((self._depth, f"prop x{count}" if count > 1 else "prop", f"{name}" + (f" '{text}'" if text else "")))

    def prop_enum(self, data, name, value, text="", **kwargs):
        if name not in data.bl_rna.properties:
            raise AssertionError(f"no property {name!r}")
        self._log.append((self._depth, "prop", f"{name}={value} '{text}'"))

    def operator(self, idname, text="", icon="NONE", **kwargs):
        self._icon(icon)
        group, _, op = idname.partition(".")
        try:
            getattr(getattr(bpy.ops, group), op).get_rna_type()
        except Exception as exc:
            raise AssertionError(f"unknown operator {idname!r}") from exc
        self._log.append((self._depth, "operator", f"{idname} '{text}'"))
        return Op()

    def popover(self, panel, text="", icon="NONE", **kwargs):
        self._icon(icon)
        if not hasattr(bpy.types, panel):
            raise AssertionError(f"unknown popover {panel!r}")
        self._log.append((self._depth, "popover", f"{panel} '{text}'"))

    def template_list(self, listtype, list_id, data, propname, active_data, active_propname, **kwargs):
        if propname not in data.bl_rna.properties or active_propname not in active_data.bl_rna.properties:
            raise AssertionError(f"bad template_list {propname!r}")
        self._log.append((self._depth, "list", propname))


CONTROLS = ("prop", "operator", "popover", "list")


def controls(log) -> int:
    total = 0
    for _depth, kind, _words in log:
        if kind.startswith("prop x"):
            total += int(kind[6:])
        elif kind in CONTROLS and not _words.startswith("curiomesh.help"):  # the small ? buttons explain, they don't add choices
            total += 1
    return total


def draw(cls, label):
    log = []
    holder = type("Holder", (), {"layout": Rec(log)})()
    for hook in ("draw_header", "draw_header_preset", "draw"):
        fn = cls.__dict__.get(hook) or (getattr(cls, hook, None) if hook == "draw" else None)
        if fn is None:
            continue
        try:
            fn(holder, bpy.context)
        except Exception as exc:
            raise AssertionError(f"[{label}] {cls.__name__}.{hook}: {exc!r}") from exc
    return log


def snapshot(panel, label, out):
    """Draw the main tab as a user sees it at the default open state, then every popover and Pro panel on its own."""
    main_total, lines = 0, []
    for cls in panel.MAIN_PANELS:
        if not cls.poll(bpy.context):
            continue
        closed = "DEFAULT_CLOSED" in getattr(cls, "bl_options", set())
        log = draw(cls, label)
        RECORDED.setdefault(label, []).append((cls.bl_label, closed, log))
        count = controls(log)
        main_total += 1 if closed else count  # a closed sub-panel shows its header only
        lines.append(f"- **{cls.bl_label}** ({'closed: header only' if closed else f'{count} controls'})")
        if not closed:
            lines += [f"  {'  ' * d}{kind}: {words}" for d, kind, words in log]
    extra = []
    for cls in (*panel.POPOVERS, *panel.PRO_PANELS):
        if hasattr(cls, "poll") and "poll" in cls.__dict__ and not cls.poll(bpy.context):
            continue
        extra.append(f"{cls.bl_label} {controls(draw(cls, label))}")
    out.append(f"### {label}: {main_total} controls in the main tab\n")
    out += lines
    out.append(f"\nPopovers and Pro panels (controls when opened): {', '.join(extra)}\n")
    print(f"STATE {label}: main tab {main_total} controls", flush=True)
    if main_total > BUDGET[label]:
        raise AssertionError(f"{label}: {main_total} controls in the main tab, budget {BUDGET[label]}")
    return main_total


def check_no_jump(panel, guided):
    """Status messages and Auto Fix progress come and go: the rows of the main panel, Adjust and Fix, and the place of
    the Fix Bad Spots button and the Quad Size slider, must stay the same, or the buttons jump under the mouse."""
    from types import SimpleNamespace

    from curiomesh import autofix_live, ladder

    classes = (panel.CURIOMESH_PT_guided, panel.CURIOMESH_PT_guided_adjust, panel.CURIOMESH_PT_guided_parts)

    def shape():
        logs = [draw(cls, "nojump") for cls in classes]
        fix = [[i for i, (_d, k, w) in enumerate(log) if k == "operator" and w.startswith("curiomesh.seg_review")] for log in logs]
        quad = [i for i, (_d, k, w) in enumerate(logs[1]) if k == "prop" and w.startswith("quad_size")]
        return [len(log) for log in logs], fix, quad

    idle = shape()
    saved = (guided.state["message"], ladder.status, autofix_live.fx["job"])
    try:
        guided.state["message"] = "Checking Part 2"
        ladder.status = lambda: "Working out more sizes: 3 ready, 5 to go (Even rows, 40%)"
        busy = shape()
        autofix_live.fx["job"] = SimpleNamespace(busy=True, status="Auto Fix, round 1: relaxing, 3 of 9 spots", fraction=0.3)
        working = shape()
    finally:
        guided.state["message"], ladder.status, autofix_live.fx["job"] = saved
    assert idle == busy == working, f"panel rows move while it works: idle {idle}, message {busy}, Auto Fix {working}"
    assert idle[1][0] or idle[1][2], "Fix Bad Spots is not drawn where it should be"


def main():
    module = load_addon()
    from curiomesh import guided, issue_panel, overlay, panel, segments

    addons = bpy.context.preferences.addons
    entry = addons.get("curiomesh") or addons.new()
    entry.module = "curiomesh"
    overlay.get_prefs().pro_mode = True
    out = ["# Recorded panel layout per state\n", "Drawn headless by tests/blender_panel_states.py against a recording layout.\n"]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.context.view_layer.objects.active = None
    snapshot(panel, "none", out)

    bpy.ops.mesh.primitive_monkey_add()
    obj = bpy.context.object
    obj.modifiers.new("s", "SUBSURF").levels = 2
    bpy.ops.object.modifier_apply(modifier="s")
    gs = bpy.context.scene.curiomesh_guided
    gs.auto_update = False
    snapshot(panel, "ready", out)
    gs.use_auto = True
    gs.auto_why = "8,000 faces in 1 loose part, smooth with few flats: an organic shape."
    draw(panel.CURIOMESH_PT_guided, "ready/auto")
    gs.use_auto = False
    for pick in range(8):  # every Detail choice draws
        gs.detail_pick = gs.bl_rna.properties["detail_pick"].enum_items[pick].identifier
        draw(panel.CURIOMESH_PT_guided, f"ready/detail{pick}")
    gs.detail_pick = "BALANCED"
    for value in ("QUICK", "NORMAL", "THOROUGH"):
        gs.cleanup = value
        assert gs.cleanup == value, (value, gs.cleanup)
    gs.cleanup = "NORMAL"
    assert gs.auto_redo_rough and not gs.auto_tidy

    if bpy.ops.curiomesh.guided_auto_start() != {"FINISHED"}:
        raise AssertionError(f"Start did not start: {guided.status()}")
    if guided.progress_info("AUTO") is None:
        raise AssertionError("no job running after Start")
    snapshot(panel, "working", out)
    for _ in range(50):
        work = guided.state["work"]
        if work is None:
            break
        while work["runner"].busy:
            work["runner"].step(0.04)
        guided._collect_work(bpy.context)
        if guided.state["work"] is work:
            break
    segments.ensure()
    if guided.state["res"] is None:
        raise AssertionError(f"no result: {guided.status()}")
    snapshot(panel, "result", out)

    res = guided.state["res"]  # a made-up bad spot on part 1, so the spot card draws
    res["issue_spots"] = [{"kinds": {"folded_over": 3, "sliver": 2}, "faces": [0, 1, 2], "part": 0, "center": [0.0, 0.0, 0.0], "radius": 0.1}]
    res["issues"] = {"totals": {"folded_over": 3, "sliver": 2}}
    issue_panel.ui.update(res=res, cur=0)
    segments.select([0])
    panel.CURIOMESH_PT_guided_parts.bl_options = set()  # Fix opened
    snapshot(panel, "fixing", out)
    draw(panel.CURIOMESH_PT_guided, "fixing/main")
    check_no_jump(panel, guided)
    if len(sys.argv) > sys.argv.index("--") + 1 if "--" in sys.argv else False:
        target = Path(sys.argv[sys.argv.index("--") + 1])
        target.write_text("\n".join(out) + "\n", encoding="utf-8")
        import json

        target.with_suffix(".json").write_text(json.dumps(RECORDED, indent=1), encoding="utf-8")
    module.unregister()
    print("PASS panel_states", flush=True)


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    sys.exit(1)
