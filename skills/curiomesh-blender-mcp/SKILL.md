---
name: curiomesh-blender-mcp
description: Drive CurioMesh inside an already running Blender through the Blender MCP server (execute_blender_code), to retopologise or fix the user's open scene. Use when the user has Blender open with the MCP connection and wants CurioMesh applied to a selected object, and headless file-to-file runs are not what they want.
---

# CurioMesh through Blender MCP

Only for agents that have a Blender MCP tool such as `execute_blender_code`. If you do not, use `curiomesh-retopo` instead. This works on the user's live scene: ask before changing anything, and never touch objects the user did not name.

Rules: the add-on's run is asynchronous in the UI, so a script must step the job itself. Keep each call short; a long call freezes Blender. Do not run this on a huge mesh without telling the user how long it takes.

## 1. Check the add-on is on

```python
import bpy
{"has_curiomesh": hasattr(bpy.context.scene, "curiomesh_guided"), "selected": [o.name for o in bpy.context.selected_objects if o.type == "MESH"], "faces": {o.name: len(o.data.polygons) for o in bpy.context.selected_objects if o.type == "MESH"}}
```

## 2. Start (one mesh selected and active)

```python
import bpy, sys
mod = next(m for m in sys.modules if m.endswith("curiomesh") or m == "curiomesh")
guided = sys.modules[mod + ".guided"]
gs = bpy.context.scene.curiomesh_guided
gs.scope = "AUTO"; gs.auto_profile = "GAME"; gs.auto_detail = 6.0
bpy.ops.curiomesh.guided_auto_start()
```

For the Auto button's own choice use `bpy.ops.curiomesh.guided_auto_all()` instead of the three setting lines. Uses: LOW_POLY, GAME, ANIMATION, SUBDIV, SCAN, SCULPT.

## 3. Step to the end

Repeat this call until it returns `done: true` (each call works about 20 seconds):

```python
import bpy, sys, time
guided = sys.modules[next(m for m in sys.modules if m.endswith("curiomesh.guided") or m == "curiomesh.guided")]
work = guided.state["work"]; r = work["runner"]; t = time.time()
while r.busy and time.time() - t < 20:
    r.step(0.04)
guided._collect_work(bpy.context)
{"done": not r.busy, "message": guided.state["message"]}
```

The state dict `work` does not clear, so test `r.busy`, not `work`.

## 4. Read, then apply

```python
res = guided.state["res"]; base = guided.state["auto_base"]
{"faces": len(res["faces"]), "view_changed_pct": base["report"]["loss"], "issues": res.get("issues", {}).get("totals")}
```

Show the user these numbers (see `curiomesh-retopo` for what is good). Only after they agree, `bpy.ops.curiomesh.guided_apply()`. It writes a new object `<name>_Retopo` next to the original; the original is untouched, and Undo works.

## 5. Fix bad faces in the open scene

Select the result in Edit Mode, set `bpy.context.scene.curiomesh_fix.source` to the original object, then `bpy.ops.curiomesh.fix_find()` (selects the bad faces and reports counts) and `bpy.ops.curiomesh.fix_run(how="ALL")` (or CLEAN, FLIP, UNTANGLE, RELAX, PATCH). Read the counts from `bpy.context.scene.curiomesh_fix.last`. If the count is not lower than before, press Undo (`bpy.ops.ed.undo()`) and stop. See `curiomesh-fix-bad-faces` for which fix suits which kind.
