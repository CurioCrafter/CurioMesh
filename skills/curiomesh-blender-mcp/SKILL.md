---
name: curiomesh-blender-mcp
description: Drive CurioMesh inside an already running Blender through the Blender MCP server (execute_blender_code), to retopologise or fix the user's open scene. Use when the user has Blender open with the MCP connection and wants CurioMesh applied to a selected object, and headless file-to-file runs are not what they want.
---

# CurioMesh through Blender MCP

Only for agents that have a Blender MCP tool such as `execute_blender_code`. If you do not, use `curiomesh-retopo` (file to file) or `curiomesh-agent` (CurioMesh's own MCP server). This works on the user's live scene: ask before changing anything, and never touch objects the user did not name.

CurioMesh (GitHub add-on, 0.9.9 and later) has one command layer, `agentcmd.run(command, args)`. It returns one dict: `ok`, `result` or `error` (`code`, `message`, `hint`, `details`), and `step`. The full command list and error codes are in `docs/AGENTS.md`; `commands.list` returns them with help. Do not reach into `guided.state`: the commands do the stepping, the History entries and the undo steps for you.

Keep each call short; a long call freezes Blender. Long jobs answer `running: true`: check `job.status` in a later call.

## 1. Find the module (every call starts with this)

```python
import bpy, sys
mod = next(m for name, m in sys.modules.items() if name.endswith(".agentcmd") or name == "agentcmd")
def cm(command, **args):
    return mod.run(command, args)
```

If `next` finds nothing, CurioMesh is not installed, is the site build, or is older than 0.9.9: tell the user.

## 2. Look, then start

```python
cm("session.info")          # version, active object, what is running
cm("scene.objects")         # meshes with face counts
cm("use.pick", use="GAME")  # AUTO, GAME, LOW_POLY, ANIMATION, SUBDIV, SCAN, SCULPT, BLOCKOUT, HARD, HULL
cm("auto.start")            # or auto.start_for_me to let CurioMesh pick the Use
```

Pass `object="Name"` to work on a mesh other than the active one.

## 3. Wait, read, decide

```python
cm("job.status")    # {running, percent, words, seconds}; call again until running is false
cm("result.numbers")  # faces, quad %, view change %, verdict
cm("result.report")   # issues per part, notes
```

Show the user these numbers (see `curiomesh-retopo` for what is good). Use `job.wait` only when no Blender window needs to stay responsive.

## 4. Improve, then apply

Run a tool, read the numbers, and run it again while it helps. If a pass gains nothing, stop; if a result is worse, go back.

```python
step = cm("history.list")        # note the step on screen
cm("tidy.keep_going")            # repeats while a pass gains something
cm("autofix.run")                # then autofix.findings
cm("history.jump", step=6)       # step back to any earlier step
```

Only after the user agrees: `cm("apply", mode="NEW_OBJECT")`. It writes `<name>_Retopo` next to the original, which stays untouched, and Ctrl+Z works.

## 5. Errors

When `ok` is false, read `error["hint"]` and `error["details"]`; nothing was changed and `step` shows it. Do not resend the same command. `BUSY` means wait; `NO_RESULT` means run `auto.start` first; `BAD_ARGS` lists the allowed values. `INTERNAL` is a bug: tell the user and include `details["trace"]`.

## 6. Fix bad faces in the open scene

`cm("issues.list")` for bad spots; `cm("fix.find")` then `cm("fix.run", how="ALL")` for Fix Selection (CLEAN, FLIP, UNTANGLE, RELAX, PATCH), also on chosen faces (`faces=[...]`). Compare the bad-face count in the replies. Keep going while it drops; stop when a round gains nothing, and `history.jump` back if it got worse. See `curiomesh-fix-bad-faces` for which fix suits which kind.
