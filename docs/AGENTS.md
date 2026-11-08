# CurioMesh for agents

An agent can do everything a person can do in CurioMesh: start Auto, set any option, read the result and its
numbers, use every Tidy Tool, Face Loops tool, Auto Fix, issue fix and part edit, step through History, bake, make
LODs, export and run diagnostics. When something goes wrong it gets a clear error back.

This page is the index. Parts exchange (sending bad parts out and bringing better ones back) has its own page:
[PARTS_EXCHANGE.md](PARTS_EXCHANGE.md). Skills that teach an agent all this are in [`skills/`](../skills).

Agent tools are in the GitHub add-on only. The extensions.blender.org build does not have them.

## One command layer, three routes

Every command is a name plus JSON arguments, and every answer is one JSON reply. The same commands sit behind all
three routes, so what works on one works on the others.

| Route | Pick it when | How it runs |
|---|---|---|
| **Mailbox** (live Blender) | You have Blender open and want an agent to work on the scene you see. | Turn on *Let an Agent Drive CurioMesh* (Pro tab, This Computer). CurioMesh reads `inbox/*.json` from a folder and writes `outbox/<same name>.json`. Files only, no network. |
| **Headless runner** `cm_do.py` | Batch work, a job on another computer, anything with no window. | `blender -b --factory-startup --python cm_do.py -- --commands job.json --out replies.jsonl` runs the commands in order and prints one reply per line. `--serve FOLDER` runs the mailbox without a window, so one Blender keeps its state across many calls. |
| **MCP server** `curiomesh_mcp.py` | Your agent speaks MCP (Claude Code and others). | Plain Python, standard library only, in `skills/curiomesh-agent/scripts/`. `python curiomesh_mcp.py --folder FOLDER` uses the mailbox of an open Blender; with no `--folder` it starts its own headless Blender. Each command becomes one MCP tool (`auto.start` is `auto_start`). |

How to choose in one line: window open and the user watching, use the mailbox (through MCP if you can). No window,
use `cm_do.py --serve` (through MCP if you can) or a plain command file.

### The mailbox

```json
// inbox/0007.json   (written as 0007.json.tmp, then renamed, so it is never read half written)
{"id": "0007", "command": "auto.start", "args": {"use": "GAME"}}
// outbox/0007.json
{"id": "0007", "ok": true, "result": {"started": true}, "step": 12, "seconds": 0.01}
```

- Messages run one per tick, in name order, on Blender's main thread. Nothing blocks the window.
- A long job (Auto, Auto Fix, a part check) answers at once with `"running": true`. Then ask `job.status`.
- `job.wait {"seconds": N}` is for headless runs: it steps the job itself until it is done or N seconds have passed.
- There is no limit on how long you wait. The only time limit is the one you pass yourself.

## The older routes

They stay and keep their flags.

| Route | What it is | When it is better |
|---|---|---|
| `curiomesh_run.py` and `curiomesh_fix.py` (in `skills/curiomesh-retopo/scripts/`) | One file in, one file and a JSON report out, run through `blender -b --python`. | A single file-to-file job where you want the whole thing in one call and do not need to look inside. |
| **Export Job** (Auto Segments panel, This Computer) | Writes a folder with the mesh, the settings and a runner, so another computer with Blender can run the same search. *Import Result* brings the answer back. | The work needs a stronger machine. See [CLOUD_WORKFLOW.md](CLOUD_WORKFLOW.md). |
| Standalone `curiomesh` command line (branch `standalone`) | A program that does not need Blender at all. | A machine with no Blender, or a plain batch of files where you only want the finished mesh. It has no History, no part edits and no tools, so for anything you want to steer, use the routes above. |

Blender-side skill `curiomesh-blender-mcp` now calls `agentcmd.run(...)` through the Blender MCP server's
`execute_blender_code`, instead of reaching into CurioMesh's internals.

## Commands

Names are `area.verb`. Every command takes an optional `object` (a mesh name); without it the command uses the
active mesh. A command that changes anything runs the same operator or function a click runs, so it gets the same
History entry and the same Ctrl+Z step. The reply carries `step`, the History step number after the change.

`commands.list` returns every command with plain-words help, its argument schema and whether it edits. Always ask
it: it is the full set. This table is the starting set.

| Area | Commands |
|---|---|
| Session and diagnostics | `session.info`, `commands.list`, `diag.check_pc`, `diag.log {lines}`, `scene.objects`, `scene.select {object}`, `file.load {path}`, `file.save {path}` |
| Settings (every option) | `settings.groups`, `settings.list {group}`, `settings.get {group, names?}`, `settings.set {group, values}`, `use.pick {use}` |
| Auto and the result | `auto.start`, `auto.start_for_me`, `job.status`, `job.wait {seconds}`, `job.stop`, `result.numbers`, `result.report`, `result.update`, `result.discard`, `size.set {detail}`, `size.parts {bigger, parts}`, `apply {mode, bake?, uvs?, multires?, lods?}` |
| Tidy Up | `tidy.again`, `tidy.keep_going`, `tidy.stop`, `tidy.back`, `tidy.levels` |
| Tidy Tools | `tools.list`, `tools.scope {scope, faces?}`, `tools.preview {tool, settings}`, `tools.run {tool, settings, keep_going?}`, `tools.again`, `tools.every {tool}` |
| Face Loops and Flow Editor | `loops.strips {kind?}`, `loops.edit {strip, how, amount}`, `loops.fix_all {keep_going?}`, `loops.ring {...}`, `loops.redraw {faces or part}`, `loops.move_pole {vertex, to}`, `flow.add {kind, points}`, `flow.list`, `flow.remove {index}`, `flow.clear` |
| Auto Fix, issues, Fix Selection | `autofix.groups`, `autofix.set {ticks}`, `autofix.run {here?, time_limit?}`, `autofix.keep_going`, `autofix.stop`, `autofix.undo`, `autofix.findings`, `issues.list`, `issues.redo {index}`, `issues.keep_original {index}`, `issues.put_back {part}`, `issues.save {path}`, `issues.compare {path}`, `fix.find {faces?}`, `fix.run {how, faces?, grow?}` |
| Parts | `parts.list`, `parts.select {parts, extend?}`, `parts.worst {count}`, `parts.rename`, `parts.lock`, `parts.merge`, `parts.halve`, `parts.from_faces {faces}`, `parts.redo {parts, use?, method?, detail?}`, `parts.tries {part}`, `parts.use_try {part, index}`, `parts.keep_original {parts}`, `parts.export`, `parts.import`, `parts.report` |
| History and undo | `history.list`, `history.jump {step}`, `history.undo`, `history.redo` |
| Bake, LOD, export | `lod.make {lod0, lod1, lod2}`, `bake.set {detail, size, sets}`, `export.game {...}`, `export.file {path}`, `job.export_folder {folder, effort}`, `job.import_result {path}` |

Things to know:

- **Every option is reachable.** `settings.list {group}` reads each property from Blender's own description of it
  (name, help, type, value, default, ranges, enum choices), so new options show up on their own. `settings.set`
  goes through the property's own update, so a value set by an agent behaves exactly like a slider dragged by hand.
  Out-of-range values and unknown names answer `BAD_ARGS` with the allowed range or choices.
- **Again and Keep going.** Every command that improves something can run again on the current result
  (`tidy.again`, `tools.again`). The Keep going versions (`tidy.keep_going`, `tools.run {keep_going}`,
  `loops.fix_all {keep_going}`, `autofix.keep_going`) repeat while a pass still gains something and stop when one
  gains nothing. Nothing is capped by a number chosen for convenience. If a time limit runs out the reply says so.
- **Most Detail Loss** is the user's own guard. It is respected, reported and changeable (`settings.set`), never
  hidden.
- **Changes are all or nothing.** A command that fails leaves the mesh and History as they were. `step` in the reply
  proves it.
- **Modal tools.** Tools that need the mouse in the panel run through their underlying function instead, so an agent
  gives numbers (a ring's centre and size, a strip id) in place of mouse moves.

## Errors

Every reply has the same shape:

```json
{"id": "0009", "ok": false,
 "error": {"code": "BORDER_MOVED", "message": "Part 8: border point 14 of loop 1 moved 0.004 m",
           "hint": "Keep every border vertex where parts.json puts it; only the inside may change",
           "details": {"part": 7, "loop": 0, "point": 14, "off": 0.004}},
 "step": 12}
```

The list of codes is fixed: new ones may be added, old ones never change meaning.

| Code | Means |
|---|---|
| `BAD_ARGS` | Missing or wrong argument. `details` gives the allowed values or range. |
| `UNKNOWN_COMMAND` | No such command. `details.close` names the nearest ones. |
| `NO_OBJECT` | No mesh to work on. |
| `NO_RESULT` | Needs a result on screen. Run `auto.start` first. |
| `BUSY` | Something is running. `details` holds `job.status`. |
| `NOT_READY` | The result does not allow this (Mirror, Even Rows, an imported job, a cut piece). |
| `NOT_FOUND` | The part, spot, strip, tool or History step is not there. |
| `REFUSED` | The tool would make things worse (more bad faces, a face stops being a quad). Nothing changed. |
| `FILE` | A file can't be read or written, or it is damaged. |
| `VERSION` | A part file from a newer format. |
| `MISMATCH` | A part file made from a different result than the one on screen. |
| `BORDER_MOVED`, `HOLE`, `SHAPE_LOST`, `VIEW_CHANGED` | Part import checks. See [PARTS_EXCHANGE.md](PARTS_EXCHANGE.md). |
| `NEEDS_WINDOW` | Only works in a live window (for example a Look First preview). The hint gives the headless way. |
| `INTERNAL` | A bug. `details.trace` holds the last lines of the traceback. Please report it. |

The MCP server adds one code of its own, `NO_ANSWER`, when your `timeout` ran out before Blender replied. The job may
still be running: ask `job.status` before you send the command again.

## A ten-line example session

An agent over MCP (or the mailbox) fixing one model for a game. Each line is one command and what the agent reads.

```
 1  session.info                          -> CurioMesh 0.9.9, Blender 4.2, active "Crate", no result yet
 2  use.pick {"use": "GAME"}              -> ok
 3  auto.start                            -> running: true
 4  job.wait {"seconds": 60}              -> running: true, "Part 9 of 14"   (ask again; no cap on asking)
 5  job.wait {"seconds": 60}              -> running: false
 6  result.numbers                        -> 3 120 faces, quads 98 %, view change 5.8 %, verdict ok
 7  autofix.run                           -> running: true ... then autofix.findings lists what was fixed
 8  tidy.keep_going                       -> passes listed; stops by itself when a pass gains nothing
 9  history.list                          -> step 14 on screen. A bad result? history.jump {"step": 6}
10  apply {"mode": "NEW_OBJECT"}          -> "Crate_Retopo" created, step 15
```

The loop in the skill is the same every time: list, act, read the numbers, keep the step or step back.
