---
name: curiomesh-agent
description: Drive CurioMesh command by command (settings, Auto, Tidy Tools, Auto Fix, parts, History, bake, export), in a Blender that is open or a headless one, through the CurioMesh MCP server or a mailbox folder. Also sends bad parts out for better topology and brings them back (parts exchange). Use when the user wants an agent to steer CurioMesh step by step and read the numbers, not run one file-to-file job.
---

# CurioMesh agent (commands and parts exchange)

CurioMesh has one command layer for agents. Every command is a name plus JSON arguments and returns one JSON reply with `ok`, a `result` or an `error`, and `step` (the History step after the command). Full list and error codes: `docs/AGENTS.md` in the CurioMesh folder. For a plain file-to-file run use `curiomesh-retopo` instead. This works in the GitHub add-on (0.9.9 and later), not in the extensions.blender.org build.

Ask before changing the user's open scene, and touch only the objects they named.

## 1. Connect

Pick one.

- **Open Blender (mailbox).** The user turns on *Let an Agent Drive CurioMesh* (Pro tab, This Computer) and gives you the folder it shows. Start the MCP server on that folder: `python <this skill>/scripts/curiomesh_mcp.py --folder FOLDER`.
- **No window.** `python <this skill>/scripts/curiomesh_mcp.py` starts its own headless Blender with `cm_do.py --serve`. Blender is found from `$BLENDER`, PATH or the usual install places; say which one you used. If CurioMesh is a repo checkout, set `CURIOMESH_REPO` or pass `--addon`. Run one Blender at a time.
- **No MCP.** Write `inbox/<id>.json` as `<id>.json.tmp` and rename it; read `outbox/<id>.json`. Or give `cm_do.py` a command file: `blender -b --factory-startup --python cm_do.py -- --commands job.json --out replies.jsonl`.

With MCP every command is a tool: `auto.start` is `auto_start`. `curiomesh_call {command, args}` reaches any command by its real name. Every tool takes `object` (mesh name) and `timeout` (seconds). Leave `timeout` out to wait until Blender answers; there is no other limit.

Start with `session.info`, then `commands.list` (help, arguments, which commands edit).

## 2. The loop: list, act, read numbers, keep or step back

1. **List.** Read the state before touching it: `scene.objects`, `result.numbers`, `parts.list`, `issues.list`, `history.list`. Read options with `settings.list {group}`: it shows every option with its help, default and range.
2. **Act.** One change at a time: `use.pick`, `settings.set`, `auto.start`, `tools.run`, `autofix.run`, `parts.redo`. A long job answers `running: true`; ask `job.status` or `job.wait {seconds}` until it is done.
3. **Read the numbers.** `result.numbers` (faces, quad %, view change %, verdict), and `result.report` for issues per part. Good looks like the retopo skill says: view change at or under the Detail, quads near 100 %, no inward, crossing, folded or sunk faces.
4. **Keep or step back.** Better: keep it, and run the same command again (`tidy.again`, `tools.again`) or its Keep going form (`tidy.keep_going`, `tools.run {keep_going: true}`, `autofix.keep_going`, `loops.fix_all {keep_going: true}`). Keep going stops by itself when a pass gains nothing. Keep going while the numbers improve; stop when a pass gains nothing. Worse or unchanged: `history.jump {step}` back to the step you noted before acting.

Every kept change is a History step, so nothing is lost by trying. Note the `step` before each experiment.

## 3. Errors

A reply with `ok: false` has `error.code`, `message`, `hint` and `details`, and the mesh is unchanged (`step` is the same as before). Read the `hint` and act on it; do not resend the same command. Common ones: `BUSY` (wait for the job), `NO_RESULT` (run `auto.start`), `BAD_ARGS` (the allowed values are in `details`), `NOT_READY` (the result does not allow it; `message` says why), `REFUSED` (it would make things worse). `INTERNAL` is a bug: tell the user and include `details.trace`. If the MCP server says `NO_ANSWER`, your timeout ran out; the job may still be running, so check `job.status`.

## 4. Parts exchange recipe

Use this when a few parts are bad and you can build better topology for them yourself (or the user has another tool for it). Format: `docs/PARTS_EXCHANGE.md`.

1. Find the bad parts: `parts.worst {count}` or `issues.list`. Mark them: `parts.select {parts}`.
2. `parts.export {parts, note}` returns a folder. Read `parts.json` and, for each part, `part_<id>.obj` (current faces), `part_<id>_source.obj` (the surface to follow) and `part_<id>_around.obj` (neighbouring faces, read only).
3. Make a better part and write it to `returned/part_<id>.obj`. Rules: every border vertex stays exactly where `parts.json` puts it (only the inside may change); faces have 3 to 64 corners, quads preferred; follow the source surface; match the flow of the around file.
4. Note the `step`, then `parts.import {folder}`. Each part is `kept` or `refused` with a code: `BORDER_MOVED` (a border vertex moved), `HOLE`, `SHAPE_LOST`, `VIEW_CHANGED` (outside Most Detail Loss; tell the user, they may raise it), `MISMATCH` (the result changed since export: export again), `VERSION`, `NOT_READY`.
5. Fix what was refused and import again. Parts that failed do not stop the others. Everything kept is one History step: `history.jump` to the noted step takes it all out.
6. Read `result.numbers` and the bad spots of the new faces (reported, not blocking). If it is worse, step back.
7. To share the parts in an issue report, `parts.report {parts, note}` writes a zip; CurioMesh sends nothing. Tell the user it contains pieces of their mesh.

## Honest limits

The commands run on Blender's main thread, one at a time. A big model's Auto still takes minutes. Part exchange needs a result that allows part redos.
