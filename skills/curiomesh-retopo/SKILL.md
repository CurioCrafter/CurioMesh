---
name: curiomesh-retopo
description: Retopologise a 3D model into a clean all-quad, low-poly mesh with the CurioMesh Blender add-on, headless, from a file (.obj .fbx .glb .blend) to a file. Use when the user wants a high-poly mesh, scan, sculpt or game asset reduced, retopologised, quad-remeshed or made game-ready, and CurioMesh is installed or its folder is at hand.
---

# CurioMesh retopo (headless)

CurioMesh reduces a mesh part by part and checks every result against the original from six sides. `scripts/curiomesh_run.py` drives it with no Blender window and writes a JSON report you read. Nothing is uploaded.

## 1. Find Blender

Use the first that exists, and say which you used:

1. `$BLENDER` (a path to the Blender executable).
2. `blender` on PATH.
3. Windows: `C:\Program Files\Blender Foundation\Blender *\blender.exe` (newest). macOS: `/Applications/Blender.app/Contents/MacOS/Blender`. Linux: `/usr/bin/blender`, `/snap/bin/blender`, `~/blender*/blender`.

CurioMesh needs Blender 3.6 to 5.2. If none is found, ask the user for the path. Do not download Blender yourself.

## 2. Find CurioMesh

- Installed add-on: nothing to pass.
- Repo folder (the one holding `guided.py`): pass `--addon <folder>` or set `CURIOMESH_REPO`.
- If `Report` says "not installed", tell the user to install it from https://github.com/CurioCrafter/CurioMesh/releases/latest (see the README).

## 3. Pick the Use and Detail

Map what the user says; if they say nothing, use `--use AUTO` (CurioMesh looks at the mesh and picks, like its Auto button).

| User wants | `--use` | Detail |
|---|---|---|
| game prop, "game ready", default | `GAME` | 6 |
| fewest faces, background prop | `LOW_POLY` | 10 |
| character or creature that bends | `ANIMATION` | 1.5 |
| subdivision / smooth later | `SUBDIV` | 1.5 |
| noisy scan or photogrammetry | `SCAN` | 2 |
| a base to sculpt on | `SCULPT` | 6 |

Detail is the percent of the view that may differ from the original: lower keeps more faces. If the user names a face count or a share ("about 5,000 faces", "a quarter"), add `--reduce-to 5000` or `--reduce-to 25%`.

## 4. Run

```
"$BLENDER" -b --factory-startup --python <this skill>/scripts/curiomesh_run.py -- IN --use GAME --detail 6 --out OUT.glb --report report.json
```

- `IN`: `.obj .fbx .glb .gltf .blend .ply .stl`. Several meshes in one file are joined. `OUT`: `.glb .fbx .obj .blend`.
- Add `--addon <folder>` when running from a repo checkout.
- Run it in the background if your tool times out. A 60,000-face model takes minutes; scans can take tens of minutes. Keep one Blender running at a time on the user's PC.
- Read `report.json` (also printed as one `CURIOMESH_REPORT {...}` line). If `ok` is false, read `error`, fix the cause and rerun; do not retry blindly.

## 5. Read the report

Keys: `faces`, `quads`, `tris`, `quad_pct`, `source_faces`, `view_changed_pct`, `seconds`, `parts`, `issues` (`totals` per kind, `spots`, `per_part`, `worst`), `verdict` (`ok` and `problems`).

What is good (from the README's examples, Game Asset at Detail 6):
- `view_changed_pct` at or under the Detail you asked for, up to about 1.5 times it. Examples land at 5.6 to 6.0.
- `quad_pct` near 100 and `tris` under 2% of `faces`.
- `issues.totals` showing no `inward`, `crossing`, `folded_over`, `sunk` and at most a handful of `flat`, `duplicate`, `overlap`. Poles, slivers and pinched faces are untidy, not wrong.
- Face count: the README's examples keep 2% to 20% of the source on hard-surface models.

## 6. Retry when it is not good

Change one thing per retry, at most three retries, and keep the best result:
- `view_changed_pct` too high (shape lost, thin parts gone): lower Detail (6 to 3, then 1.5). Faces go up.
- Too many faces for the user: raise Detail (6 to 10), or switch to `LOW_POLY`, or use `--reduce-to`.
- Defect kinds in `issues.totals` (inward, folded, crossing, sunk): run the `curiomesh-fix-bad-faces` skill on the output. If one whole part is bad, rerun with `--keep-original-bad`.
- Organic model with ugly flow: try `--use ANIMATION` or `SCULPT`.
- Run fails or hangs on a huge mesh: set `CURIOMESH_HELPERS=2`, or decimate nothing and ask the user.

## 7. Report back

Tell the user: output path, faces before and after, quads %, view changed %, the issue totals, and what you changed on any retry. Say plainly if it still has bad faces.

## Honest limits

Quads can look slanted on smooth surfaces. Fine ornaments go jagged at high reduction. It does not place edge loops for joints on its own. Baking (`--bake normal`, `--bake-size 2048`) runs Cycles inside Blender and can take minutes.
