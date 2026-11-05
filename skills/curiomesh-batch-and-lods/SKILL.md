---
name: curiomesh-batch-and-lods
description: Retopologise many model files in one go, make LOD sets (_LOD0, _LOD1, _LOD2) for a game engine, and bake a normal map with CurioMesh in Blender, headless. Use when the user has a folder of meshes to reduce, wants LODs, or wants detail baked to a normal map.
---

# Batch, LODs and baking (CurioMesh)

Uses `curiomesh_run.py` from the sibling skill `curiomesh-retopo` (its `scripts/` folder). Find Blender and the add-on as that skill says. Run one Blender at a time: the user's PC stays usable that way, and CurioMesh already uses helper processes.

## Many files

`curiomesh_run.py` takes one file per run. Loop in your shell, one after the other, never in parallel:

```
for f in models/*.obj; do
  "$BLENDER" -b --factory-startup --python <retopo skill>/scripts/curiomesh_run.py -- "$f" --use GAME --detail 6 --out "out/$(basename "${f%.*}").glb" --report "out/$(basename "${f%.*}").json"
done
```

PowerShell: `Get-ChildItem models\*.obj | ForEach-Object { & $env:BLENDER -b --factory-startup --python ... }`.

Then read every JSON, and give the user one table: file, source faces, faces, quad %, view changed %, defect totals, `verdict.ok`. Retry only the files whose `verdict.ok` is false, following the retry steps of `curiomesh-retopo`. A file that fails (`ok: false`) does not stop the others.

CurioMesh's own `Batch Remesh Selected` button belongs to the older whole-mesh remesher, not Auto Segments. Do not use it here.

## LOD sets

```
"$BLENDER" -b --factory-startup --python <retopo skill>/scripts/curiomesh_run.py -- model.obj --use GAME --lods 50,25,12.5 --out out/model.glb --report out/lods.json
```

This writes `out/model_LOD0.glb`, `_LOD1`, `_LOD2` (each a share of the source faces; 50, 25 and 12.5 are the defaults). Only the first runs the full search. The report lists, per level, `faces`, `quads`, and `view_changed_pct`. The view change grows with each level: judge each level on its own. LOD2 at 12.5% of a small model can lose thin parts; if `view_changed_pct` is above about 15, raise that level's percent.

## Baking

Add `--bake normal` (keep materials, add a normal map) or `--bake normal_color` (new UVs and one baked material), and `--bake-size 1024|2048|4096`. It bakes the original's detail onto the result using Cycles on the CPU, so it can take minutes. Export as `.glb` or `.fbx` so textures travel with the file; check that the output folder holds the images. A failed bake leaves the quads as they are and the report notes say "Bake failed".

Low-poly game models: bake, and have the engine triangulate on import.
