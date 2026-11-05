---
name: curiomesh-fix-bad-faces
description: Find and fix bad faces (turned inward, folded, crossing, sunk, flat slivers, duplicates) in a CurioMesh retopo result, in a measure, fix, re-measure loop that never makes the mesh worse. Use when a retopologised mesh has shading errors, black or inside-out faces, folds, or when the curiomesh-retopo report lists defects.
---

# Fix bad faces (CurioMesh)

Uses `curiomesh_fix.py` and `curiomesh_run.py` from the sibling skill `curiomesh-retopo` (its `scripts/` folder, next to this skill's folder). Find Blender and the add-on the way that skill says (`$BLENDER`, PATH, common install paths; `--addon <repo folder>` for a checkout).

## What the kinds mean

| Kind | Meaning | First fix |
|---|---|---|
| `inward` | face points into the model | Flip Inward, then Untangle |
| `sunk` | face sits under your original surface | Untangle |
| `folded_over`, `crossing` | faces fold or cut through each other | Untangle, then Patch |
| `flat`, `duplicate` | zero-area or doubled faces | Clean Up |
| `bent` | badly warped quad | Relax, then Patch |
| `overlap` | face lying on another | Clean Up, then Patch |
| `sliver`, `pinched`, `pole`, `concave` | untidy, not wrong | leave alone unless the user asks |

## The loop

Use `.obj` or `.blend` for the result you fix (retopo with `--out result.obj`). A `.glb` stores triangles only, so every quad comes back as two triangles and the counts no longer match the retopo report.

1. Measure only (no `--fix`). The original is needed so "inward" is known:
   ```
   "$BLENDER" -b --factory-startup --python <retopo skill>/scripts/curiomesh_fix.py -- RESULT.obj --original ORIGINAL.obj --report m0.json
   ```
   `before.bad` is the count of bad faces, `before` has the count per kind, `bad_face_ids` lists them (capped at 2000).
2. If `before.bad` is 0, stop: nothing to fix.
3. Fix. Start with `--fix ALL` (Clean Up, Flip Inward, Untangle, Relax, Patch in turn; a move that makes it worse is undone). For one kind, name the fix from the table:
   ```
   ... curiomesh_fix.py -- RESULT.obj --original ORIGINAL.obj --fix FLIP,UNTANGLE --out fixed1.obj --report m1.json
   ```
   To touch only some faces: `--faces 12,40-55` (numbers from `bad_face_ids`).
4. Read `m1.json`: `before`, `after`, `steps` (each with `kept`, `bad_before`, `bad_after`). The script keeps a step only when it lowers `bad`, so `after.bad` is never above `before.bad`.
5. Loop: take `fixed1.obj` as the new RESULT and go to step 3 with a different fix list (for example what is left: if folds remain, `UNTANGLE,PATCH`). Stop when:
   - `after.bad` is 0, or
   - a round does not lower `bad` (`improved` is false), or
   - you have done 4 rounds.
6. Keep the best file. Never overwrite the original or the last good result; write each round to a new file and compare `after.bad`.

## When fixes do not help

- Defects pile up in one part (look at `issues.per_part` in the retopo report): rerun the retopo with `--keep-original-bad`. That puts the original faces back on every part with a problem spot. The report's `issues` and `kept_original_parts` show the effect. The face count rises.
- Many defects everywhere: the Detail is too high for this model. Rerun `curiomesh-retopo` with a lower Detail.
- `--fix` steps all show `kept: false` and bad faces remain: report the count and kinds to the user and stop. Do not loop.
- No `--original`: Flip Inward and Untangle cannot run (the report says so). Ask for the original file.

## Report back

Give before and after `bad` and the per-kind counts, which steps were kept, the output file, and what is still bad.
