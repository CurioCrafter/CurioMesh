# Parts exchange (format version 1)

Parts exchange is for the case where CurioMesh made most of a model well, but a few parts are bad. You (or your
agent) mark those parts, CurioMesh writes them to a folder with their borders, someone or something makes better
topology for them, and CurioMesh stitches the new parts back in, checks them and keeps the whole swap as one undoable
History step.

The format is plain OBJ plus JSON so that anything can read it: an agent's Python, a script, another 3D program.
Agent commands: `parts.export`, `parts.import`, `parts.report` (see [AGENTS.md](AGENTS.md)). Panel buttons: *Send to
Agent*, *Bring Back*, *Save for a Report*. GitHub add-on only.

## The folder

```
<name>-parts-<yyyymmdd-hhmmss>/
  parts.json                 the manifest
  README.txt                 what to do, in five lines
  part_<id>.obj              the part's current topology (local numbering, quads kept, n-gons allowed)
  part_<id>_source.obj       the original faces for this part: the surface to follow
  part_<id>_around.obj       one ring of neighbouring result faces, read only, to match the flow at the border
  returned/part_<id>.obj     what you write back (same rules as part_<id>.obj)
```

`parts.json`:

```json
{
  "format": "curiomesh-parts",
  "version": [1, 0],
  "made_by": "CurioMesh 0.9.9",
  "object": "Statue",
  "result": {"source_faces": 120000, "source_verts": 60210, "result_faces": 9800, "result_key": "3f9a1c0e"},
  "units": "metres, object space, Z up",
  "rules": "Keep every border vertex exactly where it is. Faces 3 to 64 corners. Quads preferred.",
  "parts": [
    {
      "id": 7, "name": "Part 8", "kind": "panel",
      "faces": 412, "view_change_pct": 6.1, "tidiness": 71.0, "issues": ["folded_over", "pole_cluster"],
      "border": [[0, 3, 9, 14, 0]],
      "border_points": [[0.120000001, 1.5, 0.0]],
      "materials": [0, 2]
    }
  ],
  "note": "free text typed when exporting"
}
```

(`border_points` holds one point per border vertex; one is shown to keep this short.)

- **Numbers.** Coordinates have 9 significant digits, enough to round-trip a 32-bit float exactly. On import,
  border points are matched within 1e-6 of the object's size and then snapped to the stored values. An editor that
  rewrites numbers a little still stitches.
- **`border`.** One closed loop per border, as local vertex numbers in `part_<id>.obj`; `border_points` has the same
  points in loop order. A returned part may renumber freely: its border is found by position, not by number.
- **Materials.** Written as `usemtl m<index>` lines. A returned file with none gets the part's first material.
- **`result_key`.** A short hash of the result that was on screen when the parts were exported (face count plus a
  hash of the border points). If the result has changed since, import answers `MISMATCH` and says which History step
  the files were made from. Jump back there (`history.jump`) or export again.
- **Versions.** `version` is `[major, minor]`. A reader takes any file with the same major and ignores keys it does
  not know. A new minor only adds keys. A new major means an old reader answers `VERSION`, naming the CurioMesh
  version that reads it.
- **What is exported.** The marked parts, or the `parts` you pass. Several at once is normal. A part whose border
  touches itself at one point is left out, with the reason in the reply and under `"skipped"` in `parts.json`.

`parts.export {parts?, folder?, note?}` answers `{folder, parts: [ids], skipped: [{id, why}]}`. Export changes
nothing, so it works even when redo is off for the result.

## What you write back

For each part you fix, put one OBJ in `returned/` named `part_<id>.obj`.

1. Keep every border vertex exactly where `parts.json` puts it. Only the inside may change.
2. Faces have 3 to 64 corners and no repeated corner. Quads are best.
3. Follow `part_<id>_source.obj`; match the flow of `part_<id>_around.obj` at the border.
4. Leave out parts you did not fix. The others are brought in on their own.

## What CurioMesh does when you bring them back

`parts.import {folder, look_first?}`, in this order:

1. **Reads and checks the files.** Format and version; every number finite; indices in range; faces of 3 to 64
   corners; the part ids exist; `result_key` matches. These files come from outside, so nothing is trusted.
2. **Border.** Each exported border point must be present once, and the loops must visit the points in the same
   order (either direction). No other vertex may lie on the border. Otherwise: `BORDER_MOVED`, with the loop, the
   point and how far off it is.
3. **No new holes, no lost shape.** The part is spliced into a copy with the borders welded. More open edges than
   before is `HOLE`; the part no longer following the source is `SHAPE_LOST`.
4. **Camera check, within Most Detail Loss.** The same pictures as a part redo. The whole object's change must stay
   within Most Detail Loss when that guard is on, and the part must keep the view as well as the one it replaces.
   Otherwise `VIEW_CHANGED`, with both numbers and the hint that Most Detail Loss can be raised. The guard is the
   user's: it is reported, not hidden.
5. **Kept as a try.** Each passing part becomes a new try with the method name "From Outside", next to the old
   result, so the old one is one click away. Bad spots in the new faces are counted and reported but do not block it.
6. **Look First, then keep.** In the panel a preview shows all returned parts at once with a readout card (faces,
   view change, tidiness, bad spots, before and after). Enter keeps, Esc cancels, Before/After works. From an agent,
   `look_first: true` opens the same preview in a live window; otherwise the import is kept at once. Keeping is one
   History entry for all returned parts together.
7. **Parts that fail.** A part that fails steps 2 to 4 is left out; the others still come in. The reply lists every
   part as `kept` or `refused` with its error code.

Undo: Ctrl+Z once takes all the returned parts out again, and `history.jump` reaches any step. The tries live in
memory like all others; after a save and reload only the applied result survives, but the returned folder is still on
disk and importing it again rebuilds them.

The result must allow part redos. If it does not (Mirror, Even Rows, an imported job, a cut piece), import answers
`NOT_READY` and names the reason.

## Worked example: a grid with a fold

A model has a flat panel, "Part 4", exported as a 4 by 4 grid of quads (25 vertices). Its border is the 16 vertices
around the outside. A fold in the middle makes four faces point backwards.

`part_3.obj` (part id 3), shortened:

```
# vertices 0 to 15 are the border, in loop order; 16 to 24 are inside
v 0.0 0.0 0.0
v 0.25 0.0 0.0
...
v 0.5 0.5 0.05      <- inside vertex 20, pushed out of line: this is the fold
...
f 1 2 18 17
f 2 3 19 18
...
```

`parts.json` lists `"id": 3`, `"issues": ["folded_over"]`, `"border": [[0,1,2,...,15,0]]` and the 16 border points.

The agent keeps the 16 border points exactly and moves the inside vertices back onto the flat panel (they can also
be re-laid however it likes, as long as faces stay 3 to 64 corners). It writes `returned/part_3.obj`.

Then:

```
parts.import {"folder": "Statue-parts-20261107-101500"}
-> ok, parts: [{"id": 3, "status": "kept", "view_change_pct": 5.9, "bad_spots": 0}], step: 15
history.list         -> step 15 "From Outside (1 part)"
```

If the agent had nudged border vertex 6 by 4 mm instead, the reply would be:

```
{"ok": false, "error": {"code": "BORDER_MOVED", "message": "Part 4: border point 6 of loop 1 moved 0.004 m",
  "hint": "Keep every border vertex where parts.json puts it; only the inside may change"}, "step": 14}
```

and `step` stays 14: nothing changed.

## Saving parts for a report

`parts.report {parts?, path?, note?}` (button: *Save for a Report*) writes one zip next to the .blend, or at the path
you give. It holds the part folder above, a `report.md` (CurioMesh, Blender and OS versions; the Use and changed
settings; the part numbers; the issue list for those parts; your note) and `settings.json`. CurioMesh does not send
it anywhere. The zip contains pieces of your mesh, so check it before you attach it to an issue.
