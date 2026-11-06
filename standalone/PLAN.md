# CurioMesh standalone: plan

A `curiomesh` program that does CurioMesh's automatic retopology without Blender, and checks the quality of any
retopology (CurioMesh's, another tool's, or done by hand) against its source.

## Why it is worth making

Free standalone auto-retopology already exists: AutoRemesher (MIT, 2026, has a CLI, OBJ only) and Instant Meshes.
They cover a surface with even quads and ask for a face count. This tool offers two things they don't:

1. **You set a quality, not a face count.** `--detail 6` means at most 6% of the view may change. It keeps the
   fewest faces that hold that, measured with the camera. It works by reducing the model, so hard edges stay and flat
   areas become big quads. That suits hard-surface models and multi-part assemblies.
2. **`curiomesh measure`** scores any retopology against its source: view change, bad faces and ranked issue spots,
   as JSON, with an exit code for build pipelines (`--max-loss`, `--max-bad`).

Release gate: before a public download, it must beat AutoRemesher on hard-surface test meshes at the same face count.

## Legal

- CurioMesh is the owner's own GPL-3.0-or-later project; this program is GPL-3.0-or-later too.
- No Blender source is copied into this crate. It compiles the add-on's own Rust kernels from `native/src`.
  The one exception: `native/src/near.rs` contains `triangles_cross`, a credited port of Blender's
  `isect_tri_tri_v3_ex` (GPL-2.0-or-later, which allows GPL-3.0-or-later). That is legal here. It means this program
  can't be relicensed as MIT or CC0 while it includes that file, and the release notice must keep the Blender
  credit.
- The Blender features the add-on uses get these replacements:

  | Blender feature | In this program |
  |---|---|
  | Decimate | `qem.rs`: own Garland-Heckbert edge-collapse reducer (published 1997 method) |
  | BVH ray casting (camera pictures) | `geom.rs` (CPU depth buffers) and `gpu.rs` (wgpu) |
  | KDTree / BVH nearest | `near.rs` own trees |
  | QuadriFlow, Remesh, UV unwrap | not used yet. The Animation and Sculpt presets need an own field remesher; UVs stay out |
- Dependency licences (the `cargo metadata` lock for Windows, 108 crates): MIT, Apache-2.0, MIT-or-Apache,
  BSD-2/3-Clause, ISC, Zlib, Unlicense, 0BSD and Unicode-3.0. All are GPL-3.0-compatible, with no copyleft-incompatible
  or unlicensed crates. New compared with the add-on: tobj (MIT), gltf (MIT OR Apache-2.0), serde_json, and the image
  decoders gltf's `import` pulls in (image, png, zune-jpeg: MIT/Apache/Zlib). Before a release, add these to a
  THIRD_PARTY_NOTICES for the binary.
- Trust (owner's rule), as of 2026-11-04: tobj 3.2M downloads, 275 GitHub stars, active; gltf 11.4M downloads, 645
  stars; serde_json 1.5B downloads. ply-rs is unmaintained (2020) and its fork has 8 stars, so PLY is read by own code.

## What the add-on does, and where it lives

| Step (add-on, `autoretopo.AutoJob._run`) | Add-on | Standalone now | To do |
|---|---|---|---|
| Read the mesh | Blender | `io.rs`: OBJ, glTF/GLB (with node transforms), PLY ASCII and binary; welds split vertices | FBX (no well-trusted crate: suggest glTF) |
| Split into loose pieces, segment big parts | Python `_split_pieces`, `triadq/segment.py` (+ native `segment_merges`) | whole object at once | port `segment.py` |
| Size each part by camera loss (`_measure_piece`, `budget.allocate`) | Python + Decimate + native pictures | one global QEM order, search on one count | per part: QEM levels per piece + `budget.allocate` (the biggest gap) |
| Reduce | Blender Decimate, or native `qem` for big parts | native `qem` | — |
| Pair triangles into quads | Python `engine._pair_candidates` + native `pair_seeds` | ported scores + native `pairing` | port the smoothed cross field (`engine._estimate_field`); try 2 seeds and keep the best like `_run_candidate` |
| Smooth grids, Balanced split, flat merging | Python `smoothgrid`, `flow`, `flatquad` | — | port (`smooth_grids` matters for curved parts) |
| Close to all quads | native `make_closable` + `close_pieces` | same native code | picks up the v093-border closing on merge |
| Copies of a part (twins) | Python `special` | — | port: repeated bolts are made once |
| Fit to source | Python `fit.py` | — | port (uses `near.rs` trees) |
| Spin edges at poles, Tidy | Python `ChordReducer`, `refill.polish` | — | port, or `chords.rs` |
| Repair / untangle (`_check`, `_untangle`, `repair.py`) | Python + native checks | — | port; needs `triangles_cross` (already in `near.rs`) |
| Defect check | `checks.py` (+ native `mesh_checks`) | broken edges, flipped, duplicate, bent/concave quads | crossings (`near.rs` has them), inward/sunk |
| Issue Finder spots | `triadq/issues.py` | spots of flagged faces that touch, scored by the same weights | the distance join (`JOIN_EDGES`), pinched faces, `compare` of two runs |
| Redo Rough Spots | `triadq/redo.py` | — | after repair |
| Look at the result (loss) | native pictures, 192 px | same: GPU, falls back to CPU | shading loss (`shaded`, 20 and 10 degrees) |
| GUI | Blender panel | — | egui/eframe (32k stars) with a wireframe preview and the spot list, once the CLI holds up |

## First comparison (2026-11-04)

These are three free bench models. The add-on's results are its 0.9.2 Game Asset runs from 2026-11-03 in Blender 5.2.
Every view change below was measured by `curiomesh measure` with the same camera (192 px, GPU), so the columns compare
fairly. The add-on's own log reported less change for its results (2.47%, 5.58% and 6.12%), so the two measures still
differ by 15-35%. Find out why before quoting numbers publicly.

| Model (source faces) | Add-on 0.9.2 | Standalone, `--use game-asset` | Standalone, `--reduce-to` the add-on's count |
|---|---|---|---|
| chess set (76,832) | 23,834 quads, 3.36%, 0 bad, 377 spots, 103 s | 5,757 quads, 5.89%, 0 bad, 67 spots, 1.3 s | 25,884 quads, 1.85%, 0 bad, 447 spots, 1.7 s |
| drill press (58,511) | 24,031 quads, 6.29%, 28 bad, 360 spots, 229 s | 20,080 quads, 5.82%, 0 bad, 310 spots, 1.8 s | 24,740 quads, 3.68%, 0 bad, 372 spots, 1.7 s |
| fire hydrant (86,316) | 37,552 quads, 7.16%, 0 bad, 791 spots, 960 s | 25,134 quads, 5.43%, 0 bad, 482 spots, 1.7 s | 39,391 quads, 2.82%, 0 bad, 750 spots, 1.7 s |

Read with care:
- In view change and time, the standalone does better than the add-on here. In layout it is behind: it has no smooth
  grids, fitting, pole spinning or tidying, and the camera can't see edge flow. Its quads still need to be looked at
  next to the add-on's.
- The add-on's times include Blender's per-part looks and its repair steps, and come from a separate bench run. The
  standalone's times are on this PC with the GPU.
- "Bad" here counts broken-edge faces and quads that `quad_valid` rejects (bent past 60 degrees or concave). The add-on's
  own folded count for the drill press was 0, so the two definitions differ.
- `--reduce-to` lands within 3-9% of the count asked for, in up to 4 looks.

### Against AutoRemesher 1.2.0 (release gate)

AutoRemesher 1.2.0 for Windows came from the official GitHub release (huxingyi/autoremesher: 3.6k stars, MIT).
VirusTotal: 0 of 66 vendors flagged it. It ran from its command line with default settings and `--target-quads`. It
undershoots the target, so it ran twice: at the add-on's count, then with the target scaled up to land near it. The
standalone ran with `--reduce-to` AutoRemesher's first count. All results were measured by `curiomesh measure`.

| Model | AutoRemesher, 1st | AutoRemesher, scaled target | Standalone at AutoRemesher's 1st count |
|---|---|---|---|
| chess set | 8,974 faces, 8.34%, 113 bad, 35 spots, 5 s | 21,338 faces, 4.85%, 71 bad, 43 spots, 16 s | 9,462 faces, 4.58%, 0 bad, 152 spots, 2 s |
| drill press | 15,455 faces, 32.80%, 1,219 bad, 326 spots, 10 s | 21,967 faces, 26.77%, 1,620 bad, 393 spots, 12 s | 16,688 faces, 8.12%, 34 bad, 260 spots, 2 s |
| fire hydrant | 17,335 faces, 20.64%, 462 bad, 127 spots, 7 s | 35,084 faces, 14.60%, 601 bad, 175 spots, 7 s | 18,897 faces, 9.69%, 0 bad, 418 spots, 2 s |

- **Shape and defects:** the standalone wins on all three. On the hard-surface drill press it keeps 4x more of the view
  than AutoRemesher at the same count, and even with AutoRemesher given 30% more faces. It also leaves 0-34 bad faces
  where AutoRemesher leaves hundreds. AutoRemesher also leaves a few triangles. That fits the angle: hard-surface parts
  and assemblies.
- **Layout:** AutoRemesher wins. Its even field quads have far fewer issue spots on the chess set and the hydrant (fewer
  poles and slivers). The standalone's layout is the next thing to improve: smooth grids, pole spinning, the cross field.
- The standalone landed 5-9% above the count asked for, so its columns have slightly more faces.
- AutoRemesher was not tuned (sharp edge 90 degrees, adaptivity 1). Tuning could narrow the shape gap. This was not tried.

### Layout: spinning edges and relaxing (`layout.rs`)

After closing, the standalone now spins edges at poles (edge rotation, as the add-on's `turn_poles` does) and slides
corners on smooth ground toward their neighbours' middle, snapping them back onto the source. Each step is kept only
if the view holds. On all three models both steps were kept. The view change went from 4.58 to 4.22% on the chess
set, 8.12 to 8.37% on the drill press and 9.69 to 9.53% on the fire hydrant.

Layout measured on every result. "Irregular" is the share of inner corners without 4 edges. "Squareness" is the
mean minimum scaled Jacobian of the quads: 1 is a square, 0 is folded.

| Model | Add-on 0.9.2 | AutoRemesher (scaled target) | Standalone before | Standalone with layout step |
|---|---|---|---|---|
| chess set | 0.445 irregular, 0.036 slivers, 0.656 square | 0.020, 0.077, 0.964 | 0.416, 0.033, 0.669 | 0.388, 0.031, 0.679 |
| drill press | 0.397, 0.203, 0.617 | 0.095, 0.088, 0.827 | 0.434, 0.240, 0.533 | 0.431, 0.219, 0.548 |
| fire hydrant | 0.411, 0.160, 0.634 | 0.017, 0.041, 0.947 | 0.423, 0.211, 0.539 | 0.405, 0.196, 0.547 |

- The layout step helps a little: up to 7% fewer irregular corners and 7-10% fewer slivers.
- The add-on is just as irregular (40-45%), so this gap comes from the method, not the port. Quads made by pairing
  a reduced triangle mesh keep the triangles' poles. More relaxing doesn't fix that: 10 passes gave the same numbers
  as 3.
- AutoRemesher's even layout comes from a field-aligned remesher. The way to get close without losing the shape lead is
  a hybrid: field-aligned rows on smooth, curved parts (the add-on's Field Rows, `fieldmesh.py`; its sweeps are already
  native in `field.rs`, the Instant Meshes method) and reduction kept on hard and flat parts. That needs the per-part
  segmenting first. It's the biggest piece of work left.

### Hybrid, first version (`rows.rs`): built, guarded, not yet winning

Each loose part with at least 1,000 source triangles is offered even rows of quads at about the faces it has after
reduction, biggest part first, within 60 s. The rows are built like this:

- A direction field and a grid are grown coarse to fine with the add-on's native sweeps.
- Cells shrink to half where the surface turns fast.
- Creases sharper than 60 degrees, and borders, are pinned.
- Points sharing a grid corner become one corner.
- Corners one cell apart are linked and the faces walked (Instant Meshes' extraction). Small holes are filled.
- Every face is split into quads (their pure-quad mode), and the quads are snapped back onto the source.

A part keeps its rows only if the view holds and the rows add no bad faces. `--no-rows` turns the step off.

- **On a clean surface it works.** A sphere comes out closed and all quads, with 9-12% irregular corners against
  roughly 40% for pairing, in under a second.
- **On the bench models it is refused almost everywhere.** The rows of nearly every part have 2-54 quads bent past 60
  degrees at tight curves, necks and rims. Every bad face is a bent quad; none are broken edges. The guard therefore
  refuses them: 33 of 35 parts on the chess set, all 10 tried on the drill press, 16 of 17 on the hydrant. Results are
  unchanged: same view change, 0 bad faces, layout as before. Only one hydrant part took rows.
- Small parts also come out with too few corners: 44-79 where 99 were expected on one chess piece. Pinning creases
  makes this worse (44 with it, 79 without). Relaxing the rows on the source fixed almost none of the bent quads.
- What the add-on does that this doesn't: it cuts parts into limbs with their own cell size (`segment_limits`), pins
  only long creases (`CREASE_LENGTH`), tidies the corners after extraction (`_split_rings`, `_merge_small`,
  `_clean_regions`), fits every corner to the planes of the source near it (`fit_regions`), and untangles folds
  (`untangle`, `_even_out_folds`). Porting `fit_regions` and `untangle` comes first: those target the bent quads.

### Hybrid, second version: the add-on's finishing steps ported

After the split into quads, the rows now go through the add-on's last steps:
- `refit`: each corner gets the patch of source nearest to it and fits that patch's planes, with a guard against new
  bad quads.
- `untangle`, then `clean`: doublets dissolved, folded quads merged away.
- `_even_out_folds` at the end.

A part's rows that miss its face count by more than 10% are tried once more with a corrected cell. The try nearer the
count is kept, and rows more than 25% off are refused.

`--reduce-to` at AutoRemesher's counts, measured as before:

| Model | Before the hybrid | Hybrid now |
|---|---|---|
| chess set | 9,462 faces, 4.22%, 0 bad; irregular 0.388, 0.031 slivers, 0.679 square, 5.7% under 0.3 | 9,491 faces, 4.65%, 0 bad; 0.334, 0.022, 0.753, 3.5%; rows on 18 parts |
| drill press | 16,688 faces, 8.37%, 0 bad; 0.431, 0.219, 0.548 | 16,716 faces, 8.19%, 0 bad; 0.433, 0.217, 0.547; rows on 1 part |
| fire hydrant | 18,897 faces, 9.53%, 0 bad; 0.405, 0.196, 0.547 | 18,920 faces, 10.15%, 0 bad; 0.405, 0.194, 0.551; rows on 1 part |

- The ports cut bent quads per part from 2-54 to mostly 1-7. On the smooth, turned chess pieces most parts now take
  rows, and the layout gets clearly better at the same face count, for 0.4 points more view change.
- The machined drill press and hydrant keep the reduction almost everywhere: their parts' rows still have a few bent
  quads or miss the count. That is the hybrid doing what it should, rows where smooth and reduction where hard. Their
  layout stays where the add-on's is.
- It costs time: 35-56 s a model instead of about 4 s. Most of it goes on parts whose rows end up refused, within the
  60 s budget (the add-on's FIELD_SECONDS). A cheap first check that a part is smooth enough would save most of it.
- Still far from AutoRemesher's evenness on the chess set: 0.334 irregular against 0.045. The add-on's other
  extraction steps (`_split_rings`, `_merge_small`, `pair_along_rows`, `pair_by_paths`) and per-limb cell sizes would
  be next.

## Code layout

- `src/lib.rs` compiles `native/src/*.rs` with `#[path]`, so the add-on's kernels and this program run the same code
  and `native/` is unchanged. `geom.rs` names `numpy::Complex64`; Cargo.toml maps `numpy` to num-complex for that.
- `src/retopo.rs`: camera, pairing scores, pair and close, the search on detail or face count.
- `src/issues.rs`: checks and spots. `src/io.rs`: files. `src/main.rs`: the command line.
- `Cargo.lock` started from `native/Cargo.lock`, so shared crates have the same versions (and build on Rust 1.88).

### What the standalone can do that Blender held back (2026-11-04)

Three ideas the add-on can't afford inside Blender, each measured on the bench models:

- **Rows on all parts at once: kept.** The add-on tries rows one part at a time on a 60 s clock to keep Blender
  responsive. Building them in parallel gave the same results 2.5-8 times faster (chess set 35 s to 4.6 s, drill
  press 56 s to 18 s, hydrant 44 s to 17 s).
- **Eight pairing orders per look: dropped.** Identical results: the pairing's augmenting search already pairs as
  many triangles as can be, so every order leaves the same number.
- **Each part sized by its own camera loss (the add-on's Auto Segments allocation): dropped.** Parts were reduced
  in parallel to 8 sizes each, judged alone at the whole view's pixel size, and faces given by best gain per face,
  with the add-on's rounds. The one global reduction order did better on 5 of 6 runs. Game asset: chess set 5,717
  faces against 7,091 for the same 6.1%, drill press 20,080 against 28,111, hydrant 25,310 against 47,076. At a face
  count: drill press 8.2% against 10.2% change, hydrant 10.2% against 14.7%. Only the chess set at a count came out
  slightly ahead (4.37% at 9,851 faces against 4.65% at 9,491). The global quadric error compares every part on one
  scale, while small parts cover only a few pixels at the whole view's pixel size, so their camera numbers are coarse.
  This is also likely why the standalone loses less view than the add-on at the add-on's own face counts. Worth
  trying in the add-on: one global reduction order instead of sizes per part.

### Tidy Up (0.9.8's Organize) tried, not shipped (2026-11-06)

`organize.settle` was ported (plain smoothing and loop straightening, all its guards: pinned hard edges and borders,
no flipped, folded or caved-in quads, held on the source, at most half an edge length of drift) and run after the pole
spinning. Game Asset at strength 0, 0.5 and 1: chess set squareness 0.741 -> 0.744 with the view change 6.06% -> 6.78%,
drill press and hydrant within 0.004 squareness and 0.2 points, irregular corners unchanged on all three. Settling
moves corners but doesn't change which corners are poles; the add-on's measured Organize gains come from
`ChordReducer.redraw_all`, which redraws the patches round poles as plain grids. That redraw (chords.py, about 800
lines) is the piece to port for a Tidy Up that pays off.
