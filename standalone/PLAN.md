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
- Not compared yet: AutoRemesher (3.6k stars, MIT, active). It is needed for the release gate.

## Code layout

- `src/lib.rs` compiles `native/src/*.rs` with `#[path]`, so the add-on's kernels and this program run the same code
  and `native/` is unchanged. `geom.rs` names `numpy::Complex64`; Cargo.toml maps `numpy` to num-complex for that.
- `src/retopo.rs`: camera, pairing scores, pair and close, the search on detail or face count.
- `src/issues.rs`: checks and spots. `src/io.rs`: files. `src/main.rs`: the command line.
- `Cargo.lock` started from `native/Cargo.lock`, so shared crates have the same versions (and build on Rust 1.88).
