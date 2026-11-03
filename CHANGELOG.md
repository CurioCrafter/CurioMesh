# Changelog

## 0.9.1 - unreleased

### Graphics card
- **The camera pictures of big meshes (150,000 triangles and up) are drawn on the graphics card** when the computer has one that works: NVIDIA, AMD, Intel or Apple, through Vulkan, DirectX 12 or Metal, with no CUDA and nothing to install. On an RTX 3080 Ti that is 1.1 to 1.4 times faster at 170,000 triangles and about 3 times at 1.3 million; smaller meshes stay on the processor, where the card is level or slower. A whole run is only 1 to 2% quicker, since pictures are a small part of a run.
- **Almost the same pictures:** the card works in single precision, so on a depth tie a pixel can pick the neighbouring triangle. On real meshes the camera counts 0.02 to 0.4% of the pixels as changed at 128 to 192 pixels a side, against the processor's pictures.
- **Use Graphics Card** in the PC section turns it off; so does `CURIOMESH_GPU=0`. If the card errors, the processor takes over for the rest of the session and the summary says so once.
- The native core grows from 1.7 MB to 7.2 MB on Windows (wgpu and its graphics-API bindings).

## 0.9.0 - 2026-11-02

### Auto Segments: fewer faces, judged by a camera
- **About 20-55% fewer quads at the same Detail.** Leftover triangles join in twos along short paths of
  quads, then quads are taken away cheapest first until the camera says the amount asked for is used up.
  No connected part may pass its own limit. Big parts are also tried reduced in small steps and at sizes
  between the old ones, so they no longer land far under the amount.
- **Use menu:** Low Poly, Game Asset (the old default), Animation, Subdivision, Scan Cleanup, Sculpt Base,
  Custom. Each sets **Detail**, **Method** and **Output**. Detail runs from 0.1% to 80%, with a plain-words
  note of what it means.
- **Methods:** Auto, Pairing, Flat Quads, QuadriFlow, Grid Fill, Flow Fill, Field Rows (opt-in; Auto offers
  it on flat, thin and box parts) and Tris Only. **Outputs:** Triangles, Quads, Mixed. When a method can't do
  a part, Auto is used there and the summary says so.
- **Reduce To** a percent or a face count, beside Keep Detail; it lands within about 10%.
- **Styles:** Keep Detail, Balanced, and Clean Rows (even rows, all quads).
- **Turned parts** (pipes, tanks, bolts, rings) become rings of quads with capped ends. **Round and
  four-cornered areas** become concentric rows and grids, wherever the camera says they hold up.
- **Part cases:** each part is named (tiny, flat plate, thin sheet, box, tube, dome, rough scan, ornament,
  organic), and a small measured table skips the sizes that case never holds.
  Idea from [cav3dw3ll3r](https://github.com/cav3dw3ll3r).
- **Fit and shading:** finished vertices are fitted to the source part by part (2-10% less visible change
  at the same face count). Apply gives each face corner its normal from the source points on its own side
  of a hard edge, and marks edges past 60 degrees sharp.
  Idea from [cav3dw3ll3r](https://github.com/cav3dw3ll3r).
- **Tidiness score** from 0 to 100 at the end: how close the layout is to hand-made meshes (sources in
  `docs/PATTERN_SOURCES.md`). Read it next to the camera number. **Tidy Quads** is an option, off by default.
- **Redraw Messy Quads:** messy parts are redrawn as pole-free grids, and quads are taken where the camera
  sees least change. It is on for Animation and Subdivision, and left off on models of more than 40 loose
  pieces. Tidiness roughly doubles (bust 28 to 62) for 7-50% more faces and 2-5x the time, capped at 240 s.
  The summary says when it ran and what it took.
- **Multires Detail:** a clean low cage keeps its faces at level 0, and your mesh's shape comes back as
  Multires levels. It is Animation's default route (QuadriFlow cage plus Multires), offered for Subdivision
  and Sculpt Base, and available for any cage with **Add Multires Detail**.

### One guided panel
- **Five steps, one main button:** pick the mesh, choose what it's for, Start, fix parts if you want, Apply.
  The **Auto** button picks the Use and settings from the mesh and free memory, says why, and starts.
  Nothing changes until Apply, and Step 5 says what Apply wrote.
- **Progress** in words, with a time range and a Stop that ends the work within a fraction of a second.
- **Summary:** "2,508 faces, 86% fewer / Looks 94% the same / Tidiness 27/100". The parts, the hover card
  and the tries use the same words.
- **Classic Remesh:** the old one-pass remesher (Quick Start) is kept, closed by default under Guided, with
  no second Auto button.
- **Heat** colours are viridis (colour-blind safe), with a legend in the viewport.
- **Panel layout:** advanced options are folded away; nothing is clipped in a narrow sidebar; tooltips are
  short. The add-on is named plain **CurioMesh** (id `curiomesh`).

### Reviewing and fixing parts
- **Review Segments:** hover a part for its numbers, click to select, and walk the parts worst first. Give a
  part another Use, Method or Detail and only it is redone. Every try is kept, to go back to.
- **Segments inside a connected mesh:** big pieces split along creases, necks and steps (a bust into head,
  torso and plinth rings). You can merge, split with a line, paint, rename; this is saved with the object.
  Seams stay put, so segments join with no gap or T-junction.
- **Live Boolean:** put boxes, spheres or any closed mesh over the result. Union, Subtract or Intersect
  picks the faces to redo, live, as you move them.
- **Show It How:** fix a spot by hand and **Remember This Fix**. **Look for Known Fixes** finds the same
  pattern elsewhere and applies it after a camera check. Fix libraries are size-capped and validated.

### Guides, loops, paint and mirror
- **Guide Curves:** draw strokes or add curves, Grease Pencil or edge meshes, and quads follow them.
- **Draw Edge Loops:** closed loops and flow lines become strips of regular quads, with poles kept outside.
- **Deformation Lines:** a line across a joint becomes evenly spaced loops, with optional Crease and Sharp
  marks. It helps tight joints; on broad bends it makes little difference.
- **Paint Density:** Weight Paint the parts that need more quads.
- **Mirror** (Auto, X, Y, Z): one half is solved and mirrored; the seam sits exactly on the plane.

### Game assets
- **Make LOD Set:** `_LOD0`, `_LOD1` and `_LOD2` at percents you choose (50, 25 and 12.5% by default).
  Only the first does the full search.
- **Bake Detail:** Normal Map or Normal + Color, 512 to 4096 pixels. Normal + Color is the recommended
  route. Glass and other see-through materials are kept as they are.
- **Keep UV Seams** is an option, off by default: it can nearly double the faces.
- **Keep Copies as Instances:** repeated parts are written once and shared.

### Big scans and speed
- **Native Turbo core** (Rust, all cores, no GPU needed): the camera pictures, pairing, closing, reduction,
  crossing checks, lookups, fitting and shading. Results match the Python path. A 1.26M-face scan went
  from 47 to about 5 minutes, and Blender stays usable.
- **Effort** (Light, Normal, Full), **This Computer** check, and big pieces cut into regions and saved to
  disk, so a stopped run resumes.
- **Export Job / Import Job Result:** run a job on another machine with Blender only (guides, paint and
  Mirror included). See `docs/CLOUD_WORKFLOW.md`.
- **Rough As Texture** (rough scans judged on average; coast rocks 434,997 to 56,585 quads) and **Leave
  Out Hidden Pieces**, both off by default. **Judge After Baking** is experimental.

### Checks and repair
- After every run the result is checked against your mesh for non-manifold edges and vertices, flipped
  edges, duplicates, holes, faces passing through each other (any size of mesh), and folded quads. Only
  what the result adds counts.
- New problems are cleared on the spot or the part is remade one size up. Clearing crossings stops at
  30 s. What is left is named in the summary and selected on Apply.

### Security and packaging
- Job results and saved parts are validated, and a .npz that unpacks past 1 GB is refused. Remembered fixes
  never use a shared temp file. A planted `nvidia-smi` in the current folder is never run.
- Releases are built only by the release workflow, from the tag, with `SHA256SUMS.txt`. The workflow has
  least privilege, with actions and installs pinned.
- `THIRD_PARTY_NOTICES.md` ships in every zip. A separate Python-only, CC0-only zip is built for
  extensions.blender.org.

### Known limits
- Some small thin parts come out with folded slivers: quads bent about 90 degrees round an edge (ship
  pinnace 40 of 59,900 quads, treasure chest 1). They are named in the summary and selected on Apply.
- Judge After Baking can leave long thin slivers on big flat areas of a scan. It is experimental, off by
  default.
- Draw Edge Loops' "Selected Faces Only" is switched off: the region's seam is not re-stitched yet.
- The optional QuadWild script needs QuadWild installed separately.

### Credits
- Ideas from [cav3dw3ll3r](https://github.com/cav3dw3ll3r): fitting each source point to its nearest vertex,
  shading from the nearest source points, and the part cases.
- Test models: free CC0 models from [Poly Haven](https://polyhaven.com). Tidiness statistics: see
  `docs/PATTERN_SOURCES.md` for sources, licenses and credits. The ported triangle test and Rust crates are
  credited in `THIRD_PARTY_NOTICES.md`.

## 0.8.0 - 2026-11-01

- **Auto Segments** (new mode in the Remesh setting of the Guided panel). Press
  Update with no points and no face count: it reduces each loose piece as far as
  a camera says it can, looking from six sides at the original and the result,
  then makes all quads. **Change Allowed** sets how much of the view may look
  different. Faces go to the parts where they remove the most visible change, and
  small parts are judged by how big they look on the object.
- One quad that can't be closed no longer makes a whole piece split into many more
  faces: it goes back to two triangles and the rest of the piece closes normally.
  This also lowers the face counts of Flat Quads and Guided on some meshes.
- **Work Live** (Guided panel). Paint areas on your mesh or on the result, each
  with its own density and fill style, and watch them remesh one at a time on a
  background thread while you keep working. Scroll sets the size, Shift+scroll
  the density, F the fill style. Only what changed is redone, and a change stops
  work that is out of date. New **Work Areas** mode in the Remesh setting.
- Region fills are faster: after each refinement round only the edges that
  changed are checked for flips, not all of them.

## 0.7.0 - 2026-10-30

- **Flat Quads** (new Method). Flat areas become as few quads as their outlines
  allow and density grades up only next to detail: a flat plate becomes one quad,
  a subdivided box six. Includes a **Flatness** setting. The target limits what's
  left after the flat areas, and very dirty scans can land above it.
- Flat Quads opens non-manifold edges, flipped normals and bow-ties into borders
  without moving geometry, and closes each loose piece on its own, so dirty meshes
  finish all quads, including large scans that failed in the other engines.
- Delaunay flips no longer create zero-area slivers along straight borders.
- **Guided Retopo** (new Guided panel). Click Detail, Loop, Sharp or Calm
  importance points on a mesh, see their reach and the combined outline as a
  live cage, and watch an adaptive quad preview that snaps to the surface update
  as you edit. Presets for characters, hard-surface props, scans and game LODs,
  Auto-suggest, lock faces, Redraw Strip, a Next spot hint, and sphere Empties
  as points.
- **Whole Mesh** remeshes everything with the points changing the quad size,
  all quads by default (Pure Quads in Advanced). **Points Only** rebuilds only
  the inside of the point spheres on a fixed seam.
- UV layers and materials carry over in both modes, including across UV seams
  and for materials linked to the object.
- Points Only regions that wrap a limb or surround an open hole are cut into
  disk-shaped pieces and all of them are rebuilt.

## 0.6.0 - 2026-10-29

- **Licence: CurioMesh is now GPL-3.0-or-later** (previously MIT, up to 0.5.1).
- **Turbo core in Rust** for topology checks, quad closing, ParityPath search,
  cross-field smoothing and cage inside tests, on all CPU cores, shipped for
  Windows, Linux and macOS in every zip with an automatic Python fallback
  (`CURIOMESH_TURBO=0` forces Python). Topology checks and quad closing are
  about 30x faster; a 50k-face all-quad remesh is 4.5x faster.
- **GitHub Actions** build the native core for every platform and attach all
  release zips; CI runs the tests on every push.
- **Region retopology.** Remesh only the faces you pick: an edit-mode
  selection, a face set, a vertex group, box cages, lattice cages or any closed
  mesh as a volume. Volumes combine with Union, Subtract and Intersect. The
  seam keeps its original vertices, so the patch joins with no T-junctions, and
  the inside is all quads. Preview with Auto-Update, then Apply (undoable) into
  the mesh or a copy.
- **Border Ring.** Jagged region borders get a ring of quads along the seam and
  an untangling pass. Bent or caved-in seam quads dropped from about 5% to 0.5%
  on a whale region and from about 31% to 2.8% on a noisy scan;
  region remeshing is about 5x faster.
- **Fill Loop** fills a closed hole with an all-quad patch that follows the
  surrounding curve. **Fit to Surface** relaxes and snaps vertices onto the
  original surface without folding quads.
- **Grid fill** for four-sided regions now follows the Quad Size setting.
- **Follow Curvature flow** (new Flow option for PairClosure and regions):
  quads also line up with the surface's main bending directions.
- **New UI.** Numbered steps, one main button per panel, a Last Result box,
  plain-language names and tooltips on every option, status words in the HUD
  so it does not rely on colour alone.
- **Preferences:** sidebar tab name, compact panels, show/hide each panel, HUD
  corner, offset, scale and opacity, and every heatmap and region colour.
- **A download for every Blender version:** separate zips for 3.6, 4.0, 4.1,
  4.2, 4.3, 4.4, 4.5, 5.0, 5.1 and 5.2, plus the two all-in-one zips, all tested
  on that Blender version. Every top-level module is now packed automatically.

## 0.5.1 - 2026-10-27

- Wider Blender support: the legacy add-on now installs and passes the test
  suites on Blender 3.6 LTS and newer; the extension package needs 4.2 LTS or
  newer. Verified on 3.6, 4.2, 4.5, 5.0, 5.1 and 5.2. Blender 3.0 and older
  are not supported.
- Release assets are named by Blender range:
  `curiomesh-<ver>-blender-3.6-plus-addon.zip` and
  `curiomesh-<ver>-blender-4.2-plus-extension.zip`.
- Tests no longer assume Python 3.11 (`tomllib`) or a specific EEVEE engine id.
- Tested on 13 real meshes (244 to 1.9M faces: scans, game assets, FBX/OBJ/GLB).
- Atlas-Q Auto now succeeds on real game assets that it used to reject
  (multi-shell, slightly dirty meshes): a few bad quads (3% on Balanced) and
  tiny loose shells no longer fail the gates, and a PairClosure result with
  only soft issues (high aspect, merged shells) is returned with a warning.
- Failures now say why: the message names the closest candidate and its reasons.
- Dirty or multi-shell meshes under Preserve topology, and sources over 200,000
  faces, go PairClosure-first. QuadriFlow is skipped on multi-shell meshes with
  open boundaries, where it could hang for minutes.
- UVs on sources over 300,000 faces are projected with a BVH instead of the
  Data Transfer modifier, which could stall for tens of minutes.
- Analyze uses the fast sampled path above 75,000 faces.
- Heatmap and HUD skip stats on meshes over 300,000 faces instead of slowing
  every redraw; the HUD hint now flags results with many poles.
- Live preview is limited to sources up to 8,000 faces so Blender stays responsive.

## 0.5.0 - 2026-10-27

- Quick Start panel with a big Remesh button, quality buttons and HUD /
  Heatmap / Live toggles; advanced settings moved to collapsed sub-panels.
- Viewport HUD dashboard: quad ratio bar, faces vs target, triangles, poles,
  engine, time and a next-step hint.
- Non-destructive triangle / n-gon / pole heatmap drawn with the GPU module.
- Live preview re-runs PairClosure into a preview object on setting or source
  changes (sources up to 20,000 faces).
- Batch remesh of all selected meshes with progress, cancel and a JSON report.
- TRIAD-Q engine vectorized with NumPy and per-mesh caching: about 100x faster
  quad-dominant and 29x faster all-quad on a 7k-triangle sphere.
- Maximum-cardinality triangle pairing (Edmonds' blossom) on top of the
  score-greedy start; `rescued_pairs` reports the augmentations.
- 4-RoSy cross field with frame transport, pinned to feature edges, drives
  pair alignment.
- README screenshots captured by `scripts/capture_readme_screenshots.py`.

## 0.4.0 hardening

- Normalize vertex-only bow-ties after QEM by deterministically duplicating
  disconnected incident fans without moving geometry or changing face-corner
  attributes; true edge-non-manifolds remain rejected.
- Make shrinkwrap projection transactional. Failed or gate-breaking projection
  restores the validated candidate and records the rollback reason.
- Add adversarial topology, standalone engine, and Blender projection rollback
  regressions grounded in multi-shell photogrammetry parts.

## 0.4.0 - 2026-07-16

- Added Atlas-Q scan analysis and solver portfolio.
- Added target-scale native QEM proxies and deterministic candidate ensembles.
- Added exact conforming PairClosure with UV-seam/material propagation.
- Added area/spatial/component-stratified symmetric RMS, p95, and maximum source
  error with preset-specific hard gates.
- Added vertex-fan manifold, true boundary-cycle, component, duplicate-face,
  winding, concave/crossed-quad, Jacobian, aspect, and target-count validation.
- Added exact-source versus exact-proxy topology proof status in output and UI.
- Moved area, diagonal, aspect, Jacobian, and fidelity gates into world space so
  non-uniform object scale cannot hide stretched quads or large error.
- Added a 60-degree local quad-warpage ceiling and per-component Euler checks.
- Added target-scale preprocessing to direct PairClosure and mode-specific
  field/pairing weights; unsafe concave triangle pairs now remain unpaired.
- Added ParityPath sparse quad transitions and deterministic target-aware
  selection against the conservative global corner closure.
- Added a topology-gated edge-rotation/QEM unlock for irreducible low-budget
  scan proxies; every round is transactional and signature preserving.
- Fixed the old collinear per-triangle pure-quad workaround.
- Fixed duplicate candidate recomputation in TRIAD-Q.
- Fixed boundary edges being reported as non-manifold.
- Fixed orphaned mesh data in Replace mode.
- Preserved material-slot order and isolated baked target materials.
- Made baking transactional: all settings and datablocks roll back on failure.
- Added real isolated install/enable smoke tests for both release zip layouts.
- Added algorithm and photogrammetry workflow documentation, unit coverage, and
  release packaging.

## 0.3.0

- Added TRIAD-Q Lite, standalone OBJ I/O, seed-swept pairing, and Blender smoke
  coverage.
