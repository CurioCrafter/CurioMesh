# Changelog

## Unreleased

- **Show it how (Teach by example)** (Auto Segments, step 4). Fix a spot by hand on the
  preview, in Edit Mode, with any tools, then press **Remember This Fix**. It stores the
  patch of quads you replaced and the ones you put there (positions relative to the
  patch, no mesh) plus three numbers for the shape, in a file in Blender's user config
  folder; nothing is uploaded, and **Export / Import** share it. **Look for Known Fixes**
  finds the same quad pattern in a result by walking the quads, checks the shape, and
  **Apply** puts the fix in after a camera looks at the spot from six sides and finds it
  no further from your original. The fixed area is also put back when you press Start
  again on the same mesh, if the quads around it come out the same. Limits: one
  area per press, the outline must stay as it was, a mirrored copy of a fix is not found,
  and the pattern must match exactly (the same face count and connections).
- **Draw Edge Loops** (Auto Segments, step 2). Draw closed loops (round an eye or the
  mouth) and open flow lines on the mesh, or add curves. Each becomes a strip of regular
  quads: three loops by default (the one you drew and one each side), evenly spaced, cut
  into the mesh before it is paired, so the poles end up outside the loops. The drawn
  loop runs as one continuous edge loop and sits on the stroke. On a bust (8,579
  faces) with two eye rings, a mouth ring and a chin line, the continuous loop along each
  stroke was 14, 14, 10 and 10 edges long (the longest 14, 14, 12, 11) where plain guide
  curves gave a median of 2 to 3 (longest 4 to 6); the loop lies 0.011 to 0.036 of an
  edge from the stroke on average (0.06 to 0.18 at worst). The tidiness score went from
  41.3 to 39.7: the strips are regular but the quads around them take the poles. Move a
  curve (G, or Edit Curve), redraw it, or delete it and the result updates by itself.
  **Selected Faces Only** redoes just the faces you selected and leaves the rest as it
  is; where the new outline does not line up with the old one a small gap or T-junction
  stays (on the bust, 226 of 398 outline vertices joined). Limits: a ring needs room for
  its loops (it is told "too small" and left out; lower Loop Gap), two strips that touch
  are not both built, and a strip that would fold on a steep surface is left out.
- **Deformation Lines** (Auto Segments, step 2). Draw a line across a joint or crease
  (elbow, knee, mouth corner, eyelid, knuckle) and it becomes an evenly spaced set of
  loops, three or more, in the same strip. Optional **Crease** and **Sharp** marks go on
  the middle loop when you Apply. A new Animation mesh type asks for these lines. Tested by
  bending a bust at the neck 70 degrees over a short span: with a ring across the neck,
  4.0% of the quads there collapse against 6.2% without it, and the average corner
  loses 0.058 instead of 0.087 of its squareness; over a broad span (0.012) it made
  no difference (4.6% without, 5.2% with). So it helps tight joints and little else.

- **Paint Density** (Auto Segments, step 2). Press Paint to go to Weight Paint on a
  vertex group and colour the parts that need more quads red; the search keeps more
  faces there and fewer elsewhere. Uses Blender's own Weight Paint and Decimate
  weights, so there is no brush of ours. Best with Reduce To, where the face count is
  fixed: on a bust aimed at 3,000 faces, the painted head went from 60% to 76% of the
  faces (81% at strength 3), and the view changed by 11.7% instead of 8.2%.
  With Keep Detail it adds faces to the painted part instead.
- **Guide curves** (Auto Segments, step 2). Draw strokes on the mesh with **Draw Guide**
  (hold the left mouse button), or add curves, Grease Pencil drawings or edge-only
  meshes with **Add Selected**. The quads follow them: an edge loop round an eye,
  a line along a panel. The strokes are laid on the surface, the triangles near them
  are split until they are small, and the edges that hug each stroke are kept as
  hard edges that no quad may cross and the flow runs along. On a bust with five
  strokes, the nearest edge was 0.30% of the size away on average (1.5% at worst)
  without guides and 0.10% (0.5% at worst) with them, for about 27% more faces.
  With Mirror on, a stroke on either side is used on both.
- **Mirror** (Auto Segments, step 2). Pick Auto, X, Y or Z and it solves one half of
  the mesh (the side with more surface) and mirrors it: the two sides are identical
  and the seam sits exactly on the plane, with no gap. Auto finds the plane and uses
  it only when 90% of the surface matches its mirror (70% when you pick the axis
  yourself); otherwise it solves the whole mesh and says how well the mesh matched.
  Tested on a mirrored bust: 100% match, watertight, all quads, nothing open on the seam.
- **Auto Segments is much faster and Stop really stops.** Waiting for the pairing no
  longer spins the main thread (it fought the pairing thread for Python's lock): a
  17,000-face bust went from 37 s to 7 s and a 61,000-face barrier from 13 s to 6 s,
  same result. Pressing Start again with another Change Allowed reuses what was
  already measured (about 1.5 to 4 s instead of 5 to 7 s). Stop now ends the
  pairing thread within a stopping point (0.06 s measured) instead of letting it
  run on. The report keeps the seconds spent looking, decimating and pairing.
- **Auto Segments is now a guided, five-step panel**: pick the mesh, choose the aim,
  press Start and watch, fix areas you don't like, Apply. One big button always shows
  the next step. A real progress bar (a text bar before Blender 4.0) shows the stage
  in words ("Looking at piece 12 of 250", "Pairing into quads, try 2") with a rough
  time left, and a Stop button. Work Live shows the same bar. A summary at the end
  gives faces before and after, percent fewer, and how much of the view changed.
  Start is its own button now; changing a setting no longer restarts the search.
- **Reduce To** (next to Keep Detail, the old Change Allowed). Ask for a percent of
  your faces or a face count and Auto Segments lands within about 10% of it, then
  reports how much the view changed.
- **Live Boolean** (step 4). Put boxes, spheres or any closed mesh over parts of the
  result, combine them with Union, Subtract or Intersect, and only the faces inside
  are redone, live, as you move the shapes. An Organized / Detailed slider sets how
  big the new quads are; everything outside stays as it is. Uses the Regions
  winding-number test and the Work Live background worker; no mesh Boolean needed.
- **Auto Segments** (new mode in the Remesh setting of the Guided panel). Press
  Update with no points and no face count: it reduces each loose piece as far as
  a camera says it can, looking from six sides at the original and the result,
  then makes all quads. **Change Allowed** sets how much of the view may look
  different. Small parts are judged by how big they look on the object, and
  Blender stays usable while it works.
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
