# Changelog

## Unreleased

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
  finish all quads. A 391,675-face hull that failed in the other engines now
  finishes.
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
  on a whale region and from about 31% to 2.8% on a noisy porthole scan;
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
