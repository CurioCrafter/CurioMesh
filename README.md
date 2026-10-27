# CurioMesh Atlas-Q Quad Retopology

CurioMesh 0.5 is a Blender 4.3+ add-on for turning dense meshes and
photogrammetry scans into lower-density, validated quad retopology. Atlas-Q is
not presented as a universal miracle solver. It is a scan-aware controller that
builds a target-scale proxy, chooses an appropriate solver path, compares
multiple candidates against the preserved source, and rejects results that do
not pass the requested topology and quad-quality gates.

## What's New In 0.5

![CurioMesh Quick Start panel, HUD and heatmap in Blender 4.5](docs/images/panel_quickstart.png)

### Get going in three clicks

Open `3D View > N-panel > CurioMesh`. The **Quick Start** panel holds
everything you need for a first result:

1. Pick a quality: **Draft**, **Balanced** or **Hero**.
2. Set **Target Faces** and the engine.
3. Press the big **Remesh** button.

The **See it** row turns on the three live views below. Advanced settings live
in collapsed sub-panels (Scan, Shape, Pipeline, PairClosure, Materials & UVs,
Batch, Diagnostics, Debug), so they stay out of the way until you need them.
Every earlier setting is still there.

### HUD dashboard

![HUD dashboard and triangle/pole heatmap on a remeshed Suzanne](docs/images/hud_heatmap.png)

A dark card in the corner of the viewport shows the active mesh at a glance:
quad percentage with a green/amber/red bar, faces against your target, leftover
triangles, poles, the engine and run time. The last line tells you the next
step, for example `146 triangles left - enable Require Quads`. It updates as
you edit or select meshes.

### Triangle and pole heatmap

One click on **Heatmap** colours problems right on the mesh, without touching
its materials: leftover triangles red, n-gons purple, 3-poles blue and 5+
poles orange. It follows edits live and skips meshes over 300,000 faces.

### Live preview

Turn on **Live** and CurioMesh re-runs the PairClosure engine into a preview
object beside the source about 0.4 s after you change target faces, feature
angle, seeds, flow mode or Require Quads, or edit the source. It works on
sources up to 20,000 faces; turning Live off removes the preview.

### Batch remesh

Select any number of meshes and press **Batch Remesh Selected** in the Batch
sub-panel. CurioMesh runs them one by one with the current settings, shows
progress in the HUD (Esc cancels), keeps going past failures, lists the results
in the panel and writes one `curiomesh_batch_<time>.json` report next to the
`.blend`.

### A faster, smarter PairClosure engine

- **About 100x faster.** Adjacency, feature detection, pair scoring and quad
  validation are vectorized with NumPy, and per-mesh work is computed once
  instead of once per seed. On a 7k-triangle noisy sphere (3 seeds) a
  quad-dominant run drops from 111 s to about 1 s, and an all-quad run from
  212 s to about 7.4 s.
- **Maximum pairing.** Triangle pairing starts from the best-scoring greedy
  pairs, then runs Edmonds' blossom matching over the valid pairs, so the most
  triangles possible end up in quads. Stranded triangles cost three quads each
  in closure, so this keeps all-quad output closer to the target.
- **Cross field.** Pairs now follow a 4-way cross field that is smoothed
  across the surface and pinned to sharp edges, boundaries, material borders
  and UV seams, so quads line up with the shape's features.

![Leftover triangles in red: greedy pairing alone vs maximum pairing](docs/images/rescue_before_after.png)

On this triangulated Suzanne (3,936 triangles, half-count target) the extra
pairing pass cuts leftover triangles from 202 to 146 (28 rescued pairs). The
count is shown in the PairClosure sub-panel and stored on the output as
`curiomesh_triadq_rescued_pairs`.

All screenshots come from Blender 4.5.1 LTS and can be regenerated with:

```powershell
blender -b --factory-startup --python-expr "import bpy; bpy.ops.wm.save_as_mainfile(filepath='empty.blend')"
blender --factory-startup --window-geometry 0 0 1920 1080 empty.blend --python scripts/capture_readme_screenshots.py
```

## What Changed In 0.4

- **Scan Intelligence** classifies organic, hard-surface, open, dirty, and very
  large inputs without changing the source object. The report explains whether
  topology counts are exact or sampled and warns about destructive repair.
- **Atlas-Q Auto** is the new default. It combines a QEM proxy, multi-seed
  QuadriFlow candidates, an optional dirty-scan repair candidate,
  area/spatial/component-stratified source-error measurement, and hard
  validation.
- **PairClosure** replaces the old per-triangle "pure quad" workaround. It
  performs feature-aware triangle pairing, compares a sparse ParityPath
  transition closure with the conservative global corner closure, and selects
  the valid result closest to the requested budget. Before closure it can
  normalize vertex-only bow-ties by
  duplicating disconnected incident fans without moving geometry, changing
  face order, or changing face-corner UVs. It deliberately refuses to present
  this as a repair for edges used by more than two faces. Every accepted face
  has exactly four distinct corners.
- **Topology and shape gates** reject edge and vertex non-manifolds, branched
  boundaries, duplicate faces, inconsistent winding, concave/crossed quads,
  excessive target error, poor Jacobians/aspect, and excessive symmetric RMS,
  p95, or maximum sampled source error.
- **Large-scan proxy budgets** keep Python-side work off the full-resolution
  scan. Blender's native collapse decimator creates a target-scale QEM proxy
  before candidate remeshing.
- `Replace` mode now releases the old orphaned mesh datablock. Material-slot
  ordering is preserved, and baking makes target materials single-user before
  editing their node trees.

The immutable evaluated source snapshot remains alive through remeshing,
projection, material mapping, UV transfer, and optional baking. A failed
candidate never replaces the selected source object. Projection is
transactional: if shrinkwrap makes an accepted candidate degenerate,
non-manifold, topologically different, or too inaccurate, CurioMesh restores
the validated pre-projection mesh and records the rollback on the output.

## Solver Portfolio

| Strategy | Use it for | Implementation and expectation |
|---|---|---|
| `Auto` | Default for most assets | Measures the scan and routes to the strongest applicable path. |
| `Organic / General` | Statues, rocks, terrain pieces, creatures, clean scans | Target-scale QEM proxy plus a deterministic multi-seed QuadriFlow ensemble. |
| `Mechanical / Feature` | Props, architecture, machinery, crisp archaeology | Preserves sharp, seam, material, and boundary evidence while comparing flow candidates. |
| `Dirty Closed Scan` | Noisy closed photogrammetry with broken connectivity | May voxel-repair one candidate. It can close holes and erase thin sheets, so it requires `Repair` topology policy. |
| `Open Fragment` | Intentional sheets, excavation fragments, partial scans | Preserves boundary loops and avoids automatic voxel closure. |
| `Exact PairClosure` | Robust fallback or simple/open triangulated surfaces | Guarantees conforming quads after validation, but can create more poles and may miss an aggressive face target. |

Atlas-Q accepts a candidate only after measured checks. `Require Locally Valid
Quads` is enabled by default. A solver process finishing successfully is not
enough. “Locally valid” deliberately does not claim a global face-to-face
self-intersection proof.

## Installation

1. Download a release archive or clone this repository.
2. In Blender, use `Edit > Preferences > Add-ons > Install from Disk`.
3. Select the CurioMesh add-on or extension zip and enable **CurioMesh Quad
   Remesher**.
4. Select a mesh and open `3D View > N-panel > CurioMesh`.

CurioMesh targets Blender 4.3+ and the test runner prefers Blender 4.5 LTS.

## Recommended Photogrammetry Workflow

1. Keep `Output` set to `New Object`.
2. Set the desired **Target Faces**. This is a soft geometric budget: topology,
   features, and valid quads take precedence over an impossible exact count.
3. Click **Analyze Without Changing Mesh**.
4. Leave `Scan Strategy` on `Auto` unless the report misclassifies known intent.
5. Use `Preserve` topology for open fragments, intentional holes, thin shells,
   mechanical parts, and most clean meshes.
6. Use `Repair` only for a closed scan whose holes and broken connectivity are
   known defects.
7. Run Atlas-Q and inspect local quad-manifold status, topology proof, p95/max
   source error, minimum scaled Jacobian, p95 aspect ratio, boundary loops, and
   silhouette.
8. Keep UV projection or bake base color. Bake normal/displacement detail in a
   dedicated external baking step; CurioMesh does not expose those maps.
   A low-poly mesh cannot encode scan-frequency detail in geometry alone.

For multi-million-face scans, Atlas-Q samples the full source, then performs
exact topology accounting on the reduced proxy. The result explicitly says
`PROXY_PRESERVED_SOURCE_UNVERIFIED`; it does not present proxy preservation as
proof about every full-resolution source edge. See
[docs/PHOTOGRAMMETRY_WORKFLOW.md](docs/PHOTOGRAMMETRY_WORKFLOW.md) for recovery
steps and expectations by scan type.

## Mathematical Design

Atlas-Q uses an error-budgeted solver selection rather than claiming one
algorithm wins on every mesh.

The initial target edge scale is

```text
h0 = sqrt(A / Nq)
```

where `A` is sampled surface area and `Nq` is the desired quad count. The
Balanced QEM proxy budget is approximately

```text
Fproxy = min(Fsource, clamp(12 Nq, 2,000, 400,000)).
```

Each valid candidate is scored from measured symmetric source distance, scaled
Jacobian, aspect ratio, target-count error, pole ratio, and quad ratio. Topology
violations are rejection conditions rather than small score penalties.

PairClosure starts with feature-aware triangle pairing. It then processes every
residual triangle with shared edge midpoints. ParityPath keeps unaffected valid
quads unchanged and solves odd quad transitions as a T-join on the quad-only
face-dual graph. Affected faces use locally validated two-, three-, or
four-quad templates with shared geometric midpoints and face-local UVs.

The conservative target closure processes every base face
`f = (v0 ... vk-1)` using a globally shared edge midpoint `mi` and a face
center `cf`:

```text
qi = (vi, mi, cf, m(i-1)).
```

This produces `Fglobal = 3T + 4Q`. ParityPath instead has
`Fparity = 3T + Q0 + 2Q2d + 3Q2c + 4Q4`, where the `Q` terms classify
unaffected, direct two-split, center-filled two-split, and four-split quads.
Both paths are conforming, preserve Euler characteristic when the base is
manifold, guarantee four-corner faces, and preserve face-local UV seams.

If Blender's collapse decimator stalls above a closure budget, PairClosure can
rotate a deterministic face-independent set of interior triangle diagonals and
retry QEM. Every round is transactional: it is retained only when manifold,
degeneracy, component, Euler, and per-component boundary-signature gates remain
exact. The final quad count and source-error gates still decide acceptance.

For a vertex-only bow-tie, incident faces around vertex `v` are partitioned by
edge adjacency into fans `F0 ... Fk`. `F0` keeps `v`; each later fan receives a
coincident duplicate `v_i`. This is an exact combinatorial normalization:
coordinates, faces, materials, and corner UV indices are unchanged. True
edge-non-manifolds remain rejected.

For a closed all-quad manifold, poles are mathematically necessary:

```text
sum_v (4 - valence(v)) = 4 chi.
```

A sphere therefore has total defect charge eight; "no extraordinary vertices"
is not a valid universal requirement. CurioMesh measures and minimizes excess
poles instead.

The detailed implemented objective, gates, and roadmap are in
[docs/ALGORITHM.md](docs/ALGORITHM.md).

## Controls

- `Target Faces`: approximate final quad budget.
- `Engine`: Atlas-Q Auto, direct QuadriFlow, PairClosure, or the v0.3 Auto alias.
- `Quality`: Draft uses one candidate; Balanced uses the configured budget;
  Hero uses at least five candidates and stricter acceptance gates.
- `Scan Strategy`: solver-family override.
- `Topology`: `Preserve` rejects changed Euler characteristic, boundary-cycle
  count, connected-component count, or open/closed state against an exact
  source or proxy reference. It also compares a sorted `(Euler, boundary
  cycles)` signature per component, so cycles cannot silently move between
  equal-Euler components;
  `Repair` permits change. Proxy-only proof is prominently labeled.
- `Require Locally Valid Quads`: rejects non-quads, repeated/zero-area faces,
  concave/crossed/inverted quads, edge/vertex non-manifolds, branched
  boundaries, duplicate faces, inconsistent winding, and quads warped more
  than 60 degrees across either diagonal. Gates are measured in world space,
  including non-uniform object scale. Global self-intersection remains
  unverified.
- `Candidate Budget`: one to eight deterministic candidates.
- `Preserve Sharp`, `Preserve Boundary`, `Treat UV Seams As Sharp`: feature
  constraints passed into proxy/remeshing stages.
- `Voxel Repair Fallback`: available for direct QuadriFlow and controlled by
  Atlas-Q's dirty-scan strategy.
- `Project Details`: projects the accepted topology back to the preserved
  evaluated source.
- `Texture Preservation`: UV projection/transfer, base-color bake, or none.
- `Output`: create a new object or replace the selected object's mesh after all
  mandatory checks pass.

## Standalone PairClosure CLI

The dependency-light TRIAD-Q/PairClosure path can still process OBJ files
outside Blender:

```powershell
python -m triadq input.obj output.obj --target-faces 4000 --mode auto --seed-count 8 --pure-quads
```

The Blender PairClosure engine now builds a target-scale QEM proxy before the
Python pairing stage. The standalone CLI performs mode-tuned field pairing and
exact quad closure but does **not** have Blender's QEM or QuadriFlow backend, so
`target-faces` remains a scoring target outside Blender.

## What Is Implemented Versus Research

Implemented now:

- Bounded scan classification and explicit uncertainty.
- Target-scale native QEM proxy construction.
- Multi-candidate QuadriFlow, feature-flow, open-flow, and dirty-flow routing.
- Exact conforming PairClosure.
- Sparse ParityPath transitions plus topology-gated edge-rotation/QEM unlock.
- Area-, spatial-, and component-stratified symmetric distance scoring with
  RMS, p95, and maximum-error gates.
- Immutable-source projection, UV/material transfer, and baking.

Not claimed or implemented:

- A proof that Atlas-Q is globally novel or universally better.
- Full mixed-integer global patch-layout optimization comparable to MIQ,
  Integer-Grid Maps, or QuadWild.
- Semantic animation loops around eyes, mouths, joints, handles, or cloth.
- Guaranteed exact face count, minimum pole count, or excellent quad shape for
  every topology and budget.
- A global triangle/quad self-intersection proof or semantic deformation loops.
- Out-of-core cancellation during Blender's synchronous QuadriFlow operation.

The new contribution is a clean-room CurioMesh synthesis: confidence-aware scan
routing, error-budgeted proxy construction, measured candidate selection, and
an exact closure/validation contract. Benchmark superiority remains unproven
until real scan comparisons are published.

## Research Basis

CurioMesh's implementation and roadmap are grounded in primary sources:

- [Instant Field-Aligned Meshes](https://igl.ethz.ch/projects/instant-meshes/)
  for scalable local orientation/position fields.
- [QuadriFlow](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13498) for
  scalable global consistency and lower singularity counts.
- [Mixed-Integer Quadrangulation](https://www.graphics.rwth-aachen.de/publication/0344/)
  and [Integer-Grid Maps](https://www.graphics.rwth-aachen.de/publication/03197/)
  for seamless parameterization and reliable quad extraction.
- [Reliable Feature-Line Driven Quad Remeshing](https://www.quadmesh.cloud/ReliableQuad.pdf)
  for feature-aware patch layouts and global quantization.
- [Pattern-Based Quadrangulation](https://igl.ethz.ch/projects/patch-quad/) for
  valid patch filling.
- [Garland-Heckbert QEM](https://www.cs.princeton.edu/courses/archive/fall03/cs526/papers/garland97.pdf)
  for target-scale proxy simplification.
- [Quasi-structured quad meshing with midpoint subdivision](https://arxiv.org/abs/2103.04652)
  for the robust all-quad closure pattern.
- [Blossom-Quad](https://orbi.uliege.be/handle/2268/113152) for graph matching
  formulations of indirect quadrangulation.
- [Practical Quad Mesh Simplification](https://vcg-legacy.isti.cnr.it/Publications/2010/TPCPP10/Tarini%20Pietroni%20Cignoni%20Panozzo%20Puppo%20-%20Practical%20Quad%20Semplification%20-%20EG%202010.pdf)
  and [Quadrilateral Mesh Simplification](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Daniels08.pdf)
  for topology-preserving local connectivity operations and quad LOD design.

CurioMesh is MIT licensed. No GPL QuadWild/BiMDF source is copied or vendored.

## Development, Tests, And Packaging

```powershell
python -m unittest discover -s tests -v
python tests\triadq_smoke.py
python -m compileall -q __init__.py atlasq.py bridge.py metrics.py operators.py textures.py ui.py triadq tests scripts
ruff check .
tests\run_blender_tests.ps1
python scripts\build_release.py
```

The release builder creates a legacy add-on zip and a Blender extension-layout
zip in `dist/`, then validates every archive entry. Generated caches and build
artifacts are excluded from Git.

## License

MIT. See [LICENSE](LICENSE).
