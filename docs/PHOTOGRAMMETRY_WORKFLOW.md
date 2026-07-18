# Photogrammetry Retopology Workflow

## Before You Run CurioMesh

- Keep the original scan in the file and use `New Object` output.
- Apply known unit scale and inspect disconnected debris, intentional openings,
  thin sheets, and flipped regions.
- Decide whether holes are evidence or defects. This determines `Preserve`
  versus `Repair`; CurioMesh cannot infer archaeological intent from geometry.
- Decide what high-frequency detail will move into color, normal, or
  displacement textures.

## Strategy By Asset

### Organic statue, rock, terrain, or creature

Use `Auto` or `Organic / General`, `Preserve`, and Balanced. Raise target count
when the p95 error or silhouette is poor. Automatic topology is suitable for
static/game/sculpt bases, not guaranteed deformation loops.

### Mechanical, architectural, or ship scan

Use `Mechanical / Feature` and `Preserve`. Keep seam/material feature options
on. Remove scan-noise edges before forcing feature preservation; a solver that
faithfully preserves false features produces faithfully bad topology.

### Open archaeological fragment

Use `Open Fragment` and `Preserve`. Do not enable dirty closed-scan repair.
Compare reported boundary-loop count with the source. Projection across nearby
opposing sheets remains a risk.

### Dirty closed photogrammetry

Analyze first. If holes and disconnected sheets are known defects, use `Dirty
Closed Scan` with `Repair`. Voxel repair can change genus, close intentional
openings, and remove features thinner than a few voxels. The output report
records the selected path and topology metrics.

### Multi-million-face scan

Atlas-Q area/spatial/component-samples the full source and builds a bounded
native QEM proxy. Connectivity gates are exact on that proxy, while the result
is prominently marked `PROXY_PRESERVED_SOURCE_UNVERIFIED`. The original remains
the projection/texture reference. Do not blindly split the model into
independent remeshing tiles: unrelated fields and singularities can create
visible seams.

## Reading The Result

- `Local Quad Manifold Valid`: mandatory for an exact-quad delivery. This
  checks vertex fans, boundary degree, winding, duplicates, and local quad
  geometry; it is not a global self-intersection proof.
- `Topology Proof`: distinguishes exact source preservation from exact proxy
  preservation and unverified full-source topology.
- `p95 Error / Diagonal`: symmetric sampled geometric deviation. Compare assets
  at the same unit scale and target count.
- `Max Sample Error / Diagonal`: catches isolated omitted pieces that averages
  and p95 can hide. Small connected source components receive sample priority
  within the documented two-million-vertex connectivity budget.
- `Min Scaled Jacobian`: must stay positive; values nearer one are squarer.
- `p95 Aspect Ratio`: identifies stretched cells even when average quality looks
  good. Both are world-space metrics, so unapplied non-uniform scale is included.
- `Extraordinary Verts`: poles are necessary on most topologies; look for
  clustering in visible or deforming areas.
- `Boundary Loops`, `Connected Components`, and `Euler Characteristic`: detect
  unintended topology change against the exact source or reduced proxy.
- `Target Error`: a large value means the requested count conflicted with the
  solver/closure. Raise the target or use a flow strategy.
- `curiomesh_vertex_fan_splits`: the number of coordinate-preserving
  vertex-only bow-tie splits applied after proxy reduction. A nonzero value is
  expected on some scan fragments; edge-non-manifold input is still rejected.
- `curiomesh_projection_status`: `ROLLED_BACK_FAILED_GATES` means shrinkwrap
  folded or otherwise invalidated the candidate, so the validated
  pre-projection topology was retained instead.

## If Atlas-Q Rejects Every Candidate

1. Read the result message and candidate JSON custom property.
2. Raise target faces; extremely coarse integer layouts can be invalid.
3. For an intentional open mesh, force `Open Fragment` + `Preserve`.
4. For a truly dirty closed mesh, use `Dirty Closed Scan` + `Repair`.
5. Use PairClosure for an exact-quad fallback and accept its pole/count tradeoff.
6. If semantic loops matter, add manual guide topology after automatic
   retopology. Geometry-only automation cannot know joints, eyelids, handles, or
   archaeological interpretation.

CurioMesh 0.4 can bake base color. Use a dedicated baker for normal and
displacement maps after approving the final topology and UVs.
