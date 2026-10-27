# Atlas-Q Algorithm And Validation Contract

This document separates shipped behavior from research directions. The code is
a clean-room CurioMesh implementation under MIT; it does not copy GPL solver
sources.

## Problem Contract

Input can be a clean authored mesh, an open fragment, or a noisy photogrammetry
scan. The requested output is a lower-density surface whose accepted faces are
all valid quads. Four different goals can conflict:

1. geometric fidelity to the scan,
2. low face count,
3. clean quad shape and few poles,
4. preservation or intentional repair of topology.

Atlas-Q makes these conflicts visible. Topology and validity are hard gates;
face count is a soft target.

## Implemented Pipeline

### 1. Immutable source

The evaluated object is copied once and retained as the reference for source
distance, projection, material assignment, UV transfer, and baking. Work is
performed on separate objects. `Replace` happens only after validation.

### 2. Bounded scan intelligence

For meshes up to the exact-analysis limit, CurioMesh measures edge incidence,
true boundary cycles, connected face components, Euler characteristic,
edge/vertex manifoldness, duplicate faces, winding consistency, degeneracy,
triangle ratio, and dihedral feature coverage. Above the limit it samples the
full source, reduces it natively, and performs exact connectivity checks on the
proxy. The basis is carried as `SOURCE_EXACT`, `PROXY_EXACT`, or `SAMPLED`.

Auto routing begins with these empirical thresholds:

- dirty: non-manifold edge ratio above `0.1%`, sampled degenerates above
  `0.5%`, or a vertex non-manifold, branched boundary, or duplicate face;
- open: one or more measured boundary components;
- hard surface: sharp-edge coverage above `5%`;
- large scan: source exceeds exact-analysis budget;
- organic: otherwise.

UVs or multiple material slots add a texture-critical warning without changing
the geometric class.

### 3. Target-scale QEM proxy

For target count `Nq`, estimated target edge length is

```text
h0 = sqrt(A / Nq).
```

CurioMesh applies Blender's native collapse decimator before Python adjacency
or standalone conversion. Proxy budgets are:

| Quality | Multiplier | Cap | Candidate count |
|---|---:|---:|---:|
| Draft | `6 Nq` | 150k | 1 |
| Balanced | `12 Nq` | 400k | configured, default 3 |
| Hero | `20 Nq` | 400k | at least 5 |

Sharp, seam, and material delimiters are requested where Blender exposes them.
The source reference is never decimated.

On explicitly budgeted PairClosure runs, Blender QEM can stall on an
irreducible triangulation even though a smaller triangulation of the same
surface exists. Atlas-Q then rotates a deterministic face-independent set of
interior triangle diagonals and retries QEM for a bounded number of rounds.
Each round snapshots the mesh and is committed only when it does not increase
triangle count and preserves all manifold/degeneracy gates plus the exact
sorted per-component `(Euler, boundary cycles)` signature. Otherwise it is
rolled back. The closure estimate used to stop early is heuristic; the actual
quad count and fidelity gates remain authoritative.

### 4. Solver candidates

Flow candidates run Blender's QuadriFlow with deterministic seeds separated by
7919. Feature/Open variants alter sharp and boundary constraints. DirtyFlow may
voxel-repair the first candidate only when topology repair is explicitly
allowed.

If no valid flow candidate remains, Atlas-Q runs PairClosure. Direct Blender
PairClosure uses the same target-scale preprocessing. It further
decimates toward a base triangle budget

```text
Nbase ~= Nq / (3 - p),  p = 0.8 initially,
```

then performs feature-aware triangle pairing. Organic, Patch, Dirty,
and Texture modes use different field-smoothing, feature-angle, and pairing
weight profiles. A proposed pair must itself be a finite, convex, non-crossed
quad with no more than 60 degrees of triangle-normal divergence across either
diagonal; unsafe pairs remain triangles for exact closure.

Pair scoring is guided by a per-face 4-RoSy cross field. Each triangle has a
tangent frame; its cross is stored as `exp(4i theta)` so the four equivalent
directions share one value. Feature edges (boundary, non-manifold, sharp,
material, UV seam) are hard constraints: faces touching one are pinned to
their longest feature edge. Free faces start from a weak longest-edge guess
and are smoothed by Jacobi averaging across non-feature edges with parallel
transport `exp(4i (alpha_i - alpha_j))`, where `alpha` is the shared edge's
angle in each face frame. A pair's field term is the mean alignment of its
four sides with the nearest of the four cross directions.

Pairing starts from a score-greedy matching (faces visited in seed order,
each taking its best still-free partner), then grows it to a
maximum-cardinality matching over the allowed pair graph with Edmonds'
blossom algorithm, so augmenting paths through odd dual cycles are found.
`rescued_pairs` reports the augmentations beyond the greedy start. Adjacency,
features, the field, and pair scores are computed once per mesh and shared
by all seeds; only the visit order, matching, and closure run per seed.

When PairClosure is explicitly requested, Atlas-Q bypasses the larger
flow-candidate proxy and builds only this closure-sized QEM proxy. Avoiding two
serial decimation passes matters for sub-1,000-quad assets: it reduces topology
drift and makes the requested final budget correspond to the closure model.
For an explicitly low-poly fallback, callers may also release sharp/UV seam
delimiters on the closure proxy. Exact topology, local quad validity, and
surface-error gates still decide acceptance; the relaxed candidate cannot
silently remove a handle, boundary cycle, or component.

Batch controllers can set an explicit closure-base budget after observing a
failed candidate's expansion ratio. If a base request `B0` produced `Q0` quads
for desired output `Qt`, the next deterministic estimate is proportional to
`B1 = B0 Qt / Q0`, with a small safety margin. The final target remains `Qt`;
only proxy construction changes, and all topology/fidelity gates are rerun.

Immediately after the QEM proxy and before triangle pairing, PairClosure checks
the local fan of every used vertex. If the incident faces split into `k > 1`
groups when adjacency is restricted to edges containing that vertex, each fan
after the first receives an exact coordinate duplicate. This removes
vertex-only bow-ties even when those fans connect elsewhere in the same global
component. Face order, material IDs, and corner UV indices do not change. The
normalizer is deterministic and refuses inputs containing an edge incident to
more than two faces; that defect needs a different, explicit repair policy.

### 5. ParityPath and conservative exact closure

After feature-aware pairing, ParityPath marks all three edges of every residual
triangle. A neighboring quad incident to an odd number of marked edges becomes
a terminal in the quad-only dual graph. Deterministic shortest paths pair those
terminals or route them to a mesh boundary; path edges are combined by
symmetric difference. This is the parity equation

```text
sum(edge splits around face) = face arity  (mod 2).
```

Unchanged quads remain one face. Two-split quad cells use a validated direct
two-quad partition when possible or a three-quad center template; four-split
cells use four quads. Residual triangles use the robust three-quad midpoint and
center template. Every local template is checked for finite, convex,
non-crossed geometry before it is committed, and the complete component
signature is checked again afterward.

For transition classes `Q0`, `Q2d`, `Q2c`, and `Q4`, the sparse count is

```text
Fparity = 3T + Q0 + 2 Q2d + 3 Q2c + 4 Q4.
```

Atlas-Q also constructs the conservative global candidate. For every base
polygon `f=(v0,...,vk-1)`, construct one globally shared geometric midpoint
for each undirected edge, one face center, and

```text
qi = (vi, midpoint(vi,v(i+1)), center(f), midpoint(v(i-1),vi)).
```

Applying this to the entire base—not only leftover triangles—is what makes the
result conforming. Shared geometric edges receive one midpoint. UV midpoints
remain face-local so seams survive. Material IDs are repeated per generated
corner quad.

For `T` unpaired triangles and `Q` paired quads, that candidate has:

```text
Fglobal = 3T + 4Q.
```

The engine returns the valid closure whose face count is closest to the target,
using the lower count as the deterministic tie-breaker. ParityPath is therefore
an optimization, not a replacement for the proven conservative construction.

If the base is a valid consistently wound manifold, the operation preserves its
Euler characteristic and manifoldness. Acceptance additionally checks vertex
fans, boundary degree, duplicates, winding, local crossing, and corner
Jacobians. It does not prove that spatially separate faces never intersect.

### 6. Symmetric source error

CurioMesh builds world-space BVHs for source and candidate and samples both
directions. Samples combine:

- systematic triangle-area strata,
- occupied cells of an 8x8x8 world-space grid,
- representatives biased toward small connected source components.

NumPy union-find provides exact component representatives up to two million
vertices/six million edges; above that explicit guard, area and spatial strata
remain active and exhaustive component coverage is not claimed. RMS, p95, and
maximum sampled distances are normalized by source bounding-box diagonal.
This catches off-surface output and omitted regions more reliably than one-way
shrinkwrap distance or index-strided vertices.

### 7. Candidate score

Only candidates passing hard gates receive a positive score. The current
measured score is

```text
S = 0.35 G + 0.25 J + 0.15 A + 0.10 C + 0.10 P + 0.05 Q,
```

where:

- `G = exp(-(p95/h0)/0.35)` is geometric fidelity,
- `J` is clamped p05 scaled Jacobian,
- `A = exp(-max(aspect95-1,0)/3)` is aspect quality,
- `C = exp(-target_error)` is count fidelity,
- `P = exp(-4 extraordinary_ratio)` penalizes excess poles,
- `Q` is quad ratio.

This is a practical controller objective, not a claim of a globally optimal
quadrangulation functional.

## Hard Acceptance Gates

With `Require Locally Valid Quads`, an output is rejected if it has:

- any face whose arity is not four,
- repeated-corner, zero-area, concave, crossed, inverted, or more-than-60-degree
  warped faces,
- non-positive corner scaled Jacobian,
- an edge incident to more than two faces or a disconnected vertex fan,
- a branched boundary, duplicate face, or inconsistent shared-edge winding,
- changed global Euler characteristic, connected-component count, open/closed
  state, or the sorted per-component `(Euler, boundary cycles)` signature under
  exact source/proxy `Preserve` analysis,
- quality-preset violations in target count, p05 Jacobian, p95 aspect, or
  symmetric RMS/p95/maximum sampled surface error.

Current hard thresholds are:

| Quality | Target error | RMS / target edge | p95 / target edge | max / target edge | p05 Jacobian | p95 aspect |
|---|---:|---:|---:|---:|---:|---:|
| Draft | 75% | 0.60 | 1.50 | 4.00 | 0.01 | 12 |
| Balanced | 35% | 0.40 | 1.00 | 3.00 | 0.03 | 8 |
| Hero | 25% | 0.25 | 0.75 | 2.00 | 0.05 | 6 |

Validation runs before and after projection. Shrinkwrap succeeding does not
override a failed quality gate. The accepted candidate mesh is snapshotted
before projection; if projection violates a gate, CurioMesh restores that mesh
and records `curiomesh_projection_status=ROLLED_BACK_FAILED_GATES`. If applying
the modifier itself fails, it records `ROLLED_BACK_APPLY_FAILED`. A rollback is
preferred over silently shipping folded or degenerate quads. “Locally valid” is not an embedded-surface
self-intersection certificate; visual/intersection inspection is still
required for a shipping asset.

Area, bounding diagonal, aspect, Jacobian, and BVH distances are evaluated in
world space. Non-uniform object scale therefore cannot make a visibly stretched
quad pass by looking square in local coordinates; applying scale can still help
the underlying solver produce better flow.

For a closed orientable all-quad mesh:

```text
E = 2F
V - F = chi
sum_interior (4 - valence) = 4 chi.
```

With boundary:

```text
sum_interior (4-valence) + sum_boundary (3-valence) = 4 chi.
```

Regression tests exercise both disk and sphere-topology examples.

## Research Roadmap, Not Shipped Claims

The strongest longer-term path is a global patch solver:

1. robust multi-scale normal/curvature confidence,
2. confidence weighting and a global (non-Jacobi) solve for the 4-RoSy field
   (the standalone engine ships a feature-pinned, diffused 4-RoSy field; see
   Solver candidates),
3. singularity simplification at target scale,
4. feature/separatrix patch layout,
5. integer arc quantization with parity and no T-junctions,
6. injective parameterization using symmetric Dirichlet energy,
7. robust extraction and local cavity templates.

A representative field objective is

```text
sum_ij w_ij rho(|q_i - exp(i4 phi_ij) q_j|^2)
  + lambda sum_i c_i |q_i - exp(i4 theta_i)|^2.
```

An adaptive curvature metric can start from the sagitta estimate

```text
h_j(x) ~= sqrt(8 epsilon / (|k_j(x)| + k_floor)).
```

These ideas derive from established MIQ, Integer-Grid Map, field-aligned,
feature-patch, and injective-parameterization research. They are not silently
represented as implemented by the current Python add-on.

## Primary Sources

- [Instant Field-Aligned Meshes](https://igl.ethz.ch/projects/instant-meshes/)
- [QuadriFlow](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13498)
- [Mixed-Integer Quadrangulation](https://www.graphics.rwth-aachen.de/publication/0344/)
- [Integer-Grid Maps](https://www.graphics.rwth-aachen.de/publication/03197/)
- [QEx robust quad extraction](https://www.graphics.rwth-aachen.de/publication/03204/)
- [Reliable Feature-Line Driven Quad Remeshing](https://www.quadmesh.cloud/ReliableQuad.pdf)
- [Pattern-Based Quadrangulation](https://igl.ethz.ch/projects/patch-quad/)
- [Garland-Heckbert QEM](https://www.cs.princeton.edu/courses/archive/fall03/cs526/papers/garland97.pdf)
- [Quasi-structured quad meshing](https://arxiv.org/abs/2103.04652)
- [SLIM locally injective maps](https://igl.ethz.ch/projects/slim/SLIM2017.pdf)
- [Blossom-Quad](https://orbi.uliege.be/handle/2268/113152)
- [Practical Quad Mesh Simplification](https://vcg-legacy.isti.cnr.it/Publications/2010/TPCPP10/Tarini%20Pietroni%20Cignoni%20Panozzo%20Puppo%20-%20Practical%20Quad%20Semplification%20-%20EG%202010.pdf)
- [Quadrilateral Mesh Simplification](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Daniels08.pdf)
