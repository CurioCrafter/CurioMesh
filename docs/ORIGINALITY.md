# Originality

CurioMesh is written for this repository. No code, UI, feature names or text were copied or translated from other
retopology tools, commercial or open source. Where an algorithm rests on published math, the module says so and cites
the paper or author: that is how we credit ideas, and the code is our own writing of them. Blender's own operators
(QuadriFlow Remesh, Voxel Remesh, Decimate, Multires, Bevel, Data Transfer) are called, not included.

The only third-party code in the repository is listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md): one
triangle test ported from Blender (GPL-2.0-or-later, credited in `native/src/near.rs`) and the Rust crates the native
core links. QuadWild, QuadriFlow and Instant Meshes source is not included.

## Module by module

| Module | What it does | Origin | Notes |
|---|---|---|---|
| `triadq/engine.py`, `native/src/matching.rs`, `pairing.rs` | Pairs triangles into quads by a maximum matching, then closes leftovers | Published: Blossom-Quad (Remacle et al. 2012), Edmonds 1965 | The matching follows the usual textbook form of Edmonds' algorithm. `pairing.rs` reproduces Python's MT19937 (Matsumoto and Nishimura 1998) so both paths pick the same pairs |
| `triadq/topology.py` (ParityPath, quad closure) | Turns a triangle/quad mesh into all quads with no T-junctions | Our design; the last-resort split is the Catmull-Clark midpoint split | Routing odd triangles along dual-graph paths is our own scheme |
| `triadq/field.py` | Cross field from curvature | Published: Viertel and Osting 2019, Rusinkiewicz 2004 | Cited in the module |
| `triadq/fieldmesh.py`, `native/src/field.rs` (Field Rows) | Quads in rows from a direction field and a lattice | Published idea: Jakob, Tarini, Panozzo, Sorkine-Hornung 2015 (Instant Field-Aligned Meshes) | Own NumPy implementation; extraction, conformal cell sizes, plane fitting and row sliding are our own |
| `triadq/rowfill.py`, `flowstroke.rows` (Redraw in Rows) | Lays a marked area out again as straight rows at the same quad size, tied to the faces round it | Published pieces: curvature tensor from edge bending (Cohen-Steiner and Morvan 2003), four-fold direction averaging in the power form (Knoppel, Crane, Pinkall and Schroder 2013), Tutte / Floater flat maps evened by stretch (Yoshizawa, Belyaev and Seidel 2004) | The constant field on a graded flat map, the tie row chosen by dynamic programming over the corners each vertex lacks, and the border easing are our own |
| `triadq/quadreduce.py`, `native/src/qem.rs` | Fewer quads / fewer triangles with least shape change | Published: Garland and Heckbert 1997 quadrics | `qem.rs` uses the thresholds of Blender's Decimate on purpose and names them in comments |
| `triadq/tidy.py` | Edge turns, collapses, doublets on a finished quad mesh | Published: Tarini et al. 2010 | Cited |
| `triadq/chords.py`, `native/src/chords.rs` (Redraw Messy Quads) | Redraws a messy patch as a plain grid via a closed walk on squared paper, judged by the camera | Our design | Daniels et al. 2008 cited as the approach not taken |
| `triadq/refill.py` | Replaces a pole-heavy patch with one grid | Published idea: Takayama et al. 2014, Marcias et al. 2015 | Cited |
| `triadq/organize.py`, `organize.py` (Organize slider) | Cleans poles, then settles quads like damped cloth on the surface | Published: Tarini et al. 2010, Mueller et al. 2005 (shape matching), Mueller et al. 2007 (position-based dynamics) | The pulls, guards and slider steps are our own |
| `triadq/trial.py`, `tidylive.py`, `tidytools.py` (Look First) | Shows a tool's change on the mesh before it is kept, with one ring handle and a numbers card | Our design (bench/tools_design/INTERACTION.md) | The ghost, wash, ring, card and keys are our own; no other tool's preview was looked at |
| `triadq/hardtidy.py` (True Up Edges) | Finds the result's hard-edge chains and lays each on the crease of the original: straight ones on one line, curved ones smoothed and evenly spaced | Published pieces: least-squares planes (principal axis fit), Taubin 1995 (shrink-free curve smoothing) | Reading each crease as where two planes fitted to the original on either side meet, and accepting it only where that line lies on the original, is our own. Set Linear / Straight-style tools work on a picked chain against itself; this finds the chains and matches them to the original |
| `triadq/hardtidy.py` (Press Panels Flat) | Finds flat panels and presses each onto the plane of the original under it; shared corners go to the point nearest all their planes | Published pieces: least-squares plane fit, regularized least squares for the shared corners | Panel growing follows our own Clean Panels (`flatquad`); a corner goes onto a plane only where the original under it lies on that plane. No pick-a-plane step and no Flatten / Make Planar behaviour |
| `triadq/tidytools.py` (Smooth Corners) | Slides corners toward their neighbours' middle, then back onto the source | Published: Taubin 1995 (umbrella smoothing), surface projection as in Tarini et al. 2010 | The example tool; the shrink step of Taubin's filter is replaced by the projection |
| `triadq/relax.py` (Source Relax) | Slides corners in their tangent plane toward the area-weighted middle of their quads, then back onto the source | Published: Botsch and Kobbelt 2004 (tangential smoothing in remeshing), Desbrun, Meyer, Schroeder and Barr 1999 (Laplacian) | The normal part of each move is dropped and the corner goes back onto the original scan, not the low-poly mesh |
| `triadq/relax.py` (Keep Lines Relax) | As Source Relax, but corners on hard edges, the outline, UV seams, part borders and guides slide only along their line | Published: Botsch and Kobbelt 2004 (feature edges relaxed along themselves) | Lines are found by what they are on the result held on its source; the outline is read from the camera's sides |
| `triadq/relax.py` (Square Up) | Pulls each quad's corner angles toward 90 degrees, keeping its two side lengths and its row directions | Published: Mueller et al. 2005 (shape matching), Mueller et al. 2007 (position-based dynamics) | The rectangle's axes come from the quad's own rows, not a best fit; a one-shot button, never a brush |
| `triadq/relax.py` (Relax Toward Detail) | Same faces, smaller where the original bends and bigger where it is flat | Published: Du, Faber and Gunzburger 1999 (centroidal Voronoi with a density), Alliez et al. 2003 | The density is read from the spread of the original's normals under each quad |
| `triadq/loopspoles.py` (Even the Band) | Spaces the points on every line across a band of strips evenly by arc length | Our design on `triadq/strips.py` rung lines | Many strips at once along rung lines; not a loop-spacing tool |
| `triadq/loopspoles.py` (Cancel Pole Pair) | Walks a picked 3-pole and 5-pole together by edge spins until both are regular | Published: Tarini et al. 2010 (edge rotation), Peng et al. 2011 (moving singularity pairs) | The search reuses `fewerpoles.PoleCleaner`'s checked spins on a patch round the two |
| `triadq/loopspoles.py` (Open Up Poles) | Turns the edges round each pole to even angles and equal lengths in its tangent plane | Our design (an angle-scaled polar map of the one-ring, as in the discrete exponential map of Schmidt, Grimm and Wyvill 2006) | Fans that get narrower keep their old place; held on the source by `organize._onto` |
| `triadq/tidytools.py` (Preview Tidy) | A findings report: two-edge corners and doublets, triangle pairs that make quads, slivers and faces turned the other way, each line with its fix worked out first and its look cost | Published: Tarini et al. 2010 (doublet removal); the pairs are a greedy matching, squarest quad first; slivers use the Issue Finder test and umbrella smoothing (Taubin 1995) held on the source | A report of only what was found, each line its own Undo step; no checklist ticked before one run and no presets |
| `triadq/tidytools.py` (Seal the Middle) | Puts corners near the mirror plane onto it, merges doubled ones and counts gaps and crossing faces left | Our design; the plane comes from the result's mirror or from `symmetry.find_plane` | Says what it fixed; no modifier is added or applied |
| `triadq/tidytools.py` (Swap Parts: Look First for Smaller/Bigger Quads, Redo This Spot, Keep Original Here, Put Reduced Back) | Shows a part's other faces (a ready size, your own faces) over the old ones before they are kept | Our design | No new maths: the faces already exist as the part's tries |
| `triadq/tidytools.py` (Detail Marker sample grid) | A grid of quads at the size a marker ball asks for, laid on the surface inside its ring, with a face estimate | Our design; the size comes from our own Decimate weight table (`budget.density_drop`) | The ball's (1 - d^2)^2 falloff is integrated ring by ring for the estimate |
| `triadq/tidytools.py` (Hard Edges preview) | Edges whose faces meet past an angle, drawn before Apply | Standard dihedral angle between face normals (Newell's normal, Newell 1972 as in Sutherland et al. 1974) | The same test Apply's Hard Edges makes with Blender's face angle |
| `triadq/autofix.py`, `autofix_live.py`, `triadq/stripfix.py` (Auto Fix, Fix every spot like this) | Each tidy tool finds its own problem spots and says whether a fix made them better; one shared function runs a tool on its spots and keeps only fixes that improve the quads within the look budget, in rounds while they help | Our design (bench/tools_design/AUTO_FIX.md) | The spot finders reuse each tool's own checks; the accept rule, the drop-the-parts-that-changed-most then halve-the-strength fitting and the round loop are our own. "Auto Fix" is a plain phrase used by no surveyed product as a feature name. Hard Edges' crease reading compares the original's normals just inside each side of an edge (standard dihedral angle) |
| `triadq/history.py`, `history.py` (History) | Steps of the add-on's own results that follow Blender's Ctrl+Z, and a list to jump back | Our design | A step number on the scene that Blender's undo restores picks the stored step |
| `triadq/patterns.py` | Tidiness score against statistics from hand-made meshes | Our design | Data sources and licences in [PATTERN_SOURCES.md](PATTERN_SOURCES.md) |
| `triadq/loopscore.py` (Loop Quality) | Scores edge loops for straightness, spacing, poles | Our design | Targets are our own reading of artist practice |
| `triadq/flatquad.py` (Flat Quads) | Few faces on flat areas, all of them on detail | Our design | Rebuilt from the owner's own earlier NEST write-up |
| `triadq/fit.py`, `triadq/cases.py` | Fits vertices and shading to the nearest source points; named test cases | Ideas from cav3dw3ll3r, credited in the README | Code is our own |
| `triadq/smoothgrid.py` | Rebuilds domes and rings as concentric rows | Our design | |
| `triadq/hardsurf.py`, `hard_surface.py` | Clean Panels, Even Cylinders, hard-edge finish | Published pieces: Coons 1967 patches, constrained Delaunay, Catmull-Clark split, EMST curve reading (Figueiredo and Miranda Gomes 1995) | Layout rules are our own. Mark Sharp, Crease and Bevel are Blender's |
| `triadq/patch.py`, `region.py` | Remesh a region on a fixed seam | Published: Tutte embedding (Floater 1997), Coons patch, generalized winding number (Jacobson et al. 2013, Van Oosterom and Strackee) | Cited |
| `triadq/segment.py`, `native/src/segment.rs` | Splits a mesh into named parts | Published: minima rule (Hoffman and Singh), Shamir 2008 survey, shape diameter (Shapira), watershed (Mangan and Whitaker) | Merge tree and settling are our own |
| `triadq/special.py`, `triadq/parts.py` | Wires, fins, cloth, turned parts, repeated pieces | Our design | |
| `triadq/repair.py`, `triadq/checks.py`, `native/src/near.rs` | Finds and clears faces that cross | Our design; `triangles_cross` ported from Blender | Port credited in the file and in THIRD_PARTY_NOTICES |
| `symmetry.py` | Finds a mirror plane, solves one half, mirrors it | Our design | |
| `flowstroke.py`, `flowedit.py`, `guides.py`, `loops.py` | Drawn strokes, rings and edge loops the quads follow | Our design | |
| `triadq/rings.py`, `faceloops.py` (Ring Round This) | Rebuilds the faces round an eye, a mouth or a hole as a filled rim, closed rings and one joining band | Published pieces: Coons 1967 patch for the rim's fill; the band's pairing is a cheapest path through a table, as in Fuchs, Kedem and Uselton 1977 (joining two outlines) | Rim reading (bulge over a fitted bowl), ring counts and corner-quad placement are our own |
| `triadq/strips.py`, `faceloops.py` (Face Loops, Fix Every Strip) | Finds the strips of quads of a result and slides, straightens, evens, adds or removes one, or straightens them all within the camera check | Published: Taubin 1995 (umbrella smoothing of the strip's middle line) | Moving a strip only along the rung lines its quads already run on, the reach-weighted spread to its neighbours and the split / merge refusals are our own |
| `triadq/texsets.py` | Groups faces into texture sets or engine-sized pieces | Our design | Plain longest-axis cut |
| `bridge.py`, `atlasq.py`, `rows_worker.py`, `multires.py`, `blockout.py` | Drive Blender's own QuadriFlow, Voxel Remesh, Decimate and Multires; Blockout fits simple solids | Our design around Blender operators | QuadriFlow is reached only through `bpy.ops.object.quadriflow_remesh` |
| `agentcmd.py`, `agent_mailbox.py`, `skills/curiomesh-agent/scripts/cm_do.py` (agent command layer) | One command name plus JSON arguments in, one JSON reply with a coded error out; a folder mailbox (inbox and outbox files) for a live window or a headless Blender | Our design (bench/tools_design/AGENT_TOOLS.md) | Argument checks are a small subset of JSON Schema; files are written to a temp name and renamed. No sockets |
| `agentops_run.py`, `agentops_tools.py`, `agentops_parts.py` (agent commands) | The panel's own actions (Auto, sizes, Tidy Up, Tidy Tools, Face Loops, Flow Editor, Auto Fix, parts, issues, History, export) as commands | Our design | Each command calls the same function its button calls (`tidylive.work_out`, `fix_select.run_fix`, `segment_goals.redo_parts`, `issue_panel` plans); settings and ranges are read from Blender's own property descriptions |
| `partpack.py`, `partload.py`, `parts_export.py`, `parts_import.py` (Send to Agent, Bring Back) | Writes marked parts with their border, source and neighbours as OBJ files plus a manifest; reads them back, checks the border, holes, shape and look, and keeps them as one History step | Our design (docs/PARTS_EXCHANGE.md) | Wavefront OBJ is the standard text format, read by our own reader. The checks reuse the part swap's camera check and Most Detail Loss |
| `skills/curiomesh-agent/scripts/curiomesh_mcp.py` | Offers the commands to an agent over the Model Context Protocol (JSON-RPC 2.0 on stdio) | Public protocol specs (JSON-RPC 2.0, Model Context Protocol) | Standard library only; talks to the mailbox folder |
| `icons/` | Panel icons | Drawn by our own script, CC0 | See `icons/LICENSE.txt` |

Names in the panel are plain descriptions of what each control does. A few are standard Blender or modelling terms
(Mark Sharp, Crease, Holding Edges, Multires, LOD), used in their ordinary meaning.

When a new algorithm is added, add a row here and cite its paper in the module.
<!-- main-only: uv begin -->

## UV system (GitHub build only)

Rows for the UV features: seam keeping, New UVs, packing and bake-safe UVs. Each package section lists its sources, what is ours and its starting constants. No code, UI or names were copied from other tools.

### K0

Scaffold only (stubs, marked binding blocks, adapter, layer read/write helpers, hash check). No published method and
no third-party code or UI: nothing to cite, no thresholds. The `UvJob` worker-thread handle follows the add-on's own
existing pattern (a daemon thread polled by the caller). Row for G1: "UV scaffold: our own plumbing, no outside source".
`turbo.py` UV bindings: wrappers only, no method.

### K4

Visibility (`native/src/uvsight.rs`, `triadq/uvsight.py`). Depth-buffer visibility, a standard technique: orthographic
depth pictures, one sample per pixel center, nearest surface wins; a triangle is drawn by edge functions over its pixel
rows (textbook barycentric rasterisation). No third-party code or UI was read or used; the six-view picture code of
`geom.rs` is our own and the new drawing routine is written fresh for arbitrary directions.
- Seen per face = hits / expected hits, `expected = area * max(0, n.d) / pixel^2` (our measure). Idea of weighting seams
  by what the camera sees: Sheffer and Hart 2002, "Seamster: inconspicuous low-distortion texture seam layout" (cited
  in SPEC 6.3; the pixel counts and the formulas here are ours).
- View directions beyond the 26 cube directions: golden-angle spiral on the sphere (Saff and Kuijlaars 1997,
  "Distributing many points on a sphere"; Vogel 1979 spiral).
- Edge pixels: scan of the right and lower neighbour of every pixel; each crossing counts `1 / (|cos a| + |sin a|)` so
  the total is the edge's length in pixels (our derivation: a line of length l crosses l(|cos a| + |sin a|) such pairs).
- Near-equal-depth rule (`DEPTH_PIXELS` = 2) is the add-on's own rule from `triadq/views.py`.
- Constants: `DEPTH_PIXELS` 2.0 (own rule), `PIXELS_MIN` 512 and `PIXELS_PER_EDGE` 4 (SPEC 5), `SUBPIXEL_HITS` 1.0 and
  `SMOOTH_ROUNDS` 2 (SPEC 5), `UNDERSIDE_WEIGHT` 0.3 (SPEC 5 start value, P0 tunes), `FRAME_PAD` 1.02 (margin so
  silhouettes stay inside the picture), `MAX_PICTURE_BYTES` 4 GiB (memory guard only: refuses what cannot fit).
Row for G1: "UV visibility: depth-buffer visibility from many views; seen share per face and inside/outline pixel length per edge;
ideas after Sheffer and Hart 2002 and Saff and Kuijlaars 1997; all code ours".

### S1c
CLI and settings plumbing only (flag parsing, JSON keys, report block). No published method; flag and key names are ours (SPEC 10.3, 9.1 card names).

### K2

QEM seams (`native/src/uvqem.rs`, call sites in both branches' `native/src/qem.rs`). No third-party code or UI read or
used; the reduction itself is the existing own `qem.rs` (Garland and Heckbert 1997 quadrics, already cited there).
- Half-edge collapses (the kept vertex keeps one endpoint's place, so a seam vertex keeps its source per-side UVs):
  Hoppe 1996, "Progressive Meshes".
- Seam-aware collapse with a UV-side check (an along-seam collapse must keep the seam's image in UV on both sides):
  Liu, Ferguson, Jacobson, Gingold 2017, "Seamless: seam erasure and seam-aware decoupling of shape from mesh resolution".
  Our form: the check is the largest distance of the replaced source UV vertices to the new segment, on both sides
  (distance to a segment is convex, so this bounds the polyline both ways); the paper's own conditions were not copied.
- Our rules: a seam vertex goes only along its own chain; an off-seam vertex may go onto a seam vertex at its place;
  seam corners (seam valence other than 2, open border, K1 chain ends) never go; a closed loop with no corner gets 3
  anchors spread evenly by 3D length; seam edges get the existing border plane quadric from both sides.
- Constants: `SEAM_PX` 0.5 (SPEC 4.3, pixels at `TEXTURE_PX_DEFAULT` 1024), `LOOP_KEEP` 3 (SPEC 4.3).
Row for G1: "QEM seams: half-edge collapses along UV seam chains with a UV-side check, after Hoppe 1996 and Liu et al.
2017; corner and loop rules and the check's form ours; all code ours".

### S3
- `standalone/src/tangents.rs`: tangent frames after Mikkelsen, "Simulation of Wrinkled Surfaces Revisited" (MSc thesis,
  University of Copenhagen, 2008), written from the thesis text only (never its reference code): per-triangle dP/du, angle
  weighted merge of corners that share vertex + UV + orientation, Gram-Schmidt, bitangent = sign x cross(n, t). Row:
  "Tangents: method of Mikkelsen 2008 as described in the thesis; all code ours".
- `standalone/src/bake.rs` fill: multi-source breadth-first search from filled texels (plain textbook BFS), each empty texel
  copies its nearest filled texel, never an average. Row: "Island-aware fill: BFS nearest-texel copy; ours".
- `standalone/src/bake.rs` gap: gap = 2^(k+1), k = floor(log2(size/256)) (SPEC 8.2, our rule). Seam step check (SPEC 8.5):
  both sides of every cut sampled bilinear at mip 0 and mip k, normal angle and colour step, length-weighted median/p95,
  worst three chains; our design. UDIM tile numbering 1001 + u + 10 v is the public convention for tile file names.
- Matched cuts % (SPEC 8.4, Ray et al. 2010; Purnomo et al. 2004) is not part of S3; it belongs to the cut/pack packages.

### P0

Bench only (`bench/uvplan/uvstats.py`, `run_p0.py`, `report_p0.py`, `make_extra.py`, `bench/assets/uvm.py`); nothing ships in the add-on, standalone or site build. No third-party code or UI was read or used; the neutral clean-room notes were not needed for this package.

| Piece | Origin | Notes |
|---|---|---|
| Stretch numbers (L2, worst 1 %, per-chart normalisation, Angle p90) | Published: Sander, Snyder, Gortler, Hoppe 2001 "Texture Mapping Progressive Meshes" (L2 stretch); per-chart scaling and p90 of sigma1/sigma2 are our own bookkeeping | Singular values from the 2x2 metric of the 3D-to-UV map |
| Closest point on a triangle (drift, owner sets, flips, split cost) | Textbook geometry: Ericson 2005 "Real-Time Collision Detection" | Written fresh in NumPy; the KD-tree is SciPy's `cKDTree` |
| Straddle (owner sets by side region), texture drift, flipped, straddlers by kind, split cost c | Our design (SPEC 9.1, 4.2) | Thresholds (eps, 2 px per-side match, 0.25 edge slack) are bench starting values |
| Seen faces / seen cut px | Our bench approximation: painter's algorithm face-id raster in 14 orthographic views | K4's `uvsight` is the real thing; this only scores baselines |
| Cost model `c_h = a * perimeter / edge_est + b` | Our design; least squares over source islands of Keep UV Seams on versus off | Fit in `cost_model_fit.json` |
| Smart Project baseline | Blender's own operator `bpy.ops.uv.smart_project`, called as a black box, defaults (66 deg, margin 0.003) | Used only as the comparison baseline; no code copied |
| Test models | Poly Haven CC0 geometry already on the PC; two derived by editing the glTF (UVs removed; synthetic TEXCOORD_1) | Not for the repo's shipped files; `.npz` exports are CC0 bench models only |

### K5

Solver and flatten (`native/src/uvflat.rs`, `triadq/uvflat.py`). No third-party code or UI read or used; no solver
crate (faer not needed: the own factor met the gate). Every method below is written fresh from the papers.
- Fill-reducing order: nested dissection by BFS level structures (George 1973, "Nested dissection of a regular finite
  element mesh"; George and Liu 1981, book "Computer Solution of Large Sparse Positive Definite Systems"). The separator
  is the smallest level within the middle 30-70 % (our choice); approximate minimum degree is not used.
- Direct solve: up-looking simplicial LDLᵀ over the elimination tree (Liu 1990, "The role of elimination trees in sparse
  factorization"; Davis 2005, "Algorithm 849: a concise sparse Cholesky factorization package"). Factored once per chart;
  each ARAP step is two triangular solves for u and v together.
- Iterative path above a memory threshold: conjugate gradients (Hestenes and Stiefel 1952) with IC(0) (Meijerink and
  van der Vorst 1977), Jacobi on a non-positive pivot. Stop rule ours: the moves still to come (step move times
  rho/(1-rho), rho from the residual over 10 iterations) under 1/50 texel, or a stall (< 1 % residual drop in 50
  iterations, reported). No iteration cap.
- Exact grid UVs of a pole-free quad chart: integer grid walk (after Kälberer, Nieser and Polthier 2007, "QuadCover";
  Bommes, Zimmer and Kobbelt 2009, "Mixed-integer quadrangulation"); strip lengths = mean 3D length per column/row (ours).
- Quad ARAP: local/global method of Liu, Zhang, Xu, Gotsman and Gortler 2008, "A local/global approach to mesh
  parameterization". One rotation per polygon; per-side weight = the cotangent weight of the rectangle with that side
  and the mean of its two neighbours (ours); triangles use cotangent weights (Pinkall and Polthier 1993) clamped >= 0.
- Straight rows On (vertices of a row/column share one unknown): constraint idea after Campen and Kobbelt 2014,
  "Dual strip weaving"; Campen, Bommes and Kobbelt 2015, "Quantized global parametrization". Soft rows: label-free
  second-difference penalty along quad loops, weight raised until the p95 bend < 2 deg (ours).
- Injective step: largest flip-free step from the per-triangle quadratic (Smith and Schaefer 2015, "Bijective
  parameterization with free boundaries"); quads tested on both diagonals; 0.9 of it taken (ours).
- LSCM (Lévy, Petitjean, Ray and Maillot 2002) assembled as Dirichlet energy minus twice the UV area (Desbrun, Meyer and
  Alliez 2002); a soft pull toward a start instead of pins (ours, SPEC 6.5); Tutte start (Tutte 1963; Floater 1997).
- Stretch number: singular values of the per-triangle map (Sander, Snyder, Gortler and Hoppe 2001), worst
  max(s1, 1/s2) per face after scaling UV area to surface area, 99th percentile over faces (our summary).
- Constants: `TEXTURE_PX` 1024 and `STOP_TEXEL_SHARE` 0.02 (SPEC 3), `STALL_WINDOW` 50 / `STALL_DROP` 0.01 (SPEC 3),
  `RATE_WINDOW` 10 (ours), `ARAP_CHANGE` 0.01 (SPEC 6.4), `STEP_SHARE` 0.9 (ours), `PULL_SHARE` 0.001, `SOFT_START` 0.1,
  `SOFT_GROW` 4 (ours), `BEND_P95_DEG` 2 (SPEC 6.5), `FACTOR_MAX_BYTES` 2e9 (memory guard; turbo passes 1/4 of free RAM),
  `ND_LEAF` 64 (ours).
Row for G1: "UV flatten: own sparse LDLᵀ with nested dissection, IC(0)/Jacobi CG above a memory threshold, exact grid
UVs, quad ARAP with shared-unknown or soft straight rows, injective steps, LSCM with a soft pull; after George 1973,
Liu 1990, Davis 2005, Meijerink and van der Vorst 1977, Liu et al. 2008, Campen and Kobbelt 2014, Smith and Schaefer
2015, Lévy et al. 2002, Tutte 1963, Floater 1997, Sander et al. 2001; all code ours".

### K7

Packer (`native/src/uvpack.rs`, `triadq/uvpack.py`). No third-party code, UI or option names were read or used; the
clean-room note's avoid-list (fixed integer grid with 10 % growth and binary search, L-strip packer, known-optimal tables,
"cavities") was followed: none of it is here.
- Rectangles: MaxRects best-short-side-fit, Jylänki 2010, "A thousand ways to pack the bin" (a published method; free-list
  split and prune written from the paper's description; box orders, the gap rule and the tie-breaks are ours).
- Rows of free intervals and the bottom-left rule: Chazelle 1983, "The bottom-left bin-packing heuristic: an efficient
  implementation". The conservative per-triangle row raster, the square dilation by the gap (Minkowski sum, textbook
  mathematical morphology), the score "least new page height, then least wasted interval length" and the push-right search
  for the leftmost fit are ours.
- Gap and mip levels: our own derivation. A mip-k texel is a block of 2^k texels; two charts are clean when the blocks they
  touch, each grown by one block (bilinear footprint), do not meet. Aligned boxes need 2^k per side, any other 2^(k+1) - 1
  (aligned+aligned 2^(k+1), aligned+loose 3*2^k - 1, loose+loose 2^(k+2) - 2). Checked by rasterising at every reduced size.
- Fill the page: largest uniform scale by a growth/shrink bracket then bisection; lo is always feasible and hi never is, so
  it cannot oscillate and the best feasible layout is kept. Keep going: other box orders and orientation policies, then
  adjacent swaps accepted only when the page then fits a scale larger by `MIN_GAIN`. Both ours; no random tries.
- Constants: `START_FILL` 0.5, `GROW` 1.25, `SCALE_TOL` 1e-5, `MIN_GAIN` 1e-4 (search tolerances, ours), `ORDERS` 5,
  `POLICIES` 4 (counts of the variants tried), `MAX_K` 20 (guards the shift width only).
Row for G1: "UV packing: MaxRects (Jylänki 2010) for boxes; conservative row-interval fill with a bottom-left free-interval
search (Chazelle 1983) for outlines; gap, mip rule, page-fill search and Keep going moves ours".

### K3

UV Check numbers and verdict (`native/src/uvscore.rs`, `triadq/uvscore.py`). Written fresh; `triadq/uvs.py` is not reused.
No third-party code or UI was read or used. P0's `uvstats.py` (our own bench scorer) defines the straddle, drift and flip
steps, which the kernel follows with exact ball queries.
- Stretch and angle: Sander, Snyder, Gortler, Hoppe 2001, "Texture mapping progressive meshes" (L2 stretch from the
  Jacobian of the 3D to UV map). Normalising each chart's UV area to its 3D area (the chart's intended scale) is ours.
- Closest point on a triangle, barycentric: Ericson 2005, "Real-Time Collision Detection" (textbook geometry).
- Overlap and space used: pixel-centre rasterisation with a tie rule that gives a centre lying exactly on a shared edge
  to one triangle only (Pineda 1988, "A parallel algorithm for polygon rasterization"; the sorted-endpoint edge
  evaluation that makes the two triangles' values exact opposites is ours). Pages are (material, integer tile).
- Faces across seams, texture drift (minimum over source points within nearest distance plus local surface error), seam
  fit (distance of result seam samples to the source seam's UV image, after the Liu, Ferguson, Jacobson, Gingold 2017
  condition cited in SPEC 4.3), seams in sight, straight rows and the verdict table: our definitions (SPEC 9.1, 9.2).
- Constants: `UV_TOL_PX` 0.01, `OWNER_UV_PX` 2, `DRIFT_EDGE_SLACK` 0.25 (SPEC 9.1, P0), `EPS_DIAG` 1e-4 and `EPS_EDGE`
  0.05 (P0's owner ball), `SEEN_MIN` 0.5 (a face seen at least half counts as seen), `ROW_LINE_TEXEL` 0.25 (SPEC 9.1),
  `STRETCH_BAD` 4 (SPEC 9.1 "area share > 4x"), `DRIFT_BAD_PX` 8, `WELD` 1e-6, `GRID_EDGES` 2 / `GRID_MAX_DIM` 128 (query
  grid size, speed only), `RASTER_MAX_PX` 8192 (memory guard only: one page is that squared bytes).
Row for G1: "UV Check: stretch after Sander et al. 2001, textbook closest-point and rasterisation rules, side-region straddle
and drift definitions ours; verdict thresholds from SPEC 9.2 (tuned by P1)".

### K1

Seam kernel (`native/src/uvseam.rs`, `triadq/uvseams.py`). No third-party code or UI was read or used; only the neutral
notes in `bench/cleanroom/` are allowed reading and none was needed for this piece.
- Closest point on a triangle with barycentric weights: textbook (Ericson 2005, "Real-Time Collision Detection", 5.1.5).
- Fold fix: a convex-combination (harmonic) fill by Gauss-Seidel sweeps with the spot's border pinned, after Tutte 1963
  ("How to draw a graph") and Floater 1997 ("Parametrization and smooth approximation of surface triangulations"); our
  form: inverse-length weights, a node per (vertex, side region) split at self-seams, grow by one ring while a spot
  still folds, put a spot back when it did not get better, leave out folds whose triangle sits wholly on the region
  border. Stop rule = SPEC 3 (1/50 texel, stall after 50 sweeps without a 1 % drop).
- Ours: islands by exact weld + per-corner UV equality, chains between seam corners (valence, open border, island or
  material pair change), the four chain classes (Ignore / Shift / Soft / Hard) and side regions; the Hard triage with
  the face-cost model `a * perimeter / edge_est + b` (SPEC 4.2); the side-aware lookup: area-majority reference corner,
  a walk that never crosses a Soft or Hard seam within two face diameters, per-corner whole-tile offsets on Shift
  chains, plane slide for corners out of reach, and a lookup stored as source corners + affine weights so every UV
  layer reads the same places.
- Constants: `UV_TOL_PX` 0.01, `IGNORE_PX` 1, `SHIFT_TOL_PX` 1, `HARD_ACROSS_EDGES` 3, `COST_A` 1 / `COST_B` 4,
  `HARD_BUDGET_SHARE` 0.5 (all SPEC 4.1 / 4.2 start values, P1 tunes), `WALK_REACH` 2 (SPEC 4.8 "about 2 face
  diameters"), `SLIDE_EDGE` 0.02 (ours: 0.25 clamped corners onto a seam and folded more faces on the bench),
  `CENTRE_WEIGHT` 0.5 (ours), `STOP_TEXEL_SHARE` 0.02 / `STALL_WINDOW` 50 / `STALL_DROP` 0.01 (SPEC 3).
Row for G1: "UV seams and per-side transfer: chain classes, side regions, Hard triage cost model and the side-aware
walk lookup ours; textbook closest point (Ericson 2005); fold fix as a harmonic fill after Tutte 1963 / Floater 1997".

### K6

Row Charts (`native/src/uvchart.rs`, `triadq/uvchart.py`). No third-party code or UI read or used; only the neutral
clean-room notes were available and none was needed. Every method below is written fresh from the papers.
- Patches by traces that all advance one edge per step from every irregular point and stop at a border, a forced line
  or another trace: Eppstein, Goodrich, Kim, Tamstorf 2008, "Motorcycle graphs: canonical quad mesh partitioning".
  Ours: turning vertices and ends of forced lines (hard edges, part and material borders, kept seams, Cut here) count
  as irregular so patches stay rectangles; head-on and same-step meetings stop both traces.
- Topology check: Euler characteristic over corner wedges (V' - E' + F) per patch and per merged chart (a chart is a
  disk iff it is 1). Annulus: the cheapest straight crossing row (ours, keeps the patch a grid); anything else:
  tree-cotree (Eppstein 2003, "Dynamic generators of topologically embedded graphs") with a maximum-cost dual spanning
  tree kept glued and dangling branches glued back (the cost rule ours).
- Cut cost: `length_px`, or `seen_px + length_px / L` with seams out of sight (idea of visibility-weighted seams:
  Sheffer, Hart 2002, "Seamster"; the two measured terms and L are ours, SPEC 6.3).
- Merge tree: greedy pair merges by cost removed per stretch added, stretch-free merges first (Garland, Willmott,
  Heckbert 2001, "Hierarchical face clustering on polygonal surfaces"; bounded distortion after Sorkine, Cohen-Or,
  Goldenthal, Lischinski 2002, "Bounded-distortion piecewise mesh parameterization"). Ours: grid unions measured from
  the glued exact grid (no solve); a grid only while it is within 1 % of ARAP (`GRID_SLACK`); pole unions in a lazy
  queue, measured by a rigid 2D fit (Kabsch's least-squares rotation, 2D case) and an ARAP solve on a 2-ring band only
  (`uvflat::quick_stretch`); a pair whose chart grew is queued again under its last measured key; segments glued one
  by one and kept only while the chart stays a disk (ring closure keeps one cut); `level(x)` applies every recorded
  merge whose stretch fits, skipping the others.
- Keep going: one discrete move per pass (cut from the worst-stretch vertex along patch borders to the chart border by
  Dijkstra, or close the cut segment with the most seen px that keeps Stretch allowed), accepted only when
  `stretch excess + seen cut px / sqrt(area px)` drops. Idea of discrete cut moves: Li, Kaufman, Kim,
  Sorkine-Hornung, Jacobson, Sheffer 2018, "OptCuts"; moves and score ours (no fracture, no dual weights).
- Constants: `HIDDEN_L` 20 (SPEC 6.3 start), `STRETCH_ALLOWED_PCT` 10 (SPEC 6.4 start), `BUILD_STRETCH` 1.0 (merges are
  recorded up to stretch 2; a setting, Again with more records more), `BAND_RINGS` 2, `PX_PER_DIAGONAL` 1024 (pixels
  when the caller gives none), `GRID_SLACK` 0.01, `SAME_STRETCH` 1e-9 (rounding noise only).
Row for G1: "Row Charts: motorcycle-graph patches of our own quads with forced lines, Euler/tree-cotree disk cuts,
visibility-weighted cut cost and a greedy merge tree with skipping levels; after Eppstein et al. 2008, Eppstein 2003,
Sheffer and Hart 2002, Garland et al. 2001, Sorkine et al. 2002, Li et al. 2018; all code ours".

### K1b

Seam snap and split leftovers (`native/src/uvsnap.rs`, `triadq/uvsnap.py`). No third-party code or UI was read or used.
Re-checked against `bench/cleanroom/UV_EMPYREAN_SEAMMEND.md` (the only allowed reading): that tool's public docs
describe a grid stamped over each island and an optional strip of faces inserted along island boundaries, with small
faces collapsed. We do neither: no strip or row of faces is ever added along a border and nothing is collapsed; only the
single result faces that the remesh, the alignment and the snap could not fit are cut, one cut per chain, along the
source polyline, under the user's face allowance spent by damage. No names from that note are used.
- Snap path: Dijkstra's shortest paths (Dijkstra 1959) over result edges inside a corridor around the chain. Ours: the
  corridor (`SNAP_REACH` x each vertex's own mean edge), the edge cost `length x (1 + k x distance / edge)`, the
  acceptance rule (mean distance < 0.35 edge and at most one turn per 5 path edges, a turn = more than 45 deg off the
  chain direction or running back), the move guard (never past `max_move`, never breaking the add-on's quad test or
  flipping a face's Newell normal; a refused vertex is held), closed chains snapped as two halves, a tangential relax of
  the rings around snapped vertices under the same guard.
- Closest point on a segment, a segment crossing a plane, closest point on a triangle (Ericson 2005, "Real-Time
  Collision Detection", 5.1.2, 5.3.1, 5.1.5); Newell's polygon normal (Newell 1972; Sutherland, Sproull, Schumacker
  1974). A uniform grid for neighbour queries (textbook).
- Split, ours: one crossing per (result edge, chain), found where the chain polyline crosses the plane through the edge
  along its faces' normal (no projection onto a face, so warped quads work), shared by both faces; a crossing nearer
  than 0.1 of the edge to an end reuses that vertex as a seam vertex (also where no cut is needed) and never moves it;
  crossings whose point is clearly nearer another result face are dropped (the far side of a thin part); chords
  between consecutive points of a chain whose midpoint is inside the face; a fan from a seam corner inside a face;
  successive cuts for several chains; pieces turned into quads and triangles by diagonals only; split triangles paired
  with a same-side neighbour when the quad passes the quad test; runs of faces joined through a new vertex are spent
  whole (no T-junctions), most damage first (area x seen x gap px), within the allowance; faces holding a whole island
  are marked for baking, faces it cannot cut are counted.
- Per-side UVs straight from the chain polylines at the crossing parameter (side A = left of the chain direction) are
  an output (`corner_uv`). Measured on P0's data they are worse than K1's side-aware per-side transfer run on the new
  mesh (concrete_road_barrier straddle 3.97 % vs 2.14 %, drift p99 87 px vs 1.0 px), so the bench and the callers
  should use the transfer; the chain UVs stay for the exact synthetic case and for S1b/A1 to look at.
- Constants: `SNAP_REACH` 1 (SPEC 4.5 "1 local edge"), `SNAP_COST_K` 4, `SNAP_MEAN_DIST` 0.35, `SNAP_TURN_EVERY` 5
  (SPEC 4.5), `SNAP_TURN_COS` cos 45 deg (ours), `ON_EDGE_SHARE` 1e-6 (ours), `RELAX_WEIGHT` 0.5 and `RELAX_RINGS` 1
  (ours; rings is a setting), `SPLIT_GAP_PX` 1 (SPEC 4.7), `CROSS_REACH` 0.5 (ours; 1.0 measured worse on
  fire_hydrant: 14.38 % vs 14.06 % straddle before the thin-part rule), `REUSE_EDGE` 0.1 (SPEC 4.7 "about 0.1 edge"),
  `NEARER_SHARE` 0.5 (ours; sparing faces at the edge's ends and an absolute 0.05-edge floor both measured worse),
  `WARP_COSINE` 0.5 (the add-on's 60 deg quad warp limit).
Row for G1: "UV seam snap and split leftovers: corridor path rule, staircase test, move guard, per-edge shared
crossings with vertex reuse, fan and successive cuts, damage-ordered face allowance ours; Dijkstra 1959; textbook
geometry (Ericson 2005, Newell 1972); no boundary strips (differs from the strip insertion in Empyrean's public docs)".

### A1c

File: `triadq/uvclassic.py`.
- Per-face pairing fallback (drop only the pairs whose two triangles disagree on the shared edge UVs) and island-aware transfer are our own design; the lookup is K1 (uvseams, SPEC 4.8). Pairing on the dual graph: Blossom-Quad, Remacle et al. 2012 (already in ORIGINALITY). No other tool consulted.

### A1

Add-on Keep (Phase A): `triadq/uvkeep.py`, `uv_hooks.py`, `uv_layers.py` (apply a candidate). Orchestration of the K1 and
K1b cores; no new published method and no third-party code or UI was read or used.
- Kinds: connected components of source faces joined across every chain that is not Hard (union of islands by label
  propagation; textbook connected components). The Hard/Soft choice and its order are SPEC 4.2 (cost model `a`, `b`
  fitted by P0, starting 1 and 4).
- Reconciliation of a Hard chain whose two sides ended up in one kind: our rule. The pair's other chains join the Hard
  set while the Hard estimate still fits half the allowance, else the Hard chain turns Soft. Both outcomes are counted
  and shown on the card, never dropped silently. Repeats until no conflict is left (each chain is promoted at most once
  and demoted at most once, so it ends).
- Over the 16-bit kind limit (`MOST_CHARTS`, Blender's own limit): the least damaging Hard chains (3D length x gap px)
  turn Soft, found by bisection on the count, and the count is reported.
- Face budget for splits: `allowance x faces_off_est - Hard estimate - extra faces already spent`, with
  `faces_off_est = faces - Hard estimate`. "any" = no budget. Our accounting, written in SPEC 4.2.
- Snap limit starting value: half the vertex's mean edge (`MOVE_SHARE` 0.5, from K1b's bench). A caller may give a
  per-vertex limit from a camera check instead.
- Apply writes the fit that matches the result by hash (SHA-256 of float32 positions and UVs, faces, materials, layer
  names); with none it writes the per-side transfer only. Nothing from other tools.

### S1a

Standalone Seam Fit, remesh-time alignment (`standalone/src/retopo.rs`, `rows.rs`, `tidy.rs`, bench `run_s1a.py`,
`standalone/examples/seamfit_gate.rs`). No third-party code or UI was read or used.
- Seams kept as edges by the reduction: `qem::run_seams` (K2's row; Hoppe 1996, Liu, Ferguson, Jacobson, Gingold 2017).
- Faces tagged by side region through pairing, closing, evening out and Tidy Up, so none of them joins faces across a
  seam; seam edges as pairing guide edges (the standalone's own drawn-strokes guide mechanism). Our wiring.
- Rows pinned along seam edges like crease lines: field aligned to feature lines (Bommes, Zimmer, Kobbelt 2009,
  "Mixed-integer quadrangulation") and a lattice line through them (Jakob, Tarini, Panozzo, Sorkine-Hornung 2015,
  "Instant Field-Aligned Meshes", position constraint). Rows kept per part only when a quick straddle count (nearest
  source face under points a quarter in from each corner) is no worse. Our rule.
- Parallel-chain rule (trim strips): of two chains sharing an island closer than `PARALLEL_EDGE` = 0.75 result edges
  (mean distance of the shorter to the other), only the one with more texture at risk (3D length x largest gap px) is
  pinned. Our rule; threshold from SPEC 4.4.
- End snap with `max_move` = 0.5 x the vertex's mean edge (`SNAP_MOVE_EDGE`, K1b's bench value) and the whole-view check;
  S1b replaces it with the standalone judge's own `max_move`.

### S1a-pair

Seam Fit face cost fix (`standalone/src/retopo.rs` `merge_leftovers`). No third-party code or UI was read or used.
- Leftover merge: two triangles pairing left over plus one quad they touch, joined edge to edge into a hexagon with no
  guide (seam) edge inside and one side tag, are cut again into two quads (best of the hexagon's three cuts by worst
  scaled Jacobian, both quads valid within the 60 degree warp limit, the cut not already an edge elsewhere). Runs while a
  pass joins any. Our design; a local re-cut of a paired mesh, no published method used.

### A2

Add-on New UVs and bake-safe UVs (`triadq/uvnew.py`, `triadq/uvbake.py`, five marked lines in `textures.py`). Pipeline glue
over the K4-K7 cores; no third-party code or UI was read or used, and no other tool's workflow is copied.
- Chart pipeline (visibility, patches and merge tree, flatten, pack): the methods are the cores' (K4, K5, K6, K7 rows);
  what is new here is only their order and the checks between them.
- Box per chart by rotating calipers: Toussaint 1983, "Solving geometric problems with the rotating calipers" (smallest-area
  rectangle on the convex hull's edges); hull by Andrew 1979 monotone chain. On charts take the rows axis from their grid
  labels instead (circular mean of the u edge angles modulo 90 degrees, our rule), so rows stay on u or v.
- Hidden-parts scale per chart: area-weighted mean of max(floor, seen) per face, no square root (SPEC 6.7, ours; weighted
  metric idea after Sander, Snyder, Gortler, Hoppe 2001 and Sander, Gortler, Snyder, Hoppe 2002).
- Whole texels per box side and the mip-clean gap: Carr, Hoberock, Crane, Hart 2006 (rectangular multi-chart geometry
  images); the gap rule itself is SPEC 8.2 / K7.
- Self-overlap guard (ours): after the solver (flip-free, but a cap with a pole of valence 5 or more can still lie over
  itself), each chart of 100 faces or more is rasterised alone at 256 px; one that overlaps or flips is replaced by the
  charts of a lower Stretch allowed that lie inside it (half, quarter, ...), last by its grid patches (exact grids, Kalberer,
  Nieser, Polthier 2007 style integer layout built by K5's `grid_layout`). No published method is copied.
- Sharpen cuts: dihedral angle over the shading angle makes a cut sharp (plain geometry). Seam step check: both sides of every
  cut sampled bilinearly in the baked maps, normal angle and 8-bit colour step, median and p95 weighted by cut length (ours),
  at mip 0 and mip k (box-filter reduction).
- Image history for Ctrl+Z: a scene-property step counter that Blender's own undo restores, and per-image 8-bit copies kept
  per step (ours; no outside source).
- Constants: `SELF_CHECK_FACES` 100, `SELF_CHECK_PX` 256, `SELF_CHECK_FILL` 0.9, `SELF_OVERLAP_PCT` 0.1 (our starting values,
  P1 tunes); Stretch allowed default 10 (P0's start; sweeps at 10, 25, 50, 100 on the 13 bench models change the chart count by under 25 %
  once the self-overlap guard has run); `SAMPLES_PER_EDGE` 5; `SHADING_ANGLE` 30 degrees.
Row for G1: "Add-on New UVs and bake-safe UVs: pipeline order and checks ours; calipers box after Toussaint 1983; texel snapping after
Carr et al. 2006; weighted density after Sander et al. 2001/2002; self-overlap guard, seam step check and image history ours".

### K7big

Packer speed and tightness on large sets (`native/src/uvpack.rs`, `triadq/uvpack.py`). No third-party code or UI was read.
- Bottom-left placement rule (the free place whose box ends lowest, then leftmost) as a second rule beside best short side
  fit: Baker, Coffman, Rivest 1980, "Orthogonal packings in two dimensions". Both rules are tried, the better layout kept (ours).
- Dropping free rectangles that no remaining box can use (a box's shortest side plus the gap on both sides): plain
  bookkeeping on the MaxRects list (Jylanki 2010), ours; the layout is unchanged, only faster.

### K6poles

Charts that no longer lie over themselves around poles, and a faster merge tree (`native/src/uvchart.rs`,
`triadq/uvchart.py`), plus the picture hand-over of K4 (`uv_fast.py`, `triadq/uvsight.py`, two accessors in
`native/src/lib.rs`). No third-party code or UI was read.
- Self-overlap refusal in the merge tree: a flip-free disk layout is one-to-one exactly when its border does not cross
  itself (Floater 2003, "One-to-one piecewise linear mappings over triangulations"). Testing only the border sides that
  moved, against a cell grid no finer than the longest side, and remembering a refused pair until its smaller chart
  changes: ours.
- Merge rounds of rising stretch (Stretch allowed halved down to 1 %, then doubled up to the build stretch): ours, on
  top of the greedy pair merges after Garland, Willmott, Heckbert 2001.
- Measuring candidates ahead on worker threads from a separate not-yet-measured heap: plain bookkeeping, ours; the
  merges made are the same as measuring one at a time.
- Tried and dropped: gluing fewer shared segments when a union closes around a pole ("cone cut", ours): 19 of 1505
  refused unions on the lion model became merges, the chart count did not drop, and the build took 2.4x longer.
- Picture hand-over in slices of 1M cells (i32 face ids, f32 depth, copied straight from the kernel's arrays): ours.
<!-- main-only: uv end -->
