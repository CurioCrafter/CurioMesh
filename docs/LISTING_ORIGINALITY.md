# Originality

CurioMesh is written for this repository. No code, UI, feature names or text were copied or translated from other
retopology tools, commercial or open source. Where an algorithm rests on published math, the module says so and cites
the paper or author: that is how we credit ideas, and the code is our own writing of them. Blender's own operators
(QuadriFlow Remesh, Voxel Remesh, Decimate, Multires, Bevel, Data Transfer) are called, not included.

CurioMesh contains Python code only and no compiled or third-party code. Its bundled data and their sources are listed in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and [PATTERN_SOURCES.md](PATTERN_SOURCES.md).

## Module by module

| Module | What it does | Origin | Notes |
|---|---|---|---|
| `triadq/engine.py` | Pairs triangles into quads by a maximum matching, then closes leftovers | Published: Blossom-Quad (Remacle et al. 2012), Edmonds 1965 | The matching follows the usual textbook form of Edmonds' algorithm |
| `triadq/topology.py` (ParityPath, quad closure) | Turns a triangle/quad mesh into all quads with no T-junctions | Our design; the last-resort split is the Catmull-Clark midpoint split | Routing odd triangles along dual-graph paths is our own scheme |
| `triadq/field.py` | Cross field from curvature | Published: Viertel and Osting 2019, Rusinkiewicz 2004 | Cited in the module |
| `triadq/fieldmesh.py` (Field Rows) | Quads in rows from a direction field and a lattice | Published idea: Jakob, Tarini, Panozzo, Sorkine-Hornung 2015 (Instant Field-Aligned Meshes) | Own NumPy implementation; extraction, conformal cell sizes, plane fitting and row sliding are our own |
| `triadq/quadreduce.py` | Fewer quads / fewer triangles with least shape change | Published: Garland and Heckbert 1997 quadrics | |
| `triadq/tidy.py` | Edge turns, collapses, doublets on a finished quad mesh | Published: Tarini et al. 2010 | Cited |
| `triadq/chords.py` (Redraw Messy Quads) | Redraws a messy patch as a plain grid via a closed walk on squared paper, judged by the camera | Our design | Daniels et al. 2008 cited as the approach not taken |
| `triadq/refill.py` | Replaces a pole-heavy patch with one grid | Published idea: Takayama et al. 2014, Marcias et al. 2015 | Cited |
| `triadq/organize.py`, `organize.py` (Organize slider) | Cleans poles, then settles quads like damped cloth on the surface | Published: Tarini et al. 2010, Mueller et al. 2005 (shape matching), Mueller et al. 2007 (position-based dynamics) | The pulls, guards and slider steps are our own |
| `triadq/patterns.py` | Tidiness score against statistics from hand-made meshes | Our design | Data sources and licences in [PATTERN_SOURCES.md](PATTERN_SOURCES.md) |
| `triadq/loopscore.py` (Loop Quality) | Scores edge loops for straightness, spacing, poles | Our design | Targets are our own reading of artist practice |
| `triadq/flatquad.py` (Flat Quads) | Few faces on flat areas, all of them on detail | Our design | |
| `triadq/fit.py`, `triadq/cases.py` | Fits vertices and shading to the nearest source points; named test cases | Ideas from cav3dw3ll3r, credited in the README | Code is our own |
| `triadq/smoothgrid.py` | Rebuilds domes and rings as concentric rows | Our design | |
| `triadq/hardsurf.py`, `hard_surface.py` | Clean Panels, Even Cylinders, hard-edge finish | Published pieces: Coons 1967 patches, constrained Delaunay, Catmull-Clark split, EMST curve reading (Figueiredo and Miranda Gomes 1995) | Layout rules are our own. Mark Sharp, Crease and Bevel are Blender's |
| `triadq/patch.py`, `region.py` | Remesh a region on a fixed seam | Published: Tutte embedding (Floater 1997), Coons patch, generalized winding number (Jacobson et al. 2013, Van Oosterom and Strackee) | Cited |
| `triadq/segment.py` | Splits a mesh into named parts | Published: minima rule (Hoffman and Singh), Shamir 2008 survey, shape diameter (Shapira), watershed (Mangan and Whitaker) | Merge tree and settling are our own |
| `triadq/special.py`, `triadq/parts.py` | Wires, fins, cloth, turned parts, repeated pieces | Our design | |
| `triadq/repair.py`, `triadq/checks.py` | Finds and clears faces that cross | Our design | |
| `symmetry.py` | Finds a mirror plane, solves one half, mirrors it | Our design | |
| `flowstroke.py`, `flowedit.py`, `guides.py`, `loops.py` | Drawn strokes, rings and edge loops the quads follow | Our design | |
| `triadq/texsets.py` | Groups faces into texture sets or engine-sized pieces | Our design | Plain longest-axis cut |
| `bridge.py`, `atlasq.py`, `rows_worker.py`, `multires.py`, `blockout.py` | Drive Blender's own QuadriFlow, Voxel Remesh, Decimate and Multires; Blockout fits simple solids | Our design around Blender operators | QuadriFlow is reached only through `bpy.ops.object.quadriflow_remesh` |
| `triadq/uvseams.py`, `triadq/uvclassic.py` | UV islands, seams and side regions of your mesh; the per-side UV transfer of Classic Remesh | Published pieces: closest point on a triangle (Ericson 2005), convex-combination fill (Tutte 1963, Floater 1997) | The side rules are our own |
| `triadq/uvsight.py` | What the camera sees of each face and edge | Published idea: Sheffer and Hart 2002 (Seamster); Saff and Kuijlaars 1997 view directions | Depth pictures and pixel counts are our own |
| `triadq/uvchart.py`, `triadq/uvhide.py` | Charts from quad patches, merge tree, seams slid out of sight | Published: Eppstein et al. 2008 (motorcycle graphs), Eppstein 2003 (tree-cotree), Garland, Willmott and Heckbert 2001, Julius, Kraevoy and Sheffer 2005 (D-Charts), Sander et al. 2003, Floater 2003, Boykov and Jolly 2001, Dinic 1970, Toussaint 1983 | Merge order, rules and the slide are our own |
| `triadq/uvflat.py` | Flattens each chart with straight rows | Published: Liu et al. 2008 (local/global ARAP), Smith and Schaefer 2015 (injective steps), Tutte 1963, Floater 1997, Hestenes and Stiefel 1952 (CG) | Straight-rows handling is our own |
| `triadq/uvpack.py`, `triadq/uvnew.py` | Packs charts on the page with a mip-safe gap; the New UVs pipeline | Published: Jylanki 2010 (MaxRects), Baker, Coffman and Rivest 1980, Chazelle 1983, Carr, Hoberock, Crane and Hart 2006, Andrew 1979 | The gap rule and the pipeline are our own |
| `triadq/uvscore.py` | UV Check numbers and verdict | Published: Sander, Snyder, Gortler and Hoppe 2001 (L2 stretch), Pineda 1988 (fill rule) | Thresholds are our own |
| `triadq/uvbake.py`, `triadq/uvtrial.py`, `uv_panel.py`, `uv_draw.py`, `uv_layers.py` | The bake's own unwrap, the UV tools (New UVs, New page layout, Cut here, Bake those spots) and their preview | Our design | |
| `icons/` | Panel icons | Drawn by our own script, CC0 | See `icons/LICENSE.txt` |

Names in the panel are plain descriptions of what each control does. A few are standard Blender or modelling terms
(Mark Sharp, Crease, Holding Edges, Multires, LOD), used in their ordinary meaning.
