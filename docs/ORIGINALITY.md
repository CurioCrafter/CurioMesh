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
| `triadq/quadreduce.py`, `native/src/qem.rs` | Fewer quads / fewer triangles with least shape change | Published: Garland and Heckbert 1997 quadrics | `qem.rs` uses the thresholds of Blender's Decimate on purpose and names them in comments |
| `triadq/tidy.py` | Edge turns, collapses, doublets on a finished quad mesh | Published: Tarini et al. 2010 | Cited |
| `triadq/chords.py`, `native/src/chords.rs` (Redraw Messy Quads) | Redraws a messy patch as a plain grid via a closed walk on squared paper, judged by the camera | Our design | Daniels et al. 2008 cited as the approach not taken |
| `triadq/refill.py` | Replaces a pole-heavy patch with one grid | Published idea: Takayama et al. 2014, Marcias et al. 2015 | Cited |
| `triadq/organize.py`, `organize.py` (Organize slider) | Cleans poles, then settles quads like damped cloth on the surface | Published: Tarini et al. 2010, Mueller et al. 2005 (shape matching), Mueller et al. 2007 (position-based dynamics) | The pulls, guards and slider steps are our own |
| `triadq/trial.py`, `tidylive.py`, `tidytools.py` (Look First) | Shows a tool's change on the mesh before it is kept, with one ring handle and a numbers card | Our design (bench/tools_design/INTERACTION.md) | The ghost, wash, ring, card and keys are our own; no other tool's preview was looked at |
| `triadq/hardtidy.py` (True Up Edges) | Finds the result's hard-edge chains and lays each on the crease of the original: straight ones on one line, curved ones smoothed and evenly spaced | Published pieces: least-squares planes (principal axis fit), Taubin 1995 (shrink-free curve smoothing) | Reading each crease as where two planes fitted to the original on either side meet, and accepting it only where that line lies on the original, is our own. Set Linear / Straight-style tools work on a picked chain against itself; this finds the chains and matches them to the original |
| `triadq/hardtidy.py` (Press Panels Flat) | Finds flat panels and presses each onto the plane of the original under it; shared corners go to the point nearest all their planes | Published pieces: least-squares plane fit, regularized least squares for the shared corners | Panel growing follows our own Clean Panels (`flatquad`); a corner goes onto a plane only where the original under it lies on that plane. No pick-a-plane step and no Flatten / Make Planar behaviour |
| `triadq/tidytools.py` (Smooth Corners) | Slides corners toward their neighbours' middle, then back onto the source | Published: Taubin 1995 (umbrella smoothing), surface projection as in Tarini et al. 2010 | The example tool; the shrink step of Taubin's filter is replaced by the projection |
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
| `triadq/texsets.py` | Groups faces into texture sets or engine-sized pieces | Our design | Plain longest-axis cut |
| `bridge.py`, `atlasq.py`, `rows_worker.py`, `multires.py`, `blockout.py` | Drive Blender's own QuadriFlow, Voxel Remesh, Decimate and Multires; Blockout fits simple solids | Our design around Blender operators | QuadriFlow is reached only through `bpy.ops.object.quadriflow_remesh` |
| `icons/` | Panel icons | Drawn by our own script, CC0 | See `icons/LICENSE.txt` |

Names in the panel are plain descriptions of what each control does. A few are standard Blender or modelling terms
(Mark Sharp, Crease, Holding Edges, Multires, LOD), used in their ordinary meaning.

When a new algorithm is added, add a row here and cite its paper in the module.
