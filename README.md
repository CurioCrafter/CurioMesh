# CurioMesh: free auto retopology and quad remesh add-on for Blender

**A free, open source Blender retopology add-on. It turns a dense sculpt, scan or high-poly game asset into a clean all-quad, low-poly mesh.**

[![Download CurioMesh](https://img.shields.io/badge/Download-free%20add--on-2ea44f?style=for-the-badge&logo=blender&logoColor=white)](https://github.com/CurioCrafter/CurioMesh/releases/latest)
[![Star on GitHub](https://img.shields.io/badge/Star-on%20GitHub-24292f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CurioCrafter/CurioMesh/stargazers)

[![Latest release](https://img.shields.io/github/v/release/CurioCrafter/CurioMesh?label=release)](https://github.com/CurioCrafter/CurioMesh/releases/latest)
[![Blender 3.6 to 5.2](https://img.shields.io/badge/Blender-3.6%20to%205.2-orange?logo=blender&logoColor=white)](https://www.blender.org/download/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

![Automatic retopology in Blender: wireframe of an air compressor before and after CurioMesh, 79,042 faces reduced to 15,881 all-quad faces, whole object and close-up](docs/images/old_military_compressor.jpg)

*[Old military compressor](https://polyhaven.com/a/old_military_compressor) (free CC0 model): 79,042 faces to 15,881 quads, 5.6% of the view changed. Wire shown on both sides.*

CurioMesh is an automatic quad retopology add-on for Blender 3.6 to 5.2. Say what the mesh is for, press one button, and it reduces
the model part by part. A camera check compares the result with your original from six sides and tells you how much of the view
changed. Then you redo only the part that still looks wrong. It runs on your PC and uploads nothing.

**Who it is for:** 3D artists, technical artists and game developers who need a lower-poly, all-quad version of a high-poly
model, a sculpt or a photogrammetry scan, and want to fix single parts instead of redoing the whole mesh.

**Try it in four steps:**

1. Download the zip for your Blender version from the [latest release](https://github.com/CurioCrafter/CurioMesh/releases/latest) ([which one?](#download-and-install-the-blender-retopology-add-on)).
2. In Blender, install it and enable **CurioMesh**.
3. Select a mesh and open `3D View > N-panel > CurioMesh`.
4. Press **Auto** in the Guided panel.

Auto picks the Use and settings from your mesh and free memory, says why, and starts. Or follow the numbered steps yourself:
choose what it is for, **Start**, fix parts in **Review Segments**, then **Apply**. Nothing changes in your scene until Apply.

**Honest limits:** quads can look slanted on smooth surfaces, fine ornaments go jagged at high reduction, it does not find edge
loops for faces and joints on its own, and big scans take minutes, sometimes tens of minutes. See [Known limits](#known-limits).

## Contents

[Examples](#examples-before-and-after-wire-on-both-sides) · [Uses](#pick-what-it-is-for) · [Styles](#choose-a-layout-style) · [Multires](#multires-detail-a-clean-cage-with-your-shape-in-the-levels) · [Review Segments](#review-segments-find-the-part-that-is-wrong-and-redo-only-that) · [Bake and UVs](#bake-detail-uvs-and-game-engines) · [Big scans](#big-scans-small-pcs-and-a-cloud-job) · [Without Blender](#curiomesh-without-blender-the-standalone-app-preview) · [Install](#download-and-install-the-blender-retopology-add-on) · [FAQ](#frequently-asked-questions-about-free-blender-retopology) · [Limits](#known-limits) · [All features](#all-features)

## What it does

| Feature | What it does |
|---|---|
| **Auto Segments and the Auto button** | Reduces the whole object to all quads, one loose part at a time, and checks every size from six sides. **Auto** picks the Use and settings for you. |
| **Presets by use** | Low Poly, Game Asset, Animation, Subdivision, Scan Cleanup, Sculpt Base. |
| **Three layout styles** | Keep Detail (fewest faces), Balanced, Clean Rows (even quads in rows). |
| **Reduce To** | Aim at a percentage of your faces or a face count, within about 10%. |
| **Review Segments** | Every part in its own colour, numbers on hover, redo one part with another method, go back to any try. |
| **Multires Detail** | A clean low cage at level 0, with the shape of your mesh back in the Multires levels above it. |
| **Make LOD Set** | Three objects named `_LOD0`, `_LOD1` and `_LOD2`, at percents you choose. |
| **Bake Detail** | Normal map, or normal map and colour, onto the new mesh. Your mesh is only read. |

More below: [all features](#all-features).

## Examples: before and after, wire on both sides

Every example below ran the real add-on on a free [Poly Haven](https://polyhaven.com) model (CC0) with the default Game
Asset settings: Keep Detail, **Detail 6%**. Detail is how much of the view may look different from your mesh. You can
download the same models and check the numbers. Each picture shows the original (left) and the result (right), the whole
object on top and a close-up below.

![Auto retopology of a chess set in Blender: 76,832 faces become 10,110 quads, wireframe before and after, whole object and close-up](docs/images/chess_set.jpg)

*Chess set (33 pieces), [Poly Haven](https://polyhaven.com/a/chess_set), CC0.*

![Quad remesh of an overhead crane in Blender: 89,964 faces become 22,541 quads, wireframe before and after, whole object and close-up](docs/images/overhead_crane.jpg)

*Overhead crane, [Poly Haven](https://polyhaven.com/a/overhead_crane), CC0.*

![Low-poly retopology of a concrete road barrier in Blender: 60,928 faces become 987 quads, wireframe before and after, whole object and close-up](docs/images/concrete_road_barrier.jpg)

*Concrete road barrier, [Poly Haven](https://polyhaven.com/a/concrete_road_barrier), CC0.*

| Model | Original faces | CurioMesh quads | Size | View changed | Wire pictures |
|---|---:|---:|---:|---:|---|
| [Marble bust](https://polyhaven.com/a/marble_bust_01) | 17,456 | 2,508 | 14% | 6.0% | [before and after](docs/images/marble_bust_01.jpg) |
| [Concrete road barrier](https://polyhaven.com/a/concrete_road_barrier) | 60,928 | 987 | 2% | 5.9% | [before and after](docs/images/concrete_road_barrier.jpg) |
| [Old military compressor](https://polyhaven.com/a/old_military_compressor) | 79,042 | 15,881 | 20% | 5.6% | [before and after](docs/images/old_military_compressor.jpg) |
| [Fire hydrant](https://polyhaven.com/a/fire_hydrant) | 86,316 | 26,771 | 31% | 4.9% | [before and after](docs/images/fire_hydrant.jpg) |
| [Treasure chest](https://polyhaven.com/a/treasure_chest) | 103,330 | 10,423 | 10% | 4.8% | [before and after](docs/images/treasure_chest.jpg) |
| [Gothic statue](https://polyhaven.com/a/gothic_statue) | 27,739 | 16,586 | 60% | 5.9% | [before and after](docs/images/gothic_statue.jpg) |
| [Lion head](https://polyhaven.com/a/lion_head) | 47,214 | 7,940 | 17% | 5.9% | [before and after](docs/images/lion_head.jpg) |
| [Horse statue](https://polyhaven.com/a/horse_statue_01) | 22,388 | 5,609 | 25% | 5.9% | [before and after](docs/images/horse_statue_01.jpg) |
| [Chess set (33 pieces)](https://polyhaven.com/a/chess_set) | 76,832 | 10,110 | 13% | 3.6% | [before and after](docs/images/chess_set.jpg) |
| [Spinning wheel](https://polyhaven.com/a/spinning_wheel_01) | 52,023 | 16,644 | 32% | 3.7% | [before and after](docs/images/spinning_wheel_01.jpg) |
| [Sofa (already low poly)](https://polyhaven.com/a/Sofa_01) | 4,101 | 1,571 | 38% | 5.9% | [before and after](docs/images/Sofa_01.jpg) |
| [Drill press](https://polyhaven.com/a/drill_press_01) | 58,511 | 16,809 | 29% | 5.8% | [before and after](docs/images/drill_press_01.jpg) |
| [Overhead crane](https://polyhaven.com/a/overhead_crane) | 89,964 | 22,541 | 25% | 5.4% | [before and after](docs/images/overhead_crane.jpg) |
| [Modular wooden pier](https://polyhaven.com/a/modular_wooden_pier) | 84,780 | 25,906 | 31% | 6.0% | [before and after](docs/images/modular_wooden_pier.jpg) |
| [Moss rocks (6 pieces)](https://polyhaven.com/a/rock_moss_set_01) | 63,127 | 20,425 | 32% | 5.9% | [before and after](docs/images/rock_moss_set_01.jpg) |
| [Namaqualand boulder](https://polyhaven.com/a/namaqualand_boulder_02) | 97,964 | 52,789 | 54% | 6.0% | [before and after](docs/images/namaqualand_boulder_02.jpg) |
| [Ship pinnace](https://polyhaven.com/a/ship_pinnace) | 168,317 | 59,904 | 36% | 5.8% | [before and after](docs/images/ship_pinnace.jpg) |
| [Dutch ship](https://polyhaven.com/a/dutch_ship_large_01) | 110,616 | 36,115 | 33% | 6.4% | [before and after](docs/images/dutch_ship_large_01.jpg) |
| [Coast rocks (scan)](https://polyhaven.com/a/coast_rocks_02) | 1,260,423 | 434,995 | 35% | 5.9% | [before and after](docs/images/coast_rocks_02.jpg) |
| [Coastal cliff (scan)](https://polyhaven.com/a/coastal_cliff_04) | 1,537,926 | 362,292 | 24% | 6.0% | [before and after](docs/images/coastal_cliff_04.jpg) |


*View changed* is the share of the picture that differs from your original, seen from six sides. CurioMesh measures it
itself. It is not a quality score, and it says nothing about how tidy the quads are. *Size* is the result as a share of the
original's faces. Folded quads (a quad that is folded or flipped) are rare but not zero: the drill press has 1 of 16,809; the treasure chest has 1 of 10,423; the ship pinnace has 40 of 59,904. Numbers are from
commit `7f94ff8`, run on Blender 5.2.3 LTS.

Rough rock and fine detail need the most faces to keep their look. Hard-surface models with big smooth areas gain the most.
The sofa starts small, at 4,101 faces, and still ends at 1,571.

## Pick what it is for

Open the **Guided** panel in the CurioMesh tab (`3D View > N-panel > CurioMesh`) and follow the numbered steps: **1.** click a mesh,
**2.** choose what it is for, **3.** press **Start**, **4.** fix parts if you want (optional), **5.** press **Apply**. Nothing changes in your scene
until Apply. The older one-pass remesher is still there as **Classic Remesh**, closed by default.

Not sure which Use to pick? Press **Auto** at the top of the panel. It looks at your mesh and the free memory on this PC, picks the Use and
settings, says in plain words why it chose them, and starts. You can still change anything before you press Apply.

Each Use sets Detail, Method and Output for you. Change any of them and the Use becomes Custom. The same bust with each Use:

| Use | CurioMesh faces | Triangles left | View changed | Wire pictures |
|---|---:|---:|---:|---|
| Low Poly | 1,492 | 1,492 | 9.4% | [before and after](docs/images/bust_low_poly.jpg) |
| Game Asset (default) | 2,508 | 0 | 6.0% | [before and after](docs/images/marble_bust_01.jpg) |
| Animation | 4,228 | 0 | 10.7% | [before and after](docs/images/bust_animation.jpg) |
| Subdivision | 7,932 | 0 | 1.4% | [before and after](docs/images/bust_subdiv.jpg) |
| Scan Cleanup | 5,638 | 0 | 1.9% | [before and after](docs/images/bust_scan.jpg) |
| Sculpt Base | 3,402 | 0 | 5.9% | [before and after](docs/images/bust_sculpt.jpg) |


The Animation row shows a higher View changed (10.7%) because its level 0 is a clean cage that keeps its own rough outline, and the Multires levels bring the shape back. That cage has 38 folded quads out of 4,228. Low Poly leaves every face a triangle, as it should, and changes 9.4% of the view.

| Use | What it is for |
|---|---|
| Low Poly | The fewest faces for a game prop or a background object. Flat areas collapse into big polygons, then triangles. Small parts may vanish. |
| Game Asset | All quads, few faces. The default. |
| Animation | All quads in even sizes, with more faces so the mesh bends cleanly. Uses a clean cage with the detail in Multires levels, and Redraw Messy Quads. Check eyes, mouth and joints by hand. |
| Subdivision | All quads, with flat panels kept as single large quads so smoothing keeps the shape. Redraw Messy Quads is on. |
| Scan Cleanup | Keeps more detail of a dirty scan. Scans have few truly flat areas, so no flat merging. |
| Sculpt Base | Even, smooth quads to sculpt on. Smoothness over detail. |

Under the Use you can pick a **Method** (Auto, Pairing, Flat Quads, QuadriFlow, Grid Fill, Flow Fill, Field Rows, Tris Only) and an
**Output** of triangles, quads or mixed. The tooltips explain hard edges, UV seams and Surface Fit.

## Choose a layout style

Quads can follow the shape, or they can run in rows. Pick what comes first when the two disagree:

- **Keep Detail** holds the amount of change you allow with the fewest faces. The quads follow the shape, so on smooth surfaces they can look slanted.
- **Balanced** holds the same change with far fewer irregular vertices, by splitting every quad. It costs more faces.
- **Clean Rows** makes even, square quads in long rows. It costs more visible change, and small sharp detail softens.

| Style | Model | CurioMesh quads | View changed | Tidiness | Wire pictures |
|---|---|---:|---:|---:|---|
| Keep Detail | Marble bust | 2,508 | 6.0% | 27 | [before and after](docs/images/marble_bust_01.jpg) |
| Balanced | Marble bust | 3,920 | 6.0% | 68 | [before and after](docs/images/bust_balanced.jpg) |
| Clean Rows | Marble bust | 14,147 | 5.7% | 97 | [before and after](docs/images/bust_rows.jpg) |
| Keep Detail | Concrete road barrier | 987 | 5.9% | 16 | [before and after](docs/images/concrete_road_barrier.jpg) |
| Balanced | Concrete road barrier | 1,195 | 6.1% | 51 | [before and after](docs/images/barrier_balanced.jpg) |
| Clean Rows | Concrete road barrier | 2,486 | 6.2% | 99 | [before and after](docs/images/barrier_rows.jpg) |


![Blender quad retopology with the Keep Detail style on a marble bust, wireframe before and after](docs/images/marble_bust_01.jpg)
![Quad retopology with the Balanced style on a marble bust, wireframe before and after](docs/images/bust_balanced.jpg)
![Quad remesh with the Clean Rows style on a marble bust, even quads in rows, wireframe before and after](docs/images/bust_rows.jpg)

*Tidiness* is a number from 0 to 100 for how close the quads' layout is to hand-made meshes. Higher is tidier. Credits and limits are in
[`docs/PATTERN_SOURCES.md`](docs/PATTERN_SOURCES.md). Read it next to View changed, because a tidy grid can still miss the surface.

## Reduce To a number of faces

Set **Aim** to Reduce To and give a percentage of your faces or a face count. The result lands within about 10% of it unless the parts
cannot go that low, and the summary tells you how much of the view changed. On the bust, asking for 1,500 faces gave 1,491
with 11.2% of the view changed.

![Reduce polygon count in Blender: marble bust reduced to about 1,500 quad faces, wireframe before and after](docs/images/bust_reduce.jpg)

## Multires Detail: a clean cage with your shape in the levels

Turn on **Multires Detail** (it is Animation's default route, and offered for Subdivision and Sculpt Base) and CurioMesh makes a clean, low quad cage.
On Apply, the cage keeps its faces at level 0 and gets a Multires modifier whose levels bring your mesh's shape back, using Blender's own Subdivide and
Reshape. Animate or export level 0, and sculpt or bake from the levels above it. Your mesh is only read. On the bust, the cage has 4,228 quads at level 0, 16,912 at level 1 and 67,648 at level 2.

![Multires levels on a marble bust in Blender: the original, then the clean quad cage at level 0 and the detail restored at levels 1 and 2, wire on every mesh](docs/images/bust_multires.jpg)

Level 0 keeps its own rough outline, so a normal map changes its shading only. For any cage you made yourself, select it and your source mesh and press
**Add Multires Detail**.

## Review Segments: find the part that is wrong, and redo only that

After a run, press **Review Segments**. The result is drawn with every part in its own colour. Hover a part to see its faces before
and after, how much of the view it changed, and its method. Click a part (Shift+click for several), pick a Use, Method or Detail for
it, and press Redo. Only that part is remade, while Blender stays usable. Its seam stays where it was, so it joins its neighbours with
no gap. **Next** and **Previous** jump to the parts that changed the most. Every try is kept, so you can go back to any earlier one at once.

Big loose pieces are also split along necks and creases, so a bust can become head, torso and plinth. You can merge, split with a
drawn line, paint or rename segments. Part kinds (tube, flat plate, rough scan) are a rough guess from the shape, and the guess can be wrong.
The [CHANGELOG](CHANGELOG.md) has the full list.

## Draw edge loops, mirror, guide curves, paint density

- **Draw Edge Loops.** Draw a closed loop around an eye or the mouth, or an open line. Each becomes a strip of regular quads with the
  poles pushed outside it. **Deformation Lines** across an elbow, knee or mouth corner become three or more evenly spaced loops. Move or redraw
  a curve and the result updates. In our test, a ring across a bent neck cut collapsed quads there from 6.2% to 4.0%. A broad bend saw no
  difference, so it helps tight joints and little else.
- **Mirror** (Auto, X, Y or Z). Solves one half and mirrors it, so both sides match and the seam is exact. Auto uses the plane only when 90% of the surface matches.
- **Guide Curves.** Draw strokes on the mesh, or add curves, Grease Pencil drawings or edge-only meshes. The quads follow them.
- **Paint Density.** Paint a vertex group red where you want more quads. It works best with Reduce To.
- **Show it how.** Fix a spot by hand, press **Remember This Fix**, and CurioMesh can find the same quad pattern in later results. It is stored on your PC.

## Redo inside shapes (Live Boolean)

Put boxes, spheres or any closed mesh over parts of the result and combine them with Union, Subtract or Intersect. Only the faces inside
are redone as you move the shapes. A slider from Organized to Detailed sets the new quad size. Everything outside stays as it was.
A redo is refused when it would make more faces than it replaces, and the panel says so.

## Bake detail, UVs and game engines

- **Bake Detail** sits next to Apply. Choose Off, **Normal Map** or **Normal + Color**, at 512 to 4096 pixels. Your mesh is only read. Normal + Color makes one new material with new UVs and is the recommended route for textured game assets.
- **Make LOD Set** makes three new objects named `_LOD0`, `_LOD1` and `_LOD2` for game engines, each a Reduce To of your mesh at percents you choose (50, 25 and 12.5% by default). Only the first runs the full search.
- **Keep UV Seams** keeps each quad inside one UV island. It is off by default, because a model cut into many islands can need up to about twice the faces.
- Apply marks edges sharper than 60 degrees as hard, and the result takes its shading from the nearest source points, so flat sides stay flat.
- Baking was tested through FBX and glTF export and re-import in Blender. It has not been tested in a game engine yet.
- Baking blocks Blender until it finishes and reads the surface nearest to each texel, so parts that touch can bleed into each other. Only the normal map and base colour are baked, not roughness or metal.

## Big scans, small PCs, and a cloud job

The **This computer** box reads your cores, free memory and graphics card and says in words how big the job is for this PC.
**Effort** (Auto, Light, Normal, Full) trades speed for a usable Blender. Pieces too big for the effort are cut into regions with their borders kept
exactly. Each region appears when it is done, and finished regions are saved so a stopped job resumes.

**Export Job** writes a folder you can run on any computer that has Blender (`blender --background --python run_job.py`).
**Import Result** brings the answer back. CurioMesh never uploads anything and has no accounts or keys.
See [docs/CLOUD_WORKFLOW.md](docs/CLOUD_WORKFLOW.md).

At the default settings, the coast rocks scan went from 1,260,423 faces to 434,995 quads (5.9% of the view changed) in 5 min and the coastal cliff scan went from 1,537,926 faces to 362,292 quads (6.0% of the view changed) in 7 min, on a PC that was also doing other work. Both stayed all quads.

Two options help on rough scans of rock, terrain or sea floor, and both are off by default. **Rough As Texture** judges rough
surfaces by their average direction instead of every pixel, so far fewer faces are kept (bumps smaller than a few pixels are lost).
**Leave Out Hidden Pieces** drops loose pieces that no outside camera can see.

## All features

<details>
<summary>The rest of the feature list (click to open)</summary>

| Feature | What it does |
|---|---|
| **Redraw Messy Quads** | Parts with many irregular vertices are redrawn as pole-free grids. On for Animation and Subdivision, and left off on models of many loose parts. |
| **Segments inside one mesh** | Splits a connected model along necks and creases (head, torso, plinth). Merge, split, paint, rename. |
| **Draw Edge Loops and Deformation Lines** | Loops round eyes and mouths, lines across joints. The quads follow. |
| **Mirror, Guide Curves, Paint Density** | Symmetric results, strokes the quads follow, and more faces where you paint. |
| **Show it how** | Fix a spot by hand once, and CurioMesh finds the same pattern in later results. |
| **Redo inside shapes (Live Boolean)** | Union, Subtract or Intersect boxes and spheres, and only what is inside is redone, live. |
| **UV and shading care** | Keep UV Seams, hard edges past 60 degrees, shading read from your source. |
| **Checks and repair** | Non-manifold edges, flipped faces, duplicates, holes, crossing faces. What cannot be fixed is selected. |
| **Tidiness score** | A 0 to 100 number for how close the quad layout is to hand-made meshes. |
| **Part cases** | Each part is named (tube, flat plate, rough scan and so on) so the search starts at sensible sizes. |
| **Rough As Texture, Leave Out Hidden Pieces** | Off by default. Far fewer faces on rough scans. Drops loose pieces no camera can see. |
| **Keep Copies as Instances** | Repeated parts become linked objects, so a copy costs no geometry. |
| **Effort levels, This computer, Export Job** | Work in small steps, or run the job on a stronger computer and import the result. |
| **Regions** | Redo any selection, face set, vertex group or cage with an exact seam. [More.](docs/OLDER_FEATURES.md) |
| **Flat Quads** | Flat areas become one big quad each. [More.](docs/OLDER_FEATURES.md) |
| **Guided points and Work Live** | Click where detail matters, and retopo area by area in the background. [More.](docs/OLDER_FEATURES.md) |
| **Classic Remesh** | The older whole-mesh remesher (Atlas-Q, Fill Loop, Fit to Surface), folded away under the Guided panel. [More.](docs/ATLAS_Q_REFERENCE.md) |
| **HUD and heatmap** | Quad percentage, leftover triangles and poles, shown in the viewport. |
| **Native Turbo core** | Rust code on all cores for the slow steps, with a Python fallback. |

</details>

## Older features

Whole-mesh Atlas-Q, **Regions** (redo any selection or cage with an exact seam), **Flat Quads**, **Guided points** and **Work Live** are all still in
CurioMesh. They are described in [docs/OLDER_FEATURES.md](docs/OLDER_FEATURES.md). The solvers, controls and maths of the whole-mesh remesher
are in [docs/ATLAS_Q_REFERENCE.md](docs/ATLAS_Q_REFERENCE.md). Release notes are in [CHANGELOG.md](CHANGELOG.md).

## CurioMesh without Blender: the standalone app (preview)

![CurioMesh Standalone: drag the line to compare the original with its new quads](docs/images/standalone_app.png)

Turn a heavy scan, CAD export or sculpt into clean quads with a desktop app for Windows, macOS and Linux. Drop
a model in, say what it's for, and drag the line across it to compare the original with the new quads. You choose
how much of the view may change; CurioMesh keeps the fewest faces that hold it and lists the spots worth a look.

- **Download:** the zip for your system from the [standalone releases](https://github.com/CurioCrafter/CurioMesh/releases?q=standalone).
  Unpack it and start `curiomesh-gui`.
- **Opens** OBJ, glTF, GLB and PLY. **Saves** OBJ and a JSON issue report.
- **Command line for pipelines:** `curiomesh model.glb -o low.obj --use game-asset`, and
  `curiomesh measure model.glb any_retopo.obj --max-loss 6 --max-bad 0` to stop a build on a broken asset from any
  tool.
- **Same core** as the add-on: its Rust reducer, quad pairing and closing, and its six-view camera.

On three free test models at about the same face count, it kept far more of the shape than AutoRemesher 1.2.0
(drill press: 8.2% of the view changed against 32.8%, 0 bad faces against 1,219). AutoRemesher's quads are more
even on smooth, organic shapes. The full table, the guide and the limits are in
[standalone/README.md](standalone/README.md); the plan and the legal notes are in
[standalone/PLAN.md](standalone/PLAN.md).

## Download and install the Blender retopology add-on

Each release has a zip for your exact Blender version, plus two all-in-one zips.
Get them from the [latest release](https://github.com/CurioCrafter/CurioMesh/releases/latest).

| Your Blender | Download | Install with |
| --- | --- | --- |
| 5.2 LTS, 5.1, 5.0, 4.5 LTS, 4.4, 4.3, 4.2 LTS | `curiomesh-<version>-blender-<your version>-extension.zip` | `Edit > Preferences > Get Extensions > ⌄ > Install from Disk` |
| 4.1, 4.0, 3.6 LTS | `curiomesh-<version>-blender-<your version>-addon.zip` | `Edit > Preferences > Add-ons > Install...` |
| Not sure | `...-blender-4.2-plus-extension.zip` (4.2 and newer) or `...-blender-3.6-plus-addon.zip` (any version from 3.6) | as above |
| 3.5 and older | not supported | |

To install:

1. Download the zip that matches your Blender version from the table above. Do not unzip it.
2. In Blender, open the menu from the table (Get Extensions for 4.2 and newer, Add-ons for older versions) and pick the zip.
3. Enable **CurioMesh**.
4. Select a mesh and open `3D View > N-panel > CurioMesh`.

The version on extensions.blender.org is Python-only (the site allows no compiled code), so it is slower on big meshes, and its tidiness score uses CC0
reference meshes only. The GitHub zips include the fast native Turbo core. `python scripts/build_release.py --blender-listing` builds that zip. Releases are built
by the release workflow from the tag, and each comes with `SHA256SUMS.txt` so you can check your download.

The last full test on every Blender version from 3.6 to 5.2 was for 0.8.0. The checks for this version ran on 3.6 LTS, 4.2 LTS and 5.2 LTS, and most of the work on Auto Segments
was done on 5.2. The viewport overlays draw with Blender's GPU module. If a build cannot draw them, the rest of the add-on keeps working.

## Frequently asked questions about free Blender retopology

**Is there a free automatic retopology add-on for Blender?**
Yes. CurioMesh is free and open source (GPL-3.0-or-later), works in Blender 3.6 to 5.2, and needs no account. Paid retopology tools exist too. This README makes no quality comparison with them.

**How do I retopologize a scan in Blender?**
Install CurioMesh, select the scan, open the CurioMesh tab, choose Auto Segments, pick the Scan Cleanup Use, and press Start. Check the result in Review Segments, redo any part that looks wrong, then press Apply. For scans over a million faces, set Effort to Light or use Export Job.

**How do I make a low poly game asset from a high poly mesh?**
Pick the Game Asset or Low Poly Use and press Start. Turn on Bake Detail with Normal + Color to carry the surface detail over as a normal map. Triangulate before you export to your engine.

**Can I get all quads from an automatic quad remesh?**
Yes. Output defaults to Quads, and the summary reports leftover triangles, which are normally zero. Quads can still be slanted on smooth surfaces with Keep Detail. Balanced and Clean Rows give tidier layouts for more faces.

**Can I redo just one part of a retopology?**
Yes. Review Segments redoes one part with its own method and detail while the seam to its neighbours stays closed. Redo Inside Shapes does the same for any area you cover with a box or sphere.

**How is it different from Blender's Quad Remesh?**
Blender's Quad Remesh works on the whole object in one pass. CurioMesh works part by part, measures each result against your original, and lets you redo single parts. QuadriFlow is also one of its Methods, for smooth organic parts.

**How do I make LODs for a game asset in Blender?**
Select the mesh, set the three percents in Make LOD Set (50, 25 and 12.5% by default) and press it. You get `_LOD0`, `_LOD1` and `_LOD2` as new objects, and only the first runs the full search.

**Can I get a clean low-poly cage with the detail in Multires levels?**
Yes. Multires Detail makes the cage and puts your mesh's shape in the levels above level 0. It is the default route for the Animation Use.

**Does it keep my UVs?**
Turn on Keep UV Seams and each quad stays inside one UV island, at the cost of more faces on models with many islands. Or bake with Normal + Color, which makes new UVs and does not need the old seams.

**Does it upload my model?**
No. Everything runs on your PC. Export Job writes a folder that you move yourself.

**Does it work in Blender 4.x and 5.x?**
Yes, from Blender 3.6 LTS up to 5.2. See [Download and install](#download-and-install-the-blender-retopology-add-on) for which versions were tested for this release.

## Known limits

- **Keep Detail quads are not always tidy.** It spends everything Detail allows, so on smooth surfaces quads can look slanted or scattered instead of running in rows. Spinning edges at poles helps (Tidiness about 28 to 39 on smooth busts), but it is still far from hand-made. Use Balanced or Clean Rows for that, at the cost of more faces. Clean Rows also loses small sharp detail.
- **Soft faces and fine detail go first.** At 6%, cloth folds, small rounded ends and fine ornaments are simplified. Lower Detail to keep more. CurioMesh does not give you a hand-made low poly yet.
- **UV seams.** By default quads may span several UV islands, so textures can smear there. Turn on Keep UV Seams (up to about twice the faces) or bake with Normal + Color.
- **Big scans are slow.** A scan of over a million faces can take a long time on a busy PC, and Blender can pause for a few seconds at a time. Use Effort, or Export Job.
- **Grid Fill and Flow Fill** only work on a part that is one open surface with one edge loop round it. Otherwise Auto is used, and the summary says so.
- **Repair is cautious.** A fix the camera dislikes, or one that adds over 3% more faces, is undone, and clearing crossings stops after 30 seconds, so a few spots can stay. They are selected on the applied mesh and named in the summary.
- **Selected Faces Only** for Draw Edge Loops does not close its seam yet and is switched off. Use Redo Inside Shapes.
- **Live Boolean** refuses a redo that would make more faces than it replaces. On a mesh already reduced hard, that can be most redos.
- **No global patch layout.** CurioMesh has none, and it does not promise exact loops round eyes, mouths or joints without your help. An optional script, `scripts/flowtool.py`, can use QuadWild if you install it separately. It is not in the panel, it covers organic parts only, and it is untested on Linux and macOS.
- **Native core builds.** The release workflow builds the Turbo core for Windows, Linux and macOS (Intel and Apple Silicon). The Intel Mac build is compiled but not run in CI, and the developers tried only Windows by hand. Without the core, CurioMesh falls back to Python code, which is much slower on big meshes.
- **Little GPU compute.** Only the camera pictures of meshes over 150,000 triangles run on the graphics card (NVIDIA, AMD, Intel or Apple, no CUDA). That is 1.1 to 3 times faster for that step, about 1 to 2% for a whole run. Everything else runs on the processor, and a graphics card does not speed it up.
- **Folded slivers on some small thin parts.** A quad can come out bent about 90 degrees round an edge of a thin part. They are named in the summary and selected on Apply, so you can fix them by hand.
- **Judge After Baking is experimental.** On big flat areas of a scan it can leave long thin slivers, which bake badly, and its score misses them. It is off by default.
- **Redraw Messy Quads costs time.** On some models it roughly doubles the tidiness score for 7 to 50% more faces and 2 to 5 times the time, and it is left off on models of more than 40 loose pieces.
- Mirror, Guide Curves and Paint Density work in Auto Segments only. Guide strokes on a Grease Pencil object need Blender 4.3 or later.

## Community

- **Questions and help:** [Q&A](https://github.com/CurioCrafter/CurioMesh/discussions/categories/q-a)
- **Show your retopology:** [Show and tell](https://github.com/CurioCrafter/CurioMesh/discussions/categories/show-and-tell). Before and after screenshots and the settings you used help everyone.
- **Ideas and feature requests:** [Ideas](https://github.com/CurioCrafter/CurioMesh/discussions/categories/ideas)
- **Bugs:** [open an issue](https://github.com/CurioCrafter/CurioMesh/issues/new) with your Blender version, your CurioMesh settings and, if you can share it, the mesh or a small `.blend`.

Pull requests are welcome. Run the tests first:

```powershell
python -m unittest discover -s tests -v
ruff check .
python scripts\build_release.py
```

`python scripts\test_blender_matrix.py` runs the smoke and install tests on every Blender build it finds.

If CurioMesh saved you time, a [star on GitHub](https://github.com/CurioCrafter/CurioMesh/stargazers) helps other Blender artists find a free retopology tool.

[![Star History Chart](https://api.star-history.com/svg?repos=CurioCrafter/CurioMesh&type=Date)](https://star-history.com/#CurioCrafter/CurioMesh&Date)

## Credits

- Test models are free CC0 models from [Poly Haven](https://polyhaven.com); thank you to Poly Haven for making them free for everyone. Models by: Marble bust: Rico Cilliers; Concrete road barrier: Amal Kumar; Old military compressor: Brian Speight; Fire hydrant: Goncalo Felicio; Treasure chest: Rico Cilliers; Gothic statue: Benny Weimer; Lion head: Tina; Horse statue: Rico Cilliers; Chess set: Riley Queen; Spinning wheel: Sofia Pahaoja; Sofa: Kirill Sannikov; Drill press: Viktor; Overhead crane: Timothy3D; Modular wooden pier: Rico Cilliers; Moss rocks: Kless Gyzen; Namaqualand boulder: Greg Zaal, Rico Cilliers; Ship pinnace: James Ray Cock, Rico Cilliers, Nicolo Zubbini, Yann Kervran; Dutch ship: James Ray Cock, Rico Cilliers, Nicolo Zubbini; Coast rocks: Rob Tuytel, Rico Cilliers; Coastal cliff: Rob Tuytel, Rico Cilliers.
- Idea from [cav3dw3ll3r](https://github.com/cav3dw3ll3r): fitting each source point to its nearest vertex, shading read from the nearest source points, and the finite test cases.
- The tidiness score is based on statistics measured from hand-made meshes. It holds no geometry. Sources and licenses are in
  [docs/PATTERN_SOURCES.md](docs/PATTERN_SOURCES.md).
- Credits for the CC-BY works, changed by reducing them to statistics ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)):
  - [Sprite Rig](https://studio.blender.org/characters/sprite/v1/) (CC-BY) Blender Foundation, studio.blender.org
  - [Rain Rig](https://studio.blender.org/characters/rain/v3/) (CC) Blender Foundation, studio.blender.org
  - [Pack Bot Rig](https://studio.blender.org/characters/pack-bot/v1/) (CC-BY) Blender Foundation, studio.blender.org
  - [Security Bot Rig](https://studio.blender.org/characters/security-bot/v1/) (CC-BY) Blender Foundation, studio.blender.org
- The triangle intersection test in the native core is ported from Blender's `isect_tri_tri_v3_ex` (GPL-2.0-or-later, Blender authors). Rust crates in the native core and their licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- The whole-mesh solvers are CurioMesh's own implementations of published papers, listed in [docs/ATLAS_Q_REFERENCE.md](docs/ATLAS_Q_REFERENCE.md). No QuadWild source is copied or included.

## License

CurioMesh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

Versions up to 0.5.1 were released under MIT. From 0.6.0 on, CurioMesh is GPL-3.0-or-later, the same family of licence Blender itself uses.
