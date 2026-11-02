# CurioMesh: free automatic retopology for Blender

**Turn a dense sculpt, scan or high-poly game asset into a clean all-quad, low-poly mesh. Free and open source.**

[![Download CurioMesh](https://img.shields.io/badge/Download-free%20add--on-2ea44f?style=for-the-badge&logo=blender&logoColor=white)](https://github.com/CurioCrafter/CurioMesh/releases/latest)
[![Star on GitHub](https://img.shields.io/badge/Star-on%20GitHub-24292f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CurioCrafter/CurioMesh/stargazers)

[![Latest release](https://img.shields.io/github/v/release/CurioCrafter/CurioMesh?label=release)](https://github.com/CurioCrafter/CurioMesh/releases/latest)
[![Blender 3.6 to 5.2](https://img.shields.io/badge/Blender-3.6%20to%205.2-orange?logo=blender&logoColor=white)](https://www.blender.org/download/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

![Automatic retopology in Blender: wireframe of an air compressor before and after CurioMesh, 79,042 faces reduced to 16,317 all-quad faces, whole object and close-up](docs/images/old_military_compressor.jpg)

*[Old military compressor](https://polyhaven.com/a/old_military_compressor) (free CC0 model): 79,042 faces to 16,317 quads, 5.3% of the view changed. Wire shown on both sides.*

CurioMesh is an automatic quad retopology add-on for Blender 3.6 to 5.2. Say what the mesh is for, press one button, and it reduces
the model part by part. A camera check compares the result with your original from six sides and tells you how much of the view
changed. Then you redo only the part that still looks wrong. It runs on your PC and uploads nothing.

**Try it in three steps:** [install the zip](#download-and-install), pick a **Use** (Game Asset, Animation, Scan Cleanup and more) and press **Start**,
then check parts in **Review Segments** and press **Apply**.

## Contents

[Examples](#examples-before-and-after-wire-on-both-sides) · [Uses](#pick-what-it-is-for) · [Styles](#choose-a-layout-style) · [Review Segments](#review-segments-find-the-part-that-is-wrong-and-redo-only-that) · [Bake and UVs](#bake-detail-uvs-and-game-engines) · [Big scans](#big-scans-small-pcs-and-a-cloud-job) · [Install](#download-and-install) · [FAQ](#frequently-asked-questions) · [Limits](#known-limits) · [All features](#all-features)

## What it does

| Feature | What it does |
|---|---|
| **Auto Segments** | Reduces the whole object to all quads, one loose part at a time, and checks every size from six sides. |
| **Presets by use** | Low Poly, Game Asset, Animation, Subdivision, Scan Cleanup, Sculpt Base. |
| **Three layout styles** | Keep Detail (fewest faces), Balanced, Clean Rows (even quads in rows). |
| **Reduce To** | Aim at a percentage of your faces or a face count, within about 10%. |
| **Review Segments** | Every part in its own colour, numbers on hover, redo one part with another method, go back to any try. |
| **Segments inside one mesh** | Splits a connected model along necks and creases (head, torso, plinth). Merge, split, paint, rename. |
| **Draw Edge Loops and Deformation Lines** | Loops round eyes and mouths, lines across joints. The quads follow. |
| **Mirror, Guide Curves, Paint Density** | Symmetric results, strokes the quads follow, and more faces where you paint. |

More below: [all features](#all-features).

## Examples: before and after, wire on both sides

Every example below ran the real add-on on a free [Poly Haven](https://polyhaven.com) model (CC0) with the default Game
Asset settings: Keep Detail, **Detail 6%**. Detail is how much of the view may look different from your mesh. You can
download the same models and check the numbers. Each picture shows the original (left) and the result (right), the whole
object on top and a close-up below.

![Chess set (33 pieces): 76,832 faces become 10,834 quads, wire on both, whole object and close-up](docs/images/chess_set.jpg)

*Chess set (33 pieces), [Poly Haven](https://polyhaven.com/a/chess_set), CC0.*

![Overhead crane: 89,964 faces become 25,090 quads, wire on both, whole object and close-up](docs/images/overhead_crane.jpg)

*Overhead crane, [Poly Haven](https://polyhaven.com/a/overhead_crane), CC0.*

![Concrete road barrier: 60,928 faces become 987 quads, wire on both, whole object and close-up](docs/images/concrete_road_barrier.jpg)

*Concrete road barrier, [Poly Haven](https://polyhaven.com/a/concrete_road_barrier), CC0.*

| Model | Original faces | CurioMesh quads | Size | View changed | Wire pictures |
|---|---:|---:|---:|---:|---|
| [Marble bust](https://polyhaven.com/a/marble_bust_01) | 17,456 | 2,632 | 15% | 6.0% | [before and after](docs/images/marble_bust_01.jpg) |
| [Concrete road barrier](https://polyhaven.com/a/concrete_road_barrier) | 60,928 | 987 | 2% | 5.9% | [before and after](docs/images/concrete_road_barrier.jpg) |
| [Old military compressor](https://polyhaven.com/a/old_military_compressor) | 79,042 | 16,317 | 21% | 5.3% | [before and after](docs/images/old_military_compressor.jpg) |
| [Fire hydrant](https://polyhaven.com/a/fire_hydrant) | 86,316 | 27,795 | 32% | 5.0% | [before and after](docs/images/fire_hydrant.jpg) |
| [Treasure chest](https://polyhaven.com/a/treasure_chest) | 103,330 | 10,789 | 10% | 4.5% | [before and after](docs/images/treasure_chest.jpg) |
| [Gothic statue](https://polyhaven.com/a/gothic_statue) | 27,739 | 18,184 | 66% | 5.9% | [before and after](docs/images/gothic_statue.jpg) |
| [Lion head](https://polyhaven.com/a/lion_head) | 47,214 | 7,995 | 17% | 6.0% | [before and after](docs/images/lion_head.jpg) |
| [Horse statue](https://polyhaven.com/a/horse_statue_01) | 22,388 | 5,623 | 25% | 5.9% | [before and after](docs/images/horse_statue_01.jpg) |
| [Chess set (33 pieces)](https://polyhaven.com/a/chess_set) | 76,832 | 10,834 | 14% | 3.4% | [before and after](docs/images/chess_set.jpg) |
| [Spinning wheel](https://polyhaven.com/a/spinning_wheel_01) | 52,023 | 14,810 | 28% | 5.4% | [before and after](docs/images/spinning_wheel_01.jpg) |
| [Sofa (already low poly)](https://polyhaven.com/a/Sofa_01) | 4,101 | 1,658 | 40% | 3.8% | [before and after](docs/images/Sofa_01.jpg) |
| [Drill press](https://polyhaven.com/a/drill_press_01) | 58,511 | 18,103 | 31% | 5.5% | [before and after](docs/images/drill_press_01.jpg) |
| [Overhead crane](https://polyhaven.com/a/overhead_crane) | 89,964 | 25,090 | 28% | 6.4% | [before and after](docs/images/overhead_crane.jpg) |
| [Modular wooden pier](https://polyhaven.com/a/modular_wooden_pier) | 84,780 | 22,411 | 26% | 6.0% | [before and after](docs/images/modular_wooden_pier.jpg) |
| [Moss rocks (6 pieces)](https://polyhaven.com/a/rock_moss_set_01) | 63,127 | 21,142 | 33% | 6.0% | [before and after](docs/images/rock_moss_set_01.jpg) |
| [Namaqualand boulder](https://polyhaven.com/a/namaqualand_boulder_02) | 97,964 | 52,789 | 54% | 6.0% | [before and after](docs/images/namaqualand_boulder_02.jpg) |
| [Ship pinnace](https://polyhaven.com/a/ship_pinnace) | 168,317 | 63,889 | 38% | 6.4% | [before and after](docs/images/ship_pinnace.jpg) |
| [Dutch ship](https://polyhaven.com/a/dutch_ship_large_01) | 110,616 | 36,591 | 33% | 5.5% | [before and after](docs/images/dutch_ship_large_01.jpg) |
| [Coast rocks (scan)](https://polyhaven.com/a/coast_rocks_02) | 1,260,423 | 434,995 | 35% | 5.9% | [before and after](docs/images/coast_rocks_02.jpg) |
| [Coastal cliff (scan)](https://polyhaven.com/a/coastal_cliff_04) | 1,537,926 | 362,292 | 24% | 6.0% | [before and after](docs/images/coastal_cliff_04.jpg) |


*View changed* is the share of the picture that differs from your original, seen from six sides. CurioMesh measures it
itself. It is not a quality score, and it says nothing about how tidy the quads are. *Size* is the result as a share of the
original's faces. Folded quads (a quad that is folded or flipped) are rare but not zero: the treasure chest has 1 of 10,789; the ship pinnace has 26 of 63,889. Numbers are from
commit `77d913d`, run on Blender 5.2.3 LTS.

Rough rock and fine detail need the most faces to keep their look. Hard-surface models with big smooth areas gain the most.
The sofa starts small, at 4,101 faces, and still ends at 1,658.

## Pick what it is for

Open the **Guided** panel in the CurioMesh tab (`3D View > N-panel > CurioMesh`), choose **Auto Segments**, pick a **Use**,
and press **Start**. Nothing changes in your scene until you press **Apply**.

Each Use sets Detail, Method and Output for you. Change any of them and the Use becomes Custom. The same bust with each Use:

| Use | CurioMesh faces | Triangles left | View changed | Wire pictures |
|---|---:|---:|---:|---|
| Low Poly | 1,396 | 1,396 | 10.4% | [before and after](docs/images/bust_low_poly.jpg) |
| Game Asset (default) | 2,632 | 0 | 6.0% | [before and after](docs/images/marble_bust_01.jpg) |
| Animation | 7,042 | 0 | 2.0% | [before and after](docs/images/bust_animation.jpg) |
| Subdivision | 6,967 | 0 | 2.1% | [before and after](docs/images/bust_subdiv.jpg) |
| Scan Cleanup | 5,478 | 0 | 2.0% | [before and after](docs/images/bust_scan.jpg) |
| Sculpt Base | 3,194 | 0 | 6.4% | [before and after](docs/images/bust_sculpt.jpg) |


| Use | What it is for |
|---|---|
| Low Poly | The fewest faces for a game prop or a background object. Flat areas collapse into big polygons, then triangles. Small parts may vanish. |
| Game Asset | All quads, few faces. The default. |
| Animation | All quads in even sizes, with more faces so the mesh bends cleanly. Check eyes, mouth and joints by hand. |
| Subdivision | All quads, with flat panels kept as single large quads so smoothing keeps the shape. |
| Scan Cleanup | Keeps more detail of a dirty scan. Scans have few truly flat areas, so no flat merging. |
| Sculpt Base | Even, smooth quads to sculpt on. Smoothness over detail. |

Under the Use you can pick a **Method** (Auto, Pairing, Flat Quads, QuadriFlow, Grid Fill, Flow Fill, Tris Only) and an
**Output** of triangles, quads or mixed. The tooltips explain hard edges, UV seams and Surface Fit.

## Choose a layout style

Quads can follow the shape, or they can run in rows. Pick what comes first when the two disagree:

- **Keep Detail** holds the amount of change you allow with the fewest faces. The quads follow the shape, so on smooth surfaces they can look slanted.
- **Balanced** holds the same change with far fewer irregular vertices, by splitting every quad. It costs more faces.
- **Clean Rows** makes even, square quads in long rows. It costs more visible change, and small sharp detail softens.

| Style | Model | CurioMesh quads | View changed | Tidiness | Wire pictures |
|---|---|---:|---:|---:|---|
| Keep Detail | Marble bust | 2,632 | 6.0% | 28 | [before and after](docs/images/marble_bust_01.jpg) |
| Balanced | Marble bust | 5,708 | 5.2% | 69 | [before and after](docs/images/bust_balanced.jpg) |
| Clean Rows | Marble bust | 14,147 | 5.7% | 97 | [before and after](docs/images/bust_rows.jpg) |
| Keep Detail | Concrete road barrier | 987 | 5.9% | 16 | [before and after](docs/images/concrete_road_barrier.jpg) |
| Balanced | Concrete road barrier | 1,195 | 6.1% | 51 | [before and after](docs/images/barrier_balanced.jpg) |
| Clean Rows | Concrete road barrier | 910 | 6.4% | 22 | [before and after](docs/images/barrier_rows.jpg) |


![Blender quad retopology with the Keep Detail style on a marble bust, wireframe before and after](docs/images/marble_bust_01.jpg)
![Quad retopology with the Balanced style on a marble bust, wireframe before and after](docs/images/bust_balanced.jpg)
![Quad remesh with the Clean Rows style on a marble bust, even quads in rows, wireframe before and after](docs/images/bust_rows.jpg)

*Tidiness* is a number from 0 to 100 for how close the quads' layout is to hand-made meshes. Higher is tidier. Credits and limits are in
[`docs/PATTERN_SOURCES.md`](docs/PATTERN_SOURCES.md). Read it next to View changed, because a tidy grid can still miss the surface.

## Reduce To a number of faces

Set **Aim** to Reduce To and give a percentage of your faces or a face count. The result lands within about 10% of it unless the parts
cannot go that low, and the summary tells you how much of the view changed. On the bust, asking for 1,500 faces gave 1,586
with 11.4% of the view changed.

![Reduce polygon count in Blender: marble bust reduced to about 1,500 quad faces, wireframe before and after](docs/images/bust_reduce.jpg)

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

At the default settings, the coast rocks scan went from 1,260,423 faces to 434,995 quads (5.9% of the view changed) in 20 min and the coastal cliff scan went from 1,537,926 faces to 362,292 quads (6.0% of the view changed) in 19 min, on a PC that was also doing other work. Both stayed all quads.

Two options help on rough scans of rock, terrain or sea floor, and both are off by default. **Rough As Texture** judges rough
surfaces by their average direction instead of every pixel, so far fewer faces are kept (bumps smaller than a few pixels are lost).
**Leave Out Hidden Pieces** drops loose pieces that no outside camera can see.

## All features

<details>
<summary>The rest of the feature list (click to open)</summary>

| Feature | What it does |
|---|---|
| **Show it how** | Fix a spot by hand once, and CurioMesh finds the same pattern in later results. |
| **Redo inside shapes (Live Boolean)** | Union, Subtract or Intersect boxes and spheres, and only what is inside is redone, live. |
| **Bake Detail** | Normal map, or normal map and colour, onto the new mesh. Your mesh is only read. |
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
| **Whole-mesh Atlas-Q, Fill Loop, Fit to Surface** | The older whole-mesh remesher and its helpers. [More.](docs/ATLAS_Q_REFERENCE.md) |
| **HUD and heatmap** | Quad percentage, leftover triangles and poles, shown in the viewport. |
| **Native Turbo core** | Rust code on all cores for the slow steps, with a Python fallback. |

</details>

## Older features

Whole-mesh Atlas-Q, **Regions** (redo any selection or cage with an exact seam), **Flat Quads**, **Guided points** and **Work Live** are all still in
CurioMesh. They are described in [docs/OLDER_FEATURES.md](docs/OLDER_FEATURES.md). The solvers, controls and maths of the whole-mesh remesher
are in [docs/ATLAS_Q_REFERENCE.md](docs/ATLAS_Q_REFERENCE.md). Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Download and install

Each release has a zip for your exact Blender version, plus two all-in-one zips.
Get them from the [latest release](https://github.com/CurioCrafter/CurioMesh/releases/latest).

| Your Blender | Download | Install with |
| --- | --- | --- |
| 5.2 LTS, 5.1, 5.0, 4.5 LTS, 4.4, 4.3, 4.2 LTS | `curiomesh-<version>-blender-<your version>-extension.zip` | `Edit > Preferences > Get Extensions > ⌄ > Install from Disk` |
| 4.1, 4.0, 3.6 LTS | `curiomesh-<version>-blender-<your version>-addon.zip` | `Edit > Preferences > Add-ons > Install...` |
| Not sure | `...-blender-4.2-plus-extension.zip` (4.2 and newer) or `...-blender-3.6-plus-addon.zip` (any version from 3.6) | as above |
| 3.5 and older | not supported | |

Then enable **CurioMesh**, select a mesh, and open `3D View > N-panel > CurioMesh`.

The last full test on every Blender version from 3.6 to 5.2 was for 0.8.0. The checks for this version ran on 3.6 LTS, 4.2 LTS and 5.2 LTS, and most of the work on Auto Segments
was done on 5.2. The viewport overlays draw with Blender's GPU module. If a build cannot draw them, the rest of the add-on keeps working.

## Frequently asked questions

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

**Does it keep my UVs?**
Turn on Keep UV Seams and each quad stays inside one UV island, at the cost of more faces on models with many islands. Or bake with Normal + Color, which makes new UVs and does not need the old seams.

**Does it upload my model?**
No. Everything runs on your PC. Export Job writes a folder that you move yourself.

**Does it work in Blender 4.x and 5.x?**
Yes, from Blender 3.6 LTS up to 5.2. See [Download and install](#download-and-install) for which versions were tested for this release.

## Known limits

- **Keep Detail quads are not always tidy.** It spends everything Detail allows, so on smooth surfaces quads can look slanted or scattered instead of running in rows. Use Balanced or Clean Rows for that, at the cost of more faces. Clean Rows also loses small sharp detail, and on some models it lands over the amount you asked for.
- **Soft faces and fine detail go first.** At 6%, cloth folds, small rounded ends and fine ornaments are simplified. Lower Detail to keep more. CurioMesh does not give you a hand-made low poly yet.
- **UV seams.** By default quads may span several UV islands, so textures can smear there. Turn on Keep UV Seams (up to about twice the faces) or bake with Normal + Color.
- **Big scans are slow.** A scan of over a million faces can take a long time on a busy PC, and Blender can pause for a few seconds at a time. Use Effort, or Export Job.
- **Grid Fill and Flow Fill** only work on a part with one open edge loop and an even number of edges around it. Otherwise Auto is used, and the summary says so.
- **Repair is cautious.** A fix that the camera dislikes, or that adds over a tenth more faces, is undone, so a few spots can stay. They are selected on the applied mesh and named in the summary.
- **Selected Faces Only** for Draw Edge Loops does not close its seam yet and is switched off. Use Redo Inside Shapes.
- **Live Boolean** refuses a redo that would make more faces than it replaces. On a mesh already reduced hard, that can be most redos.
- **No global patch layout.** CurioMesh has none, and it does not promise exact loops round eyes, mouths or joints without your help. An optional script, `scripts/flowtool.py`, can use QuadWild if you install it separately. It is not in the panel, it covers organic parts only, and it is untested on Linux and macOS.
- **Windows first.** The native Turbo core is built for Windows only today. On other systems CurioMesh falls back to slower Python code, which is much slower on big meshes. The release workflow is meant to add the other platforms.
- **CPU only.** There is no GPU compute.
- **No LOD set button yet.** Run Reduce To several times to make levels.
- Mirror, Guide Curves and Paint Density work in Auto Segments only. Guide strokes on a Grease Pencil object need Blender 4.3 or later.
- Stop is felt between steps. One reduction of a piece with several hundred thousand faces cannot be interrupted.

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

- Test models are free CC0 models from [Poly Haven](https://polyhaven.com). Models by: Marble bust: Rico Cilliers; Concrete road barrier: Amal Kumar; Old military compressor: Brian Speight; Fire hydrant: Goncalo Felicio; Treasure chest: Rico Cilliers; Gothic statue: Benny Weimer; Lion head: Tina; Horse statue: Rico Cilliers; Chess set: Riley Queen; Spinning wheel: Sofia Pahaoja; Sofa: Kirill Sannikov; Drill press: Viktor; Overhead crane: Timothy3D; Modular wooden pier: Rico Cilliers; Moss rocks: Kless Gyzen; Namaqualand boulder: Greg Zaal, Rico Cilliers; Ship pinnace: James Ray Cock, Rico Cilliers, Nicolo Zubbini, Yann Kervran; Dutch ship: James Ray Cock, Rico Cilliers, Nicolo Zubbini; Coast rocks: Rob Tuytel, Rico Cilliers; Coastal cliff: Rob Tuytel, Rico Cilliers.
- Idea from [cav3dw3ll3r](https://github.com/cav3dw3ll3r): fitting each source point to its nearest vertex, shading read from the nearest source points, and the finite test cases.
- The tidiness score is based on statistics measured from hand-made meshes. It holds no geometry. Sources and licenses are in
  [docs/PATTERN_SOURCES.md](docs/PATTERN_SOURCES.md). Credits for the CC-BY works, changed by reducing them to statistics
  ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)):
  - Sprite Rig (CC-BY) Blender Foundation, studio.blender.org
  - Rain Rig (CC) Blender Foundation, studio.blender.org
  - Pack Bot Rig (CC-BY) Blender Foundation, studio.blender.org
  - Security Bot Rig (CC-BY) Blender Foundation, studio.blender.org
- The whole-mesh solvers are CurioMesh's own implementations of published papers, listed in [docs/ATLAS_Q_REFERENCE.md](docs/ATLAS_Q_REFERENCE.md). No QuadWild source is copied or included.

## License

CurioMesh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

Versions up to 0.5.1 were released under MIT. From 0.6.0 on, CurioMesh is GPL-3.0-or-later, the same family of licence Blender itself uses.

<!--
Blocks to switch on when the branches are merged. Do not publish them before:

## Auto button (branch `multires`, not in `next`)
One Auto button at the top of Quick Start and Auto Segments looks at the mesh and the free memory, picks the Use and the settings, says why, and starts.
Nothing changes until Apply.

## Multires Detail (branch `multires`, not in `next`)
A clean, low cage keeps its faces at level 0 and the shape of your mesh comes back as Multires levels (Blender's Subdivide and Reshape; your mesh is only read).
Offered for Animation, Subdivision and Sculpt Base. Needs wire pictures of the bust at level 0, 1 and 2: run regen.py on the merged commit.
-->
