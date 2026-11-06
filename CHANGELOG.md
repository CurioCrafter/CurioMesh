# Changelog

## 0.9.8.1 - unreleased

### QA bugs
- **register() twice** no longer raises "already registered": the second call unregisters first and starts clean.
- **Selected Faces Only** is switched off in 0.9.8 and hidden in the panel (its seam is not stitched yet); Start now
  ignores a leftover setting instead of storing a selection that did nothing.
- **Base Quads on a mesh of many loose pieces**: when the result is far above Base Quads, the preview line says the
  loose pieces each keep a few quads of their own.
- **Reduce To / LOD set** that cannot reach its size on a mesh of many loose pieces now says so in the result box and the
  LOD report.

## 0.9.8 - 2026-11-06

### Hard surface
- **Hard Surface** use (under More Uses): Game Asset at Detail 5% with the two hard-surface rebuilds on and every hard
  edge marked Sharp and Creased on Apply. The two rebuilds are also switches in Options; they are off in every other use.
- **Clean Panels**: a part made of flat faces (box, plank, plate, bracket, bevel strip) is also offered laid out by
  panels: each convex flat area one grid along its own edges (a rectangle an a x b grid, a round cap a grid with four
  corner poles, never a fan), a flat area round one hole a ring of quads round it, other flat areas cut and split so
  they stay all quads. Every panel border stays an exact edge loop and both sides of it agree, so it is watertight. A
  chamfer strip comes out as one row. The camera keeps a layout only where the view holds; it counts as half its faces
  when weighed against a reduced size.
- **Even Cylinders**: a turned part is also read by the longest path through the tree of its rings, which holds where
  the old ring walk wandered (on the hydrant 16 parts that were 8 to 20% off now fit within about 1%), and a scanned
  tube is read wall by wall so its inside stays hollow. A visibly round part (radius at least 5% of the model) never
  drops below about 16 sides: the scanned concrete pipe that became a 12-sided prism now keeps 192 faces and 3.9% view
  change instead of 111 faces and 5.6%.
- **Round parts at least 16 sides** in every use: a turned part whose radius is a tenth of its size or more is no
  longer swept with 8 or 12 steps, and a hollow profile is never simplified until its walls fold onto each other.
- **Faces lying on each other count as a problem**: the result line says "New problems: 22 faces lying on each other"
  instead of "No new problems".
- **Hard Edges** on Apply (Options): Leave, **Mark Sharp + Crease** (no extra faces, a Subdivision Surface keeps the
  edge), or **Holding Edges**: a loop on both sides of every hard edge at a % of the next face or in millimetres
  (Blender's bevel, two segments, square profile, so the shape does not move).
- Measured (Game Asset, Keep Detail 6%, rebuilds off -> on, same branch): overhead crane Issue Finder total 5,612 ->
  5,392, inward+folded+crossing 481 -> 425, slivers on flat parts 1,084 -> 1,014, Tidiness 40 -> 42; fire hydrant
  inward+folded+crossing 234 -> 145 but poles 872 -> 1,430; old compressor worse (inward+folded+crossing 208 -> 287,
  slivers on flat parts 506 -> 682); treasure chest a little worse; drill press, barrier and four other scans
  (column, stool, barrel, pipe joint) unchanged: the camera turned the rebuilds down. 0 to 25% more time. That is why
  the rebuilds are opt-in.

### A tidier panel
- **One path, in order.** The CurioMesh panel now shows only what the step you are on needs: pick a mesh, **What is
  it for?**, **Start**, then **Adjust**, **Fix** and **Finish**, each appearing when it can do something. Before a
  result the panel has about 20 controls (it had 40 with More Options open); with a result about 25 (it had 97).
- **What is it for?** is eight picture tiles with CurioMesh's own coloured icons: Auto, Game, Low Poly, Animation,
  Subdiv, Scan, Sculpt and Blockout (Hulls is under the small arrow). **Auto** lets Start pick the Use from your mesh
  and says what it picked and why; it no longer resets Mirror.
- **Detail** is one dropdown: Keep Most, Keep a Lot, Balanced, Lighter, Lightest, Custom %, Share of Faces or Exact
  Face Count. Picking one keeps your Use instead of turning it into Custom.
- **Two popovers** hold the rest of the setup: **Shape the flow...** (edge loops, flow lines, joint lines, detail
  markers) and **Options...** (Rows, Cleanup, Multires, UV seams, Output).
- **Cleanup: Quick / Normal / Thorough** replaces the three tidy switches (Normal turns on Redo Rough Spots, Thorough
  also Redraw Messy Quads and Tidy Quads). The switches are still in the Pro tab one by one.
- **A verdict** under the result, with a green, amber or red icon: "Good for game asset", "Check 2 spots" or
  "5 spots with bad faces left", the numbers in grey under it, and the next thing to do. Tidiness gets a word
  ("like a quick hand retopo"). The thresholds are first guesses, kept in one table (`verdict.py`).
- **Adjust**: Quad Size, Rows (was Organization) and Tidy Up (was Organize) with words at both ends, five icon-only
  view switches, and Fine-tune... for the three Tweak sliders and Most Detail Loss. It warns when the sliders would
  replace parts you changed by hand.
- **Fix** walks the bad spots one at a time: **Redo this spot** remakes the part under it with smaller quads, **Keep
  original here** shows the faces it adds, asks first, and **Put reduced back** undoes it. Keep Original on all parts
  also shows its cost and asks. The part list, a short part card, and the hand tools and part finder in two popovers.
- **Finish**: one big Apply, New Object / Replace Mesh, the only Bake row (its size and texture sets behind the gear),
  LOD set... and Discard. Export for game and the star / report buttons after Apply.
- **CurioMesh Pro tab** (the Pro button in the panel header, or the preference): Engine (Style, Method, Output, the
  cleanup switches, what is kept, Multires, Tweak), This Computer, Inside Shapes, Show It How, Issue Reports, Classic
  Guided (Whole Mesh, Points Only, Work Areas), Classic Remesh and Regions. A file saved with a Classic Guided mode
  says so in the main panel, with a button to open it in Pro.
- **Plain tooltips** on every setting and button: what it does, then when to use it or what it costs, in two short
  sentences. "Segment" is "part" everywhere you read it.
- Every operator name and saved setting is unchanged, so scripts, agent skills and older .blend files keep working.
  New: `curiomesh.pick_use`, `curiomesh.issue_redo`, `curiomesh.issue_keep_original`, `curiomesh.issue_put_back`.

### Organize
- **Organize** (a slider under the result, next to Quad Size and Organization): about the same faces, laid out tidier.
  Moving it the first time works out four strengths in the background (a few seconds on a bust, under ten on a
  15,000-face machine); after that dragging only swaps meshes. 0.25 redraws the messy patches around poles as plain
  grids and spins edges to walk the poles left together; higher strengths then let the quads settle into even rows,
  held on your mesh, with hard edges, borders and the mirror seam kept where they are. A mirrored result is organized
  on one half and mirrored again, so the seam stays exact.
- **A meter** under the slider: "Tidiness 38 -> 56, view changed 6.2% -> 7.6%, poles 1,077 -> 722, faces 2,607 ->
  2,410". The Issue Finder and the red bad-face overlay follow the strength on screen, and Apply writes it.
- **Most Detail Loss**: Organize goes no further than this much more of the view changed than at 0 (it says where it
  was held).
- **Loop Flow**: draws the edge loops of the result in a colour each, and its poles as dots (blue: three edges, red:
  five or more).
- Measured at Organize 1 on five free test models: Tidiness up by 12 to 18 points on the bust, lion, horse (with
  Symmetrize) and barrier and by 4 on the compressor; poles down by a third on the organic ones; 4 to 13% fewer faces;
  the view changed by 0.8 to 1.4 points more; Issue Finder totals down on all five, though 7 to 80 bent quads appear
  where there were almost none.
- **Loop Quality**: a 0-100 score of the edge loops the way an artist reads them (straight rows, no zig-zags, evenly
  stacked loops, even quad widths, square corners, few and spread-out poles, few loops that die after a few quads),
  shown in the Tidy Up line: "Loop Quality 16 -> 32 (straight 0 -> 17, even spacing 39 -> 56)" on the bust.
- **Straighter rows at the top of the slider**: 0.5 now settles the quads fully and 0.75 and 1 straighten the loops
  (each corner moves onto the smooth curve through its loop and to the middle of its neighbours along it, held on
  your mesh, with the same guards). Drawn guide curves and edge loops stay where they are. Honest limit: on the bust
  this adds only about 3 Loop Quality points over the settle; with about a third of the corners being poles the loops
  are three or four quads long, so the rows cannot get much straighter without fewer poles.

### Flow Editor (experimental, off by default)
- Turn on **Flow Editor (Experimental)** in the add-on preferences and **Draw on the mesh** appears under Shape the
  flow. Draw on a finished result and the quads around the stroke are redone on their seams; the rest stays as it is.
- **Draw** recognises what you drew: a closed loop becomes a **Ring** (a clean closed edge loop with rows round it),
  a line along a sharp crease an **Edge**, any other line a **Flow** line the rows follow. Change it with one click
  (or F, R, E while drawing). On a test set of 57 drawn shapes (clean and shaky circles, ellipses, open arcs, loops
  with a 5-15% gap, zig-zags, figure-eights, strokes along and across a crease) it was right on all 57.
- **Flow**, **Ring** and **Edge** force a kind; **Place Circle** (drag to size, Shift for an ellipse); **Eye**,
  **Mouth** and **Nose-Mouth** place those shapes where you click; **Erase** and **Erase All**. Strokes are curves in
  the guide list, saved with the .blend and used by Start too; with a mirrored result they work on both sides.
- Not yet measured inside Blender: on the bust outside Blender, a ring round the forehead redid 268 faces into 417 in
  1.6 s with five closed loops round it, and a flow line redid 79 faces in 0.5 s, both with clean seams.

### Fixes
- **The Apply button was missing in 0.9.0 to 0.9.7** for everyone who kept the default sidebar tab: the "5. Apply" panel
  was never registered. The new Finish section has it, and a new test (`tests/blender_registration.py`) checks that every
  panel and button CurioMesh defines is registered and gone again after disabling.
- **A crash in Blender 5.2** (inside Blender's attribute carry-over after QuadriFlow) is avoided: CurioMesh only asks
  for it when the mesh has colours or vertex groups to keep.
- **From the Blender Extensions review:** background work runs through one task runner; helper Blenders get the add-on
  folder from their own environment instead of a `sys.path` change; no saved data uses pickle; the computer's graphics
  card is read from Blender instead of running `nvidia-smi`; the depsgraph handlers are gone (the overlay and paint
  watch check cheaply when they draw or only while painting); saved fixes and resume points live in the extension's own
  user folder. "TRIAD-Q Lite" is now just "TRIAD-Q". The GitHub zips keep the native core and its speed.
- A feature tour for testing in Blender (`tests/blender_feature_tour.py`), with an inventory of every button.

### Auto Segments
- **Reduction no longer pushes a thin ring or flange through the part it sits on.** The native reducer now refuses a
  collapse that makes its new triangles pass through other triangles of the piece (more crossings than before the
  collapse). Taken alone at the same sizes, the crane's ringed post went from 37-39 crossing pairs to none and the
  drill press flange from 22-55 to none. The test only runs where a surface facing the other way is near (thin
  rods, rings, plates, parts close together), and each nearby triangle is measured once per collapse; taking a whole
  model down is still about 2.5-3 times slower than before (crane 1.1 to 2.7 s, ship pinnace 2.7 to 8 s).
- **Repair remakes parts one size up before filling crossings spot by spot**, and parts still turned inward or folded
  over after untangling are remade too; a size up is kept only if it leaves fewer such faces. The fills used to spend
  the whole 30 s on spots they could not clear before any part was remade: on the chess set the knight head went from
  47 inward and 20 crossing faces to none, and the run from 50 to 16 s. Giving the fills more time (up to 120 s) was
  tried and cleared no more parts on four of the six test models, so remakes and fills share the 30 s, counted from
  after the source is checked.
- Measured on six hard-surface test models (Game Asset, Keep Detail, 6%): serious faces (inward, folded, crossing)
  went from 1,969 to 1,775 in all (chess set 71 to 0, compressor 246 to 221, pinnace 579 to 466,
  chest 246 to 233; crane 430 to 451 and drill press 397 to 404 barely moved), at the same face count overall (crane
  +5%), measured with the repair taking 30 s for remakes and 30 s more for fills (pinnace 202 s instead of 134 s;
  with the shared 30 s it took 163 s for the same counts). Most of what is left sits where a part is thinner than its
  faces or touches another part back to back.

### Issue Finder
- **Touching parts** is a kind of its own: a face on its own part's surface with another part pressed against it back
  to back was counted as turned inward. It now is "touching parts", a note that does not count as a bad face, is not
  painted red and is not repaired. A face really turned inside out (the same part on both sides of it) stays inward.
  On the test results 5-20% of the inward faces were touching parts (drill press 34 of 202, crane 18 of 203).

## 0.9.7 - 2026-11-05

### Auto Segments
- **Fewer long thin quads on flat areas** (Keep Detail; Balanced uses the same step): before pairing, flat areas that reduction left as
  long thin triangles get a few points inside, graded from small at their edges to large in the middle, so flat
  plates get far fewer fans of thin spokes from bolts and corners. Their outline stays exactly where it was. On four
  test models the face count stayed within 2%, thin quads on flat areas dropped by a third or more on two of them (the
  other two barely changed), and the Issue Finder total went down or stayed level.

### Symmetrize, Detail Markers and Tweak
- **Symmetrize** (next to Mirror): mirrors even when the two sides differ, as on scans and sculpts. The result is built
  on one side (Keep Side: Auto, + or -) and mirrored, so it is exactly symmetric. **Set Plane** puts the plane at an
  offset you type instead of finding it. The panel says how far apart the sides were ("the sides differ by up to
  38.8 mm") and, with a result, how clean the seam is.
- **A cleaner mirror seam**: a vertex that strayed over the plane goes back, seam vertices closer than a tenth of an
  edge are merged, and faces left without area or lying in the plane are dropped. The Issue Finder's seam counts on a
  horse statue: 4 faces crossing the plane and 2 concave faces before, none after; no open, non-manifold or duplicate
  seam edges on it or a bust, at every Quad Size.
- **Detail Markers** (was Paint Density): paint **More** or **Less Detail** in Weight Paint, or add marker balls at the
  3D cursor and scale them. They steer where the faces go at every Quad Size: on a bust, a ball on the mouth and one
  on the hair doubled the faces per area in the mouth against the hair at all three sizes measured. With Mirror,
  markers on either side count.
- With markers, the search now also tries sizes in between its steps, so it lands closer to the Detail.
- **Tweak** (a fold under the result): **Rows on Round Areas** (how readily round, smooth areas become rows of quads
  that follow their curve), **Flat Area Size** (how big the quads inside flat areas may get) and **Avoid Poles** (spin
  edges at poles, Keep Detail). Each reruns the search with Auto-Update; what was measured is kept, so it takes
  seconds on a bust. On a sofa Rows on Round Areas gave 1,741 / 1,682 / 1,866 faces at 0 / 0.5 / 1; Avoid Poles off
  left 22% more poles on a bust.

### Know the cost before you press Start
- **Cost tags** in the panel: Bake Detail, LOD Set, Clean Rows / Field Rows / QuadriFlow and Multires say "Slow" or
  "Heavy" with the time they add on this computer, and whether they block Blender.
- **A fuller time estimate** before Start: it now counts the style's slowdown and the bake ("plus about 1 minute of
  baking that blocks Blender").
- **Start, Make LOD Set and Auto ask first** when the job is rated large or expected to take over 10 minutes, and Apply
  asks when its bake alone takes over 2 minutes. The dialog says the time here and the cheaper ways (Effort Light, Keep
  Detail, a smaller bake, Export Job). "Don't ask again this session" is in it.
- **A small-PC notice** in This Computer (4 cores or fewer, 8 GB or less, or no graphics card for Cycles).

### Quad Size and Organization sliders
- **Quad Size**, under the result: smaller quads to the left, bigger to the right. Each step is a whole result of the
  search at another Detail, so it holds its own amount of change and puts the faces where the shape needs them. The
  sizes are worked out in the background once the search is done, reusing what it measured, so most take seconds;
  until they are ready the nearest finished one is shown, with a note.
- **Organization**: fewest faces (Keep Detail) to the left, Balanced in the middle, even rows (Clean Rows) to the
  right. In between, loose pieces change over one at a time, smallest first, and the whole object is looked at again.
- Dragging either slider is a live preview: each size keeps its own preview mesh, so a step is a mesh swap (about
  1 ms on a 17,000 face bust), at most ten times a second. The segments of the result are found once the slider
  rests, and the sizes still being worked out wait while you drag.
- A size that another one beats on both faces and change is skipped, so the face count only ever goes one way.
- **Bigger Quads / Smaller Quads** for the selected parts only, the rest stays: a whole loose piece takes its faces
  from the next size, a segment of a bigger piece is remade for Reduce at a lower or higher Detail and kept only if it
  really has fewer (or more) faces.

## 0.9.6 - 2026-11-05

### Game engines
- **Texture Sets** on the bake (1 to 16): Normal + Color onto several sets of maps, each its own material, with whole
  parts kept together in sets of about equal area. Lit in Godot, three 1024 sets shade like one.
- **More engines in Export for Game Engine:** Roblox (FBX unit scale, tangents, 20,000 triangles per mesh, 1024
  textures), Web and glTF (.glb for three.js, Babylon.js, Bevy, Sketchfab), CryEngine and O3DE (DirectX normal maps),
  and **Custom**: FBX, .glb or OBJ, OpenGL or DirectX normal maps, a triangle limit and a largest texture size.
- **Auto-split:** a mesh over the engine's triangle limit is written as pieces under it, cut along texture sets and
  parts. FBX and OBJ files now point at their PNG maps, so engines link the textures on import.

### README
- The star chart is drawn from GitHub's own star dates, as a quad mesh (`scripts/star_chart.py`).

## 0.9.5 - 2026-11-05

### Game engines
- **Export for Game Engine** (File > Export > CurioMesh Game Asset, and a button after Apply): the selected meshes and
  their baked maps for Unreal (FBX, PNG maps, normal map green flipped for DirectX), Unity (FBX turned and scaled for
  Unity, PNG maps) or Godot (.glb with the textures inside). The bake's triangles are kept, and your images are not changed.
- Baking is now tested in Unreal 5.7, Unity 6 and Godot 4.7, lit from 8 sides next to the source, not only by
  re-importing in Blender. Imported with default settings, the exported files came closest to the source in all three. A hand-made export
  lost out: Unreal needed Flip Green set by hand, a plain glTF export let Godot choose other triangles, and a plain FBX
  export carried no textures.

## 0.9.4 - 2026-11-04

### Skills for AI agents
- **Agent skills** in `skills/`: `curiomesh-retopo`, `curiomesh-fix-bad-faces`, `curiomesh-batch-and-lods` and
  `curiomesh-blender-mcp`. An agent can retopologise a file headless, read a JSON report (faces, quads, view changed,
  Issue Finder totals per part), fix bad faces in a measure-fix-measure loop that keeps a change only when it lowers the
  bad-face count, make LOD sets and batch many files.
- Two headless scripts, `curiomesh_run.py` (Auto Segments to a .glb/.fbx/.obj/.blend plus a report) and `curiomesh_fix.py`
  (Fix Selection tools on any mesh, before and after counts per kind).
- Install in Claude Code with `/plugin marketplace add CurioCrafter/CurioMesh` then `/plugin install curiomesh-skills@curiomesh`,
  or unzip the new `curiomesh-<version>-agent-skills.zip` from a release into a skills folder. It is not in the
  extensions.blender.org zip.

### Blockout and Hulls
- **Blockout** (a new Use): every loose part of your mesh becomes the low-poly solid that fits it best: an oriented
  box, a cylinder (axis found from the part, capped with quads) or a ball (a quad sphere, stretched to an ellipsoid),
  or a small convex hull when that matches clearly better. The camera picks, from the part's own six views. A big
  part no single solid fits is cut at its creases by the segment finder and each segment gets its own solid. A part
  that still looks wrong keeps a reduced copy of itself, so nothing comes out crushed; the result says which parts
  did. All quads except the hulls. Each solid takes the material most of its part had.
- **Hulls** (a new Use): every part becomes its own reduced convex hull, for a quick faceted low-poly look. Concave
  parts are cut at their creases first.
- **Primitive Sides** sets the sides of each cylinder and the detail of each ball (small parts get half), and the
  size of the hulls. **Each Part Its Own Object** makes Apply write one object per loose part.
- Apply, Undo, Bake Detail and Make LOD Set work as with the other Uses; in a LOD set each level halves the sides.

### Segments you shape yourself
- **Live segment size.** Segment Detail and the new **Largest Segment**, **Smallest Segment** and **Crease Strength**
  sliders recolour the segments while you drag, with the segment count beside them. The colours come from the merge
  tree the last find kept, so a change shows in milliseconds even on a big mesh; a moment after you stop, the
  segments are found for real (borders settled on the creases) and kept. **Border Smoothness** sets how round the
  borders are where no crease holds them; it shows when the segments are kept.
- **Each mesh keeps its own segment settings.** They are stored on the object and saved with the .blend.
- **Paint Size.** Paint the CM_SegmentSize vertex group in Weight Paint: red makes smaller segments there, blue
  larger ones, green leaves it to the sliders. The colours follow each stroke.
- **Edit segments by hand.** **Split in Two** cuts the selected segment along its own natural border, no line needed.
  **Grow** and **Shrink** move a segment's border with a brush (Ctrl paints the other way). **Lock** keeps a segment
  face for face when the size settings change; segments you merge, split, grow, shrink or paint are locked for you.
  In Review Segments, M merges, K splits in two, L locks, G and S grow and shrink. **Undo Edit** (Ctrl+Z in
  Review Segments) goes back through the last 20 hand edits, and the next Auto run uses the edited segments.
- **A remade segment never loses its shape.** A Redo or Use remake whose camera check is much worse than the
  segment had (or whose surface or size shrinks far below your mesh there) is refused: the segment keeps what it
  had and the panel says so. Before, the check only asked for more faces, and a segment that had never been
  measured (one you split, merged or painted) had no check at all, so a round tank could come back as a few flat
  triangles.
- **No segment finer than the result can hold.** Where the result's quads are bigger than a found segment (painted
  very small, for example), the segment joins a neighbour instead of being marked on faces that are not it. Grow,
  Shrink and Paint New mark the result through the faces of your mesh, so a segment covers the same surface on both.
- **Locked segments stay locked when the size changes.** A small segment that is dropped no longer joins a locked
  neighbour, which made the locked one bigger and lose its lock.
- **Redo and Use remakes run to the end again.** A part redo stopped after its first step and sat at "Remaking..."
  for good.

### Fixes
- Without the native core (blocked by the system, or the Python-only build) fitting to your mesh stopped with an
  error. It now skips only its crossing test.
- Report a Problem filled in "CurioMesh ?" in the Blender 4.2+ zips; it now reads the version.

## 0.9.3 - 2026-11-04

### Finding what is still wrong
- **Issue Finder** (an "Issues" box in step 4, Fix Parts, after an Auto run): every problem area in the result is
  found and ranked, worst first. It counts faces turned inward, sunk into your mesh, crossing, folded over, bent,
  flat, duplicated or lying on another face (the check the run already makes, not repeated), and also tight or
  untidy topology: slivers, pinched faces, poles (a corner with seven or more edges, or two or fewer) and concave
  quads. Faces that touch or lie close together are one spot, scored by how bad the kinds are and how many faces;
  real defects are never lumped in with untidy faces next to them. **View** moves the 3D view to a spot,
  **Select** selects its faces on the preview in Edit Mode, **Next Issue** and **Previous** step through them, and
  with Show Bad Faces the spots are painted orange (the current one yellow). Each spot names the part it is in, as
  the Fix Parts list numbers them.
- **Save Issue Report and Compare With Report...** keep a record (a JSON file next to your .blend, or where you
  choose) and, after a fix and a new run, say how many of its spots are fixed, still there or new, with the
  totals per kind.

### Fixing bad faces yourself
- **Keep Original Where Bad** (in the Issues box, and in step 5, Apply, when problem spots were found): every part
  with a problem spot gets your original faces back, unchanged. The reduced version stays in that part's tries, so
  Fix Parts switches it back with one click. After any part is redone or put back, the problem spots are looked for
  again, so the Issues box always matches the result on screen.
- **Fix Selection** (a new panel in Edit Mode, CurioMesh tab), for any mesh: select a rough area, or nothing for the
  whole mesh. **Find Bad Faces** checks it (plus a few rings around it, set by Grow) and leaves only the bad faces
  selected, with a count per kind. Then fix only those: **Clean Up** (duplicate and flat faces), **Flip Inward**
  (only where that makes a face point out), **Untangle** (corners moved back onto your original), **Relax** (corners
  smoothed, then put back on the original) or **Patch** (bad faces removed and the hole filled again from its edge).
  **Fix All** runs them in turn and undoes any move that leaves more bad faces. Each is one Undo step and says how
  many bad faces there were before and after. The original is the mesh Auto last worked on, or the one you pick.
- After Apply, a small box links to the GitHub page and to a problem report with your Blender, CurioMesh and system
  versions already filled in.

### Fewer bad faces
- **Redo Rough Spots** (Auto settings, off by default: it adds up to a minute for a few percent fewer bad faces):
  after the final check, the spots it still flags get a small region each, grown a few rings of faces round the
  spot. The region's inner corners are put back towards your mesh, part or all of the way, and the way that leaves
  the fewest bad faces there is kept, if it leaves fewer than before and bends no quad. The region's edge never
  moves, so it joins the rest exactly, and the face count doesn't change. If the regions together cross or bend a
  face outside them, only the regions touching it go back; the rest are kept if nothing else broke and the camera
  still agrees. The summary says how many spots were redone and how many bad faces fewer there are.
- **Copies land where they fit best.** A part that is almost symmetric across its own plane also fits turned the
  wrong way onto its mirror image, within the copy tolerance, and the first placement found was taken: such a copy
  sat a little off and some of its faces turned inward. Placements are now compared and the closest is kept.
- **Fitting to your mesh no longer tangles the quads.** Each corner is pulled onto its own share of your mesh, so on
  thin walls and in creases neighbouring corners were pulled different ways and the quads between them folded over
  each other or passed through each other. Corners of a quad the fit folded, bent or pushed through another quad now
  stay where they were. On big rough scans this leaves far fewer crossings for the final check to untangle.

- **Turning folded faces back now finishes on big scans.** The step that moves the corners of inward and folded-over
  faces back used to stop at a 15 second limit on very big rough scans, so many spots were never visited and two runs
  could differ. Corners that share no face are now judged together, which makes the step 10 to 30 times faster (a
  few seconds on a 200,000-face scan) with exactly the result the old step reached when it had the time. It always
  does its full rounds, so the same model gives the same result; a 60 second limit stays as a safety net.

## 0.9.2 - 2026-11-03

### Faces that turn inward or fold over
- **Parts are reduced by CurioMesh's own reducer, not Blender's Decimate**, whenever the native core is
  there and no density is painted. Decimate lets a face turn almost 90 degrees in each collapse, and many
  such turns fold it over. The reducer now refuses a collapse that turns a face more than 60 degrees at
  once, more than 90 degrees from where it faced in the source, or more than 120 degrees from the source
  surface nearest it (a face slid over to the back of a thin plate), or folds two faces around the new
  point over each other. So fewer folds are made, and less has to be turned back at the end.
- **Show Bad Faces** (the red "Bad" button in the segment view, on by default): after Auto Segments, the faces
  the defect check flags (turned inward, sunk into your mesh, crossing, folded over, bent, flat, duplicated or
  lying on another face) are painted red, and the summary line gives their count. It reuses the check the run
  already makes.
- **A regression test for bad faces** (`tests/test_regression_gate.py`, runs without Blender or the native core):
  rebuilt parts of a triangle tube, their copies and a frozen result must come out with no faces turned inward
  and no defects; with the native core it also checks them against the source.

### Tidier quads
- **Keep Detail spins edges at poles** before the final check (Tarini et al. 2010's edge rotation, as
  Redraw Messy Quads already did): an edge between two quads turns where that leaves fewer corners with more or
  fewer than four quads. No corner moves and no face is added, so rows run longer. Kept only if the camera
  change stays within a point of what it was and inside what Keep Detail counts as a fit.
- **Clean Rows lands near the face count asked for** with Reduce To. QuadriFlow is asked again, up to three
  times, following what the earlier tries gave, since the small parts that are paired and closed don't shrink
  with it. The try nearest the amount is kept.

### Faster
- **Use Several Cores also shares out big parts.** A loose part of more than 120,000 faces is cut into regions
  that the helper Blenders measure at once; before, one big part kept one core busy while the others waited.
  The region borders are kept as they are while each region is reduced, then joined again, so there are no gaps
  or T-junctions. Regions skip the wire, fin and cloth rebuilds, whose fit alone can take a minute on a part
  this big, and are not looked for as copies. Results can differ a little from one core, by a few percent in faces.

### Bake Detail
- **Textured models keep their colours.** A glTF material that is plainly opaque was treated as see-through
  (it reads as blended in Blender), so the colour bake skipped it and the result kept the old texture on new
  UVs. The chess board came out plain white. Only a really blended material, or one with alpha or
  transmission, is kept out of the bake now.
- **More texels per face on detailed parts.** When the new unwrap makes thousands of small islands, the gap
  left between them used up most of the texture. The gap now shrinks until the islands fill the texture, so
  curved parts get a sharper normal map and look less faceted. On a model baked before, the earlier bake's UV
  layer is left alone while it does this.

### Graphics card and native core
- **A smaller native file.** Each platform now builds only its own graphics-card backend (DirectX 12 on Windows,
  Metal on macOS, Vulkan on Linux), takes shaders in WGSL only, strips debug names and builds the card libraries
  for size. The Windows file goes from 7.2 MB to 5.0 MB; pictures are unchanged.
- **The summary says when the card drew pictures** ("Camera pictures of the big parts drawn on the graphics card"),
  not only when it failed.

### Repeatable results
- **`CURIOMESH_NO_TIME_LIMITS=1` gives the same result every time.** Two runs of the same model and settings can
  differ a little (Old Military Compressor, Game Asset: 16,149 faces both times but different vertex positions,
  251 and 248 self-intersections). The cause is the clock: the repair (30 s), untangle (15 s), redraw (240 s) and
  row (60 s) time limits stop at another point on a busier computer. With this environment variable set, every
  step runs to its end; two runs gave identical meshes, with 14 and 12 helpers and the graphics card on.

### Fixed
- **A second start can no longer leave "Finding the segments" stuck.** Moving Segment Detail (or any other start of the
  find) while a search, redo or remake is running no longer stops that job for a find that only adds labels; a find
  started over an older find stops it completely, including its waiting loop, before the new one begins.
- **Reduce To keeps the borders of a part cut into regions** (very big meshes), so it no longer opens gaps there.
- **No gap where a UV seam crosses a region border.** Straightening the seams could merge away a border vertex of
  a region that a seam passed through.
- **Small pieces that touch a cut part, or each other, at one corner** are no longer treated as regions, so they
  keep their rebuilds, copies and repair.
- **Edge spinning and quad removal work when the biggest part was rebuilt.** Every paired corner was frozen then,
  so neither changed anything.
- **Field Rows no longer stops Auto** on a part with a face that uses one corner twice.

## 0.9.1 - 2026-11-02

### Faces that turn inward, fold over or pass through each other
- **Turned parts and Field Rows face out again** on a source made of triangles. The vote that decides which
  way a rebuilt part faces read a triangle's normal backwards, so revolved parts (bolts, knobs, chess pieces,
  turned legs) and their copies came out inside out.
- **Copies face like their own source.** A part that matches its twin both turned and mirrored is now placed
  the way its own faces say.
- **No more quads folded across sharp edges** when leftover triangles are joined: a joined quad must face
  close to every face it replaces, or the triangle is left for closure. This costs a few percent more faces
  on hard-surface parts.
- **Folded faces are turned back** before the final check: corners of faces that face against the source or
  lie folded over a neighbour move to the middle of their neighbours, on the source. Kept only if nothing
  gets worse and the camera agrees.
- **A per-face defect check** (`triadq.checks.defects`): faces turned inward, sunk into the source, crossing,
  folded over, bent, without area, duplicated or lying on another face, each against the source.

### Faster
- **Use Several Cores** (on by default, under Effort): Auto Segments measures its parts in helper Blenders at
  once, as many as the free cores and memory allow (Light two at most, Normal half the free cores, Full all but
  one), never past 85% of the computer's memory. The parts are taken back in order, so the sizes picked are the
  same as on one core. Small jobs stay in one Blender; Stop, an error or quitting Blender ends the helpers.

### Graphics card
- **The camera pictures of big meshes (150,000 triangles and up) are drawn on the graphics card** when the computer has one that works: NVIDIA, AMD, Intel or Apple, through Vulkan, DirectX 12 or Metal, with no CUDA and nothing to install. On an RTX 3080 Ti that is 1.1 to 1.4 times faster at 170,000 triangles and about 3 times at 1.3 million; smaller meshes stay on the processor, where the card is level or slower. A whole run is only 1 to 2% quicker, since pictures are a small part of a run.
- **Almost the same pictures:** the card works in single precision, so on a depth tie a pixel can pick the neighbouring triangle. On real meshes the camera counts 0.02 to 0.4% of the pixels as changed at 128 to 192 pixels a side, against the processor's pictures.
- **Use Graphics Card** in the PC section turns it off; so does `CURIOMESH_GPU=0`. If the card errors, the processor takes over for the rest of the session and the summary says so once.
- The native core grows from 1.7 MB to 7.2 MB on Windows (wgpu and its graphics-API bindings).

## 0.9.0 - 2026-11-02

### Auto Segments: fewer faces, judged by a camera
- **About 20-55% fewer quads at the same Detail.** Leftover triangles join in twos along short paths of
  quads, then quads are taken away cheapest first until the camera says the amount asked for is used up.
  No connected part may pass its own limit. Big parts are also tried reduced in small steps and at sizes
  between the old ones, so they no longer land far under the amount.
- **Use menu:** Low Poly, Game Asset (the old default), Animation, Subdivision, Scan Cleanup, Sculpt Base,
  Custom. Each sets **Detail**, **Method** and **Output**. Detail runs from 0.1% to 80%, with a plain-words
  note of what it means.
- **Methods:** Auto, Pairing, Flat Quads, QuadriFlow, Grid Fill, Flow Fill, Field Rows (opt-in; Auto offers
  it on flat, thin and box parts) and Tris Only. **Outputs:** Triangles, Quads, Mixed. When a method can't do
  a part, Auto is used there and the summary says so.
- **Reduce To** a percent or a face count, beside Keep Detail; it lands within about 10%.
- **Styles:** Keep Detail, Balanced, and Clean Rows (even rows, all quads).
- **Turned parts** (pipes, tanks, bolts, rings) become rings of quads with capped ends. **Round and
  four-cornered areas** become concentric rows and grids, wherever the camera says they hold up.
- **Part cases:** each part is named (tiny, flat plate, thin sheet, box, tube, dome, rough scan, ornament,
  organic), and a small measured table skips the sizes that case never holds.
  Idea from [cav3dw3ll3r](https://github.com/cav3dw3ll3r).
- **Fit and shading:** finished vertices are fitted to the source part by part (2-10% less visible change
  at the same face count). Apply gives each face corner its normal from the source points on its own side
  of a hard edge, and marks edges past 60 degrees sharp.
  Idea from [cav3dw3ll3r](https://github.com/cav3dw3ll3r).
- **Tidiness score** from 0 to 100 at the end: how close the layout is to hand-made meshes (sources in
  `docs/PATTERN_SOURCES.md`). Read it next to the camera number. **Tidy Quads** is an option, off by default.
- **Redraw Messy Quads:** messy parts are redrawn as pole-free grids, and quads are taken where the camera
  sees least change. It is on for Animation and Subdivision, and left off on models of more than 40 loose
  pieces. Tidiness roughly doubles (bust 28 to 62) for 7-50% more faces and 2-5x the time, capped at 240 s.
  The summary says when it ran and what it took.
- **Multires Detail:** a clean low cage keeps its faces at level 0, and your mesh's shape comes back as
  Multires levels. It is Animation's default route (QuadriFlow cage plus Multires), offered for Subdivision
  and Sculpt Base, and available for any cage with **Add Multires Detail**.

### One guided panel
- **Five steps, one main button:** pick the mesh, choose what it's for, Start, fix parts if you want, Apply.
  The **Auto** button picks the Use and settings from the mesh and free memory, says why, and starts.
  Nothing changes until Apply, and Step 5 says what Apply wrote.
- **Progress** in words, with a time range and a Stop that ends the work within a fraction of a second.
- **Summary:** "2,508 faces, 86% fewer / Looks 94% the same / Tidiness 27/100". The parts, the hover card
  and the tries use the same words.
- **Classic Remesh:** the old one-pass remesher (Quick Start) is kept, closed by default under Guided, with
  no second Auto button.
- **Heat** colours are viridis (colour-blind safe), with a legend in the viewport.
- **Panel layout:** advanced options are folded away; nothing is clipped in a narrow sidebar; tooltips are
  short. The add-on is named plain **CurioMesh** (id `curiomesh`).

### Reviewing and fixing parts
- **Review Segments:** hover a part for its numbers, click to select, and walk the parts worst first. Give a
  part another Use, Method or Detail and only it is redone. Every try is kept, to go back to.
- **Segments inside a connected mesh:** big pieces split along creases, necks and steps (a bust into head,
  torso and plinth rings). You can merge, split with a line, paint, rename; this is saved with the object.
  Seams stay put, so segments join with no gap or T-junction.
- **Live Boolean:** put boxes, spheres or any closed mesh over the result. Union, Subtract or Intersect
  picks the faces to redo, live, as you move them.
- **Show It How:** fix a spot by hand and **Remember This Fix**. **Look for Known Fixes** finds the same
  pattern elsewhere and applies it after a camera check. Fix libraries are size-capped and validated.

### Guides, loops, paint and mirror
- **Guide Curves:** draw strokes or add curves, Grease Pencil or edge meshes, and quads follow them.
- **Draw Edge Loops:** closed loops and flow lines become strips of regular quads, with poles kept outside.
- **Deformation Lines:** a line across a joint becomes evenly spaced loops, with optional Crease and Sharp
  marks. It helps tight joints; on broad bends it makes little difference.
- **Paint Density:** Weight Paint the parts that need more quads.
- **Mirror** (Auto, X, Y, Z): one half is solved and mirrored; the seam sits exactly on the plane.

### Game assets
- **Make LOD Set:** `_LOD0`, `_LOD1` and `_LOD2` at percents you choose (50, 25 and 12.5% by default).
  Only the first does the full search.
- **Bake Detail:** Normal Map or Normal + Color, 512 to 4096 pixels. Normal + Color is the recommended
  route. Glass and other see-through materials are kept as they are.
- **Keep UV Seams** is an option, off by default: it can nearly double the faces.
- **Keep Copies as Instances:** repeated parts are written once and shared.

### Big scans and speed
- **Native Turbo core** (Rust, all cores, no GPU needed): the camera pictures, pairing, closing, reduction,
  crossing checks, lookups, fitting and shading. Results match the Python path. A 1.26M-face scan went
  from 47 to about 5 minutes, and Blender stays usable.
- **Effort** (Light, Normal, Full), **This Computer** check, and big pieces cut into regions and saved to
  disk, so a stopped run resumes.
- **Export Job / Import Job Result:** run a job on another machine with Blender only (guides, paint and
  Mirror included). See `docs/CLOUD_WORKFLOW.md`.
- **Rough As Texture** (rough scans judged on average; coast rocks 434,997 to 56,585 quads) and **Leave
  Out Hidden Pieces**, both off by default. **Judge After Baking** is experimental.

### Checks and repair
- After every run the result is checked against your mesh for non-manifold edges and vertices, flipped
  edges, duplicates, holes, faces passing through each other (any size of mesh), and folded quads. Only
  what the result adds counts.
- New problems are cleared on the spot or the part is remade one size up. Clearing crossings stops at
  30 s. What is left is named in the summary and selected on Apply.

### Security and packaging
- Job results and saved parts are validated, and a .npz that unpacks past 1 GB is refused. Remembered fixes
  never use a shared temp file. A planted `nvidia-smi` in the current folder is never run.
- Releases are built only by the release workflow, from the tag, with `SHA256SUMS.txt`. The workflow has
  least privilege, with actions and installs pinned.
- `THIRD_PARTY_NOTICES.md` ships in every zip. A separate Python-only, CC0-only zip is built for
  extensions.blender.org.

### Known limits
- Some small thin parts come out with folded slivers: quads bent about 90 degrees round an edge (ship
  pinnace 40 of 59,900 quads, treasure chest 1). They are named in the summary and selected on Apply.
- Judge After Baking can leave long thin slivers on big flat areas of a scan. It is experimental, off by
  default.
- Draw Edge Loops' "Selected Faces Only" is switched off: the region's seam is not re-stitched yet.
- The optional QuadWild script needs QuadWild installed separately.

### Credits
- Ideas from [cav3dw3ll3r](https://github.com/cav3dw3ll3r): fitting each source point to its nearest vertex,
  shading from the nearest source points, and the part cases.
- Test models: free CC0 models from [Poly Haven](https://polyhaven.com). Tidiness statistics: see
  `docs/PATTERN_SOURCES.md` for sources, licenses and credits. The ported triangle test and Rust crates are
  credited in `THIRD_PARTY_NOTICES.md`.

## 0.8.0 - 2026-11-01

- **Auto Segments** (new mode in the Remesh setting of the Guided panel). Press
  Update with no points and no face count: it reduces each loose piece as far as
  a camera says it can, looking from six sides at the original and the result,
  then makes all quads. **Change Allowed** sets how much of the view may look
  different. Faces go to the parts where they remove the most visible change, and
  small parts are judged by how big they look on the object.
- One quad that can't be closed no longer makes a whole piece split into many more
  faces: it goes back to two triangles and the rest of the piece closes normally.
  This also lowers the face counts of Flat Quads and Guided on some meshes.
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
  on a large organic scan and from about 31% to 2.8% on a noisy scan;
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
