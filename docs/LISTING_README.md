# CurioMesh

Quad retopology for scans and game assets: CurioMesh turns a dense sculpt, scan or high-poly model into a clean, mostly
all-quad, low-poly mesh, one loose part at a time. A camera check compares the result with your original from six sides
and says how much of the view changed, so you can redo only the parts that still look wrong. Everything runs inside
Blender on your computer.

This is the extensions.blender.org build. It contains Python code only.

## Use

1. Select a mesh and open `3D View > Sidebar > CurioMesh`.
2. Under **What is it for?** click a tile (**Game** for most models, or **Auto**), then press **Start**.
3. Read the verdict, then use **Adjust**, **Fix** and **Finish**. Nothing changes in your scene until **Apply**.

## Features

- **Auto Segments:** reduces each part to the fewest faces that keep the amount of view change you ask for. There are
  three layout styles: Fewest Faces, Balanced and Even Rows.
- **Uses:** Low Poly, Game Asset, Animation, Subdivision, Scan Cleanup, Sculpt Base and Hard Surface.
- **Adjust:** the Quad Size, Rows and Organize sliders, and Tidy Up with its Loop Quality and Loop Flow meters.
- **Fix:** the Issue Finder (faces turned inward, folded over, crossing or sunk), Redo this spot, Keep original here,
  and Fix Selection in Edit Mode.
- **Hard Surface:** Clean Panels, Even Cylinders, and hard edges marked on Apply.
- **Shape the flow:** edge loops, deformation lines, guide curves, Detail Markers, Mirror and Symmetrize, and the Flow
  Editor (experimental, off by default).
- **Finish:** Apply, Bake Detail (normal or normal + color maps), LOD sets, Multires Detail and Export for Game Engine:
  presets for Unreal, Unity, Godot, Roblox, glTF/web and DirectX engines (CryEngine, O3DE), or Custom (FBX, glTF or OBJ).
- **Job folders:** Export Job writes a folder with a Python script, `run_job.py`, that you run yourself in any
  Blender that has CurioMesh installed (`blender --background --python run_job.py`). Import Result brings the result
  back.
- **Advanced tab** (off by default): Regions (redo a boxed area on a fixed seam, Fill Loop), Blockout volumes and
  Redo Inside Shapes, the Smart Solver (Atlas-Q), remembered fixes you can look for, apply, and export or import as
  JSON files, Materials & UVs and texture sets, the older Classic Remesh, Classic Guided and QuadriFlow methods, the
  quality heatmap and viewport HUD, Work Live, Import Job Result, Judge After Baking (experimental) and Diagnostics
  (Measure Active Mesh, Check This Computer).

## Notes

- The **Advanced** tab (off by default, in the add-on preferences) holds every setting and the older tools.
- **Background Blender Processes** (add-on preferences, off by default): when you turn it on, CurioMesh may start
  background copies of the same Blender, never any other program. They run only while a job runs and close when it
  ends. Two features use them:
  - **Use Several Cores** (Advanced tab) measures the parts of a big job in several of them at once.
  - The **Even Rows** style runs Blender's own QuadriFlow on whole pieces in one. QuadriFlow can't be interrupted and
    takes long on a full-detail piece, so this keeps your Blender usable and lets **Stop** end it at once. With
    background processes off, Even Rows is not available and the panel says so.
  - The **QuadriFlow** method (and Classic Remesh with QuadriFlow) runs Blender's own QuadriFlow inside your Blender,
    like Remesh > QuadriFlow does, on one reduced part at a time; Blender is busy while each part runs.
- Blender stays usable while a job runs: the work is done in short steps from a timer. A few single steps (reducing
  a part, pairing, the topology check) can still hold Blender for up to about 3.5 seconds on a mid-size mesh; the wait
  cursor shows while they run, and **Stop** takes effect after them. **Apply** is available once the job has finished,
  or after **Stop**. The first time you move Quad Size or Rows after a result, the other sizes for those sliders are
  worked out the same way, and the panel says so while it happens; single steps there can hold Blender for up to
  about 7 seconds at a time on a 50,000-60,000-face mesh, and Blender is then left idle at least as long before the
  next step. Working out all the sizes can take several minutes; **Stop Working Out Sizes** under the sliders ends it
  at any time (the sizes already made stay). Auto alone does not start this.
- The Batch panel's report goes to the extension's user folder. Issue reports and exports go where you pick in the
  file browser.
- Remembered fixes and resume points are kept in the extension's user folder. **Remember This Fix** also saves the fix
  on the mesh, as a custom property `curiomesh_kept`, so it travels with the .blend; **Forget Kept Fixes** removes it. A job's working files go in a folder
  there too (`jobs`) and are deleted when the job ends; folders left by a Blender that quit mid-job are cleared later.
  Resume points of a stopped job are kept for 3 days; **Clear Saved Progress** in the add-on preferences deletes them.
- Segments you edit by hand (merge, split, paint, rename, lock, a Use, Bigger/Smaller Quads on a part) are saved on your object, as a face attribute
  `curiomesh_segment` and a custom property `curiomesh_segments`, so the next run and a saved .blend keep them. Finding
  segments alone writes nothing on your object. Delete both to forget them.
- The viewport HUD is off by default: turn it on with **HUD On in New Scenes** in the preferences (or the HUD button on
  the Advanced tab). Its timer runs only while the HUD, the heatmap or Live is on.
- Ideas credited: fitting each source point to its nearest vertex, shading read from the nearest source points, and
  the finite test cases come from ideas by cav3dw3ll3r (https://github.com/cav3dw3ll3r).
- **Report a Problem** (Advanced tab, Issue Reports) opens GitHub's new-issue form in your web browser, with your
  Blender and CurioMesh versions filled in (nothing else about your computer). Filing it needs a GitHub account. The
  add-on itself never connects to the internet.
- Bundled data: `triadq/patterns.json` holds statistics measured on CC0 and public-domain meshes (see
  `docs/PATTERN_SOURCES.md`). `triadq/cases_table.json` holds camera-check results measured on CC0 Poly Haven models. The
  icons are CC0 (`icons/LICENSE.txt`).

Issues: https://github.com/CurioCrafter/CurioMesh/issues

License: GPL-3.0-or-later.
