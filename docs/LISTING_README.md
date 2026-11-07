# CurioMesh

Quad retopology for scans and game assets: CurioMesh turns a dense sculpt, scan or high-poly model into a clean, mostly
all-quad, low-poly mesh, one loose part at a time. A camera check compares the result with your original from six sides
and says how much of the view changed, so you can redo only the parts that still look wrong. Everything runs inside
Blender on your computer.

CurioMesh contains Python code only and no compiled or third-party code.

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
- Blender stays usable while a job runs: the work is done in short steps from a timer, most of them well under a
  tenth of a second. The longest single step is Blender's own Decimate on a whole part, one call that can't be cut
  up: up to about 0.7 seconds on a 60,000-face mesh. The wait cursor shows while it runs, and **Stop** takes effect
  right after it. **Apply** is available once the job has finished, or after **Stop**. The other sizes for the Quad
  Size and Rows sliders are worked out only when you press **Work Out Other Sizes** under the sliders; neither Auto
  nor moving a slider starts it. While it runs the panel says so. It works in the same short steps (again up to about
  0.7 seconds for one Decimate on a 60,000-face mesh) and leaves Blender idle at least twice as long as each step
  took, so it uses about a third of Blender's time at most. Working out all the sizes can take several minutes; **Stop
  Working Out Sizes** ends it at any time (the sizes already made stay).
- The result object carries a few CurioMesh custom properties (such as `curiomesh_engine`, `curiomesh_elapsed_ms`,
  `curiomesh_quad_ratio`, `curiomesh_strategy`) that the HUD reads; Debug Console Logs adds a few more.
- The Batch panel's report goes to the extension's user folder. Issue reports and exports go where you pick in the
  file browser.
- Remembered fixes and resume points are kept in the extension's user folder. **Remember This Fix** also saves the fix
  on the mesh, as a custom property `curiomesh_kept`, so it travels with the .blend; **Forget Kept Fixes** removes it. A job's working files go in a folder
  there too (`jobs`) and are deleted when the job ends; folders left by a Blender that quit mid-job are cleared later.
  Resume points of a stopped job are kept for 3 days; **Clear Saved Progress** in the add-on preferences deletes them.
- While a Region or Live preview is shown, the preview is a wireframe object and your mesh is hidden under it. Turning
  the preview off, disabling CurioMesh or opening the .blend with CurioMesh on removes the preview and shows your mesh
  again. If you save with a preview on and open the file without CurioMesh, delete the preview object and unhide your
  mesh (Alt+H).
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

License: GPL-3.0-or-later.
