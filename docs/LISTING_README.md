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
  three layout styles: Keep Detail, Balanced and Clean Rows.
- **Uses:** Low Poly, Game Asset, Animation, Subdivision, Scan Cleanup, Sculpt Base and Hard Surface.
- **Adjust:** the Quad Size, Rows and Organize sliders, and Tidy Up with its Loop Quality and Loop Flow meters.
- **Fix:** the Issue Finder (faces turned inward, folded over, crossing or sunk), Redo this spot, Keep original here,
  and Fix Selection in Edit Mode.
- **Hard Surface:** Clean Panels, Even Cylinders, and hard edges marked on Apply.
- **Shape the flow:** edge loops, deformation lines, guide curves, Detail Markers, Mirror and Symmetrize, and the Flow
  Editor (experimental, off by default).
- **Finish:** Apply, Bake Detail (normal or normal + color maps), LOD sets, Multires Detail and Export for Game Engine
  (Unreal, Unity, Godot, Roblox).
- **Job folders:** Export Job writes a folder you can run in any Blender that has CurioMesh installed
  (`blender --background --python run_job.py`). Import Result brings the result back.

## Notes

- Big jobs can spread their parts over helper processes. These are background copies of the same Blender, started only
  while a job runs and closed when it ends. Turn this off with **Use Several Cores** in the Pro tab.
- The Clean Rows style runs Blender's own QuadriFlow in a background copy of Blender, so a QuadriFlow crash on a hard
  mesh cannot close your session.
- Remembered fixes and resume points are kept in the extension's user folder.
- **Report a Problem** opens a GitHub issue in your web browser. The add-on itself never connects to the internet.
- Bundled data: `triadq/patterns.json` holds statistics measured on CC0 and public-domain meshes (see
  `docs/PATTERN_SOURCES.md`). `triadq/cases_table.json` holds camera-check results measured on CC0 Poly Haven models. The
  icons are CC0 (`icons/LICENSE.txt`).

Source code, documentation and issues: https://github.com/CurioCrafter/CurioMesh

License: GPL-3.0-or-later.
