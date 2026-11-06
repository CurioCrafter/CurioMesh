## Clean quads from heavy models, now a whole folder at a time

CurioMesh turns heavy scans, CAD exports and sculpts into clean all-quad meshes, without Blender. You say what the model is for, it keeps how the model looks, and it tells you when the result is good to go.

![The CurioMesh app: the original chess set on the left, its 5,717 new quads on the right, and the verdict "Good for game asset"](https://raw.githubusercontent.com/CurioCrafter/CurioMesh/standalone-v0.2.0/docs/images/standalone_app.png)

### Which file do I download?

| Your computer | Download | Start |
|---|---|---|
| Windows 10 or 11 | `CurioMesh-0.2.0-windows-x64.zip` | `curiomesh-gui.exe` |
| Mac with Apple silicon (M1 and later) | `CurioMesh-0.2.0-macos-apple-silicon.zip` | right-click `curiomesh-gui`, Open |
| Mac with an Intel processor | `CurioMesh-0.2.0-macos-intel.zip` | right-click `curiomesh-gui`, Open |
| Linux (x64) | `CurioMesh-0.2.0-linux-x64.zip` | `./curiomesh-gui` |

Unpack the zip anywhere and start the app. Nothing to install. Each zip also holds `curiomesh`, the command line, and a short guide.

### What's new in 0.2.0

**A verdict on every result.** A green, amber or red mark tells you where you stand: "Good for game asset", "Check 2 spots" or "5 spots with bad faces left", with the next step under it. It uses the same rules as the Blender add-on's new panel, so both say the same about the same mesh.

**Batch a whole folder.** Drop a folder on the window or press Batch a Folder. Every model gets the same settings, a bar of its own and a row in the results table: faces before and after, view change, bad faces and time. A model that fails is noted and the rest carry on. Export the table as CSV, or open any result to compare it.

![The Batch screen after three models: all done, 0 bad faces each](https://raw.githubusercontent.com/CurioCrafter/CurioMesh/standalone-v0.2.0/docs/images/standalone_batch.png)

**Start by saying what it's for.** Game Asset, Scan Cleanup, Low Poly, Sculpt Base and Animation, with the add-on's own icons, or your own limit on how much of the view may change. The app remembers your choice and your recent models.

![The Home screen: pick what the model is for, then open a model or batch a folder](https://raw.githubusercontent.com/CurioCrafter/CurioMesh/standalone-v0.2.0/docs/images/standalone_home.png)

**Mirror symmetric models.** Mirror Auto finds the mirror plane, solves one half and copies it, so both sides match exactly and the seam is closed.

**Ready for game engines.** Send to Unreal, Unity, Godot, Roblox or glTF writes the right file type (GLB or OBJ), triangulates when the engine wants triangles and splits meshes over Roblox's 20,000-triangle limit.

**For pipelines.** `curiomesh batch models/ -o done/`, settings files and presets (`curiomesh preset save`), `curiomesh sizes model.glb` to compare seven quad sizes at once, and `curiomesh measure` to score any retopology against its source and stop a build on a bad one.

### How it holds up

The Game Asset preset on three free test models, all 0 bad faces:

| Model | Before | After | View change | Time |
|---|---|---|---|---|
| Chess set | 76,832 faces | 5,717 quads | 6.1% | 3 s |
| Drill press | 58,511 faces | 20,080 quads | 6.2% | 25 s |
| Fire hydrant | 86,316 faces | 25,310 quads | 6.0% | 12 s |

At the same face counts as AutoRemesher 1.2.0, CurioMesh kept more of the shape on all three (drill press: 8.2% of the view changed against 32.8%) and left 0 bad faces where AutoRemesher left 113 to 1,219. AutoRemesher's quads are more even on smooth, organic shapes. Every number here was measured with `curiomesh measure`; the full table is in the guide.

### Good to know

- This is a preview. The Blender add-on still does more: UVs, baking, guided edge loops and part-by-part fixes.
- The programs aren't code-signed yet, so Windows and macOS ask once before the first run. On Windows choose More info, then Run anyway.
- Opens OBJ, glTF, GLB and PLY. Saves OBJ, or GLB for engines that read glTF.
- Check a download against `SHA256SUMS.txt`.

Free and open source under GPL-3.0-or-later. Something wrong or missing? [Open an issue](https://github.com/CurioCrafter/CurioMesh/issues).
