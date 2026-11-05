CurioMesh Standalone 0.2.0: more than opening one model. Batch whole folders, pick settings up front, mirror symmetric models, try several quad sizes, and write files ready for game engines.

**Download** the zip for your system below, unpack it and start `curiomesh-gui`. Both programs are inside, with a short guide.

- **Batch.** `curiomesh batch models/ -o done/` retopologizes every model in a folder with the same settings, carries on past a model that fails, and writes a report per model as JSON and CSV.
- **Presets and settings files.** Game Asset, Scan Cleanup, Low Poly, Sculpt Base and Animation are built in, after the Blender add-on's uses. Save your own with `curiomesh preset save`, or keep settings in a small JSON file.
- **Mirror.** `--mirror auto` solves one half of a symmetric model and mirrors it, with a closed seam and two identical sides.
- **Quad sizes.** `curiomesh sizes model.glb` works out the model from small quads to big ones and lists faces and view change for each.
- **For game engines.** Writes `.glb` as well as OBJ. `--engine unreal|unity|godot|roblox|gltf` sets triangles, the file type and Roblox's 20,000-triangle limit per mesh.
- **Fix:** glTF models whose texture files are missing now open.

Still a preview: no UVs, textures or FBX yet, and the programs aren't code-signed, so Windows and macOS ask once before the first run. GPL-3.0-or-later.

Check a download against `SHA256SUMS.txt`.
