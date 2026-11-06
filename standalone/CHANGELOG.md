# CurioMesh Standalone changelog

## 0.3.0

The first full release (0.1 and 0.2 were previews).

- **Up to date with CurioMesh 0.9.8.** The add-on's verdict on every result (green, amber or red: "Good for game
  asset", "Check 2 spots", "5 spots with bad faces left") with what to do next, the add-on's own icons for each use,
  and the rarely changed options folded under More options, so Retopologize and the result stay in view.
- **One Mac download.** A single universal app runs on Apple silicon and Intel Macs.
- **Honest about rough scans.** A result that kept nearly every face says so ("Kept nearly every face") with what
  to try instead, on screen and on the command line, instead of calling it good.
- **Checked on 22 free test models** (props, machines, statues, rocks, scans, up to 1.5 million faces) with the Game
  Asset preset: no crashes, 0 bad faces on every one, each within its 6% limit.
- Fixes: the first-run tour no longer sits on top of the open guide; numbered steps in the guide lose their extra
  bullet; "1× fewer" is gone when a result has about as many faces as the model.

## 0.2.0

- **Batch.** `curiomesh batch FILES-OR-FOLDERS -o FOLDER` retopologizes many models with the same settings. A model
  that fails is noted and the rest carry on. A report per model (faces, view change, bad faces, issue spots, time, ok
  or failed) is written as JSON and CSV. `--jobs N` works on several at once, at most half the processor cores.
- **Settings files and presets.** All settings fit in a small JSON file (`--settings FILE.json`). Built-in presets
  follow the Blender add-on's uses: Game Asset, Scan Cleanup, Low Poly, Sculpt Base and Animation.
  `curiomesh preset list | show NAME | save NAME` lists them, prints one as a settings file, or saves your own.
- **Mirror.** `--mirror auto|x|y|z` solves one half, mirrors it and closes the seam, so both sides are identical.
  `auto` finds the mirror plane and only mirrors a symmetric model. Ported from the add-on's Symmetrize.
- **Quad sizes.** `curiomesh sizes MODEL` works out the model at 7 detail values, from small quads to big ones, and
  lists the faces and view change of each. Every size shares one reduction and one set of camera pictures, so the
  extra sizes are quick. Ported from the add-on's Quad Size slider.
- **Export for game engines.** `.glb` output next to OBJ, `--engine unreal|unity|godot|roblox|gltf`, and the
  options under them: `--triangulate`, `--max-tris N` (cut into pieces under a triangle limit, Roblox 20,000),
  `--scale S` and `--up z`.
- **Fix:** a glTF model whose texture files are missing now opens. Only the geometry is read.
- For apps built on the library: `settings::Settings` and `settings::run`, `batch::run` with progress and a cancel
  flag, `ladder::run`, `mirror`, `export::write`, and `retopo::Prepared` for many searches on one model.

## 0.1.1

The first preview: the desktop app and the command line, OBJ, glTF, GLB and PLY in, OBJ out, `curiomesh measure`.
