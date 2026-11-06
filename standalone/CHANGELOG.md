# CurioMesh Standalone changelog

## 0.4.0 (in progress)

- **Judge as baked**, the add-on's Judge After Baking: a model that will get a normal map baked from the original
  (a scan, a sculpt) is judged as it will look then. Fine bumps the map paints back don't count; moved outlines, a
  surface far from the original and shading still wrong over a few pixels do. Scan Cleanup turns it on; the app has
  it under More options, the command line as `--baked`. Game Asset with it on: a boulder scan 97,964 -> 1,847 quads
  (was 116,974), a cliff scan 1.5 million -> 991, a lion head 47,214 -> 1,708, all 0 bad faces.
- **Keeps the source's UVs.** `--keep-uvs` (app: More options, Keep the UVs) gives the result the UVs of the model
  it came from, so the model's own textures fit it. UVs are read from OBJ, glTF/.glb (`TEXCOORD_0`) and PLY; OBJ
  results get `vt`, smooth `vn` and a .mtl naming the colour textures, .glb results get UVs, normals and the
  textures inside. A face that lands across a UV seam takes the island under its centre, so seams stay clean.
- **Bakes normal and colour maps.** `--bake normal,color` and `--bake-size N` (default 2048) bake a tangent-space
  normal map of the source's surface and its colour texture into the result's UVs, as PNGs next to the result,
  named in the .mtl and packed into the .glb. Islands are padded so mip-maps don't bleed.
- A model without UVs says that UVs and baking need a source with UVs; CurioMesh doesn't unwrap.
- Settings files gain `keep_uvs`, `bake` and `bake_size`.
- Measured on two free test models (Game Asset): a marble bust, 7,038 faces, 0.4% of faces with UV area over 4x
  off, maps baked in 0.7 s; a drill press with many small UV islands, 20,822 faces, 3.3% over 4x off, 1.9 s.
- **Tidy Up**, the add-on's Organize redraw: patches of quads round poles are redrawn as plain grids where one fits
  the outline with fewer poles, new corners on the original surface, hard edges, borders and materials kept. A slider
  under Options ("More detail" to "Tidier", off by default) and `--tidy` on the command line; kept only if the view
  changes at most 1 point more. Game Asset at full Tidy Up: poles on a marble bust 2,521 -> 909, a lion head 8,020 ->
  3,311, a fire hydrant 9,634 -> 7,987, a chess set 1,924 -> 1,640, with 0.2 to 0.9 points more view change, 5 to 21%
  fewer faces and 0 bad faces. On a horse statue it was not kept (the view would have changed 1.2 points more).
- **Fix one spot without redoing the model.** Pick a spot in the list and choose **Redo this spot** (smaller quads
  that follow the original more closely) or **Keep original here** (the original's own faces back there, as quads;
  it shows the faces it adds and asks first). **Put back** undoes the last fix. Only the spot and a ring round it
  change: the edge of the fixed area keeps its corners, so nothing outside moves and no cracks open. Results stay all
  quads, and a fix that would add bad faces is refused with the reason. The numbers, verdict and spot list update.
- **`curiomesh fix SOURCE RESULT --spot N --redo|--keep-original -o OUT`** does the same on the command line.

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
