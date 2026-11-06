# CurioMesh Standalone

Turn a heavy scan, CAD export or sculpt into a clean all-quad mesh, without Blender. You choose how much of the
model's look may change, and CurioMesh keeps the fewest faces that hold it. Then it tells you where the result
still needs a look.

![CurioMesh: the original model and its new quads side by side](https://raw.githubusercontent.com/CurioCrafter/CurioMesh/standalone/docs/images/standalone_app.png)

This is the standalone version of [CurioMesh](https://github.com/CurioCrafter/CurioMesh), the free retopology
add-on for Blender. It runs the add-on's own Rust core, and gives the same verdict on a result as the add-on's panel.

## Get it

Download the zip for your system from the
[releases page](https://github.com/CurioCrafter/CurioMesh/releases?q=standalone), unpack it anywhere and start
**curiomesh-gui**. Nothing to install.

- **Windows:** `curiomesh-gui.exe`. Windows may warn about an unknown publisher the first time, because the
  programs aren't code-signed yet. Choose More info, then Run anyway.
- **macOS (Apple silicon and Intel, one download):** right-click `curiomesh-gui`, choose Open, then Open again (the
  app isn't notarized yet).
- **Linux:** `./curiomesh-gui`. Needs OpenGL and X11 or Wayland.

## Use the app

1. **Open a model.** Drop an OBJ, glTF, GLB or PLY file on the window, or click Open a model. You can also drop the
   file on the program's icon.
2. **Say what it's for.** Game asset keeps few faces and lets up to 6% of the view change. Scan cleanup keeps
   more detail (2%). Your own limit sets any other share, and Aim for a face count asks for a number of faces
   instead.
3. **Retopologize.** A model of 50,000 to 100,000 faces takes a few seconds to a minute.
4. **Compare.** Drag the line across the model to see the original on the left and the new quads on the right.
   Drag anywhere else to turn the model, scroll to zoom, double-click to frame it again.
5. **Check the spots.** The list shows places worth a look, worst first: slivers, faces at poles and bent quads.
   Click one to fly to it and see its faces in red.
6. **Save OBJ.** Save issue report writes the same list as JSON.

## How much may the view change?

CurioMesh looks at the model from six sides, before and after, and counts the pixels where the outline, the depth
or the direction of the surface differs. That share is the view change. Game asset allows 6%, which keeps
silhouettes and hard edges while flat and gently curved areas become large quads. Lower numbers keep more faces.

Bad faces are faces on broken edges, flipped or duplicated faces, and quads bent past 60 degrees or folded. A good
result has none.

## Use the command line

`curiomesh` does the same work in scripts and asset pipelines. `curiomesh --help` lists everything.

### One model

```
curiomesh model.glb -o model_low.obj --use game-asset --report issues.json
curiomesh scan.ply -o scan_low.obj --reduce-to 5000
curiomesh statue.obj -o statue_low.obj --mirror auto
curiomesh crate.glb -o crate_game.glb --engine godot
```

The name after `-o` decides the file type: `.obj` or `.glb`.

### Many models: `batch`

```
curiomesh batch models/ -o done/ --use game-asset
curiomesh batch a.glb b.obj c.ply -o done/ --engine roblox --jobs 2
```

Give files, folders (every .obj, .gltf, .glb and .ply directly inside), or both. Each result is written as
`NAME_curiomesh.obj` (or `.glb`) in the `-o` folder, with the same settings for every model. A model that can't be
read or fails is noted and the rest carry on. When all are done, `curiomesh_report.json` and `curiomesh_report.csv` in
the same folder list per model: faces, quads and triangles, how much of the view changed, bad faces, issue spots, time,
and ok or failed. `--jobs N` works on N models at once (default 1, at most half your processor cores). The program
exits with code 1 when any model failed.

### Quad sizes: `sizes`

```
curiomesh sizes model.glb --detail 6 -o sizes/
```

Works out the model at 7 detail values around `--detail` (a quarter of it up to four times it), from small quads to
big ones, and prints each one's faces and view change. Sizes that another size beats on both counts are left out.
With `-o` each size worth keeping is saved, so you can pick by eye.

### Settings and presets

| Option | What it does |
|---|---|
| `--use NAME` | A preset (see below) or one you saved. |
| `--settings FILE.json` | Settings from a file. Keys left out keep their defaults. |
| `--detail P` | Keep the fewest faces whose view changes by at most P percent. Default 6. |
| `--reduce-to N` | Aim for about N faces instead. |
| `--mirror auto`, `x`, `y` or `z` | Solve one half, mirror it and close the seam, so both sides are the same. `auto` mirrors only when the model is symmetric (90% of it matches its mirror); naming an axis needs 70%. |
| `--no-rows` | Keep every part as reduced; don't try even rows of quads on smooth parts. |
| `--pixels N` | Camera picture size per side. Default 192. |
| `--cpu` | Draw the camera pictures on the processor instead of the graphics card. |
| `--report FILE.json` | Write the numbers and the issue spots as JSON. |

Later options win: `--use scan --detail 3` is Scan Cleanup at 3%.

Built-in presets, after the Blender add-on's uses:

| Preset | Detail | Also |
|---|---|---|
| `game-asset` | 6% | The default. All quads, few faces. |
| `scan` | 2% | Scan Cleanup: keeps more of a scan's detail. |
| `low-poly` | 10% | Written as triangles, no rows. Small parts may vanish. |
| `sculpt-base` | 6% | Mirrored when the model is symmetric. |
| `animation` | 1.5% | More, evener quads; mirrored when symmetric. Check eyes, mouth and joints by hand. |

```
curiomesh preset list                       # built-in and saved presets
curiomesh preset show scan > my.json        # a preset as a settings file to edit
curiomesh preset save props --use low-poly --detail 8 --engine unity
curiomesh batch props/ -o done/ --use props
```

Saved presets live in `%APPDATA%\CurioMesh\presets` on Windows, `~/Library/Application Support/CurioMesh/presets` on macOS and `~/.config/CurioMesh/presets` on Linux. A
settings file looks like this:

```json
{
  "curiomesh_settings": 1,
  "detail": 6.0,
  "reduce_to": null,
  "pixels": 192,
  "gpu": true,
  "rows": true,
  "mirror": "off",
  "export": { "format": "obj", "triangulate": false, "max_tris": 0, "scale": 1.0, "up": "y" }
}
```

### For game engines

| Option | What it does |
|---|---|
| `--engine unreal` | One .glb in triangles. Unreal 5 reads glTF and turns it to its own axes and centimetres. |
| `--engine unity` | OBJ in triangles, Y up, in metres. |
| `--engine godot` | One .glb in triangles. |
| `--engine roblox` | OBJ in triangles, cut into pieces of at most 20,000 triangles (Roblox's limit per mesh). |
| `--engine gltf` | One .glb in triangles for three.js, Babylon.js, Bevy, Sketchfab and other glTF readers. |
| `--triangulate` | Write triangles. Quads are cut along their shorter diagonal. |
| `--max-tris N` | Cut the mesh into pieces of at most N triangles, along materials and then across its longest side. |
| `--scale S` | Multiply every coordinate by S. |
| `--up z` | OBJ only: Z up instead of Y up. A .glb is always Y up, as glTF requires. |
| `--format obj` or `glb` | The file type for `batch` and `sizes`. |

Pieces are separate objects in one file. No textures or UVs are written: the standalone doesn't bake.

### Check any retopology: `measure`

`curiomesh measure SOURCE RESULT` scores any retopology against its source: one made by CurioMesh, by another tool
or by hand. It prints the view change, the bad faces and the worst spots. With `--max-loss` or `--max-bad` it exits
with code 1 when the result is over the limit, so a build can stop on a broken asset.

## How it compares

All measured with `curiomesh measure` on three free models from the CurioMesh test set, at about the same face
count. AutoRemesher 1.2.0 ran with its default settings.

| Model | AutoRemesher 1.2.0 | CurioMesh Standalone |
|---|---|---|
| Chess set | 8,974 faces, 8.3% view change, 113 bad faces | 9,491 faces, 4.7% view change, 0 bad faces |
| Drill press | 15,455 faces, 32.8%, 1,219 bad | 16,716 faces, 8.2%, 0 bad |
| Fire hydrant | 17,335 faces, 20.6%, 462 bad | 18,920 faces, 10.2%, 0 bad |

On these three, CurioMesh keeps much more of the shape on hard-surface parts, and its results have no broken or folded faces.
AutoRemesher's quads are more even: on smooth, organic models its rows of quads are cleaner. CurioMesh lays even
rows only on smooth parts where they keep the view, and keeps the shape-preserving reduction elsewhere. CurioMesh's
results here have 5-9% more faces.

## Known limits

- The Blender add-on still does more: part-by-part sizing, guided edge loops, UVs, baking, repair, Redo Rough
  Spots and Tidy Up. The standalone writes OBJ and .glb without UVs or textures, and no FBX.
- A rough scan can keep nearly all its faces at a small limit: the camera sees every bump turn. The result then says
  "Kept nearly every face"; Low Poly, a higher limit or a face count reduce it.
- Quad flow on organic models is less even than a field-aligned remesher's.
- Large models draw thinned out in the viewport; the saved result is complete.

## Licence

GPL-3.0-or-later, like the add-on. Source: the `standalone` branch of
[CurioCrafter/CurioMesh](https://github.com/CurioCrafter/CurioMesh/tree/standalone).
