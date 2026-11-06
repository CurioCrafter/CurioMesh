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
6. **Fix a spot** (optional). With a spot picked, **Redo this spot** remakes the faces round it with smaller quads
   that follow the original more closely. **Keep original here** puts the original's own faces back round it, as
   quads; it first lights up the faces it would add and says how many, and is applied with Apply. **Put back** undoes
   the last fix. Only that spot changes: everything outside it stays exactly as it was, with no cracks at the edge.
   A fix that would add bad faces isn't made, and says why. The numbers, verdict and spot list update after each fix.
7. **Save OBJ.** Save issue report writes the same list as JSON.

## How much may the view change?

CurioMesh looks at the model from six sides, before and after, and counts the pixels where the outline, the depth
or the direction of the surface differs. That share is the view change. Game asset allows 6%, which keeps
silhouettes and hard edges while flat and gently curved areas become large quads. Lower numbers keep more faces.

Bad faces are faces on broken edges, flipped or duplicated faces, and quads bent past 60 degrees or folded. A good
result has none.

### Tidy Up

Tidy Up (the slider under Options, `--tidy` on the command line) takes away poles, the corners where other than four
edges meet. Round each pole, worst first, it looks at the patch of quads up to three rings out and asks whether a plain
grid fits the patch's outline with fewer poles; if it does, the patch is redrawn as that grid with its new corners on
the original surface. Hard edges, borders and material changes stay where they are, and no redraw may bring in a bad
face. The result is kept only if the view changes at most 1 point more; the line under the result says what it did,
such as "Tidy Up: poles 2,521 -> 909, view changed 4.8% -> 5.4%". It works best on organic models (a marble bust lost
two thirds of its poles), less on machines with many hard edges. The grids are usually a little coarser than the
patches they replace, so the result also has 5 to 20% fewer faces.

### Edge loops you draw

Where an edge loop must run, draw it: round an eye or the mouth, round the neck, along a joint, across a panel. In the
app, click **Draw loops** under Edge loops and drag on the model; each stroke shows as a line on its surface. Right-drag
turns the view while drawing (drag on the divider to move it), **Undo** takes the last stroke away and **Clear** all of
them. Retopologize then gives the result an edge loop along every stroke, with quads on both sides of it, and the line
under the result says how closely it followed, such as "Loops followed: mean 0.2%, worst 0.9% of the model's size": how
far, along each stroke, the nearest edge of the result is, on average and at the worst point.

On the command line, `--strokes FILE.json` reads the strokes: a list of polylines, each a list of `[x, y, z]` points in
the model's own coordinates. A stroke whose ends meet is closed into a full loop.

```json
[[[0.05, -0.04, 0.28], [0.03, 0.05, 0.28], [-0.06, 0.04, 0.28], [-0.07, -0.08, 0.28], [0.05, -0.04, 0.28]]]
```

How it works: each stroke is laid on the surface; at every face count tried, the reduced triangles near it are split
until they are small, the cheapest chain of their edges that hugs the stroke is found and its corners are pulled onto
it, and no quad is paired across that chain. A stroke, or the part of it, more than 2% of the model's size away from
the surface is left out, as is one shorter than 2% of the size; the result says so. Rows of quads are not offered
when there are strokes (they rebuild a part from scratch), and evening out and Tidy Up are kept only if the loops stay
where they were. With mirroring, strokes on the other half are mirrored onto the half that is solved.

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
| `--tidy T` | Tidy Up, 0 to 1 (default 0): fewer poles. 0.5 is one round, 1 is two rounds and spun edges. See below. |
| `--strokes FILE.json` | Edge loops along these strokes (see Edge loops you draw). A settings file can hold them too, as `strokes`. |
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
  "tidy": 0.0,
  "mirror": "off",
  "export": { "format": "obj", "triangulate": false, "max_tris": 0, "scale": 1.0, "up": "y" },
  "keep_uvs": false,
  "bake": [],
  "bake_size": 2048
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

Pieces are separate objects in one file.

### UVs and baked maps

| Option | What it does |
|---|---|
| `--keep-uvs` | Give the result the source's UVs, so the source's own textures fit it. The OBJ gets a .mtl naming each material's colour texture; the .glb carries the textures inside. |
| `--bake normal`, `color` or `normal,color` | Bake maps into the result's UVs, as PNGs next to the result (`NAME_normal.png`, `NAME_color.png`; with several materials, one pair per material). Implies `--keep-uvs`. |
| `--bake-size N` | Pixels per side of the maps. Default 2048. |

In the app: More options, Keep the UVs and Bake.

- **The source needs UVs.** They are read from OBJ (`vt`), glTF and .glb (`TEXCOORD_0`) and PLY (`u v` or `s t`).
  CurioMesh doesn't unwrap: a model without UVs gets a result without them, and says so.
- **How UVs are carried over.** Each corner of the result takes the UV of the nearest point on the source. A face
  that lands across a UV seam takes the UV island under its centre, and its other corners are looked up again on
  that island only. A corner on a seam has one UV per face, as in the source.
- **The normal map** holds the source's smooth surface where the result is flatter: a game engine shows that detail
  on the result. Each texel looks along the result's normal for the source surface (both ways, within a distance a
  bit larger than the gap between the two), or takes the nearest source point. It is tangent-space, OpenGL style
  (green up), with tangents worked out per triangle from the UVs and averaged where the UVs are shared: close to the
  MikkTSpace tangents engines use, not identical, so a faint line can show along a UV seam. The result is written
  with the smooth normals the bake used.
- **The colour map** is the source's colour texture sampled through the source's UVs.
- Islands are padded by a few pixels so mip-maps don't bleed the background in.
- UVs that repeat outside the 0-1 square (a tiling texture) can't be baked in full: the maps cover the square only.

### Check any retopology: `measure`

`curiomesh measure SOURCE RESULT` scores any retopology against its source: one made by CurioMesh, by another tool
or by hand. It prints the view change, the bad faces and the worst spots. With `--max-loss` or `--max-bad` it exits
with code 1 when the result is over the limit, so a build can stop on a broken asset.

### Fix one spot: `fix`

`curiomesh fix SOURCE RESULT --spot N --redo -o OUT` remakes spot N (numbered as `measure` lists them) with smaller
quads on the source; `--keep-original` instead puts the source's own faces back there. The rest of RESULT is left as
it is. It prints the faces there before and after, the bad faces and the view change before and after. A fix that
would add bad faces, or can't be made (a spot along a broken edge, or where the source is tangled), is refused with
the reason, nothing is written and it exits with code 1.

Redo this spot helps where quads cut across the shape or bend; it keeps slivers and poles (smaller copies of them).
Keep original brings back the detail and costs more faces.

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

- The Blender add-on still does more: part-by-part sizing, rows of loops beside a drawn loop, new UVs (unwrapping), repair, Redo
  Rough Spots and Tidy Up. The standalone keeps the source's UVs and bakes normal and colour maps, but can't unwrap a
  model that has none, and writes no FBX.
- A rough scan judged as it is can keep nearly all its faces at a small limit: the camera sees every bump turn. If you
  will bake a normal map, turn on **Judge as baked** (More options, or `--baked`; Scan Cleanup has it on): the bumps
  then don't count, and a scan drops to a few thousand faces.
- Quad flow on organic models is less even than a field-aligned remesher's.
- Large models draw thinned out in the viewport; the saved result is complete.
- Fixing a spot is refused where the spot's edge runs along a broken edge, and Keep original where the source
  under it is tangled or has fewer borders than the spot.

## Licence

GPL-3.0-or-later, like the add-on. Source: the `standalone` branch of
[CurioCrafter/CurioMesh](https://github.com/CurioCrafter/CurioMesh/tree/standalone).
