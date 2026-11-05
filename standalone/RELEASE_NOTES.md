## 0.2.0: the desktop app does more

- **Start screen, settings first.** Choose what the model is for before you open it: Game asset, Scan cleanup, Low poly, Sculpt base or your own limit, each with a one-line explanation. Measuring detail and the other options are there too. Your choices and a list of recent files are remembered.
- **Batch.** Add files or a folder (or drop them), pick an output folder and run them one after another in the background. Every model has its own progress bar, there is an overall bar and a Cancel button. A table shows faces before and after, view change, bad faces and time. Open a result to compare it, show it in its folder, or export the table as CSV.
- **Guide.** A built-in guide explains every preset and setting, how to read view change and bad faces, the compare view, batch, and tips for scans, game assets and sculpts. Every control has a hover tip, and a short first-run tour can be skipped.
- **Easier navigation.** A bar at the top switches between Home, the open model and Batch.

## 0.1.1

The first preview of CurioMesh without Blender: a desktop app and a command line that turn heavy scans, CAD exports and sculpts into clean all-quad meshes.

**Download** the zip for your system below, unpack it and start `curiomesh-gui`. Both programs are inside, with a short guide.

- **Keep the look, drop the faces.** You choose how much of the view may change (6% for a game asset, 2% for a scan) and CurioMesh keeps the fewest quads that hold it. Hard edges and silhouettes stay; flat areas become large quads.
- **See the difference.** Drag a line across the model to compare the original and the new quads side by side.
- **Know where to look.** Bad faces are counted and the spots worth a look are listed worst first. Click one to fly to it.
- **Check any retopology.** `curiomesh measure` scores a result from any tool against its source and can stop a build when it is over a limit.
- **Opens** OBJ, glTF, GLB and PLY. **Saves** OBJ.

On three free test models at about the same face count, CurioMesh kept far more of the shape than AutoRemesher 1.2.0 (drill press: 8.2% of the view changed against 32.8%, 0 bad faces against 1,219). AutoRemesher's quads are more even on smooth, organic shapes. The full table is in the guide.

This is a preview: the Blender add-on still does more (UVs, baking, guided loops, repair), and the programs aren't code-signed yet, so Windows and macOS ask once before the first run. GPL-3.0-or-later.

Check a download against `SHA256SUMS.txt`.
