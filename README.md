# CurioMesh Quad Remesher

CurioMesh is a Blender add-on for practical automatic quad retopology. Version
0.3 includes the production QuadriFlow pipeline from v0.2 and adds TRIAD-Q Lite,
an experimental standalone quad-dominant remeshing engine that can run inside or
outside Blender.

## What It Does

- Runs QuadriFlow in face-count mode with deterministic seed control.
- Provides an experimental `TRIAD-Q Lite` engine: feature extraction, field
  estimation, seed-swept triangle pairing, material/UV seam awareness, optional
  pure-quad repair, and standalone OBJ input/output.
- Keeps an evaluated source snapshot alive through transfer, projection, and
  optional baking, so `REPLACE` mode does not project from the newly remeshed
  mesh back onto itself.
- Cleans duplicate/degenerate mesh data and recalculates normals before
  remeshing.
- Retries through Blender voxel remesh when QuadriFlow rejects messy input.
- Projects the output back to the source with Shrinkwrap.
- Preserves material slots, maps material indices from the nearest source face,
  and transfers or projects UVs.
- Reports face count, quad ratio, target error, extraordinary vertex ratio,
  non-manifold edges, boundary edges, UV validity, material preservation,
  elapsed time, and the engine path used.

## Installation

1. Download or clone this repository.
2. Zip the folder that contains `__init__.py`.
3. In Blender, open `Edit > Preferences > Add-ons > Install`, select the zip,
   and enable **CurioMesh Quad Remesher**.
4. Select a mesh and open `3D View > N-panel > CurioMesh`.

CurioMesh targets Blender 4.3+ and is validated against Blender 4.5 LTS.

## Workflow

1. Select a mesh object.
2. Choose a target face count and quality preset.
3. Leave `Output` on `New Object` while evaluating results.
4. Use `Replace` when you want the selected object updated in place.
5. Keep `Project UVs` enabled for textured assets. It first tries Blender data
   transfer, then falls back to BVH barycentric UV projection.
6. Use `Bake Color` or `Bake If UVs Fail` for assets where projection cannot
   preserve acceptable texture detail.

## Controls

- `Target Faces`: approximate number of output quads.
- `Engine`: `QuadriFlow` for production, `TRIAD-Q Lite` for the experimental
  standalone CurioMesh engine, or `Auto` for the production default.
- `Quality`: Draft, Balanced, or Hero defaults for cleanup and preservation.
- `Preserve Sharp`, `Preserve Boundary`, `Treat UV Seams As Sharp`: feature
  preservation hints for QuadriFlow and preprocessing.
- `Use Mesh Symmetry`: forwards mesh symmetry settings to QuadriFlow.
- `Cleanup Strength`: duplicate/degenerate cleanup before remeshing.
- `Voxel Repair Fallback`: retries through voxel remesh if QuadriFlow rejects
  the source mesh.
- `Project Details`: applies a Shrinkwrap pass onto the preserved source
  snapshot.
- `Texture Preservation`: `Project UVs`, `Transfer Only`, `Bake Color`, or
  `None`.
- `Apply Source Modifiers`: in `Replace` mode, remove original modifiers after
  baking their evaluated result into the remeshed mesh.
- `TRIAD-Q Seeds`, `TRIAD-Q Feature Angle`, `TRIAD-Q Pure Quads`, and
  `TRIAD-Q Flow`: experimental controls shown when the TRIAD-Q Lite engine is
  selected.

## TRIAD-Q Lite CLI

TRIAD-Q Lite can run without Blender on OBJ files:

```powershell
python -m triadq input.obj output.obj --target-faces 4000 --mode balanced --seed-count 8
```

It emits a JSON report with mode classification, selected seed, quad ratio,
target-face error, extraordinary vertex ratio, feature-edge count, aspect
penalty, and score. The current implementation is a practical prototype: it is
best at converting triangulated surfaces into feature-aware quad-dominant
meshes, not at replacing QuadriFlow as a full simplification/remeshing solver.

## Development And Tests

Python syntax check:

```powershell
python -m compileall -q __init__.py bridge.py metrics.py operators.py textures.py ui.py triadq tests
```

Ruff check:

```powershell
ruff check .
```

Standalone TRIAD-Q Lite smoke tests:

```powershell
python tests\triadq_smoke.py
```

Headless Blender smoke tests:

```powershell
tests\run_blender_tests.ps1
```

If Blender is not installed, the runner can download the official Blender 4.5.1
LTS portable build into `.tools/`:

```powershell
tests\run_blender_tests.ps1 -InstallBlender
```

The smoke test loads the add-on directly from this checkout, registers it,
and runs remesh cases for a UV/material sphere, torus, open grid, and
bad-normal repair mesh, plus TRIAD-Q Lite cube and grid cases.

## Notes

- QuadriFlow is high quality but not magic. It is useful for subdivision-ready
  meshes and sculpt-friendly retopology, but it is not a replacement for manual
  animation topology on characters.
- TRIAD-Q Lite is intentionally marked experimental. It is the first standalone
  CurioMesh engine path and is useful for iteration, tests, and triangulated
  mesh conversion, while QuadriFlow remains the default production engine.
- Voxel repair can save messy input but may erase thin or open details. Disable
  it when boundary fidelity matters more than robustness.
- The old broken `core/` C++ prototype was removed from the shipped add-on.
  CurioMesh now advertises only implemented behavior.

## License

MIT. See `LICENSE`.
