## CurioMesh Quad Remesher (Blender 4.5 add-on)

CurioMesh is a quad-dominant remesher for Blender focused on speed, feature preservation, and keeping your materials/UVs intact. It includes:

- Native C++ core (`curio_core`) for fast greedy tri→quad pairing
- Robust Python fallback that works without the native module
- Advanced UV transfer/projection with a configurable “cage”, anti-banding relaxers, and optional PBR baking

---

## Install

- Zip the `curiomesh` folder (the one containing `__init__.py`) and install via Blender Preferences → Add‑ons → Install.
- Enable the add‑on. The UI appears in 3D View → N‑panel → `CurioMesh`.

## Build native core (optional)

- Windows helper: run `build_windows.bat` inside `curiomesh/`.
- Generic CMake:
  - `cmake -S curiomesh/core -B build/curio_core`
  - `cmake --build build/curio_core --config Release`
- Place the produced `curio_core.*.pyd` next to `curiomesh/__init__.py`.
- If the native module is missing, the Python fallback runs automatically.

---

## File-by-file overview

- `__init__.py`
  - Blender entry. Registers/unregisters classes and attaches `Scene.curiomesh_settings`.

- `ui.py`
  - `CURIOMESH_PG_settings`: all user‑visible options (remeshing, projection, anti‑banding, baking, metrics).
  - `CURIOMESH_PT_panel`: draws the sidebar; exposes controls, presets, cage preview, and metrics.
  - Maps settings to the `curiomesh.remesh` operator call.

- `operators.py`
  - `CURIOMESH_OT_remesh`: main operator. Evaluates the selected mesh and calls `bridge.run_curiomesh(...)` with all settings.
  - `CURIOMESH_OT_apply_preset`: Draft/Balanced/Hero/Mesh Detail/Sphere Soft/Nearest kNN/Bake Hero presets.
  - `CURIOMESH_OT_compute_metrics`: fills face/extraordinary counts via `metrics.quick_metrics`.
  - `CURIOMESH_OT_preview_cage`: builds a projection cage (Box, Sphere, or Mesh Copy), supports several inflate modes, can visualize rays.
  - `CURIOMESH_OT_exploded_bake`: separates loose parts, offsets outward to avoid bake cross‑talk, and optionally reassembles.
  - `naive_tritoquad_on_bmesh(bm, ...)`: simple greedy triangle pairing used by the Python fallback.

- `bridge.py` (core pipeline)
  - `_load_native`: tries to import `curio_core`.
  - `_mesh_to_numpy`: converts Blender mesh to numpy arrays for the native core.
  - `_gather_source_materials` / `_apply_materials` / `_ensure_polygon_materials`: robust material copy helpers.
  - `_apply_data_transfer_uvs`: attempts Blender Data‑Transfer (operator or modifier) to copy UVs.
  - `_project_uvs_via_bvh`: nearest-surface projection using BVH and normal-aware ray casting; optional k‑NN blending; loop-wise UV assignment to preserve seams.
  - `_project_uvs_with_cage`: advanced cage projector. Builds/uses a cage, samples many directions, respects backface thresholds, sharp back‑off, axis bias, aim modes, and blends multiple samples per polygon/loop.
  - `_relax_uvs_vertex_average` / `_relax_uvs_seams`: anti‑banding relax passes.
  - `_uv_mapping_is_suspicious`: flags collapsed/distorted UVs to trigger bake fallback.
  - `_compute_auto_voxel_size`: heuristic for VDB preprocess.
  - `_preprocess_with_ops`: optional Voxel Remesh and adaptive decimation toward `target_faces`.
  - `_fallback_python`: bmesh-based remeshing with feature/seam protection, optional smoothing, and “pure quads” conversion.
  - `_replace_object_mesh`: creates or replaces the object data, preserves materials, and executes UV transfer/projection.
  - `_exploded_bake_internal`: scaffold for an automated exploded-bake pass (operator already implements an external helper).
  - `run_curiomesh(...)`: orchestrates the whole process and optional baking.

- `textures.py`
  - Minimal Cycles baker: ensures target UVs, creates images, assigns an Image Texture node per material, and bakes selected maps (COLOR/ROUGHNESS/METALLIC/NORMAL/EMIT) with selected‑to‑active.

- `metrics.py`
  - `quick_metrics(mesh)`: face count, extraordinary vertex count (valence ≠ 4), and extraordinary‑ratio.

- `core/pybind_module.cpp`
  - pybind11 bridge exposing `curio_core.remesh(verts, faces, ...)` to Python; returns numpy arrays for `verts`, `quads`, and `tris`.

- `core/curio_core.cpp`
  - Very small C++ remesher: computes face normals and greedily pairs adjacent triangles that share an edge if their normals are within a `feature_angle` threshold. Produces quads; unpaired triangles pass through.

- `core/quix_core.cpp`
  - An alternative/experimental implementation of the same greedy pairing idea (not used by the default CMake target).

- `core/CMakeLists.txt`
  - Build recipe for the `curio_core` Python extension.

- `build_windows.bat`
  - Windows convenience script to configure and build the native module.

- `models/`
  - Reserved for future assets (currently empty).

---

## Execution pipeline (what happens when you click “Remesh”)

1) The operator evaluates the active object to a temporary mesh copy (so modifiers are applied if needed).
2) Preprocess (optional):
   - Voxel Remesh (auto or user size) to clean up topology.
   - Adaptive decimation toward `target_faces` (Un‑Subdivide when massively dense, then Collapse). The preprocessed mesh becomes the working input.
3) Remeshing:
   - Native path (if `curio_core` is available): greedy tri→quad pairing with a feature-angle filter.
   - Fallback path: bmesh pairing with seam/sharp protection, optional smoothing; can force pure quads by splitting longest edges for leftover tris.
4) Output object & material preservation:
   - Replace original mesh data or create a new object; copy materials and set polygon `material_index` consistently.
5) UV strategy:
   - Attempt Data‑Transfer of the active UV from source to target.
   - If UVs are missing or look suspicious, use the selected projection mode (Nearest or Cage). AUTO tries sensible defaults.
6) Projection details:
   - Nearest (BVH): normal-aware ray casting from polygon centers/axes; front‑face only; optional k‑NN blending; UVs via barycentric interpolation per loop.
   - Cage: emit rays from a Box/Sphere/Mesh Copy cage; supports aim modes (poly centers, vertices, boundary edges or an aim object), axis bias, ray density, sharp back‑off, and consistency checks. Reports hit/miss stats.
7) Anti‑banding:
   - Vertex-average relax followed by seam-aware relax to reduce visible lines.
8) Baking (optional):
   - PBR baking of selected maps; or “post-bake Base Color” after projection; or a forced bake fallback when UVs are bad.
9) Cleanup & summary logs.

---

## Panel options (complete reference)

- Remeshing
  - `target_faces` (int): Desired face count after preprocess/remeshing.
  - `feature_weight` (float) / `adaptivity` (float): Passed to the native core (reserved for future heuristics).
  - `engine` (AUTO/CPU/CUDA/METAL/VULKAN): Execution hint for future native backends.
  - `use_vdb_pre` (bool) / `vdb_voxel_size` (float, 0=auto): Voxel Remesh preprocess controls.
  - `feature_angle` (deg): Dihedral threshold for preserving sharp features.
  - `smooth_iters` (int) / `smooth_lambda` (float): Laplacian smoothing on fallback and general post‑ops.
  - `preserve_seams` / `preserve_sharp` (bool): Respect UV seams and sharp edges (fallback).
  - `output_mode` (REPLACE/NEW): Replace geometry or create a new object.
  - `pure_quads` (bool): Try to eliminate leftover triangles by splitting edges.

- Bake PBR
  - `bake_pbr` (bool), `bake_resolution` (px), `bake_margin` (px), `bake_maps` (COLOR/ROUGHNESS/METALLIC/NORMAL/EMIT), `uv_method` (SMART/UNWRAP).
  - Quality: `post_bake_color` (bool), `post_bake_res` (px), `post_bake_margin` (px).
  - Fallback: `force_bake_fallback` (bool) forces baking when UV transfer/projection failed or is detected as suspicious.

- Projection
  - `projection_mode` (AUTO/TRANSFER/NEAREST/CAGE/BAKE).
  - `projection_expand` (float): Fraction of bounds diagonal to expand (>0) or shrink (<0) the cage.
  - `cage_mode` (BOX/SPHERE/MESH). For MESH, choose `cage_inflate`:
    - `WEDGE`: per‑vertex offset along bisectors of clustered face normals (`wedge_angle_thresh`).
    - `SOLIDIFY`: uniform offset using Solidify‑like thickness (`solidify_thickness_factor`).
    - `SDF`: volumetric offset via voxel remesh (`sdf_voxel_factor`, `sdf_smooth_iters`).
  - `use_preview_cage` (bool), `preview_cage_name` (string), `lock_cage` (bool).

- Advanced projection controls
  - `cage_blend_samples` (int): Extra samples blended per polygon (Cage).
  - `backface_thresh` (0..1): Reject hits whose surface normal opposes the ray beyond this threshold.
  - `ray_dir_mode` (NORMALS_AXES/NORMALS_ONLY/CENTER): Candidate ray directions.
  - `ray_aim_mode` (POLY_CENTER/VERTEX/EDGE); optional `ray_aim_object`.
  - `ray_density` (int), `axis_bias` (±X/±Y/±Z/NONE), `cage_center_offset` (vec3).
  - `split_sharp` (bool), `sharp_angle` (deg), `sharp_backoff` (fraction of diagonal).
  - `uv_consistency_check` (bool), `uv_consistency_thresh` (float).
  - `neighbor_blend` (bool), `neighbor_blend_samples` (int) for Nearest mode.
  - `relax_uvs` (bool), `relax_iters`, `relax_alpha`, `relax_threshold`.
  - `projection_solver` (AUTO/LOOP_NEAREST/POLY_PLANE/KNN_BLEND/RBF_FIELD/SPECTRAL/OT/SGMM_NDF/TENSOR_ANISO/PRT/DR) and associated parameters (present for future research integrations; current implementation uses the enhanced BVH/Cage projectors).

- Metrics
  - "Compute" fills: `metrics_faces`, `metrics_extraordinary`, `metrics_ratio`.

- Presets
  - Draft / Balanced / Hero / Mesh Detail / Sphere Soft / Nearest kNN / Bake Hero adjust coherent sets of parameters for speed vs. quality.

---

## Programmatic usage

```python
import bpy
s = bpy.context.scene.curiomesh_settings
s.target_faces = 12000
s.projection_mode = 'CAGE'
# Run on active object
bpy.ops.curiomesh.remesh('INVOKE_DEFAULT')
```

To drive everything from a script, you can also pass the operator keyword args directly (the operator mirrors the settings in `CURIOMESH_PG_settings`).

---

## Troubleshooting

- No result / empty output: enable "Debug Console Logs" in the panel and check Blender’s System Console. The Python fallback runs if `curio_core` is missing.
- Banding after projection: increase `cage_blend_samples`, enable `relax_uvs`, try `neighbor_blend` in Nearest mode, or enable `post_bake_color`.
- Crosstalk on thin surfaces: raise `backface_thresh`, enable `split_sharp`, and consider `ray_dir_mode= NORMALS_ONLY`.
- Black bakes: ensure Cycles is active and materials use nodes; the add‑on creates and activates Image Texture nodes per material automatically.

---

## Development notes

- The native core currently implements greedy tri pairing with a feature‑angle filter. The Python fallback adds smoothing and an optional pure‑quads phase.
- Advanced solver names exposed in the UI are present for future integrations; the current projector implementations are the BVH‑Nearest and Cage methods described above.

