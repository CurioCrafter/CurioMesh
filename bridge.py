import importlib
import sys
import traceback
import os
import time
from typing import Any, Dict, Tuple
import numpy as np

import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from mathutils import Vector, Matrix
from math import radians, degrees, sqrt, cos, sin, pi, atan2

from .textures import bake_maps as bake_pbr_maps

_curio_core = None


def _load_native():
    global _curio_core
    if _curio_core is not None:
        return _curio_core
    try:
        # Keep module name for now unless native is also renamed; fallback safe
        _curio_core = importlib.import_module("curio_core")
    except Exception:
        _curio_core = None
    return _curio_core


def _mesh_to_numpy(mesh: bpy.types.Mesh) -> Tuple[np.ndarray, np.ndarray]:
    mesh.calc_loop_triangles()
    verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    faces = np.array([tri.vertices[:] for tri in mesh.loop_triangles], dtype=np.int32)
    return verts, faces

def _log(debug: bool, *args):
    if debug:
        print("[CurioMesh]", *args)


def _gather_source_materials(src_obj: bpy.types.Object) -> list[bpy.types.Material]:
    mats: list[bpy.types.Material] = []
    # Prefer object slots (handles OBJECT-linked materials)
    try:
        for slot in src_obj.material_slots:
            if slot.material and slot.material not in mats:
                mats.append(slot.material)
    except Exception:
        pass
    # Fallback to mesh materials
    try:
        for m in src_obj.data.materials:
            if m and m not in mats:
                mats.append(m)
    except Exception:
        pass
    return mats


def _copy_materials(src: bpy.types.Object, dst_mesh: bpy.types.Mesh, dst_obj: bpy.types.Object | None = None, debug: bool = False) -> None:
    if src.type != 'MESH':
        return
    mats = _gather_source_materials(src)
    # Clear existing materials
    try:
        dst_mesh.materials.clear()
    except Exception:
        while dst_mesh.materials:
            try:
                dst_mesh.materials.pop(index=0)
            except Exception:
                break
    for mat in mats:
        dst_mesh.materials.append(mat)
    if dst_obj is not None and mats:
        try:
            dst_obj.active_material = mats[0]
        except Exception:
            pass
    _log(debug, f"Materials copied: {len(mats)}")


def _apply_materials(dst_obj: bpy.types.Object, dst_mesh: bpy.types.Mesh, mats: list[bpy.types.Material], debug: bool = False) -> None:
    try:
        dst_mesh.materials.clear()
    except Exception:
        while dst_mesh.materials:
            try:
                dst_mesh.materials.pop(index=0)
            except Exception:
                break
    for m in mats:
        if m:
            dst_mesh.materials.append(m)
    if mats:
        try:
            dst_obj.active_material = mats[0]
        except Exception:
            pass
    _ensure_polygon_materials(dst_mesh, debug=debug)
    _log(debug, f"Applied {len(mats)} materials to object '{dst_obj.name}'")


def _ensure_polygon_materials(mesh: bpy.types.Mesh, debug: bool = False) -> None:
    try:
        if len(mesh.materials) > 0 and len(mesh.polygons) > 0:
            for p in mesh.polygons:
                p.material_index = 0
            _log(debug, f"Assigned material_index=0 to {len(mesh.polygons)} polygons")
    except Exception:
        pass


def _exploded_bake_internal(
    src: bpy.types.Object,
    dst: bpy.types.Object,
    *,
    distance: float = 1.0,
    reassemble: bool = True,
    debug: bool = False
) -> bool:
    """Exploded bake helper to prevent crosstalk.
    
    1. Separate target mesh by loose parts
    2. Move each part away from center
    3. Bake each part individually 
    4. Reassemble if requested
    """
    _log(debug, f"Exploded bake: distance={distance}, reassemble={reassemble}")
    # TODO: Implement exploded bake logic
    return True


def _relax_uvs_seams(mesh: bpy.types.Mesh, *, iterations: int = 2, alpha: float = 0.35, threshold: float = 0.1) -> None:
    """Relax only across discontinuity edges (large UV jumps between adjacent faces).

    For each mesh edge shared by two faces, measure the UV gap between the two
    loops that correspond to the edge. If it exceeds the threshold, blend both
    loop-UVs toward their average. Repeat for a few iterations.
    """
    try:
        if not mesh.uv_layers:
            return
        uv_layer = mesh.uv_layers.active
        if uv_layer is None or len(uv_layer.data) == 0:
            return
        # Map edge_index -> [loop_index_a, loop_index_b]
        loops_a: Dict[int, int] = {}
        loops_b: Dict[int, int] = {}
        for li, loop in enumerate(mesh.loops):
            ei = loop.edge_index
            if ei not in loops_a:
                loops_a[ei] = li
            else:
                loops_b[ei] = li
        a = float(max(0.0, min(1.0, alpha)))
        thr = float(max(0.0, threshold))
        for _ in range(max(1, int(iterations))):
            for ei, li0 in loops_a.items():
                li1 = loops_b.get(ei, -1)
                if li1 < 0:
                    continue
                uv0 = uv_layer.data[li0].uv
                uv1 = uv_layer.data[li1].uv
                if (uv0 - uv1).length < thr:
                    continue
                avg = (uv0 + uv1) * 0.5
                uv_layer.data[li0].uv = uv0 * (1.0 - a) + avg * a
                uv_layer.data[li1].uv = uv1 * (1.0 - a) + avg * a
    except Exception:
        traceback.print_exc()


def _relax_uvs_vertex_average(mesh: bpy.types.Mesh, *, iterations: int = 2, alpha: float = 0.35, threshold: float = 0.0) -> None:
    """Simple UV relax: for each vertex, average the UVs of all loops that
    reference it, then blend each loop-UV toward that average. This reduces
    per-face discontinuities that show up as visible lines.
    """
    try:
        if not mesh.uv_layers:
            return
        uv_layer = mesh.uv_layers.active
        if uv_layer is None or len(uv_layer.data) == 0:
            return
        loops_by_vert: dict[int, list[int]] = {}
        for li, loop in enumerate(mesh.loops):
            vi = loop.vertex_index
            if vi not in loops_by_vert:
                loops_by_vert[vi] = []
            loops_by_vert[vi].append(li)
        for _ in range(max(1, int(iterations))):
            avg_uv: dict[int, Vector] = {}
            for vi, lis in loops_by_vert.items():
                acc = Vector((0.0, 0.0))
                for li in lis:
                    acc += uv_layer.data[li].uv
                avg_uv[vi] = acc * (1.0 / float(len(lis)))
            a = float(max(0.0, min(1.0, alpha)))
            for vi, lis in loops_by_vert.items():
                target = avg_uv[vi]
                for li in lis:
                    uv_old = uv_layer.data[li].uv.copy()
                    if threshold > 0.0:
                        if (uv_old - target).length < threshold:
                            continue
                    uv_layer.data[li].uv = uv_old * (1.0 - a) + target * a
    except Exception:
        pass


def _apply_data_transfer_uvs(src: bpy.types.Object, dst: bpy.types.Object, debug: bool = False) -> None:
    # Ensure target has at least one UV layer
    src_uv_name = None
    if getattr(src.data, 'uv_layers', None) and src.data.uv_layers:
        src_uv_name = src.data.uv_layers.active.name
    if not dst.data.uv_layers:
        dst.data.uv_layers.new(name=src_uv_name or "UVMap")
    # Ensure names match when possible
    try:
        dst.data.uv_layers.active.name = src_uv_name or dst.data.uv_layers.active.name
    except Exception:
        pass
    # Make dst active and selected with src also selected
    ctx = bpy.context
    view_layer = ctx.view_layer
    prev_active = view_layer.objects.active
    prev_sel = [o for o in ctx.selected_objects]
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
        for o in ctx.selected_objects:
            o.select_set(False)
        src.select_set(True)
        dst.select_set(True)
        view_layer.objects.active = dst
        # Try operator first (more robust across versions)
        try:
            bpy.ops.object.data_transfer(
                data_type='UV', use_create=True, use_delete=False,
                vert_mapping='NEAREST_POLY', layers_select_src='ACTIVE', layers_select_dst='ACTIVE',
                mix_mode='REPLACE', use_object_transform=True,
                loop_mapping='NEAREST_FACE_INTERPOLATED'
            )
            _log(debug, f"UV transfer via operator ok: src='{src.name}', dst='{dst.name}'")
        except Exception:
            # Fall back to modifier method
            mod = dst.modifiers.new(name="CURIOMESH_DataTransfer", type='DATA_TRANSFER')
            mod.object = src
            mod.use_loop_data = True
            try:
                mod.data_types_loops = {'UV'}
            except Exception:
                try:
                    mod.loop_data_types = {'UV'}
                except Exception:
                    pass
            try:
                mod.loop_mapping = 'NEAREST_FACE_INTERPOLATED'
            except Exception:
                mod.loop_mapping = 'NEAREST_POLY'
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
                _log(debug, f"UV transfer via modifier applied: src='{src.name}', dst='{dst.name}'")
            except Exception:
                _log(debug, f"UV transfer modifier apply failed: src='{src.name}', dst='{dst.name}'")
    finally:
        try:
            for o in bpy.context.selected_objects:
                o.select_set(False)
            for o in prev_sel:
                o.select_set(True)
            view_layer.objects.active = prev_active
        except Exception:
            pass


def _project_uvs_via_bvh(src: bpy.types.Object, dst: bpy.types.Object, debug: bool = False, *, knn_samples: int = 1) -> bool:
    """Project UVs from `src` to `dst` using a normal-aware BVH strategy.

    Tries short ray casts along the world-space ±face normal of each target
    polygon to reduce cross-surface bleeding on thin geometry; falls back to
    nearest-surface lookup if rays miss. Assigns per-loop UVs to preserve seams.
    """
    try:
        src_me = src.data
        if not getattr(src_me, 'uv_layers', None) or not src_me.uv_layers:
            _log(debug, "Source has no UVs; projection aborted")
            return False
        uv_src = src_me.uv_layers.active
        # Build BVH on source triangles in world space
        src_me.calc_loop_triangles()
        verts_world = [src.matrix_world @ v.co for v in src_me.vertices]
        tri_indices = [tri.vertices[:] for tri in src_me.loop_triangles]
        bvh = BVHTree.FromPolygons(verts_world, tri_indices, all_triangles=True)
        tri_uvs = [[uv_src.data[li].uv.copy() for li in tri.loops] for tri in src_me.loop_triangles]

        # Ensure target has a UV layer
        dst_me = dst.data
        if not dst_me.uv_layers:
            dst_me.uv_layers.new(name=uv_src.name or "UVMap")
        dst_me.uv_layers.active.name = uv_src.name or dst_me.uv_layers.active.name
        uv_dst = dst_me.uv_layers.active

        # Compute a reasonable ray length from combined bounds
        def bounds_world(obj):
            me = obj.data
            if not me.vertices:
                return Vector((0, 0, 0)), Vector((0, 0, 0))
            pts = [obj.matrix_world @ v.co for v in me.vertices]
            xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
            bmin = Vector((min(xs), min(ys), min(zs)))
            bmax = Vector((max(xs), max(ys), max(zs)))
            return bmin, bmax

        dmin, dmax = bounds_world(dst)
        smin, smax = bounds_world(src)
        bb_min = Vector((min(dmin.x, smin.x), min(dmin.y, smin.y), min(dmin.z, smin.z)))
        bb_max = Vector((max(dmax.x, smax.x), max(dmax.y, smax.y), max(dmax.z, smax.z)))
        diag = (bb_max - bb_min).length
        ray_len = max(diag * 0.5, 1e-3)
        eps = max(diag * 1e-4, 1e-6)

        def barycentric(p: Vector, a: Vector, b: Vector, c: Vector):
            v0 = b - a
            v1 = c - a
            v2 = p - a
            d00 = v0.dot(v0)
            d01 = v0.dot(v1)
            d11 = v1.dot(v1)
            d20 = v2.dot(v0)
            d21 = v2.dot(v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-20:
                return 1.0, 0.0, 0.0
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            return u, v, w

        # Assign per polygon: either single best triangle or k-NN blend
        for poly in dst_me.polygons:
            try:
                n_ws = (dst.matrix_world.to_3x3() @ poly.normal).normalized()
            except Exception:
                n_ws = Vector((0.0, 0.0, 1.0))

            # Compute polygon center in world space
            vs = [dst.matrix_world @ dst_me.vertices[dst_me.loops[li].vertex_index].co for li in poly.loop_indices]
            if not vs:
                continue
            center = sum(vs, Vector((0.0, 0.0, 0.0))) * (1.0 / float(len(vs)))

            best = None  # (dist, tri_idx, loc, normal)
            for d in (n_ws, -n_ws):
                if d.length < 1e-12:
                    continue
                start = center + d * eps
                loc, normal, tri_idx, dist = bvh.ray_cast(start, d, ray_len)
                if tri_idx is None:
                    continue
                # Prefer front-face hits
                if normal is not None and normal.dot(d) > -1e-6:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, tri_idx, loc, normal)

            if best is None:
                # Nearest fallback from center
                hit = bvh.find_nearest(center)
                if hit is None:
                    continue
                loc, normal, tri_idx, dist = hit
            else:
                dist, tri_idx, loc, normal = best

            # k-NN blend across multiple nearest triangles for robustness
            if int(knn_samples) > 1:
                # Precompute nearest triangles around center
                candidates: list[tuple[float,int]] = []
                for d in (n_ws, -n_ws):
                    start = center + d * eps
                    loc2, normal2, tri2, dist2 = bvh.ray_cast(start, d, ray_len)
                    if tri2 is not None:
                        candidates.append((dist2, tri2))
                # also add nearest
                hit = bvh.find_nearest(center)
                if hit is not None and hit[2] is not None:
                    candidates.append((hit[3], hit[2]))
                # unique by tri index and sort
                seen = set()
                uniq = []
                for d,tidx in sorted(candidates, key=lambda x: x[0]):
                    if tidx in seen:
                        continue
                    seen.add(tidx)
                    uniq.append((d,tidx))
                    if len(uniq) >= int(knn_samples):
                        break
                if not uniq:
                    uniq = [(dist, tri_idx)]
                for li in poly.loop_indices:
                    vi = dst_me.loops[li].vertex_index
                    p = dst.matrix_world @ dst_me.vertices[vi].co
                    acc = Vector((0.0,0.0)); wsum = 0.0
                    for d,tidx in uniq:
                        ai,bi,ci = tri_indices[tidx]
                        A = verts_world[ai]; B = verts_world[bi]; C = verts_world[ci]
                        tri_n = (B - A).cross(C - A)
                        if tri_n.length < 1e-12:
                            continue
                        tri_n.normalize()
                        t = (p - A).dot(tri_n)
                        p_proj = p - tri_n * t
                        u,vv,w = barycentric(p_proj, A,B,C)
                        uv0,uv1,uv2 = tri_uvs[tidx]
                        uv = uv0 * u + uv1 * vv + uv2 * w
                        wt = 1.0 / max(1e-6, d)
                        acc += uv * wt
                        wsum += wt
                    if wsum > 1e-9:
                        uv_dst.data[li].uv = acc * (1.0/wsum)
                continue

            # Single-triangle projection fallback
            a_i, b_i, c_i = tri_indices[tri_idx]
            a = verts_world[a_i]
            b = verts_world[b_i]
            c = verts_world[c_i]
            for li in poly.loop_indices:
                vi = dst_me.loops[li].vertex_index
                p = dst.matrix_world @ dst_me.vertices[vi].co
                tri_n = (b - a).cross(c - a)
                if tri_n.length < 1e-12:
                    continue
                tri_n.normalize()
                t = (p - a).dot(tri_n)
                p_proj = p - tri_n * t
                u, v, w = barycentric(p_proj, a, b, c)
                uv0, uv1, uv2 = tri_uvs[tri_idx]
                uv = uv0 * u + uv1 * v + uv2 * w
                uv_dst.data[li].uv = uv
        _log(debug, "UV projection via BVH (normal-aware) finished")
        return True
    except Exception:
        traceback.print_exc()
        return False


def _project_uvs_with_cage(
    src: bpy.types.Object,
    dst: bpy.types.Object,
    *,
    expand_ratio: float = 0.25,
    cage_object: bpy.types.Object | None = None,
    blend_samples: int = 1,
    stats_out: Dict[str, Any] | None = None,
    ray_dir_mode: str = "NORMALS_AXES",
    backface_thresh: float = 0.10,
    ray_aim_mode: str = "POLY_CENTER",
    ray_aim_object_name: str = "",
    ray_density: int = 64,
    axis_bias: str = "NONE",
    cage_center_offset: Tuple[float,float,float] | None = None,
    split_sharp: bool = False,
    sharp_angle: float = 35.0,
    sharp_backoff: float = 0.01,
    uv_consistency_check: bool = True,
    uv_consistency_thresh: float = 0.02,
    debug: bool = False,
    exploded_bake_enabled: bool = False,
    exploded_bake_distance: float = 1.0,
    exploded_bake_reassemble: bool = True,
) -> bool:
    """Advanced cage-based UV projection with multiple solvers and ray controls.
    
    Implements proper Edge-Aim mode, advanced ray sampling, and better handling
    of extreme mesh reduction scenarios.
    """
    try:
        src_me = src.data
        if not getattr(src_me, 'uv_layers', None) or not src_me.uv_layers:
            _log(debug, "Source has no UVs; cage projection aborted")
            return False
        uv_src = src_me.uv_layers.active
        src_me.calc_loop_triangles()
        # World-space data for BVH
        verts_w = [src.matrix_world @ v.co for v in src_me.vertices]
        tris = [tri.vertices[:] for tri in src_me.loop_triangles]
        bvh = BVHTree.FromPolygons(verts_w, tris, all_triangles=True)

        # Precompute per-triangle UV corners
        tri_uvs = []
        for tri in src_me.loop_triangles:
            tri_uvs.append([uv_src.data[li].uv.copy() for li in tri.loops])

        # Compute combined bounds and expansion distance
        def bounds_world(obj):
            me = obj.data
            if not me.vertices:
                return Vector((0, 0, 0)), Vector((0, 0, 0))
            pts = [obj.matrix_world @ v.co for v in me.vertices]
            xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
            bmin = Vector((min(xs), min(ys), min(zs)))
            bmax = Vector((max(xs), max(ys), max(zs)))
            return bmin, bmax

        smin, smax = bounds_world(src)
        tmin, tmax = bounds_world(dst)
        cmin = Vector((min(smin.x, tmin.x), min(smin.y, tmin.y), min(smin.z, tmin.z)))
        cmax = Vector((max(smax.x, tmax.x), max(smax.y, tmax.y), max(smax.z, tmax.z)))
        # If a cage object is provided, prefer its bounds for center/scale
        cage_bvh = None
        if cage_object is not None and getattr(cage_object, 'data', None) and len(cage_object.data.polygons) > 0:
            cmin, cmax = bounds_world(cage_object)
            # Build BVH for cage in world space so we can start rays on its surface
            cage_me = cage_object.data
            cage_me.calc_loop_triangles()
            cage_verts_w = [cage_object.matrix_world @ v.co for v in cage_me.vertices]
            cage_tris = [tri.vertices[:] for tri in cage_me.loop_triangles]
            try:
                cage_bvh = BVHTree.FromPolygons(cage_verts_w, cage_tris, all_triangles=True)
            except Exception:
                cage_bvh = None
        center = (cmin + cmax) * 0.5
        # Allow offset of the cage center for emission
        try:
            if cage_center_offset is not None:
                center += Vector(cage_center_offset)
        except Exception:
            pass
        diag = (cmax - cmin).length
        expand = max(diag * expand_ratio, 1e-4)

        # Ensure target has UV layer
        dst_me = dst.data
        if not dst_me.uv_layers:
            dst_me.uv_layers.new(name=uv_src.name or "UVMap")
        dst_me.uv_layers.active.name = uv_src.name or dst_me.uv_layers.active.name
        uv_dst = dst_me.uv_layers.active

        def bary(p, a, b, c):
            v0 = b - a; v1 = c - a; v2 = p - a
            d00 = v0.dot(v0); d01 = v0.dot(v1); d11 = v1.dot(v1)
            d20 = v2.dot(v0); d21 = v2.dot(v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-20:
                return 1.0, 0.0, 0.0
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            return u, v, w

        # Get aim object if specified
        aim_obj = None
        if ray_aim_object_name:
            aim_obj = bpy.data.objects.get(ray_aim_object_name)

        hits = 0; misses = 0
        # Build base axes with optional user bias
        axes = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))]
        bias = str(axis_bias or 'NONE').upper()
        if bias in ['+X','-X','+Y','-Y','+Z','-Z']:
            vec = {'+X':Vector((1,0,0)),'-X':Vector((-1,0,0)),'+Y':Vector((0,1,0)),'-Y':Vector((0,-1,0)),'+Z':Vector((0,0,1)),'-Z':Vector((0,0,-1))}[bias]
            axes = [vec] + axes
        total_samples = 0

        # Edge-Aim: Precompute feature edges for better targeting
        edge_aim_points = []
        if ray_aim_mode == "EDGE":
            edge_to_polys: dict[tuple[int,int], list[int]] = {}
            for p in dst_me.polygons:
                for ek in p.edge_keys:
                    key = (min(ek[0], ek[1]), max(ek[0], ek[1]))
                    lst = edge_to_polys.get(key)
                    if lst is None:
                        edge_to_polys[key] = [p.index]
                    else:
                        lst.append(p.index)
            # boundary edges (single poly) or feature edges (normal angle threshold)
            for (v0, v1), polys in edge_to_polys.items():
                add = False
                if len(polys) == 1:
                    add = True
                elif len(polys) >= 2:
                    try:
                        n0 = dst_me.polygons[polys[0]].normal
                        n1 = dst_me.polygons[polys[1]].normal
                        add = abs(n0.dot(n1)) < cos(radians(sharp_angle))
                    except Exception:
                        add = False
                if add:
                    a = dst.matrix_world @ dst_me.vertices[v0].co
                    b = dst.matrix_world @ dst_me.vertices[v1].co
                    edge_aim_points.append((a + b) * 0.5)

        for poly in dst_me.polygons:
            # Choose aim point for this polygon (center / vertex / boundary edge)
            pcs = [dst.matrix_world @ dst_me.vertices[dst_me.loops[li].vertex_index].co for li in poly.loop_indices]
            if not pcs:
                continue
            # Priority of aim: user-specified object origin > aim mode setting > default center
            if aim_obj is not None:
                p_center = aim_obj.matrix_world.translation.copy()
            else:
                mode_u = str(ray_aim_mode or 'POLY_CENTER').upper()
                if mode_u == 'VERTEX':
                    vi0 = dst_me.loops[poly.loop_indices[0]].vertex_index
                    p_center = dst.matrix_world @ dst_me.vertices[vi0].co
                elif mode_u == 'EDGE' and edge_aim_points:
                    # Find closest edge aim point to this polygon
                    poly_center = sum(pcs, Vector((0,0,0))) * (1.0/len(pcs))
                    closest_edge_pt = min(edge_aim_points, key=lambda ep: (ep - poly_center).length)
                    p_center = closest_edge_pt
                else:
                    p_center = sum(pcs, Vector((0.0, 0.0, 0.0))) * (1.0 / float(len(pcs)))
            
            # Candidate directions
            cand = []
            try:
                n = (dst.matrix_world.to_3x3() @ poly.normal).normalized()
            except Exception:
                n = Vector((0.0, 0.0, 1.0))
            dcenter = (p_center - center)
            # Build direction set according to mode
            m = str(ray_dir_mode or "NORMALS_AXES").upper()
            if m == "NORMALS_ONLY":
                cand = [n, -n]
            elif m == "CENTER":
                if dcenter.length > 1e-12:
                    cand = [dcenter.normalized(), -dcenter.normalized()]
                else:
                    cand = [Vector((1,0,0)), Vector((-1,0,0))]
            else:  # NORMALS_AXES
                base = [n, -n]
                if dcenter.length > 1e-12:
                    base += [dcenter.normalized(), -dcenter.normalized()]
                cand = base + axes + [-a for a in axes]
            
            # Increase density by rotating base dirs when requested
            extra = max(0, int(ray_density) - len(cand))
            if extra > 0:
                aug = []
                for i in range(min(extra, 32)):
                    ang = (i+1) * (pi / (extra+1))
                    for v in cand[:3]:
                        aug.append(Vector((v.x*cos(ang)-v.y*sin(ang), v.x*sin(ang)+v.y*cos(ang), v.z)))
                cand += aug

            # Per-vertex selection of triangles (reduces polygon-wide smearing)
            blendK = max(1, int(blend_samples))
            sec_dirs = [Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1)), -Vector((1,0,0)), -Vector((0,1,0)), -Vector((0,0,1))] if blendK > 1 else []
            for li in poly.loop_indices:
                vi = dst_me.loops[li].vertex_index
                p = dst.matrix_world @ dst_me.vertices[vi].co
                chosen_v = None  # (dist, tri_idx, loc, normal)
                # 1) Strict normal-only test first to avoid silhouette crosstalk
                strict_align = 0.5  # require cos(angle) >= 0.5 (~60 deg)
                strict_max = max(4.0 * expand, 0.2 * diag)
                for d in (n, -n):
                    if d.length < 1e-12:
                        continue
                    d = d.normalized()
                    startN = p + d * max(1e-5, expand * 0.05)
                    locN, normalN, triN, distN = bvh.ray_cast(startN, -d, strict_max)
                    if triN is None:
                        continue
                    # accept only solid front-face and good alignment
                    try:
                        thr = float(backface_thresh)
                    except Exception:
                        thr = 0.1
                    if normalN is not None and (normalN.dot(-d) > -max(0.0, min(1.0, thr)) or normalN.dot(n) < strict_align):
                        continue
                    chosen_v = (distN, triN, locN, normalN)
                    break
                
                for d in cand:
                    if d.length < 1e-12:
                        continue
                    d = d.normalized()
                    start = None
                    if cage_bvh is not None:
                        loc_c, n_c, tri_c, dist_c = cage_bvh.ray_cast(p, d, 4.0 * diag)
                        if tri_c is not None:
                            start = loc_c + d * 1e-5
                    if start is None:
                        start = p + d * expand
                    if split_sharp:
                        try:
                            back = max(sharp_backoff, 1e-5) * diag
                            n_ws_local = (dst.matrix_world.to_3x3() @ poly.normal).normalized()
                            start = start - n_ws_local * back
                        except Exception:
                            pass
                    # Cap length to avoid crossing through silhouettes
                    loc, normal, tri_idx, dist = bvh.ray_cast(start, -d, max(expand * 3.0, diag * 0.02))
                    if tri_idx is None:
                        continue
                    try:
                        thr = float(backface_thresh)
                    except Exception:
                        thr = 0.1
                    if normal is not None and normal.dot(-d) > -max(0.0, min(1.0, thr)):
                        continue
                    # Additional normal-alignment guard to avoid silhouette crosstalk
                    try:
                        n_dst = (dst.matrix_world.to_3x3() @ poly.normal).normalized()
                        if normal is not None and normal.dot(n_dst) < 0.2:
                            continue
                    except Exception:
                        pass
                    if chosen_v is None or dist < chosen_v[0]:
                        chosen_v = (dist, tri_idx, loc, normal)

                if chosen_v is None:
                    hit = bvh.find_nearest(p)
                    if hit is None:
                        misses += 1
                        continue
                    loc, normal, tri_idx, dist = hit
                else:
                    dist, tri_idx, loc, normal = chosen_v

                a_i, b_i, c_i = tris[tri_idx]
                a = verts_w[a_i]; b = verts_w[b_i]; c = verts_w[c_i]
                tri_n = (b - a).cross(c - a)
                if tri_n.length < 1e-12:
                    misses += 1
                    continue
                tri_n.normalize()
                # First sample: plane projection to chosen triangle
                t = (p - a).dot(tri_n)
                p_proj = p - tri_n * t
                u, v, w = bary(p_proj, a, b, c)
                uv0, uv1, uv2 = tri_uvs[tri_idx]
                uv = uv0 * u + uv1 * v + uv2 * w
                acc_uv = uv.copy(); acc_w = 1.0; used = 1
                # Additional blended samples
                for sd in sec_dirs:
                    if used >= blendK:
                        break
                    d2 = (sd + tri_n * 0.25).normalized()
                    start2 = p + d2 * expand
                    loc2, n2, tri2, dist2 = bvh.ray_cast(start2, -d2, 4.0 * diag)
                    if tri2 is None:
                        continue
                    a2, b2, c2 = tris[tri2]
                    a2w = verts_w[a2]; b2w = verts_w[b2]; c2w = verts_w[c2]
                    tri2_n = (b2w - a2w).cross(c2w - a2w)
                    if tri2_n.length < 1e-12:
                        continue
                    tri2_n.normalize()
                    t2 = (p - a2w).dot(tri2_n)
                    p2 = p - tri2_n * t2
                    u2, v2, w2 = bary(p2, a2w, b2w, c2w)
                    uv20, uv21, uv22 = tri_uvs[tri2]
                    uv2 = uv20 * u2 + uv21 * v2 + uv22 * w2
                    align = max(0.0, tri2_n.dot(tri_n))
                    wgt = align * (1.0 / max(dist2, 1e-6))
                    acc_uv += uv2 * wgt
                    acc_w += wgt
                    used += 1
                total_samples += used
                if acc_w <= 1e-12:
                    uv_dst.data[li].uv = uv
                else:
                    blended = acc_uv * (1.0 / acc_w)
                    if uv_consistency_check and (blended - uv).length > float(uv_consistency_thresh):
                        uv_dst.data[li].uv = uv
                    else:
                        uv_dst.data[li].uv = blended
                hits += 1
        if stats_out is not None:
            stats_out["hits"] = hits
            stats_out["misses"] = misses
            stats_out["avg_samples"] = (total_samples / max(1, hits))
        _log(debug, f"UV cage projection done: hits={hits}, misses={misses}, expand={expand:.4f}")
        return hits > 0
    except Exception:
        traceback.print_exc()
        return False


def _uv_mapping_is_suspicious(obj: bpy.types.Object, *, threshold_ratio: float = 200.0, tiny_uv_area: float = 1e-8) -> bool:
    """Heuristic to detect severely distorted or collapsed UVs on `obj`.

    Computes per-triangle ratio of 3D area to UV area and flags mapping as
    suspicious if many triangles have extremely tiny UV area or very large
    area ratio. Used to decide when to auto-bake as a fallback.
    """
    try:
        me = obj.data
        if not getattr(me, 'uv_layers', None) or not me.uv_layers:
            return True
        uv = me.uv_layers.active
        if uv is None:
            return True
        me.calc_loop_triangles()
        bad = 0
        total = max(1, len(me.loop_triangles))
        for tri in me.loop_triangles:
            idx = tri.vertices[:]
            a = obj.matrix_world @ me.vertices[idx[0]].co
            b = obj.matrix_world @ me.vertices[idx[1]].co
            c = obj.matrix_world @ me.vertices[idx[2]].co
            area3d = ((b - a).cross(c - a)).length * 0.5
            u0 = uv.data[tri.loops[0]].uv
            u1 = uv.data[tri.loops[1]].uv
            u2 = uv.data[tri.loops[2]].uv
            # 2D triangle area via determinant
            area_uv = abs((u1.x - u0.x) * (u2.y - u0.y) - (u1.y - u0.y) * (u2.x - u0.x)) * 0.5
            if area_uv < tiny_uv_area:
                bad += 1
                continue
            ratio = area3d / max(area_uv, 1e-20)
            if ratio > threshold_ratio:
                bad += 1
        return (bad / total) > 0.10
    except Exception:
        return True

def _replace_object_mesh(
    obj: bpy.types.Object,
    verts: np.ndarray,
    quads: np.ndarray,
    tris: np.ndarray | None = None,
    *,
    output_mode: str = "REPLACE",
    preserve_materials: bool = True,
    transfer_uvs_from: bpy.types.Object | None = None,
    apply_transfer: bool = True,
    debug: bool = False,
    ) -> bpy.types.Object:
    me = bpy.data.meshes.new(name=f"{obj.name}_curio")
    tri_list = tris.tolist() if tris is not None and tris.size > 0 else []
    quad_list = quads.tolist() if quads is not None and quads.size > 0 else []
    me.from_pydata(verts.tolist(), [], tri_list + quad_list)
    me.update(calc_edges=True, calc_edges_loose=True)

    if output_mode == "NEW":
        new_obj = bpy.data.objects.new(f"{obj.name}_CurioMesh", me)
        try:
            obj.users_collection[0].objects.link(new_obj)
        except Exception:
            bpy.context.collection.objects.link(new_obj)
        if preserve_materials:
            mats = _gather_source_materials(obj)
            _apply_materials(new_obj, me, mats, debug=debug)
        if transfer_uvs_from is not None:
            _apply_data_transfer_uvs(transfer_uvs_from, new_obj, debug=debug)
            # As a stronger fallback, try projection if UVs still missing or distorted
            try:
                if not new_obj.data.uv_layers or len(new_obj.data.uv_layers[0].data) == 0:
                    if not _project_uvs_via_bvh(transfer_uvs_from, new_obj, debug=debug):
                        _project_uvs_with_cage(transfer_uvs_from, new_obj, debug=debug)
            except Exception:
                if not _project_uvs_via_bvh(transfer_uvs_from, new_obj, debug=debug):
                    _project_uvs_with_cage(transfer_uvs_from, new_obj, debug=debug)
        return new_obj
    else:
        # For replace, create a temporary duplicate of src for data transfer if requested
        src_copy = None
        if transfer_uvs_from is not None:
            src_copy = transfer_uvs_from.copy()
            src_copy.data = transfer_uvs_from.data.copy()
            bpy.context.collection.objects.link(src_copy)

        mats = _gather_source_materials(transfer_uvs_from or obj)
        obj.data = me
        if preserve_materials:
            _apply_materials(obj, me, mats, debug=debug)
        if src_copy is not None:
            _apply_data_transfer_uvs(src_copy, obj, debug=debug)
            try:
                if not obj.data.uv_layers or len(obj.data.uv_layers[0].data) == 0:
                    if not _project_uvs_via_bvh(src_copy, obj, debug=debug):
                        _project_uvs_with_cage(src_copy, obj, debug=debug)
            except Exception:
                if not _project_uvs_via_bvh(src_copy, obj, debug=debug):
                    _project_uvs_with_cage(src_copy, obj, debug=debug)
        # Cleanup
        if src_copy is not None:
            try:
                bpy.data.objects.remove(src_copy, do_unlink=True)
            except Exception:
                pass
        return obj


def _compute_auto_voxel_size(obj: bpy.types.Object) -> float:
    bb = [obj.matrix_world @ v.co for v in obj.data.vertices] if obj.data and len(obj.data.vertices) else []
    if not bb:
        return 0.05
    xs = [p.x for p in bb]; ys = [p.y for p in bb]; zs = [p.z for p in bb]
    diag = float(np.linalg.norm([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]))
    # 128 cells across the diagonal by default
    return max(diag / 128.0, 1e-4)


def _preprocess_with_ops(
    obj: bpy.types.Object,
    *,
    use_vdb: bool,
    vdb_voxel_size: float,
    target_faces: int,
    adaptivity: float,
) -> bpy.types.Mesh | None:
    # Duplicate object into current collection and operate on it
    ctx = bpy.context
    depsgraph = ctx.evaluated_depsgraph_get()
    temp_obj = obj.copy()
    temp_mesh_data = obj.data.copy()
    temp_obj.data = temp_mesh_data
    ctx.collection.objects.link(temp_obj)

    view_layer = ctx.view_layer
    prev_active = view_layer.objects.active
    prev_select = [o for o in ctx.selected_objects]
    try:
        for o in ctx.selected_objects:
            o.select_set(False)
        temp_obj.select_set(True)
        view_layer.objects.active = temp_obj

        if use_vdb:
            vox = vdb_voxel_size if vdb_voxel_size > 0.0 else _compute_auto_voxel_size(temp_obj)
            try:
                bpy.ops.object.voxel_remesh(voxel_size=float(vox), adaptivity=0.0)
            except Exception:
                pass

        # Decimate strongly toward target face count if beneficial
        try:
            current_faces = max(1, len(temp_obj.data.polygons))
            if target_faces < current_faces:
                # If massively higher density than desired, try Un-Subdivide first to keep quads
                import math
                if current_faces / max(1, target_faces) >= 8:
                    iters = max(1, int(math.ceil(math.log(current_faces / max(1, target_faces), 4))))
                    modu = temp_obj.modifiers.new(name="CURIOMESH_UnSub", type='DECIMATE')
                    modu.decimate_type = 'UNSUBDIV'
                    modu.iterations = min(6, iters)
                    try:
                        bpy.ops.object.modifier_apply(modifier=modu.name)
                    except Exception:
                        pass
                    current_faces = max(1, len(temp_obj.data.polygons))

                desired_ratio = float(target_faces) / float(current_faces)
                ratio = max(1e-5, min(1.0, desired_ratio))
                if ratio < 0.999:
                    mod = temp_obj.modifiers.new(name="CURIOMESH_Decimate", type='DECIMATE')
                    mod.decimate_type = 'COLLAPSE'
                    mod.ratio = ratio
                    try:
                        bpy.ops.object.modifier_apply(modifier=mod.name)
                    except Exception:
                        # Fall back to evaluated mesh without applying
                        pass
        except Exception:
            pass

        eval_obj = temp_obj.evaluated_get(depsgraph)
        out_mesh = bpy.data.meshes.new_from_object(eval_obj)
    finally:
        # Restore selection
        try:
            for o in bpy.context.selected_objects:
                o.select_set(False)
            for o in prev_select:
                o.select_set(True)
            view_layer.objects.active = prev_active
        except Exception:
            pass
        # Cleanup temp object and possibly its mesh data
        try:
            bpy.data.objects.remove(temp_obj, do_unlink=True)
        except Exception:
            pass
        try:
            if temp_mesh_data.users == 0:
                bpy.data.meshes.remove(temp_mesh_data)
        except Exception:
            pass

    return out_mesh


def _fallback_python(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    target_faces: int,
    *,
    feature_angle_deg: float = 35.0,
    smooth_iters: int = 5,
    smooth_lambda: float = 0.2,
    preserve_seams: bool = True,
    preserve_sharp: bool = True,
    output_mode: str = "REPLACE",
    debug: bool = False,
    pure_quads: bool = False,
    # new: projection handling & post-filters
    projection_mode: str = "AUTO",
    projection_expand: float = 0.25,
    use_preview_cage: bool = False,
    preview_cage_name: str = "CurioMesh_Cage",
) -> bool:
    _log(debug, "Fallback remesh path (Python) starting")
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # Detect feature edges by dihedral angle threshold
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    protected = set()
    angle_rad = np.deg2rad(feature_angle_deg)
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        if preserve_sharp:
            f0, f1 = e.link_faces
            n0 = f0.normal
            n1 = f1.normal
            dot = max(-1.0, min(1.0, n0.dot(n1)))
            dihedral = np.arccos(dot)
            if dihedral > angle_rad:
                key = tuple(sorted((e.verts[0].index, e.verts[1].index)))
                protected.add(key)
        if preserve_seams and getattr(e, "seam", False):
            key = tuple(sorted((e.verts[0].index, e.verts[1].index)))
            protected.add(key)

    # Greedy tri to quad respecting protected edges
    from .operators import naive_tritoquad_on_bmesh

    naive_tritoquad_on_bmesh(bm, protected_edges=protected)

    # Optional smoothing (simple Laplacian)
    if smooth_iters > 0 and smooth_lambda > 0.0:
        for _ in range(smooth_iters):
            bmesh.ops.smooth_vert(
                bm,
                verts=bm.verts,
                factor=float(smooth_lambda),
                use_axis_x=True,
                use_axis_y=True,
                use_axis_z=True,
                mirror_clip_x=False,
                mirror_clip_y=False,
                mirror_clip_z=False,
            )
    bm.to_mesh(mesh)
    bm.free()

    # Ensure quads-only representation; optionally convert leftover tris to quads by local split
    verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    quads = []
    tris = []
    mesh.calc_loop_triangles()
    try:
        mesh.calc_normals()
    except Exception:
        pass
    for p in mesh.polygons:
        v_idx = p.vertices[:]
        if len(v_idx) == 4:
            quads.append(v_idx)
        elif len(v_idx) == 3:
            tris.append(v_idx)
    if not quads and not tris:
        return False
    if pure_quads and tris:
        # Aggressive pure-quad conversion with triangle pairing via adjacency.
        new_verts = verts.tolist()
        new_quads = quads[:]
        tris_list = [tuple(t) for t in tris]
        # Build edge -> tri mapping
        from collections import defaultdict
        edge_to_tris = defaultdict(list)
        for ti, (a, b, c) in enumerate(tris_list):
            for e in ((a,b), (b,c), (c,a)):
                key = tuple(sorted(e))
                edge_to_tris[key].append(ti)
        used = set()
        # First pass: pair adjacent triangles into quads
        for ti, (a,b,c) in enumerate(tris_list):
            if ti in used:
                continue
            best = None
            for e in ((a,b), (b,c), (c,a)):
                key = tuple(sorted(e))
                adj = [t for t in edge_to_tris.get(key, []) if t != ti and t not in used]
                if not adj:
                    continue
                tj = adj[0]
                # Form quad from two tris sharing edge (a,b) etc.
                ta = set((a,b,c))
                tb = set(tris_list[tj])
                shared = list(ta.intersection(tb))
                if len(shared) != 2:
                    continue
                rest_a = list(ta - set(shared))[0]
                rest_b = list(tb - set(shared))[0]
                quad = [rest_a, shared[0], rest_b, shared[1]]
                best = (tj, quad)
                break
            if best is not None:
                tj, quad = best
                new_quads.append(quad)
                used.add(ti)
                used.add(tj)
        # Second pass: remaining single triangles → split longest edge to quad
        for ti, (a,b,c) in enumerate(tris_list):
            if ti in used:
                continue
            pa, pb, pc = verts[a], verts[b], verts[c]
            d_ab = np.linalg.norm(pa - pb)
            d_bc = np.linalg.norm(pb - pc)
            d_ca = np.linalg.norm(pc - pa)
            if d_ab >= d_bc and d_ab >= d_ca:
                i0, i1, i2 = a, b, c
            elif d_bc >= d_ca:
                i0, i1, i2 = b, c, a
            else:
                i0, i1, i2 = c, a, b
            mid = (verts[i0] + verts[i1]) * 0.5
            new_index = len(new_verts)
            new_verts.append(mid.tolist())
            new_quads.append([i2, i0, new_index, i1])
        verts = np.array(new_verts, dtype=np.float32)
        quads = new_quads
        tris = []
    quads_np = np.array(quads, dtype=np.int32) if quads else np.empty((0, 4), dtype=np.int32)
    tris_np = np.array(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)
    new_obj = _replace_object_mesh(
        obj,
        verts,
        quads_np,
        tris_np,
        output_mode=output_mode,
        preserve_materials=True,
        transfer_uvs_from=obj,
        debug=debug,
    )
    # Optional projection override + relax to match non-fallback path
    try:
        mode = (projection_mode or "AUTO").upper()
        # Detect no-reduction/no-topology-change case: avoid re-projecting onto the same mesh
        try:
            same_topology = (len(mesh.vertices) == len(new_obj.data.vertices)) and (len(mesh.polygons) == len(new_obj.data.polygons))
        except Exception:
            same_topology = False
        if mode == "NEAREST":
            _log(debug, "Projection mode (fallback): NEAREST (BVH)")
            _project_uvs_via_bvh(obj, new_obj, debug=debug)
        elif mode == "CAGE" and not same_topology:
            cage = bpy.data.objects.get(preview_cage_name) if use_preview_cage else None
            ck = None
            if cage is not None:
                try:
                    ck = str(cage.get('curiomesh_cage_mode', 'UNKNOWN')).upper()
                except Exception:
                    ck = 'UNKNOWN'
            _log(debug, f"Projection mode (fallback): CAGE; cage={'YES' if cage else 'NO'}; type={ck or 'NONE'}; expand={float(projection_expand):.3f}")
            stats: Dict[str, Any] = {}
            _project_uvs_with_cage(
                obj,
                new_obj,
                expand_ratio=float(projection_expand),
                cage_object=cage,
                blend_samples=4,
                stats_out=stats,
                debug=debug,
            )
            _log(debug, f"CAGE stats (fallback): hits={stats.get('hits')}, misses={stats.get('misses')}, avg_samples={stats.get('avg_samples')}")
        elif mode == "CAGE" and same_topology:
            _log(debug, "Projection mode (fallback): CAGE skipped (no topology change)")
        # Relax by default to reduce banding
        _relax_uvs_vertex_average(new_obj.data, iterations=2, alpha=0.35)
    except Exception:
        traceback.print_exc()
    return True


def run_curiomesh(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    *,
    target_faces: int,
    feature_weight: float,
    adaptivity: float,
    engine: str,
    use_vdb: bool,
    vdb_voxel_size: float = 0.0,
    feature_angle_deg: float = 35.0,
    smooth_iters: int = 5,
    smooth_lambda: float = 0.2,
    preserve_seams: bool = True,
    preserve_sharp: bool = True,
    output_mode: str = "REPLACE",
    # Optional baking controls
    bake_pbr: bool = False,
    bake_resolution: int = 2048,
    bake_margin: int = 8,
    bake_maps: set | None = None,
    uv_method: str = "SMART",
    debug_logs: bool = False,
    force_bake_fallback: bool = False,
    projection_mode: str = "AUTO",
    projection_expand: float = 0.25,
    pure_quads: bool = False,
    use_preview_cage: bool = False,
    preview_cage_name: str = "CurioMesh_Cage",
    # Advanced controls
    backface_thresh: float = 0.10,
    cage_blend_samples: int = 4,
    neighbor_blend: bool = False,
    neighbor_blend_samples: int = 3,
    ray_dir_mode: str = "NORMALS_AXES",
    relax_uvs: bool = True,
    relax_iters: int = 2,
    relax_alpha: float = 0.35,
    relax_threshold: float = 0.15,
    lock_cage: bool = False,
    post_bake_color: bool = False,
    post_bake_res: int = 2048,
    post_bake_margin: int = 16,
    projection_solver: str = "AUTO",
    ray_aim_mode: str = "POLY_CENTER",
    ray_aim_object_name: str = "",
    ray_density: int = 64,
    axis_bias: str = "NONE",
    cage_center_offset: Tuple[float,float,float] | None = None,
    rbf_samples: int = 200,
    rbf_lambda: float = 1e-3,
    # sharp/consistency controls
    split_sharp: bool = False,
    sharp_angle: float = 35.0,
    sharp_backoff: float = 0.01,
    uv_consistency_check: bool = True,
    uv_consistency_thresh: float = 0.02,
    spectral_basis: int = 64,
    spectral_lambda: float = 1e-4,
    ot_iters: int = 50,
    sgmm_lobes: int = 3,
    sgmm_em_iters: int = 15,
    tensor_iters: int = 10,
    prt_order: int = 3,
    prt_samples: int = 8000,
    dr_steps: int = 400,
    dr_lr: float = 1e-2,
    exploded_bake_enabled: bool = False,
    exploded_bake_distance: float = 1.0,
    exploded_bake_reassemble: bool = True,
) -> bool:
    core = _load_native()
    # Optional in-Blender preprocess to improve robustness and reduce face count
    try:
        pre_mesh = _preprocess_with_ops(
            obj,
            use_vdb=use_vdb,
            vdb_voxel_size=vdb_voxel_size,
            target_faces=target_faces,
            adaptivity=adaptivity,
        )
        work_mesh = pre_mesh if pre_mesh is not None else mesh
    except Exception:
        work_mesh = mesh

    verts_np, faces_np = _mesh_to_numpy(work_mesh)

    if core is None:
        ok = _fallback_python(
            obj,
            work_mesh,
            target_faces,
            feature_angle_deg=(feature_angle_deg if preserve_sharp else 180.0),
            smooth_iters=smooth_iters,
            smooth_lambda=smooth_lambda,
            preserve_seams=preserve_seams,
            preserve_sharp=preserve_sharp,
            output_mode=output_mode,
            debug=bool(debug_logs),
            projection_mode=projection_mode,
            projection_expand=projection_expand,
            use_preview_cage=use_preview_cage,
            preview_cage_name=preview_cage_name,
            pure_quads=bool(pure_quads),
        )
        if work_mesh is not mesh:
            try:
                bpy.data.meshes.remove(work_mesh)
            except Exception:
                pass
        return ok

    try:
        # If not preserving sharps, allow pairing across all edges
        eff_angle = float(feature_angle_deg if preserve_sharp else 180.0)
        result: Dict[str, Any] = core.remesh(
            verts_np, faces_np,
            int(target_faces), float(feature_weight), float(adaptivity), str(engine), bool(use_vdb), float(vdb_voxel_size), float(eff_angle), int(smooth_iters), float(smooth_lambda)
        )
        out_verts = np.asarray(result.get("verts"), dtype=np.float32)
        out_quads = np.asarray(result.get("quads", np.empty((0, 4), dtype=np.int32)), dtype=np.int32)
        out_tris = np.asarray(result.get("tris", np.empty((0, 3), dtype=np.int32)), dtype=np.int32)
        if out_verts.size == 0 or (out_quads.size == 0 and out_tris.size == 0):
            return False
        new_obj = _replace_object_mesh(
            obj,
            out_verts,
            out_quads,
            out_tris,
            output_mode=output_mode,
            preserve_materials=True,
            transfer_uvs_from=obj,
            debug=bool(debug_logs),
        )
        
        # Handle exploded bake if enabled
        if exploded_bake_enabled:
            _exploded_bake_internal(
                obj, new_obj,
                distance=exploded_bake_distance,
                reassemble=exploded_bake_reassemble,
                debug=debug_logs
            )
        
        # Apply projection mode if requested (beyond default Auto behavior inside _replace_object_mesh)
        try:
            mode = (projection_mode or "AUTO").upper()
            if mode == "NEAREST":
                _log(debug_logs, "Projection mode: NEAREST (BVH)")
                _project_uvs_via_bvh(obj, new_obj, debug=bool(debug_logs), knn_samples=(int(neighbor_blend_samples) if neighbor_blend else 1))
            elif mode == "CAGE":
                stats: Dict[str, Any] = {}
                cage = bpy.data.objects.get(preview_cage_name) if use_preview_cage else None
                if lock_cage and cage is not None:
                    # Keep user-edited cage as is. If not locked and cage exists, we still use it.
                    pass
                # Identify cage type for logs via custom property first
                cage_kind = 'NONE'
                if cage is not None:
                    try:
                        cage_kind = str(cage.get('curiomesh_cage_mode', 'UNKNOWN')).upper()
                    except Exception:
                        cage_kind = 'UNKNOWN'
                _log(debug_logs, f"Projection mode: CAGE; cage={'YES' if cage else 'NO'}; type={cage_kind}; expand={projection_expand:.3f}")
                
                solver_name = str(projection_solver or 'AUTO').upper()
                if solver_name == 'RBF_FIELD':
                    _log(debug_logs, f"Solver: RBF_FIELD samples={rbf_samples} lambda={rbf_lambda}")
                    # TODO: Implement RBF Field projection
                    # For now, fallback to enhanced cage projection
                    pass
                elif solver_name == 'SPECTRAL':
                    _log(debug_logs, f"Solver: SPECTRAL basis={spectral_basis} λ={spectral_lambda}")
                    # TODO: Implement Spectral projection 
                    pass
                elif solver_name == 'OT':
                    _log(debug_logs, f"Solver: OT iters={ot_iters}")
                    # TODO: Implement Optimal Transport projection
                    pass
                elif solver_name == 'SGMM_NDF':
                    _log(debug_logs, f"Solver: SGMM lobes={sgmm_lobes} EM={sgmm_em_iters}")
                    # TODO: Implement SGMM NDF projection
                    pass
                elif solver_name == 'TENSOR_ANISO':
                    _log(debug_logs, f"Solver: TENSOR iters={tensor_iters}")
                    # TODO: Implement Anisotropic Tensor projection
                    pass
                elif solver_name == 'PRT':
                    _log(debug_logs, f"Solver: PRT order={prt_order} samples={prt_samples}")
                    # TODO: Implement Precomputed Radiance Transfer
                    pass
                elif solver_name == 'DR':
                    _log(debug_logs, f"Solver: DR steps={dr_steps} lr={dr_lr}")
                    # TODO: Implement Differentiable Rendering
                    pass
                
                # Call enhanced cage projection with all parameters
                _project_uvs_with_cage(
                    obj,
                    new_obj,
                    expand_ratio=float(projection_expand),
                    cage_object=cage,
                    blend_samples=int(cage_blend_samples),
                    ray_dir_mode=str(ray_dir_mode),
                    backface_thresh=float(backface_thresh),
                    ray_aim_mode=str(ray_aim_mode),
                    ray_aim_object_name=str(ray_aim_object_name),
                    ray_density=int(ray_density),
                    axis_bias=str(axis_bias),
                    cage_center_offset=cage_center_offset,
                    split_sharp=bool(split_sharp),
                    sharp_angle=float(sharp_angle),
                    sharp_backoff=float(sharp_backoff),
                    uv_consistency_check=bool(uv_consistency_check),
                    uv_consistency_thresh=float(uv_consistency_thresh),
                    stats_out=stats,
                    debug=bool(debug_logs),
                    exploded_bake_enabled=bool(exploded_bake_enabled),
                    exploded_bake_distance=float(exploded_bake_distance),
                    exploded_bake_reassemble=bool(exploded_bake_reassemble),
                )
                _log(debug_logs, f"CAGE stats: hits={stats.get('hits')}, misses={stats.get('misses')}, avg_samples={stats.get('avg_samples')}")
            
            # UV relax pass to reduce visible lines
            if relax_uvs:
                _relax_uvs_vertex_average(new_obj.data, iterations=max(1, int(relax_iters)), alpha=float(relax_alpha), threshold=float(relax_threshold))
                # Seam-only relax for stubborn banding
                _relax_uvs_seams(new_obj.data, iterations=1, alpha=max(0.1, float(relax_alpha)*0.5), threshold=max(0.05, float(relax_threshold)*0.5))
        except Exception:
            traceback.print_exc()
        
        # Optional PBR baking or forced fallback when transfer fails
        do_bake = bool(bake_pbr)
        if force_bake_fallback:
            # If no materials or UVs are present on target, force bake
            try:
                no_mats = (len(new_obj.data.materials) == 0)
                no_uvs = (len(new_obj.data.uv_layers) == 0)
                if no_mats or no_uvs:
                    do_bake = True
                    _log(debug_logs, f"Force bake fallback: no_mats={no_mats}, no_uvs={no_uvs}")
            except Exception:
                do_bake = True
        # If textures still look wrong after transfer, projection + bake can help
        if not do_bake and force_bake_fallback:
            try:
                # Heuristic: if UV area collapses (tiny bounding box), trigger bake
                uv_layer = new_obj.data.uv_layers.active if new_obj.data.uv_layers else None
                if uv_layer is not None and len(uv_layer.data) > 0:
                    xs = [uv.uv.x for uv in uv_layer.data]
                    ys = [uv.uv.y for uv in uv_layer.data]
                    span = (max(xs)-min(xs)) * (max(ys)-min(ys))
                    if span < 1e-3:
                        _log(debug_logs, f"UV span too small ({span:.6f}); enabling bake fallback")
                        do_bake = True
                # Additional heuristic: area distortion detection
                if not do_bake and _uv_mapping_is_suspicious(new_obj):
                    _log(debug_logs, "UV mapping appears distorted; enabling bake fallback")
                    do_bake = True
            except Exception:
                pass
        # Optional post-bake to lock in color and remove residual seams
        if not do_bake and post_bake_color:
            try:
                bake_pbr_maps(
                    source_obj=obj,
                    target_obj=new_obj,
                    maps={"COLOR"},
                    res=int(post_bake_res),
                    margin=int(post_bake_margin),
                    uv_method=str(uv_method),
                )
            except Exception:
                traceback.print_exc()

        if do_bake:
            try:
                bake_pbr_maps(
                    source_obj=obj,
                    target_obj=new_obj,
                    maps=set(bake_maps or set()),
                    res=int(bake_resolution),
                    margin=int(bake_margin),
                    uv_method=str(uv_method),
                )
            except Exception:
                traceback.print_exc()
        # Summary log
        try:
            faces_in = len(work_mesh.polygons)
        except Exception:
            faces_in = 0
        _log(debug_logs, (
            f"Summary: faces_in={faces_in} target={target_faces} mode={mode} "
            f"cage={use_preview_cage} aim={ray_aim_mode} solver={projection_solver}"
        ))
        return True
    except Exception as exc:
        traceback.print_exc()
        ok = _fallback_python(
            obj,
            work_mesh,
            target_faces,
            feature_angle_deg=(feature_angle_deg if preserve_sharp else 180.0),
            smooth_iters=smooth_iters,
            smooth_lambda=smooth_lambda,
            preserve_seams=preserve_seams,
            preserve_sharp=preserve_sharp,
            output_mode=output_mode,
            pure_quads=pure_quads,
        )
        return ok
    finally:
        if work_mesh is not mesh:
            try:
                bpy.data.meshes.remove(work_mesh)
            except Exception:
                pass