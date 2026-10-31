# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Guided Retopo: importance points steer a live, adaptive quad preview you can partly redo by hand.

Points (Detail / Loop / Sharp / Calm) live on the source object. A density field built from them drives
either the whole mesh (adaptive triangle remesh -> TRIAD-Q PairClosure -> fit to surface) or only the union
of the point spheres (the region machinery in triadq/patch.py, seam kept). Faces locked on the preview stay
fixed; Redraw Strip rebuilds a selected strip; Next spot points at the worst area.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import overlay, region
from .triadq import areas, patch
from .triadq.engine import RemeshOptions, remesh_mesh
from .triadq.types import MeshData

PREVIEW_NAME = "CurioMesh_GuidedPreview"
POINT_TAG = "curiomesh_point"
DEBOUNCE_SECONDS = 0.35
AREA_DEBOUNCE_SECONDS = 0.12
POLL_SECONDS = 0.2
LIVE_QUAD_CAP = 2_500  # predicted quads above which the live preview waits for "Update"
HARD_QUAD_CAP = 60_000
LIVE_REGION_CAP = 3_000
LIVE_SOURCE_FACES = 120_000
LIVE_SLOW_MS = 4_000.0  # after an update this slow, wait for "Update" instead of freezing Blender on every edit

KIND_ITEMS = [
    ("DETAIL", "Detail", "More, smaller quads around the point (eyes, fingers, a face)", "SNAP_VERTEX", 0),
    ("LOOP", "Loop", "Edge loops ring around the point (eyes, mouth, elbows): the quads flow around it", "MESH_CIRCLE", 1),
    ("SHARP", "Sharp", "Keep hard edges near the point; the quads are smaller and the creases are not rounded off", "SHARPCURVE", 2),
    ("CALM", "Calm", "Fewer, bigger quads near the point (flat or boring areas). Also cuts a hole in the region of Points Only", "SMOOTHCURVE", 3),
]
KIND_KEYS = ("DETAIL", "LOOP", "SHARP", "CALM")
SCOPE_ITEMS = [
    ("WHOLE", "Whole Mesh", "Remesh the whole mesh; the points change the quad size across it", "MESH_DATA", 0),
    ("POINTS", "Points Only", "Remesh only inside the point spheres; the rest of the mesh stays exactly as it is", "SELECT_SET", 1),
    ("AREAS", "Work Areas", "Each point is its own area with its own density and fill style. They are remeshed one at a time in the background, so you can keep working", "BRUSH_DATA", 2),
]
APPLY_ITEMS = [
    ("NEW", "New Object", "Write the result to a new object called <name>_Retopo", "DUPLICATE", 0),
    ("REPLACE", "Replace Mesh", "Put the result into your object (Undo restores it)", "FILE_REFRESH", 1),
]
PRESET_ITEMS = [
    ("CHARACTER", "Character", "Organic bodies and faces: Detail points, flowing quads", "ARMATURE_DATA", 0),
    ("HARD_SURFACE", "Hard-Surface Prop", "Props with hard edges: Sharp points, grid-like quads", "MOD_BEVEL", 1),
    ("SCAN", "Scan / Photogrammetry", "Noisy scans: broad strokes, smoothed curvature, flowing quads", "OUTLINER_OB_POINTCLOUD", 2),
    ("GAME_LOD", "Game LOD", "Few quads, spent only where it matters", "MOD_DECIM", 3),
]
PRESETS = {
    "CHARACTER": dict(base_faces=3000, kind="DETAIL", strategy="FLOW", feature=40.0, radius=0.08, smooth=1),
    "HARD_SURFACE": dict(base_faces=1500, kind="SHARP", strategy="GRID", feature=25.0, radius=0.06, smooth=0),
    "SCAN": dict(base_faces=4000, kind="DETAIL", strategy="FLOW", feature=50.0, radius=0.10, smooth=3),
    "GAME_LOD": dict(base_faces=800, kind="DETAIL", strategy="FLOW", feature=40.0, radius=0.10, smooth=1),
}

state = {
    "source": "",
    "sig": None,
    "changed_at": 0.0,
    "dirty": False,
    "message": "",
    "ms": 0.0,
    "src": None,
    "res": None,
    "lock": None,
    "spot": None,
    "hover": None,
    "draw": None,
    "draw_key": None,
    "batches": None,
    "L0": 0.1,
    "predicted": 0,
    "work": None,
}


def status() -> str:
    return state["message"]


# --------------------------------------------------------------------------- points


@dataclass
class Pt:
    kind: str
    co: np.ndarray
    nrm: np.ndarray
    r: float
    s: float
    fill: str = "FLOW"


def _on_point_edit(self, context):
    overlay.tag_redraw()


class CURIOMESH_PG_guided_point(bpy.types.PropertyGroup):
    co: bpy.props.FloatVectorProperty(name="Location", description="Where the point sits on the surface (object space)", size=3, subtype="XYZ", update=_on_point_edit)
    normal: bpy.props.FloatVectorProperty(name="Normal", description="Surface direction at the point", size=3, default=(0.0, 0.0, 1.0))
    kind: bpy.props.EnumProperty(name="Type", description="What this point asks for", items=KIND_ITEMS, default="DETAIL", update=_on_point_edit)
    radius: bpy.props.FloatProperty(name="Radius", description="How far the point reaches (object-space units)", default=0.1, min=1e-4, soft_max=10.0, subtype="DISTANCE", update=_on_point_edit)
    strength: bpy.props.FloatProperty(name="Strength", description="How strongly the point changes the quad size: 1 = about three times smaller (or bigger for Calm) at the centre", default=1.0, min=0.1, max=4.0, update=_on_point_edit)
    fill: bpy.props.EnumProperty(name="Fill Style", description="Pattern of the new quads in this area (Work Areas only)", items=region.STRATEGY_ITEMS, default="FLOW", update=_on_point_edit)
    auto: bpy.props.BoolProperty(name="Suggested", description="Placed by Auto-suggest (replaced when you suggest again)", default=False)


def _preset_update(self, context):
    p = PRESETS[self.preset]
    self.base_faces = p["base_faces"]
    self.strategy = p["strategy"]
    self.new_kind = p["kind"]
    self.feature_angle = p["feature"]
    diag = state["src"]["diag"] if state["src"] else 0.0
    if diag > 0:
        self.new_radius = p["radius"] * diag
    overlay.tag_redraw()


def _enabled_update(self, context):
    if self.enabled:
        _start(context)
    else:
        _stop()


def _redraw_update(self, context):
    overlay.tag_redraw()


class CURIOMESH_PG_guided(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Guided Preview", description="Show the point cage and keep a live preview of the new quads next to your mesh", default=False, update=_enabled_update)
    preset: bpy.props.EnumProperty(name="Mesh Type", description="Starting density, point type and fill style for this kind of mesh", items=PRESET_ITEMS, default="CHARACTER", update=_preset_update)
    scope: bpy.props.EnumProperty(name="Remesh", description="Which part of the mesh is rebuilt", items=SCOPE_ITEMS, default="WHOLE")
    strategy: bpy.props.EnumProperty(name="Fill Style", description="Pattern of the new quads in a region: Flow follows the surface, Grid is a regular grid", items=region.STRATEGY_ITEMS, default="FLOW")
    base_faces: bpy.props.IntProperty(name="Base Quads", description="Roughly how many quads the whole mesh would get with no points. The points make some areas denser or coarser than this", default=3000, min=50, soft_max=60000)
    new_kind: bpy.props.EnumProperty(name="New Point Type", description="Type of the next point you place (keys 1-4 while placing)", items=KIND_ITEMS, default="DETAIL", update=_redraw_update)
    new_radius: bpy.props.FloatProperty(name="New Radius", description="Reach of the next point (scroll while placing). 0 picks a size from the mesh", default=0.0, min=0.0, soft_max=10.0, subtype="DISTANCE")
    new_strength: bpy.props.FloatProperty(name="New Strength", description="Strength of the next point (Shift + scroll while placing)", default=1.0, min=0.1, max=4.0)
    auto_update: bpy.props.BoolProperty(name="Auto-Update", description="Rebuild the preview shortly after you change a point or a setting", default=True)
    show_cage: bpy.props.BoolProperty(name="Show Cage", description="Draw each point's reach and the combined region outline", default=True, update=_redraw_update)
    apply_mode: bpy.props.EnumProperty(name="Result", description="Where the finished quads are written", items=APPLY_ITEMS, default="NEW")
    fit_iterations: bpy.props.IntProperty(name="Surface Fit Passes", description="Relax-and-project passes that settle the new quads onto your original surface", default=2, min=0, max=30)
    feature_angle: bpy.props.FloatProperty(name="Hard Edge Angle", description="Edges sharper than this (degrees) count as hard edges and are kept", default=40.0, min=5.0, max=170.0)
    pure_quads: bpy.props.BoolProperty(name="Pure Quads", description="Make every face of a Whole Mesh result a quad. Turn off for a faster preview of big meshes; a few triangles stay", default=True)
    suggest_count: bpy.props.IntProperty(name="Suggestions", description="How many points Auto-suggest places", default=8, min=1, max=40)


# --------------------------------------------------------------------------- density maths


def _fac(kind: str, s: float) -> float:
    """Edge-length multiplier at the centre of a point."""
    if kind == "DETAIL":
        return 1.0 / (1.0 + 2.0 * s)
    if kind == "CALM":
        return 1.0 + 2.0 * s
    if kind == "SHARP":
        return 1.0 / (1.0 + s)
    return float(np.sqrt(1.0 / (1.0 + 2.0 * s) / (1.0 + 0.5 * s)))  # LOOP, isotropic mean


def _weight(P: np.ndarray, p: Pt) -> np.ndarray:
    t = np.clip(np.linalg.norm(P - p.co, axis=1) / max(p.r, 1e-9), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)  # 1 at the centre, 0 at the radius (smoothstep)


def multiplier(P: np.ndarray, pts: list[Pt], E: np.ndarray | None = None) -> np.ndarray:
    """Target edge length / base edge length at points P. E: unit edge directions (Loop points go anisotropic)."""
    f = np.ones(len(P))
    for p in pts:
        w = _weight(P, p)
        if p.kind == "LOOP" and E is not None:
            fr = 1.0 + w * (1.0 / (1.0 + 2.0 * p.s) - 1.0)  # across the ring: small steps = many rings
            ft = 1.0 + w * (1.0 / (1.0 + 0.5 * p.s) - 1.0)  # along the ring
            d = P - p.co
            d = d - np.outer(d @ p.nrm, p.nrm)
            d /= np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
            c2 = np.einsum("ij,ij->i", d, E) ** 2
            f *= 1.0 / np.sqrt(c2 / fr**2 + (1.0 - c2) / ft**2)
        else:
            f *= 1.0 + w * (_fac(p.kind, p.s) - 1.0)
    return np.clip(f, 0.125, 8.0)


def region_sd(P: np.ndarray, pts: list[Pt], scope: str) -> np.ndarray:
    """Signed distance of the point-sphere region: union of adding spheres minus the subtracting ones (<0 inside)."""
    add = {"POINTS": ("DETAIL", "LOOP"), "AREAS": KIND_KEYS}.get(scope, ("DETAIL", "LOOP", "SHARP"))
    sub = {"POINTS": ("CALM", "SHARP"), "AREAS": ()}.get(scope, ("CALM",))

    def union(kinds):
        sd = np.full(len(P), np.inf)
        for p in pts:
            if p.kind in kinds:
                sd = np.minimum(sd, np.linalg.norm(P - p.co, axis=1) - p.r)
        return sd

    return np.maximum(union(add), -union(sub))


def stepped_intensity(kind: str, strength: float, up: bool, step: float = 0.25) -> tuple[str, float]:
    """One scroll notch: up is always denser. Detail fades into Calm and back through the base density."""
    was = -strength if kind == "CALM" else strength
    now = was + (step if up else -step)
    if abs(now) < 0.1 or now * was < 0:
        now = 0.1 if up else -0.1
    now = float(np.clip(now, -4.0, 4.0))
    return "CALM" if now < 0 else ("DETAIL" if kind == "CALM" else kind), round(abs(now), 3)


# --------------------------------------------------------------------------- mesh helpers


def _edge_table(tris: np.ndarray):
    """Unique edges of a triangle array: (lo, hi, face0, face1 or -1, count)."""
    n = int(tris.max()) + 1 if len(tris) else 1
    a = tris.ravel()
    b = tris[:, [1, 2, 0]].ravel()
    f = np.repeat(np.arange(len(tris)), 3)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * n + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    count = np.diff(np.r_[start, len(ks)])
    f1 = np.where(count > 1, f[order][np.minimum(start + 1, len(ks) - 1)], -1)
    return lo[order][start], hi[order][start], f[order][start], f1, count


def _tri_normals(co: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]


def _vert_normals(co: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.cross(co[tris[:, 1]] - co[tris[:, 0]], co[tris[:, 2]] - co[tris[:, 0]])  # area weighted
    out = np.zeros_like(co)
    for k in range(3):
        for axis in range(3):
            out[:, axis] += np.bincount(tris[:, k], n[:, axis], len(co))
    return out / np.maximum(np.linalg.norm(out, axis=1), 1e-30)[:, None]


def _creases(co: np.ndarray, tris: np.ndarray, degrees: float):
    """(edge table, crease edge mask, crease vertex mask, boundary vertex mask)."""
    lo, hi, f0, f1, count = _edge_table(tris)
    nrm = _tri_normals(co, tris)
    inner = f1 >= 0
    cos = np.ones(len(lo))
    cos[inner] = np.einsum("ij,ij->i", nrm[f0[inner]], nrm[f1[inner]])
    crease = inner & (cos < np.cos(np.radians(degrees)))
    cv = np.zeros(len(co), dtype=bool)
    cv[lo[crease]] = cv[hi[crease]] = True
    bv = np.zeros(len(co), dtype=bool)
    bv[lo[~inner]] = bv[hi[~inner]] = True
    return (lo, hi, f0, f1, count), crease, cv, bv


def _fan(faces) -> np.ndarray:
    return np.asarray([(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)], dtype=np.int64).reshape(-1, 3)


def _source_data(obj) -> dict:
    """Welded copy of the source (glTF/FBX imports split vertices at seams), cached per geometry."""
    mesh = obj.data
    nv = len(mesh.vertices)
    co = np.empty(nv * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    sig = (obj.name, nv, len(mesh.polygons), round(float(co.sum()), 5), mesh.as_pointer())
    cached = state["src"]
    if cached is not None and cached["sig"] == sig:
        return cached
    _co, faces, starts, _sizes = region._mesh_arrays(obj)
    mat = np.zeros(len(faces), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", mat)
    diag = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0))) or 1.0
    key = np.round(co / (diag * 1e-6)).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    wco = co[first]
    keep, wfaces, wloops = [], [], []
    for i, f in enumerate(faces):
        first: dict[int, int] = {}
        for corner, v in enumerate(f):
            first.setdefault(int(inv[v]), int(starts[i]) + corner)
        if len(first) >= 3:
            wfaces.append(tuple(first))
            wloops.append(tuple(first.values()))
            keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    tris = _fan(wfaces)
    area = 0.5 * np.linalg.norm(np.cross(wco[tris[:, 1]] - wco[tris[:, 0]], wco[tris[:, 2]] - wco[tris[:, 0]]), axis=1)
    data = {
        "sig": sig,
        "diag": diag,
        "co": wco,
        "faces": wfaces,
        "tris": tris,
        "tri_area": area,
        "tri_cent": wco[tris].mean(axis=1),
        "tri_face": np.repeat(np.arange(len(wfaces)), [len(f) - 2 for f in wfaces]),
        "tri_loops": _fan(wloops),
        "loops": wloops,
        "mat": mat[keep],
        "bvh": None,
    }
    state["src"] = data
    return data


def _bvh(src: dict) -> BVHTree:
    if src["bvh"] is None:
        src["bvh"] = BVHTree.FromPolygons(src["co"].tolist(), src["tris"].tolist())
    return src["bvh"]


def _centroids(co: np.ndarray, faces) -> np.ndarray:
    return np.asarray([co[list(f)].mean(axis=0) for f in faces]).reshape(-1, 3)


def _nearest_tris(bvh: BVHTree, points: np.ndarray) -> np.ndarray:
    return np.asarray([bvh.find_nearest(p)[2] for p in points.tolist()], dtype=np.int64)


def _barycentric(co: np.ndarray, tris: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of each point's projection on its own triangle, kept inside the triangle."""
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    v0, v1, v2 = b - a, c - a, points - a
    d00, d01, d11 = (v0 * v0).sum(axis=1), (v0 * v1).sum(axis=1), (v1 * v1).sum(axis=1)
    d20, d21 = (v2 * v0).sum(axis=1), (v2 * v1).sum(axis=1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-30, np.inf, denom)  # degenerate triangle: use its first corner
    v, w = (d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom
    bary = np.clip(np.stack((1.0 - v - w, v, w), axis=1), 0.0, 1.0)
    return bary / bary.sum(axis=1, keepdims=True)


def _uv_islands(src: dict, uv: np.ndarray) -> np.ndarray:
    """UV island of each source face: faces that share an edge with the same UVs on both sides belong together."""
    rows, owner = [], []
    for i, (face, loops) in enumerate(zip(src["faces"], src["loops"])):
        for k in range(len(face)):
            n = (k + 1) % len(face)
            a, b = (k, n) if face[k] < face[n] else (n, k)
            rows.append((face[a], face[b], *uv[loops[a]], *uv[loops[b]]))
            owner.append(i)
    rows = np.asarray(rows)
    rows[:, 2:] = np.round(rows[:, 2:], 5)
    _u, inv = np.unique(rows, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    owner = np.asarray(owner, dtype=np.int64)
    order = np.argsort(inv, kind="stable")
    twin = inv[order][1:] == inv[order][:-1]
    return patch._labels(len(src["faces"]), owner[order][:-1][twin], owner[order][1:][twin])


def _transfer_uvs(mesh, source, src, res) -> None:
    """Copy every UV layer: each corner reads the source at its nearest point on the UV island of its own face."""
    co, faces = res["co"], res["faces"]
    flat = np.fromiter((v for f in faces for v in f), dtype=np.int64)
    bvh = _bvh(src)
    corner_tri = _nearest_tris(bvh, co)[flat]
    owner_tri = np.repeat(_nearest_tris(bvh, _centroids(co, faces)), [len(f) for f in faces])
    points = co[flat]
    for layer in source.data.uv_layers:
        uv = np.empty(len(source.data.loops) * 2, dtype=np.float64)
        layer.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        island = _uv_islands(src, uv)[src["tri_face"]]
        tri = corner_tri.copy()
        off = np.flatnonzero(island[corner_tri] != island[owner_tri])  # a corner on a UV seam: stay on this face's side
        for chart in np.unique(island[owner_tri[off]]).tolist():
            members = np.flatnonzero(island == chart)
            used, local = np.unique(src["tris"][members], return_inverse=True)
            chart_bvh = BVHTree.FromPolygons(src["co"][used].tolist(), local.reshape(-1, 3).tolist())
            here = off[island[owner_tri[off]] == chart]
            tri[here] = members[_nearest_tris(chart_bvh, points[here])]
        out = np.einsum("ijk,ij->ik", uv[src["tri_loops"][tri]], _barycentric(src["co"], src["tris"][tri], points))
        target = mesh.uv_layers.new(name=layer.name)
        target.data.foreach_set("uv", out.astype(np.float32).ravel())
    active = source.data.uv_layers.active_index
    if 0 <= active < len(mesh.uv_layers):
        mesh.uv_layers.active_index = active


# --------------------------------------------------------------------------- whole-mesh path


def _read(me):
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    tris = np.empty(len(me.polygons) * 3, dtype=np.int64)
    me.polygons.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    return co.reshape(-1, 3), tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]


def _write_co(me, co):
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()


def _bm_edges(bm, lo, hi, idx):
    bm.verts.ensure_lookup_table()
    return [e for e in (bm.edges.get((bm.verts[int(lo[i])], bm.verts[int(hi[i])])) for i in idx) if e is not None]


def _targets(co, lo, hi, pts, L0):
    d = co[hi] - co[lo]
    length = np.linalg.norm(d, axis=1)
    return length, L0 * multiplier(0.5 * (co[lo] + co[hi]), pts, d / np.maximum(length, 1e-30)[:, None])


def _split_pass(me, pts, L0) -> int:
    co, tris = _read(me)
    lo, hi, _f0, _f1, _c = _edge_table(tris)
    length, target = _targets(co, lo, hi, pts, L0)
    idx = np.flatnonzero(length > 4.0 / 3.0 * target)
    if not len(idx):
        return 0
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        edges = _bm_edges(bm, lo, hi, idx)
        bmesh.ops.subdivide_edges(bm, edges=edges, cuts=1, use_grid_fill=False)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="SHORT_EDGE")
        bm.to_mesh(me)
    finally:
        bm.free()
    return len(idx)


def _neighbours(count, lo, hi):
    a = np.concatenate((lo, hi))
    b = np.concatenate((hi, lo))
    order = np.argsort(a, kind="stable")
    start = np.searchsorted(a[order], np.arange(count + 1))
    return start, b[order]


def _collapse_pass(me, pts, L0, crease_deg) -> int:
    co, tris = _read(me)
    (lo, hi, _f0, _f1, count), crease, cv, bv = _creases(co, tris, crease_deg)
    length, target = _targets(co, lo, hi, pts, L0)
    ok = (length < 0.8 * target) & (count == 2) & ~bv[lo] & ~bv[hi] & (~(cv[lo] | cv[hi]) | (crease & cv[lo] & cv[hi]))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return 0
    idx = idx[np.argsort(length[idx])]
    start, nbr = _neighbours(len(co), lo, hi)
    locked = np.zeros(len(co), dtype=bool)
    chosen = []
    for i in idx.tolist():
        a, b = int(lo[i]), int(hi[i])
        if locked[a] or locked[b]:
            continue
        na, nb = nbr[start[a] : start[a + 1]], nbr[start[b] : start[b + 1]]
        if len(set(na.tolist()) & set(nb.tolist())) != 2 or len(na) + len(nb) - 4 < 4:
            continue
        ring = np.union1d(na, nb)
        mid = 0.5 * (co[a] + co[b])
        far = np.linalg.norm(co[ring[(ring != a) & (ring != b)]] - mid, axis=1).max()
        if far > 4.0 / 3.0 * target[i]:
            continue
        locked[ring] = locked[a] = locked[b] = True
        chosen.append(i)
    if not chosen:
        return 0
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.collapse(bm, edges=_bm_edges(bm, lo, hi, chosen))
        bm.to_mesh(me)
    finally:
        bm.free()
    return len(chosen)


def _flip_pass(me, crease_deg) -> int:
    co, tris = _read(me)
    (lo, hi, f0, f1, count), crease, _cv, _bv = _creases(co, tris, crease_deg)
    val = np.bincount(np.concatenate((lo, hi)), minlength=len(co))
    inner = (count == 2) & ~crease
    g0, g1 = np.where(inner, f0, 0), np.where(inner, f1, 0)
    n = len(co)
    c = np.where(inner, np.clip(tris[g0].sum(axis=1) - lo - hi, 0, n - 1), 0)
    d = np.where(inner, np.clip(tris[g1].sum(axis=1) - lo - hi, 0, n - 1), 0)
    dev = np.abs
    before = dev(val[lo] - 6) + dev(val[hi] - 6) + dev(val[c] - 6) + dev(val[d] - 6)
    after = dev(val[lo] - 7) + dev(val[hi] - 7) + dev(val[c] - 5) + dev(val[d] - 5)
    new_len = np.linalg.norm(co[c] - co[d], axis=1)
    old_len = np.linalg.norm(co[hi] - co[lo], axis=1)
    nrm = _tri_normals(co, tris)
    flat = np.einsum("ij,ij->i", nrm[g0], nrm[g1]) > 0.85
    good = inner & flat & (after < before) & (new_len < 1.5 * old_len) & (c != d) & (val[lo] > 3) & (val[hi] > 3)
    idx = np.flatnonzero(good)
    if not len(idx):
        return 0
    used = np.zeros(len(tris), dtype=bool)
    existing = set(zip(lo.tolist(), hi.tolist()))
    chosen = []
    for i in idx[np.argsort(before[idx] - after[idx])[::-1]].tolist():
        if used[f0[i]] or used[f1[i]] or (min(c[i], d[i]), max(c[i], d[i])) in existing:
            continue
        used[f0[i]] = used[f1[i]] = True
        existing.add((min(c[i], d[i]), max(c[i], d[i])))
        chosen.append(i)
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        for e in _bm_edges(bm, lo, hi, chosen):
            bmesh.utils.edge_rotate(e, True)
        bm.to_mesh(me)
    finally:
        bm.free()
    return len(chosen)


def _smooth_pass(me, project, crease_deg, strength=0.5) -> None:
    co, tris = _read(me)
    (lo, hi, _f0, _f1, _c), _crease, cv, bv = _creases(co, tris, crease_deg)
    nrm = _vert_normals(co, tris)
    total = np.zeros_like(co)
    for axis in range(3):
        total[:, axis] = np.bincount(lo, co[hi, axis], len(co)) + np.bincount(hi, co[lo, axis], len(co))
    deg = np.bincount(np.concatenate((lo, hi)), minlength=len(co)).astype(np.float64)
    delta = total / np.maximum(deg, 1)[:, None] - co
    delta -= nrm * np.einsum("ij,ij->i", delta, nrm)[:, None]
    free = ~bv & ~cv & (deg > 0)
    moved = co.copy()
    moved[free] += strength * delta[free]
    if project is not None:
        moved[free] = project(moved[free])
    _write_co(me, moved)


def adaptive_triangles(co, tris, pts, L0, project, crease_deg, max_iter=8):
    """Isotropic remeshing (split / collapse / flip / smooth) towards the per-edge target length."""
    me = bpy.data.meshes.new("CM_guided_tmp")
    try:
        me.from_pydata(co.tolist(), [], tris.tolist())
        me.update()
        for it in range(max_iter):
            changed = _split_pass(me, pts, L0) + _collapse_pass(me, pts, L0, crease_deg)
            if it >= 1:
                _flip_pass(me, crease_deg)
            _smooth_pass(me, project, crease_deg)
            if changed < 0.01 * max(len(me.vertices), 1) and it >= 2:
                break
        _flip_pass(me, crease_deg)
        _smooth_pass(me, project, crease_deg)
        out_co, out_tris = _read(me)
    finally:
        bpy.data.meshes.remove(me)
    used = np.zeros(len(out_co), dtype=bool)
    used[out_tris.ravel()] = True
    remap = np.cumsum(used) - 1
    return out_co[used], remap[out_tris]


def _proxy(context, obj, src, max_tris):
    """Source triangles, decimated (native, fast) when there are many more than the result needs."""
    tris = src["tris"]
    if len(tris) <= max_tris:
        return src["co"], tris
    welded = bpy.data.meshes.new("CM_guided_proxy")  # the welded copy: seams split by an importer would tear apart
    welded.from_pydata(src["co"].tolist(), [], tris.tolist())
    welded.update()
    tmp = bpy.data.objects.new("CM_guided_proxy", welded)
    context.scene.collection.objects.link(tmp)
    try:
        mod = tmp.modifiers.new("d", "DECIMATE")
        mod.ratio = max(0.01, max_tris / max(len(tris), 1))
        evaluated = tmp.evaluated_get(context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", co)
            mesh.calc_loop_triangles()
            t = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
            mesh.loop_triangles.foreach_get("vertices", t)
        finally:
            evaluated.to_mesh_clear()
    finally:
        data = tmp.data
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.meshes.remove(data)
    co = co.reshape(-1, 3)
    t = t.reshape(-1, 3)
    key = np.round(co / (src["diag"] * 1e-6)).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    t = inv.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return co[first], t


def _whole_mesh(context, obj, src, pts, gs, L0, project):
    qe = float(np.sum(src["tri_area"] / (L0 * multiplier(src["tri_cent"], pts)) ** 2))
    state["predicted"] = int(qe)
    pco, ptris = _proxy(context, obj, src, int(np.clip(8 * qe, 3000, 60000)))
    crease = min(gs.feature_angle, 30.0) if any(p.kind == "SHARP" for p in pts) else gs.feature_angle
    ico, itris = adaptive_triangles(pco, ptris, pts, L0, project, crease)
    mesh = MeshData(vertices=ico, faces=[tuple(t) for t in itris.tolist()], name="guided")
    open_before = int((_edge_table(itris)[4] == 1).sum())
    note = ""
    for angle in dict.fromkeys((crease, max(crease, 45.0))):  # a low angle can make the engine leave holes
        options = RemeshOptions(target_faces=max(1, len(itris) // 2), mode="AUTO", seed_count=3, feature_angle_deg=angle, force_quads=bool(gs.pure_quads))
        out, report = remesh_mesh(mesh, options)
        if report.success and out.faces and int((_edge_table(_fan(out.faces))[4] == 1).sum()) <= open_before:
            note = ""
            break
        note = "PairClosure left open edges" if report.success else f"PairClosure failed: {report.message}"
    if not out.faces:
        return None, note
    co = out.vertices
    faces = [tuple(f) for f in out.faces]
    if project is not None and gs.fit_iterations > 0:
        _t, _c, cv, bv = _creases(co, _fan(faces), gs.feature_angle)
        co = patch.fit_to_surface(co, faces, cv | bv, project, gs.fit_iterations, 0.5)
    mat = src["mat"][src["tri_face"][_nearest_tris(_bvh(src), _centroids(co, faces))]]
    return {"co": co, "faces": faces, "new": np.ones(len(faces), bool), "locked": np.zeros(len(faces), bool), "mat": mat}, note


# --------------------------------------------------------------------------- region path


def _region_pass(base_co, base_faces, mask, pts, L0, gs, project, base_locked, base_mat, fixed_target=0.0):
    """Remesh the faces in `mask` (seam kept). Returns (result dict or None, warnings)."""
    comps, warns = patch.prepare_regions(base_co, base_faces, mask, min_island=4, max_hole=4, smooth=2, split_rings=True, frozen=base_locked)
    results = []
    for comp in comps:
        cent = _centroids(base_co, [base_faces[i] for i in comp])
        target = fixed_target or L0 * float(multiplier(cent, pts).mean())
        res = patch.remesh_region(base_co, base_faces, comp, target_length=target, strategy=gs.strategy, falloff=2.5 * target, max_faces=region.MAX_PATCH_FACES)
        if res.ok and project is not None and gs.fit_iterations > 0:
            region._fit_result(base_co, res, project, gs.fit_iterations)
        if not res.ok:
            warns.append(f"Region ({len(comp)} faces) skipped: {res.message}")
        results.append(res)
    return _merge_results(base_co, base_faces, results, base_locked, base_mat), warns


def _merge_results(base_co, base_faces, results, base_locked, base_mat):
    """The base mesh with every finished patch put in. None when no patch worked."""
    good = [r for r in results if r.ok]
    if not good:
        return None
    co, faces, new = region._merged_arrays({"co": base_co, "faces": base_faces, "results": good})
    removed = np.zeros(len(base_faces), dtype=bool)
    for r in good:
        removed[r.removed_faces] = True
    kept = ~removed
    locked = np.concatenate((base_locked[kept], np.zeros(int(new.sum()), bool)))
    mat = None
    if base_mat is not None:
        extra = [base_mat[s] if 0 <= s < len(base_mat) else 0 for r in good for s in (r.face_source or [-1] * len(r.faces))]
        mat = np.concatenate((base_mat[kept], np.asarray(extra, dtype=np.int32)))
    return {"co": co, "faces": faces, "new": new, "locked": locked, "mat": mat, "removed": removed}


# --------------------------------------------------------------------------- gather / suggest / spot


def gather(obj, gs=None) -> list[Pt]:
    """Points on the object plus tagged empties."""
    pts = [Pt(p.kind, np.asarray(p.co, float), np.asarray(p.normal, float), float(p.radius), float(p.strength), p.fill) for p in obj.curiomesh_points]
    inv = obj.matrix_world.inverted()
    scale = max(float(np.mean(obj.matrix_world.to_scale())), 1e-9)
    default = gs.new_kind if gs is not None else "DETAIL"
    for o in bpy.data.objects:
        if o.type != "EMPTY" or o.parent not in (None, obj) or not (o.get(POINT_TAG) is not None or o.name.startswith("CM_Point")):
            continue
        local = inv @ o.matrix_world.translation
        kind = str(o.get("curiomesh_kind", default)).upper()
        ok, _loc, normal, _i = obj.closest_point_on_mesh(local)
        pts.append(
            Pt(
                kind if kind in KIND_KEYS else default,
                np.asarray(local, float),
                np.asarray(normal if ok else (0, 0, 1), float),
                max(float(o.empty_display_size) * float(np.mean(o.matrix_world.to_scale())) / scale, 1e-4),
                float(o.get("curiomesh_strength", 1.0)),
                str(o.get("curiomesh_fill", "FLOW")),
            )
        )
    return pts


def add_point(obj, co, normal, kind, radius, strength, auto=False, fill="FLOW"):
    p = obj.curiomesh_points.add()
    p.co, p.normal, p.kind, p.radius, p.strength, p.auto, p.fill = tuple(co), tuple(normal), kind, radius, strength, auto, fill
    obj.curiomesh_point_index = len(obj.curiomesh_points) - 1
    return p


def suggest(obj, gs) -> list[dict]:
    """High-curvature spots, hard-edge spots and small holes, as point descriptions (nothing is written)."""
    src = _source_data(obj)
    co, tris, diag = src["co"], src["tris"], src["diag"]
    preset = PRESETS[gs.preset]
    (lo, hi, _f0, _f1, count), _crease, cv, bv = _creases(co, tris, gs.feature_angle)
    vn = _vert_normals(co, tris)
    length = np.linalg.norm(co[hi] - co[lo], axis=1)
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", vn[lo], vn[hi]), -1, 1))
    k = ang / np.maximum(length, 1e-12)
    deg = np.maximum(np.bincount(lo, minlength=len(co)) + np.bincount(hi, minlength=len(co)), 1)
    curv = (np.bincount(lo, k, len(co)) + np.bincount(hi, k, len(co))) / deg
    for _ in range(preset["smooth"]):
        curv = 0.5 * curv + 0.5 * (np.bincount(lo, curv[hi], len(co)) + np.bincount(hi, curv[lo], len(co))) / deg
    curv = curv * (~bv)
    top = float(np.percentile(curv[curv > 0], 97)) if (curv > 0).any() else 1.0
    score = curv / max(top, 1e-12) + cv * 0.8
    radius = gs.new_radius or preset["radius"] * diag
    out = []
    be = np.flatnonzero(count == 1)  # small holes (eyes, mouths) become Loop points
    if len(be):
        parent: dict[int, int] = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in be.tolist():
            parent[find(int(lo[i]))] = find(int(hi[i]))
        groups: dict[int, list[int]] = {}
        for v in {int(x) for x in np.concatenate((lo[be], hi[be]))}:
            groups.setdefault(find(v), []).append(v)
        for verts in groups.values():
            vp = co[verts]
            perimeter = float(length[be][np.isin(lo[be], verts)].sum())
            if len(verts) >= 4 and perimeter < 0.25 * diag:
                c = vp.mean(axis=0)
                out.append({"kind": "LOOP", "co": c, "r": max(1.8 * float(np.linalg.norm(vp - c, axis=1).max()), 0.02 * diag)})
    taken = [np.asarray(o["co"]) for o in out]
    for v in np.argsort(-score).tolist():
        if len(out) >= gs.suggest_count or score[v] < 0.35:
            break
        if all(np.linalg.norm(co[v] - t) > radius for t in taken):
            kind = "SHARP" if cv[v] and preset["kind"] == "SHARP" else "DETAIL"
            out.append({"kind": kind, "co": co[v], "r": radius})
            taken.append(co[v])
    for o in out:
        ok, loc, normal, _i = obj.closest_point_on_mesh(Vector(o["co"]))
        o["co"] = np.asarray(loc if ok else o["co"], float)
        o["normal"] = np.asarray(normal if ok else (0, 0, 1), float)
    return out[: gs.suggest_count]


def next_spot(res, project, L0):
    """The worst area of a result: leftover triangles, n-gons, poles and surface error. (position, text) or None."""
    co, faces = res["co"], res["faces"]
    size = np.asarray([len(f) for f in faces])
    flat = np.concatenate([np.asarray(f) for f in faces])
    owner = np.repeat(np.arange(len(faces)), size)
    valence = np.bincount(flat, minlength=len(co))
    _t, _c, _cv, bv = _creases(co, _fan(faces), 180.0)
    pole = (valence != 4) & ~bv
    pole_w = np.where(valence >= 6, 1.5, 1.0) * pole
    score = np.where(size == 3, 2.0, 0.0) + np.where(size > 4, 3.0, 0.0) + np.bincount(owner, pole_w[flat], len(faces)) / np.maximum(size, 1) * 2.0
    cent = _centroids(co, faces)
    if project is not None:
        score = score + np.clip(np.linalg.norm(project(cent) - cent, axis=1) / max(0.25 * L0, 1e-9), 0.0, 3.0)
    vs = np.bincount(flat, score[owner], len(co))
    smooth = np.bincount(owner, vs[flat], len(faces)) / np.maximum(size, 1)
    i = int(np.argmax(smooth))
    if smooth[i] < 1.5:
        return None
    near = np.linalg.norm(cent - cent[i], axis=1) < 3 * L0
    tris = int(((size == 3) & near).sum())
    poles = int(pole[np.unique(flat[np.isin(owner, np.flatnonzero(near))])].sum())
    return cent[i], f"{tris} triangles and {poles} poles around here"


# --------------------------------------------------------------------------- update


def state_source():
    obj = bpy.data.objects.get(state["source"])
    return obj if obj is not None and obj.type == "MESH" else None


def _source_mesh(context):
    o = context.object
    if o is not None and o.type == "MESH" and o.name != PREVIEW_NAME:
        return o
    return state_source()


def _signature(context, src_obj) -> tuple:
    gs = context.scene.curiomesh_guided
    parts = [
        src_obj.name,
        len(src_obj.data.vertices),
        len(src_obj.data.polygons),
        gs.scope,
        gs.strategy,
        gs.base_faces,
        gs.fit_iterations,
        gs.pure_quads,
        round(gs.feature_angle, 2),
        tuple(round(x, 5) for row in src_obj.matrix_world for x in row),
        state["lock"]["id"] if state["lock"] else 0,
    ]
    for p in src_obj.curiomesh_points:
        parts.append((p.kind, p.fill, round(p.radius, 5), round(p.strength, 4), tuple(round(x, 5) for x in p.co)))
    for o in bpy.data.objects:
        if o.type == "EMPTY" and (o.get(POINT_TAG) is not None or o.name.startswith("CM_Point")):
            parts.append((o.name, o.parent, round(o.empty_display_size, 4), tuple(round(x, 4) for x in o.matrix_world.translation), str(o.get("curiomesh_kind", "")), o.get("curiomesh_strength", 1.0)))
    return tuple(parts)


def remove_preview() -> None:
    obj = bpy.data.objects.get(PREVIEW_NAME)
    if obj is not None:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    state.update(res=None, spot=None, draw=None, draw_key=None, batches=None)


def _make_mesh(name, source, res, uvs=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(res["co"].tolist(), [], res["faces"])
    mesh.update()
    for m in source.data.materials:
        mesh.materials.append(m)
    if res.get("mat") is not None and len(mesh.polygons) == len(res["mat"]):
        mesh.polygons.foreach_set("material_index", np.asarray(res["mat"], dtype=np.int32))
    if uvs:
        _transfer_uvs(mesh, source, _source_data(source), res)
    return mesh


def _build_preview(context, source, res, uvs=True) -> None:
    obj = bpy.data.objects.get(PREVIEW_NAME)
    mesh = _make_mesh(PREVIEW_NAME + "_Mesh", source, res, uvs)
    if obj is None:
        obj = bpy.data.objects.new(PREVIEW_NAME, mesh)
        (source.users_collection[0] if source.users_collection else context.scene.collection).objects.link(obj)
    else:
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
    matrix = source.matrix_world.copy()
    matrix.translation.x += max(source.dimensions.x, 0.1) * 1.25
    obj.matrix_world = matrix
    obj.show_wire = True
    obj.show_all_edges = True
    obj.hide_select = False


# --------------------------------------------------------------------------- work areas


def stop_work() -> None:
    if state["work"] is not None:
        state["work"]["runner"].stop()
        state["work"] = None


def _area_target(p: Pt, L0: float) -> float:
    return L0 * _fac(p.kind, p.s)


def _find_spot(res, project, L0) -> None:
    if res is not None:
        try:
            spot = next_spot(res, project, L0)
            res["spot"] = {"pos": spot[0], "text": spot[1]} if spot else None
        except Exception:  # the hint must never break an update
            res["spot"] = None


def _start_areas(source, gs, pts, src, L0) -> bool:
    """Hand the areas to the worker. Blender stays free; _collect_work picks up each finished piece."""
    lock = state["lock"]
    co, faces = (lock["co"], lock["faces"]) if lock else (src["co"], src["faces"])
    frozen = lock["locked"] if lock else np.zeros(len(faces), dtype=bool)
    mat = lock.get("mat") if lock else src["mat"]
    key = (src["sig"], lock["id"] if lock else 0, gs.fit_iterations, L0)
    if state["work"] is None or state["work"]["key"] != key:
        stop_work()
        project = region._projector(_bvh(src))
        iterations = gs.fit_iterations
        runner = areas.AreaRunner(
            co, faces, frozen,
            fit=(lambda r: region._fit_result(co, r, project, iterations)) if iterations > 0 else None,
            merge=lambda results: _merge_results(co, faces, results, frozen, mat),
            finish=lambda res: _find_spot(res, project, L0),
            max_faces=region.MAX_PATCH_FACES,
        )
        state["work"] = {"key": key, "runner": runner, "cent": _centroids(co, faces), "seen": 0, "areas": []}
    work = state["work"]
    items = [areas.Area(p.co, p.r, _area_target(p, L0), p.fill) for p in pts]
    groups = areas.group_areas(items)
    jobs = [areas.Job(m & ~frozen, items[g[0]].target, items[g[0]].strategy, g) for g, m in zip(groups, areas.area_masks(work["cent"], items, groups)) if (m & ~frozen).any()]
    active = source.curiomesh_point_index
    jobs.sort(key=lambda j: (active not in j.tag, -j.tag[-1]))  # the area you just touched first, then newest first
    work["areas"] = pts
    state.update(L0=L0, ms=0.0, spot=None)
    state["message"] = "Working..."
    work["runner"].start(jobs)
    return True


def _collect_work(context) -> None:
    """Show what the worker finished since the last look."""
    work = state["work"]
    source = state_source()
    if work is None or source is None:
        return
    runner = work["runner"]
    done, total = runner.progress
    if runner.output_id != work["seen"]:
        work["seen"] = runner.output_id
        if runner.output is None:
            remove_preview()
        else:
            state["res"] = runner.output
            _build_preview(context, source, runner.output, uvs=False)  # the UVs are copied when you Apply
        state["draw_key"] = None
    res = state["res"]
    made = f"{int(res['new'].sum()):,} new quads in {len(res['faces']):,} faces" if res is not None else "No quads yet"
    if runner.busy:
        state["message"] = f"Working on piece {min(done + 1, max(total, 1))} of {max(total, 1)}. {made}"
    elif not work.get("settled") == runner.output_id:
        work["settled"] = runner.output_id
        state["message"] = made + (f" | {runner.warnings[0]}" if runner.warnings else "")
        state["spot"] = res.get("spot") if res is not None else None
    overlay.tag_redraw()


def finish_work(context) -> None:
    """Wait for the worker, for whoever needs the finished result now (Apply)."""
    if state["work"] is not None:
        state["work"]["runner"].wait()
        _collect_work(context)


def run_update(context, force: bool = False) -> bool:
    """Recompute the preview now. Returns True when a preview was produced."""
    gs = context.scene.curiomesh_guided
    source = _source_mesh(context)
    if source is None:
        state["message"] = "Select a mesh first"
        return False
    state["source"] = source.name
    if source.mode == "EDIT":
        source.update_from_editmode()
    pts = gather(source, gs)
    lock = state["lock"]
    if not pts and lock is None:
        stop_work()
        remove_preview()
        state["message"] = "Place a point or press Auto-suggest"
        return False
    started = time.perf_counter()
    src = _source_data(source)
    L0 = float(np.sqrt(src["tri_area"].sum() / max(gs.base_faces, 1)))
    if gs.scope == "AREAS":
        return _start_areas(source, gs, pts, src, L0)
    stop_work()
    project = region._projector(_bvh(src))
    notes = ""
    res = None
    if lock is not None:
        mask = ~lock["locked"]
        if gs.scope == "POINTS" and not lock["free_all"]:
            mask &= region_sd(_centroids(lock["co"], lock["faces"]), pts, "POINTS") < 0
        if not mask.any():
            state["message"] = "Nothing left to remesh: all faces here are locked"
            return False
        res, warns = _region_pass(lock["co"], lock["faces"], mask, pts, L0, gs, project, lock["locked"], lock.get("mat"))
        notes = warns[0] if warns else ""
    else:
        if gs.scope == "POINTS":
            mask = region_sd(_centroids(src["co"], src["faces"]), pts, "POINTS") < 0
            if not mask.any():
                state["message"] = "Points Only needs a Detail or Loop point on the mesh"
                return False
            if not force and int(mask.sum()) > LIVE_REGION_CAP:
                state["message"] = f"Region has {int(mask.sum()):,} faces: press Update"
                return False
            if mask.mean() > 0.92:
                notes = "Region covers the mesh: used Whole Mesh"
            else:
                res, warns = _region_pass(src["co"], src["faces"], mask, pts, L0, gs, project, np.zeros(len(src["faces"]), bool), src["mat"])
                notes = warns[0] if warns else ""
        if res is None:
            qe = float(np.sum(src["tri_area"] / (L0 * multiplier(src["tri_cent"], pts)) ** 2))
            if qe > HARD_QUAD_CAP:
                state["message"] = f"Would make {int(qe):,} quads: lower Base Quads or the Detail strength"
                return False
            if not force and (qe > LIVE_QUAD_CAP or len(src["faces"]) > LIVE_SOURCE_FACES):
                state["message"] = f"About {int(qe):,} quads: press Update"
                return False
            res, extra = _whole_mesh(context, source, src, pts, gs, L0, project)
            notes = notes or extra
    if res is None:
        state["message"] = notes or "Nothing could be remeshed"
        return False
    state["res"] = res
    state["L0"] = L0
    _build_preview(context, source, res)
    sizes = [len(f) for f in res["faces"]]
    state["ms"] = (time.perf_counter() - started) * 1000.0
    state["spot"] = None
    try:
        spot = next_spot(res, project, L0)
        state["spot"] = {"pos": spot[0], "text": spot[1]} if spot else None
    except Exception as exc:  # the hint must never break an update
        notes = notes or f"hint: {exc}"
    summary = f"{len(sizes):,} faces, {sizes.count(4) / max(len(sizes), 1) * 100:.0f}% quads"
    if not res["new"].all():  # the kept faces are whatever the source had: count only the new ones
        summary = f"{int(res['new'].sum()):,} new quads in {len(sizes):,} faces"
    state["message"] = f"Preview: {summary}, {state['ms'] / 1000.0:.1f} s" + (f" | {notes}" if notes else "")
    state["draw_key"] = None
    overlay.tag_redraw()
    return True


def _start(context) -> None:
    obj = context.object
    if obj is not None and obj.type == "MESH" and obj.name != PREVIEW_NAME:
        state["source"] = obj.name
    state.update(sig=None, dirty=True, changed_at=0.0)
    gs = context.scene.curiomesh_guided
    src_obj = state_source()
    if src_obj is not None and gs.new_radius <= 0:
        gs.new_radius = PRESETS[gs.preset]["radius"] * _source_data(src_obj)["diag"]
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_SECONDS)
    overlay.tag_redraw()


def _stop() -> None:
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    stop_work()
    remove_preview()
    state.update(sig=None, message="", lock=None, hover=None)
    overlay.tag_redraw()


def _poll():
    try:
        context = bpy.context
        gs = getattr(context.scene, "curiomesh_guided", None)
        if gs is None or not gs.enabled:
            return None
        active = context.object
        if active is not None and active.type == "MESH" and active.name not in (PREVIEW_NAME, state["source"]) and active.mode == "OBJECT":
            state.update(source=active.name, lock=None, sig=None)
            stop_work()
            remove_preview()
        source = state_source()
        preview = bpy.data.objects.get(PREVIEW_NAME)
        if source is None or source.mode != "OBJECT" or (preview is not None and preview.mode != "OBJECT"):
            return POLL_SECONDS
        _collect_work(context)
        sig = _signature(context, source)
        now = time.perf_counter()
        if sig != state["sig"]:
            state.update(sig=sig, changed_at=now, dirty=True)
            overlay.tag_redraw()
        elif state["dirty"] and now - state["changed_at"] >= (AREA_DEBOUNCE_SECONDS if gs.scope == "AREAS" else DEBOUNCE_SECONDS):
            state["dirty"] = False
            if gs.auto_update and (gs.scope == "AREAS" or state["ms"] <= LIVE_SLOW_MS):
                run_update(context)
            elif gs.auto_update:
                state["message"] = f"The last update took {state['ms'] / 1000.0:.0f} s: press Update"
            overlay.tag_redraw()
    except Exception as exc:
        state["message"] = f"Guided error: {exc}"[:120]
        state["dirty"] = False
    return POLL_SECONDS


# --------------------------------------------------------------------------- drawing


def _unit_sphere(lat=10, lon=16) -> np.ndarray:
    out = []
    for i in range(lat):
        t0, t1 = np.pi * i / lat, np.pi * (i + 1) / lat
        for j in range(lon):
            p0, p1 = 2 * np.pi * j / lon, 2 * np.pi * (j + 1) / lon
            q = [np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]) for t, p in ((t0, p0), (t0, p1), (t1, p1), (t1, p0))]
            out += [q[0], q[1], q[2], q[0], q[2], q[3]]
    return np.asarray(out)


_SPHERE = None


def _ring(center, normal, r, segments=48) -> np.ndarray:
    n = np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    a = np.cross(n, (1, 0, 0) if abs(n[0]) < 0.9 else (0, 1, 0))
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.asarray(center, float) + r * (np.outer(np.cos(t), a) + np.outer(np.sin(t), b))
    return np.stack((ring, np.roll(ring, -1, axis=0)), axis=1).reshape(-1, 3)


def _outline(src, pts, scope) -> np.ndarray:
    """Segments where the region boundary crosses the source surface."""
    co, tris = src["co"], src["tris"]
    sd = region_sd(co, pts, scope)
    inside = sd < 0
    mixed = inside[tris].any(axis=1) & ~inside[tris].all(axis=1)
    t = tris[mixed]
    if not len(t):
        return np.zeros((0, 3))
    points, flags = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = t[:, a], t[:, b]
        cross = inside[va] != inside[vb]
        da, db = sd[va], sd[vb]
        den = np.where(cross, da - db, 1.0)
        den = np.where(np.isfinite(den), den, 1.0)
        u = np.where(cross, np.clip(np.nan_to_num(da, posinf=0.0, neginf=0.0) / den, 0, 1), 0.0)
        points.append(co[va] + (co[vb] - co[va]) * u[:, None])
        flags.append(cross)
    segs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        pick = flags[i] & flags[j]
        segs.append(np.stack((points[i][pick], points[j][pick]), axis=1).reshape(-1, 3))
    return np.concatenate(segs)


def _colors():
    prefs = overlay.get_prefs()
    kinds = {k: overlay.pref_color(prefs, f"guided_{k.lower()}_color") for k in KIND_KEYS}
    return kinds, overlay.pref_color(prefs, "guided_outline_color"), overlay.pref_color(prefs, "guided_lock_color"), overlay.pref_color(prefs, "region_seam_color")


def _world(matrix, arr):
    m = np.asarray(matrix, dtype=np.float64)
    return np.asarray(arr, float) @ m[:3, :3].T + m[:3, 3]


def _draw_data(source, gs):
    global _SPHERE
    pts = gather(source, gs)
    preview = bpy.data.objects.get(PREVIEW_NAME)
    key = (state["sig"], id(state["res"]), id(state["spot"]), gs.scope, gs.show_cage)
    if state["draw_key"] == key and state["draw"] is not None:
        return state["draw"]
    if _SPHERE is None:
        _SPHERE = _unit_sphere()
    m = source.matrix_world
    scale = float(np.mean(m.to_scale()))
    data = {"spheres": {}, "rings": {}, "dots": {}, "outline": np.zeros((0, 3), np.float32), "lock": np.zeros((0, 3), np.float32), "spot": None}
    for kind in KIND_KEYS:
        mine = [p for p in pts if p.kind == kind]
        if mine:
            data["spheres"][kind] = np.concatenate([_world(m, _SPHERE * p.r + p.co) for p in mine]).astype(np.float32)
            rings = [_world(m, _ring(p.co, p.nrm, p.r)) for p in mine]
            rings += [_world(m, _ring(p.co, p.nrm, 0.5 * p.r)) for p in mine if p.kind == "LOOP"]
            if gs.scope == "AREAS" and preview is not None:  # the same areas on the result, side by side
                rings += [_world(preview.matrix_world, _ring(p.co, p.nrm, p.r)) for p in mine]
            data["rings"][kind] = np.concatenate(rings).astype(np.float32)
            data["dots"][kind] = _world(m, [p.co for p in mine]).astype(np.float32)
    if pts and gs.show_cage:
        seg = _outline(_source_data(source), pts, gs.scope)
        data["outline"] = _world(m, seg).astype(np.float32) if len(seg) else seg.astype(np.float32)
    res = state["res"]
    if res is not None and preview is not None and res["locked"].any():
        tri = [(f[0], f[k], f[k + 1]) for f, lk in zip(res["faces"], res["locked"]) if lk for k in range(1, len(f) - 1)]
        data["lock"] = _world(preview.matrix_world, res["co"][np.asarray(tri).reshape(-1)]).astype(np.float32)
    if state["spot"] is not None and preview is not None:
        c = _world(preview.matrix_world, [state["spot"]["pos"]])[0]
        r = 1.6 * state["L0"] * scale
        up = np.array([0.0, 0.0, 1.0])
        side = np.array([1.0, 0.0, 0.0])
        data["spot"] = np.concatenate((_ring(c, up, r), _ring(c, side, r))).astype(np.float32)
    state["draw"], state["draw_key"], state["batches"] = data, key, None
    return data


def _draw() -> None:
    try:
        gs = getattr(bpy.context.scene, "curiomesh_guided", None)
        source = state_source()
        if gs is None or not gs.enabled or source is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        data = _draw_data(source, gs)
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        colors, outline, lock_color, spot_color = _colors()

        def batch(kind, arr):
            return batch_for_shader(shader, kind, {"pos": arr}) if arr is not None and len(arr) else None

        if state["batches"] is None:
            state["batches"] = {
                "spheres": {k: batch("TRIS", v) for k, v in data["spheres"].items()},
                "rings": {k: batch("LINES", v) for k, v in data["rings"].items()},
                "dots": {k: batch("POINTS", v) for k, v in data["dots"].items()},
                "outline": batch("LINES", data["outline"]),
                "lock": batch("TRIS", data["lock"]),
                "spot": batch("LINES", data["spot"]),
            }
        b = state["batches"]
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_mask_set(False)
        shader.bind()
        if gs.show_cage:
            gpu.state.depth_test_set("LESS_EQUAL")
            for kind, bt in b["spheres"].items():
                if bt is not None:
                    shader.uniform_float("color", (*colors[kind][:3], 0.16))
                    bt.draw(shader)
            gpu.state.line_width_set(2.0)
            for kind, bt in b["rings"].items():
                if bt is not None:
                    shader.uniform_float("color", colors[kind])
                    bt.draw(shader)
            if b["outline"] is not None:
                gpu.state.depth_test_set("ALWAYS")
                gpu.state.line_width_set(3.5)
                shader.uniform_float("color", outline)
                b["outline"].draw(shader)
        gpu.state.depth_test_set("ALWAYS")
        gpu.state.point_size_set(9.0)
        for kind, bt in b["dots"].items():
            if bt is not None:
                shader.uniform_float("color", colors[kind])
                bt.draw(shader)
        hv = state["hover"]
        if hv is not None:
            gpu.state.line_width_set(2.5)
            shader.uniform_float("color", colors[hv["kind"]])
            batch("LINES", _world(source.matrix_world, _ring(hv["co"], hv["nrm"], hv["r"])).astype(np.float32)).draw(shader)
            preview = bpy.data.objects.get(PREVIEW_NAME)
            if preview is not None:
                batch("LINES", _world(preview.matrix_world, _ring(hv["co"], hv["nrm"], hv["r"])).astype(np.float32)).draw(shader)
        work = state["work"]
        running = work["runner"].current if work is not None and work["runner"].busy else None
        if running:  # the area the algorithm is on right now: a pulsing ring on both meshes
            pulse = 0.55 + 0.45 * float(np.sin(time.perf_counter() * 6.0))
            gpu.state.depth_test_set("ALWAYS")
            gpu.state.line_width_set(4.0)
            shader.uniform_float("color", (*spot_color[:3], pulse))
            preview = bpy.data.objects.get(PREVIEW_NAME)
            for i in running:
                if i < len(work["areas"]):
                    p = work["areas"][i]
                    for owner in (source, preview):
                        if owner is not None:
                            batch("LINES", _world(owner.matrix_world, _ring(p.co, p.nrm, p.r)).astype(np.float32)).draw(shader)
        if b["lock"] is not None:
            gpu.state.depth_test_set("LESS_EQUAL")
            shader.uniform_float("color", lock_color)
            b["lock"].draw(shader)
        if b["spot"] is not None:
            gpu.state.depth_test_set("ALWAYS")
            gpu.state.line_width_set(3.0)
            shader.uniform_float("color", spot_color)
            b["spot"].draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.point_size_set(1.0)
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("NONE")
    except Exception as exc:
        overlay.LAST_DRAW_ERROR = repr(exc)


# --------------------------------------------------------------------------- operators


def _gs(context):
    return context.scene.curiomesh_guided


def surface_hit(obj, origin: Vector, direction: Vector):
    """Raycast in world space. (local location, local normal) or None."""
    inv = obj.matrix_world.inverted()
    ok, loc, normal, _i = obj.ray_cast(inv @ origin, (inv.to_3x3() @ direction).normalized())
    return (np.asarray(loc, float), np.asarray(normal, float)) if ok else None


class _SurfaceTool:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.object
        return o is not None and o.type == "MESH" and o.name != PREVIEW_NAME and o.mode == "OBJECT" and context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        gs = _gs(context)
        self.obj = context.object
        state["source"] = self.obj.name
        if gs.new_radius <= 0:
            gs.new_radius = PRESETS[gs.preset]["radius"] * _source_data(self.obj)["diag"]
        if not gs.enabled:
            gs.enabled = True
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _status(self, context):
        gs = _gs(context)
        context.workspace.status_text_set(
            f"Place Points [{gs.new_kind.title()}]  Click: add  Ctrl+Click: remove  Scroll: radius {gs.new_radius:.3g}  "
            f"Shift+Scroll: strength {gs.new_strength:.1f}  1-4: type  Enter/Esc: done"
        )

    def _view_region(self, context, event):
        """The 3D view under the mouse. None over the sidebar, a header or another area, which keep working."""

        def under_mouse(r):
            return r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height

        area = context.area
        if area is None or area.type != "VIEW_3D" or not under_mouse(area):
            return None
        if any(r.type != "WINDOW" and r.width > 1 and r.height > 1 and under_mouse(r) for r in area.regions):
            return None
        return next((r for r in area.regions if r.type == "WINDOW" and under_mouse(r)), None)

    def _ray(self, region_, event):
        from bpy_extras import view3d_utils

        coord = (event.mouse_x - region_.x, event.mouse_y - region_.y)
        return surface_hit(self.obj, view3d_utils.region_2d_to_origin_3d(region_, region_.data, coord), view3d_utils.region_2d_to_vector_3d(region_, region_.data, coord))

    def _finish(self, context):
        context.workspace.status_text_set(None)
        state["hover"] = None
        overlay.tag_redraw()

    def modal(self, context, event):
        gs = _gs(context)
        if context.area is not None:
            context.area.tag_redraw()
        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        view = self._view_region(context, event)
        if view is None:
            state["hover"] = None
            return {"PASS_THROUGH"}
        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        if event.type == "MOUSEMOVE":
            hit = self._ray(view, event)
            state["hover"] = {"co": hit[0], "nrm": hit[1], "r": gs.new_radius, "kind": gs.new_kind} if hit else None
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and not event.alt and not event.ctrl:
            up = event.type == "WHEELUPMOUSE"
            if event.shift:
                gs.new_strength = float(np.clip(gs.new_strength + (0.1 if up else -0.1), 0.1, 4.0))
            else:
                gs.new_radius = gs.new_radius * (1.1 if up else 1 / 1.1)
            if state["hover"]:
                state["hover"]["r"] = gs.new_radius
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type in {"ONE", "TWO", "THREE", "FOUR"} and event.value == "PRESS":
            gs.new_kind = KIND_KEYS[("ONE", "TWO", "THREE", "FOUR").index(event.type)]
            if state["hover"]:
                state["hover"]["kind"] = gs.new_kind
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = self._ray(view, event)
            if hit is None:
                return {"RUNNING_MODAL"}
            if event.ctrl:
                pts = self.obj.curiomesh_points
                near = [(float(np.linalg.norm(np.asarray(p.co) - hit[0])), i) for i, p in enumerate(pts) if np.linalg.norm(np.asarray(p.co) - hit[0]) <= p.radius]
                if near:
                    pts.remove(min(near)[1])
                    self.obj.curiomesh_point_index = max(0, min(self.obj.curiomesh_point_index, len(pts) - 1))
            else:
                add_point(self.obj, hit[0], hit[1], gs.new_kind, gs.new_radius, gs.new_strength)
            return {"RUNNING_MODAL"}
        if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._finish(context)


class CURIOMESH_OT_guided_place(_SurfaceTool, bpy.types.Operator):
    bl_idname = "curiomesh.guided_place"
    bl_label = "Place Points"
    bl_description = (
        "Click the surface to add a point. Ctrl+click removes one, scroll sets the radius, Shift+scroll the strength, "
        "keys 1-4 pick Detail / Loop / Sharp / Calm, Enter or Esc finishes"
    )


class CURIOMESH_OT_guided_work(_SurfaceTool, bpy.types.Operator):
    bl_idname = "curiomesh.guided_work"
    bl_label = "Work Live"
    bl_description = (
        "Remesh the mesh area by area while you watch. Click or drag on your mesh or on the result to add an area. "
        "Over an area, scroll sets its size, Shift+scroll its density and F its fill style. It redoes only what changed, in the background"
    )

    def invoke(self, context, event):
        _gs(context).scope = "AREAS"
        self.dab = None
        self.under = None
        return super().invoke(context, event)

    def _status(self, context):
        gs = _gs(context)
        p = self._area()
        kind, strength, fill, radius = (p.kind, p.strength, p.fill, p.radius) if p else (gs.new_kind, gs.new_strength, gs.strategy, gs.new_radius)
        head = f"Area {self.under + 1}" if p else "New area"
        context.workspace.status_text_set(
            f"{head}: {kind.title()} {strength:.2f}, {fill.title()}, size {radius:.3g}  |  Click/Drag: add  Ctrl+Click: remove  "
            "Scroll: size  Shift+Scroll: density  F: fill style  Enter/Esc: done"
        )

    def _area(self):
        pts = self.obj.curiomesh_points
        return pts[self.under] if self.under is not None and self.under < len(pts) else None

    def _ray(self, region_, event):
        hit = super()._ray(region_, event)
        preview = bpy.data.objects.get(PREVIEW_NAME)
        if hit is not None or preview is None:
            return hit
        from bpy_extras import view3d_utils

        coord = (event.mouse_x - region_.x, event.mouse_y - region_.y)
        try:  # same object-space shape as your mesh, only moved aside
            return surface_hit(preview, view3d_utils.region_2d_to_origin_3d(region_, region_.data, coord), view3d_utils.region_2d_to_vector_3d(region_, region_.data, coord))
        except RuntimeError:  # the result was swapped this very moment
            return None

    def _hover(self, context, hit):
        gs = _gs(context)
        self.under = None
        if hit is not None:
            near = [(float(np.linalg.norm(np.asarray(p.co) - hit[0])) / max(p.radius, 1e-9), i) for i, p in enumerate(self.obj.curiomesh_points)]
            near = [n for n in near if n[0] <= 1.0]
            self.under = min(near)[1] if near else None
        p = self._area()
        if self.under is not None:
            self.obj.curiomesh_point_index = self.under
        state["hover"] = {"co": hit[0], "nrm": hit[1], "r": p.radius if p else gs.new_radius, "kind": p.kind if p else gs.new_kind} if hit else None
        self._status(context)

    def _add(self, gs, hit):
        add_point(self.obj, hit[0], hit[1], gs.new_kind, gs.new_radius, gs.new_strength, fill=gs.strategy)
        self.dab = hit[0]

    def _brush_follows(self, gs, p):
        gs.new_kind, gs.new_strength, gs.new_radius, gs.strategy = p.kind, p.strength, p.radius, p.fill

    def modal(self, context, event):
        gs = _gs(context)
        if context.area is not None:
            context.area.tag_redraw()
        if event.type in {"RET", "NUMPAD_ENTER", "ESC"} and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self.dab = None
        view = self._view_region(context, event)
        if view is None:
            state["hover"] = None
            self.dab = None
            return {"PASS_THROUGH"}
        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            self._finish(context)
            return {"FINISHED"}
        if event.type == "MOUSEMOVE":
            hit = self._ray(view, event)
            if self.dab is not None and hit is not None and np.linalg.norm(hit[0] - self.dab) > 0.75 * gs.new_radius:
                self._add(gs, hit)
            self._hover(context, hit)
            return {"RUNNING_MODAL"}
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and not event.alt and not event.ctrl:
            up = event.type == "WHEELUPMOUSE"
            p = self._area()
            kind, strength, radius = (p.kind, p.strength, p.radius) if p else (gs.new_kind, gs.new_strength, gs.new_radius)
            if event.shift:
                kind, strength = stepped_intensity(kind, strength, up)
            else:
                radius *= 1.1 if up else 1 / 1.1
            if p:
                p.kind, p.strength, p.radius = kind, strength, radius
                self._brush_follows(gs, p)
            else:
                gs.new_kind, gs.new_strength, gs.new_radius = kind, strength, radius
            if state["hover"]:
                state["hover"].update(r=radius, kind=kind)
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type == "F" and event.value == "PRESS":
            p = self._area()
            fill = "GRID" if (p.fill if p else gs.strategy) == "FLOW" else "FLOW"
            if p:
                p.fill = fill
            gs.strategy = fill
            self._status(context)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = self._ray(view, event)
            if hit is None:
                return {"RUNNING_MODAL"}
            self._hover(context, hit)
            if event.ctrl:
                if self.under is not None:
                    self.obj.curiomesh_points.remove(self.under)
                    self.obj.curiomesh_point_index = max(0, min(self.under, len(self.obj.curiomesh_points) - 1))
                    self.under = None
            else:
                self._add(gs, hit)
            return {"RUNNING_MODAL"}
        if event.type in {"MIDDLEMOUSE", "TRACKPADPAN", "TRACKPADZOOM", "MOUSEROTATE"} or event.type.startswith("NUMPAD") or event.alt:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}


class CURIOMESH_OT_guided_add_point(bpy.types.Operator):
    bl_idname = "curiomesh.guided_add_point"
    bl_label = "Add Point"
    bl_description = "Add an importance point at the 3D cursor, snapped to the surface of the active mesh"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Type", items=KIND_ITEMS, default="DETAIL")
    radius: bpy.props.FloatProperty(name="Radius", default=0.0, min=0.0, subtype="DISTANCE")
    location: bpy.props.FloatVectorProperty(name="Location", size=3, subtype="XYZ", default=(0.0, 0.0, 0.0), description="World-space spot (0,0,0 uses the 3D cursor)")

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        obj = _source_mesh(context)
        gs = _gs(context)
        world = Vector(self.location) if any(self.location) else context.scene.cursor.location
        ok, loc, normal, _i = obj.closest_point_on_mesh(obj.matrix_world.inverted() @ world)
        if not ok:
            self.report({"ERROR"}, "No surface found")
            return {"CANCELLED"}
        radius = self.radius or gs.new_radius or PRESETS[gs.preset]["radius"] * _source_data(obj)["diag"]
        add_point(obj, loc, normal, self.kind, radius, gs.new_strength)
        return {"FINISHED"}


class _PointsOp:
    @classmethod
    def poll(cls, context):
        o = _source_mesh(context)
        return o is not None and len(o.curiomesh_points) > 0


class CURIOMESH_OT_guided_remove_point(_PointsOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_remove_point"
    bl_label = "Remove Point"
    bl_description = "Remove the point selected in the list"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        o = _source_mesh(context)
        o.curiomesh_points.remove(o.curiomesh_point_index)
        o.curiomesh_point_index = max(0, min(o.curiomesh_point_index, len(o.curiomesh_points) - 1))
        return {"FINISHED"}


class CURIOMESH_OT_guided_clear_points(_PointsOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_clear_points"
    bl_label = "Clear Points"
    bl_description = "Remove every point from the mesh (tagged empties stay)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _source_mesh(context).curiomesh_points.clear()
        return {"FINISHED"}


class CURIOMESH_OT_guided_add_empty(bpy.types.Operator):
    bl_idname = "curiomesh.guided_add_empty"
    bl_label = "Add Empty Point"
    bl_description = (
        "Add a sphere Empty at the 3D cursor as a point. Move and scale it like any object; its size is the radius. "
        "Custom properties curiomesh_kind and curiomesh_strength set the type and strength"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        obj = _source_mesh(context)
        gs = _gs(context)
        empty = bpy.data.objects.new("CM_Point", None)
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = gs.new_radius or PRESETS[gs.preset]["radius"] * _source_data(obj)["diag"]
        empty.location = context.scene.cursor.location
        empty[POINT_TAG] = 1
        empty["curiomesh_kind"] = gs.new_kind
        empty["curiomesh_strength"] = gs.new_strength
        (obj.users_collection[0] if obj.users_collection else context.scene.collection).objects.link(empty)
        return {"FINISHED"}


class CURIOMESH_OT_guided_suggest(bpy.types.Operator):
    bl_idname = "curiomesh.guided_suggest"
    bl_label = "Auto-suggest Points"
    bl_description = "Place points at high-curvature spots, hard edges and small holes (eyes, mouths). Edit them afterwards; suggesting again replaces the earlier suggestions"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        obj = _source_mesh(context)
        gs = _gs(context)
        for i in reversed(range(len(obj.curiomesh_points))):
            if obj.curiomesh_points[i].auto:
                obj.curiomesh_points.remove(i)
        found = suggest(obj, gs)
        for f in found:
            add_point(obj, f["co"], f["normal"], f["kind"], f["r"], gs.new_strength, auto=True)
        state["source"] = obj.name
        self.report({"INFO"}, f"Placed {len(found)} point(s). Edit them in the list or with Place Points")
        return {"FINISHED"}


class CURIOMESH_OT_guided_update(bpy.types.Operator):
    bl_idname = "curiomesh.guided_update"
    bl_label = "Update"
    bl_description = "Rebuild the preview now, even when it is too big for automatic updates"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _source_mesh(context) is not None

    def execute(self, context):
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        gs = _gs(context)
        obj = _source_mesh(context)
        state["source"] = obj.name
        if not gs.enabled:
            gs.enabled = True
        ok = run_update(context, force=True)
        state["sig"] = _signature(context, obj)
        state["dirty"] = False
        if not ok:
            self.report({"WARNING"}, state["message"])
        return {"FINISHED"} if ok else {"CANCELLED"}


class _PreviewOp:
    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(PREVIEW_NAME) is not None and state["res"] is not None


class CURIOMESH_OT_guided_edit_preview(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_edit_preview"
    bl_label = "Select Faces on Preview"
    bl_description = "Enter Edit Mode on the preview so you can select faces to lock or a strip to redraw"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        preview = bpy.data.objects[PREVIEW_NAME]
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        preview.hide_select = False
        preview.select_set(True)
        context.view_layer.objects.active = preview
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        return {"FINISHED"}


def _preview_selection():
    preview = bpy.data.objects[PREVIEW_NAME]
    if preview.mode == "EDIT":
        preview.update_from_editmode()
    sel = np.zeros(len(preview.data.polygons), dtype=bool)
    preview.data.polygons.foreach_get("select", sel)
    return sel


def _leave_preview(context):
    if context.object is not None and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    source = state_source()
    if source is not None:
        bpy.ops.object.select_all(action="DESELECT")
        source.select_set(True)
        context.view_layer.objects.active = source


def _set_lock(res, locked, free_all=False):
    stop_work()
    state["lock"] = {"co": res["co"], "faces": res["faces"], "locked": locked, "mat": res.get("mat"), "free_all": free_all, "id": time.perf_counter_ns()}


class CURIOMESH_OT_guided_lock(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_lock"
    bl_label = "Lock Faces"
    bl_description = "Keep the faces selected on the preview exactly as they are; later updates rebuild only the rest, around them"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        sel = _preview_selection()
        res = state["res"]
        if len(sel) != len(res["faces"]) or not sel.any():
            self.report({"WARNING"}, "Select some preview faces in Edit Mode first")
            return {"CANCELLED"}
        _leave_preview(context)
        _set_lock(res, res["locked"] | sel, state["lock"]["free_all"] if state["lock"] else False)
        ok = run_update(context, force=True)
        state["sig"] = _signature(context, state_source())
        state["dirty"] = False
        self.report({"INFO"} if ok else {"WARNING"}, f"{int(state['lock']['locked'].sum()):,} faces locked. {state['message']}")
        return {"FINISHED"}


class CURIOMESH_OT_guided_unlock(bpy.types.Operator):
    bl_idname = "curiomesh.guided_unlock"
    bl_label = "Unlock All"
    bl_description = "Forget all locked faces and rebuild everything from your mesh"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return state["lock"] is not None

    def execute(self, context):
        state["lock"] = None
        if state_source() is not None and _gs(context).enabled:
            run_update(context, force=True)
        return {"FINISHED"}


class CURIOMESH_OT_guided_redraw_strip(_PreviewOp, bpy.types.Operator):
    bl_idname = "curiomesh.guided_redraw_strip"
    bl_label = "Redraw Strip"
    bl_description = "Rebuild only the faces selected on the preview (a strip or loop of faces) with the region tool. The rest of the preview is frozen afterwards"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        sel = _preview_selection()
        res = state["res"]
        gs = _gs(context)
        if len(sel) != len(res["faces"]) or not sel.any():
            self.report({"WARNING"}, "Select the strip of preview faces in Edit Mode first")
            return {"CANCELLED"}
        _leave_preview(context)
        source = state_source()
        co, faces = res["co"], res["faces"]
        edge = float(np.mean([np.linalg.norm(co[f[k]] - co[f[(k + 1) % len(f)]]) for i, f in enumerate(faces) if sel[i] for k in range(len(f))]))
        new, warns = _region_pass(co, faces, sel, [], state["L0"], gs, region._projector(_bvh(_source_data(source))), res["locked"], res.get("mat"), fixed_target=edge)
        if new is None:
            self.report({"WARNING"}, warns[0] if warns else "Could not redraw that strip")
            return {"CANCELLED"}
        _set_lock(new, ~new["new"], free_all=True)
        new["locked"] = state["lock"]["locked"]
        state["res"] = new
        _build_preview(context, source, new)
        state["draw_key"] = None
        state["sig"] = _signature(context, source)
        state["message"] = f"Strip rebuilt: {int(new['new'].sum())} new faces. The rest is frozen (Unlock All to start over)"
        overlay.tag_redraw()
        return {"FINISHED"}


class CURIOMESH_OT_guided_frame(bpy.types.Operator):
    bl_idname = "curiomesh.guided_frame"
    bl_label = "Frame"
    bl_description = "Move the 3D view to the next spot worth fixing"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return state["spot"] is not None and bpy.data.objects.get(PREVIEW_NAME) is not None

    def execute(self, context):
        rv3d = getattr(context.space_data, "region_3d", None) or context.region_data
        if rv3d is None:
            return {"CANCELLED"}
        preview = bpy.data.objects[PREVIEW_NAME]
        rv3d.view_location = preview.matrix_world @ Vector(state["spot"]["pos"])
        rv3d.view_distance = max(12.0 * state["L0"] * float(np.mean(preview.matrix_world.to_scale())), 1e-3)
        return {"FINISHED"}


class CURIOMESH_OT_guided_apply(bpy.types.Operator):
    bl_idname = "curiomesh.guided_apply"
    bl_label = "Apply"
    bl_description = "Write the preview as <name>_Retopo (or into your object, see Result). Undo restores"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return state["res"] is not None and state_source() is not None

    def execute(self, context):
        gs = _gs(context)
        source = state_source()
        res = state["res"]
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        finish_work(context)
        res = state["res"]
        if res is None:
            self.report({"WARNING"}, "Nothing to apply")
            return {"CANCELLED"}
        mesh = _make_mesh(source.name + "_Retopo", source, res)
        for poly in mesh.polygons:
            poly.use_smooth = True
        if gs.apply_mode == "REPLACE":
            source.data = mesh
            out = source
        else:
            out = bpy.data.objects.new(source.name + "_Retopo", mesh)
            (source.users_collection[0] if source.users_collection else context.scene.collection).objects.link(out)
            out.matrix_world = source.matrix_world.copy()
            for slot, copy in zip(source.material_slots, out.material_slots):
                if slot.link == "OBJECT":  # materials kept on the object, not in the mesh
                    copy.link = "OBJECT"
                    copy.material = slot.material
        count = len(mesh.polygons)
        gs.enabled = False  # removes the preview, stops the timer
        bpy.ops.object.select_all(action="DESELECT")
        out.select_set(True)
        context.view_layer.objects.active = out
        self.report({"INFO"}, f"{count:,} faces written to {out.name}")
        return {"FINISHED"}


class CURIOMESH_OT_guided_stop(bpy.types.Operator):
    bl_idname = "curiomesh.guided_stop"
    bl_label = "Cancel"
    bl_description = "Turn the guided preview off and remove it. Your points stay on the mesh"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        _leave_preview(context)
        _gs(context).enabled = False
        return {"FINISHED"}


# --------------------------------------------------------------------------- panel


class CURIOMESH_UL_guided_points(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "kind", text="", icon_only=True)
        row.prop(item, "fill", text="", icon_only=True)
        row.prop(item, "radius", text="")
        row.prop(item, "strength", text="")


def _is_problem(message: str) -> bool:
    return any(w in message for w in ("skipped", "error", "Error", "failed", "Would make")) or message.startswith(("Select", "Place", "Nothing", "Points Only"))


class CURIOMESH_PT_guided(bpy.types.Panel):
    bl_label = "Guided"
    bl_idname = "CURIOMESH_PT_guided"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return bool(overlay.get_prefs().show_guided)

    def draw_header(self, context):
        self.layout.label(icon="PIVOT_ACTIVE")

    def draw(self, context):
        layout = self.layout
        gs = context.scene.curiomesh_guided
        compact = bool(overlay.get_prefs().compact_mode)
        obj = _source_mesh(context)
        previewing = state["res"] is not None and bpy.data.objects.get(PREVIEW_NAME) is not None
        if obj is None:
            layout.box().label(text="Select a mesh to begin", icon="INFO")

        layout.label(text="1. Choose a preset & place points", icon="PIVOT_ACTIVE")
        layout.prop(gs, "preset", text="")
        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("curiomesh.guided_place", text="Place Points", icon="MOUSE_LMB")
        row.operator("curiomesh.guided_suggest", text="Auto-suggest", icon="LIGHT")
        row = layout.row()
        row.scale_y = 1.4
        row.operator("curiomesh.guided_work", icon="BRUSH_DATA")
        if obj is not None:
            row = layout.row()
            row.template_list("CURIOMESH_UL_guided_points", "", obj, "curiomesh_points", obj, "curiomesh_point_index", rows=3)
            col = row.column(align=True)
            col.operator("curiomesh.guided_add_point", text="", icon="ADD")
            col.operator("curiomesh.guided_remove_point", text="", icon="REMOVE")
            col.operator("curiomesh.guided_clear_points", text="", icon="TRASH")
            col.operator("curiomesh.guided_add_empty", text="", icon="EMPTY_AXIS")
            if 0 <= obj.curiomesh_point_index < len(obj.curiomesh_points):
                item = obj.curiomesh_points[obj.curiomesh_point_index]
                col = layout.column(align=True)
                col.prop(item, "kind", text="")
                col.prop(item, "fill", text="")
                col.prop(item, "radius")
                col.prop(item, "strength")
                col.prop(item, "co", text="Location")
            elif not len(obj.curiomesh_points) and not compact:
                layout.label(text="Detail = smaller quads, Calm = bigger,", icon="INFO")
                layout.label(text="Loop = rings around, Sharp = keep edges")
        col = layout.column(align=True)
        col.prop(gs, "new_kind", text="Next Point")
        col.prop(gs, "new_radius", text="Radius")
        col.prop(gs, "new_strength", text="Strength")

        layout.label(text="2. Watch the live preview", icon="HIDE_OFF")
        layout.prop(gs, "enabled", toggle=True, text="Live Preview & Cage", icon="HIDE_OFF" if gs.enabled else "HIDE_ON")
        layout.row(align=True).prop(gs, "scope", expand=True)
        layout.row(align=True).prop(gs, "strategy", expand=True)
        layout.prop(gs, "base_faces")
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("curiomesh.guided_update", text="Update", icon="FILE_REFRESH")
        row.prop(gs, "auto_update", text="Auto", toggle=True)
        message = status()
        if message:
            box = layout.box()
            box.alert = _is_problem(message)
            for i, chunk in enumerate(message.split(" | ")[:2]):
                overlay.wrap_label(box, chunk, icon=("ERROR" if box.alert else "INFO") if i == 0 else "BLANK1", width=36, limit=2)
        if state["spot"] is not None:
            box = layout.box()
            box.label(text="Next spot to fix", icon="ZOOM_ALL")
            overlay.wrap_label(box, state["spot"]["text"], width=36, limit=2)
            box.operator("curiomesh.guided_frame", icon="VIEWZOOM")

        layout.label(text="3. Lock, redraw, apply", icon="CHECKMARK")
        col = layout.column(align=True)
        col.enabled = previewing
        col.operator("curiomesh.guided_edit_preview", icon="EDITMODE_HLT")
        row = col.row(align=True)
        row.operator("curiomesh.guided_lock", icon="LOCKED")
        row.operator("curiomesh.guided_redraw_strip", icon="MOD_SOLIDIFY")
        if state["lock"] is not None:
            layout.operator("curiomesh.guided_unlock", icon="UNLOCKED", text=f"Unlock All ({int(state['lock']['locked'].sum()):,})")
        row = layout.row(align=True)
        row.scale_y = 1.6 if previewing else 1.2
        sub = row.row(align=True)
        sub.enabled = previewing
        sub.operator("curiomesh.guided_apply", icon="CHECKMARK")
        row.operator("curiomesh.guided_stop", icon="X")
        layout.prop(gs, "apply_mode", text="")
        if not previewing and not compact:
            layout.label(text="Nothing changes until you press Apply.", icon="LOCKED")


class CURIOMESH_PT_guided_advanced(bpy.types.Panel):
    bl_label = "Advanced"
    bl_idname = "CURIOMESH_PT_guided_advanced"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CurioMesh"
    bl_parent_id = "CURIOMESH_PT_guided"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(icon="PREFERENCES")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        gs = context.scene.curiomesh_guided
        layout.prop(gs, "show_cage")
        layout.prop(gs, "feature_angle")
        layout.prop(gs, "fit_iterations")
        layout.prop(gs, "pure_quads")
        layout.prop(gs, "suggest_count")


PANEL_CLASSES = (CURIOMESH_PT_guided, CURIOMESH_PT_guided_advanced)

CLASSES = (
    CURIOMESH_PG_guided_point,
    CURIOMESH_PG_guided,
    CURIOMESH_OT_guided_place,
    CURIOMESH_OT_guided_work,
    CURIOMESH_OT_guided_add_point,
    CURIOMESH_OT_guided_remove_point,
    CURIOMESH_OT_guided_clear_points,
    CURIOMESH_OT_guided_add_empty,
    CURIOMESH_OT_guided_suggest,
    CURIOMESH_OT_guided_update,
    CURIOMESH_OT_guided_edit_preview,
    CURIOMESH_OT_guided_lock,
    CURIOMESH_OT_guided_unlock,
    CURIOMESH_OT_guided_redraw_strip,
    CURIOMESH_OT_guided_frame,
    CURIOMESH_OT_guided_apply,
    CURIOMESH_OT_guided_stop,
    CURIOMESH_UL_guided_points,
    CURIOMESH_PT_guided,
    CURIOMESH_PT_guided_advanced,
)

_draw_handles: list = []


@bpy.app.handlers.persistent
def _on_load_post(*_args):
    stop_work()
    state.update(source="", sig=None, src=None, res=None, lock=None, spot=None, draw=None, draw_key=None, batches=None, message="")
    for sc in bpy.data.scenes:
        gs = getattr(sc, "curiomesh_guided", None)
        if gs is not None and gs.enabled:
            gs["enabled"] = False  # the timer is gone after a load; the user turns it on again


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_guided = bpy.props.PointerProperty(type=CURIOMESH_PG_guided)
    bpy.types.Object.curiomesh_points = bpy.props.CollectionProperty(type=CURIOMESH_PG_guided_point)
    bpy.types.Object.curiomesh_point_index = bpy.props.IntProperty(default=0)
    _draw_handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW"))
    bpy.app.handlers.load_post.append(_on_load_post)


def unregister() -> None:
    while _draw_handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handles.pop(), "WINDOW")
        except Exception:
            pass
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    stop_work()
    try:
        remove_preview()
    except Exception:
        pass
    state.update(source="", sig=None, src=None, res=None, lock=None, spot=None, hover=None, message="")
    for owner, name in ((bpy.types.Scene, "curiomesh_guided"), (bpy.types.Object, "curiomesh_points"), (bpy.types.Object, "curiomesh_point_index")):
        if hasattr(owner, name):
            delattr(owner, name)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
