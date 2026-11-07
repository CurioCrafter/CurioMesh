# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Classic path UV handling (package A1c): pairing that no longer refuses UV meshes, and per-side texture transfer.

Our own design and code. `pair` replaces the `_pair_natively` call in the engine: a mesh with no UVs goes straight to
`_pair_natively` (identical result), a UV mesh gets only the pairs whose two triangles agree on the UVs of their shared
edge (a pair across a seam stays two triangles, per face, instead of one bad pair dropping every UV). `preserve_textures`
replaces the island-blind DataTransfer with K1's side-aware transfer (`uvseams`, SPEC 4.8) when the native core is on.
No `bpy` import: only the mesh objects handed in are touched.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import uvseams


def pair(mesh, prep, seeds):
    """Drop-in for `engine._pair_natively(mesh, prep, seeds)`; returns the same kind of result (None: Python pairs)."""

    from . import engine as E  # late: the engine imports this module

    if mesh.uvs is None or mesh.face_uvs is None:
        return E._pair_natively(mesh, prep, seeds)  # no UVs: untouched native fast path
    face_a, face_b, quads, scores = prep.candidates
    mesh, bad = _sanitize(mesh)
    keep = _agree(mesh, face_a, face_b, quads, bad)
    if keep is None:
        return None
    if keep.all() and E._turbo is None and not bad.any():
        return None  # nothing to filter, Python pairs lazily as before
    prep = replace(prep, candidates=(face_a[keep], face_b[keep], quads[keep], scores[keep]), table=None)
    if E._turbo is not None:
        bare = replace(mesh, uvs=None, face_uvs=None)  # same native kernel, UVs attached below
        found = E._pair_natively(bare, prep, seeds)
        if found is not None:
            return {seed: (_with_uvs(mesh, prep, out), breaks, rescued) for seed, (out, breaks, rescued) in found.items()}
    return {seed: E._pair_triangles(mesh, prep, seed) for seed in seeds}


def _sanitize(mesh):
    """(mesh, bad): triangles whose UV id is out of range get one extra (0, 0) UV so no id points outside the table
    downstream; `bad` marks them. They are never paired (their UVs mean nothing) and stay triangles."""

    fuv = np.asarray(mesh.face_uvs, np.int64).reshape(-1, 3)
    bad = ~np.all((fuv >= 0) & (fuv < len(mesh.uvs)), axis=1)
    if not bad.any():
        return mesh, bad
    fuv = fuv.copy()
    fuv[bad] = len(mesh.uvs)
    uvs = np.vstack([np.asarray(mesh.uvs, float).reshape(-1, 2), [[0.0, 0.0]]])
    return replace(mesh, uvs=uvs, face_uvs=[tuple(r) for r in fuv.tolist()]), bad


def _agree(mesh, face_a, face_b, quads, bad):
    """Candidates whose two triangles use the same UV at both shared-edge vertices and have valid UV ids; None if the table is unusable."""

    faces = np.asarray(mesh.faces, np.int64).reshape(-1, 3)
    fuv = np.asarray(mesh.face_uvs, np.int64).reshape(-1, 3)
    if len(fuv) != len(faces):
        return None
    ok_face = ~bad

    def at(f, v):
        return np.take_along_axis(fuv[f], (faces[f] == v[:, None]).argmax(1)[:, None], 1)[:, 0]

    keep = ok_face[face_a] & ok_face[face_b]
    for k in (1, 3):  # quad = (opp_a, x, opp_b, y): x and y are the shared edge
        keep &= at(face_a, quads[:, k]) == at(face_b, quads[:, k])
    return keep


def _with_uvs(mesh, prep, out):
    """`out` (quads, then the unpaired triangles, from the native kernel) with the face UVs of `mesh` attached."""

    face_a, face_b, quads, _ = prep.candidates
    by_quad = {tuple(q): (a, b) for q, a, b in zip(quads.tolist(), face_a.tolist(), face_b.tolist())}
    by_set = {tuple(sorted(q)): ab for q, ab in by_quad.items()}
    tri_at = {}
    for i, tri in enumerate(mesh.faces):
        tri_at.setdefault(tuple(tri), i)
    face_uvs = []
    for face in out.faces:
        if len(face) == 3:
            face_uvs.append(tuple(mesh.face_uvs[tri_at[tuple(face)]]))
            continue
        a, b = by_quad.get(tuple(face)) or by_set[tuple(sorted(face))]
        have = {}
        for f in (a, b):
            have.update(zip(mesh.faces[f], mesh.face_uvs[f]))
        face_uvs.append(tuple(have[v] for v in face))
    out.uvs, out.face_uvs = mesh.uvs, face_uvs
    return out


# ---------------------------------------------------------------- texture transfer (bridge._preserve_textures)
def read_uv(mesh, name):
    flat = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers.get(name).data.foreach_get("uv", flat)
    return flat.reshape(-1, 2).astype(np.float64)


def write_uv(mesh, uv, name):
    layer = mesh.uv_layers.get(name) or mesh.uv_layers.new(name=name)
    layer.data.foreach_set("uv", np.ascontiguousarray(uv, dtype=np.float32).reshape(-1))


def _polys(obj):
    """(points in world space, indptr, flat vertex ids, material index per face) of an object's mesh."""

    mesh = obj.data
    points = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3)
    matrix = getattr(obj, "matrix_world", None)
    if matrix is not None:
        m = np.array([[matrix[r][c] for c in range(4)] for r in range(4)], float)
        points = points @ m[:3, :3].T + m[:3, 3]
    total = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_total", total)
    flat = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", flat)
    material = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("material_index", material)
    indptr = np.concatenate(([0], np.cumsum(total))).astype(np.int64)
    return points, indptr, flat.astype(np.int64), material


def _ensure_fast() -> None:
    """Install the compiled UV kernels if nobody has yet (the add-on's own `install()` call comes with the H1 hooks)."""

    if uvseams.FAST.get("transfer") is None:
        try:
            from .. import uv_fast

            uv_fast.install()
        except Exception:
            pass


# The fold fix of SPEC 4.8 step 4 (harmonic fill of folded faces and their ring) is off in the Classic path: measured on the
# nine bench models it moves the UVs of faces that are only straddling a seam, which raised texture drift p99 from
# 0.8-2 px to 7-44 px (barrier, lion, bust, hydrant) while it removed at most half of a few dozen flipped faces. Drift is
# the Classic gate; Spot-fix style repair belongs to the tools that have a preview.
FIX_FOLDS = False


def preserve_textures(source, target, config, allow_python_bvh: bool = True):
    """Per-side transfer of every UV layer. Returns "TRANSFERRED", or None to let the old DataTransfer/BVH code run
    (not a transfer mode, no source UVs, no native core, or the transfer gave nothing usable); the reason is stored on
    the target as `curiomesh_uv_note` when it is a missing native core."""

    if str(config.texture_mode or "PROJECT").upper() not in {"TRANSFER", "PROJECT"}:
        return None
    layers = [layer.name for layer in source.data.uv_layers]
    if not layers:
        return None
    _ensure_fast()
    if uvseams.FAST.get("transfer") is None:
        try:
            target["curiomesh_uv_note"] = "Native core off: textures moved with the older island-blind transfer."
        except Exception:
            pass
        return None
    try:
        sp, si, sf, material = _polys(source)
        rp, ri, rf, _ = _polys(target)
        active = source.data.uv_layers.active.name if source.data.uv_layers.active else layers[0]
        layers.sort(key=lambda n: n != active)  # the active layer first: it makes the lookup the others reuse
        first = read_uv(source.data, layers[0])
        out = uvseams.transfer(sp, si, sf, first.ravel(), rp, ri, rf, material=material, fix_folds=FIX_FOLDS)
        if not out.get("done", True):
            return None
        results = {layers[0]: out["uv"]}
        for name in layers[1:]:
            results[name] = uvseams.layer(si, sf, read_uv(source.data, name).ravel(), rp, ri, rf, out, fix_folds=FIX_FOLDS)[0]
        for name, uv in results.items():
            uv = np.asarray(uv, float).reshape(-1, 2)
            bad = ~np.isfinite(uv).all(axis=1)
            if bad.mean() > 0.5:
                return None
            uv[bad] = 0.0  # corners the lookup could not reach; the legacy check below still judges the layer
            write_uv(target.data, uv, name)
        target.data.uv_layers.active = target.data.uv_layers.get(active)
    except Exception:
        return None
    return "TRANSFERRED"
