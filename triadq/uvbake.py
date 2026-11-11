# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Bake-safe UVs for the add-on (package A2): the bake method `CURIO`, the gap that is also the bake margin, the sharpen
rule for cuts, the seam step check after a bake and the previous pixels of baked images (so Ctrl+Z reaches them).

`textures.py` calls into this file from five marked lines; nothing else in the bake path was edited. The module holds
only a little state (the last run's message and numbers, the image history) and imports `bpy` lazily, so the pure
parts (gap, step check, history rules) are tested without Blender.

SPEC 8: hard edges are forced cuts; a cut becomes a sharp edge only where its dihedral angle is over the shading angle
(Sharpen cuts: Off / Only creased / All); the gap is 2^(k+1) texels for "clean down to 256 px"; the bake margin is that
gap; after the bake the normal and colour maps are sampled on both sides of every cut (bilinear, mip 0 and mip k).
"""

from __future__ import annotations

from math import radians
import time

import numpy as np

from . import uvnew, uvscore

MODE = "CURIO"  # "CURIO": New UVs make the bake layer; anything else: Smart Project as before (UVs: Off)
SHARPEN = "creased"  # off / creased / all
SHADING_ANGLE = radians(30.0)  # a cut is sharp where the faces on its two sides turn more than this
DEFAULT_RESOLUTION = 2048
SAMPLES_PER_EDGE = 5
HISTORY_KEY = "curio_uv_step"  # scene counter: Blender's undo restores it, which tells the handler which pixels belong

# What the last run did, for the card and the messages: {"message", "numbers", "verdict", "seam_step", "seconds", "settings"}.
LAST: dict = {}
_resolution = [DEFAULT_RESOLUTION]
_cache: dict = {}


def gap_px(resolution: int, clean_px: float = uvnew.CLEAN_DOWN_TO_PX) -> int:
    """The gap in texels at the bake size, which is also the bake margin (SPEC 8.2). Remembers the size for the unwrap
    that follows (the caller passes only the margin on, as a fraction of the square)."""

    _resolution[0] = max(int(resolution), 1)
    return uvnew.gap_texels(resolution, clean_px)


def _forget() -> None:
    LAST.clear()


# ---------------------------------------------------------------------------------------------------------------------
# The unwrap (called from textures.ensure_target_uvs)
# ---------------------------------------------------------------------------------------------------------------------


def mesh_arrays(mesh) -> tuple:
    """(points, indptr, flat, material per face, forced vertex pairs from sharp edges) of a Blender mesh."""

    nv, nf, nl = len(mesh.vertices), len(mesh.polygons), len(mesh.loops)
    points = np.empty(nv * 3)
    mesh.vertices.foreach_get("co", points)
    starts, sizes = np.empty(nf, np.int64), np.empty(nf, np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", sizes)
    if nf and (starts[0] != 0 or not np.array_equal(starts[1:], np.cumsum(sizes)[:-1])):
        raise ValueError("The mesh's faces are not stored in corner order.")
    loops = np.empty(nl, np.int64)
    mesh.loops.foreach_get("vertex_index", loops)
    material = np.empty(nf, np.int64)
    mesh.polygons.foreach_get("material_index", material)
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    forced = np.zeros((0, 2), np.int64)
    if len(mesh.edges):
        sharp = np.empty(len(mesh.edges), bool)
        mesh.edges.foreach_get("use_edge_sharp", sharp)
        ends = np.empty(len(mesh.edges) * 2, np.int64)
        mesh.edges.foreach_get("vertices", ends)
        forced = ends.reshape(-1, 2)[sharp]
    return points.reshape(-1, 3), indptr, loops, material, forced


def ensure_target_uvs(obj, method: str = "SMART", new_layer: str = "", island_margin: float = 0.02) -> bool:
    """The `CURIO` dispatch of `textures.ensure_target_uvs`. True when it handled the call (UVs made, or nothing to do);
    False hands the call back to Smart Project, with a message in `LAST["message"]`."""

    steps = ensure_target_uvs_steps(obj, method, new_layer, island_margin)
    while True:
        try:
            if next(steps) == uvnew.WAIT:
                time.sleep(0.0005)
        except StopIteration as end:
            return end.value


def ensure_target_uvs_steps(obj, method: str = "SMART", new_layer: str = "", island_margin: float = 0.02, stopped=lambda: False, tick=lambda fraction: None):
    """`ensure_target_uvs` as a generator for a timer: the unwrap yields between short steps (`uvnew.WAIT` while the
    faster core works); `tick(fraction)` gets the progress. `stopped()` True ends it early with False, so Smart Project
    makes the layer."""

    if not (method == "CURIO" or (method == "SMART" and MODE == "CURIO")):
        return False
    if obj.type != "MESH" or (obj.data.uv_layers and not new_layer):
        return True  # the old function would return here too
    mesh = obj.data
    started = time.perf_counter()
    resolution = _resolution[0]
    try:
        points, indptr, flat, material, forced = mesh_arrays(mesh)
        key = (obj.name, mesh.name, len(points), len(flat), resolution, float(points.sum()), forced.tobytes(), material.tobytes())  # sharp edges are cuts
        uv = _cache.get(key)
        if uv is None:
            settings = uvnew.Settings(texture_px=resolution)
            run = uvnew.NewUVs(points, indptr, flat, material if material.max(initial=0) > 0 else None, forced, settings)
            run._whole = True  # as run(): no first layout for a preview
            while not run.done:
                if stopped():
                    LAST.clear()
                    LAST["message"] = "New UVs were stopped. Smart Project was used."
                    return False
                tick(run.step())
                yield uvnew.WAIT if run.waiting else None
            if run.error is not None:
                raise run.error
            result = run.result
            uv = result.uv
            if not np.isfinite(uv).all():
                raise ValueError("the layout has points that are not numbers")
            _cache.clear()  # one remembered layout is enough: bake_detail asks again only while it lowers the margin
            _cache[key] = uv
            LAST.update(message="; ".join(result.messages), numbers=None, run=run, seconds=time.perf_counter() - started)
    except Exception as exc:  # any failure: Smart Project makes the layer, and the user is told why
        LAST.clear()
        LAST["message"] = f"New UVs could not be made ({exc}). Smart Project was used."
        return False
    from .. import uv_layers

    layer = None
    if new_layer:
        layer = mesh.uv_layers.new(name=new_layer)
        mesh.uv_layers.active = layer
        layer.active_render = True
    uv_layers.write_uv(mesh, uv, layer.name if layer is not None else None)
    sharpen_cuts(mesh, SHARPEN, SHADING_ANGLE)
    return True


def sharpen_cuts(mesh, mode: str, angle: float) -> int:
    """Sharp edges where the active UV map cuts: none (off), only where the dihedral angle is over `angle` (creased), or
    all. Returns how many edges were set. Replaces "sharpen every cut", which facets organic results."""

    mode = mode.lower()
    if mode == "off" or not len(mesh.polygons):
        return 0
    points, indptr, flat, _, _ = mesh_arrays(mesh)
    uv = np.empty(len(mesh.loops) * 2, np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2).astype(np.float64)
    cut = uvnew.cut_edges(indptr, flat, uv)
    ids, ne = uvnew.edge_ids(indptr, flat)
    edge_of = np.empty(len(mesh.loops), np.int64)
    mesh.loops.foreach_get("edge_index", edge_of)
    mesh_edge = np.full(ne, -1, np.int64)
    mesh_edge[ids] = edge_of
    pick = cut
    if mode != "all":
        pick = cut & (creased(points, indptr, flat, ids, ne) > angle)
    chosen = mesh_edge[pick]
    for e in chosen[chosen >= 0].tolist():
        mesh.edges[e].use_edge_sharp = True
    return int(len(chosen))


def creased(points, indptr, flat, ids, ne) -> np.ndarray:
    """Dihedral turning angle (radians) at every edge (0 for border edges)."""

    nf = len(indptr) - 1
    owner = np.repeat(np.arange(nf), np.diff(indptr))
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    cr = np.cross(points[flat], points[flat[nxt]])
    m = np.zeros((nf, 3))
    for k in range(3):
        m[:, k] = np.bincount(owner, weights=cr[:, k], minlength=nf)
    size = np.linalg.norm(m, axis=1)
    normal = np.divide(m, size[:, None], out=np.zeros_like(m), where=size[:, None] > 0)
    count = np.bincount(ids, minlength=ne)
    order = np.argsort(ids, kind="stable")
    first = np.concatenate([[0], np.cumsum(count)])[:-1]
    out = np.zeros(ne)
    two = np.flatnonzero(count == 2)
    if len(two):
        fa, fb = owner[order[first[two]]], owner[order[first[two] + 1]]
        out[two] = np.arccos(np.clip((normal[fa] * normal[fb]).sum(axis=1), -1.0, 1.0))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Seam step check after the bake (SPEC 8.5)
# ---------------------------------------------------------------------------------------------------------------------


def _bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """img (h, w, c), uv (n, 2) on the 0-1 square (texel centres at half integers), clamped at the border."""

    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy


def _mip(img: np.ndarray, k: int) -> np.ndarray:
    """Box-filtered image reduced by 2^k (sizes that do not divide are cropped)."""

    f = 1 << k
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    if f == 1 or h == 0 or w == 0:
        return img
    return img[:h, :w].reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))


def seam_step(points, indptr, flat, uv, normal_map=None, color_map=None, mip: int = 0, weights=None) -> dict:
    """Step across the cuts of a UV layout in baked maps: normal angle (degrees) and colour difference (8-bit steps),
    median and p95 over the cut length (or `weights`, per edge), and the three worst cuts.

    Maps are (h, w, 3 or 4) float arrays in 0..1, rows from the bottom (Blender's order). Returns a dict with
    `normal_p50`, `normal_p95`, `color_p50`, `color_p95`, `worst` [(edge, normal deg)] and `edges` (cuts measured)."""

    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    ids, ne = uvnew.edge_ids(indptr, flat)
    cut = uvnew.cut_edges(indptr, flat, uv)
    count = np.bincount(ids, minlength=ne)
    order = np.argsort(ids, kind="stable")
    first = np.concatenate([[0], np.cumsum(count)])[:-1]
    sel = np.flatnonzero(cut & (count == 2))
    out = {"normal_p50": float("nan"), "normal_p95": float("nan"), "color_p50": float("nan"), "color_p95": float("nan"), "worst": [], "edges": int(len(sel))}
    if len(sel) == 0:
        return out
    nxt = np.arange(1, len(flat) + 1)
    nxt[indptr[1:] - 1] = indptr[:-1]
    a, b = order[first[sel]], order[first[sel] + 1]
    same = flat[a] == flat[b]
    a0, a1 = uv[a], uv[nxt[a]]
    b0, b1 = np.where(same[:, None], uv[b], uv[nxt[b]]), np.where(same[:, None], uv[nxt[b]], uv[b])
    t = (np.arange(SAMPLES_PER_EDGE) + 0.5) / SAMPLES_PER_EDGE
    pa = (a0[:, None, :] * (1 - t)[None, :, None] + a1[:, None, :] * t[None, :, None]).reshape(-1, 2)
    pb = (b0[:, None, :] * (1 - t)[None, :, None] + b1[:, None, :] * t[None, :, None]).reshape(-1, 2)
    if weights is None:
        weights = np.linalg.norm(points[flat[nxt[a]]] - points[flat[a]], axis=1)
    else:
        weights = np.asarray(weights, dtype=np.float64)[sel]
    w = np.repeat(np.asarray(weights, dtype=np.float64), SAMPLES_PER_EDGE)
    w = np.where(w > 0, w, 1e-12)
    if normal_map is not None:
        img = _mip(np.asarray(normal_map, dtype=np.float64)[..., :3], mip)
        na, nb = _bilinear(img, pa) * 2 - 1, _bilinear(img, pb) * 2 - 1
        na /= np.maximum(np.linalg.norm(na, axis=1, keepdims=True), 1e-9)
        nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-9)
        ang = np.degrees(np.arccos(np.clip((na * nb).sum(axis=1), -1, 1)))
        out["normal_p50"], out["normal_p95"] = uvscore._wquantile(ang, w, 0.5), uvscore._wquantile(ang, w, 0.95)
        per_edge = ang.reshape(-1, SAMPLES_PER_EDGE).max(axis=1)
        worst = np.argsort(-per_edge)[:3]
        out["worst"] = [(int(sel[i]), float(per_edge[i])) for i in worst]
    if color_map is not None:
        img = _mip(np.asarray(color_map, dtype=np.float64)[..., :3], mip)
        diff = np.abs(_bilinear(img, pa) - _bilinear(img, pb)).max(axis=1) * 255.0
        out["color_p50"], out["color_p95"] = uvscore._wquantile(diff, w, 0.5), uvscore._wquantile(diff, w, 0.95)
    return out


def _image_array(image) -> np.ndarray:
    w, h = image.size
    flat = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(flat)
    return flat.reshape(h, w, 4)


def after_bake(obj, images) -> dict | None:
    """The one call `textures.bake_detail` makes after a bake: remembers the images for Ctrl+Z and runs the seam step
    check on the active UV map (images[0] is the normal map, images[1] the colour map when there is one). The numbers
    land in `LAST["seam_step"]`; None when the bake made no maps."""

    if not images:
        return None
    remember_images(_scene(), images)
    mesh = obj.data
    try:
        points, indptr, flat, _, _ = mesh_arrays(mesh)
        uv = np.empty(len(mesh.loops) * 2, np.float32)
        mesh.uv_layers.active.data.foreach_get("uv", uv)
        k = uvnew.mip_level(images[0].size[0])
        out = seam_step(points, indptr, flat, uv.reshape(-1, 2), _image_array(images[0]), _image_array(images[1]) if len(images) > 1 else None, mip=0)
        coarse = seam_step(points, indptr, flat, uv.reshape(-1, 2), _image_array(images[0]), None, mip=k)
        out["normal_p95_mip"] = coarse["normal_p95"]
    except Exception as exc:  # a check that cannot run never fails the bake
        out = {"error": str(exc)}
    LAST["seam_step"] = out
    return out


# What each image in the material holds, so Bake those spots bakes that data into it. Our own baked maps say it in their
# node label (CurioMesh_Normal ...); other images say it by the socket they feed.
_LABEL_BAKE = {"CurioMesh_BaseColor": "DIFFUSE", "CurioMesh_Normal": "NORMAL", "CurioMesh_Roughness": "ROUGHNESS",
               "CurioMesh_Emit": "EMIT"}
_SOCKET_BAKE = {"Normal": "NORMAL", "Roughness": "ROUGHNESS", "Emission": "EMIT", "Emission Color": "EMIT"}


def _fed_sockets(node, seen=None):
    """(node, socket) pairs an image node's Color output reaches, through reroutes."""
    seen = set() if seen is None else seen
    out = []
    for socket in getattr(node, "outputs", ()):
        if getattr(socket, "name", "") not in ("Color", "Output"):
            continue
        for link in getattr(socket, "links", ()):
            to = link.to_node
            if getattr(to, "type", "") == "REROUTE":
                if id(to) not in seen:
                    seen.add(id(to))
                    out += _fed_sockets(to, seen)
            else:
                out.append((to, link.to_socket))
    return out


def bake_type(image, nodes) -> str | None:
    """The bake type for `image` (one of its material's TEX_IMAGE nodes is among `nodes`): NORMAL only for a normal map
    (feeds a Normal Map node's Color or a Normal input), ROUGHNESS, EMIT, DIFFUSE for colour images. None for a data
    image no bake can make (metallic, AO, height, masks): it is left as it is."""

    for node in nodes:
        if getattr(node, "type", "") != "TEX_IMAGE" or node.image is not image:
            continue
        if getattr(node, "label", "") in _LABEL_BAKE:
            return _LABEL_BAKE[node.label]
        for to, socket in _fed_sockets(node):
            if getattr(to, "type", "") == "NORMAL_MAP" and socket.name == "Color":
                return "NORMAL"
            if getattr(to, "type", "") not in ("NORMAL_MAP", "BUMP", "DISPLACEMENT") and socket.name in _SOCKET_BAKE:
                return _SOCKET_BAKE[socket.name]
    if image.colorspace_settings.name != "Non-Color":
        return "DIFFUSE"
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Baked images follow Ctrl+Z
# ---------------------------------------------------------------------------------------------------------------------

# image name -> [(step, pixels (h, w, 4))] states, ascending: uint8 for 8-bit images, float32 for float (HDR) images so
# Ctrl+Z gives back values above 1 and below 0 exactly. The step is the scene's HISTORY_KEY counter, which
# Blender's own undo puts back; `restore()` then sets each image to the newest state whose step is not newer.
_states: dict = {}
_applied: dict = {}


def _scene():
    import bpy

    return bpy.context.scene


def _step(scene) -> int:
    return int(scene.get(HISTORY_KEY, 0))


def _pixels_state(image) -> np.ndarray:
    pixels = _image_array(image)
    if getattr(image, "is_float", False):
        return pixels.copy()
    return np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)


def remember_images(scene, images, before: bool = False) -> int:
    """Store the images' pixels as the state of a new step (`before` True: of the current step, call it BEFORE the
    pixels of existing images change, so undo can bring them back). Call it again after the change without `before`.
    Returns the step. The counter is a scene property, so Blender's undo and redo move it with everything else."""

    step = _step(scene)
    if not before:
        step += 1
        scene[HISTORY_KEY] = step
    for image in images:
        history = _states.setdefault(image.name, [])
        if before and any(s == step for s, _ in history):
            continue
        history[:] = [(s, p) for s, p in history if s != step]
        history.append((step, _pixels_state(image)))
        history.sort(key=lambda t: t[0])
        _applied[image.name] = step
    return step


def restore(scene) -> list:
    """Bring every remembered image to the state that belongs to the scene's step counter. Returns the names changed."""

    import bpy

    step, changed = _step(scene), []
    for name, history in _states.items():
        image = bpy.data.images.get(name)
        if image is None:
            continue
        older = [s for s, _ in history if s <= step]
        if not older:
            continue  # the image did not exist yet at this step; the undo removes it
        want = max(older)
        if _applied.get(name) == want:
            continue
        pixels = next(p for s, p in history if s == want)
        if tuple(image.size) != (pixels.shape[1], pixels.shape[0]):
            continue
        values = pixels.astype(np.float32) / 255.0 if pixels.dtype == np.uint8 else pixels
        image.pixels.foreach_set(values.reshape(-1))
        image.update()
        _applied[name] = want
        changed.append(name)
    return changed


def bring_back(scene, step: int) -> list:
    """A History jump to a step whose images had the states of counter `step`: a new counter step that holds those
    states (the same arrays, no copies; images not remembered by then keep what they show), so Ctrl+Z and Ctrl+Shift+Z
    of the jump move the pixels with it. Returns the names changed."""

    import bpy

    new = max([_step(scene)] + [s for history in _states.values() for s, _ in history]) + 1
    scene[HISTORY_KEY] = new
    changed = []
    for name, history in _states.items():
        older = [s for s, _ in history if s <= step]
        want = max(older) if older else _applied.get(name)
        pixels = next((p for s, p in history if s == want), None)
        if pixels is None:
            continue
        history.append((new, pixels))
        image = bpy.data.images.get(name)
        if image is not None and _applied.get(name) != want and tuple(image.size) == (pixels.shape[1], pixels.shape[0]):
            try:
                image.pixels.foreach_set((pixels.astype(np.float32) / 255.0).reshape(-1))
            except RuntimeError:  # an image with no pixel data now (unloaded): nothing to put back
                continue
            image.update()
            changed.append(name)
        _applied[name] = new
    return changed


def _on_undo(*_args) -> None:
    try:
        restore(_scene())
    except Exception:
        pass


def register() -> None:
    import bpy

    bpy.app.handlers.persistent(_on_undo)  # kept when another .blend is opened, like the other undo handlers
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _on_undo not in handlers:
            handlers.append(_on_undo)


def unregister() -> None:
    import bpy

    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        while _on_undo in handlers:
            handlers.remove(_on_undo)
