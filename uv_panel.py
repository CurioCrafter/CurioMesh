# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""The UVs row, its popover and the UV tools (package A3, SPEC 10.1). Main-only file (on the site build's delete list).

    | UVs   [ Auto ▾ ]   GOOD  2 % across seams   [ Look ]  [ ⋯ ] |

- `CurioUVSettings` (`scene.curiomesh_uv`) holds the popover's values and hands them to the cores (`sync`).
- Seam Fit is a Look First tool on the result (`tidylive.LiveTool`): preview, Apply = one Ctrl+Z step and one History
  entry; the result's own Apply then writes exactly the fit that was shown (`uvtrial.remember_applied`).
- New, New page layout, Cut here / Join here and Bake those spots change the UVs of a mesh that has been applied (the
  active mesh). They open the same way: a card with the verdict and numbers before -> after, Enter / Esc / B, Again (A),
  Keep going (K, Esc stops), one Ctrl+Z step and one History entry per Apply.
- Auto: Seam Fit and Bake those spots are Auto Fix passes (`triadq/uvtrial.py`), so a kept pass is a History step and
  Apply writes its fit. Nothing runs from draw().
"""

from __future__ import annotations

import math
import time

import bpy
import numpy as np

from . import history, overlay, tidylive, uv_draw, uv_layers
from .triadq import uvbake, uvkeep, uvnew, uvscore, uvtrial
from .translations import tip, tr

BUDGET = 0.008  # seconds of work per modal tick, like Look First
REST = 0.15  # seconds after the value stops moving: the run starts again
DRAG_PX = 300.0

MODES = (("AUTO", "Auto", "Keeps the UVs your mesh came with, or makes new ones when it has none"),
         ("KEEP", "Keep source", "The result's UVs sit where your mesh's UVs were. Seam Fit says how closely seams are followed"),
         ("NEW", "New", "Makes new UVs for the result: few seams, out of sight, straight rows"),
         ("OFF", "Off", "As before: no Seam Fit, and a bake unwraps with Smart Project"))
FITS = (("LIGHT", "Light", "Follows the seams that matter and adds few faces"),
        ("EXACT", "Exact", "Keeps every seam as a border. Can cost several times the faces"),
        ("OFF", "Off", "Seams are not followed; UVs are carried over side by side"))
ROWS = (("AUTO", "Auto", "Straight where it costs no stretch, else Soft"), ("ON", "On", "Every quad row on one line in the UVs"),
        ("SOFT", "Soft", "Curved parts may bend a little"), ("OFF", "Off", "Rows are not kept straight"))
SHARPEN = (("OFF", "Off", "No cut becomes a sharp edge"), ("CREASED", "Only creased", "Cuts where the surface turns become sharp edges"),
           ("ALL", "All", "Every cut becomes a sharp edge"))

live: dict = {"session": None, "handles": []}
_FIT_CACHE: dict = {"key": None, "value": None}


def _g():
    from . import guided  # late: guided imports panel, which imports this module

    return guided


def settings(scene=None):
    scene = scene or bpy.context.scene
    return getattr(scene, "curiomesh_uv", None)


def _gs(scene=None):
    scene = scene or bpy.context.scene
    return getattr(scene, "curiomesh_guided", None)


def _source_has_uvs() -> bool:
    try:
        source = _g().state_source()
    except Exception:
        return False
    return source is not None and bool(source.data.uv_layers)


def keep_mode(us, gs) -> str:
    """"light", "exact" or "off": what the search and Apply do with your mesh's UVs. The old Keep UV Seams switch still
    means what it meant: on = Exact."""
    if us is None:
        return "exact" if getattr(gs, "keep_uvs", False) else "off"
    if us.uvs in ("NEW", "OFF"):
        return "off"
    if us.uvs == "AUTO" and getattr(gs, "keep_uvs", False):
        return "exact"
    if us.uvs == "AUTO" and not _source_has_uvs():
        return "off"
    return {"LIGHT": "light", "EXACT": "exact", "OFF": "off"}[us.seam_fit]


def sync(scene=None) -> None:
    """Hand the popover's values to the cores. Cheap; called on every change and by the Auto timer."""
    us, gs = settings(scene), _gs(scene)
    if us is None:
        return
    uvkeep.OVERRIDE.update(mode=keep_mode(us, gs), allowance=float(us.extra_faces) / 100.0, split=bool(us.split_seams), texture_px=float(us.texture_size))
    uvbake.MODE = "SMART" if us.uvs == "OFF" else "CURIO"
    uvbake.SHARPEN = us.sharpen_cuts.lower()


def _sync(self, context):
    sync(context.scene)


def new_settings(us) -> uvnew.Settings:
    density = None if us.fill_page else max(float(us.pixel_density), 1e-6) * max(bpy.context.scene.unit_settings.scale_length, 1e-9)  # px per metre -> px per mesh unit
    return uvnew.Settings(stretch_allowed=float(us.stretch_allowed), rows=us.straight_rows.lower(), out_of_sight=bool(us.out_of_sight),
                          hidden_pct=float(us.hidden_parts), views=int(us.views), all_sides=bool(us.all_sides), texture_px=int(us.texture_size),
                          clean_px=int(us.clean_down_to), time_per_pass=float(us.time_per_pass), density=density)


class CurioUVSettings(bpy.types.PropertyGroup):
    uvs: bpy.props.EnumProperty(name="UVs", items=MODES, default="AUTO", update=_sync,
                                description="Auto keeps the UVs your mesh came with, or makes new ones when it has none")
    seam_fit: bpy.props.EnumProperty(name="Seam Fit", items=FITS, default="LIGHT", update=_sync,
                                     description="Light follows the seams that matter and adds few faces. Exact keeps every seam as a border and can cost several times the faces")
    extra_faces: bpy.props.FloatProperty(name="Extra faces for seams", default=100.0 * uvkeep.DEFAULT_ALLOWANCE, min=0.0, soft_max=100.0, precision=0, subtype="PERCENTAGE", update=_sync,
                                         description="Faces Seam Fit may add, compared with not fitting seams. What it can't fit is listed")
    split_seams: bpy.props.BoolProperty(name="Split faces on seams", default=True, update=_sync, description="Cuts a face along a seam when nothing else fits it")
    stretch_allowed: bpy.props.FloatProperty(name="Stretch allowed", default=10.0, min=0.1, soft_min=1.0, soft_max=25.0, precision=0, subtype="PERCENTAGE",
                                             description="How far the texture may stretch inside an island, in percent")
    straight_rows: bpy.props.EnumProperty(name="Straight rows", items=ROWS, default="AUTO",
                                          description="On keeps every quad row on one line in the UVs. Soft lets curved parts bend a little")
    out_of_sight: bpy.props.BoolProperty(name="Seams out of sight", default=True, description="Puts cuts where the camera check sees least")
    hidden_parts: bpy.props.FloatProperty(name="Hidden parts get", default=50.0, min=0.0, max=100.0, precision=0, subtype="PERCENTAGE",
                                          description="How many pixels hidden surfaces get, compared with seen ones")
    all_sides: bpy.props.BoolProperty(name="Seen from all sides", default=True, description="Off counts the underside less, for props that stand on something")
    views: bpy.props.IntProperty(name="Views", default=6, min=1, soft_max=26, description="How many directions the camera check looks from. Higher can take longer")
    time_per_pass: bpy.props.FloatProperty(name="Time per pass (s)", default=2.0, min=0.05, soft_max=30.0, precision=1,
                                           description="How long one pass may run before it shows what it has; Keep going continues. Higher can take longer")
    fill_page: bpy.props.BoolProperty(name="Fill the page", default=True, description="The islands get as many pixels as fit on the page")
    pixel_density: bpy.props.FloatProperty(name="Pixel density", default=512.0, min=0.001, soft_max=8192.0, description="Pixels per metre at the texture size")
    texture_size: bpy.props.IntProperty(name="Texture size", default=2048, min=16, soft_max=16384, max=65536, update=_sync,
                                        description="The texture these UVs are made for. Gap and pixel snapping use it")
    clean_down_to: bpy.props.IntProperty(name="Gap: clean down to (px)", default=uvnew.CLEAN_DOWN_TO_PX, min=1, soft_max=4096,
                                         description="The gap keeps islands apart down to this mip size")
    sharpen_cuts: bpy.props.EnumProperty(name="Sharp edges at cuts", items=SHARPEN, default="CREASED", update=_sync,
                                         description="Which cuts become sharp edges for the normal map")


# --------------------------------------------------------------------------- Seam Fit's inputs (from the result)


def _fit_inputs():
    """What Seam Fit works with on the result on screen, worked out once per result and source."""
    guided = _g()
    source, res = guided.state_source(), guided.state["res"]
    if source is None or res is None:
        return None
    gs = _gs()
    sync()
    s = uvkeep.settings_of(gs)
    key = (id(res), len(res["faces"]), source.name, repr(s))  # every setting the fit reads (Texture size too: Auto calls this without a reset)
    if _FIT_CACHE["key"] == key:
        return _FIT_CACHE["value"]
    if not source.data.uv_layers:
        value = None
    elif s.mode != "light":
        value = {"refusal": uvtrial.NOT_LIGHT}
    else:
        sa, plan = uvkeep.plan_for(source, guided._source_data(source), gs)
        value = None if sa is None else dict(source=sa, plan=plan, settings=s, material=res.get("mat"))
    _FIT_CACHE.update(key=key, value=value)
    return value


class CURIOMESH_OT_uv_seam_fit(tidylive.LiveTool, bpy.types.Operator):
    bl_idname = "curiomesh.uv_seam_fit"
    bl_label = "Seam Fit"
    bl_description = "Moves points onto your mesh's UV seams and splits faces along the seams that are left, previewed first. Use it when textures must stay where they were"
    bl_options = {"REGISTER", "UNDO"}
    trial = uvtrial.SeamFit
    faces: bpy.props.FloatProperty(name="Extra faces for seams", description="0..1 is 0..100 % extra faces; 1 splits every face on a seam", default=uvkeep.DEFAULT_ALLOWANCE, min=0.0, max=1.0)
    split: bpy.props.BoolProperty(name="Split faces on seams", default=True)
    keep_going: bpy.props.BoolProperty(name="Keep going", description="Passes repeat while fewer faces are left for baking", default=False)

    @classmethod
    def poll(cls, context):
        return _g().state["res"] is not None and not tidylive.busy()

    def invoke(self, context, event):
        us = settings(context.scene)
        if us is not None and not self.properties.is_property_set("faces"):
            self.faces = min(1.0, float(us.extra_faces) / 100.0)
        _FIT_CACHE["key"] = None
        return super().invoke(context, event)


# --------------------------------------------------------------------------- the UV-only tools on a mesh


def target(context):
    """The mesh a UV tool works on: the active mesh object, not the result preview."""
    obj = context.object
    if obj is None or obj.type != "MESH" or overlay.is_preview(obj) or not len(obj.data.polygons):
        return None
    return obj


def _source_mesh_for(obj):
    """Your original mesh as UV Check numbers need it, when it has UVs and is not `obj` itself."""
    try:
        source = _g().state_source()
    except Exception:
        return None
    if source is None or source == obj or not source.data.uv_layers:
        return None
    p, ip, fl, uv, mat = uv_layers.geometry(source.data)
    return uvscore.mesh(p, ip, fl, uv, mat)


def _bake_inputs():
    """Bake those spots in Auto Fix: the applied mesh on screen, when the result has been applied to it."""
    obj = target(bpy.context)
    if obj is None or not _g().state.get("applied") or not obj.data.uv_layers:
        return None
    points, indptr, flat, material, _forced = uvbake.mesh_arrays(obj.data)
    us = settings()
    return uvtrial.make_uv_base(points, indptr, flat, material if material.max(initial=0) > 0 else None, uv_layers.read_uv(obj.data),
                                float(us.texture_size) if us is not None else uvtrial.CHECK_PX, _source_mesh_for(obj))


class UVSession:
    """One open UV tool on `obj`: its work in slices, the shown candidate, the card's state."""

    def __init__(self, context, obj, kind: str, settings_: dict):
        self.obj, self.kind = obj, kind
        self.trial = uvtrial.TRIALS[kind]()
        self.settings = dict(self.trial.defaults, **settings_)
        mesh = obj.data
        points, indptr, flat, material, forced = uvbake.mesh_arrays(mesh)
        uv = uv_layers.read_uv(mesh)
        us = settings(context.scene)
        tex = float(us.texture_size) if us is not None else uvtrial.CHECK_PX
        self.base = uvtrial.make_uv_base(points, indptr, flat, material if material.max(initial=0) > 0 else None, uv, tex,
                                         _source_mesh_for(obj) if kind == "bake" else None, forced if len(forced) else None)
        self.cand = self.prev = None
        self.status, self.error, self.message = "getting ready", "", ""
        self.before = False
        self.chips: dict = {}
        self.drag = None
        self.moved_at = -1e9
        self.area = self.region = None
        self.region_ptr = 0
        self.longest = 0.0  # the longest slice so far, in seconds (the test reads it)
        self.passes = 0
        self._batch_key, self._batches = None, {}
        self.inset_key, self.inset_batch = None, None
        self.closed = False
        refusal = self.trial.check(self.base)
        self.work = None if refusal else self._run()
        if refusal:
            self.cand = uvtrial.UVCand(None, refusal=refusal)
            self.status = ""

    def _run(self):
        self.status = "working"
        cand = yield from self.trial.run(self.base, dict(self.settings), self.prev)
        self.cand = cand
        if cand.refusal:
            self.status = ""
            return
        self.status = "measuring"
        yield from uvtrial.measure(self.trial, self.base, cand)
        self.status = "ready"
        while self.settings.get("keep_going") and not cand.refusal:  # Keep going: again from the shown one while it gains
            self.status = f"keep going, pass {self.passes + 1}"
            nxt = yield from self.trial.run(self.base, dict(self.settings), cand)
            if nxt.refusal or nxt is cand:
                break
            yield from uvtrial.measure(self.trial, self.base, nxt)
            if not self.trial.gained(cand, nxt):
                break
            cand, self.passes = nxt, self.passes + 1
            self.cand = cand
        if self.settings.get("keep_going"):
            self.message = f"Keep going: {self.passes} more pass{'es' if self.passes != 1 else ''}, then nothing more to gain."
        self.status = "ready"

    def going(self) -> bool:
        return self.work is not None and bool(self.settings.get("keep_going")) and self.cand is not None and bool(self.cand.verdict)

    def shown_uv(self):
        if self.before or self.cand is None or self.cand.uv is None:
            return self.base.uv
        return self.cand.uv

    def batches(self) -> dict:
        key = (id(self.shown_uv()), self.before, id(self.cand))
        if key != self._batch_key:
            try:
                self._batches = uv_draw.build(self)
            except Exception as exc:  # drawing must never end the tool
                self._batches, self.message = {}, f"Preview not drawn: {exc}"
            self._batch_key = key
        return self._batches

    def pump(self, budget: float = BUDGET) -> bool:
        if self.closed:
            return False
        now = time.perf_counter()
        if self.work is None and self.drag is None and self.moved_at > 0 and now - self.moved_at >= REST:
            self.moved_at = -1e9
            self.work = self._run()
        if self.work is None:
            return False
        t0 = time.perf_counter()
        done, _ = uvtrial.drive(self.work, budget)
        self.longest = max(self.longest, time.perf_counter() - t0)
        if done:
            self.work = None
        return True

    def set_value(self, value: float) -> None:
        h = self.trial.handle
        if not h:
            return
        self.settings[h] = min(max(float(value), 0.0), 1.0)
        self.work, self.status, self.moved_at = None, "...", time.perf_counter()

    def again(self) -> None:
        """Again: the next run starts from the candidate on screen."""
        if self.cand is not None and self.cand.uv is not None and not self.cand.refusal:
            self.prev = self.cand
        self.work, self.message = self._run(), ""

    def toggle_keep_going(self) -> None:
        self.settings["keep_going"] = not self.settings.get("keep_going")
        if self.settings["keep_going"] and self.work is None:
            self.passes = 0
            self.prev = self.cand if self.cand is not None and self.cand.uv is not None else None
            self.work = self._run()

    def stop(self) -> bool:
        """Esc while Keep going works: stop at once and keep the best so far on screen. False when nothing was running."""
        if not self.going():
            return False
        self.work, self.settings["keep_going"] = None, False
        self.status, self.message = "ready", f"Stopped. Kept the best of {self.passes + 1} pass{'es' if self.passes else ''}."
        return True

    def ready(self) -> bool:
        c = self.cand
        return self.work is None and c is not None and c.uv is not None and not c.refusal and c.exact and bool(c.verdict)

    def finish(self) -> None:
        while self.work is not None:
            self.pump(1.0)


def numbers_text(s: UVSession) -> str:
    c = s.cand
    words = [w for w, _ in uvtrial.card_numbers(s.trial, c)]
    if c is not None and c.verdict:
        words.insert(0, c.verdict[0])
    return ", ".join(words)


def _pick_edge(s: UVSession, region, rv3d, x, y):
    """(a, b) vertex pair of the edge under the mouse, or None."""
    from bpy_extras import view3d_utils
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree

    base = s.base
    if getattr(s, "_tree", None) is None:
        faces = [tuple(base.flat[base.indptr[i]:base.indptr[i + 1]].tolist()) for i in range(base.faces)]
        s._tree = BVHTree.FromPolygons(base.points.tolist(), faces, all_triangles=False, epsilon=0.0)
    inv = Matrix(s.obj.matrix_world).inverted()
    origin = inv @ view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y))
    ray = (inv.to_3x3() @ view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))).normalized()
    loc, _n, face, _d = s._tree.ray_cast(Vector(origin), Vector(ray))
    if face is None:
        return None
    ring = base.flat[base.indptr[face]:base.indptr[face + 1]]
    p = np.asarray(loc)
    best, pair = math.inf, None
    for a, b in zip(ring, np.roll(ring, -1)):
        pa, pb = base.points[a], base.points[b]
        d = pb - pa
        t = float(np.clip(np.dot(p - pa, d) / max(np.dot(d, d), 1e-30), 0.0, 1.0))
        far = float(np.linalg.norm(pa + t * d - p))
        if far < best:
            best, pair = far, (int(a), int(b))
    return pair


def _images_of(obj) -> list:
    out = []
    for slot in obj.material_slots:
        m = slot.material
        if m is None or not m.use_nodes:
            continue
        for node in m.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None and node.image not in out:
                out.append(node.image)
    return out


def _raster_mask(uv, tris, w, h, grow: int) -> np.ndarray:
    """Pixels inside the triangles (UV corner rows of `tris`), grown by `grow` px: the new islands' pixels."""
    mask = np.zeros((h, w), bool)
    for t in tris:
        p = uv[t] * (w, h)
        x0, y0 = np.floor(p.min(axis=0)).astype(int) - grow
        x1, y1 = np.ceil(p.max(axis=0)).astype(int) + grow
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
        l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
        slack = grow / max(np.ptp(p, axis=0).max(), 1.0)
        inside = (l1 >= -slack) & (l2 >= -slack) & (1 - l1 - l2 >= -slack)
        mask[y0:y1 + 1, x0:x1 + 1] |= inside
    return mask


def bake_spots(context, obj, cand) -> str:
    """Write the candidate's UVs and bake your original mesh into just its new islands. "" or what went wrong."""
    from . import textures

    source = _g().state_source()
    images = _images_of(obj)
    if source is None or not images:
        return "Bake those spots needs a baked texture on this mesh and your original mesh: bake on Apply first."
    scene = context.scene
    uvbake.remember_images(scene, images, before=True)
    old = {im.name: np.asarray(im.pixels[:], np.float32) for im in images}
    old_uv = uv_layers.read_uv(obj.data)  # put back when the bake fails: new islands over an unbaked texture show garbage
    uv_layers.write_uv(obj.data, cand.uv)
    obj.data.update()
    margin = uvbake.gap_px(images[0].size[0])
    snap = textures._snapshot_bake_settings(scene) + [(scene.cycles, "device", scene.cycles.device)]  # set to CPU below: your render device comes back
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    baked = cand.extra.get("baked", cand.faces)  # every face given a new island, by all Keep going passes, not just the last
    tris = uvtrial._fan(np.asarray(cand.extra.get("indptr")), baked) if cand.extra.get("indptr") is not None else None
    try:
        for o in selected:
            o.select_set(False)
        source.select_set(True)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 16
        scene.render.bake.use_selected_to_active = True
        scene.render.bake.margin = margin
        for im in images:
            normal = im.colorspace_settings.name == "Non-Color"
            for slot in obj.material_slots:
                if slot.material is not None and slot.material.use_nodes:
                    for node in slot.material.node_tree.nodes:
                        if node.type == "TEX_IMAGE" and node.image == im:
                            slot.material.node_tree.nodes.active = node
            if not normal:
                scene.render.bake.use_pass_direct = False
                scene.render.bake.use_pass_indirect = False
                scene.render.bake.use_pass_color = True
            r = bpy.ops.object.bake(type="NORMAL" if normal else "DIFFUSE")
            if "FINISHED" not in r:
                raise RuntimeError(f"bake did not finish: {sorted(r)}")
            w, h = im.size
            new = np.asarray(im.pixels[:], np.float32).reshape(h, w, 4)
            if tris is not None:
                keep = _raster_mask(cand.uv, tris, w, h, margin)
                merged = np.where(keep[:, :, None], new, old[im.name].reshape(h, w, 4))
                im.pixels.foreach_set(merged.reshape(-1))
            im.update()
    except Exception as exc:
        for im in images:
            im.pixels.foreach_set(old[im.name])
        if old_uv is not None:
            uv_layers.write_uv(obj.data, old_uv)
            obj.data.update()
        return f"Bake those spots could not bake: {exc}"
    finally:
        textures._restore_properties(snap)
        for o in context.selected_objects:
            o.select_set(False)
        for o in selected:
            if o.name in bpy.data.objects:
                o.select_set(True)
        context.view_layer.objects.active = active
    uvbake.remember_images(scene, images)
    return ""


def _apply(context, s: UVSession) -> str:
    """Keep the shown candidate on the mesh. "" or why not."""
    c = s.cand
    if s.kind == "bake":
        c.extra["indptr"] = s.base.indptr
        err = bake_spots(context, s.obj, c)
        if err:
            return err
    else:
        uv_layers.write_uv(s.obj.data, c.uv)
        s.obj.data.update()
    history.before(context)
    history.record(context, f"UVs: {s.trial.name}", numbers_text(s))
    LAST.update(key=c.key, written=uv_layers_hash(s.obj.data), name=s.trial.name)
    return ""


LAST: dict = {}  # what the last UV Apply wrote: the previewed candidate's hash and the hash read back from the mesh


def uv_layers_hash(mesh) -> str:
    return uvtrial.uv_hash(uv_layers.read_uv(mesh))


def _add_handlers() -> None:
    if not live["handles"]:
        live["handles"] = [bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), "WINDOW", "POST_VIEW"),
                           bpy.types.SpaceView3D.draw_handler_add(_draw_pixel, (), "WINDOW", "POST_PIXEL")]


def _remove_handlers() -> None:
    while live["handles"]:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(live["handles"].pop(), "WINDOW")
        except Exception:
            pass


def _draw_view() -> None:
    s = live["session"]
    if s is not None:
        try:
            uv_draw.draw_view(s)
        except Exception as exc:
            overlay.LAST_DRAW_ERROR = f"UVs: {exc!r}"


def _draw_pixel() -> None:
    s = live["session"]
    if s is not None:
        try:
            uv_draw.draw_pixel(s)
        except Exception as exc:
            overlay.LAST_DRAW_ERROR = f"UVs: {exc!r}"


def close() -> None:
    s = live["session"]
    if s is not None:
        s.closed = True
        live["session"] = None
    overlay.tag_redraw()


KINDS = (("new", "New", "New UVs for this mesh"), ("layout", "New page layout", "Pack the islands as they are again"),
         ("cut", "Cut here", "Click edges to cut or join, then new UVs"), ("bake", "Bake those spots", "Give faces across seams their own islands and bake them"))


class CURIOMESH_OT_uv_tool(bpy.types.Operator):
    bl_idname = "curiomesh.uv_tool"
    bl_label = "UV tool"
    bl_description = "Shows the UV change on the mesh first. Enter keeps it, Esc cancels, B shows before and after"
    bl_options = {"REGISTER", "UNDO"}

    tool: bpy.props.EnumProperty(name="Tool", items=KINDS, default="new")
    value: bpy.props.FloatProperty(name="Value", description="The dial: Stretch allowed, Gap or faces to include", default=-1.0, min=-1.0, max=1.0)
    keep_going: bpy.props.BoolProperty(name="Keep going", description="Passes repeat while they gain; Esc stops", default=False)

    @classmethod
    def description(cls, context, props):
        return tip({"new": "Makes new UVs for the active mesh: few seams, out of sight, straight rows. Shown first, Enter keeps them",
                "layout": "Packs the islands as they are onto the page again, with the gap. Keep going tries other orders and keeps the best",
                "cut": "Click an edge to cut the UVs there, click it again to join it back. New UVs follow your cuts",
                "bake": "Gives each face that still crosses a seam its own small island and bakes your mesh into just those. The rest of the texture is untouched"}[props.tool])

    @classmethod
    def poll(cls, context):
        return target(context) is not None and not tidylive.busy()

    def _settings(self, context) -> dict:
        us = settings(context.scene)
        out = {"keep_going": bool(self.keep_going)}
        if us is not None:
            out["new"] = new_settings(us)
            out["time"] = float(us.time_per_pass)
        trial = uvtrial.TRIALS[self.tool]
        if self.value >= 0.0 and trial.handle:
            out[trial.handle] = float(self.value)
        elif trial.handle == "gap" and us is not None:  # the dial opens at the popover's "clean down to"
            sizes = [c for c in uvtrial.CLEAN if c <= us.texture_size] or [4]
            near = min(range(len(sizes)), key=lambda i: abs(math.log2(sizes[i]) - math.log2(max(us.clean_down_to, 1))))
            out["gap"] = near / max(len(sizes) - 1, 1)
        elif trial.handle == "stretch" and us is not None:
            out["stretch"] = min(1.0, float(us.stretch_allowed) / max(uvtrial.STRETCH_DIAL, float(us.stretch_allowed)))
        return out

    def _open(self, context):
        obj = target(context)
        if obj is None:
            return None, "Select a mesh first: UV tools work on the active mesh."
        self._edit = obj.mode == "EDIT"
        if self._edit:  # the mesh's arrays are read and written in Object Mode; Edit Mode comes back after
            bpy.ops.object.mode_set(mode="OBJECT")
        s = UVSession(context, obj, self.tool, self._settings(context))
        return s, s.error

    def _back_to_edit(self) -> None:
        if getattr(self, "_edit", False):
            try:
                bpy.ops.object.mode_set(mode="EDIT")
            except RuntimeError:
                pass
            self._edit = False

    def execute(self, context):
        """No preview: run to the end and keep it (F9, scripts)."""
        try:
            return self._execute(context)
        finally:
            self._back_to_edit()

    def _execute(self, context):
        s, err = self._open(context)
        if err or s is None:
            self.report({"WARNING"}, tr(err))
            return {"CANCELLED"}
        s.finish()
        if not s.ready():
            self.report({"WARNING"}, tr((s.cand.refusal if s.cand is not None else "") or f"{s.trial.name} did not finish"))
            return {"CANCELLED"}
        err = _apply(context, s)
        if err:
            self.report({"WARNING"}, tr(err))
            return {"CANCELLED"}
        return {"FINISHED"}

    def invoke(self, context, event):
        area, region = tidylive._view(context)
        if region is None:
            self.report({"WARNING"}, tr("Open a 3D view first: the preview is drawn there."))
            return {"CANCELLED"}
        close()
        tidylive.close("")
        s, err = self._open(context)
        if err or s is None:
            self.report({"WARNING"}, tr(err))
            return {"CANCELLED"}
        s.area, s.region, s.region_ptr = area, region, region.as_pointer()
        live["session"] = self._s = s
        _add_handlers()
        self._timer = context.window_manager.event_timer_add(1.0 / 60.0, window=context.window)
        hint = f"{s.trial.name}: " + ("click an edge to cut or join it. " if s.trial.picks else "") + "Enter keeps it, Esc cancels, B before/after, A again, K keep going."
        area.header_text_set(tr(hint))
        context.window_manager.modal_handler_add(self)
        overlay.tag_redraw()
        return {"RUNNING_MODAL"}

    def _finish(self, context) -> None:
        if live["session"] is self._s:
            live["session"] = None
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _remove_handlers()
        try:
            self._s.area.header_text_set(None)
        except Exception:
            pass
        self._back_to_edit()
        overlay.tag_redraw()

    def cancel(self, context):
        close()
        self._finish(context)

    def _keep(self, context):
        s = self._s
        err = _apply(context, s)
        self.keep_going = bool(s.settings.get("keep_going"))
        if s.trial.handle:
            self.value = float(s.settings.get(s.trial.handle, self.value))
        close()
        self._finish(context)
        if err:
            self.report({"WARNING"}, tr(err))
            return {"CANCELLED"}
        self.report({"INFO"}, tr(f"{s.trial.name}: {numbers_text(s)}"))
        return {"FINISHED"}

    def modal(self, context, event):
        s = self._s
        if s.closed or live["session"] is not s:
            self._finish(context)
            return {"CANCELLED"}
        if event.type == "TIMER":
            if s.pump(BUDGET):
                s.area.tag_redraw()
            if getattr(s, "applying", False) and s.ready():
                return self._keep(context)
            return {"PASS_THROUGH"}
        r = s.region
        x, y = event.mouse_x - r.x, event.mouse_y - r.y
        inside = 0 <= x < r.width and 0 <= y < r.height
        if s.drag is not None:
            if event.type == "MOUSEMOVE":
                s.set_value(s.drag[1] + (x - s.drag[0]) / (DRAG_PX * tidylive._ui_scale()) * (0.1 if event.shift else 1.0))
                s.moved_at = time.perf_counter()
                s.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                s.drag, s.moved_at = None, time.perf_counter()
                return {"RUNNING_MODAL"}
        if event.value != "PRESS":
            return {"PASS_THROUGH"} if not inside or event.type in tidylive.NAVIGATION or event.type == "MOUSEMOVE" else {"RUNNING_MODAL"}
        if event.type in {"RET", "NUMPAD_ENTER"}:
            s.applying = True
            return {"RUNNING_MODAL"}
        if event.type in {"ESC", "RIGHTMOUSE"} or (event.type == "Z" and (event.ctrl or event.oskey)):
            if event.type == "ESC" and s.stop():
                s.area.tag_redraw()
                return {"RUNNING_MODAL"}
            close()
            self._finish(context)
            return {"CANCELLED"}
        if event.type == "B":
            s.before = not s.before
            s.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "A" and not (event.ctrl or event.shift or event.alt):
            s.again()
            return {"RUNNING_MODAL"}
        if event.type == "K":
            s.toggle_keep_going()
            s.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if not inside or event.type in tidylive.NAVIGATION or event.type.startswith(("NDOF", "NUMPAD")):
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE":
            for name, (x0, y0, x1, y1) in s.chips.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if name == "apply":
                        s.applying = True
                    elif name == "cancel":
                        close()
                        self._finish(context)
                        return {"CANCELLED"}
                    elif name == "before":
                        s.before = not s.before
                    elif name == "again":
                        s.again()
                    elif name == "keep":
                        if not s.stop():
                            s.toggle_keep_going()
                    elif name == "value" and s.trial.handle:
                        s.drag = (x, float(s.settings.get(s.trial.handle, 0.0)))
                    s.area.tag_redraw()
                    return {"RUNNING_MODAL"}
            if s.trial.picks:
                pair = _pick_edge(s, r, r.data, x, y)
                if pair is not None:
                    s.settings["cuts"] = uvtrial.toggle_cut(tuple(s.settings.get("cuts", ())), *pair)
                    s.prev, s.moved_at, s.work = None, time.perf_counter(), None
                    s.message = f"{len(s.settings['cuts'])} edge{'s' if len(s.settings['cuts']) != 1 else ''} cut by you"
                    s.area.tag_redraw()
            return {"RUNNING_MODAL"}
        return {"RUNNING_MODAL"}


# --------------------------------------------------------------------------- Look, Auto, the row and the popover


class CURIOMESH_OT_uv_look(bpy.types.Operator):
    bl_idname = "curiomesh.uv_look"
    bl_label = "Look"
    bl_description = "Opens the UV tool the UVs choice implies: Seam Fit on the result while UVs keep your mesh's, New on an applied mesh"

    @classmethod
    def poll(cls, context):
        return not tidylive.busy() and live["session"] is None

    def invoke(self, context, event):
        us, gs = settings(context.scene), _gs(context.scene)
        sync(context.scene)
        mode = keep_mode(us, gs)
        if _g().state["res"] is not None:
            if mode == "light":
                return bpy.ops.curiomesh.uv_seam_fit("INVOKE_DEFAULT")
            if us is not None and (us.uvs == "NEW" or (us.uvs == "AUTO" and not _source_has_uvs())):
                self.report({"INFO"}, tr("New UVs are made on Apply. After Apply, Look shows them on the mesh."))
                return {"CANCELLED"}
            if mode == "exact":
                self.report({"INFO"}, tr("Seam Fit: Exact keeps every seam as a border during the search, so there is nothing to fit after it."))
                return {"CANCELLED"}
            self.report({"INFO"}, tr("UVs are Off: nothing to look at."))
            return {"CANCELLED"}
        if target(context) is None:
            self.report({"WARNING"}, tr("Make a result first with Start, or select a mesh."))
            return {"CANCELLED"}
        return bpy.ops.curiomesh.uv_tool("INVOKE_DEFAULT", tool="new" if not context.object.data.uv_layers else "layout")


def draw_row(layout, context) -> None:
    """The UVs row (one call from panel.py, beside Bake): mode, verdict, one number, Look and the popover."""
    us = settings(context.scene)
    if us is None:
        return
    row = layout.row(align=True)
    row.prop(us, "uvs", text="UVs")
    row.operator("curiomesh.uv_look", text="Look", icon="HIDE_OFF")
    row.popover(panel="CURIOMESH_PT_pop_uv", text="", icon="THREE_DOTS")


class CURIOMESH_PT_pop_uv(bpy.types.Panel):
    bl_label = "UVs"
    bl_idname = "CURIOMESH_PT_pop_uv"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 15

    def draw(self, context):
        layout, us, gs = self.layout, settings(context.scene), _gs(context.scene)
        if us is None:
            return
        layout.prop(us, "uvs", expand=True)
        mode = keep_mode(us, gs)
        keep = us.uvs in ("AUTO", "KEEP") and mode != "off" or us.uvs == "KEEP"
        new = us.uvs == "NEW" or (us.uvs == "AUTO" and not _source_has_uvs())
        col = layout.column(align=True)
        if keep:
            col.prop(us, "seam_fit", expand=True)
            if getattr(gs, "keep_uvs", False) and us.uvs == "AUTO":
                overlay.wrap_label(col, "Keep UV Seams is on, so Auto keeps every seam (Exact).", icon="INFO", width=36, limit=2)
            sub = col.column(align=True)
            sub.active = us.seam_fit == "LIGHT"
            sub.prop(us, "extra_faces")
            sub.prop(us, "split_seams")
            r = col.row(align=True)
            r.enabled = _g().state["res"] is not None
            r.operator("curiomesh.uv_seam_fit", text="Seam Fit", icon="HIDE_OFF")
            op = r.operator("curiomesh.uv_seam_fit", text="", icon="SHADERFX")  # fix every spot like this: no face budget
            op.faces = 1.0
        if new or us.uvs == "NEW":
            col = layout.column(align=True)
            col.prop(us, "stretch_allowed")
            col.prop(us, "straight_rows")
            col.prop(us, "out_of_sight")
            col.prop(us, "hidden_parts")
        col = layout.column(align=True)
        col.prop(us, "all_sides")
        col.prop(us, "views")
        col.prop(us, "time_per_pass")
        col = layout.column(align=True)
        col.prop(us, "fill_page")
        sub = col.row(align=True)  # always drawn (greyed while Fill the page is on): the rows below never jump
        sub.active = not us.fill_page
        sub.prop(us, "pixel_density")
        col.prop(us, "texture_size")
        col.prop(us, "clean_down_to")
        col.label(text=tr(f"Gap {uvnew.gap_texels(us.texture_size, us.clean_down_to)} px: clean down to {us.clean_down_to} px ({uvnew.mip_level(us.texture_size, us.clean_down_to)} levels)"))
        layout.prop(us, "sharpen_cuts")
        layout.separator()
        layout.label(text="On the active mesh:")
        grid = layout.grid_flow(columns=2, align=True)
        for kind, words, _tip in KINDS:
            grid.operator("curiomesh.uv_tool", text=tr(words)).tool = kind
        op = layout.operator("curiomesh.uv_tool", text="Bake every spot", icon="SHADERFX")  # fix every spot like this
        op.tool, op.value, op.keep_going = "bake", 1.0, True




CLASSES = (CurioUVSettings, CURIOMESH_OT_uv_seam_fit, CURIOMESH_OT_uv_tool, CURIOMESH_OT_uv_look, CURIOMESH_PT_pop_uv)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.curiomesh_uv = bpy.props.PointerProperty(type=CurioUVSettings)
    uvtrial.FIT["inputs"] = _fit_inputs
    uvtrial.BAKE.update(inputs=_bake_inputs, write=lambda obj, c: bake_spots(bpy.context, obj, c))
    try:
        from . import uv_fast

        uv_fast.install()
    except ImportError:
        pass
    uvbake.register()


def unregister() -> None:
    close()
    _remove_handlers()
    uvbake.unregister()
    uvtrial.FIT["inputs"] = None
    uvtrial.BAKE.update(inputs=None, write=None)
    if hasattr(bpy.types.Scene, "curiomesh_uv"):
        del bpy.types.Scene.curiomesh_uv
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
