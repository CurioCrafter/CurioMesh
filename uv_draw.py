# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""What a UV tool's preview draws (package A3, SPEC 10.1): a checker through the candidate's UVs (32 texels a square),
the cuts in pink, the spots it found as dashed rings, a 180 px UV square inset, and the card with the verdict chip,
the deciding rule, the numbers before -> after, and Apply / Cancel / After-Before / Again / Keep going.

Batches are rebuilt only when the shown candidate changes, never per frame. Nothing here changes the mesh.
"""

from __future__ import annotations

import math

import numpy as np

from . import overlay, tidylive
from .triadq import uvnew, uvtrial

INSET_PX = 180.0
CHECKER_TEXELS = 32.0  # texels per checker square
LIFT = 0.004  # of the mesh's size: the checker sits just above the surface
EDGES_MOST = 200_000
LEVEL_COLOURS = {"GOOD": (0.243, 0.812, 0.431, 1.0), "CHECK": (1.0, 0.72, 0.16, 1.0), "BAD": (1.0, 0.353, 0.373, 1.0)}
GREY = (0.85, 0.87, 0.9, 1.0)

_shader = {"checker": None, "failed": False}


def _checker_shader():
    """A checker in the fragment shader: the squares stay sharp at any zoom. None where shaders cannot be made."""
    if _shader["checker"] is not None or _shader["failed"]:
        return _shader["checker"]
    try:
        import gpu

        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant("MAT4", "viewProjectionMatrix")
        info.push_constant("FLOAT", "squares")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_in(2, "VEC4", "tint")
        iface = gpu.types.GPUStageInterfaceInfo("curio_uv_checker")
        iface.smooth("VEC2", "vUV")
        iface.smooth("VEC4", "vTint")
        info.vertex_out(iface)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source("void main() { vUV = uv; vTint = tint; gl_Position = viewProjectionMatrix * vec4(pos, 1.0); }")
        info.fragment_source("void main() { float c = mod(floor(vUV.x * squares) + floor(vUV.y * squares), 2.0);"
                             " vec3 base = mix(vec3(0.25), vec3(0.8), c); fragColor = vec4(base * vTint.rgb, vTint.a); }")
        _shader["checker"] = gpu.shader.create_from_info(info)
    except Exception:
        _shader["failed"] = True
    return _shader["checker"]


def _tint(labels: np.ndarray) -> np.ndarray:
    """A light colour per island, so neighbouring islands read apart."""
    h = (labels.astype(np.float64) * 0.6180339887) % 1.0
    rgb = 0.75 + 0.25 * np.stack([np.cos(2 * math.pi * (h + k / 3.0)) for k in range(3)], axis=1)
    return np.concatenate([rgb, np.full((len(h), 1), 0.85)], axis=1).astype(np.float32)


def build(s) -> dict:
    """Batches of the shown UVs of session `s` (After: the candidate's, Before: the mesh's own)."""
    import gpu
    from gpu_extras.batch import batch_for_shader

    base = s.base
    uv = s.shown_uv()
    out = {}
    if uv is None:
        return out
    mw = np.asarray(s.obj.matrix_world, dtype=np.float64)
    size = float(np.ptp(base.points, axis=0).max()) if len(base.points) else 1.0
    normals = np.zeros_like(base.points)
    fan = uvtrial._fan(base.indptr, np.arange(base.faces))
    if len(fan):
        p = base.points[base.flat[fan]]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        for k in range(3):
            np.add.at(normals, base.flat[fan[:, k]], n)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    lifted = base.points + LIFT * size * normals
    world = lifted @ mw[:3, :3].T + mw[:3, 3]
    labels, _n = uvtrial.islands(base.indptr, base.flat, uv)
    shader = _checker_shader()
    if shader is not None and len(fan):
        corners = fan.reshape(-1)
        out["checker"] = (shader, batch_for_shader(shader, "TRIS", {"pos": world[base.flat[corners]].astype(np.float32), "uv": uv[corners].astype(np.float32),
                                                                     "tint": _tint(labels[base.corner_face[corners]])}))
    flat = gpu.shader.from_builtin("UNIFORM_COLOR")
    cut = uvnew.cut_edges(base.indptr, base.flat, uv)
    if cut.any():
        ids, _ne = uvnew.edge_ids(base.indptr, base.flat)
        nxt = np.arange(1, len(base.flat) + 1)
        nxt[base.indptr[1:] - 1] = base.indptr[:-1]
        first = np.unique(ids, return_index=True)[1]
        corner = first[cut][:EDGES_MOST]
        seg = np.stack([world[base.flat[corner]], world[base.flat[nxt[corner]]]], axis=1).reshape(-1, 3)
        out["cuts"] = (flat, batch_for_shader(flat, "LINES", {"pos": seg.astype(np.float32)}))
    cand = s.cand
    if cand is not None and cand.found is not None and len(cand.found) and not s.before:
        f = np.asarray(cand.found, np.float64) @ mw[:3, :3].T + mw[:3, 3]
        r = 0.02 * size
        ring = []
        for k in range(0, 16, 2):  # dashed: every other piece of a 16-piece circle, facing up the view's z
            a0, a1 = 2 * math.pi * k / 16, 2 * math.pi * (k + 1) / 16
            ring.append((math.cos(a0) * r, math.sin(a0) * r, math.cos(a1) * r, math.sin(a1) * r))
        ring = np.asarray(ring)
        seg = np.zeros((len(f), len(ring), 2, 3))
        seg[:, :, 0, 0], seg[:, :, 0, 1] = f[:, None, 0] + ring[None, :, 0], f[:, None, 1] + ring[None, :, 1]
        seg[:, :, 1, 0], seg[:, :, 1, 1] = f[:, None, 0] + ring[None, :, 2], f[:, None, 1] + ring[None, :, 3]
        seg[:, :, :, 2] = f[:, None, None, 2]
        out["found"] = (flat, batch_for_shader(flat, "LINES", {"pos": seg.reshape(-1, 3).astype(np.float32)}))
    # the UV square inset: every edge of the shown UVs, in 0..1
    nxt = np.arange(1, len(base.flat) + 1)
    nxt[base.indptr[1:] - 1] = base.indptr[:-1]
    ids_uv = np.stack([uv, uv[nxt]], axis=1)[:EDGES_MOST]
    out["inset"] = ids_uv.reshape(-1, 2).astype(np.float32)
    return out


def draw_view(s) -> None:
    """POST_VIEW: the checker, the cuts and the found rings."""
    import gpu
    from bpy import context

    b = s.batches()
    if not b:
        return
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.blend_set("ALPHA")
    if "checker" in b:
        shader, batch = b["checker"]
        shader.bind()
        shader.uniform_float("viewProjectionMatrix", context.region_data.perspective_matrix)
        shader.uniform_float("squares", float(s.base.texture_px) / CHECKER_TEXELS)
        batch.draw(shader)
    gpu.state.depth_test_set("NONE")
    for key, colour, width in (("cuts", tidylive.PINK, 3.0), ("found", tidylive.WHITE, 2.0)):
        if key in b:
            shader, batch = b[key]
            for c, w in ((tidylive.HALO, width + 2.0), (colour, width)):
                gpu.state.line_width_set(w)
                shader.bind()
                shader.uniform_float("color", c)
                batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


def _inset(shader, s, region, scale) -> None:
    from gpu_extras.batch import batch_for_shader

    b = s.batches()
    if "inset" not in b:
        return
    side = INSET_PX * scale
    x0 = region.width - overlay._sidebar_inset() - side - 12 * scale
    y0 = 12 * scale
    overlay._fill(shader, overlay._rounded_rect(x0 - 4 * scale, y0 - 4 * scale, side + 8 * scale, side + 8 * scale, 4 * scale), tidylive.CARD_BG)
    key = ("inset", id(s.shown_uv()), x0, y0, side)
    if s.inset_key != key:
        pts = b["inset"]
        inside = ((pts >= -1.0) & (pts <= 2.0)).all(axis=1).reshape(-1, 2).all(axis=1).repeat(2)  # tiles beside 0-1 are clipped, not wrapped
        p = pts[inside] * side + np.array([x0, y0], np.float32)
        s.inset_batch = batch_for_shader(shader, "LINES", {"pos": p})
        s.inset_key = key
    import gpu

    gpu.state.scissor_test_set(True)
    gpu.state.scissor_set(int(x0), int(y0), int(side), int(side))
    shader.bind()
    shader.uniform_float("color", (0.18, 0.77, 0.71, 0.9))
    s.inset_batch.draw(shader)
    gpu.state.scissor_test_set(False)
    shader.uniform_float("color", (0.9, 0.9, 0.95, 0.5))
    batch_for_shader(shader, "LINE_STRIP", {"pos": [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side), (x0, y0)]}).draw(shader)
    overlay._text(x0, y0 + side + 6 * scale, max(10, int(10 * scale)), GREY, "UV square" + ("  (before)" if s.before else ""))


def card_lines(s) -> tuple:
    """(title, verdict (level, title) or None, [(number words, sign)], note)."""
    t = s.trial
    title = f"{t.name}   {t.value_text(dict(s.settings, texture_px=s.base.texture_px))}   {s.status}"
    if s.error:
        return t.name, None, [], s.error
    c = s.cand
    if c is not None and c.refusal:
        return title, None, [], c.refusal
    verdict = (c.verdict[0], c.verdict[1]) if c is not None and c.verdict else None
    note = s.message or (c.note if c is not None else "")
    return title, verdict, uvtrial.card_numbers(t, c), note


def draw_pixel(s) -> None:
    """POST_PIXEL: the card and the UV square."""
    import gpu
    from bpy import context

    region = context.region
    if region is None or s.region_ptr != region.as_pointer():
        return
    scale = tidylive._ui_scale()
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    _inset(shader, s, region, scale)
    title, verdict, numbers, note = card_lines(s)
    w, h = 470 * scale, 118 * scale
    x, y = 16 * scale, 16 * scale
    overlay._fill(shader, overlay._rounded_rect(x, y, w, h, 6 * scale), tidylive.CARD_BG)
    overlay._fill(shader, overlay._rounded_rect(x, y, 5 * scale, h, 2 * scale), tidylive.TEAL)
    size = max(11, int(11 * scale))
    pad = 12 * scale
    top = y + h - 18 * scale
    chips = {}
    tw = overlay._text(x + pad, top, size, (1, 1, 1, 1), title)
    chips["value"] = (x + pad, top - 4 * scale, x + pad + tw, top + 14 * scale)  # drag the title's value to change it
    line = top - 20 * scale
    if verdict is not None:
        cw = tidylive._chip(shader, x + pad, line - 6 * scale, verdict[0], LEVEL_COLOURS.get(verdict[0], tidylive.SLATE), size, scale, 0.8)
        overlay._text(x + pad + cw + 8 * scale, line, size, GREY, verdict[1][:60])
    elif s.cand is not None and s.cand.refusal or s.error:
        overlay._text(x + pad, line, size, tidylive.HELD, (s.error or s.cand.refusal)[:72])
    line -= 20 * scale
    cx = x + pad
    for text, sign in numbers:
        colour = tidylive.BETTER if sign > 0 else tidylive.WORSE if sign < 0 else GREY
        cx += overlay._text(cx, line, size, colour, text) + 4 * scale
        if sign:
            tidylive._triangle(shader, cx + 4 * scale, line + 4 * scale, 4 * scale, sign > 0, colour)
            cx += 10 * scale
        cx += 10 * scale
    line -= 18 * scale
    if note:
        overlay._text(x + pad, line, size, GREY, note[:80])
    cx, cy = x + pad, y + 8 * scale
    going = bool(s.settings.get("keep_going"))
    for name, text, lit in (("apply", "Apply  Enter", tidylive.TEAL), ("cancel", "Cancel  Esc", tidylive.SLATE),
                            ("before", "Before  B" if s.before else "After  B", tidylive.SLATE if s.before else tidylive.TEAL),
                            ("again", "Again  A", tidylive.SLATE), ("keep", ("Stop  Esc" if s.going() else "Keep going  K"), tidylive.PINK if going else tidylive.SLATE)):
        cw = tidylive._chip(shader, cx, cy, text, lit, size, scale)
        chips[name] = (cx, cy, cx + cw, cy + 20 * scale)
        cx += cw + 6 * scale
    s.chips = chips
    gpu.state.blend_set("NONE")
