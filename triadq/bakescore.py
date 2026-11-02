# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""What a normal map can't hide: judging a low mesh as it will look after its bake.

A bake repaints the normals but moves nothing. So fine bumps don't count; what counts is an outline that has moved, a
surface far from where it was (the map's detail then slides as the view turns), and shading that is still wrong over a
few pixels with the source's normals painted on. The pictures passed in hold, for the low mesh, the source's normal
nearest each hit, which is about what the bake would paint there.
"""

from __future__ import annotations

import math

import numpy as np

DEPTH_PIXELS = 3.0  # a surface may sit this far from the source before the map's detail visibly slides
OUTLINE_PIXELS = 1  # an outline may move this far
BLOCK = 4  # shading is compared over squares of this many pixels: what a shift of fine bumps does cancels out in them
TURNED_DEG = 30.0
OUTLINE_WEIGHT = 0.5  # every outline pixel moved: half the view counts as changed, as thin parts show in few pixels


def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    """The mask, widened by this many pixels."""

    out = mask.copy()
    for _ in range(pixels):
        wider = out.copy()
        wider[1:] |= out[:-1]
        wider[:-1] |= out[1:]
        wider[:, 1:] |= out[:, :-1]
        wider[:, :-1] |= out[:, 1:]
        out = wider
    return out


def outline_moved(hit_a: np.ndarray, hit_b: np.ndarray) -> tuple[int, int]:
    """(pixels outside the other picture's outline by more than the limit, pixels on the first picture's outline)."""

    moved = (hit_a & ~grow(hit_b, OUTLINE_PIXELS)) | (hit_b & ~grow(hit_a, OUTLINE_PIXELS))
    return int(moved.sum()), int((hit_a & grow(~hit_a, 1)).sum())


def _blocks(values: np.ndarray) -> np.ndarray:
    high, wide = values.shape[0] // BLOCK, values.shape[1] // BLOCK
    return values[: high * BLOCK, : wide * BLOCK].reshape((high, BLOCK, wide, BLOCK) + values.shape[2:]).sum(axis=(1, 3))


def surface_changed(hit_a, normal_a, depth_a, hit_b, normal_b, depth_b, pixel: float) -> np.ndarray:
    """The pixels showing both meshes that still look different after the bake."""

    both = hit_a & hit_b
    changed = both & (np.abs(depth_a - depth_b) > DEPTH_PIXELS * pixel)
    a = normal_a * both[..., None]
    b = normal_b * both[..., None]
    b = b * np.where((a * b).sum(axis=2, keepdims=True) < 0.0, -1.0, 1.0)  # winding isn't shape
    sum_a, sum_b = _blocks(a), _blocks(b)
    size = np.linalg.norm(sum_a, axis=2) * np.linalg.norm(sum_b, axis=2)
    turned = (sum_a * sum_b).sum(axis=2) < math.cos(math.radians(TURNED_DEG)) * size
    high, wide = turned.shape
    changed[: high * BLOCK, : wide * BLOCK] |= both[: high * BLOCK, : wide * BLOCK] & np.repeat(np.repeat(turned, BLOCK, axis=0), BLOCK, axis=1)
    return changed


def picture_counts(reference, view, pixel: float) -> tuple[int, int]:
    """(pixels that count as changed after a bake, pixels showing the object), over all the pictures."""

    wrong, total, moved, outline = 0, 0, 0, 0
    for (hit_a, normal_a, depth_a, _face_a), (hit_b, normal_b, depth_b, _face_b) in zip(reference, view):
        wrong += int(surface_changed(hit_a, normal_a, depth_a, hit_b, normal_b, depth_b, pixel).sum())
        total += int((hit_a | hit_b).sum())
        off, line = outline_moved(hit_a, hit_b)
        moved += off
        outline += line
    return wrong + int(OUTLINE_WEIGHT * total * moved / max(outline, 1)), total
