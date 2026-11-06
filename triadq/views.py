# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""What the camera counts: which pixels of a picture changed, and which pixels show a given part at all.

A rough surface (a scanned rock, a sea floor) shades as noise, so one pixel's normal says little: any change of the mesh
turns most of them, and that's not what a viewer sees change. With `rough` on, where the source's normals spread widely
over a small block of pixels, the block's average direction is compared instead. Smooth surfaces stay pixel by pixel.
"""

from __future__ import annotations

import math

import numpy as np

TURNED_DEG = 20.0
DEPTH_PIXELS = 2.0  # a surface may move this many pixels towards or away from the camera
ROUGH_BLOCK = 4  # pixels per side of a block
ROUGH_SPREAD = 0.94  # a block is rough if the average of its unit normals is shorter than this: about 20 degrees of spread
SIDES = tuple((x, y, z) for x, y, z in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))) + tuple((x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1))
MOST_STEPS = 64  # per side of one look at a part, so a large part can't take minutes


def _block_sum(values: np.ndarray) -> np.ndarray:
    high, wide = values.shape[0] // ROUGH_BLOCK, values.shape[1] // ROUGH_BLOCK
    return values[: high * ROUGH_BLOCK, : wide * ROUGH_BLOCK].reshape((high, ROUGH_BLOCK, wide, ROUGH_BLOCK) + values.shape[2:]).sum(axis=(1, 3))


def _per_pixel(blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    high, wide = blocks.shape
    out[: high * ROUGH_BLOCK, : wide * ROUGH_BLOCK] = np.repeat(np.repeat(blocks, ROUGH_BLOCK, axis=0), ROUGH_BLOCK, axis=1)
    return out


def surface_changed(hit_a, normal_a, depth_a, hit_b, normal_b, depth_b, pixel: float, rough: bool = False) -> np.ndarray:
    """The pixels showing both meshes where the surface has turned or moved. A is the source."""

    both = hit_a & hit_b
    facing = (normal_a * normal_b).sum(axis=2)
    turned = np.abs(facing) < math.cos(math.radians(TURNED_DEG))  # winding isn't shape
    if rough:
        count = _block_sum(hit_a.astype(np.float64))
        spread = np.linalg.norm(_block_sum(normal_a * hit_a[..., None]), axis=2) < ROUGH_SPREAD * count
        is_rough = _per_pixel(spread & (count > 0), both.shape)
        same_way = normal_b * np.where(facing < 0.0, -1.0, 1.0)[..., None]
        sum_a, sum_b = _block_sum(normal_a * both[..., None]), _block_sum(same_way * both[..., None])
        apart = (sum_a * sum_b).sum(axis=2) < math.cos(math.radians(TURNED_DEG)) * np.linalg.norm(sum_a, axis=2) * np.linalg.norm(sum_b, axis=2)
        turned = np.where(is_rough, _per_pixel(apart, both.shape), turned)
    return both & (turned | (np.abs(depth_a - depth_b) > DEPTH_PIXELS * pixel))


def picture_counts(reference, other, pixel: float, rough: bool = False) -> tuple[int, int]:
    """(pixels a viewer would see as different in outline, direction or depth, pixels showing the object)."""

    wrong, total = 0, 0
    for (hit_a, normal_a, depth_a, _face_a), (hit_b, normal_b, depth_b, _face_b) in zip(reference, other):
        wrong += int(surface_changed(hit_a, normal_a, depth_a, hit_b, normal_b, depth_b, pixel, rough).sum()) + int((hit_a ^ hit_b).sum())
        total += int((hit_a | hit_b).sum())
    return wrong, total


def pixels_seen(cast, labels, low, high, pixel: float, want: int, reach: float, enough: int | None = None) -> int:
    """Pixels of part `want` a camera sees from the fourteen sides and corners, with everything else in the way.

    `cast(origin, direction)` gives the index of the first face a ray hits, or None; `labels` the part of each face. `reach`
    is how far outside the object the camera stands. Counting stops at `enough`. Over MOST_STEPS a side the pixels are
    sampled more thinly and each counts for what it stands for.
    """

    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    middle = (low + high) / 2.0
    half = float(np.linalg.norm(high - low)) / 2.0 + pixel
    step = max(pixel, 2.0 * half / MOST_STEPS)
    worth = (step / pixel) ** 2
    across = (np.arange(-half + step / 2.0, half, step)).tolist()
    seen = 0.0
    for side in SIDES:
        view = np.asarray(side, dtype=np.float64)
        view /= np.linalg.norm(view)
        right = np.cross(view, (0.0, 0.0, 1.0) if abs(view[2]) < 0.9 else (1.0, 0.0, 0.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, view)
        start = (middle + view * reach).tolist()
        direction = (-view).tolist()
        right, up = right.tolist(), up.tolist()
        for x in across:
            line = (start[0] + right[0] * x, start[1] + right[1] * x, start[2] + right[2] * x)
            for y in across:
                found = cast((line[0] + up[0] * y, line[1] + up[1] * y, line[2] + up[2] * y), direction)
                if found is not None and labels[found] == want:
                    seen += worth
                    if enough is not None and seen >= enough:
                        return int(seen)
    return int(seen)


__all__ = ["picture_counts", "pixels_seen", "surface_changed"]
