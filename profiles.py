# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Auto Segments presets by use. A preset only sets Detail, Method and Output; every one stays editable (that is Custom)."""

from __future__ import annotations

import ctypes
import math
import os

METHOD_ITEMS = [
    ("AUTO", "Auto", "The camera decides how many faces each part keeps, then pairs triangles into quads and closes the rest", "CAMERA_DATA", 0),
    ("PAIRING", "Pairing", "Pairs triangles into quads from more starting points, so quads come out more even. Slower, a few more faces", "MOD_LATTICE", 1),
    ("FLAT", "Flat Quads", "Flat areas become one big quad each, then the rest is paired. Fewest faces on hard-surface and architecture", "MESH_PLANE", 2),
    ("QUADRIFLOW", "QuadriFlow", "Blender's own QuadriFlow on each part. Even flowing quads on smooth organic parts; it can fail on thin or open parts (Auto is used then)", "MOD_REMESH", 3),
    ("GRID", "Grid Fill", "A regular grid. Needs a part with one open edge loop; a closed part falls back to Auto", "MESH_GRID", 4),
    ("FLOW", "Flow Fill", "Quad loops that follow the surface. Needs a part with one open edge loop; a closed part falls back to Auto", "MOD_WAVE", 5),
    ("FIELD", "Field Rows", "Lays long rows of quads over each part from a direction field, along its long creases. Most organized, with more shape change on busy parts. Parts it can't do well use Auto. Up to 30,000 quads per part", "OUTLINER_OB_LATTICE", 7),
    ("TRIS", "Tris Only", "No pairing at all: the reduced triangles as they are. About twice the faces of the same shape in quads", "MESH_DATA", 6),
]
OUTPUT_ITEMS = [
    ("TRIS", "Triangles", "Every face is a triangle", "MESH_DATA", 0),
    ("QUADS", "Quads", "All quads where the shape allows it (extra faces are added to close the rest)", "MESH_GRID", 1),
    ("MIXED", "Mixed", "Keep the paired quads and leave the leftover triangles: no closing, so no extra faces", "MESH_PLANE", 2),
]
# key: (label, tooltip, detail %, method, output). Detail is how much of the view may change: lower keeps more.
PROFILES = {
    "LOW_POLY": ("Low Poly", "Fewest faces for a game prop or background: flat areas collapse into big polygons, then triangles. Small parts may vanish", 10.0, "FLAT", "TRIS"),
    "GAME": ("Game Asset", "All quads, few faces, good for a game model. The default", 6.0, "AUTO", "QUADS"),
    "ANIMATION": ("Animation", "All quads, even sizes and more faces, so the mesh bends cleanly. Check eyes, mouth and joints by hand", 1.5, "QUADRIFLOW", "QUADS"),
    "SUBDIV": ("Subdivision", "All quads with flat panels kept as single large quads, so smoothing keeps the shape", 1.5, "FLAT", "QUADS"),
    "SCAN": ("Scan Cleanup", "Keeps more of the detail of a dirty scan (scans have few true flats, so no flat merging)", 2.0, "AUTO", "QUADS"),
    "SCULPT": ("Sculpt Base", "Even, smooth quads to sculpt on: smoothness over detail", 6.0, "PAIRING", "QUADS"),
}
PROFILE_ITEMS = [(k, v[0], v[1]) for k, v in PROFILES.items()] + [("CUSTOM", "Custom", "Your own Detail, Method and Output")]


MULTIRES_USES = {"ANIMATION", "SUBDIV", "SCULPT"}  # the uses where a clean base matters more than its own shape
MOST_FACES_PER_GB = 1_000_000  # of the top Multires level: its grids and the mesh shown take about this much memory
FEWEST_CAGE, MOST_CAGE = 1500, 12000


def cage_faces(source_faces: int) -> int:
    """Faces for a clean Multires cage: a quarter of the source, within what bends and draws well."""
    return int(min(max(source_faces // 4, FEWEST_CAGE), MOST_CAGE))


def top_faces(base_faces: int, levels: int) -> int:
    return int(base_faces) * 4 ** int(levels)


def multires_words(base_faces: int, levels: int) -> str:
    """The plain summary under the Multires Detail switch."""
    return (
        f"About {base_faces:,} clean quads at level 0, up to about {top_faces(base_faces, levels):,} faces at level {levels}. "
        "Level 0 keeps its own rough shape and is what you animate; the detail is in the levels above. "
        "A normal map only changes the shading at level 0, never the outline."
    )


def free_memory_gb() -> float:
    """Free memory on this computer, or a modest guess when it can't be read."""
    try:
        if os.name == "nt":
            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("total", "free", "pagefile", "free_pagefile", "virtual", "free_virtual", "extended")]
            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.free / 2**30
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (AttributeError, OSError, ValueError):
        return 8.0


def levels_for(base_faces: int, free_gb: float, source_faces: int = 10**9) -> int:
    """Multires levels this computer holds, at least 1. Past about the source's own faces a level only copies its flat
    triangles (measured: from there the shading stops getting closer), so one level above that is the most."""
    room = min(max(free_gb - 1.0, 0.5) * MOST_FACES_PER_GB, 4.0 * source_faces) / max(base_faces, 1)
    return int(max(1, math.floor(math.log(max(room, 1.0), 4.0))))


def choose(facts: dict, free_gb: float) -> dict:
    """What the Auto button picks from `detail.mesh_facts` and the free memory: a Use, Multires levels (0 is off), a bake and why."""
    faces, parts = facts["faces"], facts["parts"]
    many_parts = parts >= 20 or facts["biggest_part"] < 0.3
    bake_size = "2048" if free_gb >= 4.0 else "1024"
    size = f"{faces:,} faces in {parts:,} loose part{'s' if parts != 1 else ''}"
    if many_parts:
        return {"use": "GAME", "levels": 0, "bake": "OFF", "bake_size": bake_size, "kind": "assembly", "why": f"{size}: an assembly. Game Asset, part by part. No Multires: its levels would round hard edges and many small parts gain little."}
    if facts["flat"] >= 0.2:
        return {"use": "GAME", "levels": 0, "bake": "OFF", "bake_size": bake_size, "kind": "hard", "why": f"{size}, {100.0 * facts['flat']:.0f}% of its edges flat: a hard-surface model. Game Asset. No Multires: its levels would round the hard edges."}
    levels = levels_for(cage_faces(faces), free_gb, faces)
    room = f"{levels} Multires level{'s' if levels != 1 else ''}: past that your mesh has no more to give, or the {free_gb:.0f} GB free runs out"
    if facts["flat"] < 0.06:
        return {"use": "SCAN", "levels": 0, "bake": "NORMAL", "bake_size": bake_size, "kind": "scan", "why": f"{size}, almost no flat edges: a scan. Scan Cleanup keeps its surface, and a baked normal map holds what is left."}
    return {"use": "ANIMATION", "levels": levels, "bake": "OFF", "bake_size": bake_size, "kind": "organic", "why": f"{size}, smooth with few flats: an organic shape. Animation with Multires Detail: clean quads at level 0, the shape back in the levels. {room}."}


def preset(key: str):
    """(detail, method, output) for a profile, or None for Custom."""
    p = PROFILES.get(key)
    return None if p is None else (p[2], p[3], p[4])


def detail_words(detail: float) -> str:
    """What a Detail value means, in plain words."""
    for limit, words in (
        (0.5, "Almost nothing may change: very many faces"),
        (2.0, "High detail: many faces, small bumps are kept"),
        (5.0, "Good detail: fine features are kept, thin parts survive"),
        (10.0, "Shape kept; only the smallest bumps may soften"),
        (20.0, "Low: small parts and thin edges start to vanish"),
        (35.0, "Very low: only the large shapes remain"),
    ):
        if detail < limit:
            return words
    return "Extreme: a rough block of the object"


LOD_LEVELS = 3
LOD_PERCENTS = (50.0, 25.0, 12.5)  # each half the last, a common starting point; the view change each LOD reports is the real test


def lod_name(source: str, level: int) -> str:
    return f"{source}_LOD{level}"


def lod_targets(percents, faces: int) -> list[int]:
    """Faces to land on per LOD: never so few that nothing is left."""
    return [max(12, int(round(p / 100.0 * faces))) for p in percents]
