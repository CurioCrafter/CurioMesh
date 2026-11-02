# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Auto Segments presets by use. A preset only sets Detail, Method and Output; every one stays editable (that is Custom)."""

from __future__ import annotations

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
    "ANIMATION": ("Animation", "All quads, even sizes and more faces, so the mesh bends cleanly. Check eyes, mouth and joints by hand", 1.5, "FIELD", "QUADS"),
    "SUBDIV": ("Subdivision", "All quads with flat panels kept as single large quads, so smoothing keeps the shape", 1.5, "FLAT", "QUADS"),
    "SCAN": ("Scan Cleanup", "Keeps more of the detail of a dirty scan (scans have few true flats, so no flat merging)", 2.0, "AUTO", "QUADS"),
    "SCULPT": ("Sculpt Base", "Even, smooth quads to sculpt on: smoothness over detail", 6.0, "PAIRING", "QUADS"),
}
PROFILE_ITEMS = [(k, v[0], v[1]) for k, v in PROFILES.items()] + [("CUSTOM", "Custom", "Your own Detail, Method and Output")]


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
        (10.0, "Balanced: small features may soften"),
        (20.0, "Low: small parts and thin edges start to vanish"),
        (35.0, "Very low: only the large shapes remain"),
    ):
        if detail < limit:
            return words
    return "Extreme: a rough block of the object"
