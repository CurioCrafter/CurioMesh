# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors

"""CurioMesh's own coloured icons (icons/*.png, drawn by a script in the CurioMesh source repository), loaded once per session.

`kw(name, fallback)` gives the keyword for any layout call: {"icon_value": id} when the picture loaded, else
{"icon": fallback} with a stock Blender icon, so the panel draws the same way headless and in tests."""

from __future__ import annotations

from pathlib import Path

FOLDER = Path(__file__).resolve().parent / "icons"
_pcoll = None


def load() -> None:
    global _pcoll
    if _pcoll is not None:
        return
    try:
        import bpy.utils.previews

        pcoll = bpy.utils.previews.new()
        for path in sorted(FOLDER.glob("*.png")):
            pcoll.load(path.stem, str(path), "IMAGE")
        _pcoll = pcoll
    except Exception:  # background mode: the stock icons stand in
        _pcoll = None


def free() -> None:
    global _pcoll
    if _pcoll is not None:
        try:
            import bpy.utils.previews

            bpy.utils.previews.remove(_pcoll)
        except Exception:
            pass
    _pcoll = None


def icon(name: str) -> int:
    """The icon id of icons/<name>.png, or 0 when it is not loaded."""
    if _pcoll is None or name not in _pcoll:
        return 0
    try:
        return int(_pcoll[name].icon_id)
    except Exception:
        return 0


def kw(name: str, fallback: str = "NONE") -> dict:
    found = icon(name)
    return {"icon_value": found} if found else {"icon": fallback}
