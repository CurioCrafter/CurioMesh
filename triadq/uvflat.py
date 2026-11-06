# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Flatten and solver twin (package K5).

Empty scaffold (K0): constants and the hook for faster callables. NumPy only, no bpy. Stages are sliceable: about 4 ms
of work per step. The reference lives in the kernel sources; a parity test compares the two.
"""

from __future__ import annotations

# Constants table: same names and values as the top of the matching .rs file (tests/test_uv_scaffold.py compares).

# Optional faster callables, set by the one adapter (uv_fast.py); the NumPy code runs when a name is missing.
FAST: dict = {}


def set_fast(**callables) -> None:
    """Install (or, with None, remove) faster callables by name."""

    for key, fn in callables.items():
        if fn is None:
            FAST.pop(key, None)
        else:
            FAST[key] = fn
