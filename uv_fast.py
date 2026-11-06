# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
"""The one adapter between the compiled UV kernels (`turbo`) and the NumPy cores in `triadq/uv*.py`.

`install()` hands each core its faster callables; the cores never import anything compiled and run their NumPy path
for every callable that is missing. Main-only file (deleted from the site build). One block per kernel, each filled by
that kernel's package: `# main-only: uv begin <kernel>` ... `# main-only: uv end <kernel>`.
"""

from __future__ import annotations

from . import turbo
from .triadq import uvchart, uvflat, uvpack, uvscore, uvseams, uvsight, uvsnap

# main-only: uv begin seam
def _install_seam() -> None:
    uvseams.set_fast(probe=turbo.uvseam_probe)
# main-only: uv end seam

# main-only: uv begin snap
def _install_snap() -> None:
    uvsnap.set_fast(probe=turbo.uvsnap_probe)
# main-only: uv end snap

# main-only: uv begin qem
def _install_qem() -> None:
    pass  # K2: called from qem::run_seams inside the compiled core; no NumPy twin
# main-only: uv end qem

# main-only: uv begin score
def _install_score() -> None:
    uvscore.set_fast(probe=turbo.uvscore_probe, check=turbo.uvscore_check, names=turbo.uvscore_names)
# main-only: uv end score

# main-only: uv begin sight
def _install_sight() -> None:
    uvsight.set_fast(probe=turbo.uvsight_probe)
# main-only: uv end sight

# main-only: uv begin flat
def _install_flat() -> None:
    uvflat.set_fast(probe=turbo.uvflat_probe)
# main-only: uv end flat

# main-only: uv begin chart
def _install_chart() -> None:
    uvchart.set_fast(probe=turbo.uvchart_probe)
# main-only: uv end chart

# main-only: uv begin pack
def _install_pack() -> None:
    uvpack.set_fast(probe=turbo.uvpack_probe)
# main-only: uv end pack


_INSTALLERS = (_install_seam, _install_snap, _install_qem, _install_score, _install_sight, _install_flat, _install_chart, _install_pack)


def install() -> bool:
    """Give every core its faster callables. False (and nothing installed) when the compiled kernels are missing."""

    if not turbo.available():
        uninstall()
        return False
    for step in _INSTALLERS:
        step()
    return True


def uninstall() -> None:
    for core in (uvseams, uvsnap, uvscore, uvsight, uvflat, uvchart, uvpack):
        core.FAST.clear()
