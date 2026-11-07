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
    uvseams.set_fast(probe=turbo.uvseam_probe, extract=turbo.uvseam_extract, triage=turbo.uvseam_triage, transfer=turbo.uvseam_transfer,
                     layer=turbo.uvseam_layer)
# main-only: uv end seam

# main-only: uv begin snap
def _install_snap() -> None:
    uvsnap.set_fast(probe=turbo.uvsnap_probe, snap=turbo.uvsnap_snap, split=turbo.uvsnap_split)
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
class _DrawJob:
    """Draws the pictures with the compiled kernel on a worker thread and hands them to the NumPy reader."""

    def __init__(self, points, polys, views, pixels, weights):
        self._pictures = turbo.uvsight_pictures(points, polys, views, pixels)
        self._job = turbo.uvsight_fill(self._pictures)

    def done(self) -> bool:
        return self._job.done()

    def stop(self) -> None:
        self._job.stop()

    def parts(self):
        """The pictures one view at a time, so the reader can yield between them."""
        self._job.result()
        import numpy as np

        n = self._pictures.frame()[0]
        for k in range(int(self._pictures.drawn)):
            yield self._pictures.face(k).reshape(n, n), self._pictures.depth(k).astype(np.float32).reshape(n, n)

    def result(self):
        got = list(self.parts())
        return [f for f, _ in got], [d for _, d in got]


def _install_sight() -> None:
    uvsight.set_fast(probe=turbo.uvsight_probe, draw_job=_DrawJob)
# main-only: uv end sight

# main-only: uv begin flat
def _install_flat() -> None:
    uvflat.set_fast(probe=turbo.uvflat_probe, flatten=turbo.uvflat_flatten, grid_layout=turbo.uvflat_grid_layout,
                    quick_stretch=turbo.uvflat_quick_stretch, stretch=turbo.uvflat_stretch)
# main-only: uv end flat

# main-only: uv begin chart
def _install_chart() -> None:
    uvchart.set_fast(probe=turbo.uvchart_probe, build=turbo.uvchart_build, level=turbo.uvchart_level, uvs=turbo.uvchart_uvs,
                     cuts=turbo.uvchart_cuts, keep_going=turbo.uvchart_pass)
# main-only: uv end chart

# main-only: uv begin pack
def _install_pack() -> None:
    uvpack.set_fast(probe=turbo.uvpack_probe, fill_page=turbo.uvpack_fill_page, pack_boxes=turbo.uvpack_pack_boxes, pack_into=turbo.uvpack_pack_into)
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
