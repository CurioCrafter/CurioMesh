// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! UV seam snap and split leftovers (package K1b). K0 stub: the frozen `max_move` input and a compile probe.

#![allow(dead_code)] // K0 stub: the frozen types are used by the other kernels' packages; drop this when filled.

use crate::geom::quad_valid;

/// Frozen cross-kernel input (K1b): per vertex, the farthest it may move in model units (0 = held).
pub type MaxMove<'a> = &'a [f64];

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let unit = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
    usize::from(quad_valid(&unit, 1e-9, 0.5))
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
