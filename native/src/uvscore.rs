// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! UV Check numbers (package K3). K0 stub: the frozen `seen_face` / `seen_edge_px` inputs and a compile probe.

#![allow(dead_code)] // K0 stub: the frozen types are used by the other kernels' packages; drop this when filled.

use crate::topo::{summary, Faces};

/// Frozen cross-kernel inputs (K4's outputs). `None` means "every face and edge fully seen" (synthetic seen = 1).
pub type SeenFace<'a> = Option<&'a [f64]>;
pub type SeenEdgePx<'a> = Option<&'a [f64]>;

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    usize::from(summary(&Faces { indptr: &indptr, flat: &flat }).edges == 7)
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
