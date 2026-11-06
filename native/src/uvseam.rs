// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! UV seam kernel (package K1): source islands, side regions, seam chains, chain classes, face-cost model, per-side
//! transfer. K0 stub: constants, the frozen cross-kernel chain types and a compile probe. Byte-identical on `main`
//! and `standalone`: uses only `topo`, `geom`, `close` and the `near` items both branches have.

#![allow(dead_code)] // K0 stub: the frozen types are used by the other kernels' packages; drop this when filled.

use crate::near::{nearest_faces, FaceTree};
use crate::topo::{edges, halves, Faces};
use std::sync::atomic::AtomicBool;

// Constants table (mirrored in triadq/uvseams.py; tests/test_uv_scaffold.py compares the two).
pub const UV_TOL_PX: f64 = 0.01;
pub const TEXTURE_PX_DEFAULT: f64 = 1024.0;

/// What a seam chain asks of the remesh (spec 4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ChainClass {
    Ignore = 0,
    Shift = 1,
    Soft = 2,
    Hard = 3,
}

/// Frozen cross-kernel input (K1's output, read by K1b, K2, K3, K6): seam chains as flat slices.
/// Chain `c` runs over `run_vertex[run_indptr[c]..run_indptr[c + 1]]` (welded vertex ids). `uv_a` / `uv_b` hold the
/// UV (2 values) of each run vertex on side A and side B; `gap_px` one value per run vertex; `class` one
/// `ChainClass` code per chain.
pub struct SeamUv<'a> {
    pub run_indptr: &'a [i64],
    pub run_vertex: &'a [i64],
    pub uv_a: &'a [f64],
    pub uv_b: &'a [f64],
    pub class: &'a [u8],
    pub gap_px: &'a [f64],
}

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    let faces = Faces { indptr: &indptr, flat: &flat };
    let n_edges = edges(&halves(&faces)).len();
    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0];
    let tree = FaceTree::new(&points, &indptr, &flat);
    let stop = AtomicBool::new(false);
    let (done, rows) = nearest_faces(&tree, &[0.5, 0.5, 0.5], &stop);
    usize::from(n_edges == 7 && done && rows[0].1 >= 0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
