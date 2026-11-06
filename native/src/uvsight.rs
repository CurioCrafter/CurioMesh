// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Visibility from face-id pictures (package K4). K0 stub: constants, the frozen output type and a compile probe.

#![allow(dead_code)] // K0 stub: the frozen types are used by the other kernels' packages; drop this when filled.

use crate::geom::pictures;
use crate::topo::{edges, halves, Faces};

// Constants table (mirrored in triadq/uvsight.py).
pub const VIEWS_DEFAULT: usize = 6;

/// Frozen cross-kernel output (read by K3, K6): how much each face and each edge is seen.
pub struct Seen {
    pub seen_face: Vec<f64>,
    pub seen_edge_px: Vec<f64>,
}

impl Seen {
    /// Synthetic "everything seen once", for tests of the kernels that read it.
    pub fn all_seen(faces: &Faces) -> Seen {
        Seen { seen_face: vec![1.0; faces.count()], seen_edge_px: vec![1.0; edges(&halves(faces)).len()] }
    }
}

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    let seen = Seen::all_seen(&Faces { indptr: &indptr, flat: &flat });
    let points = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let up = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let (hit, _, _, _) = pictures(&points, &[0, 1, 2], &up, [0.3, 0.3, 0.0], 1.0, 4);
    usize::from(seen.seen_face.len() == 2 && seen.seen_edge_px.len() == 7 && hit.len() == 6 * 16)
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
