// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Patches, merge tree, cuts (package K6). K0 stub: the frozen `part_label` input and a compile probe.

#![allow(dead_code)] // K0 stub: the frozen types are used by the other kernels' packages; drop this when filled.

use crate::close::STOPPED;
use crate::topo::{edges, halves, Dsu, Faces};

/// Frozen cross-kernel input (K6): the part of every face. Passed in, never computed here, because `segment.rs` is
/// not compiled into the standalone.
pub type PartLabel<'a> = &'a [u32];

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    let n_edges = edges(&halves(&Faces { indptr: &indptr, flat: &flat })).len();
    let mut dsu = Dsu::new(3);
    dsu.union(0, 1);
    usize::from(n_edges == 7 && dsu.find(0) == dsu.find(1) && dsu.find(2) != dsu.find(0) && !STOPPED.is_empty())
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
