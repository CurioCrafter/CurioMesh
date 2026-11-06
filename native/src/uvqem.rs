// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Seam-aware quadric edge collapse (package K2), called from `qem::run_seams` on each branch. K0 stub: a compile
//! probe. Byte-identical on `main` and `standalone` (their `qem.rs` copies differ, so this file stands alone).

use crate::topo::{edges, halves, Faces};

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let flat = [0_i64, 1, 2, 3, 1, 4, 5, 2];
    let indptr = [0_i64, 4, 8];
    let h = halves(&Faces { indptr: &indptr, flat: &flat });
    usize::from(h.face.len() == 8 && edges(&h).len() == 7)
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
