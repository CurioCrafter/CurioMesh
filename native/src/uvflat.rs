// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Sparse solver and flatten (package K5). K0 stub: a compile probe.

use crate::topo::{collect, PARALLEL_MIN};

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let squares = collect(4, |i| i * i);
    usize::from(squares == [0, 1, 4, 9] && PARALLEL_MIN > 0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }
}
