// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! CurioMesh without Blender. The heavy kernels are the add-on's own native core, compiled straight from
//! `native/src` so both always run the same code; only the PyO3 bindings (`native/src/lib.rs`) are left out.
// ponytail: #[path] includes instead of splitting native/ into a lib crate + binding crate; split it when a third user
// of the kernels appears or the module names stop matching.

#![allow(dead_code, clippy::all)]

#[path = "../../native/src/chords.rs"]
mod chords;
#[path = "../../native/src/close.rs"]
pub mod close;
#[path = "../../native/src/geom.rs"]
pub mod geom;
#[path = "../../native/src/gpu.rs"]
pub mod gpu;
#[path = "../../native/src/join.rs"]
mod join;
#[path = "../../native/src/matching.rs"]
mod matching;
#[path = "../../native/src/near.rs"]
pub mod near;
#[path = "../../native/src/pairing.rs"]
pub mod pairing;
#[path = "../../native/src/qem.rs"]
pub mod qem;
#[path = "../../native/src/topo.rs"]
pub mod topo;

pub mod io;
pub mod issues;
pub mod retopo;
