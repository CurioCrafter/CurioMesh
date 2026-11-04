// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Puts the CurioMesh icon into the Windows programs, so Explorer and the taskbar show it.

fn main() {
    println!("cargo:rerun-if-changed=assets/curiomesh.ico");
    println!("cargo:rerun-if-changed=assets/curiomesh.rc");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile("assets/curiomesh.rc", embed_resource::NONE).manifest_optional().unwrap();
    }
}
