# Third-party notices

CurioMesh is licensed GPL-3.0-or-later (see `LICENSE`). This file lists the
third-party code and data that ship with it, and the notices their licenses ask for.

## CurioMesh Turbo (the `_native` binaries)

The optional `_curiomesh_turbo` binaries in `_native/` are built from the Rust source
in `native/` of the CurioMesh repository (https://github.com/CurioCrafter/CurioMesh,
the same tag as this release). That source is the Corresponding Source for the binaries.

They statically link these Rust crates. Where a crate offers "MIT OR Apache-2.0",
CurioMesh uses it under the MIT license.

| Crate | Version | License | Copyright |
|---|---|---|---|
| crossbeam-deque, crossbeam-epoch, crossbeam-utils | 0.8.8, 0.9.21, 0.8.23 | MIT OR Apache-2.0 | (c) 2019 The Crossbeam Project Developers |
| either | 1.19.0 | MIT OR Apache-2.0 | (c) 2015, the either authors (https://github.com/rayon-rs/either) |
| libc | 0.2.191 | MIT OR Apache-2.0 | (c) The Rust Project Developers |
| matrixmultiply | 0.3.11 | MIT OR Apache-2.0 | (c) 2016-2023 Ulrik Sverdrup "bluss"; (c) 2021 DutchGhost |
| ndarray | 0.17.2 | MIT OR Apache-2.0 | (c) 2015-2021 Ulrik Sverdrup "bluss", Jim Turner, and ndarray developers |
| num-complex, num-integer, num-traits | 0.4.6, 0.1.47, 0.2.19 | MIT OR Apache-2.0 | (c) 2014 The Rust Project Developers |
| numpy (rust-numpy) | 0.30.0 | BSD-2-Clause | (c) 2017, Toshiki Teramura |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | Aleksey Kladov and once_cell contributors |
| pyo3, pyo3-ffi | 0.30.1 | MIT OR Apache-2.0 | (c) 2017-present PyO3 Project and Contributors |
| rawpointer | 0.2.1 | MIT OR Apache-2.0 | (c) 2015 bluss |
| rayon, rayon-core | 1.12.0, 1.13.0 | MIT OR Apache-2.0 | (c) 2010 The Rust Project Developers |
| rustc-hash | 2.1.3 | MIT OR Apache-2.0 | The Rust Project Developers |
| wgpu, wgpu-core, wgpu-hal, wgpu-types, wgpu-naga-bridge, wgpu-core-deps-*, naga, naga-types | 30.0.1 | MIT OR Apache-2.0 | (c) gfx-rs developers (https://github.com/gfx-rs/wgpu). Draws the camera pictures on the graphics card (Vulkan, DX12, Metal) |
| pollster | 0.4.0 | Apache-2.0 OR MIT | (c) Joshua Barretto |
| ash | 0.38.0 | MIT OR Apache-2.0 | (c) Maik Klein and ash contributors |
| gpu-allocator | 0.28.0 | MIT OR Apache-2.0 | (c) Traverse Research |
| windows, windows-* | 0.62.2 and related | MIT OR Apache-2.0 | (c) Microsoft Corporation |
| objc2, block2, objc2-* , dispatch2, raw-window-metal (macOS only) | 0.6.5 and related | MIT, or MIT OR Apache-2.0 OR Zlib | (c) Mads Marquart and contributors |
| libloading | 0.8.9 | ISC | (c) Simonas Kazlauskas |
| codespan-reporting, spirv | 0.13.1, 0.4.0 | Apache-2.0 | (c) Brendan Zabarauskas; (c) Lei Zhang |
| arrayvec, bit-set, bit-vec, bitflags, bytemuck, document-features, equivalent, foldhash (Zlib), half, hashbrown, indexmap, litrs, lock_api, log, ordered-float (MIT), parking_lot, parking_lot_core, presser, profiling, range-alloc, raw-window-handle, renderdoc-sys, scopeguard, smallvec, static_assertions, thiserror, unicode-width, zerocopy, libm (MIT) | see Cargo.lock | MIT OR Apache-2.0 (or as marked) | (c) their authors; each crate's license file is in its source package |
| Rust standard library | (toolchain) | MIT OR Apache-2.0 | The Rust Project Developers |

Crates used only while building (proc-macro2, quote, syn, unicode-ident, heck,
pyo3-macros, pyo3-macros-backend, pyo3-build-config, target-lexicon, autocfg) are not
part of the binaries. All are MIT OR Apache-2.0, except target-lexicon
(Apache-2.0 WITH LLVM-exception) and unicode-ident (also Unicode-3.0).

### MIT license (for the crates above that use it)

```
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

### ISC license (libloading)

```
Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

The two crates marked Apache-2.0 (and pollster, used under either license) are under the Apache License 2.0,
https://www.apache.org/licenses/LICENSE-2.0, whose text is in each crate's source package.

### BSD 2-Clause license (rust-numpy)

```
BSD 2-Clause License

Copyright (c) 2017, Toshiki Teramura
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Code ported from Blender

`triangles_cross` in `native/src/near.rs` is a step-by-step port of
`isect_tri_tri_v3_ex` from Blender's `source/blender/blenlib/intern/math_geom.cc`
(SPDX-FileCopyrightText: 2001-2002 NaN Holding BV, and the Blender authors;
GPL-2.0-or-later). It is distributed here under GPL-3.0-or-later.

## Data

- `triadq/patterns.json` holds statistics measured on hand-made meshes, some
  licensed CC-BY 4.0 by the Blender Foundation (Blender Studio). Credits, links and
  licenses are in `docs/PATTERN_SOURCES.md`.
- `triadq/cases_table.json` holds camera-check results measured on Poly Haven models
  (CC0 1.0, https://polyhaven.com). No geometry from them is included.

## Not included

CurioMesh does not include QuadWild, QuadWild-BiMDF, QuadriFlow or Instant Meshes.
QuadriFlow is used through Blender's own QuadriFlow Remesh operator. The developer script
`scripts/flowtool.py` (not in the add-on zips) can call a separately installed
QuadWild-BiMDF (GPL-3.0, https://github.com/cgg-bern/quadwild-bimdf) as an outside
program.
