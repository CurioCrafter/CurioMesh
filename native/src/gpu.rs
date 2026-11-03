// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

//! `geom::pictures` on the graphics card (wgpu: Vulkan, DX12 or Metal, so NVIDIA, AMD, Intel and Apple all work).
//!
//! One compute thread per triangle fills six depth buffers (see `gpu.wgsl`). The card works in f32 and gets the
//! mesh relative to the picture center; the answers are then redone in f64 on the CPU for the pixels that were
//! hit, so a pixel that picks the same triangle as the CPU path reports the same depth and normal to the last bit.
//! What can differ: a pixel whose nearest depths agree to about 1e-7 (coplanar neighbours, a pixel center exactly
//! on a shared edge) may pick the other triangle, and the card counts a pixel center within 0.001 pixel outside a
//! triangle as inside. Ties of equal f32 depth go to the lowest triangle index, like the CPU path.
//!
//! The device is created on first use and kept for the process. Any failure returns an `Err`; callers fall back.

use std::sync::{Mutex, OnceLock};
use rayon::prelude::*;

const SHADER: &str = include_str!("gpu.wgsl");
const GROUPS_WIDE: u32 = 1024;
const MIN_SPLIT: usize = 16384; // items per rayon task: small meshes stay on one thread, where waking the pool costs more than the work

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    depth: wgpu::ComputePipeline,
    pick: wgpu::ComputePipeline,
    name: String,
    max_buffer: u64,
}

static GPU: OnceLock<Result<Mutex<Gpu>, String>> = OnceLock::new();

fn open() -> Result<Mutex<Gpu>, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err("only a software adapter".into());
    }
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("curiomesh"),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            ..wgpu::Limits::downlevel_defaults()
        },
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pictures"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let make = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let (depth, pick) = (make("depth"), make("pick"));
    Ok(Mutex::new(Gpu {
        max_buffer: (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size),
        device,
        queue,
        depth,
        pick,
        name: format!("{} ({:?})", info.name, info.backend),
    }))
}

fn gpu() -> Result<&'static Mutex<Gpu>, String> {
    GPU.get_or_init(|| std::panic::catch_unwind(open).unwrap_or_else(|_| Err("the graphics driver crashed".into())))
        .as_ref()
        .map_err(|e| e.clone())
}

/// The adapter in use ("NVIDIA GeForce RTX 3080 Ti (Vulkan)"), or why there is none.
pub fn describe() -> Result<String, String> {
    gpu().map(|g| g.lock().unwrap_or_else(|e| e.into_inner()).name.clone())
}

/// A buffer holding what `fill` writes into a scratch copy of it (the fill may run on all cores). The mapped memory
/// is write-only, so the scratch copy is filled first and copied over in one go.
fn filled(d: &wgpu::Device, size: usize, usage: wgpu::BufferUsages, fill: impl FnOnce(&mut [u8])) -> wgpu::Buffer {
    let padded = size.next_multiple_of(4).max(16);
    let buffer = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: padded as u64, usage, mapped_at_creation: true });
    let mut scratch = vec![0u8; padded];
    fill(&mut scratch[..size]);
    buffer.slice(..).get_mapped_range_mut().expect("mapped at creation").copy_from_slice(&scratch);
    buffer.unmap();
    buffer
}

/// Same arguments and result as `geom::pictures`.
pub fn pictures(
    points: &[f64],
    tris: &[i64],
    normals: &[f64],
    center: [f64; 3],
    half: f64,
    pixels: usize,
) -> Result<(Vec<bool>, Vec<f64>, Vec<f64>, Vec<i64>), String> {
    let lock = gpu()?;
    let g = lock.lock().unwrap_or_else(|e| e.into_inner());
    let (n, cells) = (tris.len() / 3, pixels * pixels);
    let out_size = 6 * cells * 4;
    if n == 0 || [points.len() * 4, tris.len() * 4, out_size].iter().any(|&b| b as u64 > g.max_buffer) || tris.len() > u32::MAX as usize {
        return Err("mesh or picture too big for the card".into());
    }
    let d = &g.device;
    let storage = wgpu::BufferUsages::STORAGE;
    let scale = pixels as f64 / (2.0 * half);
    let params = [pixels as u32, n as u32, GROUPS_WIDE * 64, (half as f32).to_bits(), (scale as f32).to_bits()];
    let params = filled(d, 20, wgpu::BufferUsages::UNIFORM, |out| {
        for (slot, word) in out.chunks_exact_mut(4).zip(params) {
            slot.copy_from_slice(&word.to_le_bytes());
        }
    });
    let pts = filled(d, points.len() * 4, storage, |out| {
        out.par_chunks_exact_mut(12).zip(points.par_chunks_exact(3)).with_min_len(MIN_SPLIT).for_each(|(slot, p)| {
            for k in 0..3 {
                slot[k * 4..k * 4 + 4].copy_from_slice(&((p[k] - center[k]) as f32).to_le_bytes());
            }
        });
    });
    let tri = filled(d, tris.len() * 4, storage, |out| {
        out.par_chunks_exact_mut(4).zip(tris.par_iter()).with_min_len(MIN_SPLIT).for_each(|(slot, &v)| slot.copy_from_slice(&(v as u32).to_le_bytes()));
    });
    let empty = |d: &wgpu::Device| filled(d, out_size, storage | wgpu::BufferUsages::COPY_SRC, |out| out.fill(0xFF));
    let (depth, ids) = (empty(d), empty(d));
    let read = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ids"),
        size: out_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = |pipeline: &wgpu::ComputePipeline| {
        let entries: Vec<_> = [&params, &pts, &tri, &depth, &ids]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries })
    };
    let groups = (n as u32).div_ceil(64);
    let mut encoder = d.create_command_encoder(&Default::default());
    for pipeline in [&g.depth, &g.pick] {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind(pipeline), &[]);
        pass.dispatch_workgroups(groups.min(GROUPS_WIDE), groups.div_ceil(GROUPS_WIDE), 1);
    }
    encoder.copy_buffer_to_buffer(&ids, 0, &read, 0, out_size as u64);
    g.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    let picked: Vec<u32> = read.slice(..).get_mapped_range().map_err(|e| e.to_string())?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    drop(g);

    // Redo the winners in f64: the depth is the CPU path's number for that triangle, the normal its unit normal.
    let mut hit = vec![false; 6 * cells];
    let mut normal = vec![0.0; 18 * cells];
    let mut depth = vec![0.0; 6 * cells];
    let mut face = vec![0_i64; 6 * cells];
    (&mut hit, &mut depth, &mut face, normal.par_chunks_mut(3), &picked)
        .into_par_iter()
        .enumerate()
        .with_min_len(MIN_SPLIT)
        .for_each(|(at, (hit, depth, face, normal, &id))| {
            if id == u32::MAX {
                return;
            }
            let (view, cell) = (at / cells, at % cells);
            let (axis, back) = (view / 2, view % 2 == 1);
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let t = &tris[id as usize * 3..id as usize * 3 + 3];
            let at_corner = |k: usize| {
                let p = &points[t[k] as usize * 3..t[k] as usize * 3 + 3];
                ((p[u] - center[u] + half) * scale - 0.5, (p[v] - center[v] + half) * scale - 0.5, p[axis])
            };
            let ((x0, y0, z0), (x1, y1, z1), (x2, y2, z2)) = (at_corner(0), at_corner(1), at_corner(2));
            let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            let (row, col) = ((cell / pixels) as f64, (cell % pixels) as f64);
            let w0 = ((y1 - y2) * row + (x2 - x1) * col + (x1 * y2 - x2 * y1)) / area;
            let w1 = ((y2 - y0) * row + (x0 - x2) * col + (x2 * y0 - x0 * y2)) / area;
            let z = w0 * z0 + w1 * z1 + (1.0 - w0 - w1) * z2;
            *hit = true;
            *face = id as i64;
            *depth = if back { z - (center[axis] - 2.0 * half) } else { center[axis] + 2.0 * half - z };
            let nn = &normals[id as usize * 3..id as usize * 3 + 3];
            let size = (nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]).sqrt().max(f64::MIN_POSITIVE);
            normal.copy_from_slice(&[nn[0] / size, nn[1] / size, nn[2] / size]);
        });
    Ok((hit, normal, depth, face))
}
