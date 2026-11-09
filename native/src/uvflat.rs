// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Sparse solver and flatten (package K5). Our own code, written from the papers named at each piece:
//!
//! - Ordering: nested dissection by level structures (George 1973, "Nested dissection of a regular finite element
//!   mesh"; George, Liu 1981, "Computer solution of large sparse positive definite systems").
//! - Direct solve: up-looking simplicial LDLᵀ over the elimination tree (Liu 1990, "The role of elimination trees in
//!   sparse factorization"; Davis 2005, "Algorithm 849: a concise sparse Cholesky factorization package").
//! - Iterative path above a memory threshold: CG (Hestenes, Stiefel 1952) with IC(0) (Meijerink, van der Vorst 1977),
//!   Jacobi when IC(0) meets a non-positive pivot.
//! - Exact grid layout of a pole-free quad chart (integer grid after Kälberer, Nieser, Polthier 2007 and Bommes,
//!   Zimmer, Kobbelt 2009/2013; strip lengths ours).
//! - Quad ARAP: the local/global method of Liu, Zhang, Xu, Gotsman, Gortler 2008 with one rotation per polygon; the
//!   per-side weights of a polygon are the cotangent weights of the rectangle with the same side lengths (ours).
//! - Straight rows: On = the vertices of a row share one unknown (constraint idea after Campen, Kobbelt 2014); Soft = a
//!   label-free second-difference penalty along quad loops (ours).
//! - Injective step: largest step with no flipped triangle (Smith, Schaefer 2015, "Bijective parameterization with
//!   free boundaries").
//! - LSCM (Lévy, Petitjean, Ray, Maillot 2002) with a soft pull toward a start instead of pins (ours), and Tutte's
//!   embedding (Tutte 1963; Floater 1997) when a start flips.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::topo::{collect, Faces, FxBuild, PARALLEL_MIN};

// Constants table (the NumPy twin `triadq/uvflat.py` holds the same names and values).
/// Texture size used for the stop rule when the caller gives none (pixels).
pub const TEXTURE_PX: f64 = 1024.0;
/// Iterative stop: no UV would move more than this share of a texel.
pub const STOP_TEXEL_SHARE: f64 = 0.02;
/// Iterative stall rule: the residual must drop by `STALL_DROP` within `STALL_WINDOW` iterations, else stop (reported).
pub const STALL_WINDOW: usize = 50;
pub const STALL_DROP: f64 = 0.01;
/// Iterations over which the residual's shrink rate is measured for the stop rule.
pub const RATE_WINDOW: usize = 10;
/// ARAP "until converged": stop when the energy changes by less than this share in one step.
pub const ARAP_CHANGE: f64 = 0.01;
/// Injective step: take this share of the largest flip-free step.
pub const STEP_SHARE: f64 = 0.9;
/// LSCM pull weight as a share of the mean diagonal of the conformal matrix.
pub const PULL_SHARE: f64 = 0.001;
/// Soft rows: starting weight (share of the mean ARAP diagonal) and growth per try until the row bend target holds.
pub const SOFT_START: f64 = 0.1;
pub const SOFT_GROW: f64 = 4.0;
/// Soft rows target: 95th percentile of the row bend angle, degrees.
pub const BEND_P95_DEG: f64 = 2.0;
/// Chart stretch reported as this quantile over faces.
pub const STRETCH_QUANTILE: f64 = 0.99;
/// Memory allowed for one direct factor (bytes) when the caller passes 0; above it the iterative path runs.
pub const FACTOR_MAX_BYTES: f64 = 2.0e9;
/// Nested dissection stops splitting below this many unknowns.
pub const ND_LEAF: usize = 64;
/// Numbers per chart in the stats rows (see `Stat`).
pub const STATS_LEN: usize = 17;

/// Columns of a stats row.
pub mod stat {
    pub const CHART: usize = 0;
    pub const VERTICES: usize = 1;
    pub const FACES: usize = 2;
    pub const PATH: usize = 3; // 0 direct, 1 IC(0) CG, 2 Jacobi CG
    pub const ROWS: usize = 4; // 0 off, 1 soft, 2 on
    pub const STEPS: usize = 5;
    pub const ENERGY: usize = 6;
    pub const FLIPS: usize = 7;
    pub const STRETCH: usize = 8;
    pub const BEND: usize = 9; // p95 row bend, degrees (NaN with no rows)
    pub const FACTOR_MS: usize = 10;
    pub const SOLVE_MS: usize = 11; // one warm global solve
    pub const TOTAL_MS: usize = 12;
    pub const STOPPED: usize = 13;
    pub const NNZ_L: usize = 14;
    pub const START: usize = 15; // 0 given, 1 LSCM, 2 Tutte, 3 LSCM with pull
    pub const STALLED: usize = 16; // iterative solves that stopped on the stall rule
}

const NONE: usize = usize::MAX;

/// Compile probe: calls every `crate::` helper this kernel is assigned. Returns 1 when all worked.
pub fn probe() -> usize {
    let squares = collect(4, |i| i * i);
    usize::from(squares == [0, 1, 4, 9] && PARALLEL_MIN > 0)
}

// ---------------------------------------------------------------------------------------------------------------
// Sparse symmetric matrices
// ---------------------------------------------------------------------------------------------------------------

/// Symmetric matrix in CSR, both triangles stored, columns sorted, duplicates summed.
pub struct Csr {
    pub n: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<u32>,
    pub val: Vec<f64>,
}

/// Collects symmetric entries; `add(i, j, v)` adds v at (i, j) and (j, i).
pub struct Builder {
    n: usize,
    t: Vec<(u32, u32, f64)>,
}

impl Builder {
    pub fn new(n: usize) -> Self {
        Builder { n, t: Vec::new() }
    }
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        self.t.push((i as u32, j as u32, v));
        if i != j {
            self.t.push((j as u32, i as u32, v));
        }
    }
    /// `w (e_a - e_b)(e_a - e_b)ᵀ`
    pub fn edge(&mut self, a: usize, b: usize, w: f64) {
        if a == b {
            return; // both ends share one unknown (Straight rows On): the term is constant
        }
        self.add(a, a, w);
        self.add(b, b, w);
        self.add(a, b, -w);
    }
    pub fn build(self) -> Csr {
        let n = self.n;
        let mut count = vec![0usize; n + 1];
        for &(i, _, _) in &self.t {
            count[i as usize + 1] += 1;
        }
        for i in 0..n {
            count[i + 1] += count[i];
        }
        let mut fill = count.clone();
        let mut cols = vec![(0u32, 0f64); self.t.len()];
        for &(i, j, v) in &self.t {
            cols[fill[i as usize]] = (j, v);
            fill[i as usize] += 1;
        }
        let (mut ptr, mut idx, mut val) = (Vec::with_capacity(n + 1), Vec::with_capacity(cols.len()), Vec::with_capacity(cols.len()));
        ptr.push(0);
        for i in 0..n {
            let row = &mut cols[count[i]..count[i + 1]];
            row.sort_unstable_by_key(|e| e.0);
            for &(j, v) in row.iter() {
                if idx.len() > *ptr.last().unwrap() && *idx.last().unwrap() == j {
                    *val.last_mut().unwrap() += v;
                } else {
                    idx.push(j);
                    val.push(v);
                }
            }
            ptr.push(idx.len());
        }
        Csr { n, ptr, idx, val }
    }
}

impl Csr {
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        (&self.idx[self.ptr[i]..self.ptr[i + 1]], &self.val[self.ptr[i]..self.ptr[i + 1]])
    }
    pub fn diag(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let (c, v) = self.row(i);
                c.iter().position(|&j| j as usize == i).map_or(0.0, |p| v[p])
            })
            .collect()
    }
    pub fn mul(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let (c, v) = self.row(i);
            y[i] = c.iter().zip(v).map(|(&j, &a)| a * x[j as usize]).sum();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Ordering: nested dissection by level structures (George 1973; George, Liu 1981)
// ---------------------------------------------------------------------------------------------------------------

/// Fill-reducing order of the graph of `a`: `perm[new] = old`. Each part is split by the smallest BFS level near its
/// middle (a vertex separator), the two sides are ordered first and the separator last; small parts keep BFS order.
pub fn nd_order(a: &Csr) -> Vec<u32> {
    let n = a.n;
    if n <= ND_LEAF {
        return (0..n as u32).collect(); // one leaf: the loop below would keep the given order
    }
    let mut perm = vec![0u32; n];
    let mut part = vec![0u32; n]; // stamp: which part a vertex belongs to right now
    let mut seen = vec![0u32; n];
    let (mut part_stamp, mut seen_stamp) = (0u32, 0u32);
    let mut stack: Vec<(Vec<u32>, usize)> = vec![((0..n as u32).collect(), 0)];
    let mut queue: Vec<u32> = Vec::new();
    let mut level: Vec<u32> = vec![0; n];
    while let Some((verts, lo)) = stack.pop() {
        if verts.len() <= ND_LEAF {
            perm[lo..lo + verts.len()].copy_from_slice(&verts);
            continue;
        }
        part_stamp += 1;
        for &v in &verts {
            part[v as usize] = part_stamp;
        }
        // BFS twice: the second root is far from the first (a pseudo-peripheral start).
        let mut bfs = |root: u32, queue: &mut Vec<u32>, level: &mut Vec<u32>, seen_stamp: &mut u32| {
            *seen_stamp += 1;
            queue.clear();
            queue.push(root);
            seen[root as usize] = *seen_stamp;
            level[root as usize] = 0;
            let mut head = 0;
            while head < queue.len() {
                let v = queue[head] as usize;
                head += 1;
                let (c, _) = a.row(v);
                for &w in c {
                    let w_ = w as usize;
                    if part[w_] == part_stamp && seen[w_] != *seen_stamp {
                        seen[w_] = *seen_stamp;
                        level[w_] = level[v] + 1;
                        queue.push(w);
                    }
                }
            }
        };
        bfs(verts[0], &mut queue, &mut level, &mut seen_stamp);
        let far = *queue.last().unwrap();
        bfs(far, &mut queue, &mut level, &mut seen_stamp);
        if queue.len() < verts.len() {
            // Disconnected: the reached component and the rest are independent; no separator needed.
            let stamp = seen_stamp;
            let reached = queue.clone();
            let rest: Vec<u32> = verts.iter().copied().filter(|&v| seen[v as usize] != stamp).collect();
            stack.push((rest, lo + reached.len()));
            stack.push((reached, lo));
            continue;
        }
        let depth = level[*queue.last().unwrap() as usize] as usize + 1;
        if depth < 3 {
            perm[lo..lo + verts.len()].copy_from_slice(&queue);
            continue;
        }
        let mut size = vec![0usize; depth];
        for &v in queue.iter() {
            size[level[v as usize] as usize] += 1;
        }
        // Smallest level whose cumulative share before it lies in 30..70 % (middle level when none).
        let total = queue.len();
        let (mut best, mut before) = (NONE, 0usize);
        let mut acc = 0usize;
        for (l, &s) in size.iter().enumerate() {
            if l > 0 && l + 1 < depth && acc * 10 >= total * 3 && acc * 10 <= total * 7 && (best == NONE || s < size[best]) {
                best = l;
                before = acc;
            }
            acc += s;
        }
        if best == NONE {
            acc = 0;
            for (l, &s) in size.iter().enumerate() {
                if acc * 2 >= total && l > 0 && l + 1 < depth {
                    best = l;
                    before = acc;
                    break;
                }
                acc += s;
            }
        }
        if best == NONE {
            perm[lo..lo + verts.len()].copy_from_slice(&queue);
            continue;
        }
        let sep = size[best];
        let mut side_a = Vec::with_capacity(before);
        let mut side_b = Vec::with_capacity(total - before - sep);
        let mut s_at = lo + total - sep;
        for &v in queue.iter() {
            let l = level[v as usize] as usize;
            if l < best {
                side_a.push(v);
            } else if l > best {
                side_b.push(v);
            } else {
                perm[s_at] = v;
                s_at += 1;
            }
        }
        let a_len = side_a.len();
        stack.push((side_b, lo + a_len));
        stack.push((side_a, lo));
    }
    perm
}

// ---------------------------------------------------------------------------------------------------------------
// Direct solve: simplicial LDLᵀ (elimination tree, up-looking rows; Liu 1990, Davis 2005)
// ---------------------------------------------------------------------------------------------------------------

pub struct Ldlt {
    pub n: usize,
    perm: Vec<u32>,
    lp: Vec<usize>,
    li: Vec<u32>,
    lx: Vec<f64>,
    d: Vec<f64>,
}

/// Elimination tree and column counts of L for `a` in the order `perm` (`pinv` its inverse).
fn symbolic(a: &Csr, perm: &[u32], pinv: &[u32]) -> (Vec<usize>, Vec<usize>) {
    let n = a.n;
    let (mut parent, mut flag, mut lnz) = (vec![NONE; n], vec![NONE; n], vec![0usize; n]);
    for k in 0..n {
        flag[k] = k;
        let (c, _) = a.row(perm[k] as usize);
        for &j in c {
            let mut i = pinv[j as usize] as usize;
            if i < k {
                while flag[i] != k {
                    if parent[i] == NONE {
                        parent[i] = k;
                    }
                    lnz[i] += 1;
                    flag[i] = k;
                    i = parent[i];
                }
            }
        }
    }
    (parent, lnz)
}

/// Entries of L the factor of `a` would hold, and the inverse permutation.
pub fn factor_size(a: &Csr, perm: &[u32]) -> (usize, Vec<u32>, Vec<usize>, Vec<usize>) {
    let mut pinv = vec![0u32; a.n];
    for (k, &p) in perm.iter().enumerate() {
        pinv[p as usize] = k as u32;
    }
    let (parent, lnz) = symbolic(a, perm, &pinv);
    (lnz.iter().sum(), pinv, parent, lnz)
}

impl Ldlt {
    /// Factor `a` (SPD) in the order `perm`; `size` = `factor_size(a, &perm)`. Err on a non-positive pivot or when stopped.
    pub fn factor(a: &Csr, perm: Vec<u32>, size: (usize, Vec<u32>, Vec<usize>, Vec<usize>), stop: &AtomicBool) -> Result<Ldlt, String> {
        let n = a.n;
        let (nnz, pinv, parent, lnz_count) = size;
        let mut lp = vec![0usize; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + lnz_count[k];
        }
        let (mut li, mut lx) = (vec![0u32; nnz], vec![0f64; nnz]);
        let (mut y, mut flag, mut lnz, mut pattern, mut d) = (vec![0f64; n], vec![NONE; n], vec![0usize; n], vec![0usize; n], vec![0f64; n]);
        for k in 0..n {
            if k % 4096 == 0 && stop.load(Ordering::Relaxed) {
                return Err(crate::close::STOPPED.to_string());
            }
            let mut top = n;
            flag[k] = k;
            let (c, v) = a.row(perm[k] as usize);
            for (&j, &val) in c.iter().zip(v) {
                let mut i = pinv[j as usize] as usize;
                if i <= k {
                    y[i] += val;
                    let mut len = 0;
                    while flag[i] != k {
                        pattern[len] = i;
                        len += 1;
                        flag[i] = k;
                        i = parent[i];
                    }
                    while len > 0 {
                        top -= 1;
                        len -= 1;
                        pattern[top] = pattern[len];
                    }
                }
            }
            d[k] = y[k];
            y[k] = 0.0;
            for &i in &pattern[top..n] {
                let yi = y[i];
                y[i] = 0.0;
                let (s, e) = (lp[i], lp[i] + lnz[i]);
                for q in s..e {
                    y[li[q] as usize] -= lx[q] * yi;
                }
                let l_ki = yi / d[i];
                d[k] -= l_ki * yi;
                li[e] = k as u32;
                lx[e] = l_ki;
                lnz[i] += 1;
            }
            if d[k] <= 0.0 || !d[k].is_finite() {
                return Err(format!("not positive definite at pivot {k}"));
            }
        }
        Ok(Ldlt { n, perm, lp, li, lx, d })
    }

    /// Solve in place for `K` right-hand sides stored interleaved (`x[i * K + r]`).
    pub fn solve<const K: usize>(&self, x: &mut [f64]) {
        let n = self.n;
        let mut y = vec![0f64; n * K];
        for k in 0..n {
            let o = self.perm[k] as usize;
            y[k * K..k * K + K].copy_from_slice(&x[o * K..o * K + K]);
        }
        for j in 0..n {
            let mut yj = [0f64; K];
            yj.copy_from_slice(&y[j * K..j * K + K]);
            for p in self.lp[j]..self.lp[j + 1] {
                let (i, l) = (self.li[p] as usize * K, self.lx[p]);
                for r in 0..K {
                    y[i + r] -= l * yj[r];
                }
            }
        }
        for j in 0..n {
            for r in 0..K {
                y[j * K + r] /= self.d[j];
            }
        }
        for j in (0..n).rev() {
            let mut s = [0f64; K];
            for p in self.lp[j]..self.lp[j + 1] {
                let (i, l) = (self.li[p] as usize * K, self.lx[p]);
                for r in 0..K {
                    s[r] += l * y[i + r];
                }
            }
            for r in 0..K {
                y[j * K + r] -= s[r];
            }
        }
        for k in 0..n {
            let o = self.perm[k] as usize;
            x[o * K..o * K + K].copy_from_slice(&y[k * K..k * K + K]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Iterative path: CG with IC(0), Jacobi on breakdown
// ---------------------------------------------------------------------------------------------------------------

/// Incomplete Cholesky with the pattern of the lower triangle (diagonal last in each row).
pub struct Ic0 {
    ptr: Vec<usize>,
    idx: Vec<u32>,
    val: Vec<f64>,
}

impl Ic0 {
    /// None when a pivot is not positive (the caller then uses Jacobi).
    pub fn new(a: &Csr) -> Option<Ic0> {
        let n = a.n;
        let (mut ptr, mut idx, mut val) = (vec![0usize], Vec::new(), Vec::new());
        for i in 0..n {
            let (c, v) = a.row(i);
            for (&j, &x) in c.iter().zip(v) {
                if j as usize <= i {
                    idx.push(j);
                    val.push(x);
                }
            }
            if idx.last().map(|&j| j as usize) != Some(i) {
                return None;
            }
            ptr.push(idx.len());
        }
        let mut pos = vec![u32::MAX; n];
        for i in 0..n {
            let (s, e) = (ptr[i], ptr[i + 1]);
            for p in s..e {
                pos[idx[p] as usize] = p as u32;
            }
            for p in s..e - 1 {
                let k = idx[p] as usize;
                let mut dot = 0.0;
                for q in ptr[k]..ptr[k + 1] - 1 {
                    let at = pos[idx[q] as usize];
                    if at != u32::MAX {
                        dot += val[at as usize] * val[q];
                    }
                }
                val[p] = (val[p] - dot) / val[ptr[k + 1] - 1];
            }
            let dd = val[e - 1] - val[s..e - 1].iter().map(|x| x * x).sum::<f64>();
            if dd <= 0.0 || !dd.is_finite() {
                return None;
            }
            val[e - 1] = dd.sqrt();
            for p in s..e {
                pos[idx[p] as usize] = u32::MAX;
            }
        }
        Some(Ic0 { ptr, idx, val })
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = r.len();
        for i in 0..n {
            let (s, e) = (self.ptr[i], self.ptr[i + 1]);
            let mut t = r[i];
            for p in s..e - 1 {
                t -= self.val[p] * z[self.idx[p] as usize];
            }
            z[i] = t / self.val[e - 1];
        }
        for i in (0..n).rev() {
            let (s, e) = (self.ptr[i], self.ptr[i + 1]);
            z[i] /= self.val[e - 1];
            let zi = z[i];
            for p in s..e - 1 {
                z[self.idx[p] as usize] -= self.val[p] * zi;
            }
        }
    }
}

pub enum Pre {
    Ic0(Ic0),
    Jacobi(Vec<f64>),
}

#[derive(Default, Clone, Copy)]
pub struct CgInfo {
    pub iterations: usize,
    pub stalled: bool,
    pub stopped: bool,
}

/// Preconditioned CG from the warm start in `x`. Stops when the moves still to come would shift no unknown by more
/// than `tol` (twice in a row), when the residual is zero, or when it stalls (less than `STALL_DROP` drop in
/// `STALL_WINDOW` iterations; reported). No iteration cap. Moves still to come (ours): this step's largest move times
/// ρ/(1-ρ), ρ = the residual's mean shrink per iteration over the last `RATE_WINDOW` iterations (a geometric tail).
pub fn pcg(a: &Csr, pre: &Pre, b: &[f64], x: &mut [f64], tol: f64, stop: &AtomicBool) -> CgInfo {
    let n = a.n;
    let mut info = CgInfo::default();
    let (mut r, mut z, mut ap) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    a.mul(x, &mut ap);
    for i in 0..n {
        r[i] = b[i] - ap[i];
    }
    let precondition = |r: &[f64], z: &mut [f64]| match pre {
        Pre::Ic0(ic) => ic.apply(r, z),
        Pre::Jacobi(d) => {
            for i in 0..n {
                z[i] = r[i] / d[i];
            }
        }
    };
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let norm = |r: &[f64]| r.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut window_start = norm(&r);
    let mut recent = [window_start; RATE_WINDOW];
    let mut small = 0;
    while rz > 0.0 {
        if stop.load(Ordering::Relaxed) {
            info.stopped = true;
            break;
        }
        a.mul(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        let mut moved = 0f64;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            moved = moved.max((alpha * p[i]).abs());
        }
        info.iterations += 1;
        let now = norm(&r);
        let back = info.iterations.min(RATE_WINDOW);
        let then = recent[info.iterations % RATE_WINDOW];
        recent[info.iterations % RATE_WINDOW] = now;
        let rho = if then > 0.0 { (now / then).powf(1.0 / back as f64).min(1.0 - 1e-12) } else { 0.0 };
        if moved * rho / (1.0 - rho) <= tol {
            small += 1;
            if small >= 2 {
                break;
            }
        } else {
            small = 0;
        }
        if info.iterations % STALL_WINDOW == 0 {
            if now > (1.0 - STALL_DROP) * window_start {
                info.stalled = true;
                break;
            }
            window_start = now;
        }
        precondition(&r, &mut z);
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    info
}

// ---------------------------------------------------------------------------------------------------------------
// One SPD system with pinned unknowns: direct when the factor fits in memory, else CG
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Path {
    Direct,
    Ic0,
    Jacobi,
}

enum Solver {
    Direct(Ldlt),
    Iter(Csr, Pre),
}

pub struct Lin {
    pub n: usize,
    solver: Solver,
    /// (row i, pinned unknown p, A[i][p]) for unpinned i: moved to the right-hand side.
    pin_cols: Vec<(u32, u32, f64)>,
    pub path: Path,
    pub nnz_l: usize,
    pub factor_ms: f64,
}

#[derive(Default, Clone, Copy)]
pub struct SolveInfo {
    pub iterations: usize,
    pub stalled: usize,
    pub stopped: bool,
}

impl Lin {
    /// `a` with the rows and columns of `pins` replaced by the identity. `max_bytes` (0 = `FACTOR_MAX_BYTES`) picks
    /// the path; `force_iterative` is for tests.
    pub fn new(a: Csr, pins: &[usize], max_bytes: f64, force_iterative: bool, stop: &AtomicBool) -> Result<Lin, String> {
        let started = Instant::now();
        let n = a.n;
        let mut pinned = vec![false; n];
        for &p in pins {
            pinned[p] = true;
        }
        let mut pin_cols = Vec::new();
        let (mut ptr, mut idx, mut val) = (vec![0usize], Vec::with_capacity(a.idx.len()), Vec::with_capacity(a.idx.len()));
        for i in 0..n {
            let (c, v) = a.row(i);
            if pinned[i] {
                idx.push(i as u32);
                val.push(1.0);
            } else {
                for (&j, &x) in c.iter().zip(v) {
                    if pinned[j as usize] {
                        pin_cols.push((i as u32, j, x));
                    } else {
                        idx.push(j);
                        val.push(x);
                    }
                }
            }
            ptr.push(idx.len());
        }
        let a = Csr { n, ptr, idx, val };
        let limit = if max_bytes > 0.0 { max_bytes } else { FACTOR_MAX_BYTES };
        let perm = nd_order(&a);
        let size = factor_size(&a, &perm);
        let nnz = size.0;
        let bytes = nnz as f64 * 12.0 + n as f64 * 64.0;
        let (solver, path, nnz_l) = if !force_iterative && bytes <= limit {
            (Solver::Direct(Ldlt::factor(&a, perm, size, stop)?), Path::Direct, nnz)
        } else {
            match Ic0::new(&a) {
                Some(ic) => (Solver::Iter(a, Pre::Ic0(ic)), Path::Ic0, 0),
                None => {
                    let d = a.diag();
                    if d.iter().any(|&x| x <= 0.0) {
                        return Err("matrix has a non-positive diagonal".into());
                    }
                    (Solver::Iter(a, Pre::Jacobi(d)), Path::Jacobi, 0)
                }
            }
        };
        Ok(Lin { n, solver, pin_cols, path, nnz_l, factor_ms: started.elapsed().as_secs_f64() * 1e3 })
    }

    /// Solve for `K` interleaved right-hand sides. On entry the rows of pinned unknowns hold their values; `x` holds the
    /// warm start (iterative path) and receives the solution. `tol` is the iterative stop rule's largest move.
    pub fn solve<const K: usize>(&self, rhs: &mut [f64], x: &mut [f64], tol: f64, stop: &AtomicBool) -> SolveInfo {
        for &(i, p, a) in &self.pin_cols {
            for r in 0..K {
                rhs[i as usize * K + r] -= a * rhs[p as usize * K + r];
            }
        }
        let mut info = SolveInfo::default();
        match &self.solver {
            Solver::Direct(f) => {
                f.solve::<K>(rhs);
                x.copy_from_slice(rhs);
            }
            Solver::Iter(a, pre) => {
                let n = self.n;
                for r in 0..K {
                    let b: Vec<f64> = (0..n).map(|i| rhs[i * K + r]).collect();
                    let mut col: Vec<f64> = (0..n).map(|i| x[i * K + r]).collect();
                    let got = pcg(a, pre, &b, &mut col, tol, stop);
                    info.iterations += got.iterations;
                    info.stalled += usize::from(got.stalled);
                    info.stopped |= got.stopped;
                    for i in 0..n {
                        x[i * K + r] = col[i];
                    }
                }
            }
        }
        info
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Chart geometry: elements, test triangles, stretch, flips
// ---------------------------------------------------------------------------------------------------------------

/// One chart with local vertex ids `0..n`.
pub struct Chart<'a> {
    pub points: &'a [f64],
    pub faces: Faces<'a>,
}

impl Chart<'_> {
    pub fn n(&self) -> usize {
        self.points.len() / 3
    }
    fn p(&self, v: usize) -> [f64; 3] {
        [self.points[3 * v], self.points[3 * v + 1], self.points[3 * v + 2]]
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// The polygon's corners in its own plane (Newell normal, first side along x), isometric for planar polygons.
/// Writes into `pts` (scratch) and `out` (cleared first), so `elements` allocates nothing per face.
fn flat_polygon(chart: &Chart, poly: &[i64], pts: &mut Vec<[f64; 3]>, out: &mut Vec<[f64; 2]>) {
    pts.clear();
    pts.extend(poly.iter().map(|&v| chart.p(v as usize)));
    out.clear();
    let k = pts.len();
    let mut nrm = [0f64; 3];
    let mut c = [0f64; 3];
    for i in 0..k {
        let (a, b) = (pts[i], pts[(i + 1) % k]);
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
        for d in 0..3 {
            c[d] += a[d] / k as f64;
        }
    }
    let ln = norm(nrm);
    let first = sub(pts[1], pts[0]);
    if ln < 1e-300 || norm(first) < 1e-300 {
        out.resize(k, [0.0, 0.0]);
        return;
    }
    let nrm = [nrm[0] / ln, nrm[1] / ln, nrm[2] / ln];
    let t = sub(first, [nrm[0] * dot(first, nrm), nrm[1] * dot(first, nrm), nrm[2] * dot(first, nrm)]);
    let lt = norm(t).max(1e-300);
    let e1 = [t[0] / lt, t[1] / lt, t[2] / lt];
    let e2 = cross(nrm, e1);
    out.extend(pts.iter().map(|&p| [dot(sub(p, c), e1), dot(sub(p, c), e2)]));
}

/// One ARAP term: `w |(u_b - u_a) - R e|²` inside element `elem`.
#[derive(Clone, Copy)]
struct Term {
    a: u32,
    b: u32,
    w: f64,
    e: [f64; 2],
}

/// A triangle used for flip and stretch tests, with its reference corners.
#[derive(Clone, Copy)]
struct TestTri {
    v: [u32; 3],
    r: [[f64; 2]; 3],
    face: u32,
}

struct Elements {
    terms: Vec<Term>,
    start: Vec<usize>, // terms of element e: start[e]..start[e + 1]
    tris: Vec<TestTri>,
}

/// Elements of a chart: triangles use clamped cotangent weights (Pinkall, Polthier 1993 form, >= 0); polygons with
/// four or more sides use, per side, the cotangent weight of the rectangle with that side and the mean of its two
/// neighbours: `0.5 * mean(adjacent) / len` (ours). Quads give four test triangles (both diagonals), larger polygons
/// a fan.
fn elements(chart: &Chart) -> Elements {
    elements_with(chart, true)
}

/// `elements`; with `with_terms` false only the test triangles (what `stretch` needs).
fn elements_with(chart: &Chart, with_terms: bool) -> Elements {
    let nf = chart.faces.count();
    let (mut terms, mut start, mut tris) = (Vec::with_capacity(4 * nf), Vec::with_capacity(nf + 1), Vec::with_capacity(4 * nf));
    start.push(0usize);
    let (mut pts, mut q, mut len) = (Vec::new(), Vec::new(), Vec::new());
    for f in 0..nf {
        let poly = chart.faces.face(f);
        let k = poly.len();
        if k < 3 {
            start.push(terms.len());
            continue;
        }
        flat_polygon(chart, poly, &mut pts, &mut q);
        if !with_terms {
            push_tris(&mut tris, poly, &q, f, k);
            continue;
        }
        len.clear();
        len.extend((0..k).map(|i| {
            let (a, b) = (q[i], q[(i + 1) % k]);
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
        }));
        for i in 0..k {
            let j = (i + 1) % k;
            let w = if k == 3 {
                // Side i..j is opposite corner o.
                let o = (i + 2) % 3;
                let (u, v) = ([q[i][0] - q[o][0], q[i][1] - q[o][1]], [q[j][0] - q[o][0], q[j][1] - q[o][1]]);
                let area2 = cross2(u, v).abs();
                if area2 < 1e-300 { 0.0 } else { (0.5 * (u[0] * v[0] + u[1] * v[1]) / area2).max(0.0) }
            } else if len[i] < 1e-300 {
                0.0
            } else {
                0.25 * (len[(i + k - 1) % k] + len[j]) / len[i]
            };
            terms.push(Term { a: poly[i] as u32, b: poly[j] as u32, w, e: [q[j][0] - q[i][0], q[j][1] - q[i][1]] });
        }
        start.push(terms.len());
        push_tris(&mut tris, poly, &q, f, k);
    }
    Elements { terms, start, tris }
}

/// A polygon's test triangles: a triangle itself, a quad both diagonals (4), larger polygons a fan.
fn push_tris(tris: &mut Vec<TestTri>, poly: &[i64], q: &[[f64; 2]], f: usize, k: usize) {
    let mut push = |c: [usize; 3]| tris.push(TestTri { v: [poly[c[0]] as u32, poly[c[1]] as u32, poly[c[2]] as u32], r: [q[c[0]], q[c[1]], q[c[2]]], face: f as u32 });
    match k {
        3 => push([0, 1, 2]),
        4 => [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]].into_iter().for_each(&mut push),
        _ => (1..k - 1).for_each(|i| push([0, i, i + 1])),
    }
}

fn uv_at(uv: &[f64], v: u32) -> [f64; 2] {
    [uv[2 * v as usize], uv[2 * v as usize + 1]]
}

fn tri_area(uv: &[f64], t: &[u32; 3]) -> f64 {
    let (a, b, c) = (uv_at(uv, t[0]), uv_at(uv, t[1]), uv_at(uv, t[2]));
    0.5 * cross2([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]])
}

/// Faces whose UV polygon has a non-positive signed area (shoelace).
pub fn count_flips(faces: &Faces, uv: &[f64]) -> usize {
    (0..faces.count())
        .filter(|&f| {
            let poly = faces.face(f);
            let k = poly.len();
            let mut s = 0.0;
            for i in 0..k {
                let (a, b) = (uv_at(uv, poly[i] as u32), uv_at(uv, poly[(i + 1) % k] as u32));
                s += cross2(a, b);
            }
            k >= 3 && !(s > 0.0)
        })
        .count()
}

/// Chart stretch: the UVs are scaled so the UV area equals the surface area, then per face the worst of its test
/// triangles' `max(σ1, 1/σ2)` (Sander, Snyder, Gortler, Hoppe 2001 singular values); returns (`STRETCH_QUANTILE`
/// over faces, per face). 1 = isometric.
pub fn stretch(chart: &Chart, uv: &[f64]) -> (f64, Vec<f64>) {
    stretch_of(&elements_with(chart, false), chart.faces.count(), uv)
}

fn stretch_of(el: &Elements, nfaces: usize, uv: &[f64]) -> (f64, Vec<f64>) {
    let mut per = vec![0f64; nfaces];
    let (mut a3, mut a2) = (0.0, 0.0);
    for t in &el.tris {
        let r = t.r;
        a3 += 0.5 * cross2([r[1][0] - r[0][0], r[1][1] - r[0][1]], [r[2][0] - r[0][0], r[2][1] - r[0][1]]).abs();
        a2 += tri_area(uv, &t.v).abs();
    }
    if a3 <= 0.0 || a2 <= 0.0 {
        return (f64::INFINITY, vec![f64::INFINITY; nfaces]);
    }
    let s = (a3 / a2).sqrt();
    for t in &el.tris {
        let r = t.r;
        let (e1, e2) = ([r[1][0] - r[0][0], r[1][1] - r[0][1]], [r[2][0] - r[0][0], r[2][1] - r[0][1]]);
        let det = cross2(e1, e2);
        let (p0, p1, p2) = (uv_at(uv, t.v[0]), uv_at(uv, t.v[1]), uv_at(uv, t.v[2]));
        let (f1, f2) = ([s * (p1[0] - p0[0]), s * (p1[1] - p0[1])], [s * (p2[0] - p0[0]), s * (p2[1] - p0[1])]);
        let value = if det.abs() < 1e-300 {
            continue;
        } else {
            // J = F E⁻¹ (2x2), singular values from the closed form.
            let inv = [[e2[1] / det, -e2[0] / det], [-e1[1] / det, e1[0] / det]];
            let j = [
                [f1[0] * inv[0][0] + f2[0] * inv[1][0], f1[0] * inv[0][1] + f2[0] * inv[1][1]],
                [f1[1] * inv[0][0] + f2[1] * inv[1][0], f1[1] * inv[0][1] + f2[1] * inv[1][1]],
            ];
            let (a, b, c, d) = (j[0][0], j[0][1], j[1][0], j[1][1]);
            let e = (a + d) / 2.0;
            let ff = (a - d) / 2.0;
            let g = (c + b) / 2.0;
            let h = (c - b) / 2.0;
            let qq = (e * e + h * h).sqrt();
            let rr = (ff * ff + g * g).sqrt();
            let (s1, s2) = (qq + rr, (qq - rr).abs());
            if cross2(f1, f2) * det <= 0.0 || s2 < 1e-300 { f64::INFINITY } else { s1.max(1.0 / s2) }
        };
        let f = t.face as usize;
        per[f] = per[f].max(value);
    }
    (quantile(&per, STRETCH_QUANTILE), per)
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    let at = ((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
    v.select_nth_unstable_by(at, |a, b| a.total_cmp(b));
    v[at]
}

/// Largest flip-free share of the step `d` (Smith, Schaefer 2015): per positive test triangle the smallest positive
/// root of its signed area along the step; `STEP_SHARE` of the smallest, at most 1.
fn injective_step(tris: &[TestTri], uv: &[f64], d: &[f64]) -> f64 {
    let mut t_min = f64::INFINITY;
    for t in tris {
        let (p0, p1, p2) = (uv_at(uv, t.v[0]), uv_at(uv, t.v[1]), uv_at(uv, t.v[2]));
        let (d0, d1, d2) = (uv_at(d, t.v[0]), uv_at(d, t.v[1]), uv_at(d, t.v[2]));
        let (e1, e2) = ([p1[0] - p0[0], p1[1] - p0[1]], [p2[0] - p0[0], p2[1] - p0[1]]);
        let (f1, f2) = ([d1[0] - d0[0], d1[1] - d0[1]], [d2[0] - d0[0], d2[1] - d0[1]]);
        let c = cross2(e1, e2);
        if c <= 0.0 {
            continue; // not positive at the start (a non-convex quad's diagonal): nothing to keep
        }
        let b = cross2(e1, f2) + cross2(f1, e2);
        let a = cross2(f1, f2);
        let root = smallest_positive_root(a, b, c);
        if root < t_min {
            t_min = root;
        }
    }
    (STEP_SHARE * t_min).min(1.0)
}

fn smallest_positive_root(a: f64, b: f64, c: f64) -> f64 {
    let scale = a.abs().max(b.abs()).max(c.abs());
    if a.abs() <= 1e-14 * scale {
        return if b < 0.0 { -c / b } else { f64::INFINITY };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return f64::INFINITY;
    }
    let q = -0.5 * (b + b.signum() * disc.sqrt());
    let (r1, r2) = (q / a, if q != 0.0 { c / q } else { f64::INFINITY });
    [r1, r2].into_iter().filter(|&r| r > 0.0).fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------------------------------------------
// Exact grid layout of a pole-free quad chart
// ---------------------------------------------------------------------------------------------------------------

pub struct Grid {
    pub uv: Vec<f64>,
    pub col: Vec<i64>,
    pub row: Vec<i64>,
}

/// Integer grid of a quad chart by a walk over faces (each quad p, q, r, s in its own order gets r = q + rot90(q - p),
/// s = p + rot90(q - p)); Err when a face is not a quad, the walk meets itself inconsistently (a pole, a closed ring)
/// or two vertices land on one grid point. Unit steps are then replaced by the mean 3D length of each column and row
/// strip. Unused vertices get NaN and label -1.
pub fn grid_layout(chart: &Chart) -> Result<Grid, String> {
    let n = chart.n();
    let nf = chart.faces.count();
    let mut ij: Vec<Option<(i64, i64)>> = vec![None; n];
    let mut edge_faces: HashMap<(i64, i64), Vec<u32>, FxBuild> = HashMap::default();
    for f in 0..nf {
        let poly = chart.faces.face(f);
        if poly.len() != 4 {
            return Err("not every face is a quad".into());
        }
        for i in 0..4 {
            let (a, b) = (poly[i], poly[(i + 1) % 4]);
            edge_faces.entry((a.min(b), a.max(b))).or_default().push(f as u32);
        }
    }
    let mut done = vec![false; nf];
    let mut queue: Vec<usize> = Vec::new();
    for seed in 0..nf {
        if done[seed] {
            continue;
        }
        if seed > 0 {
            return Err("chart is not connected".into());
        }
        let poly = chart.faces.face(seed);
        for (c, &(i, j)) in [(0i64, 0i64), (1, 0), (1, 1), (0, 1)].iter().enumerate() {
            ij[poly[c] as usize] = Some((i, j));
        }
        done[seed] = true;
        queue.push(seed);
        let mut head = 0;
        while head < queue.len() {
            let f = queue[head];
            head += 1;
            let poly = chart.faces.face(f);
            for i in 0..4 {
                let (a, b) = (poly[i], poly[(i + 1) % 4]);
                for &g in &edge_faces[&(a.min(b), a.max(b))] {
                    let g = g as usize;
                    if done[g] {
                        continue;
                    }
                    let gp = chart.faces.face(g);
                    // In g the shared side runs b -> a; place the other two corners from it.
                    let at = (0..4).find(|&c| gp[c] == b && gp[(c + 1) % 4] == a).ok_or("faces are not consistently oriented")?;
                    let (p, q) = (ij[b as usize].unwrap(), ij[a as usize].unwrap());
                    let step = (q.0 - p.0, q.1 - p.1);
                    let rot = (-step.1, step.0);
                    let r = (q.0 + rot.0, q.1 + rot.1);
                    let s = (p.0 + rot.0, p.1 + rot.1);
                    for (v, want) in [(gp[(at + 2) % 4], r), (gp[(at + 3) % 4], s)] {
                        match ij[v as usize] {
                            None => ij[v as usize] = Some(want),
                            Some(have) if have != want => return Err("rows do not close: a pole or a ring inside".into()),
                            _ => {}
                        }
                    }
                    done[g] = true;
                    queue.push(g);
                }
            }
        }
    }
    let mut taken: HashMap<(i64, i64), usize, FxBuild> = HashMap::default();
    for (v, x) in ij.iter().enumerate() {
        if let Some(k) = x {
            if taken.insert(*k, v).is_some() {
                return Err("two vertices on one grid point".into());
            }
        }
    }
    let used: Vec<(i64, i64)> = ij.iter().flatten().copied().collect();
    if used.is_empty() {
        return Ok(Grid { uv: vec![f64::NAN; 2 * n], col: vec![-1; n], row: vec![-1; n] });
    }
    let (i0, i1) = (used.iter().map(|x| x.0).min().unwrap(), used.iter().map(|x| x.0).max().unwrap());
    let (j0, j1) = (used.iter().map(|x| x.1).min().unwrap(), used.iter().map(|x| x.1).max().unwrap());
    let (ni, nj) = ((i1 - i0) as usize, (j1 - j0) as usize);
    let (mut wsum, mut wcnt, mut hsum, mut hcnt) = (vec![0f64; ni], vec![0usize; ni], vec![0f64; nj], vec![0usize; nj]);
    for (&(a, b), _) in edge_faces.iter() {
        let (pa, pb) = (ij[a as usize].unwrap(), ij[b as usize].unwrap());
        let len = norm(sub(chart.p(a as usize), chart.p(b as usize)));
        if pa.1 == pb.1 {
            let k = (pa.0.min(pb.0) - i0) as usize;
            wsum[k] += len;
            wcnt[k] += 1;
        } else {
            let k = (pa.1.min(pb.1) - j0) as usize;
            hsum[k] += len;
            hcnt[k] += 1;
        }
    }
    let prefix = |sum: &[f64], cnt: &[usize]| {
        let mut out = vec![0f64; sum.len() + 1];
        for k in 0..sum.len() {
            out[k + 1] = out[k] + if cnt[k] > 0 { sum[k] / cnt[k] as f64 } else { 0.0 };
        }
        out
    };
    let (us, vs) = (prefix(&wsum, &wcnt), prefix(&hsum, &hcnt));
    let mut g = Grid { uv: vec![f64::NAN; 2 * n], col: vec![-1; n], row: vec![-1; n] };
    for (v, x) in ij.iter().enumerate() {
        if let Some((i, j)) = *x {
            g.col[v] = i - i0;
            g.row[v] = j - j0;
            g.uv[2 * v] = us[(i - i0) as usize];
            g.uv[2 * v + 1] = vs[(j - j0) as usize];
        }
    }
    Ok(g)
}

// ---------------------------------------------------------------------------------------------------------------
// Loops (label-free): consecutive triples along quad rows
// ---------------------------------------------------------------------------------------------------------------

/// Triples (a, v, b) where a - v - b continues one quad row straight through v: interior vertices with four quads
/// (the two opposite pairs) and border vertices with two quads (the two border neighbours). Poles end rows.
pub fn loop_triples(faces: &Faces, n: usize) -> Vec<[u32; 3]> {
    // For every corner of a quad: (vertex, previous, next).
    let mut around: Vec<Vec<(i64, i64, bool)>> = vec![Vec::new(); n];
    for f in 0..faces.count() {
        let poly = faces.face(f);
        let k = poly.len();
        for c in 0..k {
            around[poly[c] as usize].push((poly[(c + k - 1) % k], poly[(c + 1) % k], k == 4));
        }
    }
    let mut out = Vec::new();
    for (v, fan) in around.iter().enumerate() {
        if fan.is_empty() || fan.iter().any(|x| !x.2) {
            continue;
        }
        let mut nb: Vec<i64> = fan.iter().flat_map(|x| [x.0, x.1]).collect();
        nb.sort_unstable();
        nb.dedup();
        let shares = |a: i64, b: i64| fan.iter().any(|x| (x.0 == a && x.1 == b) || (x.0 == b && x.1 == a));
        if fan.len() == 4 && nb.len() == 4 {
            for i in 0..4 {
                for j in i + 1..4 {
                    if !shares(nb[i], nb[j]) && nb[i] < nb[j] {
                        out.push([nb[i] as u32, v as u32, nb[j] as u32]);
                    }
                }
            }
        } else if fan.len() == 2 && nb.len() == 3 {
            // Border vertex with two quads: the neighbour in both quads is the inner one.
            let inner = nb.iter().copied().find(|&w| fan.iter().all(|x| x.0 == w || x.1 == w));
            if let Some(inner) = inner {
                let ends: Vec<i64> = nb.iter().copied().filter(|&w| w != inner).collect();
                out.push([ends[0] as u32, v as u32, ends[1] as u32]);
            }
        }
    }
    out
}

/// 95th percentile of the turning angle (degrees) along the triples.
pub fn bend_p95(triples: &[[u32; 3]], uv: &[f64]) -> f64 {
    let angles: Vec<f64> = triples
        .iter()
        .map(|t| {
            let (a, b, c) = (uv_at(uv, t[0]), uv_at(uv, t[1]), uv_at(uv, t[2]));
            let (x, y) = ([b[0] - a[0], b[1] - a[1]], [c[0] - b[0], c[1] - b[1]]);
            cross2(x, y).atan2(x[0] * y[0] + x[1] * y[1]).abs().to_degrees()
        })
        .collect();
    quantile(&angles, 0.95)
}

// ---------------------------------------------------------------------------------------------------------------
// Quad ARAP with rows, injective steps
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rows {
    Off,
    Soft,
    On,
    Auto,
}

pub struct ArapReport {
    pub steps: usize,
    pub energy: f64,
    pub path: Path,
    pub nnz_l: usize,
    pub factor_ms: f64,
    pub solve_ms: f64,
    pub stalled: usize,
    pub stopped: bool,
}

/// Unknown of every vertex for the u and v systems (On: shared per column / row label).
struct Maps {
    u: Vec<u32>,
    v: Vec<u32>,
    nu: usize,
    nv: usize,
}

fn identity_maps(n: usize) -> Maps {
    let id: Vec<u32> = (0..n as u32).collect();
    Maps { u: id.clone(), v: id, nu: n, nv: n }
}

fn label_maps(col: &[i64], row: &[i64]) -> Maps {
    let squeeze = |lab: &[i64]| {
        let mut seen: HashMap<i64, u32, FxBuild> = HashMap::default();
        let ids: Vec<u32> = lab.iter().map(|&l| { let k = seen.len() as u32; *seen.entry(l).or_insert(k) }).collect();
        (ids, seen.len())
    };
    let ((u, nu), (v, nv)) = (squeeze(col), squeeze(row));
    Maps { u, v, nu, nv }
}

/// Runs quad ARAP on `uv` in place. `steps` = 0 runs until the energy changes by less than `ARAP_CHANGE`; `held`
/// vertices stay where they are (none: one vertex is pinned, the translation null space). `soft` adds the row penalty
/// with that weight along `triples`; `maps` = On (rows share one unknown).
#[allow(clippy::too_many_arguments)]
fn arap_run(
    el: &Elements,
    n: usize,
    uv: &mut [f64],
    held: Option<&[bool]>,
    triples: &[[u32; 3]],
    soft: f64,
    on: Option<&Maps>,
    steps: usize,
    tol: f64,
    max_bytes: f64,
    force_iterative: bool,
    stop: &AtomicBool,
) -> Result<ArapReport, String> {
    let ident;
    let maps = match on {
        Some(m) => m,
        None => {
            ident = identity_maps(n);
            &ident
        }
    };
    let shared = on.is_none();
    let pins_for = |map: &[u32]| -> Vec<usize> {
        let mut p: Vec<usize> = match held {
            Some(h) => (0..n).filter(|&v| h[v]).map(|v| map[v] as usize).collect(),
            None => Vec::new(),
        };
        if p.is_empty() {
            p.push(map[0] as usize);
        }
        p.sort_unstable();
        p.dedup();
        p
    };
    let build = |map: &[u32], m: usize| {
        let mut b = Builder::new(m);
        b.t.reserve(4 * el.terms.len() + m + if soft > 0.0 { 9 * triples.len() } else { 0 });
        for t in &el.terms {
            if t.w > 0.0 {
                b.edge(map[t.a as usize] as usize, map[t.b as usize] as usize, t.w);
            }
        }
        if soft > 0.0 {
            for t in triples {
                let (a, c, d) = (map[t[0] as usize] as usize, map[t[1] as usize] as usize, map[t[2] as usize] as usize);
                let coef = [(a, 1.0), (c, -2.0), (d, 1.0)];
                for (x, &(i, ci)) in coef.iter().enumerate() {
                    b.add(i, i, soft * ci * ci);
                    for &(j, cj) in &coef[x + 1..] {
                        // (i, j) and (j, i); on one unknown both land on the diagonal
                        b.add(i, j, if i == j { 2.0 } else { 1.0 } * soft * ci * cj);
                    }
                }
            }
        }
        for i in 0..m {
            b.add(i, i, 1e-12); // keeps an isolated unknown (zero-length sides) solvable; far below every real weight
        }
        b.build()
    };
    let pins_u = pins_for(&maps.u);
    let lin_u = Lin::new(build(&maps.u, maps.nu), &pins_u, max_bytes, force_iterative, stop)?;
    let (lin_v, pins_v) = if shared {
        (None, pins_u.clone())
    } else {
        let p = pins_for(&maps.v);
        (Some(Lin::new(build(&maps.v, maps.nv), &p, max_bytes, force_iterative, stop)?), p)
    };
    let factor_ms = lin_u.factor_ms + lin_v.as_ref().map_or(0.0, |l| l.factor_ms);
    let nnz_l = lin_u.nnz_l + lin_v.as_ref().map_or(0, |l| l.nnz_l);
    let nel = el.start.len() - 1;
    let zero = 1e-20 * el.terms.iter().map(|t| t.w * (t.e[0] * t.e[0] + t.e[1] * t.e[1])).sum::<f64>();
    let mut report = ArapReport { steps: 0, energy: 0.0, path: lin_u.path, nnz_l, factor_ms, solve_ms: 0.0, stalled: 0, stopped: false };
    let mut last = f64::INFINITY;
    let mut new = vec![0f64; 2 * n];
    let (mut rot, mut rhs, mut d): (Vec<[f64; 2]>, Vec<f64>, Vec<f64>) = (Vec::with_capacity(nel), vec![0f64; 2 * n], vec![0f64; 2 * n]);
    loop {
        // Local step: one rotation per element (2x2 Procrustes: the angle of Σ w du eᵀ's rotational part).
        let fit = |e: usize| {
            let mut m = [0f64; 4];
            for t in &el.terms[el.start[e]..el.start[e + 1]] {
                let (pa, pb) = (uv_at(uv, t.a), uv_at(uv, t.b));
                let du = [pb[0] - pa[0], pb[1] - pa[1]];
                m[0] += t.w * du[0] * t.e[0];
                m[1] += t.w * du[0] * t.e[1];
                m[2] += t.w * du[1] * t.e[0];
                m[3] += t.w * du[1] * t.e[1];
            }
            // The angle's cosine and sine straight from the vector (no atan2/cos/sin); zero gives the identity.
            let (x, y) = (m[0] + m[3], m[2] - m[1]);
            let h = (x * x + y * y).sqrt();
            if h == 0.0 { [1.0, 0.0] } else { [x / h, y / h] }
        };
        if nel >= PARALLEL_MIN {
            rot.clear();
            rot.par_extend((0..nel).into_par_iter().map(fit));
        } else {
            rot.clear();
            rot.extend((0..nel).map(fit));
        }
        let mut energy = 0.0;
        for e in 0..nel {
            let [c, s] = rot[e];
            for t in &el.terms[el.start[e]..el.start[e + 1]] {
                let (pa, pb) = (uv_at(uv, t.a), uv_at(uv, t.b));
                let re = [c * t.e[0] - s * t.e[1], s * t.e[0] + c * t.e[1]];
                energy += t.w * ((pb[0] - pa[0] - re[0]).powi(2) + (pb[1] - pa[1] - re[1]).powi(2));
            }
        }
        report.energy = energy;
        let converged = if steps > 0 {
            report.steps >= steps
        } else {
            report.steps > 0 && (last - energy).abs() <= ARAP_CHANGE * last.max(1e-300)
        };
        if converged || energy <= zero || report.stopped {
            break;
        }
        last = energy;
        // Global step: one solve with the fixed matrix.
        let started = Instant::now();
        let mut info = SolveInfo::default();
        if shared {
            rhs.fill(0.0);
            for e in 0..nel {
                let [c, s] = rot[e];
                for t in &el.terms[el.start[e]..el.start[e + 1]] {
                    let re = [t.w * (c * t.e[0] - s * t.e[1]), t.w * (s * t.e[0] + c * t.e[1])];
                    for r in 0..2 {
                        rhs[2 * t.a as usize + r] -= re[r];
                        rhs[2 * t.b as usize + r] += re[r];
                    }
                }
            }
            for &p in &pins_u {
                rhs[2 * p] = uv[2 * p];
                rhs[2 * p + 1] = uv[2 * p + 1];
            }
            new.copy_from_slice(uv);
            info = lin_u.solve::<2>(&mut rhs, &mut new, tol, stop);
        } else {
            for (axis, (map, m, lin, pins)) in [(&maps.u, maps.nu, &lin_u, &pins_u), (&maps.v, maps.nv, lin_v.as_ref().unwrap(), &pins_v)].into_iter().enumerate() {
                let (mut rhs, mut x, mut cnt) = (vec![0f64; m], vec![0f64; m], vec![0usize; m]);
                for e in 0..nel {
                    let [c, s] = rot[e];
                    for t in &el.terms[el.start[e]..el.start[e + 1]] {
                        let re = if axis == 0 { c * t.e[0] - s * t.e[1] } else { s * t.e[0] + c * t.e[1] };
                        rhs[map[t.a as usize] as usize] -= t.w * re;
                        rhs[map[t.b as usize] as usize] += t.w * re;
                    }
                }
                for v in 0..n {
                    let k = map[v] as usize;
                    x[k] += uv[2 * v + axis];
                    cnt[k] += 1;
                }
                for k in 0..m {
                    x[k] /= cnt[k].max(1) as f64;
                }
                for &p in pins.iter() {
                    rhs[p] = x[p];
                }
                let got = lin.solve::<1>(&mut rhs, &mut x, tol, stop);
                info.iterations += got.iterations;
                info.stalled += got.stalled;
                info.stopped |= got.stopped;
                for v in 0..n {
                    new[2 * v + axis] = x[map[v] as usize];
                }
            }
        }
        let ms = started.elapsed().as_secs_f64() * 1e3;
        if report.steps == 1 || report.solve_ms == 0.0 {
            report.solve_ms = ms; // the second solve is the first warm one; the first stands in until then
        }
        report.stalled += info.stalled;
        if info.stopped || stop.load(Ordering::Relaxed) {
            report.stopped = true;
            break;
        }
        for i in 0..2 * n {
            d[i] = new[i] - uv[i];
        }
        let t = injective_step(&el.tris, uv, &d);
        for i in 0..2 * n {
            uv[i] += t * d[i];
        }
        report.steps += 1;
    }
    Ok(report)
}

// ---------------------------------------------------------------------------------------------------------------
// Starts: LSCM (pull or two pins) and Tutte
// ---------------------------------------------------------------------------------------------------------------

/// Directed border sides (a -> b with no b -> a) chained into loops; the longest loop first.
fn border_loops(faces: &Faces, n: usize) -> Vec<Vec<u32>> {
    let mut sides: HashMap<(i64, i64), (), FxBuild> = HashMap::default();
    for f in 0..faces.count() {
        let poly = faces.face(f);
        for i in 0..poly.len() {
            sides.insert((poly[i], poly[(i + 1) % poly.len()]), ());
        }
    }
    let mut next = vec![NONE; n];
    for &(a, b) in sides.keys() {
        if !sides.contains_key(&(b, a)) {
            next[a as usize] = b as usize;
        }
    }
    let mut seen = vec![false; n];
    let mut loops = Vec::new();
    for s in 0..n {
        if next[s] == NONE || seen[s] {
            continue;
        }
        let mut lp = Vec::new();
        let mut v = s;
        while v != NONE && !seen[v] {
            seen[v] = true;
            lp.push(v as u32);
            v = next[v];
        }
        loops.push(lp);
    }
    loops.sort_by_key(|l| std::cmp::Reverse(l.len()));
    loops
}

/// Tutte embedding (uniform weights; Tutte 1963, Floater 1997): the longest border loop on a circle by arc length,
/// counter-clockwise, every other vertex the mean of its neighbours. Flip-free on a disk.
pub fn tutte(chart: &Chart, max_bytes: f64, stop: &AtomicBool) -> Result<Vec<f64>, String> {
    let n = chart.n();
    let loops = border_loops(&chart.faces, n);
    let border = loops.first().ok_or("chart has no border (closed): it needs a cut")?;
    let mut b = Builder::new(n);
    for f in 0..chart.faces.count() {
        let poly = chart.faces.face(f);
        for i in 0..poly.len() {
            let (a, c) = (poly[i] as usize, poly[(i + 1) % poly.len()] as usize);
            b.edge(a, c, 0.5); // an inner side is seen from both of its faces: 0.5 + 0.5
        }
    }
    let lens: Vec<f64> = (0..border.len()).map(|i| norm(sub(chart.p(border[i] as usize), chart.p(border[(i + 1) % border.len()] as usize)))).collect();
    let total: f64 = lens.iter().sum::<f64>().max(1e-300);
    let radius = total / std::f64::consts::TAU;
    let mut uv = vec![0f64; 2 * n];
    let mut acc = 0.0;
    let pins: Vec<usize> = border.iter().map(|&v| v as usize).collect();
    for (i, &v) in border.iter().enumerate() {
        let th = std::f64::consts::TAU * acc / total;
        uv[2 * v as usize] = radius * th.cos();
        uv[2 * v as usize + 1] = radius * th.sin();
        acc += lens[i];
    }
    let lin = Lin::new(b.build(), &pins, max_bytes, false, stop)?;
    let mut rhs = vec![0f64; 2 * n];
    for &p in &pins {
        rhs[2 * p] = uv[2 * p];
        rhs[2 * p + 1] = uv[2 * p + 1];
    }
    let mut x = uv.clone();
    lin.solve::<2>(&mut rhs, &mut x, radius * 1e-9, stop);
    Ok(x)
}

/// LSCM (Lévy et al. 2002): per triangle, area-weighted Cauchy-Riemann residual `|J∇u - ∇v|²` on the polygon's own
/// plane (quads as both diagonals at half weight). With `pull = Some(g)`: `+ w Σ|x_i - g_i|²`, no pins (ours); else
/// two far border vertices pinned at their 3D distance.
pub fn lscm(chart: &Chart, pull: Option<&[f64]>, max_bytes: f64, stop: &AtomicBool) -> Result<Vec<f64>, String> {
    let n = chart.n();
    let el = elements(chart);
    let mut b = Builder::new(2 * n);
    // Conformal energy = Dirichlet energy - 2 * UV area (Lévy et al. 2002; Desbrun, Meyer, Alliez 2002). Dirichlet: per
    // triangle side (cot of the opposite corner) / 2 on u and on v, unclamped; quads as both diagonals at half weight.
    for t in &el.tris {
        let r = t.r;
        let k = chart.faces.face(t.face as usize).len();
        let weight = if k == 4 { 0.5 } else { 1.0 };
        for c in 0..3 {
            let (i, j) = ((c + 1) % 3, (c + 2) % 3);
            let (u, v) = ([r[i][0] - r[c][0], r[i][1] - r[c][1]], [r[j][0] - r[c][0], r[j][1] - r[c][1]]);
            let area2 = cross2(u, v);
            if area2.abs() < 1e-300 {
                continue;
            }
            let w = weight * 0.5 * (u[0] * v[0] + u[1] * v[1]) / area2;
            let (a, bb) = (t.v[i] as usize, t.v[j] as usize);
            b.edge(2 * a, 2 * bb, w);
            b.edge(2 * a + 1, 2 * bb + 1, w);
        }
    }
    // UV area = 1/2 Σ over border sides i -> j of (u_i v_j - u_j v_i): only the border couples u with v.
    for lp in border_loops(&chart.faces, n) {
        for k in 0..lp.len() {
            let (i, j) = (lp[k] as usize, lp[(k + 1) % lp.len()] as usize);
            b.add(2 * i, 2 * j + 1, -0.5);
            b.add(2 * j, 2 * i + 1, 0.5);
        }
    }
    let mut a = b.build();
    let mut rhs = vec![0f64; 2 * n];
    let mut x0 = vec![0f64; 2 * n];
    let pins: Vec<usize> = match pull {
        Some(g) => {
            let mean = a.diag().iter().sum::<f64>() / (2 * n).max(1) as f64;
            let w = PULL_SHARE * mean.max(1e-300);
            for i in 0..2 * n {
                rhs[i] = w * g[i];
            }
            // Merge the pull into `a` (same pattern on the diagonal).
            for i in 0..2 * n {
                let (s, e) = (a.ptr[i], a.ptr[i + 1]);
                match a.idx[s..e].iter().position(|&j| j as usize == i) {
                    Some(p) => a.val[s + p] += w,
                    None => return Err("conformal matrix lacks a diagonal".into()),
                }
            }
            x0.copy_from_slice(g);
            Vec::new()
        }
        None => {
            let loops = border_loops(&chart.faces, n);
            let border = loops.first().ok_or("chart has no border (closed): it needs a cut")?;
            let far = |from: usize| border.iter().copied().max_by(|&a, &b| norm(sub(chart.p(a as usize), chart.p(from))).total_cmp(&norm(sub(chart.p(b as usize), chart.p(from))))).unwrap() as usize;
            let b0 = far(border[0] as usize);
            let b1 = far(b0);
            if b0 == b1 {
                return Err("border too small to pin".into());
            }
            let dist = norm(sub(chart.p(b0), chart.p(b1)));
            rhs[2 * b1] = dist;
            vec![2 * b0, 2 * b0 + 1, 2 * b1, 2 * b1 + 1]
        }
    };
    let lin = Lin::new(a, &pins, max_bytes, false, stop)?;
    let scale = (0..n).map(|v| norm(chart.p(v))).fold(0.0, f64::max).max(1e-300);
    lin.solve::<1>(&mut rhs, &mut x0, scale * 1e-9, stop);
    Ok(x0)
}

// ---------------------------------------------------------------------------------------------------------------
// Flatten one chart, many charts, quick stretch
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum Start {
    /// Use the given UVs (a glued grid); Tutte when they flip or are missing.
    Given,
    /// LSCM pulled toward the given UVs (pole charts); the given UVs, then Tutte, when that flips.
    Pull,
    /// LSCM with two far border vertices pinned; Tutte when it flips.
    Free,
}

#[derive(Clone, Copy)]
pub struct FlatOpts {
    pub rows: Rows,
    pub start: Start,
    /// Stretch allowed as a share (0.10 = 10 %): Auto keeps On only when the chart's stretch stays within `1 + this`.
    pub stretch_allowed: f64,
    /// ARAP steps; 0 = until the energy changes by less than `ARAP_CHANGE`.
    pub steps: usize,
    pub max_factor_bytes: f64,
    pub texture_px: f64,
    pub force_iterative: bool,
}

impl Default for FlatOpts {
    fn default() -> Self {
        FlatOpts { rows: Rows::Auto, start: Start::Given, stretch_allowed: 0.10, steps: 0, max_factor_bytes: 0.0, texture_px: TEXTURE_PX, force_iterative: false }
    }
}

pub struct FlatOut {
    pub uv: Vec<f64>,
    pub stats: [f64; STATS_LEN],
}

fn finite(uv: &[f64]) -> bool {
    uv.iter().all(|x| x.is_finite())
}

/// On is possible only when every face is a quad whose sides alternate between "same column" and "same row".
fn labels_fit(faces: &Faces, col: &[i64], row: &[i64]) -> bool {
    (0..faces.count()).all(|f| {
        let p = faces.face(f);
        if p.len() != 4 {
            return false;
        }
        let side = |i: usize| {
            let (a, b) = (p[i] as usize, p[(i + 1) % 4] as usize);
            (col[a] == col[b], row[a] == row[b])
        };
        let s: Vec<(bool, bool)> = (0..4).map(side).collect();
        s.iter().all(|x| x.0 != x.1) && s[0].0 != s[1].0 && s[0] == s[2] && s[1] == s[3]
    })
}

/// Flatten one chart (section 6.5): start, quad ARAP with rows, injective steps. `start` per vertex (2n), `labels`
/// (column, row) per vertex for On (-1 = none). Returns UVs per vertex; a stop returns the flip-free state reached.
pub fn flatten_chart(chart: &Chart, start: Option<&[f64]>, labels: Option<(&[i64], &[i64])>, opts: &FlatOpts, stop: &AtomicBool) -> Result<FlatOut, String> {
    let started = Instant::now();
    let n = chart.n();
    let mut stats = [f64::NAN; STATS_LEN];
    stats[stat::VERTICES] = n as f64;
    stats[stat::FACES] = chart.faces.count() as f64;
    let given = start.filter(|s| s.len() == 2 * n && finite(s));
    let flip_free = |uv: &[f64]| count_flips(&chart.faces, uv) == 0;
    let mb = opts.max_factor_bytes;
    let (mut uv, kind) = match (opts.start, given) {
        (Start::Given, Some(g)) if flip_free(g) => (g.to_vec(), 0.0),
        (Start::Pull, Some(g)) => match lscm(chart, Some(g), mb, stop) {
            Ok(x) if flip_free(&x) => (x, 3.0),
            _ if flip_free(g) => (g.to_vec(), 0.0),
            _ => (tutte(chart, mb, stop)?, 2.0),
        },
        (Start::Free, _) | (_, None) => match lscm(chart, None, mb, stop) {
            Ok(x) if flip_free(&x) => (x, 1.0),
            _ => (tutte(chart, mb, stop)?, 2.0),
        },
        _ => (tutte(chart, mb, stop)?, 2.0),
    };
    stats[stat::START] = kind;
    let el = elements(chart);
    let triples = loop_triples(&chart.faces, n);
    let span = {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for v in 0..n {
            for r in 0..2 {
                lo[r] = lo[r].min(uv[2 * v + r]);
                hi[r] = hi[r].max(uv[2 * v + r]);
            }
        }
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt().max(1e-300)
    };
    let texture = if opts.texture_px > 0.0 { opts.texture_px } else { TEXTURE_PX };
    let tol = STOP_TEXEL_SHARE * span / texture;
    let labelled = labels.filter(|(c, r)| c.len() == n && r.len() == n && c.iter().chain(r.iter()).all(|&x| x >= 0) && labels_fit(&chart.faces, c, r));
    let want_on = match opts.rows {
        Rows::On | Rows::Auto => labelled.is_some(),
        _ => false,
    };
    let run = |uv: &mut Vec<f64>, mode: Rows, soft: f64| -> Result<ArapReport, String> {
        let maps = if mode == Rows::On { labelled.map(|(c, r)| label_maps(c, r)) } else { None };
        arap_run(&el, n, uv, None, &triples, if mode == Rows::Soft { soft } else { 0.0 }, maps.as_ref(), opts.steps, tol, mb, opts.force_iterative, stop)
    };
    let mut used = Rows::Off;
    let mut report = None;
    if want_on {
        let mut trial = uv.clone();
        let r = run(&mut trial, Rows::On, 0.0)?;
        let ok = opts.rows == Rows::On || stretch_of(&el, chart.faces.count(), &trial).0 <= 1.0 + opts.stretch_allowed;
        if ok || r.stopped {
            uv = trial;
            used = Rows::On;
            report = Some(r);
        }
    }
    if report.is_none() {
        let mut mode = match opts.rows {
            Rows::Off => Rows::Off,
            _ if triples.is_empty() => Rows::Off,
            _ => Rows::Soft,
        };
        if mode == Rows::Off {
            report = Some(run(&mut uv, Rows::Off, 0.0)?);
        } else {
            // Soft: raise the weight until the row bend target holds or stops improving.
            let mean_diag = {
                let mut d = vec![0f64; n];
                for t in &el.terms {
                    d[t.a as usize] += t.w;
                    d[t.b as usize] += t.w;
                }
                d.iter().sum::<f64>() / n.max(1) as f64
            };
            let mut w = SOFT_START * mean_diag;
            let base = uv.clone();
            let mut best: Option<(f64, Vec<f64>, ArapReport)> = None;
            loop {
                let mut trial = base.clone();
                let r = run(&mut trial, Rows::Soft, w)?;
                let bend = bend_p95(&triples, &trial);
                let improved = best.as_ref().map_or(true, |b| bend < (1.0 - STALL_DROP) * b.0);
                let stopped = r.stopped;
                if improved {
                    best = Some((bend, trial, r));
                }
                if bend <= BEND_P95_DEG || !improved || stopped {
                    break;
                }
                w *= SOFT_GROW;
            }
            let (_, trial, r) = best.unwrap();
            // Auto refuses Soft rows that break Stretch allowed, as it does for On: Off is the fallback.
            if opts.rows == Rows::Auto && !r.stopped && stretch_of(&el, chart.faces.count(), &trial).0 > 1.0 + opts.stretch_allowed {
                report = Some(run(&mut uv, Rows::Off, 0.0)?);
                mode = Rows::Off;
            } else {
                uv = trial;
                report = Some(r);
            }
        }
        used = mode;
    }
    let r = report.unwrap();
    stats[stat::PATH] = match r.path {
        Path::Direct => 0.0,
        Path::Ic0 => 1.0,
        Path::Jacobi => 2.0,
    };
    stats[stat::ROWS] = match used {
        Rows::Soft => 1.0,
        Rows::On => 2.0,
        _ => 0.0,
    };
    stats[stat::STEPS] = r.steps as f64;
    stats[stat::ENERGY] = r.energy;
    stats[stat::FLIPS] = count_flips(&chart.faces, &uv) as f64;
    stats[stat::STRETCH] = stretch_of(&el, chart.faces.count(), &uv).0;
    stats[stat::BEND] = if triples.is_empty() { f64::NAN } else { bend_p95(&triples, &uv) };
    stats[stat::FACTOR_MS] = r.factor_ms;
    stats[stat::SOLVE_MS] = r.solve_ms;
    stats[stat::STOPPED] = f64::from(u8::from(r.stopped));
    stats[stat::NNZ_L] = r.nnz_l as f64;
    stats[stat::STALLED] = r.stalled as f64;
    stats[stat::TOTAL_MS] = started.elapsed().as_secs_f64() * 1e3;
    Ok(FlatOut { uv, stats })
}

/// Local copy of the faces `ids` of a mesh: (points, indptr, flat, global vertex of each local one).
pub fn sub_chart(points: &[f64], faces: &Faces, ids: &[u32]) -> (Vec<f64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut local: HashMap<i64, i64, FxBuild> = HashMap::default();
    let (mut indptr, mut flat, mut global) = (vec![0i64], Vec::new(), Vec::new());
    for &f in ids {
        for &v in faces.face(f as usize) {
            let k = *local.entry(v).or_insert_with(|| {
                global.push(v);
                global.len() as i64 - 1
            });
            flat.push(k);
        }
        indptr.push(flat.len() as i64);
    }
    let pts = global.iter().flat_map(|&v| points[3 * v as usize..3 * v as usize + 3].iter().copied()).collect();
    (pts, indptr, flat, global)
}

/// Every chart of a mesh in one call: `chart[f]` (< 0 = skip), `start` per corner (2 per corner, NaN = none), `col`
/// and `row` per corner (-1 = none). Returns UVs per corner (NaN for skipped faces) and one stats row per chart id
/// (ascending), `STATS_LEN` numbers each; a chart that cannot be flattened gets NaN UVs and its stats row says so
/// (STEPS = -1). Charts run in parallel when the mesh is large.
#[allow(clippy::too_many_arguments)]
pub fn flatten(points: &[f64], faces: &Faces, chart: &[i64], start: &[f64], col: &[i64], row: &[i64], opts: &FlatOpts, stop: &AtomicBool) -> (Vec<f64>, Vec<f64>) {
    let mut ids: Vec<i64> = chart.iter().copied().filter(|&c| c >= 0).collect();
    ids.sort_unstable();
    ids.dedup();
    let slot: HashMap<i64, usize, FxBuild> = ids.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
    for (f, &c) in chart.iter().enumerate() {
        if c >= 0 {
            members[slot[&c]].push(f as u32);
        }
    }
    let one = |k: usize| -> (Vec<f64>, [f64; STATS_LEN]) {
        let (pts, indptr, flat, global) = sub_chart(points, faces, &members[k]);
        let n = global.len();
        let (mut s, mut c, mut r) = (vec![f64::NAN; 2 * n], vec![-1i64; n], vec![-1i64; n]);
        let mut corner = 0usize;
        for &f in &members[k] {
            let (lo, hi) = (faces.indptr[f as usize] as usize, faces.indptr[f as usize + 1] as usize);
            for h in lo..hi {
                let v = flat[corner] as usize;
                if s[2 * v].is_nan() && 2 * h + 1 < start.len() {
                    s[2 * v] = start[2 * h];
                    s[2 * v + 1] = start[2 * h + 1];
                }
                if h < col.len() && c[v] < 0 {
                    c[v] = col[h];
                }
                if h < row.len() && r[v] < 0 {
                    r[v] = row[h];
                }
                corner += 1;
            }
        }
        let ch = Chart { points: &pts, faces: Faces { indptr: &indptr, flat: &flat } };
        let has_start = finite(&s);
        let labels = if col.is_empty() { None } else { Some((&c[..], &r[..])) };
        match flatten_chart(&ch, if has_start { Some(&s) } else { None }, labels, opts, stop) {
            Ok(mut out) => {
                out.stats[stat::CHART] = ids[k] as f64;
                // Back to corners, in the members' face order.
                let mut uv = Vec::with_capacity(2 * flat.len());
                for &v in &flat {
                    uv.push(out.uv[2 * v as usize]);
                    uv.push(out.uv[2 * v as usize + 1]);
                }
                (uv, out.stats)
            }
            Err(_) => {
                let mut st = [f64::NAN; STATS_LEN];
                st[stat::CHART] = ids[k] as f64;
                st[stat::VERTICES] = n as f64;
                st[stat::FACES] = members[k].len() as f64;
                st[stat::STEPS] = -1.0;
                st[stat::STOPPED] = f64::from(u8::from(stop.load(Ordering::Relaxed)));
                (vec![f64::NAN; 2 * flat.len()], st)
            }
        }
    };
    let results: Vec<(Vec<f64>, [f64; STATS_LEN])> = if chart.len() >= PARALLEL_MIN && ids.len() > 1 {
        (0..ids.len()).into_par_iter().map(one).collect()
    } else {
        (0..ids.len()).map(one).collect()
    };
    let mut uv = vec![f64::NAN; 2 * faces.flat.len()];
    let mut stats = Vec::with_capacity(ids.len() * STATS_LEN);
    for (k, (cuv, st)) in results.into_iter().enumerate() {
        let mut at = 0;
        for &f in &members[k] {
            let (lo, hi) = (faces.indptr[f as usize] as usize, faces.indptr[f as usize + 1] as usize);
            for h in lo..hi {
                uv[2 * h] = cuv[2 * at];
                uv[2 * h + 1] = cuv[2 * at + 1];
                at += 1;
            }
        }
        stats.extend_from_slice(&st);
    }
    (uv, stats)
}

/// Stretch of a merge candidate without a full flatten (section 6.4): ARAP until converged on the vertices marked
/// `free` (the band around the shared border), every other vertex held at `uv`. Returns (stretch over the faces that
/// touch a free vertex, stretch over the whole chart) and leaves the band's UVs in `uv`.
pub fn quick_stretch(chart: &Chart, uv: &mut [f64], free: &[bool], max_bytes: f64, stop: &AtomicBool) -> Result<(f64, f64), String> {
    quick_stretch_per(chart, uv, free, max_bytes, stop).map(|(band, all, _)| (band, all))
}

/// `quick_stretch` that also returns the per-face stretch of the chart (what `stretch` gives for the final `uv`),
/// so a caller needing both does not run `stretch` a second time.
pub fn quick_stretch_per(chart: &Chart, uv: &mut [f64], free: &[bool], max_bytes: f64, stop: &AtomicBool) -> Result<(f64, f64, Vec<f64>), String> {
    let n = chart.n();
    if uv.len() != 2 * n || free.len() != n {
        return Err("quick_stretch arrays are inconsistent".into());
    }
    let el = elements(chart);
    let held: Vec<bool> = free.iter().map(|&f| !f).collect();
    let span = (0..n).map(|v| uv[2 * v].abs().max(uv[2 * v + 1].abs())).fold(0.0, f64::max).max(1e-300);
    arap_run(&el, n, uv, Some(&held), &[], 0.0, None, 0, STOP_TEXEL_SHARE * span / TEXTURE_PX, max_bytes, false, stop)?;
    let (all, per) = stretch_of(&el, chart.faces.count(), uv);
    let band: Vec<f64> = (0..chart.faces.count()).filter(|&f| chart.faces.face(f).iter().any(|&v| free[v as usize])).map(|f| per[f]).collect();
    Ok((if band.is_empty() { 1.0 } else { quantile(&band, STRETCH_QUANTILE) }, all, per))
}

/// The vertices within `k` rings (by shared faces) of `seeds` (the band `quick_stretch` frees; used by uvchart).
#[allow(dead_code)]
pub fn band(faces: &Faces, n: usize, seeds: &[u32], k: usize) -> Vec<bool> {
    let mut vf: Vec<Vec<u32>> = vec![Vec::new(); n];
    for f in 0..faces.count() {
        for &v in faces.face(f) {
            vf[v as usize].push(f as u32);
        }
    }
    let mut inside = vec![false; n];
    let mut front: Vec<u32> = seeds.to_vec();
    for &s in seeds {
        inside[s as usize] = true;
    }
    for _ in 0..k {
        let mut next = Vec::new();
        for &v in &front {
            for &f in &vf[v as usize] {
                for &w in faces.face(f as usize) {
                    if !inside[w as usize] {
                        inside[w as usize] = true;
                        next.push(w as u32);
                    }
                }
            }
        }
        front = next;
    }
    inside
}

// ---------------------------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------------------------

#[cfg(test)]
pub mod tests {
    use super::*;

    /// (points, indptr, flat) of an nx x ny quad grid, spacing h, height z(x, y).
    pub fn grid(nx: usize, ny: usize, h: f64, z: impl Fn(f64, f64) -> f64) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut pts = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                let (x, y) = (i as f64 * h, j as f64 * h);
                pts.extend_from_slice(&[x, y, z(x, y)]);
            }
        }
        let (mut indptr, mut flat) = (vec![0i64], Vec::new());
        let id = |i: usize, j: usize| (j * (nx + 1) + i) as i64;
        for j in 0..ny {
            for i in 0..nx {
                flat.extend_from_slice(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
                indptr.push(flat.len() as i64);
            }
        }
        (pts, indptr, flat)
    }

    fn never() -> AtomicBool {
        AtomicBool::new(false)
    }

    #[test]
    fn probe_builds() {
        assert_eq!(super::probe(), 1);
    }

    #[test]
    fn ldlt_and_ic0_solve_a_grid_laplacian() {
        let (pts, ip, fl) = grid(8, 8, 1.0, |_, _| 0.0);
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        let el = elements(&ch);
        let n = ch.n();
        let mut b = Builder::new(n);
        for t in &el.terms {
            b.edge(t.a as usize, t.b as usize, t.w);
        }
        let a = b.build();
        let truth: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut rhs = vec![0f64; n];
        a.mul(&truth, &mut rhs);
        rhs[0] = truth[0];
        let stop = never();
        let direct = Lin::new(Builder::new(0).build(), &[], 0.0, false, &stop);
        assert!(direct.is_ok());
        let mut b2 = Builder::new(n);
        for t in &el.terms {
            b2.edge(t.a as usize, t.b as usize, t.w);
        }
        let lin = Lin::new(b2.build(), &[0], 0.0, false, &stop).unwrap();
        assert_eq!(lin.path, Path::Direct);
        let mut x = vec![0f64; n];
        let mut r = rhs.clone();
        lin.solve::<1>(&mut r, &mut x, 1e-12, &stop);
        let err = x.iter().zip(&truth).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1e-9, "direct error {err}");
        // Iterative path on the 8x8 grid chart: IC(0) must not fall back to Jacobi.
        let mut b3 = Builder::new(n);
        for t in &el.terms {
            b3.edge(t.a as usize, t.b as usize, t.w);
        }
        let it = Lin::new(b3.build(), &[0], 0.0, true, &stop).unwrap();
        assert_eq!(it.path, Path::Ic0, "IC(0) fell back on the 8x8 grid chart");
        let mut x = vec![0f64; n];
        let mut r = rhs.clone();
        let info = it.solve::<1>(&mut r, &mut x, 1e-12, &stop);
        let err = x.iter().zip(&truth).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 1e-6 && !info.stopped, "IC(0) CG error {err}");
    }

    #[test]
    fn flat_grid_stays_unchanged() {
        let (pts, ip, fl) = grid(8, 8, 0.5, |_, _| 0.0);
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        let g = grid_layout(&ch).unwrap();
        for v in 0..ch.n() {
            assert!((g.uv[2 * v] - pts[3 * v]).abs() < 1e-12 && (g.uv[2 * v + 1] - pts[3 * v + 1]).abs() < 1e-12);
        }
        for rows in [Rows::Off, Rows::Soft, Rows::On] {
            let opts = FlatOpts { rows, steps: 5, ..FlatOpts::default() };
            let out = flatten_chart(&ch, Some(&g.uv), Some((&g.col, &g.row)), &opts, &never()).unwrap();
            let moved = out.uv.iter().zip(&g.uv).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(moved < 1e-9, "{rows:?} moved the flat grid by {moved}");
        }
    }

    #[test]
    fn grid_layout_refuses_a_pole() {
        // Three quads around one interior vertex (valence 3) are not a grid.
        let pts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -0.5, -1.0, 0.0, 0.5, -1.0, 0.0];
        let ip = [0i64, 4, 8, 12];
        let fl = [0i64, 1, 2, 3, 0, 3, 4, 5, 0, 5, 6, 1];
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        assert!(grid_layout(&ch).is_err());
    }

    /// A bumpy quad grid with one extra pole: a 5x5 grid with its centre vertex split into a valence-5 star.
    fn pole_chart() -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        // Rings around a centre: 5 spokes, two rings, quads between.
        let (k, rings) = (5usize, 3usize);
        let mut pts = vec![0.0, 0.0, 0.3];
        for r in 1..=rings {
            for s in 0..2 * k {
                let th = std::f64::consts::TAU * s as f64 / (2 * k) as f64;
                let rad = r as f64;
                pts.extend_from_slice(&[rad * th.cos(), rad * th.sin(), 0.3 * (rad * 0.7).cos()]);
            }
        }
        let ring = |r: usize, s: usize| (1 + (r - 1) * 2 * k + s % (2 * k)) as i64;
        let (mut ip, mut fl) = (vec![0i64], Vec::new());
        for s in 0..k {
            fl.extend_from_slice(&[0, ring(1, 2 * s), ring(1, 2 * s + 1), ring(1, 2 * s + 2)]);
            ip.push(fl.len() as i64);
        }
        for r in 1..rings {
            for s in 0..2 * k {
                fl.extend_from_slice(&[ring(r, s), ring(r + 1, s), ring(r + 1, s + 1), ring(r, s + 1)]);
                ip.push(fl.len() as i64);
            }
        }
        (pts, ip, fl)
    }

    #[test]
    fn soft_rows_stay_positive_definite_on_a_pole_chart() {
        let (pts, ip, fl) = pole_chart();
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        assert!(grid_layout(&ch).is_err(), "pole chart must not be a grid");
        let n = ch.n();
        let el = elements(&ch);
        let triples = loop_triples(&ch.faces, n);
        assert!(!triples.is_empty());
        for w in [0.1, 10.0, 1e4] {
            let mut b = Builder::new(n);
            for t in &el.terms {
                b.edge(t.a as usize, t.b as usize, t.w);
            }
            for t in &triples {
                let coef = [(t[0] as usize, 1.0), (t[1] as usize, -2.0), (t[2] as usize, 1.0)];
                for (x, &(i, ci)) in coef.iter().enumerate() {
                    b.add(i, i, w * ci * ci);
                    for &(j, cj) in &coef[x + 1..] {
                        b.add(i, j, w * ci * cj);
                    }
                }
            }
            // Factor succeeds (every pivot positive) with the one null-space pin.
            Lin::new(b.build(), &[0], 0.0, false, &never()).expect("Soft system not SPD");
        }
        let opts = FlatOpts { rows: Rows::Soft, start: Start::Free, ..FlatOpts::default() };
        let out = flatten_chart(&ch, None, None, &opts, &never()).unwrap();
        assert_eq!(out.stats[stat::FLIPS], 0.0);
        assert_eq!(out.stats[stat::ROWS], 1.0);
    }

    #[test]
    fn rows_on_keep_every_loop_on_one_value() {
        // A curved (cylinder-like) grid: ARAP moves it, On keeps rows straight.
        let (pts, ip, fl) = grid(10, 6, 0.3, |x, _| (x * 0.8).sin());
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        let g = grid_layout(&ch).unwrap();
        let opts = FlatOpts { rows: Rows::On, steps: 10, ..FlatOpts::default() };
        let out = flatten_chart(&ch, Some(&g.uv), Some((&g.col, &g.row)), &opts, &never()).unwrap();
        assert_eq!(out.stats[stat::ROWS], 2.0);
        let (s, _) = stretch(&ch, &out.uv);
        assert!(s < 1.5, "On collapsed the chart: stretch {s}");
        for v in 0..ch.n() {
            for w in 0..ch.n() {
                if g.row[v] == g.row[w] {
                    assert_eq!(out.uv[2 * v + 1], out.uv[2 * w + 1]);
                }
                if g.col[v] == g.col[w] {
                    assert_eq!(out.uv[2 * v], out.uv[2 * w]);
                }
            }
        }
    }

    #[test]
    fn folded_start_ends_with_no_flips() {
        let (pts, ip, fl) = grid(8, 8, 1.0, |x, y| 0.2 * (x * 0.5).sin() * (y * 0.4).cos());
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        let mut start: Vec<f64> = (0..ch.n()).flat_map(|v| [pts[3 * v], pts[3 * v + 1]]).collect();
        // Fold: mirror the right half over x = 4.
        for v in 0..ch.n() {
            if start[2 * v] > 4.0 {
                start[2 * v] = 8.0 - start[2 * v];
            }
        }
        assert!(count_flips(&ch.faces, &start) > 0);
        let opts = FlatOpts { rows: Rows::Off, steps: 30, ..FlatOpts::default() };
        let out = flatten_chart(&ch, Some(&start), None, &opts, &never()).unwrap();
        assert_eq!(out.stats[stat::FLIPS], 0.0);
        assert_eq!(out.stats[stat::START], 2.0); // Tutte
        // Every ARAP step kept every test triangle positive.
        let el = elements(&ch);
        assert!(el.tris.iter().all(|t| tri_area(&out.uv, &t.v) > 0.0));
    }

    #[test]
    fn injective_step_cuts_a_folding_move() {
        let uv = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let d = [0.0, 0.0, 0.0, 0.0, 0.0, -2.0]; // would flip at t = 0.5
        let tris = [TestTri { v: [0, 1, 2], r: [[0.0; 2]; 3], face: 0 }];
        let t = injective_step(&tris, &uv, &d);
        assert!((t - 0.45).abs() < 1e-12, "{t}");
    }

    #[test]
    fn lscm_with_pull_and_pins_are_flip_free_on_a_curved_patch() {
        let (pts, ip, fl) = grid(6, 6, 1.0, |x, y| 0.1 * (x * x + y * y) / 6.0);
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        let free = lscm(&ch, None, 0.0, &never()).unwrap();
        assert_eq!(count_flips(&ch.faces, &free), 0);
        let g = grid_layout(&ch).unwrap();
        let pulled = lscm(&ch, Some(&g.uv), 0.0, &never()).unwrap();
        assert_eq!(count_flips(&ch.faces, &pulled), 0);
        let (s, _) = stretch(&ch, &pulled);
        assert!(s < 1.2, "pulled LSCM stretch {s}");
    }

    #[test]
    fn many_charts_in_one_call_and_quick_stretch() {
        let (pts, ip, fl) = grid(8, 4, 1.0, |x, _| 0.3 * (x * 0.6).sin());
        let faces = Faces { indptr: &ip, flat: &fl };
        let chart: Vec<i64> = (0..faces.count()).map(|f| if f % 8 < 4 { 0 } else { 1 }).collect();
        let start = vec![f64::NAN; 2 * fl.len()];
        let opts = FlatOpts { rows: Rows::Off, start: Start::Free, ..FlatOpts::default() };
        let (uv, stats) = flatten(&pts, &faces, &chart, &start, &[], &[], &opts, &never());
        assert_eq!(stats.len(), 2 * STATS_LEN);
        assert!(uv.iter().all(|x| x.is_finite()));
        assert_eq!(stats[stat::FLIPS] + stats[STATS_LEN + stat::FLIPS], 0.0);
        let ch = Chart { points: &pts, faces };
        let g = grid_layout(&ch).unwrap();
        let seeds: Vec<u32> = (0..ch.n() as u32).filter(|&v| g.col[v as usize] == 4).collect();
        let free = band(&ch.faces, ch.n(), &seeds, 1);
        let mut uv2 = g.uv.clone();
        let (b, all) = quick_stretch(&ch, &mut uv2, &free, 0.0, &never()).unwrap();
        assert!(b.is_finite() && all.is_finite() && b >= 1.0);
        for v in 0..ch.n() {
            if !free[v] {
                assert_eq!(uv2[2 * v], g.uv[2 * v]);
            }
        }
    }

    /// Gate: factor + 20 ARAP steps on a 100k-vertex chart. Run with
    /// `cargo test --release --lib uvflat::tests::gate_100k -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gate_100k() {
        let side = 316; // 317² = 100,489 vertices
        let (pts, ip, fl) = grid(side, side, 0.01, |x, y| 0.3 * (x * 2.0).sin() * (y * 1.5).cos());
        let ch = Chart { points: &pts, faces: Faces { indptr: &ip, flat: &fl } };
        let g = grid_layout(&ch).unwrap();
        let el = elements(&ch);
        let started = Instant::now();
        let mut uv = g.uv.clone();
        let r = arap_run(&el, ch.n(), &mut uv, None, &[], 0.0, None, 20, 1e-9, 0.0, false, &never()).unwrap();
        let total = started.elapsed().as_secs_f64();
        println!(
            "gate_100k: vertices {} nnz(L) {} factor {:.0} ms, one warm solve {:.1} ms, factor + 20 steps {:.3} s, energy {:.3e}, flips {}",
            ch.n(), r.nnz_l, r.factor_ms, r.solve_ms, total, r.energy, count_flips(&ch.faces, &uv)
        );
        assert_eq!(r.steps, 20);
    }
}

