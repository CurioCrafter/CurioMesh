// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Triangle pairing for one seed: `engine._pair_triangles` on flat arrays, every seed on its own thread. Python's
//! `random.Random(seed).shuffle` is reproduced exactly, so the pairs are the same as the Python path's.

use crate::matching;
use rayon::prelude::*;

/// Python's `random.Random(seed)`: MT19937 seeded by `init_by_array([seed])`, for a seed below 2^32.
struct Mersenne {
    state: Box<[u32; 624]>,
    at: usize,
}

impl Mersenne {
    fn new(seed: u32) -> Self {
        let mut s = Box::new([0u32; 624]);
        s[0] = 19650218;
        for i in 1..624 {
            s[i] = 1812433253u32.wrapping_mul(s[i - 1] ^ (s[i - 1] >> 30)).wrapping_add(i as u32);
        }
        let mut i = 1usize;
        for _ in 0..624 {
            s[i] = (s[i] ^ (s[i - 1] ^ (s[i - 1] >> 30)).wrapping_mul(1664525)).wrapping_add(seed); // a key of one word
            i += 1;
            if i >= 624 {
                s[0] = s[623];
                i = 1;
            }
        }
        for _ in 0..623 {
            s[i] = (s[i] ^ (s[i - 1] ^ (s[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                s[0] = s[623];
                i = 1;
            }
        }
        s[0] = 0x8000_0000;
        Mersenne { state: s, at: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.at >= 624 {
            let s = &mut self.state;
            for k in 0..624 {
                let y = (s[k] & 0x8000_0000) | (s[(k + 1) % 624] & 0x7fff_ffff);
                s[k] = s[(k + 397) % 624] ^ (y >> 1) ^ (if y & 1 == 1 { 0x9908_b0df } else { 0 });
            }
            self.at = 0;
        }
        let mut y = self.state[self.at];
        self.at += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// `order = list(range(count)); random.Random(seed).shuffle(order)`.
pub fn shuffled(count: usize, seed: u32) -> Vec<i64> {
    let mut order: Vec<i64> = (0..count as i64).collect();
    let mut random = Mersenne::new(seed);
    for i in (1..count).rev() {
        let n = (i + 1) as u32;
        let bits = 32 - n.leading_zeros();
        let j = loop {
            let r = random.next() >> (32 - bits);
            if r < n {
                break r;
            }
        };
        order.swap(i, j as usize);
    }
    order
}

/// The graph every seed shares: for each face its candidates, best score first, as in `engine._pairing_prep`.
pub struct Allowed {
    indptr: Vec<i64>,
    other: Vec<i64>,
    row: Vec<u32>,
}

pub fn allowed(faces: usize, face_a: &[i64], face_b: &[i64], scores: &[f64], minimum: Option<f64>) -> Allowed {
    let keep = |m: usize| minimum.is_none_or(|limit| scores[m] > limit);
    let mut indptr = vec![0i64; faces + 1];
    for m in 0..face_a.len() {
        if keep(m) {
            indptr[face_a[m] as usize + 1] += 1;
            indptr[face_b[m] as usize + 1] += 1;
        }
    }
    for f in 0..faces {
        indptr[f + 1] += indptr[f];
    }
    let total = indptr[faces] as usize;
    let mut filled = indptr.clone();
    let mut items = vec![(0.0f64, 0i64, 0u32); total];
    for m in 0..face_a.len() {
        if keep(m) {
            for (from, to) in [(face_a[m], face_b[m]), (face_b[m], face_a[m])] {
                let at = &mut filled[from as usize];
                items[*at as usize] = (scores[m], to, m as u32);
                *at += 1;
            }
        }
    }
    let spans: Vec<&mut [(f64, i64, u32)]> = {
        let mut rest: &mut [(f64, i64, u32)] = &mut items;
        let mut spans = Vec::with_capacity(faces);
        for f in 0..faces {
            let (head, tail) = rest.split_at_mut((indptr[f + 1] - indptr[f]) as usize);
            spans.push(head);
            rest = tail;
        }
        spans
    };
    // `list.sort(key=score, reverse=True)` is stable; the caller has already ruled out NaN.
    spans.into_par_iter().with_min_len(4096).for_each(|span| {
        span.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    });
    Allowed {
        indptr,
        other: items.iter().map(|x| x.1).collect(),
        row: items.iter().map(|x| x.2).collect(),
    }
}

pub struct Paired {
    pub quads: Vec<i64>,   // 4 per quad, in the order they are emitted
    pub tris: Vec<i64>,    // triangles left over, ascending
    pub owner: Vec<i64>,   // the face whose material each output face takes: quads first, then triangles
    pub rescued: i64,
}

/// `_pair_triangles` for one seed, without UVs.
pub fn pair(graph: &Allowed, candidates: &[i64], seed: u32) -> Paired {
    let faces = graph.indptr.len() - 1;
    let order = shuffled(faces, seed);
    let mut mate = vec![-1i64; faces];
    for &face in &order {
        let f = face as usize;
        if mate[f] != -1 {
            continue;
        }
        for &other in &graph.other[graph.indptr[f] as usize..graph.indptr[f + 1] as usize] {
            if mate[other as usize] == -1 {
                mate[f] = other;
                mate[other as usize] = face;
                break;
            }
        }
    }
    let rescued = matching::augment(&graph.indptr, &graph.other, &order, &mut mate);
    let mut used = vec![false; faces];
    let mut out = Paired {
        quads: Vec::with_capacity(faces * 2),
        tris: Vec::new(),
        owner: Vec::with_capacity(faces),
        rescued,
    };
    for &face in &order {
        let f = face as usize;
        if used[f] || mate[f] == -1 {
            continue;
        }
        let other = mate[f];
        used[f] = true;
        used[other as usize] = true;
        // `quads_by_pair`: when two triangles share two edges, the last candidate listed wins.
        let (from, to) = (graph.indptr[f] as usize, graph.indptr[f + 1] as usize);
        let row = (from..to).filter(|&k| graph.other[k] == other).map(|k| graph.row[k]).max().unwrap() as usize;
        out.quads.extend_from_slice(&candidates[row * 4..row * 4 + 4]);
        out.owner.push(face);
    }
    for f in 0..faces {
        if !used[f] {
            out.tris.push(f as i64);
            out.owner.push(f as i64);
        }
    }
    out
}

pub fn pair_seeds(graph: &Allowed, candidates: &[i64], seeds: &[u32]) -> Vec<Paired> {
    seeds.par_iter().map(|&seed| pair(graph, candidates, seed)).collect()
}
