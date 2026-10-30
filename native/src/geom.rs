// SPDX-License-Identifier: GPL-3.0-or-later
//! Numeric kernels: generalized winding number and cross-field relaxation loops.

use numpy::Complex64;
use rayon::prelude::*;

/// `patch.winding_number`: Van Oosterom & Strackee solid angles summed per point.
/// Points run in parallel; triangles whose bounding box is far from the point are
/// still evaluated (the result must stay exact), so the cull lives in `inside_volume`.
pub fn winding_number(points: &[f64], tris: &[f64]) -> Vec<f64> {
    let tau = 2.0 * std::f64::consts::PI;
    let one = |p: &[f64]| {
        let mut total = 0.0;
        for t in tris.chunks_exact(9) {
            let a = [t[0] - p[0], t[1] - p[1], t[2] - p[2]];
            let b = [t[3] - p[0], t[4] - p[1], t[5] - p[2]];
            let c = [t[6] - p[0], t[7] - p[1], t[8] - p[2]];
            let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dot = |u: &[f64; 3], v: &[f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            let (la, lb, lc) = (norm(&a), norm(&b), norm(&c));
            let bxc = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            let det = dot(&a, &bxc);
            let den = la * lb * lc + dot(&a, &b) * lc + dot(&b, &c) * la + dot(&c, &a) * lb;
            total += det.atan2(den);
        }
        total / tau
    };
    if (points.len() / 3) * (tris.len() / 9) >= crate::topo::PARALLEL_MIN {
        points.par_chunks(3).map(one).collect()
    } else {
        points.chunks(3).map(one).collect()
    }
}

#[inline]
fn mul(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// Inner loop of `engine._estimate_field` (plain Jacobi diffusion).
pub fn cross_jacobi(
    cross: &mut [Complex64],
    face_a: &[i64],
    face_b: &[i64],
    transport: &[Complex64],
    pinned: &[bool],
    degree: &[f64],
    steps: usize,
) {
    let n = cross.len();
    let zero = Complex64::new(0.0, 0.0);
    let (mut acc_a, mut acc_b) = (vec![zero; n], vec![zero; n]);
    for _ in 0..steps {
        acc_a.fill(zero);
        acc_b.fill(zero);
        for k in 0..face_a.len() {
            let (fa, fb) = (face_a[k] as usize, face_b[k] as usize);
            let into_a = mul(cross[fb], transport[k]);
            let into_b = mul(cross[fa], transport[k].conj());
            acc_a[fa].re += into_a.re;
            acc_a[fa].im += into_a.im;
            acc_b[fb].re += into_b.re;
            acc_b[fb].im += into_b.im;
        }
        for f in 0..n {
            if pinned[f] {
                continue;
            }
            let scale = 1.0 / degree[f];
            let re = cross[f].re + (acc_a[f].re + acc_b[f].re);
            let im = cross[f].im + (acc_a[f].im + acc_b[f].im);
            cross[f] = Complex64::new(re * scale, im * scale);
        }
    }
}

/// Inner loop of `field.estimate_field_gl` (Ginzburg-Landau relaxation).
#[allow(clippy::too_many_arguments)]
pub fn cross_gl(
    cross: &mut [Complex64],
    face_a: &[i64],
    face_b: &[i64],
    transport: &[Complex64],
    pinned: &[bool],
    denom: &[f64],
    pin_term: &[Complex64],
    mu: f64,
    dt: f64,
    steps: usize,
) {
    let n = cross.len();
    let zero = Complex64::new(0.0, 0.0);
    let mut acc = vec![zero; n];
    let mut next = cross.to_vec();
    for step in 0..steps {
        acc.fill(zero);
        // np.bincount over concat((face_a, face_b)): all face_a terms first.
        for k in 0..face_a.len() {
            let v = mul(cross[face_b[k] as usize], transport[k]);
            let a = &mut acc[face_a[k] as usize];
            a.re += v.re;
            a.im += v.im;
        }
        for k in 0..face_b.len() {
            let v = mul(cross[face_a[k] as usize], transport[k].conj());
            let a = &mut acc[face_b[k] as usize];
            a.re += v.re;
            a.im += v.im;
        }
        let sharpen = step >= steps / 2;
        for f in 0..n {
            if pinned[f] {
                next[f] = cross[f];
                continue;
            }
            let scale = 1.0 / denom[f];
            let re = acc[f].re + mu * cross[f].re + pin_term[f].re;
            let im = acc[f].im + mu * cross[f].im + pin_term[f].im;
            let mut r = Complex64::new(re * scale, im * scale);
            if sharpen {
                let s = dt * (1.0 - r.re.hypot(r.im).powi(2));
                r = Complex64::new(r.re + s * r.re, r.im + s * r.im);
            }
            next[f] = r;
        }
        cross.copy_from_slice(&next);
    }
}
