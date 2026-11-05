// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Quad Size: whole results at several Detail values (a ladder), so an app can offer a slider from small quads to big
//! ones. A port of the add-on's `triadq/ladder.py` (`FACTORS`, `details`, `useful`); here every rung shares one
//! reduction order and one set of source pictures (`retopo::Prepared`), so a rung costs only its own looks.

use crate::io::Mesh;
use crate::retopo::{Outcome, Prepared};
use crate::settings::{self, Settings};
use std::sync::atomic::{AtomicBool, Ordering};

pub const FACTORS: [f64; 7] = [0.25, 0.4, 0.63, 1.0, 1.59, 2.52, 4.0]; // of the Detail asked for: smallest quads first
pub const LOWEST: f64 = 0.1;
pub const HIGHEST: f64 = 80.0;

/// One result on the ladder.
pub struct Rung {
    pub detail: f64,
    pub faces: usize,
    pub loss: f64,
    pub outcome: Outcome,
}

/// The Detail of every rung for `base`, smallest first, rounded so the same rung always has the same value.
pub fn details(base: f64) -> Vec<f64> {
    let mut out: Vec<f64> = FACTORS.iter().map(|f| ((base * f).clamp(LOWEST, HIGHEST) * 1000.0).round() / 1000.0).collect();
    out.dedup();
    out
}

/// Indexes of the rungs worth showing: none that another rung beats on both faces and view change. A higher Detail can
/// land on more faces; leaving it out keeps the slider's face count going one way.
pub fn useful(rungs: &[(usize, f64)]) -> Vec<usize> {
    (0..rungs.len()).filter(|&i| {
        let (f, c) = rungs[i];
        !rungs.iter().any(|&(f2, c2)| f2 <= f && c2 <= c && (f2 < f || c2 < c))
    }).collect()
}

/// Work out a rung for each of `details` (the Detail asked for first, then outward), calling `on_rung` as each is ready.
/// `reduce_to` in `settings` is ignored: each rung searches on its own Detail. Stops between rungs when `cancel` is set.
/// Returns the rungs sorted by Detail.
pub fn run(source: &Mesh, settings: &Settings, details: &[f64], mut on_rung: impl FnMut(&Rung), mut log: impl FnMut(&str), cancel: &AtomicBool) -> Result<Vec<Rung>, String> {
    let work = settings::work_on(source, settings, &mut log);
    let prepared = Prepared::new(&work.mesh, settings.pixels, settings.gpu, &mut log);
    let mut order = details.to_vec();
    order.sort_by(|a, b| (a / settings.detail).ln().abs().total_cmp(&(b / settings.detail).ln().abs()));
    let mut rungs = Vec::new();
    for detail in order {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let mut one = settings.retopo();
        (one.detail, one.reduce_to) = (detail, None);
        log(&format!("Quad size at detail {detail}%"));
        let out = prepared.solve(&one, &mut log)?;
        let out = settings::finish(source, settings, &work, out, &mut log);
        let rung = Rung { detail, faces: out.mesh.faces(), loss: out.loss, outcome: out };
        on_rung(&rung);
        rungs.push(rung);
    }
    rungs.sort_by(|a, b| a.detail.total_cmp(&b.detail));
    Ok(rungs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn details_and_useful() {
        assert_eq!(details(6.0), vec![1.5, 2.4, 3.78, 6.0, 9.54, 15.12, 24.0]);
        assert_eq!(details(0.2)[0], LOWEST);
        // The second rung has more faces and more change than the first: not worth a stop on the slider.
        assert_eq!(useful(&[(100, 2.0), (120, 3.0), (80, 4.0)]), vec![0, 2]);
    }
}
