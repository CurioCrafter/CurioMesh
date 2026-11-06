// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Tidy Up: fewer poles by redrawing the patches round them as plain grids (the add-on's `ChordReducer.redraw_all` in
//! `triadq/chords.py`, as Organize's `clean_poles` runs it: without the picture judge).
//!
//! Walk once round a patch of quads: a corner with k faces outside the patch wants 4 - k inside, a turn of k - 2
//! quarter turns. Laid out on squared paper, a walk that closes without crossing itself holds a plain grid with no
//! poles, and every outline corner that got its wish is regular too. `chords::closed_walk` finds the walk that
//! disappoints the fewest corners; when it beats the patch, the patch is swapped for that grid, its new corners placed on
//! the source.

use crate::io::Mesh;
use crate::layout::{cross, dot, jacobian, normal, sub, Limits};
use crate::{chords, issues, near};
use std::collections::{HashMap, HashSet};

type V3 = [f64; 3];
const TURNS: [i64; 3] = [1, 0, -1]; // left, straight on, right: 1, 2 or 3 faces inside the patch
const STEPS: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
const HARD_DEG: f64 = 35.0; // faces meeting sharper than this make a hard edge, which patches stop at
const LONGEST_WALK: usize = 44;
const SHAPE_PRICE: f64 = 0.25; // between equally tidy walks, take the one whose turns match the patch's real corners
const WALK_SLACK: f64 = 6.0; // a walk may disappoint the corners this much past what each wants
const NEW_QUALITY: f64 = 0.05; // no new quad less square than this
const NEW_TURNED_DEG: f64 = 20.0; // nor turned further than this from the old quad nearest it
const MOST_LEFT: f64 = 0.003; // an old corner may end this share of the object's size off the new grid
const SWEEPS: usize = 24;

struct Work<'a> {
    points: Vec<V3>,
    faces: Vec<Option<[i64; 4]>>,
    tag: Vec<i64>,
    users: Vec<Vec<usize>>,
    border: Vec<bool>,
    locked: Vec<bool>, // border corners and corners where tags (material, part) meet
    refused: HashSet<(Vec<i64>, Vec<usize>, i64, usize)>,
    limits: Limits,
    most_left: f64,
    hard: f64,
    source: &'a near::FaceTree,
}

fn mid(q: &[V3; 4]) -> V3 {
    [0, 1, 2].map(|k| (q[0][k] + q[1][k] + q[2][k] + q[3][k]) / 4.0)
}
fn far2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

impl Work<'_> {
    fn quad(&self, f: usize) -> [V3; 4] {
        self.faces[f].unwrap().map(|v| self.points[v as usize])
    }

    /// Pole energy of a corner with `n` faces.
    fn cost(&self, v: i64, n: usize) -> i64 {
        let n = n as i64;
        if self.border[v as usize] { if n > 2 { (n - 2).pow(2) } else { 0 } } else { (n - 4).pow(2) }
    }

    /// No hard edge runs through this corner, and it is free to go.
    fn smooth(&self, v: i64) -> bool {
        if self.locked[v as usize] {
            return false;
        }
        let normals: Vec<V3> = self.users[v as usize].iter().map(|&f| normal(&self.quad(f))).collect();
        normals.iter().enumerate().all(|(k, a)| normals[..k].iter().all(|b| dot(*a, *b) >= self.hard))
    }

    /// The quads within `radius` rings of `seed`, stopping at hard edges, borders and other tags.
    fn patch(&self, seed: i64, radius: usize) -> Option<HashSet<usize>> {
        if !self.smooth(seed) {
            return None;
        }
        let kind = self.tag[self.users[seed as usize][0]];
        let mut region = HashSet::new();
        let (mut frontier, mut seen) = (vec![seed], HashSet::from([seed]));
        for _ in 0..radius {
            let mut next = Vec::new();
            for &v in &frontier {
                if !self.smooth(v) {
                    continue;
                }
                for &f in &self.users[v as usize] {
                    if self.tag[f] == kind && region.insert(f) {
                        next.extend(self.faces[f].unwrap());
                    }
                }
            }
            frontier = next.into_iter().filter(|&w| seen.insert(w)).collect();
        }
        Some(region)
    }

    /// Radians the patch's quads take up at this corner.
    fn angle_inside(&self, v: i64, region: &HashSet<usize>) -> f64 {
        let here = self.points[v as usize];
        self.users[v as usize].iter().filter(|f| region.contains(f)).map(|&f| {
            let face = self.faces[f].unwrap();
            let at = face.iter().position(|&w| w == v).unwrap();
            let (one, two) = (sub(self.points[face[(at + 1) % 4] as usize], here), sub(self.points[face[(at + 3) % 4] as usize], here));
            let n = cross(one, two);
            dot(n, n).sqrt().atan2(dot(one, two))
        }).sum()
    }

    /// The patch's one outline, in the faces' own direction. None unless the patch is a plain disk.
    fn outline(&self, region: &HashSet<usize>) -> Option<Vec<i64>> {
        let mut halves = HashSet::new();
        for &f in region {
            let q = self.faces[f].unwrap();
            for k in 0..4 {
                halves.insert((q[k], q[(k + 1) % 4]));
            }
        }
        let mut following = HashMap::new();
        for &(a, b) in &halves {
            if !halves.contains(&(b, a)) && following.insert(a, b).is_some() {
                return None; // two parts of the patch touch only at this corner
            }
        }
        let start = *following.keys().min()?; // the lowest corner first: the same walk every run
        let mut ring = vec![start];
        let mut v = following[&start];
        while v != start {
            ring.push(v);
            v = *following.get(&v)?;
            if ring.len() > following.len() {
                return None;
            }
        }
        (ring.len() == following.len()).then_some(ring)
    }

    /// Try to redraw the patch round a corner as a grid. Returns how much the pole energy fell, 0 if nothing changed.
    fn redraw(&mut self, seed: i64, radius: usize) -> i64 {
        let Some(region) = self.patch(seed, radius) else { return 0 };
        if region.len() < 3 {
            return 0;
        }
        let Some(ring) = self.outline(&region) else { return 0 };
        if ring.len() > LONGEST_WALK {
            return 0;
        }
        let mut inside: HashMap<i64, usize> = HashMap::new();
        for &f in &region {
            for v in self.faces[f].unwrap() {
                *inside.entry(v).or_default() += 1;
            }
        }
        let on_ring: HashSet<i64> = ring.iter().copied().collect();
        let loose: Vec<i64> = inside.keys().copied().filter(|v| !on_ring.contains(v)).collect();
        if loose.iter().any(|&v| self.locked[v as usize]) {
            return 0;
        }
        // A disk: V - E + F == 1 with one outline. Anything else has a hole or a handle in it.
        let edges = (4 * region.len() + ring.len()) / 2;
        if inside.len() as i64 - edges as i64 + region.len() as i64 != 1 {
            return 0;
        }
        let before: i64 = inside.keys().map(|&v| self.cost(v, self.users[v as usize].len())).sum();
        if before == 0 {
            return 0;
        }
        let key = (ring.clone(), ring.iter().map(|&v| self.users[v as usize].len()).collect(), before, region.len());
        if !self.refused.insert(key) {
            return 0;
        }
        let (mut costs, mut wishes) = (Vec::new(), Vec::new());
        for &v in &ring {
            let outside = self.users[v as usize].len() - inside[&v];
            let turn = self.angle_inside(v, &region) / std::f64::consts::FRAC_PI_2;
            let row = [1, 2, 3].map(|n| self.cost(v, outside + n));
            costs.push([0, 1, 2].map(|t| row[t] as f64 + SHAPE_PRICE * (turn - (t + 1) as f64).powi(2)));
            wishes.push(row);
        }
        let floor: f64 = costs.iter().map(|r| r[0].min(r[1]).min(r[2])).sum();
        let Some((_, turns)) = chords::closed_walk(&costs, (before as f64 - 0.5).min(floor + WALK_SLACK)) else { return 0 };
        let Some(spots) = laid_out(&turns) else { return 0 };
        let after: i64 = wishes.iter().zip(&turns).map(|(row, &t)| row[t]).sum();
        if after >= before {
            return 0;
        }
        let cells = grid_cells(&spots);
        if cells.is_empty() || cells.len() > region.len() {
            return 0;
        }
        if self.place_grid(&region, &ring, &spots, &cells, &loose) { before - after } else { 0 }
    }

    /// Work out the grid on the surface, and swap it in if it fits the old patch.
    fn place_grid(&mut self, region: &HashSet<usize>, ring: &[i64], spots: &[(i64, i64)], cells: &[(i64, i64)], loose: &[i64]) -> bool {
        let mut at: HashMap<(i64, i64), V3> = spots.iter().zip(ring).map(|(&s, &v)| (s, self.points[v as usize])).collect();
        let n = ring.len() as f64;
        let center = ring.iter().fold([0.0; 3], |s, &v| {
            let p = self.points[v as usize];
            [s[0] + p[0] / n, s[1] + p[1] / n, s[2] + p[2] / n]
        });
        let mut middle = Vec::new();
        for &(x, y) in cells {
            for s in [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)] {
                if !at.contains_key(&s) {
                    at.insert(s, center);
                    middle.push(s);
                }
            }
        }
        for _ in 0..if middle.is_empty() { 0 } else { SWEEPS } {
            for &(x, y) in &middle {
                let near: Vec<V3> = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)].iter().filter_map(|s| at.get(s).copied()).collect();
                let k = near.len() as f64;
                at.insert((x, y), near.iter().fold([0.0; 3], |s, p| [s[0] + p[0] / k, s[1] + p[1] / k, s[2] + p[2] / k]));
            }
        }
        for s in &middle {
            match self.source.nearest(at[s]) {
                Some((p, _, _)) => {
                    at.insert(*s, p);
                }
                None => return false,
            }
        }
        let old: Vec<(V3, V3)> = region.iter().map(|&f| {
            let q = self.quad(f);
            (mid(&q), normal(&q))
        }).collect();
        let nearest = |list: &[(V3, V3)], p: V3| *list.iter().min_by(|a, b| far2(a.0, p).total_cmp(&far2(b.0, p))).unwrap();
        let turned = NEW_TURNED_DEG.to_radians().cos();
        let mut new = Vec::with_capacity(cells.len());
        for &(x, y) in cells {
            let q = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)].map(|s| at[&s]);
            let (m, nq) = (mid(&q), normal(&q));
            if jacobian(&q) < NEW_QUALITY || !self.limits.ok(&q) || dot(nq, nearest(&old, m).1) < turned {
                return false;
            }
            new.push((m, nq));
        }
        for &v in loose {
            let p = self.points[v as usize];
            let (m, nq) = nearest(&new, p);
            if dot(sub(p, m), nq).abs() > self.most_left {
                return false;
            }
        }
        // Names: the outline keeps its corners, the middle gets new ones.
        let mut names: HashMap<(i64, i64), i64> = spots.iter().copied().zip(ring.iter().copied()).collect();
        let quads: Vec<[i64; 4]> = {
            let mut next = self.points.len() as i64;
            for s in &middle {
                names.insert(*s, next);
                next += 1;
            }
            cells.iter().map(|&(x, y)| [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)].map(|s| names[&s])).collect()
        };
        // A grid edge across the patch between two outline corners must not be an edge the outside already has.
        let place: HashMap<i64, usize> = ring.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        for q in &quads {
            for k in 0..4 {
                let (a, b) = (q[k], q[(k + 1) % 4]);
                let (Some(&i), Some(&j)) = (place.get(&a), place.get(&b)) else { continue };
                if (i + 1) % ring.len() == j || (j + 1) % ring.len() == i {
                    continue;
                }
                let taken = self.users[a as usize].iter().any(|&f| {
                    let o = self.faces[f].unwrap();
                    !region.contains(&f) && (0..4).any(|t| (o[t] == a && o[(t + 1) % 4] == b) || (o[t] == b && o[(t + 1) % 4] == a))
                });
                if taken {
                    return false;
                }
            }
        }
        // It fits: swap it in.
        let tag = self.tag[*region.iter().next().unwrap()];
        for &f in region {
            for v in self.faces[f].take().unwrap() {
                self.users[v as usize].retain(|&g| g != f);
            }
        }
        for s in &middle {
            self.points.push(at[s]);
            self.users.push(Vec::new());
            self.border.push(false);
            self.locked.push(false);
        }
        for q in quads {
            let f = self.faces.len();
            self.faces.push(Some(q));
            self.tag.push(tag);
            for v in q {
                self.users[v as usize].push(f);
            }
        }
        true
    }
}

/// The walk's points from its turns, or None if it crosses itself or goes round the wrong way.
fn laid_out(turns: &[usize]) -> Option<Vec<(i64, i64)>> {
    let (mut x, mut y, mut heading, mut area) = (0i64, 0i64, 0i64, 0i64);
    let mut seen = HashSet::new();
    let mut points = Vec::with_capacity(turns.len());
    for &t in turns {
        if !seen.insert((x, y)) {
            return None;
        }
        points.push((x, y));
        heading = (heading + TURNS[t]) & 3;
        let (dx, dy) = STEPS[heading as usize];
        area += x * dy - y * dx;
        (x, y) = (x + dx, y + dy);
    }
    (area > 0).then_some(points)
}

/// The unit squares inside a closed walk of unit steps that doesn't cross itself.
fn grid_cells(points: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut crossings: HashMap<i64, Vec<i64>> = HashMap::new();
    for (k, &(x, y)) in points.iter().enumerate() {
        let (u, w) = points[(k + 1) % points.len()];
        if x == u {
            crossings.entry(y.min(w)).or_default().push(x);
        }
    }
    let mut cells = Vec::new();
    for (y, mut row) in crossings {
        row.sort_unstable();
        for pair in row.chunks_exact(2) {
            cells.extend((pair[0]..pair[1]).map(|x| (x, y)));
        }
    }
    cells.sort_unstable();
    cells
}

/// Inner corners with other than four edges.
pub fn poles(mesh: &Mesh) -> usize {
    let nv = mesh.points.len() / 3;
    let mut faces_at = vec![0usize; nv];
    let mut edges: HashMap<(i64, i64), usize> = HashMap::new();
    for f in 0..mesh.faces() {
        let face = mesh.face(f);
        for k in 0..face.len() {
            let (a, b) = (face[k], face[(k + 1) % face.len()]);
            faces_at[a as usize] += 1;
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut border = vec![false; nv];
    for (&(a, b), &n) in &edges {
        if n != 2 {
            border[a as usize] = true;
            border[b as usize] = true;
        }
    }
    (0..nv).filter(|&v| faces_at[v] > 0 && !border[v] && faces_at[v] != 4).count()
}

/// Redraw the patches round poles, worst first, for `rounds` rounds or until that stops paying. Returns the pole energy
/// taken away (0 and the mesh unchanged if it isn't all quads, or if the redraw would bring in bad faces).
pub fn redraw_poles(mesh: &mut Mesh, source: &near::FaceTree, rounds: usize) -> i64 {
    let nf = mesh.faces();
    if nf == 0 || (0..nf).any(|f| mesh.face(f).len() != 4) {
        return 0; // ponytail: all-quad results only, which is what closing gives
    }
    let nv = mesh.points.len() / 3;
    let mut users = vec![Vec::new(); nv];
    let mut edge_use: HashMap<(i64, i64), usize> = HashMap::with_capacity(nf * 2);
    let faces: Vec<Option<[i64; 4]>> = (0..nf).map(|f| Some(mesh.face(f).try_into().unwrap())).collect();
    for (f, q) in faces.iter().enumerate() {
        let q = q.unwrap();
        for k in 0..4 {
            users[q[k] as usize].push(f);
            let (a, b) = (q[k], q[(k + 1) % 4]);
            *edge_use.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut border = vec![false; nv];
    for (&(a, b), &n) in &edge_use {
        if n != 2 {
            border[a as usize] = true;
            border[b as usize] = true;
        }
    }
    let locked: Vec<bool> = (0..nv).map(|v| border[v] || users[v].iter().any(|&f| mesh.material[f] != mesh.material[users[v][0]])).collect();
    let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for c in mesh.points.chunks_exact(3) {
        for k in 0..3 {
            low[k] = low[k].min(c[k]);
            high[k] = high[k].max(c[k]);
        }
    }
    let size = far2(high, low).sqrt();
    let mut work = Work {
        points: mesh.points.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
        faces,
        tag: mesh.material.clone(),
        users,
        border,
        locked,
        refused: HashSet::new(),
        limits: Limits::of(&mesh.points),
        most_left: MOST_LEFT * size,
        hard: HARD_DEG.to_radians().cos(),
        source,
    };
    let mut total = 0;
    for _ in 0..rounds {
        let mut gained = 0;
        let mut order: Vec<i64> = (0..work.users.len() as i64).filter(|&v| {
            let n = work.users[v as usize].len();
            n > 0 && n != 4 && !work.locked[v as usize]
        }).collect();
        order.sort_by_key(|&v| -(work.users[v as usize].len() as i64 - 4).abs());
        for v in order {
            let n = work.users[v as usize].len();
            if n == 0 || n == 4 {
                continue;
            }
            for radius in 1..=3 {
                let gain = work.redraw(v, radius);
                if gain > 0 {
                    gained += gain;
                    break;
                }
            }
        }
        total += gained;
        if gained == 0 {
            break;
        }
    }
    if total == 0 {
        return 0;
    }
    let mut out = Mesh { points: work.points.iter().flatten().copied().collect(), indptr: vec![0], materials: mesh.materials.clone(), ..Default::default() };
    for (q, &tag) in work.faces.iter().zip(&work.tag) {
        if let Some(q) = q {
            out.flat.extend(q);
            out.indptr.push(out.flat.len() as i64);
            out.material.push(tag);
        }
    }
    let out = crate::io::compact(&out);
    if issues::check(&out).bad > issues::check(mesh).bad {
        return 0; // ponytail: the whole pass is refused, not just the redraw that brought the bad face in
    }
    *mesh = out;
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat 6x6 grid of quads with one edge spun, leaving two 3-poles and two 5-poles, redraws to fewer poles, stays on
    /// its plane and keeps its border.
    #[test]
    fn a_pole_pair_on_a_flat_grid_is_redrawn() {
        let n = 7i64;
        let mut mesh = Mesh { indptr: vec![0], ..Default::default() };
        for j in 0..n {
            for i in 0..n {
                mesh.points.extend([i as f64 + 0.1 * ((i * 7 + j * 3) % 5) as f64 / 5.0, j as f64, 0.0]);
            }
        }
        let at = |i: i64, j: i64| j * n + i;
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                mesh.flat.extend([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
                mesh.indptr.push(mesh.flat.len() as i64);
                mesh.material.push(0);
            }
        }
        // Spin the edge (3,3)-(3,2) between quads (2,2) and (3,2): the hexagon's other diagonal.
        let (f, g) = ((2 * 6 + 2) as usize, (2 * 6 + 3) as usize);
        let h = [at(3, 3), at(2, 3), at(2, 2), at(3, 2), at(4, 2), at(4, 3)];
        mesh.flat[f * 4..f * 4 + 4].copy_from_slice(&[h[1], h[2], h[3], h[4]]);
        mesh.flat[g * 4..g * 4 + 4].copy_from_slice(&[h[4], h[5], h[0], h[1]]);
        let border_before: Vec<V3> = (0..n * n).filter(|v| { let (i, j) = (v % n, v / n); i == 0 || j == 0 || i == n - 1 || j == n - 1 }).map(|v| [0, 1, 2].map(|k| mesh.points[v as usize * 3 + k])).collect();
        let poles_before = poles(&mesh);
        assert_eq!(poles_before, 4);
        let plane = near::FaceTree::new(&[-50., -50., 0., 50., -50., 0., 50., 50., 0., -50., 50., 0.], &[0, 4], &[0, 1, 2, 3]);
        assert!(redraw_poles(&mut mesh, &plane, 2) > 0);
        assert!(poles(&mesh) < poles_before, "poles {} -> {}", poles_before, poles(&mesh));
        assert!(mesh.points.chunks_exact(3).all(|p| p[2].abs() < 1e-9));
        let points: Vec<V3> = mesh.points.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        assert!(border_before.iter().all(|b| points.contains(b)));
        assert_eq!(issues::check(&mesh).bad, 0);
    }
}
