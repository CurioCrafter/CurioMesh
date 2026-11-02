// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! `quadreduce.join_loose_triangles` on flat arrays, a line-by-line mirror so the faces come out the same.

use crate::topo::FxBuild;
use std::collections::HashMap;

type P = [f64; 3];
type Edge = (i64, i64);
const JOIN_QUALITY: f64 = 0.1;

pub fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn cross(a: P, b: P) -> P {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn unit(a: P) -> P {
    let length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if length < 1e-30 {
        return [0.0; 3];
    }
    [a[0] / length, a[1] / length, a[2] / length]
}
pub fn normal(p: &[P; 4]) -> P {
    unit(cross(sub(p[2], p[0]), sub(p[3], p[1])))
}

/// `quadreduce._quality`.
pub fn quality(p: &[P; 4]) -> f64 {
    let n = normal(p);
    let mut worst = 1.0f64;
    for k in 0..4 {
        let out = sub(p[(k + 1) % 4], p[k]);
        let back = sub(p[(k + 3) % 4], p[k]);
        let size = (dot(out, out) * dot(back, back)).sqrt();
        if size < 1e-30 {
            return -1.0;
        }
        let value = dot(cross(out, back), n) / size;
        if value < worst {
            worst = value;
        }
    }
    if worst <= 0.0 {
        return worst;
    }
    for (x, y, z, w) in [(0, 1, 2, 3), (1, 2, 3, 0)] {
        let fold = dot(unit(cross(sub(p[y], p[x]), sub(p[z], p[x]))), unit(cross(sub(p[z], p[x]), sub(p[w], p[x]))));
        let value = (fold - 0.5) * 2.0;
        if value < worst {
            worst = value;
        }
    }
    worst
}

fn key(a: i64, b: i64) -> Edge {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn edges(face: &[i64]) -> impl Iterator<Item = Edge> + '_ {
    (0..face.len()).map(move |k| key(face[k], face[(k + 1) % face.len()]))
}

/// The search's `parents` dict: a stamp per face, so starting a new search costs nothing.
struct Parents {
    stamp: Vec<u32>,
    parent: Vec<i64>,
    now: u32,
}

impl Parents {
    fn clear(&mut self) {
        self.now += 1;
    }
    fn contains(&self, f: u32) -> bool {
        self.stamp.get(f as usize) == Some(&self.now)
    }
    fn insert(&mut self, f: u32, parent: i64) {
        if f as usize >= self.stamp.len() {
            self.stamp.resize(f as usize * 2 + 16, 0);
            self.parent.resize(f as usize * 2 + 16, -1);
        }
        self.stamp[f as usize] = self.now;
        self.parent[f as usize] = parent;
    }
    fn get(&self, f: u32) -> i64 {
        self.parent[f as usize]
    }
}

struct Join {
    points: Vec<P>,
    faces: Vec<Option<Vec<i64>>>,
    owner: Vec<i64>, // the input face whose material each face has
    material: Vec<i64>,
    users: HashMap<Edge, Vec<u32>, FxBuild>,
    loose: Vec<bool>,
}

/// The outline of a polygon and a face that shares exactly one edge with it.
fn merged(first: &[i64], face: &[i64]) -> Option<Vec<i64>> {
    let n = face.len();
    for k in 0..n {
        let (v, u) = (face[k], face[(k + 1) % n]);
        // The dict `at` keeps a vertex's last index, but an outline repeats none, so `position` finds the same one.
        if let Some(at) = first.iter().position(|&x| x == u) {
            if first[(at + 1) % first.len()] == v {
                let rest: Vec<i64> = (0..n - 2).map(|m| face[(k + 2 + m) % n]).collect();
                if rest.iter().any(|w| first.contains(w)) {
                    return None;
                }
                let mut out = first[..=at].to_vec();
                out.extend_from_slice(&rest);
                out.extend_from_slice(&first[at + 1..]);
                return Some(out);
            }
        }
    }
    None
}

fn with_split(outline: &[i64], edge: Edge, middle: i64) -> Option<Vec<i64>> {
    for k in 0..outline.len() {
        if key(outline[k], outline[(k + 1) % outline.len()]) == edge {
            let mut out = outline[..=k].to_vec();
            out.push(middle);
            out.extend_from_slice(&outline[k + 1..]);
            return Some(out);
        }
    }
    None
}

impl Join {
    fn worst(&self, quads: &[Vec<i64>], facing: P) -> f64 {
        let mut low = 1.0f64;
        for quad in quads {
            let spots = [self.points[quad[0] as usize], self.points[quad[1] as usize], self.points[quad[2] as usize], self.points[quad[3] as usize]];
            if dot(normal(&spots), facing) <= 0.3 {
                return -1.0;
            }
            let value = quality(&spots);
            if value < low {
                low = value;
            }
        }
        low
    }

    fn cut_hexagon(&mut self, hexagon: &[i64], merged_away: &[Edge]) -> Option<Vec<Vec<i64>>> {
        let corners: Vec<P> = hexagon.iter().map(|&v| self.points[v as usize]).collect();
        let mut total = [0.0; 3];
        for k in 1..5 {
            let n = cross(sub(corners[k], corners[0]), sub(corners[k + 1], corners[0]));
            total = [total[0] + n[0], total[1] + n[1], total[2] + n[2]];
        }
        let facing = unit(total);
        let mut best: Option<Vec<Vec<i64>>> = None;
        let mut best_worst = JOIN_QUALITY;
        for k in 0..3 {
            let edge = key(hexagon[k], hexagon[k + 3]);
            if self.users.contains_key(&edge) && !merged_away.contains(&edge) {
                continue;
            }
            let mut second = hexagon[k + 3..].to_vec();
            second.extend_from_slice(&hexagon[..=k]);
            let quads = vec![hexagon[k..k + 4].to_vec(), second];
            let value = self.worst(&quads, facing);
            if value >= best_worst {
                best = Some(quads);
                best_worst = value;
            }
        }
        if best.is_some() {
            return best;
        }
        let middle = self.points.len() as i64;
        // Python 3.12's `sum` is compensated, and Blender 5 has that: the same sum here, or the middle would differ in the last digit.
        let mean = |c: usize| {
            let (mut high, mut low) = (0.0f64, 0.0f64);
            for corner in &corners {
                let x = corner[c];
                let t = high + x;
                low += if high.abs() >= x.abs() { (high - t) + x } else { (x - t) + high };
                high = t;
            }
            (if low != 0.0 && low.is_finite() { high + low } else { high }) / 6.0
        };
        self.points.push([mean(0), mean(1), mean(2)]);
        for k in 0..2 {
            let ring: Vec<i64> = (0..6).map(|m| hexagon[(k + m) % 6]).collect();
            let quads = vec![vec![middle, ring[0], ring[1], ring[2]], vec![middle, ring[2], ring[3], ring[4]], vec![middle, ring[4], ring[5], ring[0]]];
            let value = self.worst(&quads, facing);
            if value >= best_worst {
                best = Some(quads);
                best_worst = value;
            }
        }
        if best.is_none() {
            self.points.pop();
        }
        best
    }

    fn face(&self, f: u32) -> &[i64] {
        self.faces[f as usize].as_deref().unwrap()
    }

    fn join(&mut self, path: &[u32]) -> bool {
        let count = path.len();
        let mut shared: Vec<Edge> = Vec::with_capacity(count - 1);
        for w in path.windows(2) {
            let mut common: Option<Edge> = None;
            let mut several = false;
            let first: Vec<Edge> = edges(self.face(w[0])).collect();
            for e in edges(self.face(w[1])) {
                if first.contains(&e) {
                    several |= common.is_some_and(|c| c != e);
                    common = Some(e);
                }
            }
            match common {
                Some(e) if !several => shared.push(e),
                _ => return false,
            }
        }
        let merged_away: Vec<Edge> = if count == 3 { shared.clone() } else { vec![shared[0], shared[shared.len() - 1]] };
        let size = self.points.len();
        let mut middles: Vec<(Edge, i64)> = Vec::new();
        for &edge in &shared[1..shared.len() - 1] {
            let at = self.points.len() as i64;
            middles.retain(|m| m.0 != edge); // a dict: the same edge twice keeps the later point
            middles.push((edge, at));
            let (pa, pb) = (self.points[edge.0 as usize], self.points[edge.1 as usize]);
            self.points.push([(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5]);
        }
        let middle_of = |edge: Edge| middles.iter().find(|m| m.0 == edge).unwrap().1;
        let mut outlines: Vec<Option<Vec<i64>>> = Vec::with_capacity(count);
        if count == 3 {
            let first = merged(self.face(path[0]), self.face(path[1])).unwrap_or_default();
            outlines.push(merged(&first, self.face(path[2])));
        } else {
            let first = merged(self.face(path[0]), self.face(path[1]));
            outlines.push(first.and_then(|o| with_split(&o, shared[1], middle_of(shared[1]))));
            for k in 2..count - 2 {
                let middle = with_split(self.face(path[k]), shared[k - 1], middle_of(shared[k - 1]));
                outlines.push(middle.and_then(|o| with_split(&o, shared[k], middle_of(shared[k]))));
            }
            let last = merged(self.face(path[count - 2]), self.face(path[count - 1]));
            outlines.push(last.and_then(|o| with_split(&o, shared[count - 3], middle_of(shared[count - 3]))));
        }
        let mut made: Vec<Vec<i64>> = Vec::new();
        for outline in &outlines {
            let cut = match outline {
                Some(o) if o.len() == 6 => self.cut_hexagon(o, &merged_away),
                _ => None,
            };
            match cut {
                Some(quads) => made.extend(quads),
                None => {
                    self.points.truncate(size);
                    return false;
                }
            }
        }
        // Faces of one path can touch away from it, so the new quads may not add a third face to any edge.
        let mut uses: HashMap<Edge, usize, FxBuild> = HashMap::default();
        for quad in &made {
            for e in edges(quad) {
                *uses.entry(e).or_insert(0) += 1;
            }
        }
        for (e, n) in &uses {
            let outside = match self.users.get(e) {
                Some(row) => {
                    let mut rest: Vec<u32> = row.iter().copied().filter(|f| !path.contains(f)).collect();
                    rest.sort_unstable();
                    rest.dedup();
                    rest.len()
                }
                None => 0,
            };
            if n + outside > 2 {
                self.points.truncate(size);
                return false;
            }
        }
        for &f in path {
            let face = self.faces[f as usize].take().unwrap();
            for e in edges(&face) {
                let row = self.users.get_mut(&e).unwrap();
                let at = row.iter().position(|&x| x == f).unwrap();
                row.remove(at);
                if row.is_empty() {
                    self.users.remove(&e);
                }
            }
        }
        let material = self.material[path[0] as usize];
        let owner = self.owner[path[0] as usize];
        for quad in made {
            let index = self.faces.len() as u32;
            for e in edges(&quad) {
                self.users.entry(e).or_default().push(index);
            }
            self.faces.push(Some(quad));
            self.material.push(material);
            self.owner.push(owner);
            self.loose.push(false);
        }
        self.loose[path[0] as usize] = false;
        self.loose[path[count - 1] as usize] = false;
        true
    }

    /// Try the other triangles within `reach` quads, nearest first.
    fn search(&mut self, start: u32, reach: usize, parents: &mut Parents) {
        parents.clear();
        parents.insert(start, -1);
        let mut level = vec![start];
        let start_material = self.material[start as usize];
        for _depth in 0..=reach {
            let mut following = Vec::new();
            for &f in &level {
                let size = self.face(f).len();
                for corner in 0..size {
                    let face = self.face(f);
                    let edge = key(face[corner], face[(corner + 1) % size]);
                    // A join that succeeds ends the search, and one that fails changes nothing, so the row holds.
                    let row = self.users[&edge].as_slice() as *const [u32];
                    for &other in unsafe { &*row } {
                        if parents.contains(other) || self.material[other as usize] != start_material {
                            continue;
                        }
                        parents.insert(other, f as i64);
                        if self.loose[other as usize] {
                            if f == start {
                                continue;
                            }
                            let mut path = vec![other];
                            while *path.last().unwrap() != start {
                                path.push(parents.get(*path.last().unwrap()) as u32);
                            }
                            if self.join(&path) {
                                return;
                            }
                        } else {
                            following.push(other);
                        }
                    }
                }
            }
            level = following;
        }
    }
}

pub struct Joined {
    pub vertices: Vec<f64>,
    pub indptr: Vec<i64>,
    pub flat: Vec<i64>,
    pub owner: Vec<i64>,
    pub joined: i64,
}

/// `material` holds an integer per face that is equal exactly where Python's material strings are.
pub fn join_loose_triangles(indptr: &[i64], flat: &[i64], vertices: &[f64], material: &[i64], farthest: usize) -> Joined {
    let count = indptr.len() - 1;
    let mut state = Join {
        points: vertices.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect(),
        faces: (0..count).map(|f| Some(flat[indptr[f] as usize..indptr[f + 1] as usize].to_vec())).collect(),
        owner: (0..count as i64).collect(),
        material: material.to_vec(),
        users: HashMap::with_capacity_and_hasher(flat.len() * 3 / 5, Default::default()),
        loose: (0..count).map(|f| indptr[f + 1] - indptr[f] == 3).collect(),
    };
    for f in 0..count {
        let face = state.faces[f].take().unwrap();
        for e in edges(&face) {
            state.users.entry(e).or_default().push(f as u32);
        }
        state.faces[f] = Some(face);
    }
    let triangles = state.loose.iter().filter(|&&l| l).count() as i64;
    let mut parents = Parents {
        stamp: vec![0; count],
        parent: vec![-1; count],
        now: 0,
    };
    let mut reach = 0usize;
    while reach < farthest {
        reach = (reach * 2).max(1);
        // `sorted(loose)` at the start of the pass: faces made since are quads, so they never are loose.
        let starts: Vec<u32> = (0..state.loose.len() as u32).filter(|&f| state.loose[f as usize]).collect();
        for start in starts {
            if state.loose[start as usize] {
                state.search(start, reach, &mut parents);
            }
        }
        if !state.loose.iter().any(|&l| l) {
            break;
        }
    }
    let mut out = Joined {
        vertices: state.points.iter().flatten().copied().collect(),
        indptr: vec![0],
        flat: Vec::with_capacity(flat.len()),
        owner: Vec::with_capacity(count),
        joined: triangles - state.loose.iter().filter(|&&l| l).count() as i64,
    };
    for (f, face) in state.faces.iter().enumerate() {
        if let Some(face) = face {
            out.flat.extend_from_slice(face);
            out.indptr.push(out.flat.len() as i64);
            out.owner.push(state.owner[f]);
        }
    }
    out
}
