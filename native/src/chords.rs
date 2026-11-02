// SPDX-License-Identifier: GPL-3.0-or-later
//! Redraw Messy Quads (`triadq/chords.py`): the cheapest closed walk on squared paper, and how many pixels of the six
//! source pictures a quad gets wrong. Both mirror the Python step for step, in the same double precision, so they give
//! the same walks and the same pixel counts.

const TURNS: [i32; 3] = [1, 0, -1];
const STEPS: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// `_closed_walk` up to the turns: (price, turn index per corner), or None.
pub fn closed_walk(costs: &[[f64; 3]], limit: f64) -> Option<(f64, Vec<usize>)> {
    let count = costs.len();
    if costs.iter().map(|row| row[0].min(row[1]).min(row[2])).sum::<f64>() > limit {
        return None;
    }
    let middle = count / 2 + 1;
    let width = 2 * middle + 1;
    let at = |h: usize, x: usize, y: usize| (h * width + x) * width + y;
    let mut grid = vec![f64::INFINITY; 4 * width * width];
    grid[at(0, middle, middle)] = 0.0;
    let mut history = Vec::with_capacity(count);
    for (step, row) in costs.iter().enumerate() {
        let mut following = vec![f64::INFINITY; 4 * width * width];
        // After `step` steps the walk is at most `step` from the start, and a walk further than the steps left can never
        // come back: what is out there is infinite or unused.
        let reach = step.min(count - step).min(middle);
        let (low, high) = (middle - reach, middle + reach + 1);
        for heading in 0..4 {
            for (turn, &price) in TURNS.iter().zip(row) {
                let new_heading = ((heading as i32 + turn) & 3) as usize;
                let (dx, dy) = STEPS[new_heading];
                for x in low..high {
                    let tx = x as i64 + dx;
                    if tx < 0 || tx >= width as i64 {
                        continue;
                    }
                    for y in low..high {
                        let ty = y as i64 + dy;
                        if ty < 0 || ty >= width as i64 {
                            continue;
                        }
                        let value = grid[at(heading, x, y)] + price;
                        let slot = &mut following[at(new_heading, tx as usize, ty as usize)];
                        if value < *slot {
                            *slot = value;
                        }
                    }
                }
            }
        }
        history.push(std::mem::replace(&mut grid, following));
    }
    let price = grid[at(0, middle, middle)];
    if price > limit {
        return None;
    }
    let mut turns = vec![0usize; count];
    let (mut heading, mut x, mut y, mut value) = (0usize, middle as i64, middle as i64, price);
    for i in (0..count).rev() {
        let (dx, dy) = STEPS[heading];
        let (px, py) = (x - dx, y - dy);
        let mut found = false;
        for (t, &turn) in TURNS.iter().enumerate() {
            let before = ((heading as i32 - turn) & 3) as usize;
            if px >= 0 && px < width as i64 && py >= 0 && py < width as i64 {
                let was = history[i][at(before, px as usize, py as usize)];
                if (was + costs[i][t] - value).abs() < 1e-9 {
                    turns[i] = t;
                    (heading, x, y, value) = (before, px, py, was);
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return None;
        }
    }
    Some((price, turns))
}

/// The six source pictures and the camera check's rules, as `PictureJudge` holds them.
pub struct Judge {
    pub hit: Vec<bool>,
    pub normal: Vec<f64>,
    pub depth: Vec<f64>,
    pub center: [f64; 3],
    pub half: f64,
    pub pixels: usize,
    pub cosine: f64,
    pub behind: f64,
}

type V3 = [f64; 3];

impl Judge {
    /// `PictureJudge.wrong`: a quad as the triangles (0, 1, 2) and (0, 2, 3).
    pub fn wrong(&self, q: &[V3; 4]) -> i64 {
        self.triangle(q[0], q[1], q[2]) + self.triangle(q[0], q[2], q[3])
    }

    fn triangle(&self, a: V3, b: V3, c: V3) -> i64 {
        let (ux, uy, uz) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        let (vx, vy, vz) = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
        let normal = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length < 1e-30 {
            return 0;
        }
        let normal = [normal[0] / length, normal[1] / length, normal[2] / length];
        let pixel = 2.0 * self.half / self.pixels as f64;
        let scale = 1.0 / pixel;
        let (half, center, last) = (self.half, self.center, self.pixels as i64 - 1);
        let (far, hidden) = (2.0 * pixel, self.behind * pixel);
        let mut total = 0;
        for axis in 0..3 {
            let facing = normal[axis];
            if facing.abs() < 1e-6 {
                continue;
            }
            let sign = if facing > 0.0 { 1.0 } else { -1.0 };
            let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
            let flat = |p: V3| ((p[i] - center[i] + half) * scale - 0.5, (p[j] - center[j] + half) * scale - 0.5);
            let ((ax, ay), (bx, by), (cx, cy)) = (flat(a), flat(b), flat(c));
            let low_x = (ax.min(bx).min(cx).ceil() as i64).max(0);
            let high_x = (ax.max(bx).max(cx).floor() as i64).min(last);
            let low_y = (ay.min(by).min(cy).ceil() as i64).max(0);
            let high_y = (ay.max(by).max(cy).floor() as i64).min(last);
            if low_x > high_x || low_y > high_y {
                continue;
            }
            let area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if area.abs() < 1e-12 {
                continue;
            }
            let view = 2 * axis + if sign > 0.0 { 0 } else { 1 };
            let (start, rise_b, rise_c, middle) = (a[axis], b[axis] - a[axis], c[axis] - a[axis], center[axis]);
            let side = self.pixels;
            for x in low_x..=high_x {
                for y in low_y..=high_y {
                    let (fx, fy) = (x as f64, y as f64);
                    let w1 = ((fx - ax) * (cy - ay) - (fy - ay) * (cx - ax)) / area;
                    let w2 = ((bx - ax) * (fy - ay) - (by - ay) * (fx - ax)) / area;
                    if w1 < 0.0 || w2 < 0.0 || w1 + w2 > 1.0 {
                        continue;
                    }
                    let k = (view * side + x as usize) * side + y as usize;
                    if !self.hit[k] {
                        total += 1;
                        continue;
                    }
                    let gap = sign * (middle - (start + w1 * rise_b + w2 * rise_c)) + 2.0 * half - self.depth[k];
                    if gap > hidden {
                        continue;
                    }
                    let m = &self.normal[3 * k..3 * k + 3];
                    if (m[0] * normal[0] + m[1] * normal[1] + m[2] * normal[2]).abs() < self.cosine || gap.abs() > far {
                        total += 1;
                    }
                }
            }
        }
        total
    }
}
