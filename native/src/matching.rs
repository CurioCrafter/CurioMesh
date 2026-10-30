// SPDX-License-Identifier: GPL-3.0-or-later
//! Maximum-cardinality matching (Edmonds' blossoms), a line-by-line mirror of
//! `engine._augment_matching` / `engine._blossom_search` so the pairs are identical.
//! Dict membership is modelled with -1 sentinels (stored values are never -1).

use std::collections::VecDeque;

struct Search<'a> {
    indptr: &'a [i64],
    indices: &'a [i64],
    mate: &'a [i64],
    base: Vec<i64>,
    parent: Vec<i64>,
    even: Vec<bool>,
    touched: Vec<usize>,
    seen: Vec<u32>,
    seen_gen: u32,
}

impl<'a> Search<'a> {
    fn touch(&mut self, v: i64) {
        self.touched.push(v as usize);
    }

    fn find(&mut self, mut vertex: i64) -> i64 {
        let mut top = vertex;
        while self.base[top as usize] != -1 {
            top = self.base[top as usize];
        }
        while vertex != top {
            let following = self.base[vertex as usize];
            self.base[vertex as usize] = top;
            vertex = following;
        }
        top
    }

    fn common_base(&mut self, mut first: i64, mut second: i64) -> i64 {
        self.seen_gen += 1;
        loop {
            first = self.find(first);
            self.seen[first as usize] = self.seen_gen;
            let partner = self.mate[first as usize];
            if partner == -1 || self.parent[partner as usize] == -1 {
                break;
            }
            first = self.parent[partner as usize];
        }
        loop {
            second = self.find(second);
            if self.seen[second as usize] == self.seen_gen {
                return second;
            }
            let partner = self.mate[second as usize];
            if partner == -1 || self.parent[partner as usize] == -1 {
                return second; // unreachable for a valid alternating tree
            }
            second = self.parent[partner as usize];
        }
    }

    fn mark_path(
        &mut self,
        mut vertex: i64,
        stop: i64,
        mut child: i64,
        marked: &mut Vec<i64>,
        odd: &mut Vec<i64>,
    ) {
        while self.find(vertex) != stop {
            let partner = self.mate[vertex as usize];
            let fv = self.find(vertex);
            marked.push(fv);
            let fp = self.find(partner);
            marked.push(fp);
            self.parent[vertex as usize] = child;
            self.touch(vertex);
            child = partner;
            odd.push(partner);
            vertex = self.parent[partner as usize];
        }
    }

    /// Returns the free end (or -1); tree vertices are left in `touched`.
    fn run(&mut self, root: i64, frozen: &[bool]) -> i64 {
        self.even[root as usize] = true;
        self.touch(root);
        let mut queue = VecDeque::from([root]);
        let mut marked = Vec::new();
        let mut odd = Vec::new();
        while let Some(vertex) = queue.pop_front() {
            let (lo, hi) = (
                self.indptr[vertex as usize] as usize,
                self.indptr[vertex as usize + 1] as usize,
            );
            for k in lo..hi {
                let other = self.indices[k];
                if frozen[other as usize]
                    || self.mate[vertex as usize] == other
                    || self.find(vertex) == self.find(other)
                {
                    continue;
                }
                let partner = self.mate[other as usize];
                if other == root || (partner != -1 && self.parent[partner as usize] != -1) {
                    let blossom_base = self.common_base(vertex, other);
                    marked.clear();
                    odd.clear();
                    self.mark_path(vertex, blossom_base, other, &mut marked, &mut odd);
                    self.mark_path(other, blossom_base, vertex, &mut marked, &mut odd);
                    for &member in &marked {
                        if member != blossom_base {
                            self.base[member as usize] = blossom_base;
                            self.touch(member);
                        }
                    }
                    for &member in &odd {
                        if !self.even[member as usize] {
                            self.even[member as usize] = true;
                            self.touch(member);
                            queue.push_back(member);
                        }
                    }
                } else if self.parent[other as usize] == -1 {
                    self.parent[other as usize] = vertex;
                    self.touch(other);
                    if partner == -1 {
                        return other;
                    }
                    self.even[partner as usize] = true;
                    self.touch(partner);
                    queue.push_back(partner);
                }
            }
        }
        -1
    }
}

/// Grows `mate` (-1 = free) to a maximum matching; returns the number of augmentations.
pub fn augment(indptr: &[i64], indices: &[i64], order: &[i64], mate: &mut [i64]) -> i64 {
    let n = mate.len();
    let mut frozen = vec![false; n];
    let mut base = vec![-1i64; n];
    let mut parent = vec![-1i64; n];
    let mut even = vec![false; n];
    let mut seen = vec![0u32; n];
    let mut seen_gen = 0u32;
    let mut rescued = 0;
    for &root in order {
        let r = root as usize;
        if mate[r] != -1 || frozen[r] || indptr[r] == indptr[r + 1] {
            continue;
        }
        let mut search = Search {
            indptr,
            indices,
            mate: &*mate,
            base,
            parent,
            even,
            touched: Vec::new(),
            seen,
            seen_gen,
        };
        let end = search.run(root, &frozen);
        let Search {
            base: b,
            parent: p,
            even: ev,
            touched,
            seen: s,
            seen_gen: g,
            ..
        } = search;
        (base, parent, even, seen, seen_gen) = (b, p, ev, s, g);
        if end < 0 {
            for &v in &touched {
                if even[v] || parent[v] != -1 {
                    frozen[v] = true;
                }
            }
        } else {
            let mut vertex = end;
            while vertex != -1 {
                let previous = parent[vertex as usize];
                let following = mate[previous as usize];
                mate[vertex as usize] = previous;
                mate[previous as usize] = vertex;
                vertex = following;
            }
            rescued += 1;
        }
        for &v in &touched {
            base[v] = -1;
            parent[v] = -1;
            even[v] = false;
        }
    }
    rescued
}
