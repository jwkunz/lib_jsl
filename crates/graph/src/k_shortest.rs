//! The `K` shortest loopless paths between two vertices by Yen's algorithm (SciPy's `yen`).
//!
//! See the book entry [Matchings, k shortest paths and Laplacians](https://jwkunz.github.io/lib_jsl/book/graph/matching-laplacian.html#yens-k-shortest-paths).

use crate::graph::Graph;
use crate::graph::OrdF64;
use lib_jsl_core::ErrorsJSL;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};


/// Dijkstra from `s` to `t` avoiding the vertices in `blocked_v` and the arcs in `blocked_e`.
fn restricted_path(g: &Graph, s: usize, t: usize, blocked_v: &[bool], blocked_e: &HashSet<(usize, usize)>) -> Option<(f64, Vec<usize>)> {
    let n = g.vertex_count();
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    dist[s] = 0.0;
    let mut heap = BinaryHeap::from([Reverse((OrdF64(0.0), s))]);
    while let Some(Reverse((OrdF64(d), u))) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        if u == t {
            break;
        }
        for &(v, w) in g.neighbors(u) {
            if blocked_v[v] || blocked_e.contains(&(u, v)) {
                continue;
            }
            if d + w < dist[v] {
                dist[v] = d + w;
                pred[v] = Some(u);
                heap.push(Reverse((OrdF64(d + w), v)));
            }
        }
    }
    if !dist[t].is_finite() {
        return None;
    }
    let mut path = vec![t];
    let mut v = t;
    while let Some(p) = pred[v] {
        path.push(p);
        v = p;
    }
    path.reverse();
    Some((dist[t], path))
}

fn arc_weight(g: &Graph, u: usize, v: usize) -> f64 {
    g.neighbors(u).iter().filter(|e| e.0 == v).map(|e| e.1).fold(f64::INFINITY, f64::min)
}

// ANCHOR: yen
/// The `k` shortest loopless paths from `source` to `sink`, shortest first, as
/// `(length, vertices)` (SciPy's `yen`). Yen's algorithm takes each path found so far and, for
/// every vertex on it (the spur), finds the shortest path from the spur that leaves the common
/// prefix differently from every path already found, avoiding the prefix's vertices; the best of
/// these candidates is the next path. Weights must be non-negative. Fewer than `k` paths are
/// returned if fewer exist.
pub fn yen(g: &Graph, source: usize, sink: usize, k: usize) -> Result<Vec<(f64, Vec<usize>)>, ErrorsJSL> {
    let n = g.vertex_count();
    if source >= n || sink >= n {
        return Err(ErrorsJSL::InvalidInputRange("yen: vertex out of range"));
    }
    if g.edges().iter().any(|e| e.weight.is_nan() || e.weight < 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("yen: weights must be non-negative"));
    }
    let mut found: Vec<(f64, Vec<usize>)> = Vec::new();
    if k == 0 {
        return Ok(found);
    }
    let none = vec![false; n];
    match restricted_path(g, source, sink, &none, &HashSet::new()) {
        Some(p) => found.push(p),
        None => return Ok(found),
    }
    let mut candidates: Vec<(f64, Vec<usize>)> = Vec::new();
    while found.len() < k {
        let last = found.last().expect("non-empty").1.clone();
        for i in 0..last.len() - 1 {
            let spur = last[i];
            let root = &last[..=i];
            let mut blocked_e = HashSet::new();
            for (_, p) in &found {
                if p.len() > i && &p[..=i] == root {
                    blocked_e.insert((p[i], p[i + 1]));
                    if !g.is_directed() {
                        blocked_e.insert((p[i + 1], p[i]));
                    }
                }
            }
            let mut blocked_v = vec![false; n];
            for &v in &root[..i] {
                blocked_v[v] = true;
            }
            if let Some((_, spur_path)) = restricted_path(g, spur, sink, &blocked_v, &blocked_e) {
                let mut path = root[..i].to_vec();
                path.extend(spur_path);
                let cost: f64 = path.windows(2).map(|w| arc_weight(g, w[0], w[1])).sum();
                if !candidates.iter().any(|c| c.1 == path) && !found.iter().any(|c| c.1 == path) {
                    candidates.push((cost, path));
                }
            }
        }
        if candidates.is_empty() {
            break;
        }
        // The cheapest candidate; ties to the one with fewer vertices, then the first found.
        let best = (0..candidates.len())
            .min_by(|&a, &b| candidates[a].0.total_cmp(&candidates[b].0).then(candidates[a].1.len().cmp(&candidates[b].1.len())))
            .expect("non-empty");
        found.push(candidates.remove(best));
    }
    Ok(found)
}
// ANCHOR_END: yen

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wikipedia_example() {
        // Vertices C, D, E, F, G, H as 0..6; the three shortest C → H paths have lengths 5, 7, 8.
        let g = Graph::from_edges(6, &[(0, 1, 3.0), (0, 2, 2.0), (1, 3, 4.0), (2, 1, 1.0), (2, 3, 2.0), (2, 4, 3.0), (3, 4, 2.0), (3, 5, 1.0), (4, 5, 2.0)], true).unwrap();
        let p = yen(&g, 0, 5, 3).unwrap();
        let lengths: Vec<f64> = p.iter().map(|x| x.0).collect();
        assert_eq!(lengths, vec![5.0, 7.0, 8.0]);
        assert_eq!(p[0].1, vec![0, 2, 3, 5]);
        // All loopless.
        for (_, path) in &p {
            let set: HashSet<_> = path.iter().collect();
            assert_eq!(set.len(), path.len());
        }
        assert_eq!(yen(&g, 0, 5, 100).unwrap().len(), 7);
    }
}
