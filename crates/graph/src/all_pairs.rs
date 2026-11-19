//! All-pairs shortest paths by the Floyd-Warshall algorithm and by Johnson's algorithm.
//!
//! See the book entry [Shortest paths](https://jwkunz.github.io/lib_jsl/book/graph/shortest-paths.html#floydwarshall).

use crate::graph::Graph;
use crate::graph::OrdF64;
use lib_jsl_core::ErrorsJSL;

/// The all-pairs distance matrix and the successor matrix for path reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct AllPairsShortestPaths {
    /// `distance[i][j]`, `f64::INFINITY` when `j` is unreachable from `i`.
    pub distance: Vec<Vec<f64>>,
    next: Vec<Vec<Option<usize>>>,
}

impl AllPairsShortestPaths {
    /// The vertices of a shortest path from `i` to `j`, or `None` if unreachable.
    pub fn path(&self, i: usize, j: usize) -> Option<Vec<usize>> {
        if i == j {
            return Some(vec![i]);
        }
        let mut path = vec![i];
        let mut v = i;
        while v != j {
            v = self.next[v][j]?;
            path.push(v);
        }
        Some(path)
    }
}

/// Floyd-Warshall, `O(V³)` time and `O(V²)` memory; negative edges are allowed. Returns an error
/// if the graph has a negative cycle (a negative diagonal entry).
pub fn floyd_warshall(g: &Graph) -> Result<AllPairsShortestPaths, ErrorsJSL> {
    let n = g.vertex_count();
    let mut distance = vec![vec![f64::INFINITY; n]; n];
    let mut next = vec![vec![None; n]; n];
    for (i, row) in distance.iter_mut().enumerate() {
        row[i] = 0.0;
        next[i][i] = Some(i);
    }
    for e in g.edges() {
        for (u, v) in if g.is_directed() { vec![(e.from, e.to)] } else { vec![(e.from, e.to), (e.to, e.from)] } {
            if e.weight < distance[u][v] {
                distance[u][v] = e.weight;
                next[u][v] = Some(v);
            }
        }
    }
    // ANCHOR: floyd
    for k in 0..n {
        for i in 0..n {
            let dik = distance[i][k];
            if !dik.is_finite() {
                continue;
            }
            for j in 0..n {
                let candidate = dik + distance[k][j];
                if candidate < distance[i][j] {
                    distance[i][j] = candidate;
                    next[i][j] = next[i][k];
                }
            }
        }
    }
    // ANCHOR_END: floyd
    if (0..n).any(|i| distance[i][i] < 0.0) {
        return Err(ErrorsJSL::RuntimeError("The graph contains a negative cycle."));
    }
    Ok(AllPairsShortestPaths { distance, next })
}

// ANCHOR: johnson
/// Johnson's algorithm (SciPy's `johnson`), `O(V E log V)`, better than Floyd-Warshall on sparse
/// graphs and allowing negative edges. Bellman-Ford from an added vertex joined to every vertex
/// by a zero edge gives potentials `h` with `w(u, v) + h(u) − h(v) ≥ 0`; Dijkstra from every
/// vertex on these reweighted edges, which keeps shortest paths shortest, then gives the
/// distances `d'(u, v) − h(u) + h(v)`. Errors on a negative cycle (any negative edge of an
/// undirected graph is one).
pub fn johnson(g: &Graph) -> Result<AllPairsShortestPaths, ErrorsJSL> {
    let n = g.vertex_count();
    let edges = g.edges();
    let arcs: Vec<(usize, usize, f64)> = edges
        .iter()
        .flat_map(|e| if g.is_directed() { vec![(e.from, e.to, e.weight)] } else { vec![(e.from, e.to, e.weight), (e.to, e.from, e.weight)] })
        .collect();
    if arcs.iter().any(|a| a.2.is_nan()) {
        return Err(ErrorsJSL::InvalidInputRange("johnson: edge weights must not be NaN"));
    }
    // Potentials: Bellman-Ford from a virtual source at distance 0 from everything.
    let mut h = vec![0.0f64; n];
    let mut changed = true;
    for _ in 0..=n {
        changed = false;
        for &(u, v, w) in &arcs {
            if h[u] + w < h[v] {
                h[v] = h[u] + w;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if changed {
        return Err(ErrorsJSL::RuntimeError("The graph contains a negative cycle."));
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(u, v, w) in &arcs {
        adj[u].push((v, (w + h[u] - h[v]).max(0.0)));
    }
    let mut distance = vec![vec![f64::INFINITY; n]; n];
    let mut next = vec![vec![None; n]; n];
    for s in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        let mut pred: Vec<Option<usize>> = vec![None; n];
        dist[s] = 0.0;
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(std::cmp::Reverse((OrdF64(0.0), s)));
        while let Some(std::cmp::Reverse((OrdF64(d), u))) = heap.pop() {
            if d > dist[u] {
                continue;
            }
            for &(v, w) in &adj[u] {
                if d + w < dist[v] {
                    dist[v] = d + w;
                    pred[v] = Some(u);
                    heap.push(std::cmp::Reverse((OrdF64(d + w), v)));
                }
            }
        }
        for t in 0..n {
            if dist[t].is_finite() {
                distance[s][t] = dist[t] - h[s] + h[t];
                // The first step from s towards t: walk back along the predecessors.
                let mut v = t;
                while let Some(p) = pred[v] {
                    if p == s {
                        break;
                    }
                    v = p;
                }
                next[s][t] = Some(v);
            }
        }
        next[s][s] = Some(s);
    }
    Ok(AllPairsShortestPaths { distance, next })
}
// ANCHOR_END: johnson


#[cfg(test)]
mod tests {
    use super::*;
    use crate::shortest_paths::bellman_ford;

    #[test]
    fn matches_single_source_runs_and_reconstructs_paths() {
        let g = Graph::from_edges(5, &[(0, 1, 6.0), (0, 3, 7.0), (1, 2, 5.0), (1, 3, 8.0), (1, 4, -4.0), (2, 1, -2.0), (3, 2, -3.0), (3, 4, 9.0), (4, 0, 2.0), (4, 2, 7.0)], true).unwrap();
        let all = floyd_warshall(&g).unwrap();
        for s in 0..5 {
            assert_eq!(all.distance[s], bellman_ford(&g, s).unwrap().distance, "source {s}");
        }
        assert_eq!(all.path(0, 4).unwrap(), vec![0, 3, 2, 1, 4]);
        assert_eq!(all.path(2, 2).unwrap(), vec![2]);
        let split = floyd_warshall(&Graph::from_edges(3, &[(0, 1, 1.0)], false).unwrap()).unwrap();
        assert!(split.path(0, 2).is_none() && split.distance[1][0] == 1.0);
        assert!(floyd_warshall(&Graph::from_edges(2, &[(0, 1, -1.0), (1, 0, -1.0)], true).unwrap()).is_err());
    }

    #[test]
    fn johnson_agrees_with_floyd_warshall() {
        let g = Graph::from_edges(5, &[(0, 1, 6.0), (0, 3, 7.0), (1, 2, 5.0), (1, 3, 8.0), (1, 4, -4.0), (2, 1, -2.0), (3, 2, -3.0), (3, 4, 9.0), (4, 0, 2.0), (4, 2, 7.0)], true).unwrap();
        let (a, b) = (johnson(&g).unwrap(), floyd_warshall(&g).unwrap());
        for i in 0..5 {
            for j in 0..5 {
                assert!((a.distance[i][j] - b.distance[i][j]).abs() < 1e-12);
                let p = a.path(i, j).unwrap();
                let len: f64 = p.windows(2).map(|w| g.neighbors(w[0]).iter().filter(|e| e.0 == w[1]).map(|e| e.1).fold(f64::INFINITY, f64::min)).sum();
                assert!((len - a.distance[i][j]).abs() < 1e-12);
            }
        }
        assert!(johnson(&Graph::from_edges(2, &[(0, 1, -1.0)], false).unwrap()).is_err());
    }
}
