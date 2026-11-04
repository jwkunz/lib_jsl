//! All-pairs shortest paths by the Floyd-Warshall algorithm.
//!
//! See the book entry [Shortest paths](https://jwkunz.github.io/lib_jsl/book/graph/shortest-paths.html#floydwarshall).

use crate::graph::Graph;
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
}
