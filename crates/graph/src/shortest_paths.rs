//! Single-source shortest paths: Dijkstra, Bellman-Ford and A*.
//!
//! See the book entry [Shortest paths](https://jwkunz.github.io/lib_jsl/book/graph/shortest-paths.html).

use crate::graph::Graph;
use lib_jsl_core::ErrorsJSL;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Distances from a source and the predecessor tree for path reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct ShortestPaths {
    pub source: usize,
    /// `f64::INFINITY` for unreachable vertices.
    pub distance: Vec<f64>,
    /// The previous vertex on a shortest path (`None` for the source and unreachable vertices).
    pub predecessor: Vec<Option<usize>>,
}

impl ShortestPaths {
    /// The vertices of a shortest path from the source to `target`, or `None` if unreachable.
    pub fn path_to(&self, target: usize) -> Option<Vec<usize>> {
        if !self.distance[target].is_finite() {
            return None;
        }
        let mut path = vec![target];
        let mut v = target;
        while let Some(p) = self.predecessor[v] {
            path.push(p);
            v = p;
        }
        path.reverse();
        Some(path)
    }
}

/// A min-heap entry ordered by cost.
#[derive(PartialEq)]
struct State {
    cost: f64,
    vertex: usize,
}
impl Eq for State {}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

fn check_source(g: &Graph, source: usize) -> Result<(), ErrorsJSL> {
    if source >= g.vertex_count() {
        return Err(ErrorsJSL::InvalidInputRange("The source vertex is out of range."));
    }
    Ok(())
}

/// Dijkstra's algorithm with a binary heap, `O((V + E) log V)`. Edge weights must be
/// non-negative.
pub fn dijkstra(g: &Graph, source: usize) -> Result<ShortestPaths, ErrorsJSL> {
    check_source(g, source)?;
    if g.edges().iter().any(|e| e.weight < 0.0 || e.weight.is_nan()) {
        return Err(ErrorsJSL::InvalidInputRange("Dijkstra's algorithm needs non-negative weights; use Bellman-Ford."));
    }
    let n = g.vertex_count();
    let mut distance = vec![f64::INFINITY; n];
    let mut predecessor = vec![None; n];
    distance[source] = 0.0;
    let mut heap = BinaryHeap::from([State { cost: 0.0, vertex: source }]);
    // ANCHOR: dijkstra
    while let Some(State { cost, vertex }) = heap.pop() {
        if cost > distance[vertex] {
            continue;
        }
        for &(next, w) in g.neighbors(vertex) {
            let candidate = cost + w;
            if candidate < distance[next] {
                distance[next] = candidate;
                predecessor[next] = Some(vertex);
                heap.push(State { cost: candidate, vertex: next });
            }
        }
    }
    // ANCHOR_END: dijkstra
    Ok(ShortestPaths { source, distance, predecessor })
}

/// The Bellman-Ford algorithm, `O(V E)`, allowing negative weights. Returns an error if a
/// negative cycle is reachable from the source (shortest paths are then undefined).
pub fn bellman_ford(g: &Graph, source: usize) -> Result<ShortestPaths, ErrorsJSL> {
    check_source(g, source)?;
    let n = g.vertex_count();
    let edges = g.edges();
    let both_ways = !g.is_directed();
    let mut distance = vec![f64::INFINITY; n];
    let mut predecessor = vec![None; n];
    distance[source] = 0.0;
    let relax = |distance: &mut Vec<f64>, predecessor: &mut Vec<Option<usize>>| {
        let mut changed = false;
        for e in &edges {
            for (u, v) in if both_ways { vec![(e.from, e.to), (e.to, e.from)] } else { vec![(e.from, e.to)] } {
                if distance[u].is_finite() && distance[u] + e.weight < distance[v] {
                    distance[v] = distance[u] + e.weight;
                    predecessor[v] = Some(u);
                    changed = true;
                }
            }
        }
        changed
    };
    for _ in 1..n {
        if !relax(&mut distance, &mut predecessor) {
            return Ok(ShortestPaths { source, distance, predecessor });
        }
    }
    if relax(&mut distance, &mut predecessor) {
        return Err(ErrorsJSL::RuntimeError("A negative cycle is reachable from the source."));
    }
    Ok(ShortestPaths { source, distance, predecessor })
}

/// A* search from `source` to `target` guided by `heuristic(v)`, an estimate of the remaining
/// cost. With an admissible (never overestimating) and consistent heuristic the returned path is
/// optimal; a zero heuristic reduces to Dijkstra. Returns `(cost, path)` or `None` if
/// unreachable.
pub fn a_star<H: Fn(usize) -> f64>(g: &Graph, source: usize, target: usize, heuristic: H) -> Result<Option<(f64, Vec<usize>)>, ErrorsJSL> {
    check_source(g, source)?;
    check_source(g, target)?;
    let n = g.vertex_count();
    let mut cost = vec![f64::INFINITY; n];
    let mut predecessor = vec![None; n];
    let mut closed = vec![false; n];
    cost[source] = 0.0;
    let mut heap = BinaryHeap::from([State { cost: heuristic(source), vertex: source }]);
    while let Some(State { vertex, .. }) = heap.pop() {
        if vertex == target {
            let paths = ShortestPaths { source, distance: cost.clone(), predecessor };
            return Ok(Some((cost[target], paths.path_to(target).expect("reached"))));
        }
        if closed[vertex] {
            continue;
        }
        closed[vertex] = true;
        for &(next, w) in g.neighbors(vertex) {
            if w < 0.0 {
                return Err(ErrorsJSL::InvalidInputRange("A* needs non-negative weights."));
            }
            let candidate = cost[vertex] + w;
            if candidate < cost[next] {
                cost[next] = candidate;
                predecessor[next] = Some(vertex);
                heap.push(State { cost: candidate + heuristic(next), vertex: next });
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classic() -> Graph {
        // The six-vertex example from CLRS-style textbooks.
        Graph::from_edges(6, &[(0, 1, 7.0), (0, 2, 9.0), (0, 5, 14.0), (1, 2, 10.0), (1, 3, 15.0), (2, 3, 11.0), (2, 5, 2.0), (3, 4, 6.0), (4, 5, 9.0)], false).unwrap()
    }

    #[test]
    fn dijkstra_and_bellman_ford_agree() {
        let g = classic();
        let d = dijkstra(&g, 0).unwrap();
        assert_eq!(d.distance, vec![0.0, 7.0, 9.0, 20.0, 20.0, 11.0]);
        assert_eq!(d.path_to(4).unwrap(), vec![0, 2, 5, 4]);
        assert_eq!(bellman_ford(&g, 0).unwrap().distance, d.distance);
        let disconnected = Graph::from_edges(3, &[(0, 1, 1.0)], true).unwrap();
        let s = dijkstra(&disconnected, 0).unwrap();
        assert!(s.distance[2].is_infinite() && s.path_to(2).is_none());
        assert!(dijkstra(&Graph::from_edges(2, &[(0, 1, -1.0)], true).unwrap(), 0).is_err());
    }

    #[test]
    fn bellman_ford_handles_negative_edges_and_detects_negative_cycles() {
        let g = Graph::from_edges(5, &[(0, 1, 6.0), (0, 3, 7.0), (1, 2, 5.0), (1, 3, 8.0), (1, 4, -4.0), (2, 1, -2.0), (3, 2, -3.0), (3, 4, 9.0), (4, 0, 2.0), (4, 2, 7.0)], true).unwrap();
        let s = bellman_ford(&g, 0).unwrap();
        assert_eq!(s.distance, vec![0.0, 2.0, 4.0, 7.0, -2.0]);
        assert_eq!(s.path_to(4).unwrap(), vec![0, 3, 2, 1, 4]);
        let cycle = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, -2.0), (2, 1, 1.0)], true).unwrap();
        assert!(bellman_ford(&cycle, 0).is_err());
    }

    #[test]
    fn a_star_on_a_grid_expands_optimally() {
        // A 20 × 20 grid with a wall; Manhattan distance is admissible.
        let w = 20;
        let id = |x: usize, y: usize| y * w + x;
        let wall = |x: usize, y: usize| x == 10 && y < 15;
        let mut edges = Vec::new();
        for y in 0..w {
            for x in 0..w {
                if wall(x, y) {
                    continue;
                }
                if x + 1 < w && !wall(x + 1, y) {
                    edges.push((id(x, y), id(x + 1, y), 1.0));
                }
                if y + 1 < w && !wall(x, y + 1) {
                    edges.push((id(x, y), id(x, y + 1), 1.0));
                }
            }
        }
        let g = Graph::from_edges(w * w, &edges, false).unwrap();
        let goal = id(19, 0);
        let manhattan = |v: usize| ((v % w).abs_diff(19) + (v / w)) as f64;
        let (cost, path) = a_star(&g, id(0, 0), goal, manhattan).unwrap().unwrap();
        assert_eq!(cost, dijkstra(&g, id(0, 0)).unwrap().distance[goal]);
        assert_eq!(cost, 19.0 + 2.0 * 15.0);
        assert_eq!(path.len(), cost as usize + 1);
        assert!(a_star(&g, id(10, 0), goal, manhattan).unwrap().is_none()); // inside the wall: isolated
    }
}
