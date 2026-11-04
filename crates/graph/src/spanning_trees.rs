//! Minimum spanning trees (forests) of undirected graphs: Kruskal and Prim.
//!
//! See the book entry [Minimum spanning trees and union-find](https://jwkunz.github.io/lib_jsl/book/graph/spanning-trees.html).

use crate::graph::{Edge, Graph, UnionFind};
use lib_jsl_core::ErrorsJSL;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A minimum spanning forest: its edges and total weight (one tree per connected component).
#[derive(Debug, Clone, PartialEq)]
pub struct SpanningForest {
    pub edges: Vec<Edge>,
    pub weight: f64,
}

fn undirected(g: &Graph) -> Result<(), ErrorsJSL> {
    if g.is_directed() {
        return Err(ErrorsJSL::InvalidInputRange("Spanning trees need an undirected graph."));
    }
    Ok(())
}

/// Kruskal's algorithm: edges by increasing weight, kept when they join two components
/// (union-find). `O(E log E)`.
pub fn kruskal(g: &Graph) -> Result<SpanningForest, ErrorsJSL> {
    undirected(g)?;
    // ANCHOR: kruskal
    let mut edges = g.edges();
    edges.sort_by(|a, b| a.weight.total_cmp(&b.weight));
    let mut sets = UnionFind::new(g.vertex_count());
    let chosen: Vec<Edge> = edges.into_iter().filter(|e| sets.union(e.from, e.to)).collect();
    // ANCHOR_END: kruskal
    let weight = chosen.iter().map(|e| e.weight).sum();
    Ok(SpanningForest { edges: chosen, weight })
}

#[derive(PartialEq)]
struct Candidate(f64, usize, usize);
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0)
    }
}

/// Prim's algorithm: grows each tree from a root by the cheapest edge leaving it (binary heap).
/// `O(E log V)`; better than Kruskal on dense graphs.
pub fn prim(g: &Graph) -> Result<SpanningForest, ErrorsJSL> {
    undirected(g)?;
    let n = g.vertex_count();
    let mut in_tree = vec![false; n];
    let mut chosen = Vec::new();
    for root in 0..n {
        if in_tree[root] {
            continue;
        }
        in_tree[root] = true;
        let mut heap: BinaryHeap<Candidate> = g.neighbors(root).iter().map(|&(v, w)| Candidate(w, root, v)).collect();
        while let Some(Candidate(w, u, v)) = heap.pop() {
            if in_tree[v] {
                continue;
            }
            in_tree[v] = true;
            chosen.push(Edge { from: u, to: v, weight: w });
            heap.extend(g.neighbors(v).iter().filter(|&&(x, _)| !in_tree[x]).map(|&(x, wx)| Candidate(wx, v, x)));
        }
    }
    let weight = chosen.iter().map(|e| e.weight).sum();
    Ok(SpanningForest { edges: chosen, weight })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_algorithms_find_the_same_minimum_weight() {
        // The classic 9-vertex example: MST weight 37.
        let g = Graph::from_edges(
            9,
            &[(0, 1, 4.0), (0, 7, 8.0), (1, 2, 8.0), (1, 7, 11.0), (2, 3, 7.0), (2, 8, 2.0), (2, 5, 4.0), (3, 4, 9.0), (3, 5, 14.0), (4, 5, 10.0), (5, 6, 2.0), (6, 7, 1.0), (6, 8, 6.0), (7, 8, 7.0)],
            false,
        )
        .unwrap();
        let (k, p) = (kruskal(&g).unwrap(), prim(&g).unwrap());
        assert_eq!((k.weight, p.weight), (37.0, 37.0));
        assert_eq!((k.edges.len(), p.edges.len()), (8, 8));
        // A forest for a disconnected graph, and directed graphs are rejected.
        let forest = Graph::from_edges(5, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0), (3, 4, 3.0)], false).unwrap();
        assert_eq!((kruskal(&forest).unwrap().weight, prim(&forest).unwrap().weight), (6.0, 6.0));
        assert!(kruskal(&Graph::new(2, true)).is_err() && prim(&Graph::new(2, true)).is_err());
    }
}
