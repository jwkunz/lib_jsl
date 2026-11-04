//! Graph representation: adjacency lists with conversions to edge lists and matrices, traversal,
//! connected components and a disjoint-set (union-find) structure.
//!
//! See the book entry [Graphs, traversal and orderings](https://jwkunz.github.io/lib_jsl/book/graph/traversal.html).

use lib_jsl_core::ErrorsJSL;
use std::collections::VecDeque;

/// A weighted edge `from → to` (both directions for undirected graphs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// A weighted graph on vertices `0..n` stored as adjacency lists. Undirected graphs store each
/// edge in both lists; [`Graph::edges`] reports it once.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    directed: bool,
    adjacency: Vec<Vec<(usize, f64)>>,
}

impl Graph {
    /// A graph with `n` vertices and no edges.
    pub fn new(n: usize, directed: bool) -> Self {
        Graph { directed, adjacency: vec![Vec::new(); n] }
    }

    /// A graph from `(from, to, weight)` triples.
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)], directed: bool) -> Result<Self, ErrorsJSL> {
        let mut g = Graph::new(n, directed);
        for &(u, v, w) in edges {
            g.add_edge(u, v, w)?;
        }
        Ok(g)
    }

    /// A graph from unweighted `(from, to)` pairs (weight 1).
    pub fn from_unweighted_edges(n: usize, edges: &[(usize, usize)], directed: bool) -> Result<Self, ErrorsJSL> {
        Self::from_edges(n, &edges.iter().map(|&(u, v)| (u, v, 1.0)).collect::<Vec<_>>(), directed)
    }

    /// A graph from a square adjacency matrix: entry `(i, j)` is the weight of `i → j`, and `None`
    /// or a NaN entry means no edge. Undirected graphs read the upper triangle.
    pub fn from_adjacency_matrix(matrix: &[Vec<Option<f64>>], directed: bool) -> Result<Self, ErrorsJSL> {
        let n = matrix.len();
        if matrix.iter().any(|row| row.len() != n) {
            return Err(ErrorsJSL::InvalidInputRange("The adjacency matrix must be square."));
        }
        let mut g = Graph::new(n, directed);
        for (i, row) in matrix.iter().enumerate() {
            for (j, entry) in row.iter().enumerate().skip(if directed { 0 } else { i }) {
                if let Some(w) = *entry
                    && !w.is_nan()
                {
                    g.add_edge(i, j, w)?;
                }
            }
        }
        Ok(g)
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> Result<(), ErrorsJSL> {
        let n = self.adjacency.len();
        if from >= n || to >= n {
            return Err(ErrorsJSL::InvalidInputRange("Edge endpoint out of range."));
        }
        self.adjacency[from].push((to, weight));
        if !self.directed && from != to {
            self.adjacency[to].push((from, weight));
        }
        Ok(())
    }

    pub fn vertex_count(&self) -> usize {
        self.adjacency.len()
    }

    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// The out-neighbours of `v` with edge weights.
    pub fn neighbors(&self, v: usize) -> &[(usize, f64)] {
        &self.adjacency[v]
    }

    /// The out-degree of `v` (the degree for undirected graphs).
    pub fn degree(&self, v: usize) -> usize {
        self.adjacency[v].len()
    }

    /// Every edge once.
    pub fn edges(&self) -> Vec<Edge> {
        let mut out = Vec::new();
        for (u, list) in self.adjacency.iter().enumerate() {
            for &(v, w) in list {
                if self.directed || u <= v {
                    out.push(Edge { from: u, to: v, weight: w });
                }
            }
        }
        out
    }

    pub fn edge_count(&self) -> usize {
        self.edges().len()
    }

    /// The adjacency matrix (`None` where there is no edge; the smallest weight for parallel
    /// edges).
    pub fn adjacency_matrix(&self) -> Vec<Vec<Option<f64>>> {
        let n = self.vertex_count();
        let mut m = vec![vec![None; n]; n];
        for (u, list) in self.adjacency.iter().enumerate() {
            for &(v, w) in list {
                m[u][v] = Some(m[u][v].map_or(w, |old: f64| old.min(w)));
            }
        }
        m
    }

    /// The graph with every edge reversed (itself for undirected graphs).
    pub fn reversed(&self) -> Graph {
        if !self.directed {
            return self.clone();
        }
        let mut g = Graph::new(self.vertex_count(), true);
        for e in self.edges() {
            g.adjacency[e.to].push((e.from, e.weight));
        }
        g
    }

    /// Vertices reachable from `source` in breadth-first order.
    pub fn breadth_first(&self, source: usize) -> Vec<usize> {
        let mut seen = vec![false; self.vertex_count()];
        let mut order = Vec::new();
        let mut queue = VecDeque::from([source]);
        seen[source] = true;
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &(v, _) in &self.adjacency[u] {
                if !seen[v] {
                    seen[v] = true;
                    queue.push_back(v);
                }
            }
        }
        order
    }

    /// Vertices reachable from `source` in depth-first (preorder), visiting neighbours in
    /// insertion order; iterative, so deep graphs do not overflow the stack.
    pub fn depth_first(&self, source: usize) -> Vec<usize> {
        let mut seen = vec![false; self.vertex_count()];
        let mut order = Vec::new();
        let mut stack = vec![source];
        while let Some(u) = stack.pop() {
            if seen[u] {
                continue;
            }
            seen[u] = true;
            order.push(u);
            for &(v, _) in self.adjacency[u].iter().rev() {
                if !seen[v] {
                    stack.push(v);
                }
            }
        }
        order
    }

    /// The component label of every vertex (weakly connected components for directed graphs),
    /// labels numbered from 0 in order of first appearance.
    pub fn connected_components(&self) -> Vec<usize> {
        let mut sets = UnionFind::new(self.vertex_count());
        for e in self.edges() {
            sets.union(e.from, e.to);
        }
        let mut label = vec![usize::MAX; self.vertex_count()];
        let mut next = 0;
        (0..self.vertex_count())
            .map(|v| {
                let root = sets.find(v);
                if label[root] == usize::MAX {
                    label[root] = next;
                    next += 1;
                }
                label[root]
            })
            .collect()
    }
}

/// A disjoint-set forest with union by rank and path halving (near-constant amortised time).
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
    sets: usize,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect(), rank: vec![0; n], sets: n }
    }

    /// The representative of `x`'s set.
    // ANCHOR: union_find
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merges the sets of `a` and `b`; returns false if they were already joined.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        let (high, low) = if self.rank[ra] >= self.rank[rb] { (ra, rb) } else { (rb, ra) };
        self.parent[low] = high;
        if self.rank[high] == self.rank[low] {
            self.rank[high] += 1;
        }
        self.sets -= 1;
        true
    }
    // ANCHOR_END: union_find

    pub fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    /// The number of disjoint sets.
    pub fn set_count(&self) -> usize {
        self.sets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representations_convert_consistently() {
        let g = Graph::from_edges(4, &[(0, 1, 2.0), (1, 2, 3.0), (2, 0, 1.5), (3, 3, 0.5)], false).unwrap();
        assert_eq!(g.edge_count(), 4);
        assert_eq!(g.degree(0), 2);
        let m = g.adjacency_matrix();
        assert_eq!((m[1][0], m[0][1], m[0][3], m[3][3]), (Some(2.0), Some(2.0), None, Some(0.5)));
        assert_eq!(Graph::from_adjacency_matrix(&m, false).unwrap().edges(), g.edges());
        let d = Graph::from_unweighted_edges(3, &[(0, 1), (1, 2)], true).unwrap();
        assert_eq!(d.reversed().neighbors(2), &[(1, 1.0)]);
        assert!(d.neighbors(2).is_empty());
        assert!(Graph::from_edges(2, &[(0, 2, 1.0)], true).is_err());
    }

    #[test]
    fn traversal_components_and_union_find() {
        // Two components: a path 0-1-2-3 with a chord, and the pair 4-5.
        let g = Graph::from_unweighted_edges(6, &[(0, 1), (1, 2), (2, 3), (0, 2), (4, 5)], false).unwrap();
        assert_eq!(g.breadth_first(0), vec![0, 1, 2, 3]);
        assert_eq!(g.depth_first(0), vec![0, 1, 2, 3]);
        assert_eq!(g.connected_components(), vec![0, 0, 0, 0, 1, 1]);
        let mut sets = UnionFind::new(5);
        assert!(sets.union(0, 1) && sets.union(3, 4) && !sets.union(1, 0));
        assert!(sets.connected(0, 1) && !sets.connected(1, 3) && sets.set_count() == 3);
        // A long path does not overflow the traversal stack.
        let n = 200_000;
        let path = Graph::from_unweighted_edges(n, &(0..n - 1).map(|i| (i, i + 1)).collect::<Vec<_>>(), true).unwrap();
        assert_eq!(path.depth_first(0).len(), n);
    }
}
