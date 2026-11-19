//! Matchings in bipartite graphs, following `scipy.sparse.csgraph`: maximum-cardinality
//! matching by Hopcroft and Karp (`maximum_bipartite_matching`), minimum-weight full matching
//! by successive shortest augmenting paths (`min_weight_full_bipartite_matching`), and the
//! structural rank of a sparsity pattern (`structural_rank`).
//!
//! A bipartite graph is given by its biadjacency pattern: `n_rows` row vertices, `n_cols` column
//! vertices and edges `(row, col)` (with weights for the weighted problem).
//!
//! See the book entry [Matchings, k shortest paths and Laplacians](https://jwkunz.github.io/lib_jsl/book/graph/matching-laplacian.html).

use crate::graph::OrdF64;
use lib_jsl_core::ErrorsJSL;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

/// A matching: the column of each row and the row of each column, `None` where unmatched.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BipartiteMatching {
    pub row_to_col: Vec<Option<usize>>,
    pub col_to_row: Vec<Option<usize>>,
}

impl BipartiteMatching {
    /// The number of matched pairs.
    pub fn size(&self) -> usize {
        self.row_to_col.iter().filter(|c| c.is_some()).count()
    }
}

fn adjacency(n_rows: usize, n_cols: usize, edges: &[(usize, usize)]) -> Result<Vec<Vec<usize>>, ErrorsJSL> {
    let mut adj = vec![Vec::new(); n_rows];
    for &(r, c) in edges {
        if r >= n_rows || c >= n_cols {
            return Err(ErrorsJSL::InvalidInputRange("bipartite matching: an edge is out of range"));
        }
        adj[r].push(c);
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    Ok(adj)
}

// ANCHOR: hopcroft_karp
/// A maximum-cardinality matching (SciPy's `maximum_bipartite_matching`) by the Hopcroft-Karp
/// algorithm, `O(E √V)`: each phase finds, by a breadth-first search from all free rows, the
/// length of the shortest augmenting paths, then augments along a maximal set of vertex-disjoint
/// paths of that length by depth-first searches. Maximum matchings are not unique; SciPy may
/// return a different one of the same size.
pub fn maximum_bipartite_matching(n_rows: usize, n_cols: usize, edges: &[(usize, usize)]) -> Result<BipartiteMatching, ErrorsJSL> {
    let adj = adjacency(n_rows, n_cols, edges)?;
    let mut row_to_col: Vec<Option<usize>> = vec![None; n_rows];
    let mut col_to_row: Vec<Option<usize>> = vec![None; n_cols];
    let inf = usize::MAX;
    let mut dist = vec![inf; n_rows];
    loop {
        // Breadth-first layering from the free rows.
        let mut queue = VecDeque::new();
        for r in 0..n_rows {
            if row_to_col[r].is_none() {
                dist[r] = 0;
                queue.push_back(r);
            } else {
                dist[r] = inf;
            }
        }
        let mut found = false;
        while let Some(r) = queue.pop_front() {
            for &c in &adj[r] {
                match col_to_row[c] {
                    None => found = true,
                    Some(r2) if dist[r2] == inf => {
                        dist[r2] = dist[r] + 1;
                        queue.push_back(r2);
                    }
                    _ => {}
                }
            }
        }
        if !found {
            break;
        }
        // Depth-first augmentation along the layers (iteratively, with per-row edge pointers).
        let mut ptr = vec![0usize; n_rows];
        for start in 0..n_rows {
            if row_to_col[start].is_some() {
                continue;
            }
            let mut stack = vec![start];
            while let Some(&r) = stack.last() {
                if ptr[r] == adj[r].len() {
                    dist[r] = inf;
                    stack.pop();
                    continue;
                }
                let c = adj[r][ptr[r]];
                ptr[r] += 1;
                match col_to_row[c] {
                    None => {
                        // Augment along the stack.
                        let mut col = c;
                        for &row in stack.iter().rev() {
                            let prev = row_to_col[row];
                            row_to_col[row] = Some(col);
                            col_to_row[col] = Some(row);
                            match prev {
                                Some(p) => col = p,
                                None => break,
                            }
                        }
                        break;
                    }
                    Some(r2) if dist[r2] == dist[r] + 1 => stack.push(r2),
                    _ => {}
                }
            }
        }
    }
    Ok(BipartiteMatching { row_to_col, col_to_row })
}
// ANCHOR_END: hopcroft_karp

/// The structural rank of a sparsity pattern (SciPy's `structural_rank`): the largest rank any
/// matrix with this pattern can have, the size of a maximum matching between rows and columns.
pub fn structural_rank(n_rows: usize, n_cols: usize, nonzeros: &[(usize, usize)]) -> Result<usize, ErrorsJSL> {
    Ok(maximum_bipartite_matching(n_rows, n_cols, nonzeros)?.size())
}


// ANCHOR: min_weight
/// A minimum-weight (or, with `maximize`, maximum-weight) matching that matches every vertex on
/// the smaller side (SciPy's `min_weight_full_bipartite_matching`). Returns `(rows, cols)` of the
/// matched pairs, sorted by row. Successive shortest augmenting paths: each row in turn is
/// matched by Dijkstra's algorithm on the reduced costs `w(r, c) − u(r) − v(c) ≥ 0`, which the
/// dual potentials `u`, `v` keep non-negative; `O(V E log V)`. Errors if no full matching exists.
pub fn min_weight_full_bipartite_matching(n_rows: usize, n_cols: usize, edges: &[(usize, usize, f64)], maximize: bool) -> Result<(Vec<usize>, Vec<usize>), ErrorsJSL> {
    if edges.iter().any(|e| !e.2.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("min_weight_full_bipartite_matching: weights must be finite"));
    }
    // Work with rows as the smaller side.
    let transpose = n_rows > n_cols;
    let (nr, nc) = if transpose { (n_cols, n_rows) } else { (n_rows, n_cols) };
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nr];
    for &(r, c, w) in edges {
        if r >= n_rows || c >= n_cols {
            return Err(ErrorsJSL::InvalidInputRange("min_weight_full_bipartite_matching: an edge is out of range"));
        }
        let (r, c) = if transpose { (c, r) } else { (r, c) };
        adj[r].push((c, if maximize { -w } else { w }));
    }
    // Initial potentials: u(r) = min_c w(r, c), v = 0, so every reduced cost is non-negative.
    let mut u: Vec<f64> = adj.iter().map(|a| a.iter().map(|e| e.1).fold(f64::INFINITY, f64::min)).collect();
    if u.iter().any(|x| x.is_infinite()) {
        return Err(ErrorsJSL::InvalidInputRange("min_weight_full_bipartite_matching: no full matching exists"));
    }
    let mut v = vec![0.0f64; nc];
    let mut row_of: Vec<Option<usize>> = vec![None; nc];
    let mut col_of: Vec<Option<usize>> = vec![None; nr];
    for start in 0..nr {
        // Dijkstra over columns: dist[c] is the reduced length of the best alternating path from
        // `start` to column c; prev[c] the row it was reached from.
        let mut dist = vec![f64::INFINITY; nc];
        let mut prev = vec![usize::MAX; nc];
        let mut done = vec![false; nc];
        let mut heap = BinaryHeap::new();
        for &(c, w) in &adj[start] {
            let d = w - u[start] - v[c];
            if d < dist[c] {
                dist[c] = d;
                prev[c] = start;
                heap.push(Reverse((OrdF64(d), c)));
            }
        }
        let mut free_col = None;
        let mut settled = Vec::new();
        while let Some(Reverse((OrdF64(d), c))) = heap.pop() {
            if done[c] || d > dist[c] {
                continue;
            }
            done[c] = true;
            settled.push(c);
            match row_of[c] {
                None => {
                    free_col = Some(c);
                    break;
                }
                Some(r) => {
                    for &(c2, w) in &adj[r] {
                        let nd = d + w - u[r] - v[c2];
                        if !done[c2] && nd < dist[c2] {
                            dist[c2] = nd;
                            prev[c2] = r;
                            heap.push(Reverse((OrdF64(nd), c2)));
                        }
                    }
                }
            }
        }
        let Some(end) = free_col else {
            return Err(ErrorsJSL::InvalidInputRange("min_weight_full_bipartite_matching: no full matching exists"));
        };
        let dend = dist[end];
        // Update the potentials so matched edges stay tight and reduced costs non-negative.
        u[start] += dend;
        for &c in &settled {
            if c != end {
                let r = row_of[c].expect("settled columns before the end are matched");
                u[r] += dend - dist[c];
                v[c] -= dend - dist[c];
            }
        }
        // Augment along the alternating path.
        let mut c = end;
        loop {
            let r = prev[c];
            let next = col_of[r];
            col_of[r] = Some(c);
            row_of[c] = Some(r);
            if r == start {
                break;
            }
            c = next.expect("an inner row on the path was matched");
        }
    }
    let mut pairs: Vec<(usize, usize)> = (0..nr).map(|r| (r, col_of[r].expect("all rows matched"))).collect();
    if transpose {
        pairs = pairs.into_iter().map(|(r, c)| (c, r)).collect();
    }
    pairs.sort_unstable();
    Ok(pairs.into_iter().unzip())
}
// ANCHOR_END: min_weight

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matchings_are_maximum_and_optimal() {
        // A 4 × 5 pattern with a maximum matching of size 3 (rows 0 and 1 share one column).
        let e = [(0, 0), (1, 0), (2, 1), (2, 2), (3, 3), (3, 4)];
        let m = maximum_bipartite_matching(4, 5, &e).unwrap();
        assert_eq!(m.size(), 3);
        for (r, c) in m.row_to_col.iter().enumerate() {
            if let Some(c) = c {
                assert!(e.contains(&(r, *c)) && m.col_to_row[*c] == Some(r));
            }
        }
        assert_eq!(structural_rank(3, 3, &[(0, 0), (0, 1), (1, 0), (2, 0)]).unwrap(), 2);
        // Weighted: brute force over all permutations of a dense 4 × 4 matrix.
        let w = [[4.0, 1.0, 3.0, 2.0], [2.0, 0.5, 5.0, 3.0], [3.0, 2.0, 2.0, 1.0], [1.5, 4.0, 3.0, 0.2]];
        let edges: Vec<(usize, usize, f64)> = (0..4).flat_map(|r| (0..4).map(move |c| (r, c, w[r][c]))).collect();
        let (rows, cols) = min_weight_full_bipartite_matching(4, 4, &edges, false).unwrap();
        let got: f64 = rows.iter().zip(&cols).map(|(&r, &c)| w[r][c]).sum();
        let mut best = f64::INFINITY;
        let mut perm = [0, 1, 2, 3];
        fn permute(k: usize, p: &mut [usize; 4], w: &[[f64; 4]; 4], best: &mut f64) {
            if k == 4 {
                *best = best.min((0..4).map(|r| w[r][p[r]]).sum());
                return;
            }
            for i in k..4 {
                p.swap(k, i);
                permute(k + 1, p, w, best);
                p.swap(k, i);
            }
        }
        permute(0, &mut perm, &w, &mut best);
        assert!((got - best).abs() < 1e-12);
        assert!(min_weight_full_bipartite_matching(2, 2, &[(0, 0, 1.0), (1, 0, 1.0)], false).is_err());
    }
}
