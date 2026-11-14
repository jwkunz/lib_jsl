//! Direct sparse solvers: a left-looking sparse LU factorisation with partial pivoting
//! (Gilbert and Peierls), a fill-reducing reverse Cuthill-McKee ordering, and `spsolve`.

use super::formats::{CscMatrix, CsrMatrix};
use crate::error::{LinalgError, Result};
use std::collections::VecDeque;

// ANCHOR: rcm
/// The reverse Cuthill-McKee ordering of the symmetric pattern of `A + Aᵀ` (SciPy's
/// `reverse_cuthill_mckee`): a breadth-first search from a vertex of low degree in each
/// connected component, visiting neighbours by increasing degree, then reversed. It concentrates
/// the entries near the diagonal (reduces the bandwidth), which limits fill-in during elimination.
pub fn reverse_cuthill_mckee(a: &CsrMatrix) -> Vec<usize> {
    let n = a.nrows.max(a.ncols);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..a.nrows {
        for p in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[p];
            if i != j {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    for list in adj.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    let degree: Vec<usize> = adj.iter().map(|l| l.len()).collect();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut by_degree: Vec<usize> = (0..n).collect();
    by_degree.sort_by_key(|&v| degree[v]);
    for &start in &by_degree {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            let mut next: Vec<usize> = adj[v].iter().copied().filter(|&w| !visited[w]).collect();
            next.sort_by_key(|&w| degree[w]);
            for w in next {
                visited[w] = true;
                queue.push_back(w);
            }
        }
    }
    order.reverse();
    order
}
// ANCHOR_END: rcm

/// The bandwidth `max |i − j|` over the stored entries of `A` with rows and columns permuted by
/// `perm` (`perm[k]` is the original index placed at position `k`).
pub fn bandwidth(a: &CsrMatrix, perm: Option<&[usize]>) -> usize {
    let n = a.nrows;
    let mut pos: Vec<usize> = (0..n).collect();
    if let Some(p) = perm {
        for (k, &v) in p.iter().enumerate() {
            pos[v] = k;
        }
    }
    let mut b = 0;
    for i in 0..n {
        for p in a.indptr[i]..a.indptr[i + 1] {
            b = b.max(pos[i].abs_diff(pos[a.indices[p]]));
        }
    }
    b
}

/// Column orderings for [`SparseLu`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// The columns in their natural order.
    Natural,
    /// Reverse Cuthill-McKee on the pattern of `A + Aᵀ`.
    ReverseCuthillMcKee,
}

/// A sparse LU factorisation `P A Q = L U` (SciPy's `splu`, with its own algorithm).
#[derive(Clone, Debug)]
pub struct SparseLu {
    n: usize,
    /// Column `k` of L: (original row, value), the unit diagonal stored first.
    l: Vec<Vec<(usize, f64)>>,
    /// Column `k` of U: (step, value), the diagonal last.
    u: Vec<Vec<(usize, f64)>>,
    /// The original row pivoted at each step.
    pivot_row: Vec<usize>,
    /// The column order.
    q: Vec<usize>,
}

// ANCHOR: splu
impl SparseLu {
    /// Factorises a square sparse matrix column by column (left-looking): each column of `A Q` is
    /// solved against the columns of `L` found so far, visiting only the rows its pattern can
    /// reach in the graph of `L` (found by a depth-first search, which also gives a valid order),
    /// and the largest remaining entry becomes the pivot.
    pub fn factor(a: &CsrMatrix, ordering: Ordering) -> Result<SparseLu> {
        let n = a.nrows;
        if a.ncols != n {
            return Err(LinalgError::NotSquare { rows: a.nrows, cols: a.ncols });
        }
        let csc: CscMatrix = a.to_csc();
        let q = match ordering {
            Ordering::Natural => (0..n).collect(),
            Ordering::ReverseCuthillMcKee => reverse_cuthill_mckee(a),
        };
        let scale = a.data.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        let mut l: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut u: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut pinv: Vec<Option<usize>> = vec![None; n];
        let mut pivot_row = Vec::with_capacity(n);
        let mut x = vec![0.0; n];
        let mut mark = vec![usize::MAX; n];
        for (k, &col) in q.iter().enumerate() {
            // Depth-first search for the reach of the column's pattern in the graph of L.
            let mut topo: Vec<usize> = Vec::new();
            let mut stack: Vec<(usize, usize)> = Vec::new();
            for p in csc.indptr[col]..csc.indptr[col + 1] {
                let r = csc.indices[p];
                if mark[r] == k {
                    continue;
                }
                mark[r] = k;
                stack.push((r, 0));
                while let Some((v, child)) = stack.pop() {
                    let children = pinv[v].map(|j| &l[j][1..]);
                    match children.and_then(|c| c.get(child)) {
                        Some(&(w, _)) => {
                            stack.push((v, child + 1));
                            if mark[w] != k {
                                mark[w] = k;
                                stack.push((w, 0));
                            }
                        }
                        None => topo.push(v),
                    }
                }
            }
            for p in csc.indptr[col]..csc.indptr[col + 1] {
                x[csc.indices[p]] = csc.data[p];
            }
            // Sparse triangular solve in reverse postorder.
            for &r in topo.iter().rev() {
                if let Some(j) = pinv[r] {
                    let ujk = x[r];
                    for &(i, lij) in &l[j][1..] {
                        x[i] -= lij * ujk;
                    }
                }
            }
            // Split into U (pivoted rows) and the candidates for L.
            let mut ucol = Vec::new();
            let mut best: Option<(usize, f64)> = None;
            for &r in &topo {
                match pinv[r] {
                    Some(j) => {
                        if x[r] != 0.0 {
                            ucol.push((j, x[r]));
                        }
                    }
                    None => {
                        if best.is_none_or(|(_, v)| x[r].abs() > v.abs()) {
                            best = Some((r, x[r]));
                        }
                    }
                }
            }
            let Some((piv, d)) = best.filter(|(_, v)| v.abs() > f64::EPSILON * scale * 1e-3) else {
                return Err(LinalgError::Singular(k));
            };
            if d == 0.0 {
                return Err(LinalgError::Singular(k));
            }
            ucol.sort_by_key(|e| e.0);
            ucol.push((k, d));
            let mut lcol = vec![(piv, 1.0)];
            for &r in &topo {
                if pinv[r].is_none() && r != piv && x[r] != 0.0 {
                    lcol.push((r, x[r] / d));
                }
            }
            for &r in &topo {
                x[r] = 0.0;
            }
            pinv[piv] = Some(k);
            pivot_row.push(piv);
            l.push(lcol);
            u.push(ucol);
        }
        Ok(SparseLu { n, l, u, pivot_row, q })
    }

    /// Solves `A x = b`.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        if b.len() != self.n {
            return Err(LinalgError::Shape(format!("right-hand side of length {}, expected {}", b.len(), self.n)));
        }
        // Forward: L y = P b, in step order.
        let mut w = b.to_vec();
        let mut y = vec![0.0; self.n];
        for j in 0..self.n {
            let yj = w[self.pivot_row[j]];
            y[j] = yj;
            for &(i, lij) in &self.l[j][1..] {
                w[i] -= lij * yj;
            }
        }
        // Backward: U z = y.
        for j in (0..self.n).rev() {
            let (_, ujj) = *self.u[j].last().expect("diagonal");
            let zj = y[j] / ujj;
            y[j] = zj;
            for &(i, uij) in &self.u[j][..self.u[j].len() - 1] {
                y[i] -= uij * zj;
            }
        }
        let mut x = vec![0.0; self.n];
        for (k, &col) in self.q.iter().enumerate() {
            x[col] = y[k];
        }
        Ok(x)
    }

    /// The number of stored entries of `L` and `U` together (the fill).
    pub fn nnz(&self) -> usize {
        self.l.iter().map(|c| c.len()).sum::<usize>() + self.u.iter().map(|c| c.len()).sum::<usize>()
    }
}
// ANCHOR_END: splu

/// Solves the sparse system `A x = b` (SciPy's `spsolve`) by a sparse LU factorisation with the
/// reverse Cuthill-McKee column ordering.
pub fn spsolve(a: &CsrMatrix, b: &[f64]) -> Result<Vec<f64>> {
    SparseLu::factor(a, Ordering::ReverseCuthillMcKee)?.solve(b)
}
