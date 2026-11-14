//! Incomplete LU factorisations as preconditioners for the Krylov solvers: ILU(0), which keeps the
//! sparsity pattern of `A`, and Saad's ILUT, which drops small entries and keeps the largest few
//! per row (the role of SciPy's `spilu`, which uses SuperLU's own threshold ILU).

use super::formats::CsrMatrix;
use super::iterative::LinearOperator;
use crate::error::{LinalgError, Result};
use std::collections::BTreeMap;

/// An incomplete factorisation `A ≈ L U`, with `L` unit lower triangular (the unit diagonal not
/// stored) and `U` upper triangular, both by rows.
#[derive(Clone, Debug)]
pub struct Ilu {
    n: usize,
    /// Row `i` of `L`: `(column < i, value)`.
    l: Vec<Vec<(usize, f64)>>,
    /// Row `i` of `U`: the diagonal first, then `(column > i, value)`.
    u: Vec<Vec<(usize, f64)>>,
}

fn check_square(a: &CsrMatrix) -> Result<usize> {
    if a.nrows != a.ncols {
        return Err(LinalgError::NotSquare { rows: a.nrows, cols: a.ncols });
    }
    Ok(a.nrows)
}

/// Replaces a zero (or tiny) pivot by a small multiple of the row norm, as is usual for
/// incomplete factorisations, which have no pivoting to fall back on.
fn safe_pivot(d: f64, row_norm: f64) -> f64 {
    let floor = 1e-8 * row_norm.max(f64::MIN_POSITIVE);
    if d.abs() < floor { if d < 0.0 { -floor } else { floor } } else { d }
}

impl Ilu {
    // ANCHOR: ilu0
    /// ILU(0): Gaussian elimination row by row (the IKJ variant) in which every update that would
    /// fall outside the sparsity pattern of `A` is discarded, so `L + U` has exactly the pattern
    /// of `A`.
    pub fn ilu0(a: &CsrMatrix) -> Result<Ilu> {
        let n = check_square(a)?;
        let mut l: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut u: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut pos = vec![usize::MAX; n];
        for i in 0..n {
            let cols = &a.indices[a.indptr[i]..a.indptr[i + 1]];
            let mut w: Vec<f64> = a.data[a.indptr[i]..a.indptr[i + 1]].to_vec();
            for (p, &j) in cols.iter().enumerate() {
                pos[j] = p;
            }
            let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            for p in 0..cols.len() {
                let k = cols[p];
                if k >= i {
                    break;
                }
                let lik = w[p] / u[k][0].1;
                w[p] = lik;
                for &(j, ukj) in &u[k][1..] {
                    if pos[j] != usize::MAX {
                        w[pos[j]] -= lik * ukj;
                    }
                }
            }
            let mut li = Vec::new();
            let mut ui = vec![(i, 0.0)];
            for (p, &j) in cols.iter().enumerate() {
                match j.cmp(&i) {
                    std::cmp::Ordering::Less => li.push((j, w[p])),
                    std::cmp::Ordering::Equal => ui[0].1 = w[p],
                    std::cmp::Ordering::Greater => ui.push((j, w[p])),
                }
                pos[j] = usize::MAX;
            }
            ui[0].1 = safe_pivot(ui[0].1, norm);
            l.push(li);
            u.push(ui);
        }
        Ok(Ilu { n, l, u })
    }
    // ANCHOR_END: ilu0

    // ANCHOR: ilut
    /// Saad's ILUT(τ, p): eliminate row by row with fill allowed, but drop every multiplier and
    /// every entry of the new row smaller than `τ ‖aᵢ‖₂`, and keep only the `p` largest entries of
    /// each of the `L` and `U` parts of the row (the diagonal is always kept).
    pub fn ilut(a: &CsrMatrix, drop_tol: f64, max_fill: usize) -> Result<Ilu> {
        let n = check_square(a)?;
        let mut l: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut u: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut w: BTreeMap<usize, f64> = BTreeMap::new();
            for p in a.indptr[i]..a.indptr[i + 1] {
                *w.entry(a.indices[p]).or_insert(0.0) += a.data[p];
            }
            let norm = w.values().map(|v| v * v).sum::<f64>().sqrt();
            let tau = drop_tol * norm;
            // Eliminate the entries left of the diagonal in increasing column order; fill created
            // by an update is always to the right of the column being eliminated.
            let mut next = 0;
            while let Some((&k, &wk)) = w.range(next..i).next() {
                next = k + 1;
                let lik = wk / u[k][0].1;
                if lik.abs() < tau {
                    w.remove(&k);
                    continue;
                }
                w.insert(k, lik);
                for &(j, ukj) in &u[k][1..] {
                    *w.entry(j).or_insert(0.0) -= lik * ukj;
                }
            }
            let diag = w.remove(&i).unwrap_or(0.0);
            let keep = |entries: Vec<(usize, f64)>| -> Vec<(usize, f64)> {
                let mut e: Vec<(usize, f64)> = entries.into_iter().filter(|(_, v)| v.abs() >= tau && *v != 0.0).collect();
                if e.len() > max_fill {
                    e.sort_by(|x, y| y.1.abs().total_cmp(&x.1.abs()));
                    e.truncate(max_fill);
                }
                e.sort_by_key(|x| x.0);
                e
            };
            let li = keep(w.range(..i).map(|(&j, &v)| (j, v)).collect());
            let mut ui = vec![(i, safe_pivot(diag, norm))];
            ui.extend(keep(w.range(i + 1..).map(|(&j, &v)| (j, v)).collect()));
            l.push(li);
            u.push(ui);
        }
        Ok(Ilu { n, l, u })
    }
    // ANCHOR_END: ilut

    /// Solves `L U x = b` by forward and back substitution.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let mut x = b.to_vec();
        for i in 0..self.n {
            let s: f64 = self.l[i].iter().map(|&(j, v)| v * x[j]).sum();
            x[i] -= s;
        }
        for i in (0..self.n).rev() {
            let s: f64 = self.u[i][1..].iter().map(|&(j, v)| v * x[j]).sum();
            x[i] = (x[i] - s) / self.u[i][0].1;
        }
        x
    }

    /// Solves `(L U)ᵀ x = b`.
    pub fn solve_transpose(&self, b: &[f64]) -> Vec<f64> {
        let mut x = b.to_vec();
        for i in 0..self.n {
            x[i] /= self.u[i][0].1;
            let xi = x[i];
            for &(j, v) in &self.u[i][1..] {
                x[j] -= v * xi;
            }
        }
        for i in (0..self.n).rev() {
            let xi = x[i];
            for &(j, v) in &self.l[i] {
                x[j] -= v * xi;
            }
        }
        x
    }

    /// The stored entries of `L` and `U` (the unit diagonal of `L` not counted).
    pub fn nnz(&self) -> usize {
        self.l.iter().map(Vec::len).sum::<usize>() + self.u.iter().map(Vec::len).sum::<usize>()
    }
}

/// As a preconditioner, `M = (L U)⁻¹`.
impl LinearOperator for Ilu {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        self.solve(x)
    }
    fn rmatvec(&self, x: &[f64]) -> Vec<f64> {
        self.solve_transpose(x)
    }
}
