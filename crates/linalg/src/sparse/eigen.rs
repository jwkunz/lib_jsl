//! A few eigenvalues of a large symmetric operator by the Lanczos process (SciPy's `eigsh`), and a
//! few singular values by Golub-Kahan bidiagonalisation (SciPy's `svds`), both with full
//! reorthogonalisation and restarts that enlarge the subspace until the wanted values converge.

use super::iterative::LinearOperator;
use crate::error::{LinalgError, Result};
use crate::{Eigh, SVD, UPLO};
use ndarray::Array2;

/// Which eigenvalues [`eigsh`] returns (SciPy's `which`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    /// Largest in magnitude (`'LM'`, SciPy's default).
    LargestMagnitude,
    /// Largest algebraic (`'LA'`).
    LargestAlgebraic,
    /// Smallest algebraic (`'SA'`).
    SmallestAlgebraic,
}

/// Eigenvalues (ascending) and eigenvectors (columns).
pub type EigenPairs = (Vec<f64>, Array2<f64>);

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// A deterministic start vector with all components non-zero, orthogonal to `locked`.
fn start_vector(n: usize, locked: &[Vec<f64>]) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * ((i as f64 * 1.618_033_988_75).fract() - 0.5)).collect();
    reorthogonalise(&mut v, locked, &[]);
    let s = norm(&v);
    v.iter_mut().for_each(|t| *t /= s);
    v
}

/// Orthogonalises `w` against the vectors in `locked` and `basis` together, twice (classical
/// Gram-Schmidt, repeated). Both sets must be in every pass: projecting against one set after the
/// other puts back components along the first, multiplied by the (large) Lanczos coefficients,
/// and the error grows geometrically from step to step.
fn reorthogonalise(w: &mut [f64], locked: &[Vec<f64>], basis: &[Vec<f64>]) {
    for _ in 0..2 {
        for q in locked.iter().chain(basis) {
            let c = dot(w, q);
            w.iter_mut().zip(q).for_each(|(wi, qi)| *wi -= c * qi);
        }
    }
}

/// How strongly `which` prefers the eigenvalue `theta` (larger is preferred).
fn preference(which: Which, theta: f64) -> f64 {
    match which {
        Which::LargestMagnitude => theta.abs(),
        Which::LargestAlgebraic => theta,
        Which::SmallestAlgebraic => -theta,
    }
}

/// Repeats `run` in the orthogonal complement of the vectors found so far, keeping the `k` most
/// preferred, until a run finds nothing that beats them. A single Krylov subspace holds only one
/// vector of each eigenspace, so a repeated eigenvalue appears once per run: deflation (locking)
/// finds the other copies.
fn with_deflation<T: Clone>(k: usize, limit: usize, key: impl Fn(&T) -> f64, locked: impl Fn(&[T]) -> usize, mut run: impl FnMut(&[T]) -> Result<Vec<T>>) -> Result<Vec<T>> {
    let mut found = run(&[])?;
    while locked(&found) < limit {
        let worst = found.iter().map(&key).fold(f64::INFINITY, f64::min);
        let more: Vec<T> = run(&found)?.into_iter().filter(|p| found.len() < k || key(p) > worst * (1.0 + 1e-12 * worst.signum()) + 1e-300).collect();
        if more.is_empty() {
            break;
        }
        found.extend(more);
        found.sort_by(|x, y| key(y).total_cmp(&key(x)));
        found.truncate(k);
    }
    Ok(found)
}

// ANCHOR: eigsh
/// `k` eigenvalues and eigenvectors of a symmetric operator (SciPy's `eigsh`). The Lanczos
/// process builds an orthonormal basis `Q` of the Krylov subspace in which `A` is tridiagonal,
/// `QᵀAQ = T`; the eigenvalues of `T` (Ritz values) approximate the extreme eigenvalues of `A`,
/// with residual norms `|β_m s_{m,i}|` known without forming the vectors. Full
/// reorthogonalisation keeps `Q` orthonormal in floating point; if the wanted Ritz values have not
/// converged to `tol` relative accuracy, the subspace is doubled.
pub fn eigsh(a: &dyn LinearOperator, k: usize, which: Which, tol: f64) -> Result<EigenPairs> {
    let (n, n2) = a.shape();
    if n != n2 || k == 0 || k > n {
        return Err(LinalgError::Shape(format!("need a square operator and 1 ≤ k ≤ n, got {n} × {n2}, k = {k}")));
    }
    let tol = if tol > 0.0 { tol } else { f64::EPSILON.sqrt() * 1e-2 };
    let mut pairs = with_deflation(k, n, |p: &(f64, Vec<f64>)| preference(which, p.0), |f| f.len(), |found| {
        let locked: Vec<Vec<f64>> = found.iter().map(|p| p.1.clone()).collect();
        lanczos(a, k, which, tol, &locked)
    })?;
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
    let values = pairs.iter().map(|p| p.0).collect();
    let vectors = Array2::from_shape_fn((n, pairs.len()), |(r, c)| pairs[c].1[r]);
    Ok((values, vectors))
}

/// One Lanczos run in the orthogonal complement of `locked`: the `k` wanted Ritz pairs, the
/// subspace doubled until their residuals `|β_m s_{m,i}|` are below `tol` times the largest Ritz
/// value.
fn lanczos(a: &dyn LinearOperator, k: usize, which: Which, tol: f64, locked: &[Vec<f64>]) -> Result<Vec<(f64, Vec<f64>)>> {
    let n = a.shape().0;
    let room = n - locked.len();
    if room == 0 {
        return Ok(Vec::new());
    }
    let mut m = (2 * k + 1).max(20).min(room);
    loop {
        let mut q = vec![start_vector(n, locked)];
        let (mut alpha, mut beta) = (Vec::with_capacity(m), Vec::with_capacity(m));
        let mut last_beta = 0.0;
        for j in 0..m {
            let mut w = a.matvec(&q[j]);
            let aj = dot(&w, &q[j]);
            alpha.push(aj);
            reorthogonalise(&mut w, locked, &q);
            let b = norm(&w);
            let breakdown = b <= f64::EPSILON * aj.abs().max(1.0);
            last_beta = if breakdown { 0.0 } else { b };
            if j + 1 == m || breakdown {
                break;
            }
            beta.push(b);
            q.push(w.iter().map(|t| t / b).collect());
        }
        let size = alpha.len();
        let t = Array2::from_shape_fn((size, size), |(i, j)| if i == j { alpha[i] } else if i + 1 == j || j + 1 == i { beta[i.min(j)] } else { 0.0 });
        let (theta, s) = t.eigh(UPLO::Lower)?;
        let mut idx: Vec<usize> = (0..size).collect();
        idx.sort_by(|&x, &y| preference(which, theta[y]).total_cmp(&preference(which, theta[x])));
        let wanted: Vec<usize> = idx.into_iter().take(k).collect();
        let scale = theta.iter().fold(0.0f64, |acc, v| acc.max(v.abs())).max(f64::MIN_POSITIVE);
        let converged = wanted.iter().all(|&i| (last_beta * s[(size - 1, i)]).abs() <= tol * scale);
        if converged || size == room || size < m {
            return Ok(wanted
                .iter()
                .map(|&i| {
                    let mut v = vec![0.0; n];
                    for (j, qj) in q.iter().enumerate() {
                        let c = s[(j, i)];
                        v.iter_mut().zip(qj).for_each(|(vi, qi)| *vi += c * qi);
                    }
                    (theta[i], v)
                })
                .collect());
        }
        m = (2 * m).min(room);
    }
}
// ANCHOR_END: eigsh

/// A singular triplet: value, left vector, right vector.
type Triplet = (f64, Vec<f64>, Vec<f64>);

/// The `k` largest singular values with their left and right singular vectors (columns),
/// descending (SciPy's `svds`, which returns them ascending). Golub-Kahan bidiagonalisation
/// `A V = U B` with full reorthogonalisation, the SVD of the small bidiagonal `B`, a doubled
/// subspace until the residuals are below `tol`, and deflation for repeated singular values.
pub fn svds(a: &dyn LinearOperator, k: usize, tol: f64) -> Result<(Vec<f64>, Array2<f64>, Array2<f64>)> {
    let (rows, cols) = a.shape();
    let p = rows.min(cols);
    if k == 0 || k > p {
        return Err(LinalgError::Shape(format!("need 1 ≤ k ≤ min(m, n) = {p}")));
    }
    let tol = if tol > 0.0 { tol } else { f64::EPSILON.sqrt() * 1e-2 };
    let found = with_deflation(k, p, |t: &Triplet| t.0, |f| f.len(), |found| {
        let lu: Vec<Vec<f64>> = found.iter().map(|t| t.1.clone()).collect();
        let lv: Vec<Vec<f64>> = found.iter().map(|t| t.2.clone()).collect();
        bidiagonalise(a, k, tol, &lu, &lv)
    })?;
    let left = Array2::from_shape_fn((rows, found.len()), |(r, c)| found[c].1[r]);
    let right = Array2::from_shape_fn((cols, found.len()), |(r, c)| found[c].2[r]);
    Ok((found.iter().map(|t| t.0).collect(), left, right))
}

/// One Golub-Kahan run with `u` kept orthogonal to `lu` and `v` to `lv`.
fn bidiagonalise(a: &dyn LinearOperator, k: usize, tol: f64, lu: &[Vec<f64>], lv: &[Vec<f64>]) -> Result<Vec<Triplet>> {
    let (rows, cols) = a.shape();
    let room = rows.min(cols) - lv.len();
    if room == 0 {
        return Ok(Vec::new());
    }
    let mut m = (2 * k + 1).max(20).min(room);
    loop {
        let mut v = vec![start_vector(cols, lv)];
        let mut u: Vec<Vec<f64>> = Vec::new();
        let (mut alpha, mut beta) = (Vec::new(), Vec::new());
        let mut last_beta = 0.0;
        for j in 0..m {
            let mut uj = a.matvec(&v[j]);
            if j > 0 {
                let b = beta[j - 1];
                uj.iter_mut().zip(&u[j - 1]).for_each(|(x, y)| *x -= b * y);
            }
            reorthogonalise(&mut uj, lu, &u);
            let aj = norm(&uj);
            if aj == 0.0 {
                break;
            }
            alpha.push(aj);
            u.push(uj.iter().map(|t| t / aj).collect());
            let mut vj = a.rmatvec(&u[j]);
            vj.iter_mut().zip(&v[j]).for_each(|(x, y)| *x -= aj * y);
            reorthogonalise(&mut vj, lv, &v);
            let bj = norm(&vj);
            last_beta = bj;
            if j + 1 == m || bj == 0.0 {
                break;
            }
            beta.push(bj);
            v.push(vj.iter().map(|t| t / bj).collect());
        }
        let size = alpha.len();
        if size == 0 {
            return Ok(Vec::new());
        }
        let b = Array2::from_shape_fn((size, size), |(i, j)| if i == j { alpha[i] } else if j == i + 1 { beta[i] } else { 0.0 });
        let (ub, sv, vbt) = b.svd(true, true)?;
        let (ub, vbt) = (ub.expect("requested"), vbt.expect("requested"));
        let smax = sv.iter().copied().fold(0.0, f64::max).max(f64::MIN_POSITIVE);
        let kk = k.min(size);
        // Residual of the i-th triplet: β_m |last component of the right singular vector of B|.
        let converged = (0..kk).all(|i| (last_beta * vbt[(i, size - 1)]).abs() <= tol * smax);
        if converged || size == room || size < m {
            return Ok((0..kk)
                .map(|c| {
                    let left = (0..rows).map(|r| (0..size).map(|j| u[j][r] * ub[(j, c)]).sum()).collect();
                    let right = (0..cols).map(|r| (0..size).map(|j| v[j][r] * vbt[(c, j)]).sum()).collect();
                    (sv[c], left, right)
                })
                .collect());
        }
        m = (2 * m).min(room);
    }
}
