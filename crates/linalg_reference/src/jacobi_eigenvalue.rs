//! The Jacobi eigenvalue method for symmetric matrices (Jacobi, 1846).
//!
//! A plane rotation `J` in the `(p, q)` plane can be chosen so that `Jᵀ A J` has a zero in position
//! `(p, q)`. Later rotations fill that zero in again, but the sum of squares of the off-diagonal
//! entries falls by `2 a_pq²` every time, so sweeping over all pairs repeatedly drives `A` to a
//! diagonal matrix of eigenvalues; the product of the rotations holds the eigenvectors. The cyclic
//! method converges quadratically once the off-diagonal part is small. It is slower than
//! tridiagonalisation followed by QL (what `lib_jsl_linalg` does) but simple, highly accurate, and
//! the basis of the one-sided Jacobi SVD.

use crate::Error;
use ndarray::{Array1, Array2};

/// Eigenvalues in ascending order, eigenvectors as columns, and the number of sweeps.
#[derive(Debug, Clone)]
pub struct JacobiEigen {
    pub values: Array1<f64>,
    pub vectors: Array2<f64>,
    pub sweeps: usize,
}

/// Cyclic Jacobi: sweeps over all pairs `p < q` until the off-diagonal norm is below
/// `tolerance × ‖A‖_F`.
pub fn jacobi_eigen(a: &Array2<f64>, tolerance: f64) -> Result<JacobiEigen, Error> {
    let n = a.nrows();
    let mut a = a.clone();
    let mut v = crate::identity(n);
    let scale = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut sweeps = 0;
    while crate::off_diagonal_norm(&a) > tolerance * scale {
        if sweeps == 100 {
            return Err("the Jacobi method did not converge");
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[[p, q]] == 0.0 {
                    continue;
                }
                // tan θ is the smaller root of t² + 2 t cot 2θ − 1 = 0, keeping |θ| ≤ π/4.
                let cot = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
                let t = cot.signum() / (cot.abs() + (cot * cot + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                rotate_columns(&mut a, p, q, c, s);
                rotate_rows(&mut a, p, q, c, s);
                rotate_columns(&mut v, p, q, c, s);
            }
        }
        sweeps += 1;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[[i, i]].total_cmp(&a[[j, j]]));
    Ok(JacobiEigen {
        values: order.iter().map(|&i| a[[i, i]]).collect(),
        vectors: Array2::from_shape_fn((n, n), |(i, j)| v[[i, order[j]]]),
        sweeps,
    })
}

/// `X ← X J` for the rotation with `J_pp = J_qq = c`, `J_pq = s`, `J_qp = −s`.
fn rotate_columns(x: &mut Array2<f64>, p: usize, q: usize, c: f64, s: f64) {
    for k in 0..x.nrows() {
        let (xp, xq) = (x[[k, p]], x[[k, q]]);
        x[[k, p]] = c * xp - s * xq;
        x[[k, q]] = s * xp + c * xq;
    }
}

/// `X ← Jᵀ X`.
fn rotate_rows(x: &mut Array2<f64>, p: usize, q: usize, c: f64, s: f64) {
    for k in 0..x.ncols() {
        let (xp, xq) = (x[[p, k]], x[[q, k]]);
        x[[p, k]] = c * xp - s * xq;
        x[[q, k]] = s * xp + c * xq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_linalg::{EigValsh, UPLO};

    #[test]
    fn matches_the_production_eigensolver() {
        let a = crate::random_symmetric(20, 7);
        let e = jacobi_eigen(&a, 1e-15).unwrap();
        let reference = a.eigvalsh(UPLO::Lower).unwrap();
        assert!((&e.values - &reference).iter().all(|d| d.abs() < 1e-12));
        let residual = a.dot(&e.vectors) - &e.vectors * &e.values;
        assert!(residual.iter().all(|r| r.abs() < 1e-12));
        assert!(e.sweeps < 12, "quadratic convergence: {} sweeps", e.sweeps);
    }
}
