//! The singular value decomposition computed the obvious way, from the eigenproblem of `AᵀA`.
//!
//! If `A = U Σ Vᵀ` then `AᵀA = V Σ² Vᵀ`, so the right singular vectors are eigenvectors of `AᵀA`,
//! the singular values are the square roots of its eigenvalues, and `u_i = A v_i / σ_i`. The trouble
//! is that forming `AᵀA` squares every singular value: those below `√ε σ_max ≈ 10⁻⁸ σ_max` fall
//! beneath the rounding error of the largest and are lost. Production SVDs (one-sided Jacobi in
//! `lib_jsl_linalg`, Golub-Kahan bidiagonalisation in LAPACK) work on `A` itself.

use crate::Error;
use ndarray::{Array1, Array2};

/// The thin SVD of an `m × n` matrix (`m ≥ n`): `U` is `m × n`, `σ` descends, `V` is `n × n`.
/// Columns of `U` belonging to zero singular values are left zero.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Array2<f64>,
    pub singular_values: Array1<f64>,
    pub v: Array2<f64>,
}

/// The SVD via the Jacobi eigenvalue method applied to `AᵀA`.
pub fn svd_via_normal_matrix(a: &Array2<f64>) -> Result<Svd, Error> {
    let (m, n) = a.dim();
    let eigen = crate::jacobi_eigenvalue::jacobi_eigen(&a.t().dot(a), 1e-15)?;
    // Jacobi orders eigenvalues ascending; singular values are conventionally descending.
    let order: Vec<usize> = (0..n).rev().collect();
    let singular_values: Array1<f64> = order.iter().map(|&i| eigen.values[i].max(0.0).sqrt()).collect();
    let v = Array2::from_shape_fn((n, n), |(i, j)| eigen.vectors[[i, order[j]]]);
    let av = a.dot(&v);
    let u = Array2::from_shape_fn((m, n), |(i, j)| {
        if singular_values[j] > 0.0 { av[[i, j]] / singular_values[j] } else { 0.0 }
    });
    Ok(Svd { u, singular_values, v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_linalg::SVD;

    #[test]
    fn reconstructs_a_well_conditioned_matrix() {
        let a = crate::random_matrix(10, 6, 9);
        let svd = svd_via_normal_matrix(&a).unwrap();
        let rebuilt = (&svd.u * &svd.singular_values).dot(&svd.v.t());
        assert!((rebuilt - &a).iter().all(|d| d.abs() < 1e-12));
        let (_, reference, _) = a.svd(false, false).unwrap();
        assert!((&svd.singular_values - &reference).iter().all(|d| d.abs() < 1e-12));
    }

    #[test]
    fn small_singular_values_are_lost() {
        let sigma = [1.0, 1e-3, 1e-6, 1e-10];
        let (u, _) = crate::qr::householder(&crate::random_matrix(8, 4, 10));
        let (v, _) = crate::qr::householder(&crate::random_matrix(4, 4, 11));
        let a = (&u * &Array1::from(sigma.to_vec())).dot(&v.t());
        let naive = svd_via_normal_matrix(&a).unwrap().singular_values;
        let (_, accurate, _) = a.svd(false, false).unwrap();
        let relative = |s: &Array1<f64>| (s[3] - sigma[3]).abs() / sigma[3];
        assert!(relative(&naive) > 1e-2, "naive σ₄ = {:e}", naive[3]);
        assert!(relative(&accurate) < 1e-4, "production σ₄ = {:e}", accurate[3]);
    }
}
