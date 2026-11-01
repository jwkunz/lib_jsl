//! Small complex linear-algebra helpers shared by the signal models and beamformers.

use crate::conventions::C64;
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};
use lib_jsl_linalg::{Eigh, UPLO};

/// `‖R v - λ v‖` summed over the eigenpairs.
fn residual(r: &Array2<C64>, vals: &[f64], vecs: &Array2<C64>) -> f64 {
    (0..vals.len()).map(|k| {
        let v = vecs.column(k);
        let rv = r.dot(&v);
        rv.iter().zip(v.iter()).map(|(a, b)| (a - b * vals[k]).norm()).sum::<f64>()
    }).sum()
}

/// Eigen-decomposition of a Hermitian matrix: eigenvalues in *descending* order and the matching
/// orthonormal eigenvectors as columns, verified against `R v = λ v`.
pub(crate) fn hermitian_eigen(r: &Array2<C64>) -> Result<(Vec<f64>, Array2<C64>), ErrorsJSL> {
    let n = r.nrows();
    if r.ncols() != n || n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Need a non-empty square matrix."));
    }
    let (vals, vecs) = r.eigh(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("Hermitian eigendecomposition failed."))?;
    let order: Vec<usize> = (0..n).rev().collect();
    let sorted: Vec<f64> = order.iter().map(|&i| vals[i]).collect();
    let vectors = Array2::from_shape_fn((n, n), |(i, c)| vecs[[i, order[c]]]);
    let scale = sorted.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    if residual(r, &sorted, &vectors) > 1e-8 * scale * n as f64 {
        return Err(ErrorsJSL::RuntimeError("The eigendecomposition did not satisfy R v = λ v."));
    }
    Ok((sorted, vectors))
}

/// The conjugate transpose.
pub(crate) fn hermitian_transpose(a: &Array2<C64>) -> Array2<C64> {
    a.t().mapv(|v| v.conj())
}

/// `wᴴ R w` (real part; exact for Hermitian `R`).
pub fn quadratic_form(w: &Array1<C64>, r: &Array2<C64>) -> f64 {
    w.iter().zip(r.dot(w).iter()).map(|(a, b)| (a.conj() * b).re).sum()
}

/// Checks that `r` is square, of size `n`, and Hermitian to a relative tolerance.
pub(crate) fn check_hermitian(r: &Array2<C64>, n: usize) -> Result<(), ErrorsJSL> {
    if r.dim() != (n, n) {
        return Err(ErrorsJSL::IncompatibleArraySizes((r.nrows(), n)));
    }
    let scale = r.iter().fold(0.0f64, |m, v| m.max(v.norm())).max(1e-300);
    for i in 0..n {
        for j in 0..n {
            if (r[[i, j]] - r[[j, i]].conj()).norm() > 1e-9 * scale {
                return Err(ErrorsJSL::InvalidInputRange("The matrix is not Hermitian."));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_pairs_are_correct_and_descending() {
        let r = ndarray::arr2(&[[C64::new(2.0, 0.0), C64::new(0.0, 1.0)], [C64::new(0.0, -1.0), C64::new(2.0, 0.0)]]);
        let (vals, vecs) = hermitian_eigen(&r).unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-12 && (vals[1] - 1.0).abs() < 1e-12);
        assert!(residual(&r, &vals, &vecs) < 1e-12);
        assert!((quadratic_form(&vecs.column(0).to_owned(), &r) - 3.0).abs() < 1e-12);
        assert!(check_hermitian(&r, 2).is_ok() && check_hermitian(&hermitian_transpose(&r).mapv(|v| v * C64::new(0.0, 1.0)), 2).is_err());
    }
}
