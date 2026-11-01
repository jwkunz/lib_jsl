//! The Cholesky factorisation `A = L Lᵀ` of a symmetric positive-definite matrix.
//!
//! Comparing entries of `A = L Lᵀ` row by row (the Cholesky-Banachiewicz order) gives each entry
//! of `L` from those already known:
//!
//! `l_jj = √(a_jj − Σ_{k<j} l_jk²)`, `l_ij = (a_ij − Σ_{k<j} l_ik l_jk) / l_jj` for `i > j`.
//!
//! It costs half as much as LU, needs no pivoting (the diagonal of a positive-definite matrix
//! dominates), and doubles as a test for positive definiteness: the square root fails exactly
//! when `A` is not positive definite.

use crate::Error;
use ndarray::Array2;

/// The lower Cholesky factor `L`.
pub fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>, Error> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[[i, k]] * l[[j, k]]).sum();
            if i == j {
                let d = a[[i, i]] - sum;
                if d <= 0.0 {
                    return Err("the matrix is not positive definite");
                }
                l[[i, i]] = d.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn reconstructs_and_detects_indefinite_matrices() {
        let a = crate::random_positive_definite(15, 3);
        let l = cholesky(&a).unwrap();
        assert!((l.dot(&l.t()) - &a).iter().all(|v| v.abs() < 1e-10));
        assert!(l.indexed_iter().all(|((i, j), v)| j <= i || *v == 0.0));
        assert!(cholesky(&array![[1.0, 2.0], [2.0, 1.0]]).is_err());
    }
}
