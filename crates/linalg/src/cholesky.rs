//! Cholesky factorisation `A = L Lᴴ` of Hermitian positive-definite matrices, and triangular
//! solves.

use crate::dense::{Mat, require_square};
use crate::error::{Diag, LinalgError, Result, UPLO};
use crate::scalar::Scalar;

/// The lower Cholesky factor of the Hermitian matrix whose `uplo` triangle is stored in `a`.
pub(crate) fn cholesky_lower<T: Scalar>(a: &Mat<T>, uplo: UPLO) -> Result<Mat<T>> {
    require_square(a.rows, a.cols)?;
    let n = a.rows;
    // Read the requested triangle as the lower triangle of a Hermitian matrix.
    let get = |i: usize, j: usize| -> T { if uplo == UPLO::Lower { a.at(i, j) } else { a.at(j, i).conj() } };
    let mut l = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let mut d = get(j, j).re();
        for k in 0..j {
            d -= l.at(j, k).abs_sqr();
        }
        if d.is_nan() || d <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite(j));
        }
        let ljj = d.sqrt();
        *l.at_mut(j, j) = T::from_real(ljj);
        for i in j + 1..n {
            let mut s = get(i, j);
            for k in 0..j {
                s -= l.at(i, k) * l.at(j, k).conj();
            }
            *l.at_mut(i, j) = s.scale(1.0 / ljj);
        }
    }
    Ok(l)
}

/// Solves `T X = B` for triangular `T` (only its `uplo` triangle is read) and `B` with any
/// number of columns, by forward or back substitution.
pub(crate) fn solve_triangular<T: Scalar>(t: &Mat<T>, uplo: UPLO, diag: Diag, b: &Mat<T>) -> Result<Mat<T>> {
    require_square(t.rows, t.cols)?;
    let n = t.rows;
    if b.rows != n {
        return Err(LinalgError::Shape(format!("triangular matrix {n} × {n} and right-hand side {} × {}", b.rows, b.cols)));
    }
    let mut x = b.clone();
    let m = b.cols;
    let order: Vec<usize> = if uplo == UPLO::Lower { (0..n).collect() } else { (0..n).rev().collect() };
    for &i in &order {
        let range: Vec<usize> = if uplo == UPLO::Lower { (0..i).collect() } else { (i + 1..n).collect() };
        for c in 0..m {
            let mut s = x.at(i, c);
            for &k in &range {
                s -= t.at(i, k) * x.at(k, c);
            }
            if diag == Diag::NonUnit {
                let d = t.at(i, i);
                if d == T::zero() {
                    return Err(LinalgError::Singular(i));
                }
                s /= d;
            }
            *x.at_mut(i, c) = s;
        }
    }
    Ok(x)
}
