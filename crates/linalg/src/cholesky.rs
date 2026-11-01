//! Cholesky factorisation `A = L Lᴴ` of Hermitian positive-definite matrices, and triangular
//! solves.

use crate::dense::{Mat, dot_conj, require_square, sub_matmul};

/// The block size of the Cholesky factorisation.
const BLOCK: usize = 16;
use crate::error::{Diag, LinalgError, Result, UPLO};
use crate::scalar::Scalar;

/// The lower Cholesky factor of the Hermitian matrix whose `uplo` triangle is stored in `a`.
///
/// Right-looking blocked algorithm: factor a diagonal block of order [`BLOCK`] with the unblocked
/// method, solve for the panel of `L` below it, then subtract the panel's outer product from the
/// trailing matrix with the matrix-multiply kernel, where almost all of the arithmetic happens.
pub(crate) fn cholesky_lower<T: Scalar>(a: &Mat<T>, uplo: UPLO) -> Result<Mat<T>> {
    require_square(a.rows, a.cols)?;
    let n = a.rows;
    // The lower triangle of the Hermitian matrix, whichever triangle was supplied.
    let mut l = Mat::<T>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            l.data[i * n + j] = if uplo == UPLO::Lower { a.at(i, j) } else { a.at(j, i).conj() };
        }
    }
    let mut panel_adjoint: Vec<T> = Vec::new();
    for k in (0..n).step_by(BLOCK) {
        let kb = BLOCK.min(n - k);
        factor_diagonal_block(&mut l, k, kb)?;
        let rest = n - k - kb;
        if rest == 0 {
            break;
        }
        // Panel: each row r of A₂₁ becomes x with x L₁₁ᴴ = r, by forward substitution.
        let (head, tail) = l.data.split_at_mut((k + kb) * n);
        for i in 0..rest {
            let row = &mut tail[i * n + k..i * n + k + kb];
            for j in 0..kb {
                let ljj = &head[(k + j) * n + k..(k + j) * n + k + j + 1];
                let s = row[j] - dot_conj(&row[..j], &ljj[..j]);
                row[j] = s.scale(1.0 / ljj[j].re());
            }
        }
        // Trailing update A₂₂ ← A₂₂ − L₂₁ L₂₁ᴴ, block row by block row, lower part only.
        // L₂₁ᴴ is copied out explicitly (the complex kernel cannot conjugate on the fly).
        panel_adjoint.clear();
        panel_adjoint.resize(kb * rest, T::zero());
        for i in 0..rest {
            for j in 0..kb {
                panel_adjoint[j * rest + i] = tail[i * n + k + j].conj();
            }
        }
        for i0 in (0..rest).step_by(BLOCK) {
            let ib = BLOCK.min(rest - i0);
            let ptr = tail.as_mut_ptr();
            // SAFETY: C (columns ≥ k + kb) is disjoint from A (columns k..k + kb) and from B,
            // which lives in a separate buffer; all three lie inside `tail` / `panel_adjoint`.
            unsafe {
                sub_matmul(
                    ib,
                    kb,
                    i0 + ib,
                    ptr.add(i0 * n + k),
                    n,
                    panel_adjoint.as_ptr(),
                    rest,
                    ptr.add(i0 * n + k + kb),
                    n,
                );
            }
        }
    }
    // The trailing updates also wrote above the diagonal of each diagonal block.
    for i in 0..n {
        l.data[i * n + i + 1..(i + 1) * n].fill(T::zero());
    }
    Ok(l)
}

/// The unblocked Cholesky-Banachiewicz factorisation of the diagonal block starting at `(k, k)`,
/// whose earlier columns have already been folded in by the trailing updates.
fn factor_diagonal_block<T: Scalar>(l: &mut Mat<T>, k: usize, kb: usize) -> Result<()> {
    let n = l.cols;
    for i in 0..kb {
        let (head, tail) = l.data.split_at_mut((k + i) * n);
        let row = &mut tail[k..k + i + 1];
        for j in 0..i {
            let lj = &head[(k + j) * n + k..(k + j) * n + k + j + 1];
            let s = row[j] - dot_conj(&row[..j], &lj[..j]);
            row[j] = s.scale(1.0 / lj[j].re());
        }
        let d = row[i].re() - dot_conj(&row[..i], &row[..i]).re();
        if d.is_nan() || d <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite(k + i));
        }
        row[i] = T::from_real(d.sqrt());
    }
    Ok(())
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
