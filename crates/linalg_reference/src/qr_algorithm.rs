//! The QR algorithm (Francis and Kublanovskaya, 1961) for the eigenvalues of a symmetric matrix.
//!
//! Factor `A_k = Q_k R_k`, then multiply in the other order: `A_{k+1} = R_k Q_k = Q_kᵀ A_k Q_k`.
//! Every iterate is similar to `A`, and the product `Q_1 Q_2 ⋯ Q_k` is exactly what subspace
//! (orthogonal) iteration on `Aᵏ` would produce, so `A_k` converges to a diagonal matrix of
//! eigenvalues, ordered by magnitude, with entry `(i, j)` decaying like `|λ_i / λ_j|ᵏ`. Equal
//! magnitudes (`±λ`) never separate.
//!
//! The **shifted** algorithm factors `A_k − μ I` instead and adds `μ I` back. A shift close to an
//! eigenvalue makes the last row converge in a few steps; that eigenvalue is then **deflated** and
//! the iteration continues on the leading block. Wilkinson's shift, the eigenvalue of the trailing
//! 2 × 2 block nearer its corner entry, always converges for symmetric matrices. Production codes
//! add one more ingredient, a preliminary reduction to tridiagonal (or Hessenberg) form, so that
//! each step costs `O(n)` (or `O(n²)`) rather than `O(n³)`.
//!
//! See the book entry [Eigenvalue algorithms](https://jwkunz.github.io/lib_jsl/book/linear_algebra/eigen-iterations.html#the-qr-algorithm).

use crate::Error;
use crate::qr::householder;
use ndarray::{Array1, Array2, s};

/// Eigenvalues in ascending order and the number of QR steps taken.
#[derive(Debug, Clone)]
pub struct QrEigenvalues {
    pub values: Array1<f64>,
    pub iterations: usize,
}

fn sorted_diagonal(a: &Array2<f64>) -> Array1<f64> {
    let mut d = a.diag().to_vec();
    d.sort_by(f64::total_cmp);
    Array1::from(d)
}

fn frobenius(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Unshifted QR iteration until the off-diagonal norm is below `tolerance × ‖A‖_F`.
pub fn unshifted_qr(a: &Array2<f64>, tolerance: f64, max_iterations: usize) -> Result<QrEigenvalues, Error> {
    let scale = frobenius(a);
    let mut a = a.clone();
    for iterations in 0..=max_iterations {
        if crate::off_diagonal_norm(&a) <= tolerance * scale {
            return Ok(QrEigenvalues { values: sorted_diagonal(&a), iterations });
        }
        let (q, r) = householder(&a);
        a = r.dot(&q);
    }
    Err("unshifted QR did not converge")
}

// ANCHOR: shifted_qr
/// QR iteration with Wilkinson shifts and deflation.
pub fn shifted_qr(a: &Array2<f64>, tolerance: f64, max_iterations: usize) -> Result<QrEigenvalues, Error> {
    let scale = frobenius(a);
    let mut a = a.clone();
    let mut m = a.nrows();
    let mut iterations = 0;
    while m > 1 {
        let last_row = a.slice(s![m - 1, ..m - 1]);
        if last_row.dot(&last_row).sqrt() <= tolerance * scale {
            m -= 1; // a[m−1, m−1] is an eigenvalue; continue on the leading block.
            continue;
        }
        if iterations == max_iterations {
            return Err("shifted QR did not converge");
        }
        let mu = wilkinson_shift(a[[m - 2, m - 2]], a[[m - 1, m - 2]], a[[m - 1, m - 1]]);
        let shift = crate::identity(m) * mu;
        let (q, r) = householder(&(&a.slice(s![..m, ..m]) - &shift));
        a.slice_mut(s![..m, ..m]).assign(&(r.dot(&q) + shift));
        iterations += 1;
    }
    Ok(QrEigenvalues { values: sorted_diagonal(&a), iterations })
}
// ANCHOR_END: shifted_qr

/// The eigenvalue of `[[a, b], [b, c]]` closer to `c`.
pub fn wilkinson_shift(a: f64, b: f64, c: f64) -> f64 {
    let d = (a - c) / 2.0;
    let sign = if d >= 0.0 { 1.0 } else { -1.0 };
    c - sign * b * b / (d.abs() + (d * d + b * b).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_linalg::{EigValsh, UPLO};
    use ndarray::array;

    #[test]
    fn shifts_and_deflation_cut_the_work() {
        let a = crate::random_symmetric(12, 8);
        let reference = a.eigvalsh(UPLO::Lower).unwrap();
        let plain = unshifted_qr(&a, 1e-13, 100_000).unwrap();
        let shifted = shifted_qr(&a, 1e-15, 1_000).unwrap();
        assert!((&plain.values - &reference).iter().all(|d| d.abs() < 1e-10));
        assert!((&shifted.values - &reference).iter().all(|d| d.abs() < 1e-12));
        assert!(shifted.iterations < 4 * 12, "{} shifted steps", shifted.iterations);
        assert!(plain.iterations > 10 * shifted.iterations, "{} vs {}", plain.iterations, shifted.iterations);
    }

    #[test]
    fn equal_magnitudes_defeat_the_unshifted_algorithm() {
        // Eigenvalues ±1: R Q reproduces A exactly, forever.
        let a = array![[0.0, 1.0], [1.0, 0.0]];
        assert!(unshifted_qr(&a, 1e-12, 1_000).is_err());
        let shifted = shifted_qr(&a, 1e-15, 10).unwrap();
        assert!((&shifted.values - &array![-1.0, 1.0]).iter().all(|d| d.abs() < 1e-14), "{shifted:?}");
    }
}
