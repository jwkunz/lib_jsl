//! Linear least squares, `min ‖A x − b‖₂` for a tall `A`, two ways.
//!
//! The **normal equations** `AᵀA x = Aᵀb` come straight from setting the gradient to zero and are
//! solved cheaply with Cholesky, but forming `AᵀA` squares the condition number. **QR** writes
//! `A = Q R`, so `‖A x − b‖ = ‖R x − Qᵀb‖` in the column space, and solves the triangular system
//! `R x = Qᵀb` directly, with error proportional to `κ(A)` rather than `κ(A)²`.

use crate::Error;
use crate::gaussian_elimination::back_substitute;
use ndarray::{Array1, Array2};

/// Solves the normal equations `AᵀA x = Aᵀb` with Cholesky: `L y = Aᵀb`, then `Lᵀ x = y`.
pub fn normal_equations(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let l = crate::cholesky::cholesky(&a.t().dot(a))?;
    let rhs = a.t().dot(b);
    let n = l.nrows();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        y[i] = (rhs[i] - (0..i).map(|j| l[[i, j]] * y[j]).sum::<f64>()) / l[[i, i]];
    }
    back_substitute(&l.t().to_owned(), &y)
}

/// Solves `R x = Qᵀb` with the Householder QR factorisation.
pub fn qr_least_squares(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let (q, r) = crate::qr::householder(a);
    back_substitute(&r, &q.t().dot(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_survives_an_ill_conditioned_polynomial_fit() {
        // Fit a degree-11 polynomial with all coefficients one on 40 points in [0, 1];
        // the Vandermonde matrix has condition number ≈ 10⁸.
        let a = Array2::from_shape_fn((40, 12), |(i, j)| (i as f64 / 39.0).powi(j as i32));
        let exact = Array1::from_elem(12, 1.0);
        let b = a.dot(&exact);
        let error = |x: &Array1<f64>| (x - &exact).iter().fold(0.0f64, |m, d| m.max(d.abs()));
        let qr = error(&qr_least_squares(&a, &b).unwrap());
        assert!(qr < 1e-6, "QR error {qr:e}");
        // Cholesky may fail outright (rounding makes AᵀA numerically indefinite), an even clearer
        // failure; when it succeeds, the answer is far less accurate.
        if let Ok(x) = normal_equations(&a, &b) {
            assert!(error(&x) > 1e3 * qr, "normal equations error {:e} vs QR {qr:e}", error(&x));
        }
    }

    #[test]
    fn both_agree_on_a_well_conditioned_problem() {
        let a = crate::random_matrix(30, 5, 12);
        let b = Array1::from_iter((0..30).map(|i| (i as f64 * 0.3).cos()));
        let (x1, x2) = (normal_equations(&a, &b).unwrap(), qr_least_squares(&a, &b).unwrap());
        assert!((&x1 - &x2).iter().all(|d| d.abs() < 1e-12));
    }
}
