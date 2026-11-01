//! Power iteration and its two refinements, for one eigenpair of a symmetric matrix.
//!
//! Multiplying a vector repeatedly by `A` amplifies its component along the eigenvector of the
//! largest-magnitude eigenvalue, at the rate `|λ₂ / λ₁|` per step. **Inverse iteration** applies
//! the same idea to `(A − σI)⁻¹`, whose dominant eigenvalue `1 / (λ − σ)` belongs to the eigenvalue
//! nearest the shift `σ`; the closer the shift, the faster the convergence. **Rayleigh quotient
//! iteration** updates the shift to the current estimate `σ = vᵀAv` at every step, which converges
//! cubically for symmetric matrices, at the price of a new factorisation per step.
//!
//! See the book entry [Eigenvalue algorithms](https://jwkunz.github.io/lib_jsl/book/linear_algebra/eigen-iterations.html#power-iteration-inverse-iteration-and-the-rayleigh-quotient).

use crate::Error;
use ndarray::{Array1, Array2};

/// An eigenvalue estimate, its unit eigenvector and the iterations taken.
#[derive(Debug, Clone)]
pub struct EigenEstimate {
    pub value: f64,
    pub vector: Array1<f64>,
    pub iterations: usize,
}

/// Iterates `v ← f(v) / ‖f(v)‖` until the eigen-residual `‖A v − (vᵀAv) v‖` is below `tolerance`.
fn iterate(
    a: &Array2<f64>,
    start: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    mut step: impl FnMut(&Array1<f64>, f64) -> Result<Option<Array1<f64>>, Error>,
) -> Result<EigenEstimate, Error> {
    let mut v = start / start.dot(start).sqrt();
    let mut value = v.dot(&a.dot(&v));
    for iterations in 1..=max_iterations {
        let Some(w) = step(&v, value)? else {
            // The shifted matrix is exactly singular: the shift is an eigenvalue.
            return Ok(EigenEstimate { value, vector: v, iterations });
        };
        v = &w / w.dot(&w).sqrt();
        let av = a.dot(&v);
        value = v.dot(&av);
        let residual = &av - &(&v * value);
        if residual.dot(&residual).sqrt() < tolerance {
            return Ok(EigenEstimate { value, vector: v, iterations });
        }
    }
    Err("the iteration did not converge")
}

/// Power iteration: `v ← A v`. Converges to the eigenvalue of largest magnitude.
pub fn power_iteration(a: &Array2<f64>, start: &Array1<f64>, tolerance: f64, max_iterations: usize) -> Result<EigenEstimate, Error> {
    iterate(a, start, tolerance, max_iterations, |v, _| Ok(Some(a.dot(v))))
}

/// Inverse iteration with a fixed shift: `v ← (A − σI)⁻¹ v`, factorising once.
pub fn inverse_iteration(
    a: &Array2<f64>,
    shift: f64,
    start: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<EigenEstimate, Error> {
    let factors = crate::lu::lu(&(a - &(crate::identity(a.nrows()) * shift)))?;
    iterate(a, start, tolerance, max_iterations, |v, _| factors.solve(v).map(Some))
}

// ANCHOR: rqi
/// Rayleigh quotient iteration: `v ← (A − σ_k I)⁻¹ v` with `σ_k = vᵀAv`, refactorising each step.
pub fn rayleigh_quotient_iteration(
    a: &Array2<f64>,
    start: &Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<EigenEstimate, Error> {
    let n = a.nrows();
    iterate(a, start, tolerance, max_iterations, |v, shift| {
        match crate::lu::lu(&(a - &(crate::identity(n) * shift))) {
            Ok(factors) => factors.solve(v).map(Some),
            Err(_) => Ok(None),
        }
    })
}
// ANCHOR_END: rqi

#[cfg(test)]
mod tests {
    use super::*;

    /// `Q diag(1, 2, …, n) Qᵀ` with a random orthogonal `Q`.
    fn known_spectrum(n: usize) -> Array2<f64> {
        let (q, _) = crate::qr::householder(&crate::random_matrix(n, n, 5));
        let d = Array2::from_shape_fn((n, n), |(i, j)| if i == j { (i + 1) as f64 } else { 0.0 });
        q.dot(&d).dot(&q.t())
    }

    #[test]
    fn shifts_buy_speed() {
        let a = known_spectrum(10);
        let start = Array1::from_iter(crate::random_matrix(10, 1, 6).iter().copied());
        let power = power_iteration(&a, &start, 1e-10, 10_000).unwrap();
        let inverse = inverse_iteration(&a, 3.2, &start, 1e-10, 10_000).unwrap();
        let rayleigh = rayleigh_quotient_iteration(&a, &inverse_iteration(&a, 3.2, &start, 1e-1, 100).unwrap().vector, 1e-10, 100).unwrap();
        assert!((power.value - 10.0).abs() < 1e-9);
        assert!((inverse.value - 3.0).abs() < 1e-9);
        assert!((rayleigh.value - 3.0).abs() < 1e-9);
        // Linear rate 0.9 versus 0.25 versus cubic.
        assert!(power.iterations > 100, "{}", power.iterations);
        assert!(inverse.iterations < 30, "{}", inverse.iterations);
        assert!(rayleigh.iterations <= 3, "{}", rayleigh.iterations);
    }
}
