//! Forward-difference gradients and gradient checking (SciPy's `optimize.approx_fprime` and
//! `optimize.check_grad`).
//!
//! See the book entry [Brute force, isotonic regression and gradient checks](https://jwkunz.github.io/lib_jsl/book/optimization/brute-isotonic.html).

// ANCHOR: approx_fprime
/// The forward-difference gradient `(f(x + hεᵢ) − f(x))/h` of `f` at `x` (SciPy's `approx_fprime`),
/// with step `epsilon` (SciPy's default is `√ε ≈ 1.49·10⁻⁸`); the step actually used is the
/// representable `(xᵢ + h) − xᵢ`. Its error is about `h|f''|/2 + 2ε|f|/h`, smallest near the default.
pub fn approx_fprime(f: impl Fn(&[f64]) -> f64, x: &[f64], epsilon: f64) -> Vec<f64> {
    let fx = f(x);
    let mut xh = x.to_vec();
    (0..x.len())
        .map(|i| {
            xh[i] = x[i] + epsilon;
            let dx = xh[i] - x[i];
            let g = (f(&xh) - fx) / dx;
            xh[i] = x[i];
            g
        })
        .collect()
}
// ANCHOR_END: approx_fprime

// ANCHOR: check_grad
/// The Euclidean norm of the difference between an analytic gradient `grad(x)` and the
/// forward-difference gradient of `f` (SciPy's `check_grad`): about `√ε` times the size of the
/// second derivatives when `grad` is right, of the order of the gradient itself when it is wrong.
pub fn check_grad(f: impl Fn(&[f64]) -> f64, grad: impl Fn(&[f64]) -> Vec<f64>, x: &[f64], epsilon: f64) -> f64 {
    let g = grad(x);
    let a = approx_fprime(f, x, epsilon);
    g.iter().zip(&a).map(|(u, v)| (u - v) * (u - v)).sum::<f64>().sqrt()
}
// ANCHOR_END: check_grad

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_gradient_checks() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let g = |x: &[f64]| vec![-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]), 200.0 * (x[1] - x[0] * x[0])];
        let h = f64::EPSILON.sqrt();
        let x = [-1.2, 1.0];
        assert!(check_grad(f, g, &x, h) < 1e-4);
        let wrong = |x: &[f64]| vec![g(x)[0], 0.0];
        assert!(check_grad(f, wrong, &x, h) > 50.0);
        let a = approx_fprime(f, &x, h);
        assert!((a[0] - g(&x)[0]).abs() < 1e-4 * g(&x)[0].abs());
    }
}
