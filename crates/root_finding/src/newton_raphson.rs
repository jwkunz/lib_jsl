//! Newton-Raphson method.
//!
//! See the book entry [Newton's method and the secant method](https://jwkunz.github.io/lib_jsl/book/root_finding/newton.html).

/// Finds a root of `f` using Newton-Raphson iteration with derivative `df`, starting at `x0`.
///
/// Stops when a step `|f(x)/f'(x)|` is below `tol` (returning the updated iterate) or `f(x)` is
/// exactly zero. Errors on a zero or non-finite derivative, divergence, or 100 iterations without
/// convergence. Converges quadratically near a simple root, linearly near a repeated one.
pub fn newton_raphson<F, D>(f: F, df: D, x0: f64, tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let mut x = x0;
    // ANCHOR: newton
    for _ in 0..100 {
        let fx = f(x);
        if fx == 0.0 {
            return Ok(x);
        }
        let d = df(x);
        if d == 0.0 || !d.is_finite() {
            return Err("newton_raphson: derivative is zero or non-finite".to_string());
        }
        let step = fx / d;
        x -= step;
        if !x.is_finite() {
            return Err("newton_raphson: iterate diverged".to_string());
        }
        if step.abs() < tol {
            return Ok(x);
        }
    }
    // ANCHOR_END: newton
    Err("newton_raphson: failed to converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt2() {
        let r = newton_raphson(|x| x * x - 2.0, |x| 2.0 * x, 1.0, 1e-14).unwrap();
        assert!((r - 2f64.sqrt()).abs() < 1e-13);
    }

    #[test]
    fn zero_derivative_errors() {
        assert!(newton_raphson(|x| x * x + 1.0, |x| 2.0 * x, 0.0, 1e-9).is_err());
    }
}
