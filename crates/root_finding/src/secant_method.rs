//! Secant method.

/// Finds a root of `f` with the secant method starting from `x0` and `x1`.
///
/// Stops when successive iterates differ by less than `tol` or `f` is exactly zero. Errors on a
/// zero secant slope, divergence, or 100 iterations without convergence. The order of convergence
/// near a simple root is the golden ratio, 1.618.
pub fn secant_method<F>(f: F, x0: f64, x1: f64, tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    let (mut x0, mut x1) = (x0, x1);
    let (mut f0, mut f1) = (f(x0), f(x1));
    for _ in 0..100 {
        if f1 == 0.0 {
            return Ok(x1);
        }
        if f1 == f0 {
            return Err("secant_method: zero slope between iterates".to_string());
        }
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        if !x2.is_finite() {
            return Err("secant_method: iterate diverged".to_string());
        }
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
        if (x1 - x0).abs() < tol {
            return Ok(x1);
        }
    }
    Err("secant_method: failed to converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cos_root() {
        let r = secant_method(|x| x.cos() - x, 0.0, 1.0, 1e-12).unwrap();
        assert!((r - 0.7390851332151607).abs() < 1e-10);
    }

    #[test]
    fn flat_function_errors() {
        assert!(secant_method(|_| 1.0, 0.0, 1.0, 1e-9).is_err());
    }
}
