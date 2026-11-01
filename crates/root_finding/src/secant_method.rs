//! Secant method.
//!
//! See the book entry [Newton's method and the secant method](https://jwkunz.github.io/lib_jsl/book/root_finding/newton.html#the-secant-method).

/// Finds a root of `f` with the secant method starting from `x0` and `x1`.
///
/// Stops when `f` is exactly zero, or successive iterates differ by less than `tol` and either `f`
/// changes sign between them or their own secant slope places the root within `tol`. Errors on a
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
        // ANCHOR: secant
        let mut x2 = x1 - f1 * (x1 - x0) / (f1 - f0);
        if !x2.is_finite() {
            return Err("secant_method: iterate diverged".to_string());
        }
        let small = (x2 - x1).abs() < tol;
        if x2 == x1 {
            // A zero step gives no local slope to check it with: probe a little way off.
            let h = (0.5 * tol).max(4.0 * f64::EPSILON * x1.abs()).max(f64::MIN_POSITIVE);
            x2 = if x0 > x1 { x1 + h } else { x1 - h };
        }
        let f2 = f(x2);
        // A small step is convergence only if f changes sign across it, or the slope between these
        // two nearby points (not the secant through a distant earlier point, which can make the
        // step tiny anywhere) puts the root within `tol`.
        if f2 == 0.0 || (small && (f2.signum() != f1.signum() || (f2 * (x2 - x1)).abs() <= (tol * (f2 - f1)).abs())) {
            return Ok(x2);
        }
        // ANCHOR_END: secant
        (x0, f0, x1, f1) = (x1, f1, x2, f2);
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
    fn a_tiny_step_after_a_wild_one_is_not_convergence() {
        // From 0 and 1.3, x¹⁰ − 1 throws an iterate to 2.2e6; the secant through it gives a step of
        // 1e-10 at x = 0.18, which was once returned as a root.
        let r = secant_method(|x| x.powi(10) - 1.0, 0.0, 1.3, 1e-13);
        assert!(r.as_ref().map_or(true, |x| (x.abs() - 1.0).abs() < 1e-12), "{r:?}");
    }

    #[test]
    fn flat_function_errors() {
        assert!(secant_method(|_| 1.0, 0.0, 1.0, 1e-9).is_err());
    }
}
