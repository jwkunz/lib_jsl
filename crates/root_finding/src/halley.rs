//! Halley's method (SciPy's `newton` given `fprime2`), cubically convergent with the second
//! derivative.
//!
//! See the book entry [Halley's method and fixed points](https://jwkunz.github.io/lib_jsl/book/root_finding/halley-fixed-point.html).

use lib_jsl_core::ErrorsJSL;

// ANCHOR: halley
/// Finds a root of `f` by Halley's method from `x0`, as SciPy's `newton` with `fprime2`: the
/// Newton step `δ = f/f'` is divided by `1 − δ f''/(2f')` whenever that correction is below one
/// in magnitude. Stops when successive iterates differ by at most `tol` (SciPy's default is
/// 1.48·10⁻⁸) or `f(x)` is exactly zero; errors on a zero derivative or after 50 iterations.
/// Converges cubically near a simple root.
pub fn halley<F, D, D2>(f: F, df: D, d2f: D2, x0: f64, tol: f64) -> Result<f64, ErrorsJSL>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
    D2: Fn(f64) -> f64,
{
    let mut x = x0;
    for _ in 0..50 {
        let fx = f(x);
        if fx == 0.0 {
            return Ok(x);
        }
        let d = df(x);
        if d == 0.0 || !d.is_finite() {
            return Err(ErrorsJSL::RuntimeError("halley: derivative is zero or non-finite"));
        }
        let mut step = fx / d;
        let adj = step * d2f(x) / d / 2.0;
        if adj.abs() < 1.0 {
            step /= 1.0 - adj;
        }
        let next = x - step;
        if !next.is_finite() {
            return Err(ErrorsJSL::RuntimeError("halley: iterate diverged"));
        }
        if (next - x).abs() <= tol {
            return Ok(next);
        }
        x = next;
    }
    Err(ErrorsJSL::RuntimeError("halley: failed to converge in 50 iterations"))
}
// ANCHOR_END: halley

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_convergence() {
        // Count the iterations Halley needs for the cube root of 10, against Newton.
        let calls = std::cell::Cell::new(0);
        let f = |x: f64| {
            calls.set(calls.get() + 1);
            x * x * x - 10.0
        };
        let r = halley(f, |x| 3.0 * x * x, |x| 6.0 * x, 1.0, 1e-15).unwrap();
        assert!((r - 10f64.cbrt()).abs() < 1e-15);
        assert!(calls.get() <= 6, "{}", calls.get());
        assert!(halley(|x| x * x + 1.0, |x| 2.0 * x, |_| 2.0, 0.0, 1e-9).is_err());
    }
}
