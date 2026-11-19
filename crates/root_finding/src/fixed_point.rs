//! Fixed points `x = g(x)` by iteration accelerated with Steffensen's (Aitken's Δ²) method
//! (SciPy's `optimize.fixed_point`).
//!
//! See the book entry [Halley's method and fixed points](https://jwkunz.github.io/lib_jsl/book/root_finding/halley-fixed-point.html).

use lib_jsl_core::ErrorsJSL;

/// How [`fixed_point`] steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FixedPointMethod {
    /// Steffensen's method: Aitken's Δ² extrapolation of two plain steps (SciPy's `"del2"`).
    Del2,
    /// Plain iteration `x ← g(x)`.
    Iteration,
}

// ANCHOR: fixed_point
/// Finds a fixed point of `g` from `x0` (SciPy's `fixed_point`). With [`FixedPointMethod::Del2`]
/// each step takes `p₁ = g(p₀)`, `p₂ = g(p₁)` and extrapolates
/// `p = p₀ − (p₁ − p₀)²/(p₂ − 2p₁ + p₀)` (or `p₂` when the denominator vanishes), which converges
/// quadratically even where plain iteration converges linearly — or not at all. Stops when the
/// relative change `|p/p₀ − 1|` (`|p|` at `p₀ = 0`) is below `xtol` (SciPy: 10⁻⁸); errors after
/// `maxiter` steps (SciPy: 500).
pub fn fixed_point<G>(g: G, x0: f64, xtol: f64, maxiter: usize, method: FixedPointMethod) -> Result<f64, ErrorsJSL>
where
    G: Fn(f64) -> f64,
{
    let mut p0 = x0;
    for _ in 0..maxiter {
        let p1 = g(p0);
        let p = match method {
            FixedPointMethod::Iteration => p1,
            FixedPointMethod::Del2 => {
                let p2 = g(p1);
                let d = p2 - 2.0 * p1 + p0;
                if d != 0.0 { p0 - (p1 - p0) * (p1 - p0) / d } else { p2 }
            }
        };
        if !p.is_finite() {
            return Err(ErrorsJSL::RuntimeError("fixed_point: iterate is not finite"));
        }
        let relerr = if p0 != 0.0 { (p / p0 - 1.0).abs() } else { p.abs() };
        if relerr < xtol {
            return Ok(p);
        }
        p0 = p;
    }
    Err(ErrorsJSL::RuntimeError("fixed_point: failed to converge"))
}
// ANCHOR_END: fixed_point

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steffensen_versus_iteration() {
        // SciPy's documentation example: √(c₁/(x + c₂)) with c₁ = 10, c₂ = 2 → 2.2.
        let g = |x: f64| (10.0 / (x + 2.0)).sqrt();
        let a = fixed_point(g, 1.2, 1e-8, 500, FixedPointMethod::Del2).unwrap();
        let b = fixed_point(g, 1.2, 1e-12, 500, FixedPointMethod::Iteration).unwrap();
        // x²(x + 2) = 10.
        for x in [a, b] {
            assert!((x * x * (x + 2.0) - 10.0).abs() < 1e-9);
        }
        // x = 3.5 x (1 − x) has a repelling fixed point at 5/7 that only Steffensen reaches.
        let g = |x: f64| 3.5 * x * (1.0 - x);
        let s = fixed_point(g, 0.7, 1e-12, 500, FixedPointMethod::Del2).unwrap();
        assert!((s - 5.0 / 7.0).abs() < 1e-12);
        assert!(fixed_point(g, 0.7, 1e-12, 500, FixedPointMethod::Iteration).is_err());
    }
}
