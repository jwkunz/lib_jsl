//! Adaptive trapezoid quadrature.
//!
//! See the book entry [Newton-Cotes rules](https://jwkunz.github.io/lib_jsl/book/integration/newton-cotes.html#adaptive-trapezoid-rule).

use crate::can_integrate::CanIntegrate1D;
use lib_jsl_core::ErrorsJSL;

/// Adaptive trapezoid quadrature: each interval is compared with its two halves and split again
/// wherever they disagree, so samples concentrate where the integrand changes quickly.
///
/// An interval `[a, b]` with midpoint `c` is accepted when the one-panel and two-panel trapezoid
/// values differ by less than its share of the tolerance; that difference is three times the
/// error of the two-panel value, so the estimate is conservative. The tolerance is halved at each
/// split, so the total error is bounded by roughly `tolerance` rather than by `tolerance` per
/// interval. Recursion stops with an error after 50 levels (intervals `2⁻⁵⁰` of the original),
/// which catches non-finite or non-integrable integrands.
pub struct AdaptiveTrapezoidIntegrationEngine {
    tolerance: f64,
    lower_bound: f64,
    upper_bound: f64,
}

/// The deepest subdivision allowed.
const MAX_DEPTH: u32 = 50;

impl AdaptiveTrapezoidIntegrationEngine {
    /// An engine on `[lower_bound, upper_bound]` aiming for an absolute error below `tolerance`.
    pub fn new(tolerance: f64, lower_bound: f64, upper_bound: f64) -> Self {
        AdaptiveTrapezoidIntegrationEngine { tolerance, lower_bound, upper_bound }
    }

    /// Changes the error tolerance.
    pub fn set_tolerance(&mut self, x: f64) {
        self.tolerance = x;
    }
}

impl CanIntegrate1D for AdaptiveTrapezoidIntegrationEngine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        let (a, b) = (self.lower_bound, self.upper_bound);
        recurse(a, f(a), b, f(b), &f, self.tolerance, 0)
    }

    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
    }

    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
    }
}

// ANCHOR: adaptive_trapezoid
fn recurse<F>(a: f64, f_a: f64, b: f64, f_b: f64, f: &F, tolerance: f64, depth: u32) -> Result<f64, ErrorsJSL>
where
    F: Fn(f64) -> f64,
{
    let c = 0.5 * (a + b);
    let f_c = f(c);
    let one_panel = 0.5 * (b - a) * (f_a + f_b);
    let two_panels = 0.25 * (b - a) * (f_a + 2.0 * f_c + f_b);
    let difference = (two_panels - one_panel).abs();
    if difference < tolerance {
        return Ok(two_panels);
    }
    if depth == MAX_DEPTH || !difference.is_finite() {
        return Err(ErrorsJSL::RuntimeError("adaptive trapezoid: the integrand is not finite or the tolerance cannot be met"));
    }
    let left = recurse(a, f_a, c, f_c, f, 0.5 * tolerance, depth + 1)?;
    let right = recurse(c, f_c, b, f_b, f, 0.5 * tolerance, depth + 1)?;
    Ok(left + right)
}
// ANCHOR_END: adaptive_trapezoid

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use super::*;
    #[test]
    fn test_trapezoid_adaptive_integration() {
        let error_limit = 1E-6;
        let upper_bound = 0.0;
        let lower_bound = PI / 2.0;
        let dut =
            AdaptiveTrapezoidIntegrationEngine::new(error_limit * 0.001, lower_bound, upper_bound);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = -1.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);

        let upper_bound = PI / 2.0;
        let lower_bound = -PI / 2.0;
        let dut =
            AdaptiveTrapezoidIntegrationEngine::new(error_limit * 0.001, lower_bound, upper_bound);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = 0.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);
    }

    #[test]
    fn concentrates_samples_where_needed_and_rejects_singularities() {
        // A narrow peak: the adaptive rule meets the tolerance on the whole interval.
        let f = |x: f64| 1.0 / (1e-4 + (x - 0.3) * (x - 0.3));
        let exact = 100.0 * ((0.7f64 / 0.01).atan() + (0.3f64 / 0.01).atan());
        let dut = AdaptiveTrapezoidIntegrationEngine::new(1e-6, 0.0, 1.0);
        assert!((dut.integrate(f).unwrap() - exact).abs() < 1e-5);
        // An infinite sample used to recurse until the stack overflowed.
        let dut = AdaptiveTrapezoidIntegrationEngine::new(1e-8, 0.0, 1.0);
        assert!(dut.integrate(|x| 1.0 / x.sqrt()).is_err());
    }
}
