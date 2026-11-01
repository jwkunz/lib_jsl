//! The composite midpoint rule.
//!
//! See the book entry [Newton-Cotes rules](https://jwkunz.github.io/lib_jsl/book/integration/newton-cotes.html).

use crate::{calculate_midpoints::panel_midpoints, can_integrate::CanIntegrate1D};
use lib_jsl_core::ErrorsJSL;

/// The composite midpoint rule: `∫ₐᵇ f ≈ h Σ f(a + (i + ½) h)` over `n` panels of width
/// `h = (b − a)/n`, sampling `f` at the centre of each panel.
///
/// Second order (the rule minus the integral is `−(b − a) h² f''(ξ) / 24`: half the trapezoid
/// rule's error, with the opposite sign), exact for straight lines, and *open*: it never evaluates `f` at the endpoints, so it
/// tolerates integrable endpoint singularities (slowly).
pub struct MidpointIntegrationEngine {
    number_of_points: usize,
    lower_bound: f64,
    upper_bound: f64,
    /// The sample points and spacing, when caching is enabled.
    cache: Option<(Vec<f64>, f64)>,
}

impl MidpointIntegrationEngine {
    /// An engine using `number_of_points` samples on `[lower_bound, upper_bound]` (the number of panels). With
    /// `cache_x_points`, the sample points are computed once and reused by every call to
    /// `integrate`.
    pub fn new(number_of_points: usize, lower_bound: f64, upper_bound: f64, cache_x_points: bool) -> Self {
        let cache = cache_x_points.then(|| panel_midpoints(lower_bound, upper_bound, number_of_points));
        MidpointIntegrationEngine { number_of_points, lower_bound, upper_bound, cache }
    }

    /// Changes the number of samples (refreshing the cache, if enabled).
    pub fn set_number_of_points(&mut self, x: usize) {
        self.number_of_points = x;
        self.refresh();
    }

    fn refresh(&mut self) {
        if self.cache.is_some() {
            self.cache = Some(panel_midpoints(self.lower_bound, self.upper_bound, self.number_of_points));
        }
    }

    fn points(&self) -> std::borrow::Cow<'_, (Vec<f64>, f64)> {
        match &self.cache {
            Some(c) => std::borrow::Cow::Borrowed(c),
            None => std::borrow::Cow::Owned(panel_midpoints(self.lower_bound, self.upper_bound, self.number_of_points)),
        }
    }
}

impl CanIntegrate1D for MidpointIntegrationEngine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        // ANCHOR: midpoint_rule
        let points = self.points();
        let (x, h) = &*points;
        if x.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("the midpoint rule needs at least one panel"));
        }
        Ok(h * x.iter().map(|&x| f(x)).sum::<f64>())
        // ANCHOR_END: midpoint_rule
    }

    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
        self.refresh();
    }

    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
        self.refresh();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_midpoint_integration() {
        let dut = MidpointIntegrationEngine::new(1000, 0.0, PI / 2.0, false);
        assert!((dut.integrate(f64::sin).unwrap() - 1.0).abs() < 1e-6);
        let dut = MidpointIntegrationEngine::new(1000, PI / 2.0, 0.0, true);
        assert!((dut.integrate(f64::sin).unwrap() + 1.0).abs() < 1e-6);
        // Exact for straight lines with a single panel.
        let one = MidpointIntegrationEngine::new(1, 1.0, 3.0, false);
        assert!((one.integrate(|x| 3.0 * x + 1.0).unwrap() - 14.0).abs() < 1e-14);
        assert!(MidpointIntegrationEngine::new(0, 0.0, 1.0, false).integrate(f64::sin).is_err());
    }

    #[test]
    fn midpoint_error_falls_by_four_when_h_halves() {
        let err = |n| (MidpointIntegrationEngine::new(n, 0.0, 1.0, false).integrate(f64::exp).unwrap() - (std::f64::consts::E - 1.0)).abs();
        let ratio = err(64) / err(128);
        assert!((ratio - 4.0).abs() < 0.01, "{ratio}");
    }
}
