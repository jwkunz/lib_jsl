//! The composite trapezoid rule.
//!
//! See the book entry [Newton-Cotes rules](https://jwkunz.github.io/lib_jsl/book/integration/newton-cotes.html).

use crate::{calculate_midpoints::grid_points, can_integrate::CanIntegrate1D};
use lib_jsl_core::ErrorsJSL;

/// The composite trapezoid rule: `∫ₐᵇ f ≈ h [½ f(x₀) + f(x₁) + ⋯ + f(xₙ₋₂) + ½ f(xₙ₋₁)]` on
/// `n` equally spaced points `xᵢ = a + i h`, `h = (b − a)/(n − 1)`.
///
/// Second order (error `(b − a) h² f''(ξ) / 12`) and exact for straight lines. For smooth
/// *periodic* integrands over a whole period it converges exponentially fast.
pub struct TrapezoidalIntegrationEngine {
    number_of_points: usize,
    lower_bound: f64,
    upper_bound: f64,
    /// The sample points and spacing, when caching is enabled.
    cache: Option<(Vec<f64>, f64)>,
}

impl TrapezoidalIntegrationEngine {
    /// An engine using `number_of_points` samples on `[lower_bound, upper_bound]`. With
    /// `cache_x_points`, the sample points are computed once and reused by every call to
    /// `integrate`.
    pub fn new(number_of_points: usize, lower_bound: f64, upper_bound: f64, cache_x_points: bool) -> Self {
        let cache = cache_x_points.then(|| grid_points(lower_bound, upper_bound, number_of_points));
        TrapezoidalIntegrationEngine { number_of_points, lower_bound, upper_bound, cache }
    }

    /// Changes the number of samples (refreshing the cache, if enabled).
    pub fn set_number_of_points(&mut self, x: usize) {
        self.number_of_points = x;
        self.refresh();
    }

    fn refresh(&mut self) {
        if self.cache.is_some() {
            self.cache = Some(grid_points(self.lower_bound, self.upper_bound, self.number_of_points));
        }
    }

    fn points(&self) -> std::borrow::Cow<'_, (Vec<f64>, f64)> {
        match &self.cache {
            Some(c) => std::borrow::Cow::Borrowed(c),
            None => std::borrow::Cow::Owned(grid_points(self.lower_bound, self.upper_bound, self.number_of_points)),
        }
    }
}

impl CanIntegrate1D for TrapezoidalIntegrationEngine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        // ANCHOR: trapezoid_rule
        let points = self.points();
        let (x, h) = &*points;
        if x.len() < 2 {
            return Err(ErrorsJSL::InvalidInputRange("the trapezoid rule needs at least two points"));
        }
        let interior: f64 = x[1..x.len() - 1].iter().map(|&x| f(x)).sum();
        Ok(h * (interior + 0.5 * (f(x[0]) + f(x[x.len() - 1]))))
        // ANCHOR_END: trapezoid_rule
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
mod test{  
    use std::f64::consts::PI;

    use super::*;
    #[test]
    fn test_trapezoid_integration(){
        let error_limit = 1E-3;
        let upper_bound = 0.0;
        let lower_bound = PI/2.0;
        let number_of_points = 10000;
        let dut = TrapezoidalIntegrationEngine::new(number_of_points, lower_bound, upper_bound,false);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = -1.0-result;
        dbg!(&error);
        assert!(error.abs() < error_limit);

        let upper_bound = PI/2.0;
        let lower_bound = -PI/2.0;
        let number_of_points = 10000;
        let dut = TrapezoidalIntegrationEngine::new(number_of_points, lower_bound, upper_bound,false);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = 0.0-result;
        dbg!(&error);
        assert!(error.abs() < error_limit);
    }
}
