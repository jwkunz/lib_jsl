//! The composite Simpson's 1/3 rule.
//!
//! See the book entry [Newton-Cotes rules](https://jwkunz.github.io/lib_jsl/book/integration/newton-cotes.html).

use crate::{calculate_midpoints::grid_points, can_integrate::CanIntegrate1D};
use lib_jsl_core::ErrorsJSL;

/// The composite Simpson's 1/3 rule on `n` equally spaced points (`n − 1` panels of width
/// `h = (b − a)/(n − 1)`): weights `h/3 · (1, 4, 2, 4, …, 2, 4, 1)`.
///
/// Fourth order (error `−(b − a) h⁴ f⁽⁴⁾(ξ) / 180`) and exact for cubics. The 1/3 rule needs an
/// even number of panels; with an odd number, the last three panels use Simpson's 3/8 rule so the
/// result stays fourth order. Two points fall back to the trapezoid rule.
pub struct Simpson13Engine {
    number_of_points: usize,
    lower_bound: f64,
    upper_bound: f64,
    /// The sample points and spacing, when caching is enabled.
    cache: Option<(Vec<f64>, f64)>,
}

impl Simpson13Engine {
    /// An engine using `number_of_points` samples on `[lower_bound, upper_bound]`. With
    /// `cache_x_points`, the sample points are computed once and reused by every call to
    /// `integrate`.
    pub fn new(number_of_points: usize, lower_bound: f64, upper_bound: f64, cache_x_points: bool) -> Self {
        let cache = cache_x_points.then(|| grid_points(lower_bound, upper_bound, number_of_points));
        Simpson13Engine { number_of_points, lower_bound, upper_bound, cache }
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

impl CanIntegrate1D for Simpson13Engine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        // ANCHOR: simpson_rule
        let points = self.points();
        let (x_points, dx) = &*points;
        let dx = *dx;
        if x_points.len() < 2 {
            return Err(ErrorsJSL::InvalidInputRange("Simpson's rule needs at least two points"));
        }
        let n = x_points.len();
        let y: Vec<f64> = x_points.iter().map(|&x| f(x)).collect();
        if n == 2 {
            return Ok(0.5 * dx * (y[0] + y[1]));
        }
        // Composite Simpson 1/3 needs an even number of panels (odd number of points): weights
        // 1, 4, 2, 4, …, 2, 4, 1. With an odd number of panels, the last three panels use
        // Simpson's 3/8 rule (weights 1, 3, 3, 1) so the result stays fourth order.
        let simpson_points = if n % 2 == 1 { n } else { n - 3 };
        let mut total = 0.0;
        if simpson_points >= 3 {
            let interior: f64 = (1..simpson_points - 1).map(|i| if i % 2 == 1 { 4.0 } else { 2.0 } * y[i]).sum();
            total += (y[0] + interior + y[simpson_points - 1]) * dx / 3.0;
        }
        if n % 2 == 0 {
            let k = n - 4;
            total += 3.0 * dx / 8.0 * (y[k] + 3.0 * y[k + 1] + 3.0 * y[k + 2] + y[k + 3]);
        }
        Ok(total)
        // ANCHOR_END: simpson_rule
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
    use std::f64::consts::PI;

    use super::*;
    #[test]
    fn test_simpsons_integration() {
        // Exact for cubics with any number of points (odd counts use 1/3, even add a 3/8 tail).
        let cubic = |x: f64| 2.0 * x * x * x - x * x + 3.0;
        let exact = 0.5 * 16.0 - 8.0 / 3.0 + 6.0; // ∫₀² of the cubic
        for n in [3, 4, 5, 6, 7, 10, 11] {
            let v = Simpson13Engine::new(n, 0.0, 2.0, false).integrate(cubic).unwrap();
            assert!((v - exact).abs() < 1e-12, "n={n}: {v}");
        }
        // Fourth-order convergence: halving h divides the error by about 16.
        let error = |n: usize| (Simpson13Engine::new(n, 0.0, PI, false).integrate(f64::sin).unwrap() - 2.0).abs();
        let ratio = error(17) / error(33);
        assert!(ratio > 15.0 && ratio < 17.0, "{ratio}");
        // Reversed bounds give the negated integral.
        let reversed = Simpson13Engine::new(101, PI / 2.0, 0.0, false).integrate(f64::sin).unwrap();
        assert!((reversed + 1.0).abs() < 1e-8);
        let two = Simpson13Engine::new(2, 0.0, 1.0, true).integrate(|x| x).unwrap();
        assert!((two - 0.5).abs() < 1e-15);
    }
}
