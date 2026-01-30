use crate::{
    integration::{calculate_midpoints::calculate_midpoints, can_integrate::CanIntegrate1D},
    prelude::ErrorsJSL,
};

// An struct the integrates a 1D real function f(x) using the trapezoidal rule
// Note: This is pedagogical and does not converge quickly
pub struct AdaptiveTrapezoidIntegrationEngine {
    tolerance: f64,   // Where to bound error on each subinterval
    lower_bound: f64, // The lower bound of x
    upper_bound: f64, // The upper obund of x
}

impl AdaptiveTrapezoidIntegrationEngine {
    // Construct from bounds and number of points
    // The boolean flag is if you want to cache the x domain points to save speed on repeated calls (true) or recompute the x points each time (false)
    pub fn new(tolerance: f64, lower_bound: f64, upper_bound: f64) -> Self {
        AdaptiveTrapezoidIntegrationEngine {
            tolerance,
            lower_bound,
            upper_bound,
        }
    }
    // Dynamically sets the number of points.  Will update cache if configured.
    pub fn set_tolerance(&mut self, x: f64) {
        self.tolerance = x;
    }
}

impl CanIntegrate1D for AdaptiveTrapezoidIntegrationEngine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        // Evaluate bounds
        let a = self.lower_bound;
        let f_a = f(a);
        let b = self.upper_bound;
        let f_b = f(b);
        // Evaluate the midpoint
        let c = 0.5 * (a + b);
        let f_c = f(c);
        // Endpoint trapezoidal rule
        let trap2 = (f_a + f_b) * 0.5 * (b - a);
        // Include the middle trapezoidal rule
        let left = (f_a + f_c) * 0.5 * (c - a);
        let right = (f_c + f_b) * 0.5 * (b - c);
        let trap3 = left + right;
        let difference = (trap3 - trap2).abs();
        if difference < self.tolerance {
            return Ok(trap3);
        } else {
            let left = recurse(a, f_a, c, f_c, &f, self.tolerance);
            let right = recurse(c, f_c, b, f_b, &f, self.tolerance);
            Ok(left + right)
        }
    }

    // Dynamically sets the lower bound.  Will update cache if configured.
    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
    }
    // Dynamically sets the upper bound.  Will update cache if configured.
    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
    }
}

fn recurse<F>(a: f64, f_a: f64, b: f64, f_b: f64, f: &F, tolerance: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let c = 0.5 * (a + b);
    let f_c = f(c);
    let trap2 = (f_a + f_b) * 0.5 * (b - a);
    let left = (f_a + f_c) * 0.5 * (c - a);
    let right = (f_c + f_b) * 0.5 * (b - c);
    let trap3 = left + right;
    let difference = (trap3 - trap2).abs();
    if difference.abs() < tolerance {
        trap3
    } else {
        let left = recurse(a, f_a, c, f_c, f, tolerance);
        let right = recurse(c, f_c, b, f_b, f, tolerance);
        left + right
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use super::*;
    #[test]
    fn test_adaptive_integration() {
        let error_limit = 1E-6;
        let upper_bound = 0.0;
        let lower_bound = PI / 2.0;
        let dut = AdaptiveTrapezoidIntegrationEngine::new(error_limit*0.001, lower_bound, upper_bound);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = -1.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);

        let upper_bound = PI / 2.0;
        let lower_bound = -PI / 2.0;
        let dut = AdaptiveTrapezoidIntegrationEngine::new(error_limit*0.001, lower_bound, upper_bound);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = 0.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);
    }
}
