use crate::{
    integration::{calculate_midpoints::calculate_midpoints, can_integrate::CanIntegrate1D},
    prelude::ErrorsJSL,
};

// An struct the integrates a 1D real function f(x) using the Simpson's 1/3 rule
// Note: This is pedagogical and does not converge quickly
pub struct Simpson13Engine {
    number_of_points: usize,        // The number of points in to sample
    lower_bound: f64,               // The lower bound of x
    upper_bound: f64,               // The upper obund of x
    cache: Option<(Vec<f64>, f64)>, // Cache x domain samples points for speed on repeated calls
}

impl Simpson13Engine {
    // Construct from bounds and number of points
    // The boolean flag is if you want to cache the x domain points to save speed on repeated calls (true) or recompute the x points each time (false)
    pub fn new(
        number_of_points: usize,
        lower_bound: f64,
        upper_bound: f64,
        cache_x_points: bool,
    ) -> Self {
        let cache = if cache_x_points {
            Some(calculate_midpoints(
                lower_bound,
                upper_bound,
                number_of_points,
            ))
        } else {
            None
        };
        Simpson13Engine {
            number_of_points,
            lower_bound,
            upper_bound,
            cache,
        }
    }
    // Dynamically sets the number of points.  Will update cache if configured.
    pub fn set_number_of_points(&mut self, x: usize) {
        self.number_of_points = x;
        self.cache = if self.cache.is_some() {
            Some(calculate_midpoints(self.lower_bound, self.upper_bound, x))
        } else {
            None
        };
    }
}

impl CanIntegrate1D for Simpson13Engine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        // Get cached points or compute from scratch
        let (x_points, dx) = if let Some(x) = &self.cache {
            x
        } else {
            &calculate_midpoints(self.lower_bound, self.upper_bound, self.number_of_points)
        };
        if x_points.len() < 2 {
            return Err(ErrorsJSL::InvalidInputRange(
                "Must have more than two points for integration",
            ));
        }
        // Simple integration and scaling
        let middle = x_points
            .iter()
            .enumerate()
            .skip(1)
            .take(x_points.len() - 2)
            .map(|(n, &x)| if n & 0b1 == 0 { 4.0 * f(x) } else { 2.0 * f(x) })
            .sum::<f64>();
        let first = f(*x_points.first().expect(""));
        let last = f(*x_points.last().expect(""));
        Ok((middle + first + last) * dx / 3.0)
    }

    // Dynamically sets the lower bound.  Will update cache if configured.
    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
        self.cache = if self.cache.is_some() {
            Some(calculate_midpoints(
                self.lower_bound,
                self.upper_bound,
                self.number_of_points,
            ))
        } else {
            None
        };
    }
    // Dynamically sets the upper bound.  Will update cache if configured.
    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
        self.cache = if self.cache.is_some() {
            Some(calculate_midpoints(
                self.lower_bound,
                self.upper_bound,
                self.number_of_points,
            ))
        } else {
            None
        };
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use super::*;
    #[test]
    fn test_simpsons_integration() {
        let error_limit = 1E-3;
        let upper_bound = 0.0;
        let lower_bound = PI / 2.0;
        let number_of_points = 10000;
        let dut = Simpson13Engine::new(number_of_points, lower_bound, upper_bound, false);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = -1.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);

        let upper_bound = PI / 2.0;
        let lower_bound = -PI / 2.0;
        let number_of_points = 10000;
        let dut = Simpson13Engine::new(number_of_points, lower_bound, upper_bound, false);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = 0.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);
    }
}
