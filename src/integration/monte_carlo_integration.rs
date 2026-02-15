use std::collections::VecDeque;

use crate::{prelude::ErrorsJSL, random::uniform_generator::{DefaultUniformRNG, UniformGenerator}};

// Integration of a function f(x0,x1,...) via random sampling
// f is a function pointer of a slice (real multi-variate input) -> real scalar
// bounds is a slice containing tuples of (lower,upper) bounds for each variable
// convergence_tuple contains the (threshold,window_size) for computing the variance to determine termination criteria
// max_samples describes a stop point if Some() otherwise integrate until convergence
pub fn monte_carlo_integration<F>(
    f: F,
    bounds: &[(f64, f64)],
    convergence_tuple: (f64, usize),
    max_samples: Option<usize>,
) -> Result<f64, ErrorsJSL>
where
    F: Fn(&[f64]) -> f64,
{
    let mut rng = DefaultUniformRNG::from_seed(0);
    let scale: f64 = bounds.iter().map(|(low, high)| high - low).product();
    let convergence_window = convergence_tuple.1;
    let convergence_threshold = convergence_tuple.0;
    let mut values = VecDeque::<f64>::with_capacity(convergence_window);
    let mut accumulator: f64 = 0.0;
    let mut sum_of_values = 0.0;
    let mut values_squared = VecDeque::<f64>::with_capacity(convergence_window);
    let mut sum_of_values_squared = 0.0;
    let mut sample_count: usize = 0;
    let expected_scale = 1.0 / convergence_window as f64;
    loop {
        let sample_x = bounds.iter()
            .map(|(low, high)| rng.next_f64() * (high - low) + low)
            .collect::<Vec<f64>>();
        sample_count += 1;

        let sample_y = f(&sample_x);
        accumulator += sample_y;
        let estimate = scale * (accumulator / sample_count as f64);

        values.push_front(estimate);
        sum_of_values += estimate;
        let estimate_squared = estimate * estimate;
        values_squared.push_front(estimate_squared);
        sum_of_values_squared += estimate_squared;

        if sample_count > convergence_window {
            let last_value = values.pop_back().expect("");
            sum_of_values -= last_value;
            let last_value_squared = values_squared.pop_back().expect("");
            sum_of_values_squared -= last_value_squared;
            let expected = sum_of_values * expected_scale;
            let expected_squared = sum_of_values_squared * expected_scale;
            let variance = expected_squared - (expected * expected);
            if variance.sqrt() < convergence_threshold {
                return Ok(expected);
            }

            if let Some(max) = max_samples {
                if sample_count > max {
                    return Ok(expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use super::*;
    fn test_function(x: &[f64]) -> f64 {
        x[0].sin()*x[1].sin()*x[2].sin()
    }
    #[test]
    fn test_monte_carlos_integration() {
        let error_limit = 1E-2;
        let lower_bound = 0.0;
        let upper_bound = PI;
        let result = monte_carlo_integration(
            &test_function,
            &[(lower_bound, upper_bound),(lower_bound,upper_bound),(lower_bound,upper_bound)],
            (error_limit*1E-3, 1000),
            None,
        )
        .unwrap();
        let error = 8.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);
    }
}
