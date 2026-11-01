//! Plain Monte Carlo integration over boxes.
//!
//! See the book entry [Monte Carlo and quasi-Monte Carlo integration](https://jwkunz.github.io/lib_jsl/book/integration/monte-carlo.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};

/// The outcome of [`monte_carlo`].
#[derive(Debug, Clone, Copy)]
pub struct MonteCarloResult {
    /// The estimated integral.
    pub value: f64,
    /// The estimated standard deviation of `value`, `V σ̂ / √N` (`V` the volume of the box).
    pub standard_error: f64,
    /// The number of integrand evaluations.
    pub evaluations: usize,
}

fn check_bounds(bounds: &[(f64, f64)]) -> Result<f64, ErrorsJSL> {
    if bounds.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("The box needs at least one dimension."));
    }
    if bounds.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("The bounds must be finite."));
    }
    Ok(bounds.iter().map(|(a, b)| b - a).product())
}

/// Running mean and variance of the samples (Welford's algorithm), numerically stable.
#[derive(Default)]
struct Welford {
    n: usize,
    mean: f64,
    m2: f64,
}

impl Welford {
    // ANCHOR: welford
    fn push(&mut self, y: f64) {
        self.n += 1;
        let delta = y - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (y - self.mean);
    }

    /// The standard error of the mean, `σ̂ / √n`.
    fn standard_error(&self) -> f64 {
        if self.n < 2 { f64::INFINITY } else { (self.m2 / (self.n - 1) as f64 / self.n as f64).sqrt() }
    }
    // ANCHOR_END: welford
}

/// Estimates `∫ f` over the box `bounds` (one `(lower, upper)` pair per dimension) from `samples`
/// uniformly distributed points: the volume times the sample mean of `f`, with its standard error.
/// The error falls like `1/√N` whatever the dimension. `seed` makes the result reproducible.
pub fn monte_carlo<F: Fn(&[f64]) -> f64>(f: F, bounds: &[(f64, f64)], samples: usize, seed: u64) -> Result<MonteCarloResult, ErrorsJSL> {
    let volume = check_bounds(bounds)?;
    if samples == 0 {
        return Err(ErrorsJSL::InvalidInputRange("At least one sample is needed."));
    }
    let mut rng = UniformRNG::from_seed(seed);
    let mut x = vec![0.0; bounds.len()];
    let mut stats = Welford::default();
    for _ in 0..samples {
        for (xi, (a, b)) in x.iter_mut().zip(bounds) {
            *xi = a + rng.next_f64() * (b - a);
        }
        stats.push(f(&x));
    }
    Ok(MonteCarloResult { value: volume * stats.mean, standard_error: volume * stats.standard_error(), evaluations: samples })
}

/// Monte Carlo integration until a target accuracy: samples are drawn in batches of
/// `convergence_tuple.1` until the standard error of the estimate falls below
/// `convergence_tuple.0` (an absolute error target), or until `max_samples` have been used, in
/// which case the current estimate is returned. Without `max_samples`, failing to converge within
/// 10⁹ samples is an error (the integrand may have infinite variance).
///
/// The standard error falls like `σ/√N`, so halving it costs four times the samples.
pub fn monte_carlo_integration<F>(f: F, bounds: &[(f64, f64)], convergence_tuple: (f64, usize), max_samples: Option<usize>) -> Result<f64, ErrorsJSL>
where
    F: Fn(&[f64]) -> f64,
{
    let volume = check_bounds(bounds)?;
    let (target, batch) = (convergence_tuple.0, convergence_tuple.1.max(2));
    let limit = max_samples.unwrap_or(1_000_000_000);
    let mut rng = UniformRNG::from_seed(0);
    let mut x = vec![0.0; bounds.len()];
    let mut stats = Welford::default();
    loop {
        for _ in 0..batch {
            for (xi, (a, b)) in x.iter_mut().zip(bounds) {
                *xi = a + rng.next_f64() * (b - a);
            }
            stats.push(f(&x));
        }
        if volume * stats.standard_error() < target {
            return Ok(volume * stats.mean);
        }
        if stats.n >= limit {
            return match max_samples {
                Some(_) => Ok(volume * stats.mean),
                None => Err(ErrorsJSL::RuntimeError("Monte Carlo: the target standard error was not reached in 10⁹ samples")),
            };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;

    fn test_function(x: &[f64]) -> f64 {
        x[0].sin() * x[1].sin() * x[2].sin()
    }

    #[test]
    fn test_monte_carlos_integration() {
        // ∫ over [0, π]³ of sin x sin y sin z = 8. Target a standard error of 0.02.
        let result = monte_carlo_integration(test_function, &[(0.0, PI); 3], (0.02, 1000), None).unwrap();
        assert!((8.0 - result).abs() < 0.1, "{result}");
    }

    #[test]
    fn the_standard_error_is_honest() {
        // Over many seeds, about 68% of estimates lie within one standard error of the truth.
        let inside = (0..200u64)
            .filter(|&seed| {
                let r = monte_carlo(test_function, &[(0.0, PI); 3], 2000, seed).unwrap();
                (r.value - 8.0).abs() < r.standard_error
            })
            .count();
        assert!((110..=160).contains(&inside), "{inside} of 200");
        // Quadrupling the samples halves the standard error.
        let a = monte_carlo(test_function, &[(0.0, PI); 3], 10_000, 1).unwrap();
        let b = monte_carlo(test_function, &[(0.0, PI); 3], 40_000, 1).unwrap();
        assert!((a.standard_error / b.standard_error - 2.0).abs() < 0.1);
    }

    #[test]
    fn validation_and_budgets() {
        assert!(monte_carlo(test_function, &[], 10, 0).is_err());
        assert!(monte_carlo(test_function, &[(0.0, f64::INFINITY)], 10, 0).is_err());
        assert!(monte_carlo(test_function, &[(0.0, 1.0)], 0, 0).is_err());
        // An unreachable target with a sample limit returns the estimate at the limit.
        let v = monte_carlo_integration(|x| x[0], &[(0.0, 1.0)], (1e-12, 100), Some(1000)).unwrap();
        assert!((v - 0.5).abs() < 0.05);
    }
}
