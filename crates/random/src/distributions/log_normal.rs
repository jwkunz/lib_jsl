use crate::{distributions::gaussian::gaussian_distribution_box_muller, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// A log-normal sample `e^X` with `X ~ N(mu, sigma²)`: mean `e^{μ + σ²/2}`, median `e^μ`.
pub fn log_normal_distribution<U: UniformGenerator>(uniform_generator: &mut U, mu: f64, sigma: f64) -> Result<f64, ErrorsJSL> {
    if sigma.is_nan() || sigma < 0.0 || !mu.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("sigma must be non-negative and mu finite"));
    }
    Ok(gaussian_distribution_box_muller(uniform_generator, mu, sigma).0.exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn moments_and_median_match_theory() {
        let mut rng = UniformRNG::from_seed(11);
        let (mu, sigma) = (0.5, 0.4);
        let mut samples: Vec<f64> = (0..200_000).map(|_| log_normal_distribution(&mut rng, mu, sigma).unwrap()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - (mu + sigma * sigma / 2.0f64).exp()).abs() < 0.01, "{mean}");
        samples.sort_by(f64::total_cmp);
        assert!((samples[samples.len() / 2] - mu.exp()).abs() < 0.01);
        assert!(samples[0] > 0.0);
        assert!(log_normal_distribution(&mut rng, 0.0, -1.0).is_err());
    }
}
