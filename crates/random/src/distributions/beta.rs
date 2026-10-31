use crate::{distributions::gamma::gamma_distribution, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// Generate a Beta(alpha, beta) random variable
pub fn beta_distribution<U>(uniform_generator: &mut U, alpha: f64, beta: f64) -> Result<f64,ErrorsJSL> where U : UniformGenerator{
    let x = gamma_distribution(uniform_generator, alpha, 1.0)?;
    let y = gamma_distribution(uniform_generator, beta, 1.0)?;
    Ok(x / (x+y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn test_beta_sample_mean_matches_alpha_over_alpha_plus_beta() {
        let mut rng = UniformRNG::from_seed(2);
        let n = 100_000;
        let mean = (0..n)
            .map(|_| beta_distribution(&mut rng, 2.0, 5.0).unwrap())
            .sum::<f64>()
            / n as f64;
        assert!((mean - 2.0 / 7.0).abs() < 0.005, "mean={mean}");
    }
}
