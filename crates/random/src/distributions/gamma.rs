use crate::{distributions::gaussian::gaussian_distribution_box_muller, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// A gamma sample with shape `alpha` and **scale** `beta` (mean `αβ`, variance `αβ²`), by the
/// Marsaglia-Tsang method (2000): a squeeze-accelerated rejection from a transformed normal, for
/// `α ≥ 1`; for `α < 1`, `Gamma(α) = Gamma(α + 1) · U^{1/α}`.
pub fn gamma_distribution<U>(uniform_generator: &mut U, alpha: f64, beta: f64) -> Result<f64,ErrorsJSL> where U : UniformGenerator{
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("alpha and beta must be positive"));
    }

    if alpha < 1.0 {
        // Boosting identity for alpha < 1
        let u = uniform_generator.next_f64();
        return Ok(gamma_distribution(uniform_generator,alpha + 1.0, 1.0).expect("Already sanitized") * u.powf(1.0 / alpha) * beta);
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let z = gaussian_distribution_box_muller(uniform_generator, 0.0, 1.0).0;
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue; // Reject
        }
        let v_cubed = v * v * v;
        let u = uniform_generator.next_f64();
        if u.ln() < 0.5 * z * z + d - d * v_cubed + d * v_cubed.ln() {
            return Ok(d * v_cubed * beta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn test_gamma_rejects_non_positive_parameters() {
        let mut rng = UniformRNG::from_seed(0);
        assert!(gamma_distribution(&mut rng, 0.0, 1.0).is_err());
        assert!(gamma_distribution(&mut rng, 1.0, 0.0).is_err());
        assert!(gamma_distribution(&mut rng, -1.0, 1.0).is_err());
    }

    #[test]
    fn test_gamma_sample_mean_matches_alpha_times_scale() {
        let mut rng = UniformRNG::from_seed(1);
        let n = 100_000;
        for (alpha, scale) in [(2.0, 3.0), (0.5, 2.0)] {
            let mean = (0..n)
                .map(|_| gamma_distribution(&mut rng, alpha, scale).unwrap())
                .sum::<f64>()
                / n as f64;
            let expected = alpha * scale;
            assert!((mean - expected).abs() < 0.05 * expected, "mean={mean}, expected={expected}");
        }
    }
}
