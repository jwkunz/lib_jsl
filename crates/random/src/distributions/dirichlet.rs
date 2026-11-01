use crate::{distributions::gamma::gamma_distribution, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// A Dirichlet sample on the probability simplex, `x_i = g_i / Σ g_j` with independent
/// `g_i ~ Gamma(α_i, 1)`. Its mean is `α_i / α₀` (`α₀ = Σ α_i`); two components give the beta
/// distribution.
pub fn dirichlet_distribution<U: UniformGenerator>(uniform_generator: &mut U, alpha: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    if alpha.len() < 2 || alpha.iter().any(|a| a.is_nan() || *a <= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Dirichlet needs at least two positive concentrations"));
    }
    let g: Vec<f64> = alpha.iter().map(|&a| gamma_distribution(uniform_generator, a, 1.0)).collect::<Result<_, _>>()?;
    let total: f64 = g.iter().sum();
    Ok(g.iter().map(|v| v / total).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn samples_lie_on_the_simplex_with_the_right_moments() {
        let mut rng = UniformRNG::from_seed(8);
        let alpha = [0.5, 2.0, 4.5];
        let a0: f64 = alpha.iter().sum();
        let m = 100_000;
        let samples: Vec<Vec<f64>> = (0..m).map(|_| dirichlet_distribution(&mut rng, &alpha).unwrap()).collect();
        assert!(samples.iter().all(|s| (s.iter().sum::<f64>() - 1.0).abs() < 1e-12 && s.iter().all(|&v| v >= 0.0)));
        for i in 0..3 {
            let mean = samples.iter().map(|s| s[i]).sum::<f64>() / m as f64;
            let var = samples.iter().map(|s| (s[i] - mean).powi(2)).sum::<f64>() / m as f64;
            let ai = alpha[i] / a0;
            assert!((mean - ai).abs() < 0.005, "{i}: {mean}");
            assert!((var - ai * (1.0 - ai) / (a0 + 1.0)).abs() < 0.003, "{i}: {var}");
        }
        assert!(dirichlet_distribution(&mut rng, &[1.0]).is_err() && dirichlet_distribution(&mut rng, &[1.0, 0.0]).is_err());
    }
}
