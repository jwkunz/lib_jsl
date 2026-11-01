//! Randomized quasi-Monte Carlo integration over boxes.
//!
//! See the book entry [Monte Carlo and quasi-Monte Carlo integration](https://jwkunz.github.io/lib_jsl/book/integration/monte-carlo.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::quasi_random::{HaltonSequence, SobolSequence, random_shift};
use lib_jsl_random::uniform_generator::UniformRNG;

/// The low-discrepancy sequence to sample with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuasiRandomSequence {
    /// Sobol (up to 21 dimensions); use a power of two for the number of points.
    Sobol,
    /// Halton (up to 100 dimensions).
    Halton,
}

/// The outcome of [`quasi_monte_carlo_integration`].
#[derive(Debug, Clone)]
pub struct QuasiMonteCarloResult {
    /// The mean of the replicate estimates.
    pub value: f64,
    /// Standard error from the spread of the replicates (NaN with a single replicate).
    pub standard_error: f64,
    /// The number of integrand evaluations (`points × replicates`).
    pub evaluations: usize,
}

/// Integrates `f` over the box `bounds` (one `(lower, upper)` pair per dimension) by randomized
/// quasi-Monte Carlo: `replicates` independent Cranley-Patterson shifts of the first `points`
/// points of a low-discrepancy sequence, each giving an unbiased estimate. The result is their
/// mean, and their spread gives a standard error. For smooth integrands the error falls almost
/// like `1/N`, against `1/√N` for plain Monte Carlo.
pub fn quasi_monte_carlo_integration<F: Fn(&[f64]) -> f64>(
    f: F,
    bounds: &[(f64, f64)],
    points: usize,
    replicates: usize,
    sequence: QuasiRandomSequence,
    seed: u64,
) -> Result<QuasiMonteCarloResult, ErrorsJSL> {
    let d = bounds.len();
    if d == 0 || points == 0 || replicates == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Dimensions, points and replicates must be positive."));
    }
    if bounds.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("The bounds must be finite."));
    }
    let base: Vec<Vec<f64>> = match sequence {
        QuasiRandomSequence::Sobol => {
            let mut s = SobolSequence::new(d)?;
            (0..points).map(|_| s.next_point()).collect()
        }
        QuasiRandomSequence::Halton => {
            let mut h = HaltonSequence::new(d)?;
            (0..points).map(|_| h.next_point()).collect()
        }
    };
    let volume: f64 = bounds.iter().map(|(a, b)| b - a).product();
    let mut rng = UniformRNG::from_seed(seed);
    let mut x = vec![0.0; d];
    let estimates: Vec<f64> = (0..replicates)
        .map(|_| {
            let mut shifted = base.clone();
            random_shift(&mut shifted, &mut rng);
            let sum: f64 = shifted
                .iter()
                .map(|u| {
                    for k in 0..d {
                        x[k] = bounds[k].0 + u[k] * (bounds[k].1 - bounds[k].0);
                    }
                    f(&x)
                })
                .sum();
            volume * sum / points as f64
        })
        .collect();
    let r = replicates as f64;
    let value = estimates.iter().sum::<f64>() / r;
    let standard_error = if replicates > 1 {
        (estimates.iter().map(|e| (e - value).powi(2)).sum::<f64>() / (r - 1.0) / r).sqrt()
    } else {
        f64::NAN
    };
    Ok(QuasiMonteCarloResult { value, standard_error, evaluations: points * replicates })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_random::uniform_generator::UniformGenerator;
    use std::f64::consts::PI;

    /// Π (π/2) sin(π x_i) over the unit cube integrates to one.
    fn product(x: &[f64]) -> f64 {
        x.iter().map(|&v| 0.5 * PI * (PI * v).sin()).product()
    }

    #[test]
    fn beats_plain_monte_carlo_and_reports_a_consistent_error() {
        let bounds = vec![(0.0, 1.0); 5];
        for sequence in [QuasiRandomSequence::Sobol, QuasiRandomSequence::Halton] {
            let r = quasi_monte_carlo_integration(product, &bounds, 4096, 16, sequence, 7).unwrap();
            let error = (r.value - 1.0).abs();
            assert!(error < 2e-3 && error < 5.0 * r.standard_error + 1e-12, "{sequence:?}: {r:?}");
            // Plain Monte Carlo with the same 65536 evaluations has a standard error near
            // σ/√N ≈ 0.0045 for this integrand.
            let mut rng = UniformRNG::from_seed(3);
            let samples: Vec<f64> = (0..r.evaluations).map(|_| product(&(0..5).map(|_| rng.next_f64()).collect::<Vec<_>>())).collect();
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let sd = (samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt();
            let mc_error = sd / (samples.len() as f64).sqrt();
            assert!(r.standard_error < mc_error / 5.0, "{sequence:?}: qmc {} mc {mc_error}", r.standard_error);
        }
    }

    #[test]
    fn scales_to_the_box_and_validates() {
        let r = quasi_monte_carlo_integration(|x| x[0] * x[1], &[(0.0, 2.0), (-1.0, 3.0)], 1024, 8, QuasiRandomSequence::Sobol, 1).unwrap();
        assert!((r.value - 8.0).abs() < (4.0 * r.standard_error).min(0.05), "{r:?}");
        let single = quasi_monte_carlo_integration(|x| x[0], &[(0.0, 1.0)], 256, 1, QuasiRandomSequence::Halton, 1).unwrap();
        assert!(single.standard_error.is_nan() && (single.value - 0.5).abs() < 1e-2);
        assert!(quasi_monte_carlo_integration(|x| x[0], &[(0.0, 1.0); 22], 16, 2, QuasiRandomSequence::Sobol, 1).is_err());
        assert!(quasi_monte_carlo_integration(|x| x[0], &[], 16, 2, QuasiRandomSequence::Sobol, 1).is_err());
    }
}
