//! Poisson random variables.

use crate::distributions::binomial::binomial_distribution;
use crate::distributions::gamma::gamma_distribution;
use crate::uniform_generator::UniformGenerator;

/// Knuth's multiplication method: the number of uniforms whose running product stays above `e^{−λ}`.
/// Expected cost `λ + 1` uniforms; used for small `λ`.
fn poisson_small<U: UniformGenerator>(rng: &mut U, lambda: f64) -> u64 {
    let limit = (-lambda).exp();
    let mut product = rng.next_f64();
    let mut k = 0;
    while product > limit {
        product *= rng.next_f64();
        k += 1;
    }
    k
}

/// A Poisson(`lambda`) sample (returned as `f64`), for `lambda ≥ 0`.
///
/// Small `λ` (below 16) uses Knuth's multiplication method, a fresh uniform per step. Larger `λ`
/// use the exact recursion of Ahrens and Dieter (Knuth, *TAOCP* 3.4.1): with `m = ⌊7λ/8⌋` and
/// `G ~ Gamma(m, 1)` (the time of the `m`-th event of a unit-rate Poisson process), the count is
/// `m + Poisson(λ − G)` if `G < λ`, and `Binomial(m − 1, λ/G)` otherwise, so the cost grows only
/// logarithmically in `λ`. (An earlier version reused one uniform throughout, with its loop
/// condition inverted, and returned −1 for every `λ`.) Returns `NaN` for negative or NaN `λ`.
pub fn poisson_distribution_knuth<U>(uniform_generator: &mut U, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    if lambda.is_nan() || lambda < 0.0 || lambda.is_infinite() {
        return f64::NAN;
    }
    let (mut lambda, mut offset) = (lambda, 0u64);
    while lambda >= 16.0 {
        let m = (0.875 * lambda).floor() as u64;
        let g = gamma_distribution(uniform_generator, m as f64, 1.0).expect("m ≥ 14 is a valid shape");
        if g < lambda {
            offset += m;
            lambda -= g;
        } else {
            let b = binomial_distribution(uniform_generator, m - 1, lambda / g).expect("λ/G lies in [0, 1]");
            return (offset + b) as f64;
        }
    }
    (offset + poisson_small(uniform_generator, lambda)) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn mean_and_variance_equal_lambda() {
        // The old sampler returned −1 for every λ.
        let mut rng = UniformRNG::from_seed(3);
        for lambda in [0.3, 4.0, 15.9, 16.0, 120.0, 5000.0] {
            let n = 100_000;
            let s: Vec<f64> = (0..n).map(|_| poisson_distribution_knuth(&mut rng, lambda)).collect();
            let mean = s.iter().sum::<f64>() / n as f64;
            let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let tol = 5.0 * (lambda / n as f64).sqrt();
            assert!((mean - lambda).abs() < tol, "λ = {lambda}: mean {mean}");
            assert!((var / lambda - 1.0).abs() < 0.03, "λ = {lambda}: variance {var}");
            assert!(s.iter().all(|&x| x >= 0.0 && x == x.floor()));
        }
        assert_eq!(poisson_distribution_knuth(&mut rng, 0.0), 0.0);
        assert!(poisson_distribution_knuth(&mut rng, -1.0).is_nan());
    }

    #[test]
    fn small_lambda_probabilities() {
        // P(0) = e^{-λ}, P(1) = λ e^{-λ}.
        let mut rng = UniformRNG::from_seed(4);
        let lambda = 1.5;
        let n = 200_000;
        let s: Vec<f64> = (0..n).map(|_| poisson_distribution_knuth(&mut rng, lambda)).collect();
        let p0 = s.iter().filter(|&&x| x == 0.0).count() as f64 / n as f64;
        let p1 = s.iter().filter(|&&x| x == 1.0).count() as f64 / n as f64;
        assert!((p0 - (-lambda).exp()).abs() < 0.004 && (p1 - lambda * (-lambda).exp()).abs() < 0.004, "{p0} {p1}");
    }
}
