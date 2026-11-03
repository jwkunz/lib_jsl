//! The nonparametric bootstrap: standard errors and confidence intervals by resampling.
//!
//! See the book entry [The bootstrap](https://jwkunz.github.io/lib_jsl/book/statistics/bootstrap.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
use lib_jsl_special_functions::{normal_cdf, normal_quantile};

/// Bootstrap replicates of a statistic.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// The statistic on the original sample.
    pub estimate: f64,
    /// The statistic on each resample.
    pub replicates: Vec<f64>,
    /// The standard deviation of the replicates.
    pub standard_error: f64,
    /// `mean(replicates) − estimate`.
    pub bias: f64,
}

fn quantile_of_sorted(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p.clamp(0.0, 1.0);
    let (lo, frac) = (h.floor() as usize, h - h.floor());
    if lo + 1 < sorted.len() { sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]) } else { sorted[lo] }
}

// ANCHOR: bootstrap
/// Resamples `data` with replacement `resamples` times and evaluates `statistic` on each.
pub fn bootstrap<S: Fn(&[f64]) -> f64>(data: &[f64], statistic: S, resamples: usize, seed: u64) -> Result<BootstrapResult, ErrorsJSL> {
    if data.len() < 2 || resamples < 2 {
        return Err(ErrorsJSL::InvalidInputRange("Need at least two observations and two resamples."));
    }
    let n = data.len();
    let mut rng = UniformRNG::from_seed(seed);
    let mut sample = vec![0.0; n];
    let replicates: Vec<f64> = (0..resamples)
        .map(|_| {
            for s in sample.iter_mut() {
                *s = data[((rng.next_f64() * n as f64) as usize).min(n - 1)];
            }
            statistic(&sample)
        })
        .collect();
    let estimate = statistic(data);
    let m = replicates.iter().sum::<f64>() / resamples as f64;
    let standard_error = (replicates.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (resamples - 1) as f64).sqrt();
    Ok(BootstrapResult { estimate, replicates, standard_error, bias: m - estimate })
}
// ANCHOR_END: bootstrap

impl BootstrapResult {
    fn sorted(&self) -> Vec<f64> {
        let mut s = self.replicates.clone();
        s.sort_by(f64::total_cmp);
        s
    }

    /// The percentile interval: the `(1 ∓ confidence)/2` quantiles of the replicates.
    pub fn percentile_interval(&self, confidence: f64) -> (f64, f64) {
        let s = self.sorted();
        let alpha = (1.0 - confidence) / 2.0;
        (quantile_of_sorted(&s, alpha), quantile_of_sorted(&s, 1.0 - alpha))
    }

// ANCHOR: bca
    /// The bias-corrected and accelerated (BCa) interval of Efron: the percentile levels are
    /// shifted by the bias correction `z₀ = Φ⁻¹(#{θ* < θ̂}/B)` and the acceleration `a` from the
    /// jackknife skewness of `statistic` on `data` (the same data and statistic as the
    /// bootstrap). Second-order accurate, unlike the percentile interval.
    pub fn bca_interval<S: Fn(&[f64]) -> f64>(&self, data: &[f64], statistic: S, confidence: f64) -> (f64, f64) {
        let s = self.sorted();
        let below = s.iter().filter(|&&r| r < self.estimate).count() as f64;
        let z0 = normal_quantile((below / s.len() as f64).clamp(1e-10, 1.0 - 1e-10), 0.0, 1.0);
        let n = data.len();
        let jackknife: Vec<f64> = (0..n).map(|i| statistic(&[&data[..i], &data[i + 1..]].concat())).collect();
        let jm = jackknife.iter().sum::<f64>() / n as f64;
        let (num, den) = jackknife.iter().fold((0.0, 0.0), |(a, b), &t| (a + (jm - t).powi(3), b + (jm - t).powi(2)));
        let acceleration = if den > 0.0 { num / (6.0 * den.powf(1.5)) } else { 0.0 };
        let level = |z: f64| normal_cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z)), 0.0, 1.0);
        let z = normal_quantile((1.0 + confidence) / 2.0, 0.0, 1.0);
        (quantile_of_sorted(&s, level(-z)), quantile_of_sorted(&s, level(z)))
    }
// ANCHOR_END: bca
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_random::distributions::exponential_distribution;

    fn mean(x: &[f64]) -> f64 {
        x.iter().sum::<f64>() / x.len() as f64
    }

    #[test]
    fn standard_error_of_the_mean_matches_theory() {
        let data: Vec<f64> = (0..100).map(|i| ((i * 37) % 100) as f64 / 10.0).collect();
        let b = bootstrap(&data, mean, 4000, 1).unwrap();
        let sd = (data.iter().map(|v| (v - mean(&data)).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
        assert!((b.standard_error / (sd / 10.0) - 1.0).abs() < 0.05, "{} vs {}", b.standard_error, sd / 10.0);
        assert!(b.bias.abs() < 0.02 && b.estimate == mean(&data));
        let (lo, hi) = b.percentile_interval(0.95);
        assert!(lo < b.estimate && b.estimate < hi && ((hi - lo) / (2.0 * 1.96 * sd / 10.0) - 1.0).abs() < 0.1);
        assert!(bootstrap(&[1.0], mean, 10, 0).is_err());
    }

    #[test]
    fn bca_intervals_cover_a_skewed_mean_at_the_nominal_rate() {
        // Means of exponential samples (true mean 1) are right-skewed; BCa shifts its interval
        // up relative to the percentile one and covers close to the nominal 90%.
        let mut rng = UniformRNG::from_seed(7);
        let trials = 200;
        let mut covered = 0;
        let mut shifted_up = 0;
        for t in 0..trials {
            let data: Vec<f64> = (0..25).map(|_| exponential_distribution(&mut rng, 1.0)).collect();
            let b = bootstrap(&data, mean, 1000, t as u64).unwrap();
            let (lo, hi) = b.bca_interval(&data, mean, 0.9);
            let (plo, phi) = b.percentile_interval(0.9);
            covered += (lo <= 1.0 && 1.0 <= hi) as usize;
            shifted_up += (lo + hi > plo + phi) as usize;
        }
        assert!((150..=195).contains(&covered), "coverage {covered}/{trials}");
        assert!(shifted_up > trials * 3 / 4, "{shifted_up}");
    }
}
