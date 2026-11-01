//! Markov chain Monte Carlo: random-walk Metropolis-Hastings.

use crate::distributions::gaussian::gaussian_distribution_box_muller;
use crate::uniform_generator::UniformGenerator;
use lib_jsl_core::ErrorsJSL;

/// Settings for [`metropolis_hastings`].
#[derive(Debug, Clone)]
pub struct MetropolisOptions {
    /// Standard deviation of the Gaussian proposal per coordinate.
    pub proposal_scale: Vec<f64>,
    /// Retained samples.
    pub samples: usize,
    /// Initial iterations discarded while the chain forgets its starting point.
    pub burn_in: usize,
    /// Keep every `thin`-th state (1 keeps all), reducing autocorrelation and storage.
    pub thin: usize,
    /// During burn-in, rescale the proposal every 100 steps towards an acceptance rate near the
    /// optimal 0.234 for random-walk proposals in many dimensions (0.44 in one).
    pub adapt_during_burn_in: bool,
}

/// A Markov chain.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    pub samples: Vec<Vec<f64>>,
    /// The fraction of proposals accepted after burn-in.
    pub acceptance_rate: f64,
    /// The proposal scale actually used after any adaptation.
    pub proposal_scale: Vec<f64>,
}

impl MarkovChain {
    /// The sample mean of each coordinate.
    pub fn mean(&self) -> Vec<f64> {
        let d = self.samples[0].len();
        let m = self.samples.len() as f64;
        (0..d).map(|i| self.samples.iter().map(|s| s[i]).sum::<f64>() / m).collect()
    }

    /// The normalized autocorrelation of coordinate `i` at `lag`.
    pub fn autocorrelation(&self, i: usize, lag: usize) -> f64 {
        let x: Vec<f64> = self.samples.iter().map(|s| s[i]).collect();
        let m = x.len();
        if lag >= m {
            return 0.0;
        }
        let mean = x.iter().sum::<f64>() / m as f64;
        let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
        (0..m - lag).map(|k| (x[k] - mean) * (x[k + lag] - mean)).sum::<f64>() / var
    }
}

/// Samples from the density proportional to `exp(log_density(x))` by random-walk
/// Metropolis-Hastings: propose `x' = x + σ ⊙ z` with `z` standard normal, accept with
/// probability `min(1, p(x')/p(x))`. Only an unnormalised log-density is needed; returning
/// `−∞` marks points outside the support.
pub fn metropolis_hastings<F, U>(log_density: F, initial: Vec<f64>, options: MetropolisOptions, rng: &mut U) -> Result<MarkovChain, ErrorsJSL>
where
    F: Fn(&[f64]) -> f64,
    U: UniformGenerator,
{
    let d = initial.len();
    if d == 0 || options.proposal_scale.len() != d || options.samples == 0 || options.thin == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Need a non-empty start, a matching proposal scale, samples and thin >= 1."));
    }
    if options.proposal_scale.iter().any(|s| s.is_nan() || *s <= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Proposal scales must be positive."));
    }
    let mut x = initial;
    let mut log_p = log_density(&x);
    if log_p.is_nan() || log_p == f64::NEG_INFINITY {
        return Err(ErrorsJSL::InvalidInputRange("The initial point must have positive density."));
    }
    let mut scale = options.proposal_scale.clone();
    let target = if d == 1 { 0.44 } else { 0.234 };
    let total = options.burn_in + options.samples * options.thin;
    let mut samples = Vec::with_capacity(options.samples);
    let (mut accepted, mut window_accepted) = (0usize, 0usize);
    let mut z = Vec::with_capacity(d + 1);
    for step in 0..total {
        z.clear();
        while z.len() < d {
            let (a, b) = gaussian_distribution_box_muller(rng, 0.0, 1.0);
            z.push(a);
            z.push(b);
        }
        let proposal: Vec<f64> = (0..d).map(|i| x[i] + scale[i] * z[i]).collect();
        let log_q = log_density(&proposal);
        let accept = log_q > f64::NEG_INFINITY && (log_q >= log_p || rng.next_f64().ln() < log_q - log_p);
        if accept {
            x = proposal;
            log_p = log_q;
            window_accepted += 1;
            if step >= options.burn_in {
                accepted += 1;
            }
        }
        if step < options.burn_in {
            if options.adapt_during_burn_in && (step + 1) % 100 == 0 {
                let rate = window_accepted as f64 / 100.0;
                let factor = ((rate - target) * 2.0).exp();
                scale.iter_mut().for_each(|s| *s *= factor);
                window_accepted = 0;
            }
        } else if (step - options.burn_in) % options.thin == options.thin - 1 {
            samples.push(x.clone());
        }
    }
    Ok(MarkovChain { samples, acceptance_rate: accepted as f64 / (options.samples * options.thin) as f64, proposal_scale: scale })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn recovers_a_correlated_gaussian() {
        // N(μ, Σ) with Σ = [[1, 0.8], [0.8, 1]], log-density up to a constant.
        let mu = [1.0, -2.0];
        let det = 1.0 - 0.64;
        let log_density = |x: &[f64]| {
            let (a, b) = (x[0] - mu[0], x[1] - mu[1]);
            -0.5 * (a * a - 1.6 * a * b + b * b) / det
        };
        let options = MetropolisOptions { proposal_scale: vec![0.1, 0.1], samples: 40_000, burn_in: 5_000, thin: 5, adapt_during_burn_in: true };
        let chain = metropolis_hastings(log_density, vec![5.0, 5.0], options, &mut UniformRNG::from_seed(3)).unwrap();
        let m = chain.mean();
        assert!((m[0] - mu[0]).abs() < 0.05 && (m[1] - mu[1]).abs() < 0.05, "{m:?}");
        let n = chain.samples.len() as f64;
        let cov = chain.samples.iter().map(|s| (s[0] - m[0]) * (s[1] - m[1])).sum::<f64>() / n;
        let var0 = chain.samples.iter().map(|s| (s[0] - m[0]).powi(2)).sum::<f64>() / n;
        assert!((cov - 0.8).abs() < 0.06 && (var0 - 1.0).abs() < 0.08, "cov {cov} var {var0}");
        // Adaptation steered the acceptance rate towards 0.234 and grew the tiny initial step.
        assert!(chain.acceptance_rate > 0.15 && chain.acceptance_rate < 0.35, "{}", chain.acceptance_rate);
        assert!(chain.proposal_scale[0] > 0.3);
        assert!(chain.autocorrelation(0, 1) < 0.9 && chain.autocorrelation(0, 50) < 0.1);
    }

    #[test]
    fn respects_the_support_and_validates() {
        // Exponential(1) on x > 0: mean 1.
        let log_density = |x: &[f64]| if x[0] > 0.0 { -x[0] } else { f64::NEG_INFINITY };
        let options = MetropolisOptions { proposal_scale: vec![1.0], samples: 50_000, burn_in: 2_000, thin: 2, adapt_during_burn_in: false };
        let chain = metropolis_hastings(log_density, vec![1.0], options.clone(), &mut UniformRNG::from_seed(4)).unwrap();
        assert!(chain.samples.iter().all(|s| s[0] > 0.0));
        assert!((chain.mean()[0] - 1.0).abs() < 0.05, "{:?}", chain.mean());
        assert!(metropolis_hastings(log_density, vec![-1.0], options.clone(), &mut UniformRNG::from_seed(4)).is_err());
        assert!(metropolis_hastings(log_density, vec![1.0, 2.0], options, &mut UniformRNG::from_seed(4)).is_err());
    }
}
