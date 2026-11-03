//! Bayesian inference by Markov chain Monte Carlo.
//!
//! Run with `cargo run -p lib_jsl_random --example random_mcmc`.

use lib_jsl_random::mcmc::{MetropolisOptions, metropolis_hastings};
use lib_jsl_random::uniform_generator::UniformRNG;

fn main() {
    // A dose-response experiment: at dose d, n animals, k responded. Model P(response) =
    // 1/(1 + e^{−(a + b d)}) with weak normal priors on a and b; sample the posterior of (a, b).
    let data = [(-0.86, 5, 0), (-0.30, 5, 1), (-0.05, 5, 3), (0.73, 5, 5)];
    let log_posterior = |p: &[f64]| {
        let (a, b) = (p[0], p[1]);
        let prior = -(a * a + b * b) / (2.0 * 100.0);
        let likelihood: f64 = data.iter().map(|&(d, n, k)| {
            let eta = a + b * d;
            // log p = −ln(1 + e^{−η}), log(1 − p) = −ln(1 + e^{η}), computed stably.
            let log_p = -(-eta).exp().ln_1p();
            let log_q = -eta.exp().ln_1p();
            k as f64 * log_p + (n - k) as f64 * log_q
        }).sum();
        prior + likelihood
    };
    let options = MetropolisOptions { proposal_scale: vec![1.0, 2.0], samples: 20_000, burn_in: 5_000, thin: 5, adapt_during_burn_in: true };
    let chain = metropolis_hastings(log_posterior, vec![0.0, 0.0], options, &mut UniformRNG::from_seed(1)).unwrap();
    let mean = chain.mean();
    println!("posterior means: a = {:.2}, b = {:.2}", mean[0], mean[1]);
    println!("acceptance rate {:.2}, lag-1 autocorrelation of a {:.2}", chain.acceptance_rate, chain.autocorrelation(0, 1));

    // The dose at which half respond, LD50 = −a/b, has a posterior too: summarise its quantiles.
    let mut ld50: Vec<f64> = chain.samples.iter().map(|s| -s[0] / s[1]).collect();
    ld50.sort_by(f64::total_cmp);
    let q = |p: f64| ld50[(p * (ld50.len() - 1) as f64) as usize];
    println!("LD50: median {:.3}, 95% interval [{:.3}, {:.3}]", q(0.5), q(0.025), q(0.975));
}
