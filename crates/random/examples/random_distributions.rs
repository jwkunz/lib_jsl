//! Sampling from probability distributions.
//!
//! Run with `cargo run -p lib_jsl_random --example random_distributions`.

use lib_jsl_random::distributions::{
    MultivariateNormal, beta_distribution, binomial_distribution, dirichlet_distribution, exponential_distribution, gamma_distribution,
    gaussian_distribution_box_muller, poisson_distribution_knuth,
};
use lib_jsl_random::uniform_generator::UniformRNG;

fn main() {
    let mut rng = UniformRNG::from_seed(2025);
    // Waiting times of a Poisson process with rate 3 per hour, and counts per hour.
    let waits: Vec<f64> = (0..5).map(|_| exponential_distribution(&mut rng, 3.0)).collect();
    println!("waiting times (hours): {waits:.3?}");
    let counts: Vec<f64> = (0..8).map(|_| poisson_distribution_knuth(&mut rng, 3.0)).collect();
    println!("events in 8 hours: {counts:?}");

    // Normal samples come in pairs from Box-Muller.
    let (z1, z2) = gaussian_distribution_box_muller(&mut rng, 0.0, 1.0);
    println!("two normals: {z1:.4}, {z2:.4}");

    // Gamma (shape, scale), beta from two gammas, binomial for 10⁶ trials.
    println!("gamma(2, 1.5): {:.4}", gamma_distribution(&mut rng, 2.0, 1.5).unwrap());
    println!("beta(2, 5): {:.4}", beta_distribution(&mut rng, 2.0, 5.0).unwrap());
    println!("binomial(10⁶, 0.3): {}", binomial_distribution(&mut rng, 1_000_000, 0.3).unwrap());

    // A Dirichlet sample is a random probability vector.
    let p = dirichlet_distribution(&mut rng, &[1.0, 2.0, 3.0]).unwrap();
    println!("Dirichlet(1, 2, 3): {p:.4?}, sum {:.6}", p.iter().sum::<f64>());

    // Correlated normal pairs with correlation 0.8.
    let mvn = MultivariateNormal::new(vec![0.0, 0.0], &[vec![1.0, 0.8], vec![0.8, 1.0]]).unwrap();
    let pairs: Vec<Vec<f64>> = (0..20_000).map(|_| mvn.sample(&mut rng)).collect();
    let corr = pairs.iter().map(|p| p[0] * p[1]).sum::<f64>() / pairs.len() as f64;
    println!("sample correlation of 20 000 pairs: {corr:.3}");
}
