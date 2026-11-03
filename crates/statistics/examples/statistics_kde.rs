//! Estimate the density of eruption gaps of a geyser-like bimodal process.

use lib_jsl_random::uniform_generator::UniformRNG;
use lib_jsl_random::distributions::gaussian_distribution_box_muller;
use lib_jsl_statistics::kernel_density::{Bandwidth, GaussianKde};

fn main() {
    let mut rng = UniformRNG::from_seed(1);
    // Waiting times in minutes: short gaps near 55, long gaps near 80.
    let data: Vec<f64> = (0..300)
        .map(|i| if i % 3 == 0 { gaussian_distribution_box_muller(&mut rng, 55.0, 6.0).0 } else { gaussian_distribution_box_muller(&mut rng, 80.0, 5.0).0 })
        .collect();
    for rule in [Bandwidth::Scott, Bandwidth::Silverman, Bandwidth::Fixed(2.0)] {
        let kde = GaussianKde::new(&data, rule).unwrap();
        let curve: Vec<String> = (40..=95).step_by(5).map(|x| format!("{:.3}", kde.pdf(x as f64))).collect();
        println!("{rule:?} (h = {:.2}): {}", kde.bandwidth(), curve.join(" "));
    }
    let kde = GaussianKde::new(&data, Bandwidth::Silverman).unwrap();
    println!("P(gap < 65) = {:.3}", kde.cdf(65.0));
    let draws = kde.resample(5, &mut rng);
    println!("five simulated gaps: {draws:.1?}");
}
