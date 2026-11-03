//! Quasi-random (low-discrepancy) sequences for integration.
//!
//! Run with `cargo run -p lib_jsl_random --example random_quasi`.

use lib_jsl_random::quasi_random::{HaltonSequence, SobolSequence, radical_inverse, random_shift};
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};

fn main() {
    // The van der Corput sequence in base 2: 1/2, 1/4, 3/4, 1/8, 5/8, …
    let vdc: Vec<f64> = (1..=8).map(|n| radical_inverse(n, 2)).collect();
    println!("van der Corput: {vdc:?}");

    // Estimate π from the fraction of points in the quarter disc, with 4096 points.
    let inside = |p: &[f64]| (p[0] * p[0] + p[1] * p[1] < 1.0) as u32 as f64;
    let n = 4096;
    let mut rng = UniformRNG::from_seed(3);
    let random = 4.0 * (0..n).map(|_| inside(&[rng.next_f64(), rng.next_f64()])).sum::<f64>() / n as f64;
    let mut sobol = SobolSequence::new(2).unwrap();
    let sobol_pi = 4.0 * (0..n).map(|_| inside(&sobol.next_point())).sum::<f64>() / n as f64;
    let mut halton = HaltonSequence::new(2).unwrap();
    let halton_pi = 4.0 * (0..n).map(|_| inside(&halton.next_point())).sum::<f64>() / n as f64;
    println!("π ≈ {random:.5} (random), {sobol_pi:.5} (Sobol), {halton_pi:.5} (Halton); π = {:.5}", std::f64::consts::PI);

    // Randomised QMC: independent random shifts of the same Sobol points give an error estimate.
    let base: Vec<Vec<f64>> = { let mut s = SobolSequence::new(2).unwrap(); (0..n).map(|_| s.next_point()).collect() };
    let estimates: Vec<f64> = (0..10)
        .map(|_| {
            let mut pts = base.clone();
            random_shift(&mut pts, &mut rng);
            4.0 * pts.iter().map(|p| inside(p)).sum::<f64>() / n as f64
        })
        .collect();
    let mean = estimates.iter().sum::<f64>() / 10.0;
    let se = (estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / 90.0).sqrt();
    println!("shifted Sobol: π ≈ {mean:.5} ± {se:.5}");
}
