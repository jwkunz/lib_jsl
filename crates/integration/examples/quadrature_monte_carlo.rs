//! Monte Carlo and randomised quasi-Monte Carlo integration, with error estimates.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_monte_carlo`.

use lib_jsl_integration::monte_carlo_integration::{monte_carlo, monte_carlo_integration};
use lib_jsl_integration::quasi_monte_carlo::{QuasiRandomSequence, quasi_monte_carlo_integration};

fn main() {
    // The volume of the unit ball in 6 dimensions, π³/6, as ∫ of an indicator over [−1, 1]⁶.
    let exact = std::f64::consts::PI.powi(3) / 6.0;
    let ball = |x: &[f64]| if x.iter().map(|v| v * v).sum::<f64>() <= 1.0 { 1.0 } else { 0.0 };
    let cube = vec![(-1.0, 1.0); 6];
    for n in [1_000, 10_000, 100_000, 1_000_000] {
        let r = monte_carlo(ball, &cube, n, 42).unwrap();
        println!("N = {n:>9}: {:.5} ± {:.5} (true error {:.5})", r.value, r.standard_error, (r.value - exact).abs());
    }

    // Sampling until a target standard error is reached.
    let v = monte_carlo_integration(ball, &cube, (0.01, 10_000), None).unwrap();
    println!("to a standard error of 0.01: {v:.4} (exact {exact:.4})");

    // A smooth integrand in 8 dimensions: randomised Sobol points beat random ones.
    let smooth = |x: &[f64]| x.iter().map(|v| (1.0 + v).ln() / (2.0 * 2f64.ln() - 1.0)).product::<f64>();
    let unit = vec![(0.0, 1.0); 8];
    let mc = monte_carlo(smooth, &unit, 1 << 16, 1).unwrap();
    let qmc = quasi_monte_carlo_integration(smooth, &unit, 1 << 13, 8, QuasiRandomSequence::Sobol, 1).unwrap();
    println!("8-dimensional product, 65536 evaluations each:");
    println!("  Monte Carlo        error {:.1e}, standard error {:.1e}", (mc.value - 1.0).abs(), mc.standard_error);
    println!("  randomised Sobol   error {:.1e}, standard error {:.1e}", (qmc.value - 1.0).abs(), qmc.standard_error);
}
