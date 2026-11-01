//! Nonlinear least squares: fitting an exponential decay with uncertainties.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_least_squares`.

use lib_jsl_optimization::least_squares::{LevenbergMarquardt, curve_fit};

fn main() {
    // Radioactive counts measured every 10 minutes, with their Poisson standard deviations.
    let t: Vec<f64> = (0..10).map(|k| 10.0 * k as f64).collect();
    let counts = [1031.0, 735.0, 589.0, 409.0, 331.0, 226.0, 187.0, 128.0, 109.0, 70.0];
    let sigma: Vec<f64> = counts.iter().map(|c: &f64| c.sqrt()).collect();
    // Model N(t) = N₀ e^(−t/τ).
    let model = |t: f64, p: &[f64]| p[0] * (-t / p[1]).exp();
    let fit = curve_fit(model, &t, &counts, Some(&sigma), vec![1000.0, 50.0], &LevenbergMarquardt::new()).unwrap();
    println!("N₀ = {:.1} ± {:.1}", fit.parameters[0], fit.standard_errors[0]);
    println!("τ  = {:.2} ± {:.2} minutes (half-life {:.2})", fit.parameters[1], fit.standard_errors[1], fit.parameters[1] * 2f64.ln());
    println!("reduced χ² = {:.2}", fit.reduced_chi_squared);

    // The underlying solver takes any residual vector, here with an analytic Jacobian.
    let residuals = |p: &[f64]| vec![p[0] - 1.0, 10.0 * (p[1] - p[0] * p[0])];
    let jacobian = |p: &[f64]| vec![vec![1.0, 0.0], vec![-20.0 * p[0], 10.0]];
    let r = LevenbergMarquardt::new().solve(residuals, Some(jacobian), vec![-1.2, 1.0]).unwrap();
    println!("Rosenbrock as least squares: {:?} in {} iterations", r.parameters, r.iterations);
}
