//! A stiff chemical reaction (Robertson's problem) with explicit and implicit solvers.
//!
//! Run with `cargo run --release -p lib_jsl_integration --example ode_stiff`.

use lib_jsl_integration::dormand_prince::dormand_prince_solve;
use lib_jsl_integration::stiff_ode::{OdeOptions, bdf, rosenbrock23};

/// Three species reacting at rates spanning nine orders of magnitude.
fn robertson(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![
        -0.04 * y[0] + 1e4 * y[1] * y[2],
        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
        3e7 * y[1] * y[1],
    ]
}

/// The Jacobian ∂f/∂y, which saves the solvers from estimating it by differences.
fn jacobian(_t: f64, y: &[f64]) -> Vec<Vec<f64>> {
    vec![
        vec![-0.04, 1e4 * y[2], 1e4 * y[1]],
        vec![0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
        vec![0.0, 6e7 * y[1], 0.0],
    ]
}

fn main() {
    let options = OdeOptions { relative_tolerance: 1e-6, absolute_tolerance: 1e-10, max_steps: 1_000_000, ..Default::default() };
    let start = [1.0, 0.0, 0.0];
    for t_end in [40.0, 4e5] {
        println!("t = {t_end}:");
        let s = bdf(robertson, Some(jacobian), 0.0, &start, t_end, 5, options).unwrap();
        println!("  BDF:            {:>7} steps, y = {:.6e}", s.t.len() - 1, s.last()[0]);
        let s = rosenbrock23(robertson, Some(jacobian), 0.0, &start, t_end, options).unwrap();
        println!("  Rosenbrock:     {:>7} steps, y = {:.6e}", s.t.len() - 1, s.last()[0]);
        match dormand_prince_solve(robertson, 0.0, &start, t_end, options) {
            Ok(s) => println!("  Dormand-Prince: {:>7} steps, y = {:.6e}", s.t.len() - 1, s.last()[0]),
            Err(e) => println!("  Dormand-Prince: failed ({e:?})"),
        }
    }
}
