//! Two-point boundary value problems by shooting and by collocation.
//!
//! Run with `cargo run --release -p lib_jsl_integration --example ode_boundary_value`.

use lib_jsl_integration::boundary_value_problems::{BvpOptions, solve_bvp_collocation, solve_bvp_shooting};

fn main() {
    // A hanging cable (catenary): y'' = a √(1 + y'²), y(−1) = y(1) = 1, with a = 1.
    let catenary = |_: f64, y: &[f64]| vec![y[1], (1.0 + y[1] * y[1]).sqrt()];
    let ends = |ya: &[f64], yb: &[f64]| vec![ya[0] - 1.0, yb[0] - 1.0];
    // Exact: y = cosh x + 1 − cosh 1.
    let exact = |x: f64| x.cosh() + 1.0 - 1f64.cosh();

    let shot = solve_bvp_shooting(catenary, ends, -1.0, 1.0, vec![1.0, 0.0], BvpOptions::default()).unwrap();
    println!("shooting:    converged {}, sag y(0) = {:.12} (exact {:.12})", shot.converged, shot.evaluate(0.0)[0], exact(0.0));

    let x: Vec<f64> = (0..=8).map(|i| -1.0 + i as f64 / 4.0).collect();
    let guess: Vec<Vec<f64>> = x.iter().map(|_| vec![1.0, 0.0]).collect();
    let options = BvpOptions { tolerance: 1e-8, ..Default::default() };
    let col = solve_bvp_collocation(catenary, ends, x, guess, options).unwrap();
    println!("collocation: converged {}, sag y(0) = {:.12}, {} mesh points", col.converged, col.evaluate(0.0)[0], col.x.len());

    // Bratu's problem u'' + e^u = 0, u(0) = u(1) = 0 has two solutions; the guess picks one.
    let bratu = |_: f64, y: &[f64]| vec![y[1], -y[0].exp()];
    let zero = |ya: &[f64], yb: &[f64]| vec![ya[0], yb[0]];
    for amplitude in [0.5, 16.0] {
        let x: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let guess: Vec<Vec<f64>> = x.iter().map(|&t| vec![amplitude * t * (1.0 - t), amplitude * (1.0 - 2.0 * t)]).collect();
        let sol = solve_bvp_collocation(bratu, zero, x, guess, options).unwrap();
        println!("Bratu from guess {amplitude:>4}·x(1 − x): u(1/2) = {:.10}", sol.evaluate(0.5)[0]);
    }
}
