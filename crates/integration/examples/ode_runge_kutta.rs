//! Initial value problems with the classical Runge-Kutta method and adaptive Dormand-Prince.
//!
//! Run with `cargo run -p lib_jsl_integration --example ode_runge_kutta`.

use lib_jsl_integration::dormand_prince::dormand_prince_solve;
use lib_jsl_integration::runge_kutta_4::rk4_solve;
use lib_jsl_integration::stiff_ode::OdeOptions;

fn main() {
    // Lotka-Volterra predator-prey model: x' = x − xy, y' = −y + xy.
    let predator_prey = |_t: f64, y: &[f64]| vec![y[0] - y[0] * y[1], -y[1] + y[0] * y[1]];
    // The quantity V = x − ln x + y − ln y is conserved exactly; its drift measures the error.
    let invariant = |y: &[f64]| y[0] - y[0].ln() + y[1] - y[1].ln();
    let start = [2.0, 1.0];

    for steps in [100, 200, 400] {
        let s = rk4_solve(predator_prey, 0.0, &start, 20.0, steps).unwrap();
        let drift = (invariant(s.y.last().unwrap()) - invariant(&start)).abs();
        println!("RK4, {steps:3} steps: invariant drift {drift:.1e} ({} evaluations)", s.function_evaluations);
    }
    for tol in [1e-6, 1e-9, 1e-12] {
        let options = OdeOptions { relative_tolerance: tol, absolute_tolerance: tol, ..Default::default() };
        let s = dormand_prince_solve(predator_prey, 0.0, &start, 20.0, options).unwrap();
        let drift = (invariant(s.y.last().unwrap()) - invariant(&start)).abs();
        println!(
            "Dormand-Prince, tolerance {tol:.0e}: drift {drift:.1e} ({} steps, {} rejected, {} evaluations)",
            s.t.len() - 1,
            s.rejected_steps,
            s.function_evaluations
        );
    }
}
