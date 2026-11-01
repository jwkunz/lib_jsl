//! Nonlinear systems by Newton's and Broyden's methods.
//!
//! Run with `cargo run -p lib_jsl_root_finding --example root_finding_systems`.

use lib_jsl_root_finding::{SystemOptions, broyden_system, newton_system};

type NoJacobian = fn(&[f64]) -> Vec<Vec<f64>>;

fn main() {
    // The intersection of the circle x² + y² = 4 with the curve y = eˣ − 1 in the first quadrant.
    let f = |v: &[f64]| vec![v[0] * v[0] + v[1] * v[1] - 4.0, v[0].exp() - 1.0 - v[1]];
    let jacobian = |v: &[f64]| vec![vec![2.0 * v[0], 2.0 * v[1]], vec![v[0].exp(), -1.0]];
    let options = SystemOptions::default();

    let exact = newton_system(f, Some(jacobian), vec![1.0, 1.0], options).unwrap();
    println!("Newton (exact Jacobian): {:?} after {} iterations", exact.x, exact.iterations);

    // Without a Jacobian, Newton's method forms one by finite differences at every step;
    // Broyden's method forms one only at the start and then updates it.
    let fd = newton_system(f, None::<NoJacobian>, vec![1.0, 1.0], options).unwrap();
    let broyden = broyden_system(f, vec![1.0, 1.0], options).unwrap();
    println!("Newton (differences): {} evaluations of F", fd.function_evaluations);
    println!("Broyden:              {} evaluations of F", broyden.function_evaluations);

    // Always check `converged`: from a poor start the solvers may stop at a minimum of ‖F‖.
    let none = newton_system(|v: &[f64]| vec![v[0] * v[0] + 1.0, v[1]], None::<NoJacobian>, vec![0.3, 0.0], options).unwrap();
    println!("x² + 1 = 0: converged = {}, residual {:.3}", none.converged, none.residual);
}
