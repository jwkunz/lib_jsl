//! Constrained minimisation by SQP and the augmented Lagrangian method.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_constrained`.

use lib_jsl_optimization::constrained::{ConstrainedOptions, ConstrainedProblem, augmented_lagrangian, sqp};

/// The box of largest volume with surface area 6 (no lid constraint): maximise xyz subject to
/// 2(xy + yz + zx) = 6 and x, y, z ≥ 0. The answer is the unit cube.
struct Box3;
impl ConstrainedProblem for Box3 {
    fn objective(&self, x: &[f64]) -> f64 {
        -x[0] * x[1] * x[2]
    }
    fn equalities(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] * x[1] + x[1] * x[2] + x[2] * x[0]) - 6.0]
    }
    fn inequalities(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|v| -v).collect()
    }
}

fn main() {
    let options = ConstrainedOptions::default();
    let r = sqp(&Box3, vec![0.5, 1.0, 2.0], options).unwrap();
    println!("SQP: {:?}, volume {:.8}, λ = {:?}, {} iterations", r.parameters, -r.objective_value, r.equality_multipliers, r.iterations);
    let r = augmented_lagrangian(&Box3, vec![0.5, 1.0, 2.0], options).unwrap();
    println!("augmented Lagrangian: {:?}, volume {:.8}, {} outer iterations", r.parameters, -r.objective_value, r.iterations);
}
