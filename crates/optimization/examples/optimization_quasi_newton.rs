//! Newton's method, BFGS and L-BFGS on the Rosenbrock function.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_quasi_newton`.

use lib_jsl_optimization::newton_raphson_method::NewtonRaphsonMethod;
use lib_jsl_optimization::optimization_traits::{
    GradientBasedMinimizationEngine, GradientFunction, HessianBasedMinimizationEngine, HessianFunction, MinimizationControls, ObjectiveFunction,
};
use lib_jsl_optimization::quasi_newton::{Bfgs, Lbfgs};

/// f(x, y) = 100 (y − x²)² + (1 − x)², minimum 0 at (1, 1).
struct Rosenbrock;
impl ObjectiveFunction for Rosenbrock {
    fn evaluate(&self, p: &[f64]) -> f64 {
        100.0 * (p[1] - p[0] * p[0]).powi(2) + (1.0 - p[0]).powi(2)
    }
}
impl GradientFunction for Rosenbrock {
    fn evaluate(&self, p: &[f64]) -> Vec<f64> {
        let t = p[1] - p[0] * p[0];
        vec![-400.0 * p[0] * t - 2.0 * (1.0 - p[0]), 200.0 * t]
    }
}
impl HessianFunction for Rosenbrock {
    fn evaluate(&self, p: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![1200.0 * p[0] * p[0] - 400.0 * p[1] + 2.0, -400.0 * p[0]], vec![-400.0 * p[0], 200.0]]
    }
}

fn main() {
    let start = vec![-1.2, 1.0];
    let mut newton = NewtonRaphsonMethod::new();
    newton.set_tolerance(Some(1e-15));
    let r = newton.hessian_based_minimize(&Rosenbrock, &Rosenbrock, &Rosenbrock, start.clone()).unwrap();
    println!("Newton:  {:?} in {} iterations", r.parameters, r.iterations);
    let r = Bfgs::new().gradient_based_minimize(&Rosenbrock, &Rosenbrock, start.clone()).unwrap();
    println!("BFGS:    {:?} in {} iterations", r.parameters, r.iterations);
    // L-BFGS keeps only the last few correction pairs: O(m n) memory instead of O(n²).
    let r = Lbfgs::new(5).gradient_based_minimize(&Rosenbrock, &Rosenbrock, start).unwrap();
    println!("L-BFGS:  {:?} in {} iterations", r.parameters, r.iterations);
}
