//! Gradient descent and nonlinear conjugate gradients.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_gradient`.

use lib_jsl_optimization::conjugate_gradient::{ConjugateGradientBeta, NonlinearConjugateGradient};
use lib_jsl_optimization::gradient_descent::GradientDescent;
use lib_jsl_optimization::optimization_traits::{GradientBasedMinimizationEngine, GradientFunction, MinimizationControls, ObjectiveFunction};

/// Least squares for a straight line through four points: f(a, b) = Σ (a + b xᵢ − yᵢ)².
struct LineFit;
const DATA: [(f64, f64); 4] = [(0.0, 1.1), (1.0, 2.9), (2.0, 5.2), (3.0, 6.8)];
impl ObjectiveFunction for LineFit {
    fn evaluate(&self, p: &[f64]) -> f64 {
        DATA.iter().map(|(x, y)| (p[0] + p[1] * x - y).powi(2)).sum()
    }
}
impl GradientFunction for LineFit {
    fn evaluate(&self, p: &[f64]) -> Vec<f64> {
        let r: Vec<f64> = DATA.iter().map(|(x, y)| p[0] + p[1] * x - y).collect();
        vec![2.0 * r.iter().sum::<f64>(), 2.0 * r.iter().zip(&DATA).map(|(ri, (x, _))| ri * x).sum::<f64>()]
    }
}

fn main() {
    // Gradient descent with a fixed learning rate: it must be below 2 / L, where L is the
    // largest eigenvalue of the Hessian (here about 30.2).
    let mut gd = GradientDescent::new(0.03);
    gd.set_max_iterations(Some(10_000));
    gd.set_tolerance(Some(1e-14));
    let r = gd.gradient_based_minimize(&LineFit, &LineFit, vec![0.0, 0.0]).unwrap();
    println!("gradient descent: a = {:.6}, b = {:.6} in {} iterations", r.parameters[0], r.parameters[1], r.iterations);

    // Conjugate gradients need no learning rate: each step is chosen by a line search.
    for beta in [ConjugateGradientBeta::FletcherReeves, ConjugateGradientBeta::PolakRibierePlus] {
        let r = NonlinearConjugateGradient::new(beta).gradient_based_minimize(&LineFit, &LineFit, vec![0.0, 0.0]).unwrap();
        println!("CG {beta:?}: a = {:.6}, b = {:.6} in {} iterations", r.parameters[0], r.parameters[1], r.iterations);
    }
}
