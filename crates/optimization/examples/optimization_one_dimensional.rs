//! Golden section search, and a strong Wolfe line search along a direction.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_one_dimensional`.

use lib_jsl_optimization::golden_section_search::GoldenSectionSearch;
use lib_jsl_optimization::line_search::strong_wolfe;
use lib_jsl_optimization::optimization_traits::{GradientFreeMinimizationEngine, GradientFunction, MinimizationControls, ObjectiveFunction};

/// The cost of a cylindrical can of volume 1 as a function of its radius: area 2πr² + 2/r.
struct CanArea;
impl ObjectiveFunction for CanArea {
    fn evaluate(&self, p: &[f64]) -> f64 {
        2.0 * std::f64::consts::PI * p[0] * p[0] + 2.0 / p[0]
    }
}

/// f(x, y) = (x − 1)⁴ + (y + 2)² for the line search.
struct Bowl;
impl ObjectiveFunction for Bowl {
    fn evaluate(&self, p: &[f64]) -> f64 {
        (p[0] - 1.0).powi(4) + (p[1] + 2.0).powi(2)
    }
}
impl GradientFunction for Bowl {
    fn evaluate(&self, p: &[f64]) -> Vec<f64> {
        vec![4.0 * (p[0] - 1.0).powi(3), 2.0 * (p[1] + 2.0)]
    }
}

fn main() {
    // Golden section needs only an interval containing the minimum (given as bounds).
    let mut search = GoldenSectionSearch::new();
    search.set_bounds(Some(vec![(0.1, 2.0)]));
    search.set_tolerance(Some(1e-8));
    let r = search.gradient_free_minimize(&CanArea, vec![1.0]).unwrap();
    let exact = (1.0 / (2.0 * std::f64::consts::PI)).cbrt();
    println!("best radius {:.8} (exact {exact:.8}) after {} evaluations", r.parameters[0], r.iterations);

    // A line search along the steepest-descent direction from (0, 0).
    let x0 = [0.0, 0.0];
    let f0 = ObjectiveFunction::evaluate(&Bowl, &x0);
    let g0 = GradientFunction::evaluate(&Bowl, &x0);
    let d: Vec<f64> = g0.iter().map(|g| -g).collect();
    let point = strong_wolfe(&Bowl, &Bowl, &x0, f0, &g0, &d, 1.0, 1e-4, 0.9).unwrap();
    println!("step {:.4} to {:?}: f falls from {f0} to {:.4}", point.alpha, point.x, point.value);
}
