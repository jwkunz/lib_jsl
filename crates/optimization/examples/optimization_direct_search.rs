//! Derivative-free minimisation: Nelder-Mead and Powell's method.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_direct_search`.

use lib_jsl_optimization::nelder_mead_method::NelderMeadMethod;
use lib_jsl_optimization::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction};
use lib_jsl_optimization::powells_method::PowellsMethod;

/// The total distance from a point to three towns, minimised at the Fermat point; it is not
/// differentiable at the towns themselves, which bothers neither method.
struct Distances;
const TOWNS: [(f64, f64); 3] = [(0.0, 0.0), (4.0, 0.0), (1.0, 3.0)];
impl ObjectiveFunction for Distances {
    fn evaluate(&self, p: &[f64]) -> f64 {
        TOWNS.iter().map(|(x, y)| ((p[0] - x).powi(2) + (p[1] - y).powi(2)).sqrt()).sum()
    }
}

fn main() {
    let r = NelderMeadMethod::new().gradient_free_minimize(&Distances, vec![3.0, 3.0]).unwrap();
    println!("Nelder-Mead: {:?}, total distance {:.8}", r.parameters, r.objective_value);
    let r = PowellsMethod::new().gradient_free_minimize(&Distances, vec![3.0, 3.0]).unwrap();
    println!("Powell:      {:?}, total distance {:.8}", r.parameters, r.objective_value);

    // Bounds are handled by projection: keep the point in the box x ≥ 2.
    let mut bounded = NelderMeadMethod::new();
    bounded.set_bounds(Some(vec![(2.0, 10.0), (-10.0, 10.0)]));
    let r = bounded.gradient_free_minimize(&Distances, vec![3.0, 3.0]).unwrap();
    println!("with x ≥ 2:  {:?}, total distance {:.8}", r.parameters, r.objective_value);
}
