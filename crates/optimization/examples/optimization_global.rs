//! Global minimisation of a function with many local minima.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_global`.

use lib_jsl_optimization::differential_evolution::DifferentialEvolution;
use lib_jsl_optimization::nelder_mead_method::NelderMeadMethod;
use lib_jsl_optimization::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction};
use lib_jsl_optimization::particle_swarm::ParticleSwarm;
use lib_jsl_optimization::simulated_annealing::SimulatedAnnealing;

/// The Rastrigin function: a bowl covered in ripples, global minimum 0 at the origin and a
/// local minimum near every point of the integer lattice.
struct Rastrigin;
impl ObjectiveFunction for Rastrigin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>()
    }
}

fn main() {
    let bounds = vec![(-5.12, 5.12); 3];
    let start = vec![3.3, -2.1, 4.4];
    let local = NelderMeadMethod::new().gradient_free_minimize(&Rastrigin, start.clone()).unwrap();
    println!("Nelder-Mead (local): f = {:.4} at {:.3?}", local.objective_value, local.parameters);

    let mut sa = SimulatedAnnealing::new();
    sa.set_bounds(Some(bounds.clone()));
    sa.set_maximum_temperature(Some(3.0));
    sa.set_cooling_coefficient(Some(0.95));
    let r = sa.gradient_free_minimize(&Rastrigin, start.clone()).unwrap();
    println!("simulated annealing: f = {:.2e}", r.objective_value);

    let r = DifferentialEvolution::new(bounds.clone()).gradient_free_minimize(&Rastrigin, start.clone()).unwrap();
    println!("differential evolution: f = {:.2e} after {} generations", r.objective_value, r.iterations);

    let r = ParticleSwarm::new(bounds).gradient_free_minimize(&Rastrigin, start).unwrap();
    println!("particle swarm: f = {:.2e} after {} iterations", r.objective_value, r.iterations);
}
