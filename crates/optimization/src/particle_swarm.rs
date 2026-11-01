//! Particle swarm optimization (Kennedy and Eberhart, with Clerc's constriction coefficients).

use crate::differential_evolution::search_box;
use crate::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction, OptimizationResult};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};

/// Global-best particle swarm optimization. Each particle moves with velocity
/// `v ← w v + c₁ r₁ (p_best − x) + c₂ r₂ (g_best − x)` (`r` uniform per coordinate), using
/// Clerc and Kennedy's constriction values `w = 0.7298`, `c₁ = c₂ = 1.49618` by default.
/// Velocities are capped at the box width and positions clamped into the box (which needs
/// finite bounds). Stops when the best value has improved by less than the tolerance (default
/// 1e-12, relative) for `stall_iterations` iterations, or after the maximum (default 2000).
#[derive(Debug, Clone)]
pub struct ParticleSwarm {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
    /// Number of particles; `None` uses `max(20, 10 · dimension)`.
    pub swarm_size: Option<usize>,
    pub inertia: f64,
    pub cognitive: f64,
    pub social: f64,
    pub stall_iterations: usize,
    pub seed: u64,
}

impl ParticleSwarm {
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        ParticleSwarm {
            max_iterations: None,
            tolerance: None,
            bounds: Some(bounds),
            swarm_size: None,
            inertia: 0.7298,
            cognitive: 1.49618,
            social: 1.49618,
            stall_iterations: 100,
            seed: 0,
        }
    }
}

impl MinimizationControls for ParticleSwarm {
    fn set_max_iterations(&mut self, max_iterations: Option<usize>) {
        self.max_iterations = max_iterations;
    }
    fn get_max_iterations(&self) -> Option<usize> {
        self.max_iterations
    }
    fn set_tolerance(&mut self, tolerance: Option<f64>) {
        self.tolerance = tolerance;
    }
    fn get_tolerance(&self) -> Option<f64> {
        self.tolerance
    }
    fn set_bounds(&mut self, bounds: Option<Vec<(f64, f64)>>) {
        self.bounds = bounds;
    }
    fn get_bounds(&self) -> Option<Vec<(f64, f64)>> {
        self.bounds.clone()
    }
}

impl<T: ObjectiveFunction> GradientFreeMinimizationEngine<T> for ParticleSwarm {
    /// `initial_parameters` seeds the first particle.
    fn gradient_free_minimize(&self, objective: &T, initial_parameters: Vec<f64>) -> Result<OptimizationResult, ErrorsJSL> {
        let d = initial_parameters.len();
        let bounds = search_box(&self.bounds, d)?;
        let n = self.swarm_size.unwrap_or((10 * d).max(20)).max(2);
        let tol = self.tolerance.unwrap_or(1e-12);
        let max_iterations = self.max_iterations.unwrap_or(2000);
        let mut rng = UniformRNG::from_seed(self.seed);
        let width: Vec<f64> = bounds.iter().map(|(lo, hi)| hi - lo).collect();
        let mut x: Vec<Vec<f64>> = (0..n).map(|_| bounds.iter().map(|(lo, hi)| lo + rng.next_f64() * (hi - lo)).collect()).collect();
        x[0] = initial_parameters.iter().zip(&bounds).map(|(v, (lo, hi))| v.clamp(*lo, *hi)).collect();
        let mut v: Vec<Vec<f64>> = (0..n).map(|_| width.iter().map(|w| (2.0 * rng.next_f64() - 1.0) * 0.1 * w).collect()).collect();
        let mut personal = x.clone();
        let mut personal_value: Vec<f64> = x.iter().map(|p| objective.evaluate(p)).collect();
        let mut g = (0..n).min_by(|&i, &j| personal_value[i].total_cmp(&personal_value[j])).expect("non-empty swarm");
        let (mut best, mut best_value) = (personal[g].clone(), personal_value[g]);
        let mut stall = 0;
        let mut iteration = 0;
        let mut converged = false;
        while iteration < max_iterations {
            iteration += 1;
            for i in 0..n {
                for k in 0..d {
                    let (r1, r2) = (rng.next_f64(), rng.next_f64());
                    let vel = self.inertia * v[i][k] + self.cognitive * r1 * (personal[i][k] - x[i][k]) + self.social * r2 * (best[k] - x[i][k]);
                    v[i][k] = vel.clamp(-width[k], width[k]);
                    x[i][k] = (x[i][k] + v[i][k]).clamp(bounds[k].0, bounds[k].1);
                }
                let value = objective.evaluate(&x[i]);
                if value < personal_value[i] {
                    personal_value[i] = value;
                    personal[i] = x[i].clone();
                }
            }
            g = (0..n).min_by(|&i, &j| personal_value[i].total_cmp(&personal_value[j])).expect("non-empty swarm");
            if best_value - personal_value[g] > tol * (1.0 + best_value.abs()) {
                stall = 0;
            } else {
                stall += 1;
            }
            if personal_value[g] < best_value {
                best_value = personal_value[g];
                best = personal[g].clone();
            }
            if stall >= self.stall_iterations {
                converged = true;
                break;
            }
        }
        Ok(OptimizationResult { parameters: best, objective_value: best_value, iterations: iteration, converged })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::differential_evolution::test_functions::*;

    #[test]
    fn swarm_finds_global_minima() {
        struct Sphere;
        impl ObjectiveFunction for Sphere {
            fn evaluate(&self, x: &[f64]) -> f64 {
                x.iter().map(|v| (v - 0.5).powi(2)).sum()
            }
        }
        let r = ParticleSwarm::new(vec![(-10.0, 10.0); 10]).gradient_free_minimize(&Sphere, vec![9.0; 10]).unwrap();
        assert!(r.converged && r.objective_value < 1e-10, "{r:?}");
        let r = ParticleSwarm::new(vec![(-5.12, 5.12); 2]).gradient_free_minimize(&Rastrigin, vec![3.0, -3.0]).unwrap();
        assert!(r.objective_value < 1e-8, "{r:?}");
        let r = ParticleSwarm::new(vec![(-32.0, 32.0); 3]).gradient_free_minimize(&Ackley, vec![10.0; 3]).unwrap();
        assert!(r.objective_value < 1e-6, "{r:?}");
    }

    #[test]
    fn stays_in_bounds_and_validates() {
        let r = ParticleSwarm::new(vec![(1.0, 2.0); 3]).gradient_free_minimize(&Rosenbrock, vec![1.5; 3]).unwrap();
        assert!(r.parameters.iter().all(|v| (1.0..=2.0).contains(v)) && r.objective_value < 1e-6, "{r:?}");
        let mut p = ParticleSwarm::new(vec![(0.0, 1.0)]);
        assert!(p.gradient_free_minimize(&Rosenbrock, vec![0.5, 0.5]).is_err());
        p.set_bounds(Some(vec![(0.0, f64::INFINITY)]));
        assert!(p.gradient_free_minimize(&Rosenbrock, vec![0.5]).is_err());
    }
}
