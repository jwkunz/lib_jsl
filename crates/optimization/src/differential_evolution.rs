//! Differential evolution (Storn and Price), a population-based global minimizer.

use crate::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction, OptimizationResult};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};

/// Differential evolution, strategy DE/rand/1/bin: each target vector is challenged by the trial
/// `a + F (b − c)` (three other random members), crossed over coordinate-wise with probability
/// `CR`, and replaced if the trial is no worse. Needs bounds (the initial population is uniform
/// in the box, and trial coordinates leaving it are reset between the parent and the bound).
/// Stops when the population's objective values agree to the tolerance (default 1e-12) or
/// after the maximum number of generations (default 1000).
#[derive(Debug, Clone)]
pub struct DifferentialEvolution {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
    /// Population size; `None` uses `max(15, 10 · dimension)`.
    pub population_size: Option<usize>,
    /// Differential weight F (default 0.8).
    pub differential_weight: f64,
    /// Crossover probability CR (default 0.9).
    pub crossover_probability: f64,
    pub seed: u64,
}

impl DifferentialEvolution {
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        DifferentialEvolution {
            max_iterations: None,
            tolerance: None,
            bounds: Some(bounds),
            population_size: None,
            differential_weight: 0.8,
            crossover_probability: 0.9,
            seed: 0,
        }
    }
}

impl MinimizationControls for DifferentialEvolution {
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

/// Validates a search box for the population methods.
pub(crate) fn search_box(bounds: &Option<Vec<(f64, f64)>>, dimension: usize) -> Result<Vec<(f64, f64)>, ErrorsJSL> {
    let b = bounds.clone().ok_or(ErrorsJSL::Misconfiguration("Population methods need finite bounds."))?;
    if b.len() != dimension || dimension == 0 {
        return Err(ErrorsJSL::Misconfiguration("Bounds length must match the number of parameters."));
    }
    if b.iter().any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi) {
        return Err(ErrorsJSL::Misconfiguration("Bounds must be finite with lower <= upper."));
    }
    Ok(b)
}

impl<T: ObjectiveFunction> GradientFreeMinimizationEngine<T> for DifferentialEvolution {
    /// `initial_parameters` joins the random initial population (so a good guess is kept).
    fn gradient_free_minimize(&self, objective: &T, initial_parameters: Vec<f64>) -> Result<OptimizationResult, ErrorsJSL> {
        let d = initial_parameters.len();
        let bounds = search_box(&self.bounds, d)?;
        let np = self.population_size.unwrap_or((10 * d).max(15)).max(4);
        let tol = self.tolerance.unwrap_or(1e-12);
        let generations = self.max_iterations.unwrap_or(1000);
        let mut rng = UniformRNG::from_seed(self.seed);
        let mut population: Vec<Vec<f64>> = (0..np).map(|_| bounds.iter().map(|(lo, hi)| lo + rng.next_f64() * (hi - lo)).collect()).collect();
        population[0] = initial_parameters.iter().zip(&bounds).map(|(v, (lo, hi))| v.clamp(*lo, *hi)).collect();
        let mut values: Vec<f64> = population.iter().map(|p| objective.evaluate(p)).collect();
        let pick = |rng: &mut UniformRNG, exclude: &[usize]| loop {
            let k = (rng.next_f64() * np as f64) as usize % np;
            if !exclude.contains(&k) {
                break k;
            }
        };
        let mut generation = 0;
        let mut converged = false;
        while generation < generations {
            generation += 1;
            for i in 0..np {
                let a = pick(&mut rng, &[i]);
                let b = pick(&mut rng, &[i, a]);
                let c = pick(&mut rng, &[i, a, b]);
                let forced = (rng.next_f64() * d as f64) as usize % d;
                let trial: Vec<f64> = (0..d)
                    .map(|j| {
                        if j == forced || rng.next_f64() < self.crossover_probability {
                            let v = population[a][j] + self.differential_weight * (population[b][j] - population[c][j]);
                            let (lo, hi) = bounds[j];
                            // Out of the box: move to a random point between the parent and the bound.
                            if v < lo {
                                lo + rng.next_f64() * (population[i][j] - lo)
                            } else if v > hi {
                                hi - rng.next_f64() * (hi - population[i][j])
                            } else {
                                v
                            }
                        } else {
                            population[i][j]
                        }
                    })
                    .collect();
                let value = objective.evaluate(&trial);
                if value <= values[i] {
                    population[i] = trial;
                    values[i] = value;
                }
            }
            let (best, worst) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            if worst - best <= tol * (1.0 + best.abs()) {
                converged = true;
                break;
            }
        }
        let best = (0..np).min_by(|&i, &j| values[i].total_cmp(&values[j])).expect("non-empty population");
        Ok(OptimizationResult { parameters: population[best].clone(), objective_value: values[best], iterations: generation, converged })
    }
}

#[cfg(test)]
pub(crate) mod test_functions {
    use crate::optimization_traits::ObjectiveFunction;
    use std::f64::consts::PI;

    /// Rastrigin: global minimum 0 at the origin among ~10ᵈ local minima.
    pub struct Rastrigin;
    impl ObjectiveFunction for Rastrigin {
        fn evaluate(&self, x: &[f64]) -> f64 {
            10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>()
        }
    }

    /// Ackley: global minimum 0 at the origin.
    pub struct Ackley;
    impl ObjectiveFunction for Ackley {
        fn evaluate(&self, x: &[f64]) -> f64 {
            let n = x.len() as f64;
            let s1 = x.iter().map(|v| v * v).sum::<f64>() / n;
            let s2 = x.iter().map(|v| (2.0 * PI * v).cos()).sum::<f64>() / n;
            -20.0 * (-0.2 * s1.sqrt()).exp() - s2.exp() + 20.0 + std::f64::consts::E
        }
    }

    pub struct Rosenbrock;
    impl ObjectiveFunction for Rosenbrock {
        fn evaluate(&self, p: &[f64]) -> f64 {
            p.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_functions::*;
    use super::*;

    #[test]
    fn finds_global_minima_of_multimodal_functions() {
        let de = DifferentialEvolution::new(vec![(-5.12, 5.12); 5]);
        let r = de.gradient_free_minimize(&Rastrigin, vec![4.0; 5]).unwrap();
        assert!(r.objective_value < 1e-8 && r.parameters.iter().all(|v| v.abs() < 1e-4), "{r:?}");
        let de = DifferentialEvolution::new(vec![(-32.0, 32.0); 3]);
        let r = de.gradient_free_minimize(&Ackley, vec![20.0; 3]).unwrap();
        assert!(r.objective_value < 1e-6, "{r:?}");
        let mut de = DifferentialEvolution::new(vec![(-2.0, 2.0); 4]);
        de.set_max_iterations(Some(3000));
        let r = de.gradient_free_minimize(&Rosenbrock, vec![0.0; 4]).unwrap();
        assert!(r.parameters.iter().all(|v| (v - 1.0).abs() < 1e-3), "{r:?}");
    }

    #[test]
    fn respects_bounds_and_validates() {
        // The unconstrained minimum (3) lies outside the box; the bound is active.
        struct Shifted;
        impl ObjectiveFunction for Shifted {
            fn evaluate(&self, x: &[f64]) -> f64 {
                (x[0] - 3.0).powi(2) + x[1] * x[1]
            }
        }
        let de = DifferentialEvolution::new(vec![(-1.0, 1.0), (-1.0, 1.0)]);
        let r = de.gradient_free_minimize(&Shifted, vec![0.0, 0.0]).unwrap();
        assert!((r.parameters[0] - 1.0).abs() < 1e-6 && r.parameters.iter().all(|v| (-1.0..=1.0).contains(v)));
        let mut bad = DifferentialEvolution::new(vec![(0.0, 1.0)]);
        assert!(bad.gradient_free_minimize(&Shifted, vec![0.0, 0.0]).is_err());
        bad.set_bounds(None);
        assert!(bad.gradient_free_minimize(&Shifted, vec![0.0]).is_err());
    }
}
