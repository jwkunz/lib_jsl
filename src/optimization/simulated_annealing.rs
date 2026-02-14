use rand_distr::{Distribution, Uniform};

use crate::{optimization::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction, OptimizationResult}, prelude::ErrorsJSL};

/// This file implements simulated annealing for optimization. Simulated annealing is a probabilistic optimization algorithm that is inspired by the process of annealing in metallurgy, where a material is heated and then slowly cooled to allow it to reach a state of minimum energy. In optimization, simulated annealing is used to find the minimum of an objective function by allowing for occasional uphill moves, which can help the algorithm escape local minima and explore the search space more effectively. The algorithm works by starting with an initial solution and then iteratively generating new candidate solutions by making small random changes to the current solution. The new solution is accepted with a probability that depends on the difference in objective function values between the current and new solutions, as well as a temperature parameter that controls the likelihood of accepting worse solutions. As the algorithm progresses, the temperature is gradually decreased, which reduces the probability of accepting worse solutions and allows the algorithm to converge to a minimum. Simulated annealing can be effective for optimizing complex, non-convex functions, but it can also be computationally expensive and may require careful tuning of the temperature schedule and other hyperparameters to achieve good performance.    

pub struct SimulatedAnnealing {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
    /// These are additional parameters specific to simulated annealing that can be set and get using the methods defined in this struct. The cooling coefficient controls how quickly the temperature decreases, while the minimum and maximum temperature define the range of temperatures that the algorithm will use during the optimization process.
    cooling_coefficient: Option<f64>,
    /// The minimum temperature is the lowest temperature that the algorithm will use during the optimization process. 
    /// It can be set to a small positive value to ensure that the algorithm continues to explore the search space even at low temperatures, or it can be set to zero to allow for a complete cooling schedule.
    minimum_temperature: Option<f64>,
    /// The maximum temperature is the highest temperature that the algorithm will use during the optimization process. 
    /// It can be set to a large value to allow for a wide exploration of the search space at the beginning of the optimization, or it can be set to a smaller value to focus the search on a narrower region of the search space.
    maximum_temperature: Option<f64>,
    /// Interations per temperature is the number of iterations that the algorithm will perform at each temperature level before cooling down. It can be set to a fixed value or it can be dynamically adjusted based on the progress of the optimization.
    iterations_per_temperature: Option<usize>,
}

impl SimulatedAnnealing {
    pub fn new() -> Self {
        SimulatedAnnealing {
            max_iterations: None,
            tolerance: None,
            bounds: None,
            cooling_coefficient:None,
            minimum_temperature: None,
            maximum_temperature: None,
            iterations_per_temperature: None,
        }
    }

    pub fn set_cooling_coefficient(&mut self, cooling_coefficient: Option<f64>) {
        self.cooling_coefficient = cooling_coefficient;
    }
    pub fn get_cooling_coefficient(&self) -> Option<f64> {
        self.cooling_coefficient
    }
    pub fn set_minimum_temperature(&mut self, minimum_temperature: Option<f64>) {
        self.minimum_temperature = minimum_temperature;
    }
    pub fn get_minimum_temperature(&self) -> Option<f64> {
        self.minimum_temperature
    }
    pub fn set_maximum_temperature(&mut self, maximum_temperature: Option<f64>) {
        self.maximum_temperature = maximum_temperature;
    }
    pub fn get_maximum_temperature(&self) -> Option<f64> {
        self.maximum_temperature
    }
    pub fn set_iterations_per_temperature(&mut self, iterations_per_temperature: Option<usize>) {
        self.iterations_per_temperature = iterations_per_temperature;
    }
    pub fn get_iterations_per_temperature(&self) -> Option<usize> {
        self.iterations_per_temperature
    }
}

impl MinimizationControls for SimulatedAnnealing {
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

fn generate_candidate_solution(current_parameters: &Vec<f64>, current_temperature: f64, rng: &mut rand::rngs::ThreadRng) -> Vec<f64> {
    current_parameters
        .iter()
        .map(|&param| {
            let perturbation = rand_distr::Normal::new(0.0, current_temperature).unwrap().sample(rng);
            param + perturbation
        })
        .collect()
}

fn project_to_bounds(parameters: &Vec<f64>, bounds: &Vec<(f64, f64)>) -> Vec<f64> {
    parameters
        .iter()
        .zip(bounds.iter())
        .map(|(&param, &(low, high))| {
            if param < low {
                low
            } else if param > high {
                high
            } else {
                param
            }
        })
        .collect()
}

impl GradientFreeMinimizationEngine for SimulatedAnnealing {
    fn gradient_free_minimize(
        &self,
        objective_function: &dyn ObjectiveFunction,
        initial_parameters: Vec<f64>,
    ) -> Result<OptimizationResult, ErrorsJSL> {
        let mut current_temperature = self.get_maximum_temperature().unwrap_or(1.0);
        let cooling_coefficient = self.get_cooling_coefficient().unwrap_or(0.9);
        let minimum_temperature = self.get_minimum_temperature().unwrap_or(1e-5);
        let maximum_steps = self.get_max_iterations().unwrap_or(usize::MAX);
        let iterations_per_temperature = self.get_iterations_per_temperature().unwrap_or(100);
        let bounds = self.get_bounds();
        if let Some(bounds) = bounds.as_ref() {
            if bounds.len() != initial_parameters.len() {
                return Err(ErrorsJSL::Misconfiguration(
                    "Bounds length must match the number of parameters.",
                ));
            }            
            if bounds.iter().any(|(lower, upper)| lower > upper) {
                return Err(ErrorsJSL::Misconfiguration(
                    "Lower bound must be less than or equal to upper bound.",
                ));
            }
        }
        let mut rng = rand::rng();
        let mut current_parameters = initial_parameters.clone();
        let mut current_objective_value = objective_function.evaluate(&current_parameters);
        let mut iteration_counter = 0;
        let mut result = OptimizationResult {
            parameters: current_parameters.clone(),
            objective_value: current_objective_value,
            iterations: iteration_counter,
            converged: false,
        };
        if initial_parameters.len() == 0 {
            return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
        }
        while current_temperature > minimum_temperature && iteration_counter < maximum_steps {
            // Here is the implementation of the simulated annealing algorithm. It assumes that the objective function is defined and that the initial parameters are a valid starting point for the optimization.
            // The algorithm generates a new candidate solution by making a small random change to the current parameters, and then decides whether to accept the new solution based on the difference in objective function values and the current temperature. The temperature is then updated according to the cooling schedule defined by the cooling coefficient.
            // If bounds are set, the new candidate solution is projected back into the feasible region defined by the bounds before evaluating the objective function.
            // The algorithm continues until the temperature drops below the minimum temperature or the maximum number of iterations is reached, at which point it returns the best solution found.
            for _ in 0..iterations_per_temperature {
                // Generate a new candidate solution by making a small random change to the current parameters. This can be done by adding a random perturbation to each parameter, where the magnitude of the perturbation is proportional to the current temperature. The new candidate solution is then projected back into the feasible region defined by the bounds, if bounds are set.
                // Evaluate the objective function at the new candidate solution and compare it to the current solution. If the new solution has a lower objective function value, it is accepted as the new current solution. If the new solution has a higher objective function value, it is accepted with a probability that depends on the difference in objective function values and the current temperature. This allows the algorithm to occasionally accept worse solutions, which can help it escape local minima and explore the search space more effectively.
                // Update the temperature according to the cooling schedule defined by the cooling coefficient. This typically involves multiplying the current temperature by the cooling coefficient, which reduces the temperature over time and allows the algorithm to converge to a minimum.
                let mut candidate_parameters = generate_candidate_solution(&current_parameters, current_temperature, &mut rng);
                if let Some(b) = bounds.as_ref() {
                    candidate_parameters = project_to_bounds(&candidate_parameters, &b); 
                }
                let candidate_objective_value = objective_function.evaluate(&candidate_parameters);
                // Always keep track of the best solution found so far, even if it is not accepted as the current solution. This allows the algorithm to return the best solution found at the end of the optimization process, rather than just the last solution that was accepted.
                if candidate_objective_value < result.objective_value {
                    result.parameters = candidate_parameters.clone();
                    result.objective_value = candidate_objective_value;
                }
                // Decide whether to accept the new candidate solution based on the difference in objective function values and the current temperature. If the new solution has a lower objective function value, it is accepted as the new current solution. If the new solution has a higher objective function value, it is accepted with a probability that depends on the difference in objective function values and the current temperature. This allows the algorithm to occasionally accept worse solutions, which can help it escape local minima and explore the search space more effectively.
                if candidate_objective_value < current_objective_value {
                    current_parameters = candidate_parameters;
                    current_objective_value = candidate_objective_value;
                } else {
                    let delta = candidate_objective_value - current_objective_value;
                    let acceptance_probability = (-delta / current_temperature).exp();
                    let sample = Uniform::new(0.0, 1.0).unwrap().sample(&mut rng);
                    if sample < acceptance_probability {
                        current_parameters = candidate_parameters;
                        current_objective_value = candidate_objective_value;
                    }   
                }
                iteration_counter+=1;
            }
            current_temperature *= cooling_coefficient;
        }
        result.converged = current_temperature <= minimum_temperature;
        result.iterations = iteration_counter;
        Ok(result)
    }
}   

#[cfg(test)]
mod tests {
    use super::*;
    struct TestObjectiveFunction;
    impl ObjectiveFunction for TestObjectiveFunction {
        fn evaluate(&self, parameters: &Vec<f64>) -> f64 {
            // This is a simple test objective function that has a minimum at (2.0, 3.0) with a value of 1.0. It is defined as f(x, y) = (x - 2)^2 + (y - 3)^2 + 1.
            let x = parameters[0];
            let y = parameters[1];
            (x - 2.0).powi(2) + (y - 3.0).powi(2) + 1.0
        }
    }
    #[test]
    fn test_simulated_annealing() {
        let objective_function = TestObjectiveFunction;
        let mut optimizer = SimulatedAnnealing::new();
        optimizer.set_cooling_coefficient(Some(0.9));
        optimizer.set_minimum_temperature(Some(1E-16));
        optimizer.set_maximum_temperature(Some(10.0));
        optimizer.set_iterations_per_temperature(Some(100));            
        optimizer.set_tolerance(Some(1e-6));
        let initial_parameters = vec![0.0, 0.0];
        let result = optimizer.gradient_free_minimize(&objective_function, initial_parameters).unwrap();
        //dbg!("Result: {:?}", &result);
        assert!(result.converged);
        assert!((result.parameters[0] - 2.0).abs() < 1e-1);
        assert!((result.parameters[1] - 3.0).abs() < 1e-1);
        assert!((result.objective_value - 1.0).abs() < 1e-1);   
    }
}