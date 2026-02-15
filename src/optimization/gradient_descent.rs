/// This file contains the implementation of the gradient descent algorithm for optimization.
/// Gradient descent is an iterative optimization algorithm used to find the minimum of a function.
/// The algorithm works by taking steps proportional to the negative of the gradient of the function at the current point. The size of the steps is determined by a learning rate, which is a hyperparameter that controls how quickly the algorithm converges to the minimum. 
/// The algorithm continues until it reaches a point where the gradient is close to zero, indicating that it has found a local minimum. 

use crate::{optimization::optimization_traits::{GradientBasedMinimizationEngine, GradientFunction, MinimizationControls, ObjectiveFunction, OptimizationResult}, prelude::ErrorsJSL};

pub struct GradientDescent {
    learning_rate: f64,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        GradientDescent {
            learning_rate,
            max_iterations: None,
            tolerance: None,
            bounds: None,
        }
    }
}

impl MinimizationControls for GradientDescent {
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

impl<T, U> GradientBasedMinimizationEngine<T, U> for GradientDescent 
where T: ObjectiveFunction, U: GradientFunction {
    fn gradient_based_minimize(
            &self,
            objective_function: &T,
            gradient_function: &U,
            initial_parameters: Vec<f64>,
        ) -> Result<OptimizationResult, ErrorsJSL> {
        if initial_parameters.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
        }

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

        // Here is the implementation of the gradient descent algorithm. It assumes that the objective function is differentiable and that the initial parameters are a valid starting point for the optimization.
        let mut result = OptimizationResult {
            parameters: initial_parameters.clone(),
            objective_value: objective_function.evaluate(&initial_parameters),
            iterations: 0,
            converged: false,
        };
        let max_iterations = self.max_iterations.unwrap_or(1000);
        let tolerance = self.tolerance.unwrap_or(1e-5);
        // The main loop of the gradient descent algorithm. It continues until the desired level of precision is achieved or the maximum number of iterations is reached.
        while result.iterations < max_iterations {
            // The gradient of the objective function at the current parameters is evaluated, and the parameters are updated by taking a step in the direction of the negative gradient, scaled by the learning rate. If bounds are set, the updated parameters are projected back into the feasible region defined by the bounds.
            let gradient = gradient_function.evaluate(&result.parameters);
            if gradient.len() != result.parameters.len() {
                return Err(ErrorsJSL::Misconfiguration(
                    "Gradient length must match the number of parameters.",
                ));
            }
            let mut new_parameters: Vec<f64> = result.parameters.iter().zip(gradient.iter()).map(|(p, g)| p - self.learning_rate * g).collect();
            if let Some(bounds) = bounds.as_ref() {
                new_parameters = new_parameters.iter().enumerate().map(|(i, p)| {
                    let (lower_bound, upper_bound) = bounds[i];
                    p.max(lower_bound).min(upper_bound)
                }).collect();
            }
            // The objective value at the new parameters is evaluated, and the convergence status is determined based on whether the change in objective value is smaller than the specified tolerance.
            let new_objective_value = objective_function.evaluate(&new_parameters);
            result.iterations += 1;
            if (result.objective_value - new_objective_value).abs() < tolerance {
                result.converged = true;
                result.parameters = new_parameters;
                result.objective_value = new_objective_value;
                break;
            }
            result.parameters = new_parameters;
            result.objective_value = new_objective_value;
        }
        Ok(result)
    }
}       

#[cfg(test)]
mod tests {
    use crate::optimization::optimization_traits::ObjectiveFunctionWithGradient;

    use super::*;
    #[test]
    fn test_gradient_descent() {
        // This is a simple test case for the gradient descent algorithm. It uses a simple quadratic function with a minimum at x = 2.0 and a minimum value of 1.0. The algorithm should be able to find the minimum within a reasonable number of iterations.
        struct TestObjectiveFunction;
        impl ObjectiveFunction for TestObjectiveFunction {
            fn evaluate(&self, parameters: &Vec<f64>) -> f64 {
                let x = parameters[0];
                (x - 2.0).powi(2) + 1.0
            }
        }
        impl GradientFunction for TestObjectiveFunction {
            fn evaluate(&self, parameters: &Vec<f64>) -> Vec<f64> {
                let x = parameters[0];
                vec![2.0 * (x - 2.0)]
            }
        }
        impl ObjectiveFunctionWithGradient for TestObjectiveFunction {}
        let objective_function = TestObjectiveFunction;
        let mut optimizer = GradientDescent::new(1E-2);
        optimizer.set_max_iterations(Some(10000));
        optimizer.set_tolerance(Some(1e-6));
        let initial_parameters = vec![0.0];
        let result = optimizer.gradient_based_minimize(&objective_function, &objective_function, initial_parameters).unwrap();
        //dbg!("Result: {:?}", &result);
        assert!(result.converged);
        assert!((result.parameters[0] - 2.0).abs() < 1e-1);
        assert!((result.objective_value - 1.0).abs() < 1e-1);
    }

    #[test]
    fn test_gradient_descent_rejects_empty_initial_parameters() {
        struct PanickingObjective;
        impl ObjectiveFunction for PanickingObjective {
            fn evaluate(&self, _: &Vec<f64>) -> f64 {
                panic!("objective should not be evaluated when initial parameters are empty");
            }
        }
        impl GradientFunction for PanickingObjective {
            fn evaluate(&self, _: &Vec<f64>) -> Vec<f64> {
                vec![]
            }
        }
        impl ObjectiveFunctionWithGradient for PanickingObjective {}

        let optimizer = GradientDescent::new(1e-2);
        let result = optimizer.gradient_based_minimize(&PanickingObjective, &PanickingObjective, vec![]);
        assert!(matches!(result, Err(ErrorsJSL::Misconfiguration(_))));
    }

    #[test]
    fn test_gradient_descent_rejects_bounds_dimension_mismatch() {
        struct Objective;
        impl ObjectiveFunction for Objective {
            fn evaluate(&self, parameters: &Vec<f64>) -> f64 {
                parameters.iter().map(|x| x * x).sum()
            }
        }
        impl GradientFunction for Objective {
            fn evaluate(&self, parameters: &Vec<f64>) -> Vec<f64> {
                parameters.iter().map(|x| 2.0 * x).collect()
            }
        }
        impl ObjectiveFunctionWithGradient for Objective {}

        let mut optimizer = GradientDescent::new(1e-2);
        optimizer.set_bounds(Some(vec![(-1.0, 1.0)]));
        let result = optimizer.gradient_based_minimize(&Objective, &Objective, vec![1.0, 2.0]);
        assert!(matches!(result, Err(ErrorsJSL::Misconfiguration(_))));
    }

    #[test]
    fn test_gradient_descent_rejects_gradient_dimension_mismatch() {
        struct Objective;
        impl ObjectiveFunction for Objective {
            fn evaluate(&self, parameters: &Vec<f64>) -> f64 {
                parameters.iter().map(|x| x * x).sum()
            }
        }
        struct BadGradient;
        impl GradientFunction for BadGradient {
            fn evaluate(&self, _: &Vec<f64>) -> Vec<f64> {
                vec![1.0]
            }
        }

        let optimizer = GradientDescent::new(1e-2);
        let result = optimizer.gradient_based_minimize(&Objective, &BadGradient, vec![1.0, 2.0]);
        assert!(matches!(result, Err(ErrorsJSL::Misconfiguration(_))));
    }
}   
