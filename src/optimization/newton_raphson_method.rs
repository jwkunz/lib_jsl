/// This file implements the Newton-Raphson method for optimization. The Newton-Raphson method is an iterative optimization algorithm used to find the minimum of a function.
/// The algorithm works by using the second-order Taylor expansion of the objective function to find a search direction and step size that will lead to a local minimum.
/// The algorithm continues until it reaches a point where the gradient is close to zero, indicating that it has found a local minimum.
/// The Newton-Raphson method can converge faster than gradient descent, but it requires the computation of the Hessian matrix, which can be computationally expensive for high-dimensional problems.       
use crate::{
    optimization::optimization_traits::{
        GradientFunction, HessianBasedMinimizationEngine, HessianFunction, MinimizationControls,
        ObjectiveFunction, OptimizationResult,
    },
    prelude::ErrorsJSL,
};
use ndarray_linalg::Inverse;
pub struct NewtonRaphsonMethod {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
}

impl MinimizationControls for NewtonRaphsonMethod {
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

impl<T, U, V> HessianBasedMinimizationEngine<T, U, V> for NewtonRaphsonMethod 
where T: ObjectiveFunction, U: GradientFunction, V: HessianFunction {
    fn hessian_based_minimize(
        &self,
        objective_function: &T,
        gradient_function: &U,
        hessian_function: &V,
        initial_parameters: Vec<f64>,
    ) -> Result<OptimizationResult, ErrorsJSL> {
        // Here is the implementation of the Newton-Raphson method. It assumes that the objective function is twice differentiable and that the initial parameters are a valid starting point for the optimization.
        let mut result = OptimizationResult {
            parameters: initial_parameters.clone(),
            objective_value: objective_function.evaluate(&initial_parameters),
            iterations: 0,
            converged: false,
        };
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
        // The main loop of the Newton-Raphson method. It continues until the desired level of precision is achieved or the maximum number of iterations is reached.
        while result.iterations < self.get_max_iterations().unwrap_or(usize::MAX) {
            // The gradient and Hessian of the objective function at the current parameters are evaluated, and the parameters are updated by taking a step in the direction of the negative gradient, scaled by the inverse of the Hessian. If bounds are set, the updated parameters are projected back into the feasible region defined by the bounds.
            let gradient = gradient_function.evaluate(&result.parameters);
            let hessian = hessian_function.evaluate(&result.parameters);
            let hessian_matrix = ndarray::Array2::from_shape_vec(
                (hessian.len(), hessian[0].len()),
                hessian.into_iter().flatten().collect(),
            )
            .unwrap();
            let hessian_inv = hessian_matrix.inv().unwrap();
            let step = hessian_inv
                .dot(&ndarray::Array1::from_vec(gradient))
                .mapv(|x| -x);
            let mut new_parameters = result
                .parameters
                .iter()
                .zip(step.iter())
                .map(|(p, s)| p + s)
                .collect::<Vec<f64>>();
            if let Some(bounds) = bounds.as_ref() {
                new_parameters = new_parameters
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let (lower, upper) = bounds[i];
                        if *p < lower {
                            lower
                        } else if *p > upper {
                            upper
                        } else {
                            *p
                        }
                    })
                    .collect();
            }
            let new_objective_value = objective_function.evaluate(&new_parameters);
            if (result.objective_value - new_objective_value).abs()
                < self.get_tolerance().unwrap_or(1e-6)
            {
                result.converged = true;
                result.parameters = new_parameters;
                result.objective_value = new_objective_value;
                break;
            }
            result.parameters = new_parameters;
            result.objective_value = new_objective_value;
            result.iterations += 1;
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::optimization_traits::ObjectiveFunctionWithGradient;
    #[test]
    fn test_newton_raphson_method() {
        // This is a simple test case for the Newton-Raphson method. It uses a simple quadratic function with a minimum at x = 2.0 and a minimum value of 1.0. The algorithm should be able to find the minimum within a reasonable number of iterations.
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
        impl HessianFunction for TestObjectiveFunction {
            fn evaluate(&self, _: &Vec<f64>) -> Vec<Vec<f64>> {
                vec![vec![2.0]]
            }
        }
        impl ObjectiveFunctionWithGradient for TestObjectiveFunction {}
        let objective_function = TestObjectiveFunction;
        let optimizer = NewtonRaphsonMethod {
            max_iterations: Some(100),
            tolerance: Some(1e-6),
            bounds: Some(vec![(0.0, 4.0)]),
        };
        let initial_parameters = vec![0.0];
        let result = optimizer
            .hessian_based_minimize(
                &objective_function,
                &objective_function,
                &objective_function,
                initial_parameters,
            )
            .unwrap();
        //dbg!("Result: {:?}", &result);
        assert!(result.converged);
        assert!((result.parameters[0] - 2.0).abs() < 1e-3);
        assert!((result.objective_value - 1.0).abs() < 1e-3);
    }
}
