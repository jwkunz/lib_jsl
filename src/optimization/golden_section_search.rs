/// This file contains the implementation of the golden section search algorithm for optimization. 
/// The golden section search is a method for finding the minimum of a unimodal function by successively narrowing the range of values inside which the minimum is known to exist. 
/// It is a simple and efficient method that does not require the use of derivatives, making it suitable for optimizing functions that are not differentiable or have noisy gradients.    
/// The algorithm works by maintaining a bracket of three points (a, b, c) such that the function values at these points satisfy f(a) > f(b) < f(c). 
/// The algorithm then evaluates the function at two new points (x1 and x2) that are determined by the golden ratio, and updates the bracket based on the function values at these points. 
/// This process is repeated until the desired level of precision is achieved.  
/// The golden section search is guaranteed to converge to the minimum of a unimodal function, and it has a convergence rate of O(log(n)), where n is the number of iterations. 
/// It is a popular choice for optimization problems where the objective function is expensive to evaluate, as it requires fewer function evaluations than other methods such as grid search or random search.  
use crate::{optimization::optimization_traits::{MinimizationControls, GradientFreeMinimizationEngine, ObjectiveFunction, OptimizationResult}, prelude::ErrorsJSL};

pub struct GoldenSectionSearch {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
}

impl GoldenSectionSearch {
    pub fn new() -> Self {
        GoldenSectionSearch {
            max_iterations: None,
            tolerance: None,
            bounds: None,
        }
    }
}

impl MinimizationControls for GoldenSectionSearch {
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

impl GradientFreeMinimizationEngine for GoldenSectionSearch {
    fn gradient_free_minimize(
        &self,
        objective_function: &dyn ObjectiveFunction,
        initial_parameters: Vec<f64>,
    ) -> Result<OptimizationResult, ErrorsJSL> {
        // Here is the implementation of the golden section search algorithm. It assumes that the objective function is unimodal and that the initial parameters are a bracket of three points (a, b, c) such that f(a) > f(b) < f(c).
        let mut result = OptimizationResult {
            parameters: initial_parameters.clone(),
            objective_value: objective_function.evaluate(&initial_parameters),
            iterations: 1,
            converged: false,
        };
        if initial_parameters.len() > 1 {
            return Err(ErrorsJSL::Misconfiguration("Golden section search only supports one-dimensional optimization.  That is, the initial parameters must be a single value."));
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
        // We will use the bounds as the initial bracket for the golden section search.  The initial parameters are ignored in this implementation, but they could be used to set the initial bracket if desired.
        let mut a = bounds.as_ref().unwrap()[0].0;
        let mut b = bounds.as_ref().unwrap()[0].1;
        let tolerance = self.tolerance.unwrap_or(1e-5);
        let max_iterations = self.max_iterations.unwrap_or(1000);
        // The golden ratio is used to determine the new points to evaluate the objective function at.  It is defined as (1 + sqrt(5)) / 2.
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // The main loop of the golden section search algorithm. It continues until the desired level of precision is achieved or the maximum number of iterations is reached.
        while b-a > tolerance && result.iterations < max_iterations {
            // The new points to evaluate the objective function at are determined by the golden ratio.  The point c is closer to a than b, and the point d is closer to b than a.
            let c = b - (b - a) / phi;
            let d = a + (b - a) / phi;
            //  The function values at the new points are evaluated, and the bracket is updated based on the function values.  If f(c) < f(d), then the minimum is in the interval [a, d], so we update b to d.  Otherwise, the minimum is in the interval [c, b], so we update a to c.
            let fc = objective_function.evaluate(&vec![c]);
            let fd = objective_function.evaluate(&vec![d]);
            if fc < fd {
                b = d;
            } else {
                a = c;
            }
            // The number of iterations is incremented, and the loop continues until the desired level of precision is achieved or the maximum number of iterations is reached.
            result.iterations += 1;
        }
        // After the loop, the optimal parameters are set to the midpoint of the final bracket, and the objective value at those parameters is evaluated.  The convergence status is determined based on whether the final bracket is smaller than the specified tolerance.
        result.parameters = vec![(a + b) / 2.0];
        result.objective_value = objective_function.evaluate(&result.parameters);
        result.converged = b - a <= tolerance;
        Ok(result)
    }


}

#[cfg(test)]
mod test {
    use super::*;
    struct TestObjectiveFunction;
    impl ObjectiveFunction for TestObjectiveFunction {
        fn evaluate(&self, parameters: &Vec<f64>) -> f64 {
            let x = parameters[0];
            // This is a simple quadratic function with a minimum at x = 2.0 and a minimum value of 1.0.  It is unimodal, so it is suitable for testing the golden section search algorithm.
            (x - 2.0).powi(2) + 1.0
        }
    }
    #[test]
    fn test_golden_section_search() {
        let objective_function = TestObjectiveFunction;
        let mut optimizer = GoldenSectionSearch::new();
        optimizer.set_tolerance(Some(1e-6));
        optimizer.set_max_iterations(Some(1000));
        optimizer.set_bounds(Some(vec![(0.0, 4.0)]));   
        let result = optimizer.gradient_free_minimize(&objective_function, vec![0.0]).unwrap();
        //dbg!("Result: {:?}", &result);
        assert!((result.parameters[0] - 2.0).abs() < 1e-3);
        assert!((result.objective_value - 1.0).abs() < 1e-3);
        assert!(result.converged);
    }
}