use num::traits::bounds;

/// This file contains the implementation of the golden section search algorithm for optimization. 
/// The golden section search is a method for finding the minimum of a unimodal function by successively narrowing the range of values inside which the minimum is known to exist. 
/// It is a simple and efficient method that does not require the use of derivatives, making it suitable for optimizing functions that are not differentiable or have noisy gradients.    
/// The algorithm works by maintaining a bracket of three points (a, b, c) such that the function values at these points satisfy f(a) > f(b) < f(c). 
/// The algorithm then evaluates the function at two new points (x1 and x2) that are determined by the golden ratio, and updates the bracket based on the function values at these points. 
/// This process is repeated until the desired level of precision is achieved.  
/// The golden section search is guaranteed to converge to the minimum of a unimodal function, and it has a convergence rate of O(log(n)), where n is the number of iterations. 
/// It is a popular choice for optimization problems where the objective function is expensive to evaluate, as it requires fewer function evaluations than other methods such as grid search or random search.  
use crate::{optimization::optimization_traits::{MinimizationEngine, ObjectiveFunction, OptimizationResult}, prelude::ErrorsJSL};

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
impl  MinimizationEngine for GoldenSectionSearch {
    fn minimize(
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
        if bounds.is_none(){
            return Err(ErrorsJSL::Misconfiguration("Golden section search requires bounds to be set.  Please set bounds using the set_bounds method before calling minimize."));
        }
        let mut a = bounds.as_ref().unwrap()[0].0;
        let mut b = bounds.as_ref().unwrap()[0].1;
        let tolerance = self.tolerance.unwrap_or(1e-5);
        let max_iterations = self.max_iterations.unwrap_or(1000);
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        while b-a > tolerance && result.iterations < max_iterations {
            let c = b - (b - a) / phi;
            let d = a + (b - a) / phi;
            let fc = objective_function.evaluate(&vec![c]);
            let fd = objective_function.evaluate(&vec![d]);
            if fc < fd {
                b = d;
            } else {
                a = c;
            }
            result.iterations += 1;
        }
        result.parameters = vec![(a + b) / 2.0];
        result.objective_value = objective_function.evaluate(&result.parameters);
        result.converged = b - a <= tolerance;
        Ok(result)
    }

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

#[cfg(test)]
mod test {
    use super::*;
    struct TestObjectiveFunction;
    impl ObjectiveFunction for TestObjectiveFunction {
        fn evaluate(&self, parameters: &Vec<f64>) -> f64 {
            let x = parameters[0];
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
        let result = optimizer.minimize(&objective_function, vec![0.0]).unwrap();
        //dbg!("Result: {:?}", &result);
        assert!((result.parameters[0] - 2.0).abs() < 1e-3);
        assert!((result.objective_value - 1.0).abs() < 1e-3);
        assert!(result.converged);
    }
}