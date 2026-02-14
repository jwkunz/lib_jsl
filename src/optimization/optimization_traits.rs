use crate::prelude::ErrorsJSL;

/// Here are traits that are used in optimization algorithms. They are defined here to avoid circular dependencies between the optimization algorithms and the traits they use.

/// - `ObjectiveFunction`: A trait for objective functions that can be optimized. It has a method `evaluate` that takes a vector of parameters and returns a single scalar value of the objective function at those parameters.
pub trait ObjectiveFunction {
    fn evaluate(&self, parameters: &Vec<f64>) -> f64;
}

/// - `GradientFunction`: A trait for gradient functions that can be used in optimization algorithms. It has a method `evaluate` that takes a vector of parameters and returns a vector of the same length representing the gradient of the objective function at those parameters.
pub trait GradientFunction {
    fn evaluate(&self, parameters: &Vec<f64>) -> Vec<f64>;
}

/// - `ObjectiveFunctionWithGradient`: A trait for objective functions that also have a gradient. It is a marker trait that indicates that the objective function implements both `ObjectiveFunction` and `GradientFunction`.
pub trait ObjectiveFunctionWithGradient: ObjectiveFunction + GradientFunction {}

/// - `OptimizationResult`: A struct that can be used to store the results of an optimization algorithm, including the optimal parameters, the value of the objective function at those parameters, the number of iterations taken to converge, and whether the algorithm converged or not.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub parameters: Vec<f64>,
    pub objective_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// - `MinimizationEngine`: A trait for optimization algorithms that can minimize an objective function. It has a method `minimize` that takes an objective function and an initial guess for the parameters, and returns the parameters that minimize the objective function. It also has methods for setting optional parameters such as maximum iterations, tolerance for convergence, and bounds for the optimization.
pub trait MinimizationEngine {
    /// The main method of the optimization engine. It takes an objective function and an initial guess for the parameters, and returns the parameters that minimize the objective function.
    fn minimize(
        &self,
        objective_function: &dyn ObjectiveFunction,
        initial_parameters: Vec<f64>,
    ) -> Result<OptimizationResult, ErrorsJSL>;
    /// These traits use options to allow for algorithms to set and get the convergence criteria.
    /// If the algorithm does not have a maximum number of iterations, this can be set to None.
    fn set_max_iterations(&mut self, max_iterations: Option<usize>);
    fn get_max_iterations(&self) -> Option<usize>;
    /// If the algorithm does not have a tolerance for convergence, this can be set to None.
    fn set_tolerance(&mut self, tolerance: Option<f64>);
    fn get_tolerance(&self) -> Option<f64>;
    /// This function can be used to set and get the optional bounds for the optimization. If the algorithm does not support bounds, this can be set to None.
    /// The bounds are represented as a vector of tuples, where each tuple contains the lower and upper bound for the corresponding parameter. If a parameter is unbounded, its corresponding tuple can be set to (f64::NEG_INFINITY, f64::INFINITY).
    fn set_bounds(&mut self, bounds: Option<Vec<(f64, f64)>>);
    fn get_bounds(&self) -> Option<Vec<(f64, f64)>>;
}
