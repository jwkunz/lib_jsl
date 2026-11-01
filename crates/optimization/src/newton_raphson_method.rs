//! Newton's method for minimization: each step solves `∇²f(x) p = −∇f(x)` and moves to `x + p`
//! (projected onto the bounds, if any). It converges quadratically near a minimum with a positive
//! definite Hessian, but it is the pure method, with no line search or trust region: from a poor
//! start, or where the Hessian is indefinite, it can move uphill or towards a saddle point. For
//! robust minimization prefer [`Bfgs`](crate::quasi_newton::Bfgs).
//!
//! See the book entry [Newton and quasi-Newton methods](https://jwkunz.github.io/lib_jsl/book/optimization/newton-quasi-newton.html#newtons-method).
use crate::{
    optimization_traits::{
        GradientFunction, HessianBasedMinimizationEngine, HessianFunction, MinimizationControls,
        ObjectiveFunction, OptimizationResult,
    },
};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::Solve;

/// Newton's method with an analytic Hessian. Stops when the objective changes by less than the
/// tolerance (default 1e-6) between iterations, or after `max_iterations` (default 100).
#[derive(Debug, Clone, Default)]
pub struct NewtonRaphsonMethod {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
}

impl NewtonRaphsonMethod {
    pub fn new() -> Self {
        Self::default()
    }
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
        let n = initial_parameters.len();
        if n == 0 {
            return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
        }
        let mut result = OptimizationResult {
            parameters: initial_parameters.clone(),
            objective_value: objective_function.evaluate(&initial_parameters),
            iterations: 0,
            converged: false,
        };
        let bounds = self.get_bounds();
        if let Some(bounds) = bounds.as_ref() {
            if bounds.len() != n {
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
        let max_iterations = self.get_max_iterations().unwrap_or(100);
        while result.iterations < max_iterations {
            // ANCHOR: newton_step
            let gradient = gradient_function.evaluate(&result.parameters);
            let hessian = hessian_function.evaluate(&result.parameters);
            if gradient.len() != n || hessian.len() != n || hessian.iter().any(|row| row.len() != n) {
                return Err(ErrorsJSL::Misconfiguration("The gradient and Hessian must match the number of parameters."));
            }
            let hessian_matrix = ndarray::Array2::from_shape_vec((n, n), hessian.into_iter().flatten().collect())
                .expect("n × n entries");
            // Solve H p = −g rather than forming the inverse.
            let step = hessian_matrix
                .solve(&ndarray::Array1::from_vec(gradient))
                .map_err(|_| ErrorsJSL::RuntimeError("The Hessian is singular."))?
                .mapv(|x| -x);
            // ANCHOR_END: newton_step
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
            result.iterations += 1;
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
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization_traits::ObjectiveFunctionWithGradient;
    #[test]
    fn test_newton_raphson_method() {
        // This is a simple test case for the Newton-Raphson method. It uses a simple quadratic function with a minimum at x = 2.0 and a minimum value of 1.0. The algorithm should be able to find the minimum within a reasonable number of iterations.
        struct TestObjectiveFunction;
        impl ObjectiveFunction for TestObjectiveFunction {
            fn evaluate(&self, parameters: &[f64]) -> f64 {
                let x = parameters[0];
                (x - 2.0).powi(2) + 1.0
            }
        }
        impl GradientFunction for TestObjectiveFunction {
            fn evaluate(&self, parameters: &[f64]) -> Vec<f64> {
                let x = parameters[0];
                vec![2.0 * (x - 2.0)]
            }
        }
        impl HessianFunction for TestObjectiveFunction {
            fn evaluate(&self, _: &[f64]) -> Vec<Vec<f64>> {
                vec![vec![2.0]]
            }
        }
        impl ObjectiveFunctionWithGradient for TestObjectiveFunction {}
        let objective_function = TestObjectiveFunction;
        let mut optimizer = NewtonRaphsonMethod::new();
        optimizer.set_tolerance(Some(1e-6));
        optimizer.set_bounds(Some(vec![(0.0, 4.0)]));
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

    struct Flat;
    impl ObjectiveFunction for Flat {
        fn evaluate(&self, p: &[f64]) -> f64 {
            p[0]
        }
    }
    impl GradientFunction for Flat {
        fn evaluate(&self, _: &[f64]) -> Vec<f64> {
            vec![1.0]
        }
    }
    impl HessianFunction for Flat {
        fn evaluate(&self, _: &[f64]) -> Vec<Vec<f64>> {
            vec![vec![0.0]]
        }
    }

    #[test]
    fn singular_hessians_and_empty_input_are_errors_not_panics() {
        // Both once panicked: `inv().unwrap()` and `hessian[0]`.
        let newton = NewtonRaphsonMethod::new();
        assert!(newton.hessian_based_minimize(&Flat, &Flat, &Flat, vec![1.0]).is_err());
        assert!(newton.hessian_based_minimize(&Flat, &Flat, &Flat, vec![]).is_err());
    }

    #[test]
    fn quadratic_convergence_on_a_smooth_convex_function() {
        // f = Σ (eˣⁱ − xᵢ): minimum 1 per coordinate at 0.
        struct ExpMinusX;
        impl ObjectiveFunction for ExpMinusX {
            fn evaluate(&self, p: &[f64]) -> f64 {
                p.iter().map(|x| x.exp() - x).sum()
            }
        }
        impl GradientFunction for ExpMinusX {
            fn evaluate(&self, p: &[f64]) -> Vec<f64> {
                p.iter().map(|x| x.exp() - 1.0).collect()
            }
        }
        impl HessianFunction for ExpMinusX {
            fn evaluate(&self, p: &[f64]) -> Vec<Vec<f64>> {
                (0..p.len()).map(|i| (0..p.len()).map(|j| if i == j { p[i].exp() } else { 0.0 }).collect()).collect()
            }
        }
        let mut newton = NewtonRaphsonMethod::new();
        newton.set_tolerance(Some(1e-15));
        let r = newton.hessian_based_minimize(&ExpMinusX, &ExpMinusX, &ExpMinusX, vec![1.0, -1.0]).unwrap();
        assert!(r.converged && r.parameters.iter().all(|x| x.abs() < 1e-7) && r.iterations < 10, "{r:?}");
    }
}
