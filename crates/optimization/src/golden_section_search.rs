//! Golden section search for the minimum of a unimodal function of one variable.
//!
//! The search keeps an interval `[a, b]` known to contain the minimum and two interior points at
//! the golden-ratio positions `b − (b − a)/φ` and `a + (b − a)/φ`. Comparing `f` at them discards
//! one end; the surviving interior point is exactly where one of the next pair must be, so each
//! iteration costs one new evaluation and shrinks the interval by `1/φ ≈ 0.618`. No derivatives
//! are used. For a function that is not unimodal on `[a, b]` it finds a local minimum.
use crate::optimization_traits::{
    GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction, OptimizationResult,
};
use lib_jsl_core::ErrorsJSL;

/// Golden section search on the interval given by `bounds` (required, one pair). The initial
/// parameters are ignored apart from their length, which must be 1. Stops when the interval is
/// shorter than the tolerance (default 1e-5) or after `max_iterations` (default 1000) iterations,
/// and returns the midpoint of the final interval; `iterations` counts evaluations of `f`.
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

impl Default for GoldenSectionSearch {
    fn default() -> Self {
        Self::new()
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

impl<T> GradientFreeMinimizationEngine<T> for GoldenSectionSearch 
where T: ObjectiveFunction {
    fn gradient_free_minimize(
        &self,
        objective_function: &T,
        initial_parameters: Vec<f64>,
    ) -> Result<OptimizationResult, ErrorsJSL> {
        if initial_parameters.len() != 1 {
            return Err(ErrorsJSL::Misconfiguration("Golden section search only supports one-dimensional optimization.  That is, the initial parameters must be a single value."));
        }
        let Some(bounds) = self.get_bounds() else {
            return Err(ErrorsJSL::Misconfiguration("Golden section search needs bounds giving the interval to search."));
        };
        if bounds.len() != 1 {
            return Err(ErrorsJSL::Misconfiguration("Bounds length must match the number of parameters."));
        }
        let (mut a, mut b) = bounds[0];
        if a > b || !a.is_finite() || !b.is_finite() {
            return Err(ErrorsJSL::Misconfiguration("The bounds must be finite, with lower ≤ upper."));
        }
        let tolerance = self.tolerance.unwrap_or(1e-5);
        let max_iterations = self.max_iterations.unwrap_or(1000);
        // ANCHOR: golden
        let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0; // 1/φ ≈ 0.618
        let mut c = b - inv_phi * (b - a);
        let mut d = a + inv_phi * (b - a);
        let (mut fc, mut fd) = (objective_function.evaluate(&[c]), objective_function.evaluate(&[d]));
        let mut evaluations = 2;
        while b - a > tolerance && evaluations < max_iterations {
            if fc < fd {
                // The minimum is in [a, d]; the old c becomes the new d.
                (b, d, fd) = (d, c, fc);
                c = b - inv_phi * (b - a);
                fc = objective_function.evaluate(&[c]);
            } else {
                // The minimum is in [c, b]; the old d becomes the new c.
                (a, c, fc) = (c, d, fd);
                d = a + inv_phi * (b - a);
                fd = objective_function.evaluate(&[d]);
            }
            evaluations += 1;
        }
        // ANCHOR_END: golden
        let x = 0.5 * (a + b);
        let result = OptimizationResult {
            parameters: vec![x],
            objective_value: objective_function.evaluate(&[x]),
            iterations: evaluations,
            converged: b - a <= tolerance,
        };
        Ok(result)
    }


}

#[cfg(test)]
mod test {
    use super::*;
    struct TestObjectiveFunction;
    impl ObjectiveFunction for TestObjectiveFunction {
        fn evaluate(&self, parameters: &[f64]) -> f64 {
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

    #[test]
    fn one_evaluation_per_iteration_and_errors_without_bounds() {
        use std::cell::Cell;
        struct Counting(Cell<usize>);
        impl ObjectiveFunction for Counting {
            fn evaluate(&self, p: &[f64]) -> f64 {
                self.0.set(self.0.get() + 1);
                (p[0] - 0.3).abs()
            }
        }
        let f = Counting(Cell::new(0));
        let mut gs = GoldenSectionSearch::new();
        gs.set_tolerance(Some(1e-8));
        gs.set_bounds(Some(vec![(0.0, 1.0)]));
        let r = gs.gradient_free_minimize(&f, vec![0.5]).unwrap();
        // 0.618^k < 1e-8 needs k = 39 reductions: 2 + 38 evaluations, plus one at the answer.
        // The old version evaluated both interior points every time.
        assert!(r.converged && (r.parameters[0] - 0.3).abs() < 1e-8, "{r:?}");
        assert!(f.0.get() <= 42, "{} evaluations", f.0.get());
        // Missing bounds once panicked on an unwrap.
        assert!(GoldenSectionSearch::new().gradient_free_minimize(&f, vec![0.5]).is_err());
    }
}
