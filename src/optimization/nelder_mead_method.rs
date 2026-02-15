use crate::optimization::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction, OptimizationResult};
use crate::prelude::ErrorsJSL;

/// This file implments the Nelder-Mead method for optimization. 
/// The Nelder-Mead method is a gradient-free optimization algorithm that is used to find the minimum of a function. 
/// The algorithm works by maintaining a simplex of n+1 points in n-dimensional space, where each point represents a candidate solution to the optimization problem. 
/// The algorithm iteratively updates the simplex by performing operations such as reflection, expansion, contraction, and shrinkage, based on the function values at the vertices of the simplex. 
/// The algorithm continues until it reaches a point where the function values at the vertices of the simplex are sufficiently close to each other, indicating that it has found a local minimum. 
/// The Nelder-Mead method can be effective for optimizing non-convex functions and functions with noisy gradients, but it can also be computationally expensive and may require careful tuning of the algorithm's parameters to achieve good performance.  

pub struct NelderMeadMethod {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
}

impl NelderMeadMethod {
    pub fn new() -> Self {
        NelderMeadMethod {
            max_iterations: None,
            tolerance: None,
            bounds: None,
        }
    }
}

impl MinimizationControls for NelderMeadMethod {
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

impl<T> GradientFreeMinimizationEngine<T> for NelderMeadMethod 
where T: ObjectiveFunction {
    fn gradient_free_minimize(
        &self,
        objective_function: &T,
        initial_parameters: Vec<f64>,
    ) -> Result<OptimizationResult, ErrorsJSL> {
        if initial_parameters.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
        }

        let dimension = initial_parameters.len();
        let max_iterations = self.get_max_iterations().unwrap_or(1000);
        let tolerance = self.get_tolerance().unwrap_or(1e-8);
        let bounds = self.get_bounds();

        if let Some(ref b) = bounds {
            if b.len() != dimension {
                return Err(ErrorsJSL::Misconfiguration(
                    "Bounds length must match the number of parameters.",
                ));
            }
            for (lower, upper) in b.iter() {
                if lower > upper {
                    return Err(ErrorsJSL::Misconfiguration(
                        "Each bound must satisfy lower <= upper.",
                    ));
                }
            }
        }

        let alpha = 1.0;
        let gamma = 2.0;
        let rho = 0.5;
        let sigma = 0.5;

        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(dimension + 1);
        simplex.push(project_to_bounds(&initial_parameters, bounds.as_ref()));
        for i in 0..dimension {
            let mut vertex = initial_parameters.clone();
            let step = if initial_parameters[i].abs() > 1e-12 {
                0.05 * initial_parameters[i].abs()
            } else {
                2.5e-4
            };
            vertex[i] += step;
            simplex.push(project_to_bounds(&vertex, bounds.as_ref()));
        }

        let mut values: Vec<f64> = simplex.iter().map(|x| objective_function.evaluate(x)).collect();

        let mut result = OptimizationResult {
            parameters: simplex[0].clone(),
            objective_value: values[0],
            iterations: 0,
            converged: false,
        };

        for iteration in 0..max_iterations {
            sort_simplex(&mut simplex, &mut values);

            result.parameters = simplex[0].clone();
            result.objective_value = values[0];
            result.iterations = iteration + 1;

            if function_value_stddev(&values) <= tolerance {
                result.converged = true;
                return Ok(result);
            }

            let centroid = centroid_excluding_worst(&simplex);
            let worst = &simplex[dimension];
            let second_worst_value = values[dimension - 1];
            let worst_value = values[dimension];

            let reflected = combine(&centroid, worst, 1.0 + alpha, -alpha);
            let reflected = project_to_bounds(&reflected, bounds.as_ref());
            let reflected_value = objective_function.evaluate(&reflected);

            if reflected_value < values[0] {
                let expanded = combine(&centroid, &reflected, 1.0 - gamma, gamma);
                let expanded = project_to_bounds(&expanded, bounds.as_ref());
                let expanded_value = objective_function.evaluate(&expanded);
                if expanded_value < reflected_value {
                    simplex[dimension] = expanded;
                    values[dimension] = expanded_value;
                } else {
                    simplex[dimension] = reflected;
                    values[dimension] = reflected_value;
                }
                continue;
            }

            if reflected_value < second_worst_value {
                simplex[dimension] = reflected;
                values[dimension] = reflected_value;
                continue;
            }

            if reflected_value < worst_value {
                let outside_contracted = combine(&centroid, &reflected, 1.0 - rho, rho);
                let outside_contracted = project_to_bounds(&outside_contracted, bounds.as_ref());
                let outside_contracted_value = objective_function.evaluate(&outside_contracted);
                if outside_contracted_value <= reflected_value {
                    simplex[dimension] = outside_contracted;
                    values[dimension] = outside_contracted_value;
                    continue;
                }
            } else {
                let inside_contracted = combine(&centroid, worst, 1.0 - rho, rho);
                let inside_contracted = project_to_bounds(&inside_contracted, bounds.as_ref());
                let inside_contracted_value = objective_function.evaluate(&inside_contracted);
                if inside_contracted_value < worst_value {
                    simplex[dimension] = inside_contracted;
                    values[dimension] = inside_contracted_value;
                    continue;
                }
            }

            let best = simplex[0].clone();
            for i in 1..=dimension {
                simplex[i] = combine(&best, &simplex[i], 1.0 - sigma, sigma);
                simplex[i] = project_to_bounds(&simplex[i], bounds.as_ref());
                values[i] = objective_function.evaluate(&simplex[i]);
            }
        }

        sort_simplex(&mut simplex, &mut values);
        result.parameters = simplex[0].clone();
        result.objective_value = values[0];
        result.converged = function_value_stddev(&values) <= tolerance;
        Ok(result)
    }
}

fn project_to_bounds(parameters: &Vec<f64>, bounds: Option<&Vec<(f64, f64)>>) -> Vec<f64> {
    if let Some(bounds) = bounds {
        parameters
            .iter()
            .zip(bounds.iter())
            .map(|(&value, &(lower, upper))| {
                if value < lower {
                    lower
                } else if value > upper {
                    upper
                } else {
                    value
                }
            })
            .collect()
    } else {
        parameters.clone()
    }
}

fn combine(a: &Vec<f64>, b: &Vec<f64>, wa: f64, wb: f64) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| wa * x + wb * y)
        .collect()
}

fn centroid_excluding_worst(simplex: &Vec<Vec<f64>>) -> Vec<f64> {
    let dimension = simplex[0].len();
    let count = simplex.len() - 1;
    let mut centroid = vec![0.0; dimension];
    for point in simplex.iter().take(count) {
        for i in 0..dimension {
            centroid[i] += point[i];
        }
    }
    for value in centroid.iter_mut() {
        *value /= count as f64;
    }
    centroid
}

fn function_value_stddev(values: &Vec<f64>) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|v| {
            let dv = *v - mean;
            dv * dv
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn sort_simplex(simplex: &mut Vec<Vec<f64>>, values: &mut Vec<f64>) {
    let mut indexed: Vec<(Vec<f64>, f64)> = simplex.drain(..).zip(values.drain(..)).collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (point, value) in indexed {
        simplex.push(point);
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestObjectiveFunction;

    impl ObjectiveFunction for TestObjectiveFunction {
        fn evaluate(&self, parameters: &Vec<f64>) -> f64 {
            let x = parameters[0];
            let y = parameters[1];
            (x - 2.0).powi(2) + (y + 1.0).powi(2) + 3.0
        }
    }

    #[test]
    fn test_nelder_mead_method() {
        let objective = TestObjectiveFunction;
        let mut optimizer = NelderMeadMethod::new();
        optimizer.set_max_iterations(Some(2000));
        optimizer.set_tolerance(Some(1e-10));
        optimizer.set_bounds(Some(vec![(-5.0, 5.0), (-5.0, 5.0)]));

        let result = optimizer
            .gradient_free_minimize(&objective, vec![0.0, 0.0])
            .unwrap();

        dbg!("Result: {:?}", &result);
        assert!(result.converged);
        assert!((result.parameters[0] - 2.0).abs() < 1e-3);
        assert!((result.parameters[1] + 1.0).abs() < 1e-3);
        assert!((result.objective_value - 3.0).abs() < 1e-6);
    }
}
