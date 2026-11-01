//! Quasi-Newton minimizers: BFGS with a dense inverse-Hessian approximation and limited-memory
//! L-BFGS. Both use a strong Wolfe line search, which keeps the curvature condition `yᵀs > 0`
//! and hence the approximation positive definite.
//!
//! See the book entry [Newton and quasi-Newton methods](https://jwkunz.github.io/lib_jsl/book/optimization/newton-quasi-newton.html#bfgs).

use crate::line_search::strong_wolfe;
use crate::optimization_traits::{GradientBasedMinimizationEngine, GradientFunction, MinimizationControls, ObjectiveFunction, OptimizationResult};
use lib_jsl_core::ErrorsJSL;
use std::collections::VecDeque;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}

macro_rules! unbounded_controls {
    ($t:ty) => {
        impl MinimizationControls for $t {
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
            /// Bounds are not supported; setting them makes minimization return an error.
            fn set_bounds(&mut self, bounds: Option<Vec<(f64, f64)>>) {
                self.bounds = bounds;
            }
            fn get_bounds(&self) -> Option<Vec<(f64, f64)>> {
                self.bounds.clone()
            }
        }
    };
}
pub(crate) use unbounded_controls;

/// Checks the shared preconditions and returns the starting value and gradient.
pub(crate) fn start<T: ObjectiveFunction, U: GradientFunction>(
    bounds: &Option<Vec<(f64, f64)>>,
    objective: &T,
    gradient: &U,
    x: &[f64],
) -> Result<(f64, Vec<f64>), ErrorsJSL> {
    if x.is_empty() {
        return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
    }
    if bounds.is_some() {
        return Err(ErrorsJSL::Misconfiguration("This method is unconstrained; bounds are not supported."));
    }
    let g = gradient.evaluate(x);
    if g.len() != x.len() {
        return Err(ErrorsJSL::Misconfiguration("Gradient length must match the number of parameters."));
    }
    Ok((objective.evaluate(x), g))
}

/// BFGS with a dense n × n inverse-Hessian approximation `H`, updated as
/// `H ← (I − ρsyᵀ) H (I − ρysᵀ) + ρssᵀ`, `ρ = 1/(yᵀs)`, and initially scaled by `yᵀs / yᵀy` after
/// the first step. Converges when `‖∇f‖∞` falls below the tolerance (default 1e-8).
#[derive(Debug, Clone, Default)]
pub struct Bfgs {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
}

impl Bfgs {
    pub fn new() -> Self {
        Self::default()
    }
}
unbounded_controls!(Bfgs);

impl<T: ObjectiveFunction, U: GradientFunction> GradientBasedMinimizationEngine<T, U> for Bfgs {
    fn gradient_based_minimize(&self, objective: &T, gradient: &U, initial: Vec<f64>) -> Result<OptimizationResult, ErrorsJSL> {
        let (mut f, mut g) = start(&self.bounds, objective, gradient, &initial)?;
        let n = initial.len();
        let tol = self.tolerance.unwrap_or(1e-8);
        let max_iterations = self.max_iterations.unwrap_or(200 * n.max(5));
        let mut x = initial;
        let mut h = vec![vec![0.0; n]; n];
        (0..n).for_each(|i| h[i][i] = 1.0);
        let mut first = true;
        let mut iterations = 0;
        let mut converged = inf_norm(&g) <= tol;
        while !converged && iterations < max_iterations {
            iterations += 1;
            let mut d: Vec<f64> = h.iter().map(|row| -dot(row, &g)).collect();
            if dot(&d, &g) >= 0.0 {
                // Lost positive definiteness to rounding: restart from steepest descent.
                (0..n).for_each(|i| (0..n).for_each(|j| h[i][j] = if i == j { 1.0 } else { 0.0 }));
                d = g.iter().map(|v| -v).collect();
                first = true;
            }
            let step0 = if first { (1.0 / inf_norm(&g)).min(1.0) } else { 1.0 };
            let Some(p) = strong_wolfe(objective, gradient, &x, f, &g, &d, step0, 1e-4, 0.9) else { break };
            let s: Vec<f64> = p.x.iter().zip(&x).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = p.gradient.iter().zip(&g).map(|(a, b)| a - b).collect();
            let sy = dot(&s, &y);
            (x, f, g) = (p.x, p.value, p.gradient);
            if sy > 1e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
                if first {
                    let scale = sy / dot(&y, &y);
                    h.iter_mut().flatten().for_each(|v| *v *= scale);
                    first = false;
                }
                // ANCHOR: bfgs_update
                let rho = 1.0 / sy;
                let hy: Vec<f64> = h.iter().map(|row| dot(row, &y)).collect();
                let yhy = dot(&y, &hy);
                for i in 0..n {
                    for j in 0..n {
                        h[i][j] += -rho * (hy[i] * s[j] + s[i] * hy[j]) + (rho * rho * yhy + rho) * s[i] * s[j];
                    }
                }
                // ANCHOR_END: bfgs_update
            }
            converged = inf_norm(&g) <= tol;
        }
        Ok(OptimizationResult { parameters: x, objective_value: f, iterations, converged })
    }
}

/// Limited-memory BFGS: keeps the last `memory` correction pairs `(s, y)` and applies the inverse
/// Hessian by the two-loop recursion in O(memory · n), so it scales to large problems.
#[derive(Debug, Clone)]
pub struct Lbfgs {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
    pub memory: usize,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Lbfgs { max_iterations: None, tolerance: None, bounds: None, memory: 10 }
    }
}

impl Lbfgs {
    pub fn new(memory: usize) -> Self {
        Lbfgs { memory: memory.max(1), ..Self::default() }
    }
}
unbounded_controls!(Lbfgs);

impl<T: ObjectiveFunction, U: GradientFunction> GradientBasedMinimizationEngine<T, U> for Lbfgs {
    fn gradient_based_minimize(&self, objective: &T, gradient: &U, initial: Vec<f64>) -> Result<OptimizationResult, ErrorsJSL> {
        let (mut f, mut g) = start(&self.bounds, objective, gradient, &initial)?;
        let n = initial.len();
        let tol = self.tolerance.unwrap_or(1e-8);
        let max_iterations = self.max_iterations.unwrap_or(200 * n.max(5));
        let mut x = initial;
        let mut pairs: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::with_capacity(self.memory);
        let mut iterations = 0;
        let mut converged = inf_norm(&g) <= tol;
        while !converged && iterations < max_iterations {
            iterations += 1;
            // ANCHOR: two_loop
            // Two-loop recursion: d = −H g.
            let mut q = g.clone();
            let mut alphas = Vec::with_capacity(pairs.len());
            for (s, y, rho) in pairs.iter().rev() {
                let a = rho * dot(s, &q);
                q.iter_mut().zip(y).for_each(|(v, yi)| *v -= a * yi);
                alphas.push(a);
            }
            let gamma = pairs.back().map_or(1.0, |(s, y, _)| dot(s, y) / dot(y, y));
            q.iter_mut().for_each(|v| *v *= gamma);
            for ((s, y, rho), a) in pairs.iter().zip(alphas.iter().rev()) {
                let b = rho * dot(y, &q);
                q.iter_mut().zip(s).for_each(|(v, si)| *v += (a - b) * si);
            }
            // ANCHOR_END: two_loop
            let mut d: Vec<f64> = q.iter().map(|v| -v).collect();
            if dot(&d, &g) >= 0.0 {
                pairs.clear();
                d = g.iter().map(|v| -v).collect();
            }
            let step0 = if pairs.is_empty() { (1.0 / inf_norm(&g)).min(1.0) } else { 1.0 };
            let Some(p) = strong_wolfe(objective, gradient, &x, f, &g, &d, step0, 1e-4, 0.9) else { break };
            let s: Vec<f64> = p.x.iter().zip(&x).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = p.gradient.iter().zip(&g).map(|(a, b)| a - b).collect();
            let sy = dot(&s, &y);
            (x, f, g) = (p.x, p.value, p.gradient);
            if sy > 1e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
                if pairs.len() == self.memory {
                    pairs.pop_front();
                }
                pairs.push_back((s, y, 1.0 / sy));
            }
            converged = inf_norm(&g) <= tol;
        }
        Ok(OptimizationResult { parameters: x, objective_value: f, iterations, converged })
    }
}

#[cfg(test)]
pub(crate) mod test_problems {
    use super::*;

    /// The n-dimensional Rosenbrock function, minimum 0 at (1, …, 1).
    pub struct Rosenbrock;
    impl ObjectiveFunction for Rosenbrock {
        fn evaluate(&self, p: &[f64]) -> f64 {
            p.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
        }
    }
    impl GradientFunction for Rosenbrock {
        fn evaluate(&self, p: &[f64]) -> Vec<f64> {
            let mut g = vec![0.0; p.len()];
            for i in 0..p.len() - 1 {
                let t = p[i + 1] - p[i] * p[i];
                g[i] += -400.0 * p[i] * t - 2.0 * (1.0 - p[i]);
                g[i + 1] += 200.0 * t;
            }
            g
        }
    }

    /// An ill-conditioned quadratic `½ Σ c_i x_i²` with `c_i` from 1 to 1e4.
    pub struct Quadratic;
    impl ObjectiveFunction for Quadratic {
        fn evaluate(&self, p: &[f64]) -> f64 {
            let n = p.len() as f64;
            p.iter().enumerate().map(|(i, v)| 0.5 * 1e4f64.powf(i as f64 / (n - 1.0)) * v * v).sum()
        }
    }
    impl GradientFunction for Quadratic {
        fn evaluate(&self, p: &[f64]) -> Vec<f64> {
            let n = p.len() as f64;
            p.iter().enumerate().map(|(i, v)| 1e4f64.powf(i as f64 / (n - 1.0)) * v).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_problems::*;
    use super::*;

    #[test]
    fn bfgs_and_lbfgs_minimize_rosenbrock() {
        for n in [2, 10] {
            let x0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
            let b = Bfgs::new().gradient_based_minimize(&Rosenbrock, &Rosenbrock, x0.clone()).unwrap();
            assert!(b.converged && b.parameters.iter().all(|v| (v - 1.0).abs() < 1e-6), "bfgs n={n}: {b:?}");
            let l = Lbfgs::new(7).gradient_based_minimize(&Rosenbrock, &Rosenbrock, x0).unwrap();
            assert!(l.converged && l.parameters.iter().all(|v| (v - 1.0).abs() < 1e-6), "lbfgs n={n}: {l:?}");
            assert!(b.iterations < 200 && l.iterations < 300, "{} {}", b.iterations, l.iterations);
        }
    }

    #[test]
    fn bfgs_is_finite_on_quadratics_and_lbfgs_scales() {
        // With exact line searches BFGS terminates in n steps on a quadratic; with inexact Wolfe
        // steps and a condition number of 1e4 it still needs only a few dozen (gradient descent
        // would need tens of thousands).
        let b = Bfgs::new().gradient_based_minimize(&Quadratic, &Quadratic, vec![1.0; 8]).unwrap();
        assert!(b.converged && b.iterations < 80, "{}", b.iterations);
        let l = Lbfgs::default().gradient_based_minimize(&Quadratic, &Quadratic, vec![1.0; 1000]).unwrap();
        assert!(l.converged && l.objective_value < 1e-12, "{l:?}");
    }

    #[test]
    fn validation() {
        let mut b = Bfgs::new();
        assert!(b.gradient_based_minimize(&Quadratic, &Quadratic, vec![]).is_err());
        b.set_bounds(Some(vec![(0.0, 1.0)]));
        assert!(b.gradient_based_minimize(&Quadratic, &Quadratic, vec![1.0]).is_err());
        // Starting at the optimum converges immediately.
        let r = Lbfgs::default().gradient_based_minimize(&Rosenbrock, &Rosenbrock, vec![1.0, 1.0]).unwrap();
        assert!(r.converged && r.iterations == 0);
    }
}
