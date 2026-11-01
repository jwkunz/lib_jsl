//! Nonlinear conjugate gradient minimization.
//!
//! See the book entry [Gradient descent and nonlinear conjugate gradients](https://jwkunz.github.io/lib_jsl/book/optimization/gradient-methods.html#nonlinear-conjugate-gradients).

use crate::line_search::strong_wolfe;
use crate::optimization_traits::{GradientBasedMinimizationEngine, GradientFunction, MinimizationControls, ObjectiveFunction, OptimizationResult};
use crate::quasi_newton::{start, unbounded_controls};
use lib_jsl_core::ErrorsJSL;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// The formula for `β` in the direction update `d ← −g + β d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConjugateGradientBeta {
    /// `β = ‖g₊‖² / ‖g‖²`.
    FletcherReeves,
    /// `β = max(0, g₊ᵀ(g₊ − g) / ‖g‖²)`, PR+ (restarts automatically on lost conjugacy).
    PolakRibierePlus,
    /// `β = g₊ᵀy / dᵀy` with `y = g₊ − g`.
    HestenesStiefel,
    /// `β = ‖g₊‖² / dᵀy`.
    DaiYuan,
}

/// Nonlinear conjugate gradients: O(n) memory, a strong Wolfe line search with `c₂ = 0.1`, the
/// initial step of each line search scaled from the previous one (Nocedal and Wright eq. 3.60),
/// and a restart to steepest descent every `n` iterations or whenever the direction is not a
/// descent direction. Converges when `‖∇f‖∞` falls below the tolerance (default 1e-8). On a
/// strictly convex quadratic with exact line searches every variant reduces to linear CG.
#[derive(Debug, Clone)]
pub struct NonlinearConjugateGradient {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
    pub beta: ConjugateGradientBeta,
}

impl NonlinearConjugateGradient {
    pub fn new(beta: ConjugateGradientBeta) -> Self {
        NonlinearConjugateGradient { max_iterations: None, tolerance: None, bounds: None, beta }
    }
}

impl Default for NonlinearConjugateGradient {
    fn default() -> Self {
        Self::new(ConjugateGradientBeta::PolakRibierePlus)
    }
}
unbounded_controls!(NonlinearConjugateGradient);

impl<T: ObjectiveFunction, U: GradientFunction> GradientBasedMinimizationEngine<T, U> for NonlinearConjugateGradient {
    fn gradient_based_minimize(&self, objective: &T, gradient: &U, initial: Vec<f64>) -> Result<OptimizationResult, ErrorsJSL> {
        let (mut f, mut g) = start(&self.bounds, objective, gradient, &initial)?;
        let n = initial.len();
        let tol = self.tolerance.unwrap_or(1e-8);
        let max_iterations = self.max_iterations.unwrap_or(1000 * n.max(5));
        let inf_norm = |v: &[f64]| v.iter().fold(0.0f64, |m, a| m.max(a.abs()));
        let mut x = initial;
        let mut d: Vec<f64> = g.iter().map(|v| -v).collect();
        let mut step = (1.0 / inf_norm(&g).max(f64::MIN_POSITIVE)).min(1.0);
        let mut previous_slope = dot(&g, &d);
        let mut since_restart = 0;
        let mut iterations = 0;
        let mut converged = inf_norm(&g) <= tol;
        while !converged && iterations < max_iterations {
            iterations += 1;
            let slope = dot(&g, &d);
            if slope >= 0.0 {
                d = g.iter().map(|v| -v).collect();
                since_restart = 0;
                continue;
            }
            let step0 = if iterations == 1 { step } else { (step * previous_slope / slope).clamp(1e-10, 1e10) };
            let Some(p) = strong_wolfe(objective, gradient, &x, f, &g, &d, step0, 1e-4, 0.1) else {
                if since_restart == 0 {
                    break;
                }
                d = g.iter().map(|v| -v).collect();
                since_restart = 0;
                continue;
            };
            step = p.alpha;
            previous_slope = slope;
            let y: Vec<f64> = p.gradient.iter().zip(&g).map(|(a, b)| a - b).collect();
            let gg = dot(&g, &g);
            let new_gg = dot(&p.gradient, &p.gradient);
            let dy = dot(&d, &y);
            // ANCHOR: cg_direction
            let beta = match self.beta {
                ConjugateGradientBeta::FletcherReeves => new_gg / gg,
                ConjugateGradientBeta::PolakRibierePlus => (dot(&p.gradient, &y) / gg).max(0.0),
                ConjugateGradientBeta::HestenesStiefel => dot(&p.gradient, &y) / dy,
                ConjugateGradientBeta::DaiYuan => new_gg / dy,
            };
            (x, f, g) = (p.x, p.value, p.gradient);
            since_restart += 1;
            let beta = if since_restart >= n || !beta.is_finite() {
                since_restart = 0;
                0.0
            } else {
                beta
            };
            d.iter_mut().zip(&g).for_each(|(di, gi)| *di = -gi + beta * *di);
            // ANCHOR_END: cg_direction
            converged = inf_norm(&g) <= tol;
        }
        Ok(OptimizationResult { parameters: x, objective_value: f, iterations, converged })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quasi_newton::test_problems::{Quadratic, Rosenbrock};
    use ConjugateGradientBeta::*;

    #[test]
    fn every_beta_formula_minimizes_rosenbrock() {
        for beta in [FletcherReeves, PolakRibierePlus, HestenesStiefel, DaiYuan] {
            let cg = NonlinearConjugateGradient::new(beta);
            for n in [2, 6] {
                let x0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
                let r = cg.gradient_based_minimize(&Rosenbrock, &Rosenbrock, x0).unwrap();
                assert!(r.converged && r.parameters.iter().all(|v| (v - 1.0).abs() < 1e-6), "{beta:?} n={n}: {r:?}");
            }
        }
    }

    #[test]
    fn handles_large_ill_conditioned_quadratics_and_validates() {
        let r = NonlinearConjugateGradient::default().gradient_based_minimize(&Quadratic, &Quadratic, vec![1.0; 500]).unwrap();
        assert!(r.converged && r.objective_value < 1e-12, "{r:?}");
        let mut cg = NonlinearConjugateGradient::default();
        assert!(cg.gradient_based_minimize(&Quadratic, &Quadratic, vec![]).is_err());
        cg.set_bounds(Some(vec![(0.0, 1.0)]));
        assert!(cg.gradient_based_minimize(&Quadratic, &Quadratic, vec![1.0]).is_err());
    }
}
