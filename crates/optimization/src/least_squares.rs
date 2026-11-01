//! Nonlinear least squares by the Levenberg-Marquardt method, and curve fitting built on it.
//!
//! The problem is `min_p F(p) = ½ Σ r_i(p)²` for a residual vector `r: ℝⁿ → ℝᵐ` (`m ≥ n`). Each
//! iteration solves the damped Gauss-Newton system `(JᵀJ + μ D) h = −Jᵀr`, where `D` is the
//! diagonal of `JᵀJ` (Marquardt's scaling, which makes the method invariant to rescaling the
//! parameters). The damping `μ` follows Nielsen's gain-ratio update: steps that achieve the
//! predicted reduction shrink `μ` towards Gauss-Newton, failed steps grow it towards a short
//! gradient-descent step.

use crate::optimization_traits::MinimizationControls;
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};
use lib_jsl_linalg::{Inverse, Solve};

/// The outcome of a least-squares solve.
#[derive(Debug, Clone)]
pub struct LeastSquaresResult {
    pub parameters: Vec<f64>,
    /// The residuals at the solution.
    pub residuals: Vec<f64>,
    /// `½ Σ r_i²` at the solution.
    pub cost: f64,
    /// The Jacobian `∂r_i/∂p_j` at the solution (row-major, m × n).
    pub jacobian: Vec<Vec<f64>>,
    pub iterations: usize,
    pub converged: bool,
}

/// Levenberg-Marquardt solver. The tolerance (default 1e-10) applies to the gradient infinity
/// norm (scaled by 1e-2), the relative step size and the relative cost reduction; bounds, when set, are enforced
/// by projecting each trial point onto the box.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardt {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
    /// Initial damping relative to the largest diagonal entry of `JᵀJ`.
    pub initial_damping: f64,
    /// Relative step for forward-difference Jacobians.
    pub difference_step: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        LevenbergMarquardt { max_iterations: None, tolerance: None, bounds: None, initial_damping: 1e-3, difference_step: 1e-7 }
    }
}

impl MinimizationControls for LevenbergMarquardt {
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

fn half_sum_squares(r: &[f64]) -> f64 {
    0.5 * r.iter().map(|v| v * v).sum::<f64>()
}

impl LevenbergMarquardt {
    pub fn new() -> Self {
        Self::default()
    }

    fn project(&self, p: &mut [f64]) {
        if let Some(b) = &self.bounds {
            p.iter_mut().zip(b).for_each(|(v, (lo, hi))| *v = v.clamp(*lo, *hi));
        }
    }

    /// Forward-difference Jacobian of `residuals` at `p` (with `r = residuals(p)` supplied).
    pub fn numerical_jacobian<R>(&self, residuals: &R, p: &[f64], r: &[f64]) -> Vec<Vec<f64>>
    where
        R: Fn(&[f64]) -> Vec<f64>,
    {
        let mut jac = vec![vec![0.0; p.len()]; r.len()];
        let mut q = p.to_vec();
        for j in 0..p.len() {
            let h = self.difference_step * p[j].abs().max(1.0);
            q[j] = p[j] + h;
            let rq = residuals(&q);
            q[j] = p[j];
            for i in 0..r.len() {
                jac[i][j] = (rq[i] - r[i]) / h;
            }
        }
        jac
    }

    /// Minimizes `½‖residuals(p)‖²` from `initial`, using the analytic `jacobian` when given and
    /// forward differences otherwise.
    pub fn solve<R, J>(&self, residuals: R, jacobian: Option<J>, initial: Vec<f64>) -> Result<LeastSquaresResult, ErrorsJSL>
    where
        R: Fn(&[f64]) -> Vec<f64>,
        J: Fn(&[f64]) -> Vec<Vec<f64>>,
    {
        let n = initial.len();
        if n == 0 {
            return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
        }
        if let Some(b) = &self.bounds
            && (b.len() != n || b.iter().any(|(lo, hi)| lo > hi))
        {
            return Err(ErrorsJSL::Misconfiguration("Bounds must match the parameters and satisfy lower <= upper."));
        }
        let tol = self.tolerance.unwrap_or(1e-10);
        let max_iterations = self.max_iterations.unwrap_or(200 * (n + 1));
        let jac_at = |p: &[f64], r: &[f64]| -> Result<Vec<Vec<f64>>, ErrorsJSL> {
            let jac = match &jacobian {
                Some(f) => f(p),
                None => self.numerical_jacobian(&residuals, p, r),
            };
            if jac.len() != r.len() || jac.iter().any(|row| row.len() != n) {
                return Err(ErrorsJSL::Misconfiguration("The Jacobian must be m × n."));
            }
            Ok(jac)
        };
        let normal_equations = |jac: &[Vec<f64>], r: &[f64]| {
            let a = Array2::from_shape_fn((n, n), |(i, j)| jac.iter().map(|row| row[i] * row[j]).sum::<f64>());
            let g = Array1::from_shape_fn(n, |i| jac.iter().zip(r).map(|(row, ri)| row[i] * ri).sum::<f64>());
            (a, g)
        };

        let mut p = initial;
        self.project(&mut p);
        let mut r = residuals(&p);
        if r.len() < n {
            return Err(ErrorsJSL::Misconfiguration("There must be at least as many residuals as parameters."));
        }
        let mut cost = half_sum_squares(&r);
        let mut jac = jac_at(&p, &r)?;
        let (mut a, mut g) = normal_equations(&jac, &r);
        let max_diag = (0..n).map(|i| a[[i, i]]).fold(0.0, f64::max);
        let mut mu = self.initial_damping * max_diag.max(f64::MIN_POSITIVE);
        let mut nu = 2.0;
        let mut converged = false;
        let mut iterations = 0;
        while iterations < max_iterations {
            if g.iter().fold(0.0f64, |m, v| m.max(v.abs())) <= 1e-2 * tol || cost == 0.0 {
                converged = true;
                break;
            }
            iterations += 1;
            let diag: Vec<f64> = (0..n).map(|i| a[[i, i]].max(1e-12 * max_diag.max(1.0))).collect();
            let mut damped = a.clone();
            (0..n).for_each(|i| damped[[i, i]] += mu * diag[i]);
            let h = match damped.solve(&(-&g)) {
                Ok(h) => h,
                Err(_) => {
                    mu *= nu;
                    nu *= 2.0;
                    continue;
                }
            };
            let p_norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            if h.dot(&h).sqrt() <= tol * (p_norm + tol) {
                converged = true;
                break;
            }
            let mut trial: Vec<f64> = p.iter().zip(h.iter()).map(|(a, b)| a + b).collect();
            self.project(&mut trial);
            let step = Array1::from_shape_fn(n, |i| trial[i] - p[i]);
            let r_trial = residuals(&trial);
            let cost_trial = half_sum_squares(&r_trial);
            // Predicted reduction of the linear model, ½ hᵀ(μ D h − g).
            let predicted = 0.5 * (0..n).map(|i| step[i] * (mu * diag[i] * step[i] - g[i])).sum::<f64>();
            let rho = if predicted > 0.0 { (cost - cost_trial) / predicted } else { -1.0 };
            if rho > 0.0 && cost_trial.is_finite() {
                let relative_drop = (cost - cost_trial) / cost.max(f64::MIN_POSITIVE);
                p = trial;
                r = r_trial;
                cost = cost_trial;
                jac = jac_at(&p, &r)?;
                (a, g) = normal_equations(&jac, &r);
                mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                if relative_drop < tol {
                    converged = true;
                    break;
                }
            } else {
                mu *= nu;
                nu *= 2.0;
                if !mu.is_finite() {
                    break;
                }
            }
        }
        Ok(LeastSquaresResult { parameters: p, residuals: r, cost, jacobian: jac, iterations, converged })
    }
}

/// The outcome of [`curve_fit`].
#[derive(Debug, Clone)]
pub struct CurveFitResult {
    pub parameters: Vec<f64>,
    /// Parameter covariance (n × n).
    pub covariance: Vec<Vec<f64>>,
    /// One-sigma parameter uncertainties, `√diag(covariance)`.
    pub standard_errors: Vec<f64>,
    /// `Σ ((y_i − f(x_i)) / σ_i)²`.
    pub chi_squared: f64,
    /// `chi_squared / (m − n)`.
    pub reduced_chi_squared: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Fits `model(x, p)` to data `(x_i, y_i)` by weighted least squares with Levenberg-Marquardt.
///
/// With `sigma = Some(σ)` each residual is divided by its standard deviation and the covariance
/// is `(JᵀJ)⁻¹` (absolute uncertainties). With `sigma = None` the residuals are unweighted and the
/// covariance is scaled by the reduced chi-squared, which estimates the unknown noise level (as
/// SciPy's `curve_fit` does by default).
pub fn curve_fit<M>(
    model: M,
    x: &[f64],
    y: &[f64],
    sigma: Option<&[f64]>,
    initial: Vec<f64>,
    solver: &LevenbergMarquardt,
) -> Result<CurveFitResult, ErrorsJSL>
where
    M: Fn(f64, &[f64]) -> f64,
{
    if x.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
    }
    if let Some(s) = sigma
        && (s.len() != y.len() || s.iter().any(|v| *v <= 0.0))
    {
        return Err(ErrorsJSL::Misconfiguration("sigma must match the data and be positive."));
    }
    let n = initial.len();
    if x.len() <= n {
        return Err(ErrorsJSL::Misconfiguration("There must be more data points than parameters."));
    }
    let weight = |i: usize| sigma.map_or(1.0, |s| 1.0 / s[i]);
    let residuals = |p: &[f64]| (0..x.len()).map(|i| (y[i] - model(x[i], p)) * weight(i)).collect::<Vec<f64>>();
    let fit = solver.solve(residuals, None::<fn(&[f64]) -> Vec<Vec<f64>>>, initial)?;
    let jtj = Array2::from_shape_fn((n, n), |(i, j)| fit.jacobian.iter().map(|row| row[i] * row[j]).sum::<f64>());
    let inv = jtj.inv().map_err(|_| ErrorsJSL::RuntimeError("The fit Jacobian is rank deficient; the covariance is undefined."))?;
    let chi_squared = 2.0 * fit.cost;
    let reduced_chi_squared = chi_squared / (x.len() - n) as f64;
    let scale = if sigma.is_some() { 1.0 } else { reduced_chi_squared };
    let covariance: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| inv[[i, j]] * scale).collect()).collect();
    let standard_errors = (0..n).map(|i| covariance[i][i].max(0.0).sqrt()).collect();
    Ok(CurveFitResult {
        parameters: fit.parameters,
        covariance,
        standard_errors,
        chi_squared,
        reduced_chi_squared,
        iterations: fit.iterations,
        converged: fit.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type NoJacobian = fn(&[f64]) -> Vec<Vec<f64>>;

    #[test]
    fn solves_the_rosenbrock_problem_with_and_without_an_analytic_jacobian() {
        // Rosenbrock as least squares: r = (10(y − x²), 1 − x).
        let r = |p: &[f64]| vec![10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]];
        let j = |p: &[f64]| vec![vec![-20.0 * p[0], 10.0], vec![-1.0, 0.0]];
        let lm = LevenbergMarquardt::new();
        let analytic = lm.solve(r, Some(j), vec![-1.2, 1.0]).unwrap();
        assert!(analytic.converged && (analytic.parameters[0] - 1.0).abs() < 1e-8 && (analytic.parameters[1] - 1.0).abs() < 1e-8);
        let numeric = lm.solve(r, None::<NoJacobian>, vec![-1.2, 1.0]).unwrap();
        assert!((numeric.parameters[0] - 1.0).abs() < 1e-6 && numeric.cost < 1e-14, "{numeric:?}");
        assert!(analytic.iterations < 50);
    }

    #[test]
    fn fits_an_exponential_decay_and_reports_consistent_uncertainties() {
        // Deterministic pseudo-noise of known standard deviation 0.02.
        let truth = [2.5, 1.3, 0.4];
        let model = |t: f64, p: &[f64]| p[0] * (-p[1] * t).exp() + p[2];
        let x: Vec<f64> = (0..60).map(|i| i as f64 * 0.1).collect();
        let noise: Vec<f64> = (0..60).map(|i| 0.02 * 3f64.sqrt() * (2.0 * ((i * 37 + 11) % 61) as f64 / 60.0 - 1.0)).collect();
        let y: Vec<f64> = x.iter().zip(&noise).map(|(&t, e)| model(t, &truth) + e).collect();
        let fit = curve_fit(model, &x, &y, None, vec![1.0, 1.0, 0.0], &LevenbergMarquardt::new()).unwrap();
        assert!(fit.converged);
        for (k, t) in truth.iter().enumerate() {
            assert!((fit.parameters[k] - t).abs() < 4.0 * fit.standard_errors[k], "param {k}: {fit:?}");
            assert!(fit.standard_errors[k] > 0.0 && fit.standard_errors[k] < 0.05);
        }
        // Unweighted reduced chi-squared estimates the noise variance (≈ 0.02²).
        assert!((fit.reduced_chi_squared.sqrt() - 0.02).abs() < 0.005);
        // Supplying the true sigma gives a reduced chi-squared near one and similar errors.
        let sigma = vec![0.02; 60];
        let weighted = curve_fit(model, &x, &y, Some(&sigma), vec![1.0, 1.0, 0.0], &LevenbergMarquardt::new()).unwrap();
        assert!((weighted.reduced_chi_squared - 1.0).abs() < 0.3);
        assert!((weighted.parameters[1] - fit.parameters[1]).abs() < 1e-6);
    }

    #[test]
    fn exact_linear_fit_bounds_and_validation() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|t| 3.0 * t - 2.0).collect();
        let fit = curve_fit(|t, p| p[0] * t + p[1], &x, &y, None, vec![0.0, 0.0], &LevenbergMarquardt::new()).unwrap();
        assert!((fit.parameters[0] - 3.0).abs() < 1e-9 && (fit.parameters[1] + 2.0).abs() < 1e-9 && fit.chi_squared < 1e-16);
        // A bound that excludes the unconstrained optimum is active at the solution.
        let mut lm = LevenbergMarquardt::new();
        lm.set_bounds(Some(vec![(0.0, 2.0), (-10.0, 10.0)]));
        let r = |p: &[f64]| x.iter().zip(&y).map(|(t, v)| v - (p[0] * t + p[1])).collect::<Vec<_>>();
        let bounded = lm.solve(r, None::<NoJacobian>, vec![1.0, 0.0]).unwrap();
        assert!((bounded.parameters[0] - 2.0).abs() < 1e-12);
        assert!(curve_fit(|t, p| p[0] * t, &x[..1], &y[..1], None, vec![1.0], &lm).is_err());
        assert!(LevenbergMarquardt::new().solve(r, None::<NoJacobian>, vec![]).is_err());
    }
}
