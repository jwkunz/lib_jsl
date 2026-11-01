//! Roots of nonlinear systems `F(x) = 0`, `F: ℝⁿ → ℝⁿ`.
//!
//! Both solvers are globalised by a backtracking line search on `½‖F‖²` (Numerical Recipes'
//! `newt` and `broydn`), so they converge from much farther away than plain iteration:
//!
//! * [`newton_system`]: Newton's method with an analytic or finite-difference Jacobian;
//!   quadratic convergence, one Jacobian per iteration.
//! * [`broyden_system`]: Broyden's "good" rank-one update of the Jacobian; superlinear
//!   convergence with a single function evaluation per iteration, re-forming the Jacobian by
//!   finite differences only when the update stops producing descent.

/// Settings for the system solvers.
#[derive(Debug, Clone, Copy)]
pub struct SystemOptions {
    /// Converged when `‖F‖∞` falls below this.
    pub function_tolerance: f64,
    /// Or when the relative step falls below this.
    pub step_tolerance: f64,
    pub max_iterations: usize,
}

impl Default for SystemOptions {
    fn default() -> Self {
        SystemOptions { function_tolerance: 1e-12, step_tolerance: 1e-14, max_iterations: 200 }
    }
}

/// The outcome of a system solve.
#[derive(Debug, Clone)]
pub struct SystemRootResult {
    pub x: Vec<f64>,
    /// `‖F(x)‖∞`.
    pub residual: f64,
    pub iterations: usize,
    pub function_evaluations: usize,
    pub converged: bool,
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, a| m.max(a.abs()))
}

fn half_norm_squared(v: &[f64]) -> f64 {
    0.5 * v.iter().map(|a| a * a).sum::<f64>()
}

/// Solves `A x = b` by Gaussian elimination with partial pivoting (`None` if singular).
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[p][k] == 0.0 || !a[p][k].is_finite() {
            return None;
        }
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..n {
            let m = a[i][k] / a[k][k];
            let (top, rest) = a.split_at_mut(i);
            rest[0][k..].iter_mut().zip(&top[k][k..]).for_each(|(x, y)| *x -= m * y);
            b[i] -= m * b[k];
        }
    }
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i][j] * b[j]).sum();
        b[i] = (b[i] - s) / a[i][i];
    }
    Some(b)
}

struct Counted<'a, F> {
    f: &'a F,
    evaluations: usize,
}

impl<F: Fn(&[f64]) -> Vec<f64>> Counted<'_, F> {
    fn eval(&mut self, x: &[f64]) -> Vec<f64> {
        self.evaluations += 1;
        (self.f)(x)
    }

    fn jacobian(&mut self, x: &[f64], fx: &[f64]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut jac = vec![vec![0.0; n]; n];
        let mut z = x.to_vec();
        for k in 0..n {
            let h = f64::EPSILON.sqrt() * x[k].abs().max(1.0);
            z[k] = x[k] + h;
            let fz = self.eval(&z);
            z[k] = x[k];
            for i in 0..n {
                jac[i][k] = (fz[i] - fx[i]) / h;
            }
        }
        jac
    }
}

/// Backtracking (quadratic, then cubic, interpolation) along `p` until
/// `½‖F‖²` satisfies the Armijo condition. Returns the new point, its residual and whether the
/// search succeeded.
fn line_search<F: Fn(&[f64]) -> Vec<f64>>(sys: &mut Counted<F>, x: &[f64], f0: f64, slope: f64, p: &[f64]) -> (Vec<f64>, Vec<f64>, bool) {
    let max_step = 100.0 * x.iter().map(|v| v * v).sum::<f64>().sqrt().max(x.len() as f64);
    let norm_p = p.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if norm_p > max_step { max_step / norm_p } else { 1.0 };
    let p: Vec<f64> = p.iter().map(|v| v * scale).collect();
    let slope = slope * scale;
    let (mut lambda, mut previous_lambda, mut previous_f) = (1.0, 1.0, f0);
    let min_lambda = 1e-12 / p.iter().zip(x).map(|(pi, xi)| pi.abs() / xi.abs().max(1.0)).fold(1e-300, f64::max);
    loop {
        let trial: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + lambda * b).collect();
        let ft = sys.eval(&trial);
        let value = half_norm_squared(&ft);
        if value.is_finite() && value <= f0 + 1e-4 * lambda * slope {
            return (trial, ft, true);
        }
        if lambda < min_lambda {
            return (trial, ft, false);
        }
        let next = if !value.is_finite() {
            0.1 * lambda
        } else if lambda == 1.0 {
            -slope / (2.0 * (value - f0 - slope))
        } else {
            let r1 = value - f0 - lambda * slope;
            let r2 = previous_f - f0 - previous_lambda * slope;
            let a = (r1 / (lambda * lambda) - r2 / (previous_lambda * previous_lambda)) / (lambda - previous_lambda);
            let b = (-previous_lambda * r1 / (lambda * lambda) + lambda * r2 / (previous_lambda * previous_lambda)) / (lambda - previous_lambda);
            let t = if a == 0.0 {
                -slope / (2.0 * b)
            } else {
                let disc = b * b - 3.0 * a * slope;
                if disc < 0.0 { 0.5 * lambda } else if b <= 0.0 { (-b + disc.sqrt()) / (3.0 * a) } else { -slope / (b + disc.sqrt()) }
            };
            t.min(0.5 * lambda)
        };
        previous_lambda = lambda;
        previous_f = value;
        lambda = if next.is_finite() { next.max(0.1 * lambda) } else { 0.1 * lambda };
    }
}

fn finish(x: Vec<f64>, fx: &[f64], iterations: usize, evaluations: usize, converged: bool) -> SystemRootResult {
    SystemRootResult { residual: inf_norm(fx), x, iterations, function_evaluations: evaluations, converged }
}

/// Globally convergent Newton's method for `F(x) = 0` from `x0`. With `jacobian = None` the
/// Jacobian is formed by forward differences.
pub fn newton_system<F, J>(f: F, jacobian: Option<J>, x0: Vec<f64>, options: SystemOptions) -> Result<SystemRootResult, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let n = x0.len();
    let mut sys = Counted { f: &f, evaluations: 0 };
    if n == 0 {
        return Err("newton_system: the starting point must not be empty".to_string());
    }
    let mut x = x0;
    let mut fx = sys.eval(&x);
    if fx.len() != n {
        return Err("newton_system: F must map ℝⁿ to ℝⁿ".to_string());
    }
    for iteration in 0..options.max_iterations {
        if inf_norm(&fx) <= options.function_tolerance {
            return Ok(finish(x, &fx, iteration, sys.evaluations, true));
        }
        let jac = match &jacobian {
            Some(j) => j(&x),
            None => sys.jacobian(&x, &fx),
        };
        let Some(step) = solve_dense(jac, fx.iter().map(|v| -v).collect()) else {
            return Ok(finish(x, &fx, iteration, sys.evaluations, false));
        };
        let f0 = half_norm_squared(&fx);
        let (x_new, f_new, ok) = line_search(&mut sys, &x, f0, -2.0 * f0, &step);
        let relative_step = x_new.iter().zip(&x).map(|(a, b)| (a - b).abs() / a.abs().max(1.0)).fold(0.0, f64::max);
        x = x_new;
        fx = f_new;
        if !ok || relative_step < options.step_tolerance {
            let converged = inf_norm(&fx) <= options.function_tolerance.max(1e-8);
            return Ok(finish(x, &fx, iteration + 1, sys.evaluations, converged));
        }
    }
    let converged = inf_norm(&fx) <= options.function_tolerance;
    Ok(finish(x, &fx, options.max_iterations, sys.evaluations, converged))
}

/// Broyden's method for `F(x) = 0` from `x0`: the Jacobian estimate `B` (initially by finite
/// differences) is corrected after each step by `B ← B + ((Δf − B s) sᵀ) / (sᵀ s)`.
pub fn broyden_system<F>(f: F, x0: Vec<f64>, options: SystemOptions) -> Result<SystemRootResult, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut sys = Counted { f: &f, evaluations: 0 };
    if n == 0 {
        return Err("broyden_system: the starting point must not be empty".to_string());
    }
    let mut x = x0;
    let mut fx = sys.eval(&x);
    if fx.len() != n {
        return Err("broyden_system: F must map ℝⁿ to ℝⁿ".to_string());
    }
    let mut b = sys.jacobian(&x, &fx);
    let mut fresh = true;
    for iteration in 0..options.max_iterations {
        if inf_norm(&fx) <= options.function_tolerance {
            return Ok(finish(x, &fx, iteration, sys.evaluations, true));
        }
        let step = solve_dense(b.clone(), fx.iter().map(|v| -v).collect());
        let f0 = half_norm_squared(&fx);
        let result = step.map(|p| {
            // Directional derivative of ½‖F‖² along p using B: Fᵀ B p = −‖F‖² for the exact solve.
            line_search(&mut sys, &x, f0, -2.0 * f0, &p)
        });
        match result {
            Some((x_new, f_new, true)) => {
                let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, c)| a - c).collect();
                let ss: f64 = s.iter().map(|v| v * v).sum();
                let relative_step = s.iter().zip(&x_new).map(|(d, a)| d.abs() / a.abs().max(1.0)).fold(0.0, f64::max);
                if ss > 0.0 {
                    let bs: Vec<f64> = b.iter().map(|row| row.iter().zip(&s).map(|(a, c)| a * c).sum()).collect();
                    for i in 0..n {
                        let u = (f_new[i] - fx[i] - bs[i]) / ss;
                        b[i].iter_mut().zip(&s).for_each(|(bij, sj)| *bij += u * sj);
                    }
                }
                x = x_new;
                fx = f_new;
                fresh = false;
                if relative_step < options.step_tolerance {
                    let converged = inf_norm(&fx) <= options.function_tolerance.max(1e-8);
                    return Ok(finish(x, &fx, iteration + 1, sys.evaluations, converged));
                }
            }
            _ if !fresh => {
                // The secant model went stale: restart from a finite-difference Jacobian.
                b = sys.jacobian(&x, &fx);
                fresh = true;
            }
            _ => return Ok(finish(x, &fx, iteration, sys.evaluations, false)),
        }
    }
    let converged = inf_norm(&fx) <= options.function_tolerance;
    Ok(finish(x, &fx, options.max_iterations, sys.evaluations, converged))
}

#[cfg(test)]
mod tests {
    use super::*;

    type NoJacobian = fn(&[f64]) -> Vec<Vec<f64>>;

    fn rosenbrock(x: &[f64]) -> Vec<f64> {
        vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
    }

    /// Power sums of (1, 2, 3): a root with a non-singular Jacobian (determinant −4).
    fn three(x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 14.0, x[0] + x[1] + x[2] - 6.0, x[0] * x[1] * x[2] - 6.0]
    }

    /// Discretised Bratu problem u'' + e^u = 0 on (0, 1) with zero boundary values.
    fn bratu(u: &[f64]) -> Vec<f64> {
        let n = u.len();
        let h = 1.0 / (n + 1) as f64;
        (0..n)
            .map(|i| {
                let left = if i == 0 { 0.0 } else { u[i - 1] };
                let right = if i + 1 == n { 0.0 } else { u[i + 1] };
                left - 2.0 * u[i] + right + h * h * u[i].exp()
            })
            .collect()
    }

    #[test]
    fn newton_converges_quadratically_with_and_without_a_jacobian() {
        let jac = |x: &[f64]| vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]];
        let r = newton_system(rosenbrock, Some(jac), vec![-1.2, 1.0], SystemOptions::default()).unwrap();
        assert!(r.converged && (r.x[0] - 1.0).abs() < 1e-12 && (r.x[1] - 1.0).abs() < 1e-12, "{r:?}");
        // The full first step overshoots in ‖F‖ (though plain Newton would recover), so the
        // merit-function line search takes a slower but safe path through the valley.
        assert!(r.iterations <= 20, "{}", r.iterations);
        let r = newton_system(three, None::<NoJacobian>, vec![0.7, 2.4, 2.6], SystemOptions::default()).unwrap();
        assert!(r.converged && r.x.iter().zip([1.0, 2.0, 3.0]).all(|(v, w)| (v - w).abs() < 1e-10), "{r:?}");
        let r = newton_system(bratu, None::<NoJacobian>, vec![0.0; 40], SystemOptions::default()).unwrap();
        assert!(r.converged && r.residual < 1e-12 && r.iterations < 10);
    }

    #[test]
    fn broyden_uses_fewer_evaluations_on_a_large_system() {
        let n = 40;
        let newton = newton_system(bratu, None::<NoJacobian>, vec![0.0; n], SystemOptions::default()).unwrap();
        let broyden = broyden_system(bratu, vec![0.0; n], SystemOptions::default()).unwrap();
        assert!(broyden.converged && broyden.residual < 1e-12, "{broyden:?}");
        assert!(newton.x.iter().zip(&broyden.x).all(|(a, b)| (a - b).abs() < 1e-9));
        assert!(broyden.function_evaluations < newton.function_evaluations, "{} vs {}", broyden.function_evaluations, newton.function_evaluations);
        let r = broyden_system(rosenbrock, vec![-1.2, 1.0], SystemOptions::default()).unwrap();
        assert!(r.converged && (r.x[0] - 1.0).abs() < 1e-10, "{r:?}");
    }

    #[test]
    fn line_search_rescues_a_far_start_and_validation() {
        // Pure Newton from x = 10 on atan overshoots and diverges; the line search does not.
        let r = newton_system(|x: &[f64]| vec![x[0].atan()], None::<NoJacobian>, vec![10.0], SystemOptions::default()).unwrap();
        assert!(r.converged && r.x[0].abs() < 1e-12, "{r:?}");
        assert!(newton_system(|x: &[f64]| vec![x[0]], None::<NoJacobian>, vec![], SystemOptions::default()).is_err());
        assert!(broyden_system(|x: &[f64]| vec![x[0], 1.0], vec![1.0], SystemOptions::default()).is_err());
        // x² + 1 has no real root: reported as not converged.
        assert!(!newton_system(|x: &[f64]| vec![x[0] * x[0] + 1.0], None::<NoJacobian>, vec![0.5], SystemOptions::default()).unwrap().converged);
    }
}
