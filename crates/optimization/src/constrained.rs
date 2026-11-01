//! Nonlinearly constrained minimization:
//! `min f(x)` subject to `h(x) = 0` (equalities) and `g(x) ≤ 0` (inequalities).
//!
//! * [`sqp`]: sequential quadratic programming (Han-Powell). Each iteration solves the quadratic
//!   model `min ½dᵀBd + ∇fᵀd` subject to the linearised constraints, with `B` a damped-BFGS
//!   approximation of the Lagrangian Hessian (Powell's damping keeps it positive definite). The
//!   QP is solved through its dual, a bound-constrained problem in the multipliers, by an active
//!   set method; steps are globalised by a line search on the ℓ₁ merit function. Superlinear
//!   convergence on smooth problems.
//! * [`augmented_lagrangian`]: the method of multipliers. Each outer iteration minimises the
//!   augmented Lagrangian (Rockafellar's form for inequalities) with L-BFGS, then updates the
//!   multipliers and, when the constraint violation stalls, the penalty. Simple and robust,
//!   but converges more slowly.
//!
//! Problems implement [`ConstrainedProblem`]; derivatives default to central differences.
//!
//! See the book entry [Constrained optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/constrained.html).

use crate::optimization_traits::{GradientBasedMinimizationEngine, GradientFunction, MinimizationControls, ObjectiveFunction};
use crate::quasi_newton::Lbfgs;
use lib_jsl_core::ErrorsJSL;

fn central_gradient(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let mut z = x.to_vec();
    (0..x.len())
        .map(|k| {
            let h = 1e-6 * x[k].abs().max(1.0);
            z[k] = x[k] + h;
            let up = f(&z);
            z[k] = x[k] - h;
            let down = f(&z);
            z[k] = x[k];
            (up - down) / (2.0 * h)
        })
        .collect()
}

fn central_jacobian(c: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], m: usize) -> Vec<Vec<f64>> {
    let mut jac = vec![vec![0.0; x.len()]; m];
    let mut z = x.to_vec();
    for k in 0..x.len() {
        let h = 1e-6 * x[k].abs().max(1.0);
        z[k] = x[k] + h;
        let up = c(&z);
        z[k] = x[k] - h;
        let down = c(&z);
        z[k] = x[k];
        for i in 0..m {
            jac[i][k] = (up[i] - down[i]) / (2.0 * h);
        }
    }
    jac
}

/// A constrained minimization problem. Only [`ConstrainedProblem::objective`] is required;
/// constraints default to none and derivatives to central differences.
pub trait ConstrainedProblem {
    fn objective(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        central_gradient(&|z| self.objective(z), x)
    }
    /// Equality constraints `h(x) = 0`.
    fn equalities(&self, _x: &[f64]) -> Vec<f64> {
        Vec::new()
    }
    /// Rows `∇h_i(x)ᵀ`.
    fn equality_jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let m = self.equalities(x).len();
        central_jacobian(&|z| self.equalities(z), x, m)
    }
    /// Inequality constraints `g(x) ≤ 0`.
    fn inequalities(&self, _x: &[f64]) -> Vec<f64> {
        Vec::new()
    }
    /// Rows `∇g_j(x)ᵀ`.
    fn inequality_jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let m = self.inequalities(x).len();
        central_jacobian(&|z| self.inequalities(z), x, m)
    }
}

/// Settings for the constrained solvers.
#[derive(Debug, Clone, Copy)]
pub struct ConstrainedOptions {
    /// Bound on the constraint violation and on the optimality (KKT) residual.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for ConstrainedOptions {
    fn default() -> Self {
        ConstrainedOptions { tolerance: 1e-8, max_iterations: 200 }
    }
}

/// The outcome of a constrained solve.
#[derive(Debug, Clone)]
pub struct ConstrainedResult {
    pub parameters: Vec<f64>,
    pub objective_value: f64,
    /// Lagrange multipliers of the equalities.
    pub equality_multipliers: Vec<f64>,
    /// Lagrange multipliers (≥ 0) of the inequalities; positive ones mark active constraints.
    pub inequality_multipliers: Vec<f64>,
    /// `max(|h|, max(g, 0))` at the solution.
    pub constraint_violation: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn violation(h: &[f64], g: &[f64]) -> f64 {
    h.iter().map(|v| v.abs()).chain(g.iter().map(|v| v.max(0.0))).fold(0.0, f64::max)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// Solves `A x = b` by Gaussian elimination with partial pivoting.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[p][k].abs() < 1e-300 || !a[p][k].is_finite() {
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

/// Minimises `½ yᵀMy + qᵀy` subject to `y_i ≥ 0` for `i ≥ first_bounded` (a convex,
/// bound-constrained QP: the dual of the SQP subproblem) by a primal active-set method.
fn bound_constrained_qp(m: &[Vec<f64>], q: &[f64], first_bounded: usize) -> Option<Vec<f64>> {
    let n = q.len();
    let trace: f64 = (0..n).map(|i| m[i][i]).sum::<f64>().max(1.0);
    let regularisation = 1e-12 * trace; // tolerates redundant (linearly dependent) constraints
    let mut y = vec![0.0; n];
    let mut active: Vec<bool> = (0..n).map(|i| i >= first_bounded).collect();
    for _ in 0..(10 * n + 50) {
        // Solve on the free set with active variables at zero.
        let free: Vec<usize> = (0..n).filter(|&i| !active[i]).collect();
        let sub: Vec<Vec<f64>> = free.iter().map(|&i| free.iter().map(|&j| m[i][j] + if i == j { regularisation } else { 0.0 }).collect()).collect();
        let rhs: Vec<f64> = free.iter().map(|&i| -q[i]).collect();
        let z = if free.is_empty() { Vec::new() } else { solve_dense(sub, rhs)? };
        let mut target = vec![0.0; n];
        free.iter().zip(&z).for_each(|(&i, &v)| target[i] = v);
        // Step towards the free-set optimum, stopping at the first bound that would be crossed.
        let mut alpha: f64 = 1.0;
        let mut blocking = None;
        for &i in &free {
            if i >= first_bounded && target[i] < 0.0 {
                let a = y[i] / (y[i] - target[i]);
                if a < alpha {
                    alpha = a;
                    blocking = Some(i);
                }
            }
        }
        for i in 0..n {
            y[i] += alpha * (target[i] - y[i]);
        }
        if let Some(i) = blocking {
            y[i] = 0.0;
            active[i] = true;
            continue;
        }
        // At the free-set optimum: release the bound with the most negative gradient, if any.
        let gradient: Vec<f64> = (0..n).map(|i| dot(&m[i], &y) + q[i]).collect();
        let release = (first_bounded..n).filter(|&i| active[i] && gradient[i] < -1e-12 * trace).min_by(|&a, &b| gradient[a].total_cmp(&gradient[b]));
        match release {
            Some(i) => active[i] = false,
            None => return Some(y),
        }
    }
    None
}

/// Sequential quadratic programming from `x0` (which need not be feasible).
pub fn sqp<P: ConstrainedProblem>(problem: &P, x0: Vec<f64>, options: ConstrainedOptions) -> Result<ConstrainedResult, ErrorsJSL> {
    let n = x0.len();
    if n == 0 {
        return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
    }
    let mut x = x0;
    let mut b: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let mut penalty: f64 = 1.0;
    let (mut lambda, mut mu) = (Vec::new(), Vec::new());
    let mut converged = false;
    let mut iterations = 0;
    while iterations < options.max_iterations {
        iterations += 1;
        let (f, grad) = (problem.objective(&x), problem.gradient(&x));
        let (h, g) = (problem.equalities(&x), problem.inequalities(&x));
        let (je, ji) = (problem.equality_jacobian(&x), problem.inequality_jacobian(&x));
        let (me, mi) = (h.len(), g.len());
        // Constraint rows A = [∇h; ∇g] and right-hand sides c = [h; g]: A d + c (= 0, ≤ 0).
        let a: Vec<&Vec<f64>> = je.iter().chain(ji.iter()).collect();
        let c: Vec<f64> = h.iter().chain(g.iter()).copied().collect();
        let m = me + mi;
        // B⁻¹ applied to ∇f and to each constraint row.
        let solve_b = |v: &[f64]| solve_dense(b.clone(), v.to_vec());
        let binv_g = solve_b(&grad).ok_or(ErrorsJSL::RuntimeError("The Hessian approximation became singular."))?;
        let binv_a: Vec<Vec<f64>> = a.iter().map(|row| solve_b(row)).collect::<Option<_>>().ok_or(ErrorsJSL::RuntimeError("The Hessian approximation became singular."))?;
        // ANCHOR: sqp_subproblem
        // Dual QP: min ½ yᵀ(A B⁻¹ Aᵀ)y + (A B⁻¹ ∇f − c)ᵀ y, inequality multipliers ≥ 0.
        let dual_m: Vec<Vec<f64>> = (0..m).map(|i| (0..m).map(|j| dot(a[i], &binv_a[j])).collect()).collect();
        let dual_q: Vec<f64> = (0..m).map(|i| dot(a[i], &binv_g) - c[i]).collect();
        let y = bound_constrained_qp(&dual_m, &dual_q, me).ok_or(ErrorsJSL::RuntimeError("The QP subproblem failed (inconsistent linearised constraints?)."))?;
        // d = −B⁻¹(∇f + Aᵀy).
        let d: Vec<f64> = (0..n).map(|k| -(binv_g[k] + (0..m).map(|i| y[i] * binv_a[i][k]).sum::<f64>())).collect();
        // ANCHOR_END: sqp_subproblem
        lambda = y[..me].to_vec();
        mu = y[me..].to_vec();
        let gl_old: Vec<f64> = (0..n).map(|k| grad[k] + (0..m).map(|i| y[i] * a[i][k]).sum::<f64>()).collect();
        let step_norm = d.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        let kkt = gl_old.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        if violation(&h, &g) <= options.tolerance && (step_norm <= options.tolerance * (1.0 + x.iter().fold(0.0f64, |s, v| s.max(v.abs()))) || kkt <= options.tolerance) {
            converged = true;
            break;
        }
        // ℓ₁ merit φ = f + ν (Σ|h| + Σ max(g, 0)) with ν above the multipliers.
        penalty = penalty.max(1.5 * y.iter().fold(0.0f64, |s, v| s.max(v.abs()))).max(1e-3);
        let infeasibility = |h: &[f64], g: &[f64]| h.iter().map(|v| v.abs()).sum::<f64>() + g.iter().map(|v| v.max(0.0)).sum::<f64>();
        let phi0 = f + penalty * infeasibility(&h, &g);
        let derivative = dot(&grad, &d) - penalty * infeasibility(&h, &g);
        let mut t = 1.0;
        let mut x_new;
        loop {
            x_new = x.iter().zip(&d).map(|(a, b)| a + t * b).collect::<Vec<f64>>();
            let phi = problem.objective(&x_new) + penalty * infeasibility(&problem.equalities(&x_new), &problem.inequalities(&x_new));
            if phi.is_finite() && phi <= phi0 + 1e-4 * t * derivative.min(0.0) || t < 1e-10 {
                break;
            }
            t *= 0.5;
        }
        // Damped BFGS update of the Lagrangian Hessian with the new multipliers.
        let grad_new = problem.gradient(&x_new);
        let (je_new, ji_new) = (problem.equality_jacobian(&x_new), problem.inequality_jacobian(&x_new));
        let a_new: Vec<&Vec<f64>> = je_new.iter().chain(ji_new.iter()).collect();
        let gl_new: Vec<f64> = (0..n).map(|k| grad_new[k] + (0..m).map(|i| y[i] * a_new[i][k]).sum::<f64>()).collect();
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(p, q)| p - q).collect();
        let mut yv: Vec<f64> = gl_new.iter().zip(&gl_old).map(|(p, q)| p - q).collect();
        let bs: Vec<f64> = b.iter().map(|row| dot(row, &s)).collect();
        let sbs = dot(&s, &bs);
        let sy = dot(&s, &yv);
        if sbs > 1e-300 {
            if sy < 0.2 * sbs {
                let theta = 0.8 * sbs / (sbs - sy);
                yv = yv.iter().zip(&bs).map(|(p, q)| theta * p + (1.0 - theta) * q).collect();
            }
            let sy = dot(&s, &yv);
            for i in 0..n {
                for j in 0..n {
                    b[i][j] += yv[i] * yv[j] / sy - bs[i] * bs[j] / sbs;
                }
            }
        }
        x = x_new;
    }
    let (h, g) = (problem.equalities(&x), problem.inequalities(&x));
    Ok(ConstrainedResult {
        objective_value: problem.objective(&x),
        constraint_violation: violation(&h, &g),
        parameters: x,
        equality_multipliers: lambda,
        inequality_multipliers: mu,
        iterations,
        converged,
    })
}

/// The augmented Lagrangian `f + λᵀh + ρ/2 ‖h‖² + 1/(2ρ) Σ (max(0, μ + ρg)² − μ²)` and its
/// gradient, as objective and gradient for the inner L-BFGS solve.
struct AugmentedLagrangian<'a, P> {
    problem: &'a P,
    lambda: &'a [f64],
    mu: &'a [f64],
    rho: f64,
}

impl<P: ConstrainedProblem> ObjectiveFunction for AugmentedLagrangian<'_, P> {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let (h, g) = (self.problem.equalities(x), self.problem.inequalities(x));
        let eq: f64 = h.iter().zip(self.lambda).map(|(hi, li)| li * hi + 0.5 * self.rho * hi * hi).sum();
        let ineq: f64 = g.iter().zip(self.mu).map(|(gj, mj)| ((mj + self.rho * gj).max(0.0).powi(2) - mj * mj) / (2.0 * self.rho)).sum();
        self.problem.objective(x) + eq + ineq
    }
}

impl<P: ConstrainedProblem> GradientFunction for AugmentedLagrangian<'_, P> {
    fn evaluate(&self, x: &[f64]) -> Vec<f64> {
        let mut grad = self.problem.gradient(x);
        let (h, g) = (self.problem.equalities(x), self.problem.inequalities(x));
        for (row, (hi, li)) in self.problem.equality_jacobian(x).iter().zip(h.iter().zip(self.lambda)) {
            let w = li + self.rho * hi;
            grad.iter_mut().zip(row).for_each(|(gk, r)| *gk += w * r);
        }
        for (row, (gj, mj)) in self.problem.inequality_jacobian(x).iter().zip(g.iter().zip(self.mu)) {
            let w = (mj + self.rho * gj).max(0.0);
            grad.iter_mut().zip(row).for_each(|(gk, r)| *gk += w * r);
        }
        grad
    }
}

/// The augmented Lagrangian method (method of multipliers) from `x0`.
pub fn augmented_lagrangian<P: ConstrainedProblem>(problem: &P, x0: Vec<f64>, options: ConstrainedOptions) -> Result<ConstrainedResult, ErrorsJSL> {
    if x0.is_empty() {
        return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
    }
    let mut x = x0;
    let mut lambda = vec![0.0; problem.equalities(&x).len()];
    let mut mu = vec![0.0; problem.inequalities(&x).len()];
    let mut rho = 10.0;
    let mut previous_violation = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut inner = Lbfgs::default();
    inner.set_max_iterations(Some(2000));
    while iterations < options.max_iterations {
        iterations += 1;
        inner.set_tolerance(Some((options.tolerance * 0.1).max(1e-12)));
        // ANCHOR: multiplier_update
        let al = AugmentedLagrangian { problem, lambda: &lambda, mu: &mu, rho };
        x = inner.gradient_based_minimize(&al, &al, x.clone())?.parameters;
        let (h, g) = (problem.equalities(&x), problem.inequalities(&x));
        // Inequality violation measured by the complementarity-aware min(−g, μ/ρ).
        let v = h.iter().map(|v| v.abs()).chain(g.iter().zip(&mu).map(|(gj, mj)| gj.max(-mj / rho).abs())).fold(0.0, f64::max);
        lambda.iter_mut().zip(&h).for_each(|(l, hi)| *l += rho * hi);
        mu.iter_mut().zip(&g).for_each(|(m, gj)| *m = (*m + rho * gj).max(0.0));
        // ANCHOR_END: multiplier_update
        if v <= options.tolerance {
            converged = true;
            break;
        }
        if v > 0.25 * previous_violation {
            rho = (rho * 10.0).min(1e10);
        }
        previous_violation = v;
    }
    let (h, g) = (problem.equalities(&x), problem.inequalities(&x));
    Ok(ConstrainedResult {
        objective_value: problem.objective(&x),
        constraint_violation: violation(&h, &g),
        parameters: x,
        equality_multipliers: lambda,
        inequality_multipliers: mu,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hock-Schittkowski problem 71, with its box as eight inequalities.
    struct Hs071;
    impl ConstrainedProblem for Hs071 {
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
        }
        fn gradient(&self, x: &[f64]) -> Vec<f64> {
            vec![x[3] * (2.0 * x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1.0, x[0] * (x[0] + x[1] + x[2])]
        }
        fn equalities(&self, x: &[f64]) -> Vec<f64> {
            vec![x.iter().map(|v| v * v).sum::<f64>() - 40.0]
        }
        fn inequalities(&self, x: &[f64]) -> Vec<f64> {
            let mut g = vec![25.0 - x.iter().product::<f64>()];
            g.extend(x.iter().map(|v| 1.0 - v));
            g.extend(x.iter().map(|v| v - 5.0));
            g
        }
    }
    const HS071_X: [f64; 4] = [1.0, 4.742_999_63, 3.821_149_98, 1.379_408_29];
    const HS071_F: f64 = 17.014_017_3;

    /// A QP with linear inequalities (SciPy's `minimize` tutorial): optimum (1.4, 1.7).
    struct Tutorial;
    impl ConstrainedProblem for Tutorial {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
        }
        fn inequalities(&self, x: &[f64]) -> Vec<f64> {
            vec![-(x[0] - 2.0 * x[1] + 2.0), -(-x[0] - 2.0 * x[1] + 6.0), -(-x[0] + 2.0 * x[1] + 2.0), -x[0], -x[1]]
        }
    }

    /// min x + y on the circle x² + y² = 2: (−1, −1) with multiplier ½.
    struct Circle;
    impl ConstrainedProblem for Circle {
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] + x[1]
        }
        fn equalities(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1] - 2.0]
        }
    }

    fn check(r: &ConstrainedResult, x: &[f64], f: f64, tol: f64, name: &str) {
        assert!(r.converged, "{name}: {r:?}");
        assert!(r.parameters.iter().zip(x).all(|(a, b)| (a - b).abs() < tol), "{name}: {r:?}");
        assert!((r.objective_value - f).abs() < tol && r.constraint_violation < 1e-6, "{name}: {r:?}");
    }

    #[test]
    fn sqp_solves_hs071_a_linear_qp_and_an_equality_problem() {
        let r = sqp(&Hs071, vec![1.0, 5.0, 5.0, 1.0], ConstrainedOptions::default()).unwrap();
        check(&r, &HS071_X, HS071_F, 1e-6, "hs071");
        assert!(r.iterations < 50, "{}", r.iterations);
        // The product constraint and the lower bound on x₁ are active; the others are not.
        assert!(r.inequality_multipliers[0] > 0.0 && r.inequality_multipliers[1] > 0.0 && r.inequality_multipliers[2..].iter().filter(|&&m| m > 1e-8).count() == 0, "{:?}", r.inequality_multipliers);
        let r = sqp(&Tutorial, vec![2.0, 0.0], ConstrainedOptions::default()).unwrap();
        check(&r, &[1.4, 1.7], 0.8, 1e-7, "tutorial");
        let r = sqp(&Circle, vec![0.5, -1.5], ConstrainedOptions::default()).unwrap();
        check(&r, &[-1.0, -1.0], -2.0, 1e-7, "circle");
        assert!((r.equality_multipliers[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn augmented_lagrangian_agrees_with_sqp() {
        let options = ConstrainedOptions { tolerance: 1e-8, ..Default::default() };
        let r = augmented_lagrangian(&Hs071, vec![1.0, 5.0, 5.0, 1.0], options).unwrap();
        check(&r, &HS071_X, HS071_F, 1e-5, "hs071");
        let r = augmented_lagrangian(&Tutorial, vec![2.0, 0.0], options).unwrap();
        check(&r, &[1.4, 1.7], 0.8, 1e-6, "tutorial");
        let r = augmented_lagrangian(&Circle, vec![0.5, -1.5], options).unwrap();
        check(&r, &[-1.0, -1.0], -2.0, 1e-6, "circle");
        assert!((r.equality_multipliers[0] - 0.5).abs() < 1e-5);
        assert!(sqp(&Circle, vec![], options).is_err() && augmented_lagrangian(&Circle, vec![], options).is_err());
    }
}
