//! Jacobian-free Newton-Krylov for large nonlinear systems `F(x) = 0` (SciPy's
//! `optimize.newton_krylov`): each Newton correction is found by a Krylov solver that needs only
//! products of the Jacobian with vectors, which are approximated by one difference of `F` each.
//!
//! See the book entry [Nonlinear systems](https://jwkunz.github.io/lib_jsl/book/root_finding/nonlinear-systems.html#jacobian-free-newton-krylov).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::sparse::{IterativeOptions, LgmresSolver, LinearOperator, bicgstab, gmres};
use std::cell::{Cell, RefCell};

/// The Krylov solver for the Newton corrections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InnerSolver {
    /// GMRES, one cycle of `inner_maxiter` steps (SciPy's `method='gmres'`).
    Gmres,
    /// BiCGSTAB, at most `inner_maxiter` steps of two products each (`method='bicgstab'`).
    Bicgstab,
    /// LGMRES (SciPy's default inner solver): one cycle of 30 Krylov steps, augmented by up to 10
    /// earlier Newton corrections, which are kept from one Newton iteration to the next and put
    /// first in the subspace (SciPy's `outer_k=10`, `prepend_outer_v=True`,
    /// `store_outer_Av=False`). `inner_maxiter` does not apply.
    Lgmres,
}

/// Options for [`newton_krylov`] (SciPy's defaults). The iteration stops when
/// `‖F‖∞ ≤ f_tol` and `‖F‖∞ ≤ f_rtol ‖F(x₀)‖∞`, and the last correction `dx` satisfies
/// `‖dx‖∞ ≤ x_tol` and `‖dx‖∞ ≤ x_rtol ‖x‖∞`.
#[derive(Clone, Copy, Debug)]
pub struct NewtonKrylovOptions {
    /// Default `ε^{1/3} ≈ 6 × 10⁻⁶`.
    pub f_tol: f64,
    pub f_rtol: f64,
    pub x_tol: f64,
    pub x_rtol: f64,
    /// The most Newton iterations; default `100 (n + 1)`.
    pub max_iter: Option<usize>,
    /// Steps of the inner Krylov solver per Newton iteration.
    pub inner_maxiter: usize,
    /// The relative difference step; default `√ε`.
    pub rdiff: Option<f64>,
    /// Backtrack along the Newton direction to reduce `‖F‖²` (SciPy's Armijo search).
    pub line_search: bool,
    pub inner: InnerSolver,
}

impl Default for NewtonKrylovOptions {
    fn default() -> Self {
        NewtonKrylovOptions {
            f_tol: f64::EPSILON.powf(1.0 / 3.0),
            f_rtol: f64::INFINITY,
            x_tol: f64::INFINITY,
            x_rtol: f64::INFINITY,
            max_iter: None,
            inner_maxiter: 20,
            rdiff: None,
            line_search: true,
            inner: InnerSolver::Gmres,
        }
    }
}

/// The result of [`newton_krylov`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct NewtonKrylovResult {
    pub x: Vec<f64>,
    /// `F(x)`.
    pub fun: Vec<f64>,
    /// Newton iterations, counted as SciPy's `nit` (convergence tests made).
    pub iterations: usize,
    /// Evaluations of `F`, including those inside the Jacobian-vector products.
    pub nfev: usize,
    /// Products of the approximate Jacobian with a vector.
    pub jacobian_products: usize,
    pub converged: bool,
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn max_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// A residual function `F: ℝⁿ → ℝⁿ`.
type Residual<'a> = &'a mut dyn FnMut(&[f64]) -> Vec<f64>;

/// `F` with an evaluation counter, shared by the Newton loop and the Jacobian operator.
struct Counted<'a> {
    f: RefCell<Residual<'a>>,
    calls: Cell<usize>,
}

impl Counted<'_> {
    fn eval(&self, x: &[f64]) -> Vec<f64> {
        self.calls.set(self.calls.get() + 1);
        (self.f.borrow_mut())(x)
    }
}

// ANCHOR: jacobian
/// The Jacobian at `x` seen only through products: `J v ≈ (F(x + h v/‖v‖) − F(x)) ‖v‖ / h`, with
/// `h = rdiff · max(1, ‖x‖∞) / max(1, ‖F(x)‖∞)` (SciPy's `KrylovJacobian`).
struct FiniteDifferenceJacobian<'a, 'b> {
    f: &'b Counted<'a>,
    x: Vec<f64>,
    fx: Vec<f64>,
    omega: f64,
    products: Cell<usize>,
}

impl LinearOperator for FiniteDifferenceJacobian<'_, '_> {
    fn shape(&self) -> (usize, usize) {
        (self.x.len(), self.x.len())
    }
    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        let nv = norm2(v);
        if nv == 0.0 {
            return vec![0.0; v.len()];
        }
        self.products.set(self.products.get() + 1);
        let sc = self.omega / nv;
        let xs: Vec<f64> = self.x.iter().zip(v).map(|(x, vi)| x + sc * vi).collect();
        self.f.eval(&xs).iter().zip(&self.fx).map(|(a, b)| (a - b) / sc).collect()
    }
    fn rmatvec(&self, _: &[f64]) -> Vec<f64> {
        unimplemented!("the transposed Jacobian is not available without derivatives")
    }
}
// ANCHOR_END: jacobian

/// SciPy's `scalar_search_armijo`: the step `α` with `φ(α) ≤ φ(0) + c₁ α φ'(0)`, trying 1, then
/// the minimiser of a quadratic, then of cubics, down to `α_min`.
fn armijo(mut phi: impl FnMut(f64) -> f64, phi0: f64, derphi0: f64, amin: f64) -> Option<f64> {
    let c1 = 1e-4;
    let alpha0 = 1.0;
    let mut phi_a0 = phi(alpha0);
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0 {
        return Some(alpha0);
    }
    let mut alpha0 = alpha0;
    let mut alpha1 = -derphi0 * alpha0 * alpha0 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0);
    let mut phi_a1 = phi(alpha1);
    if phi_a1 <= phi0 + c1 * alpha1 * derphi0 {
        return Some(alpha1);
    }
    while alpha1 > amin {
        let factor = alpha0 * alpha0 * alpha1 * alpha1 * (alpha1 - alpha0);
        let a = (alpha0 * alpha0 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1 * alpha1 * (phi_a0 - phi0 - derphi0 * alpha0)) / factor;
        let b = (-alpha0.powi(3) * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1.powi(3) * (phi_a0 - phi0 - derphi0 * alpha0)) / factor;
        let mut alpha2 = (-b + (b * b - 3.0 * a * derphi0).abs().sqrt()) / (3.0 * a);
        let phi_a2 = phi(alpha2);
        if phi_a2 <= phi0 + c1 * alpha2 * derphi0 {
            return Some(alpha2);
        }
        if alpha1 - alpha2 > alpha1 / 2.0 || 1.0 - alpha2 / alpha1 < 0.96 {
            alpha2 = alpha1 / 2.0;
        }
        alpha0 = alpha1;
        alpha1 = alpha2;
        phi_a0 = phi_a1;
        phi_a1 = phi_a2;
    }
    None
}

// ANCHOR: newton_krylov
/// Solves `F(x) = 0` for large `n` without forming the Jacobian (SciPy's `newton_krylov`). Each
/// Newton iteration solves `J dx = −F(x)` inexactly with a Krylov method, to the relative
/// accuracy `η` of the Eisenstat-Walker forcing sequence (choice 2, `γ = 0.9`), using finite-
/// difference products `J v`; an Armijo line search on `‖F‖²` then sets the step. An optional
/// preconditioner `M ≈ J⁻¹` (SciPy's `inner_M`) applies to the inner solves.
pub fn newton_krylov<F>(mut f: F, x0: &[f64], preconditioner: Option<&dyn LinearOperator>, options: &NewtonKrylovOptions) -> Result<NewtonKrylovResult, ErrorsJSL>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("newton_krylov: the starting point must not be empty"));
    }
    let counted = Counted { f: RefCell::new(&mut f), calls: Cell::new(0) };
    let mut x = x0.to_vec();
    let mut fx = counted.eval(&x);
    if fx.len() != n {
        return Err(ErrorsJSL::InvalidInputRange("newton_krylov: F must map ℝⁿ to ℝⁿ"));
    }
    let max_iter = options.max_iter.unwrap_or(100 * (n + 1));
    let rdiff = options.rdiff.unwrap_or(f64::EPSILON.sqrt());
    let mut fx_norm = norm2(&fx);
    let f0_norm = max_norm(&fx);
    let mut dx = vec![f64::INFINITY; n];
    let (gamma, eta_max, eta_threshold): (f64, f64, f64) = (0.9, 0.9999, 0.1);
    let mut eta: f64 = 1e-3;
    let mut augmented = LgmresSolver::new(30, 10);
    augmented.store_outer_av = false;
    augmented.prepend_outer_v = true;
    let (mut checks, mut products, mut converged) = (0, 0, false);
    for _ in 0..max_iter {
        checks += 1;
        let (f_inf, dx_inf) = (max_norm(&fx), max_norm(&dx));
        if f_inf == 0.0 || (f_inf <= options.f_tol && f_inf / options.f_rtol <= f0_norm && dx_inf <= options.x_tol && dx_inf / options.x_rtol <= max_norm(&x)) {
            converged = true;
            break;
        }
        // Inexact Newton direction.
        let omega = rdiff * max_norm(&x).max(1.0) / max_norm(&fx).max(1.0);
        let jac = FiniteDifferenceJacobian { f: &counted, x: x.clone(), fx: fx.clone(), omega, products: Cell::new(0) };
        let tol = eta.min(eta * fx_norm);
        let inner = IterativeOptions { rtol: tol, atol: 0.0, max_iterations: Some(options.inner_maxiter) };
        let sol = match options.inner {
            InnerSolver::Gmres => gmres(&jac, &fx, None, preconditioner, options.inner_maxiter, inner)?,
            InnerSolver::Bicgstab => bicgstab(&jac, &fx, None, preconditioner, inner)?,
            InnerSolver::Lgmres => augmented.solve(&jac, &fx, None, preconditioner, IterativeOptions { max_iterations: Some(1), ..inner })?,
        };
        products += jac.products.get();
        dx = sol.x.iter().map(|v| -v).collect();
        if norm2(&dx) == 0.0 {
            return Err(ErrorsJSL::Message("newton_krylov: the inner solve returned a zero step".into()));
        }
        // Line search on φ(s) = ‖F(x + s dx)‖².
        let (s, x_new, f_new) = if options.line_search {
            let mut last: Option<(f64, Vec<f64>)> = None;
            let s = armijo(
                |s| {
                    let xt: Vec<f64> = x.iter().zip(&dx).map(|(a, d)| a + s * d).collect();
                    let v = counted.eval(&xt);
                    let p = if v.iter().all(|t| t.is_finite()) { norm2(&v).powi(2) } else { f64::INFINITY };
                    last = Some((s, v));
                    p
                },
                fx_norm * fx_norm,
                -fx_norm * fx_norm,
                1e-2,
            )
            .unwrap_or(1.0);
            let x_new: Vec<f64> = x.iter().zip(&dx).map(|(a, d)| a + s * d).collect();
            let f_new = match last {
                Some((ls, v)) if ls == s => v,
                _ => counted.eval(&x_new),
            };
            (s, x_new, f_new)
        } else {
            let x_new: Vec<f64> = x.iter().zip(&dx).map(|(a, d)| a + d).collect();
            let f_new = counted.eval(&x_new);
            (1.0, x_new, f_new)
        };
        let _ = s;
        let fx_norm_new = norm2(&f_new);
        x = x_new;
        fx = f_new;
        // Eisenstat-Walker forcing term.
        let eta_a = gamma * fx_norm_new * fx_norm_new / (fx_norm * fx_norm);
        eta = if gamma * eta * eta < eta_threshold { eta_max.min(eta_a) } else { eta_max.min(eta_a.max(gamma * eta * eta)) };
        fx_norm = fx_norm_new;
    }
    let nfev = counted.calls.get();
    Ok(NewtonKrylovResult { x, fun: fx, iterations: checks, nfev, jacobian_products: products, converged })
}
// ANCHOR_END: newton_krylov

#[cfg(test)]
mod tests {
    use super::*;

    /// The Bratu problem `Δu + λ eᵘ = 0` on the unit square with zero boundary values, by
    /// 5-point differences on an `m × m` interior grid.
    fn bratu(m: usize, lambda: f64) -> impl FnMut(&[f64]) -> Vec<f64> {
        let h2 = 1.0 / ((m + 1) * (m + 1)) as f64;
        move |u: &[f64]| {
            let at = |i: isize, j: isize| if i < 0 || j < 0 || i >= m as isize || j >= m as isize { 0.0 } else { u[i as usize * m + j as usize] };
            (0..m * m)
                .map(|k| {
                    let (i, j) = ((k / m) as isize, (k % m) as isize);
                    (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - 4.0 * at(i, j)) / h2 + lambda * at(i, j).exp()
                })
                .collect()
        }
    }

    #[test]
    fn solves_the_bratu_problem() {
        let m = 20;
        for inner in [InnerSolver::Gmres, InnerSolver::Bicgstab, InnerSolver::Lgmres] {
            let o = NewtonKrylovOptions { inner, inner_maxiter: 60, ..Default::default() };
            let r = newton_krylov(bratu(m, 6.0), &vec![0.0; m * m], None, &o).unwrap();
            assert!(r.converged, "{inner:?}: {} iterations", r.iterations);
            assert!(max_norm(&r.fun) <= o.f_tol);
            // The lower solution at λ = 6 peaks at 0.792975 on this grid (SciPy agrees).
            let peak = r.x.iter().cloned().fold(0.0, f64::max);
            assert!((peak - 0.792_974_648).abs() < 1e-6, "{peak}");
        }
    }

    #[test]
    fn small_system_and_errors() {
        // x² + y² = 4, x y = 1.
        let f = |v: &[f64]| vec![v[0] * v[0] + v[1] * v[1] - 4.0, v[0] * v[1] - 1.0];
        let o = NewtonKrylovOptions { f_tol: 1e-12, ..Default::default() };
        let r = newton_krylov(f, &[2.0, 0.5], None, &o).unwrap();
        assert!(r.converged);
        assert!((r.x[0] * r.x[0] + r.x[1] * r.x[1] - 4.0).abs() < 1e-11 && (r.x[0] * r.x[1] - 1.0).abs() < 1e-11);
        assert!(newton_krylov(f, &[], None, &o).is_err());
        assert!(newton_krylov(|v: &[f64]| vec![v[0]], &[1.0, 2.0], None, &o).is_err());
    }
}
