//! Bound-constrained minimisation by a truncated Newton method (the method of SciPy's
//! `minimize(method="TNC")`, Nash's truncated Newton with bounds).
//!
//! See the book entry [Levenberg-Marquardt for systems, mixed-integer programming and truncated Newton](https://jwkunz.github.io/lib_jsl/book/optimization/milp-tnc.html#truncated-newton-with-bounds).

use crate::bounded::BoundedResult;
use lib_jsl_core::ErrorsJSL;

/// Options for [`minimize_tnc`].
#[derive(Clone, Copy, Debug)]
pub struct TncOptions {
    /// Stop when the largest component of the projected gradient is at most this.
    pub pgtol: f64,
    /// Also stop when the decrease of `f` in an iteration is at most this times `|f|` (default 0:
    /// only the gradient test and the evaluation limit stop the iteration).
    pub ftol: f64,
    /// The largest number of evaluations of `(f, ∇f)`, including those for Hessian-vector
    /// products; `None` for `max(1000, 100 n)` (SciPy's TNC allows `max(100, 10 n)`, but stops on
    /// a much looser test).
    pub maxfun: Option<usize>,
    /// The largest number of conjugate-gradient iterations per Newton step (default
    /// `min(50, n)`; SciPy's `maxCGit` defaults to `n/2`, which its preconditioner makes up for).
    pub max_cg: Option<usize>,
}

impl Default for TncOptions {
    fn default() -> Self {
        TncOptions { pgtol: 1e-8, ftol: 0.0, maxfun: None, max_cg: None }
    }
}

type DynFg<'a> = &'a mut dyn FnMut(&[f64]) -> (f64, Vec<f64>);

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ANCHOR: tnc
/// Minimises `f` subject to `lb ≤ x ≤ ub` by a truncated Newton method; `fg` returns `f` and its
/// gradient. Each iteration fixes the variables on a bound whose gradient points outwards, and
/// solves the Newton equations `H p = −g` for the others only approximately, by conjugate
/// gradients stopped at a relative residual `min(½, √‖g‖)` or at negative curvature; the
/// Hessian-vector products are forward differences of the gradient, so no Hessian is needed.
/// A projected backtracking line search along `p` keeps the iterates in the box.
pub fn minimize_tnc(mut fg: impl FnMut(&[f64]) -> (f64, Vec<f64>), x0: &[f64], lb: &[f64], ub: &[f64], options: TncOptions) -> Result<BoundedResult, ErrorsJSL> {
    let n = x0.len();
    if n == 0 || lb.len() != n || ub.len() != n {
        return Err(ErrorsJSL::IncompatibleArraySizes((lb.len(), n)));
    }
    if (0..n).any(|i| lb[i].is_nan() || ub[i].is_nan() || lb[i] > ub[i]) {
        return Err(ErrorsJSL::InvalidInputRange("minimize_tnc: need lb ≤ ub"));
    }
    let maxfun = options.maxfun.unwrap_or((100 * n).max(1000));
    let max_cg = options.max_cg.unwrap_or(n.clamp(1, 50));
    let mut x: Vec<f64> = (0..n).map(|i| x0[i].clamp(lb[i], ub[i])).collect();
    let (mut f, mut g) = fg(&x);
    let mut nfev = 1;
    let mut iterations = 0;
    let free_of = |x: &[f64], g: &[f64]| -> Vec<bool> { (0..n).map(|i| !((x[i] <= lb[i] && g[i] > 0.0) || (x[i] >= ub[i] && g[i] < 0.0))).collect() };
    let pg_norm = |x: &[f64], g: &[f64]| -> f64 { (0..n).map(|i| ((x[i] - g[i]).clamp(lb[i], ub[i]) - x[i]).abs()).fold(0.0, f64::max) };
    let mut converged = false;
    while nfev < maxfun {
        if pg_norm(&x, &g) <= options.pgtol {
            converged = true;
            break;
        }
        let free = free_of(&x, &g);
        let gf: Vec<f64> = (0..n).map(|i| if free[i] { g[i] } else { 0.0 }).collect();
        let gnorm = dot(&gf, &gf).sqrt();
        // Truncated conjugate gradients on H p = −g over the free variables.
        let hv = |v: &[f64], fg: DynFg<'_>, x: &[f64], g: &[f64]| -> Vec<f64> {
            let vn = dot(v, v).sqrt();
            let xn = dot(x, x).sqrt();
            let h = f64::EPSILON.sqrt() * (1.0 + xn) / vn.max(1e-300);
            let xh: Vec<f64> = (0..n).map(|i| x[i] + h * v[i]).collect();
            let (_, gh) = fg(&xh);
            (0..n).map(|i| if free[i] { (gh[i] - g[i]) / h } else { 0.0 }).collect()
        };
        let mut p = vec![0.0; n];
        let mut r: Vec<f64> = gf.iter().map(|v| -v).collect();
        let mut d = r.clone();
        let mut rr = dot(&r, &r);
        let eta = 0.5f64.min(gnorm.sqrt());
        for k in 0..max_cg {
            let hd = hv(&d, &mut fg, &x, &g);
            nfev += 1;
            let curv = dot(&d, &hd);
            if curv <= 1e-16 * dot(&d, &d) {
                if k == 0 {
                    p.clone_from(&d);
                }
                break;
            }
            let alpha = rr / curv;
            for i in 0..n {
                p[i] += alpha * d[i];
                r[i] -= alpha * hd[i];
            }
            let rr_new = dot(&r, &r);
            if rr_new.sqrt() <= eta * gnorm {
                break;
            }
            let beta = rr_new / rr;
            rr = rr_new;
            for i in 0..n {
                d[i] = r[i] + beta * d[i];
            }
        }
        if dot(&p, &gf) >= 0.0 {
            p = gf.iter().map(|v| -v).collect();
        }
        // Projected backtracking (Armijo) line search.
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            let xt: Vec<f64> = (0..n).map(|i| (x[i] + step * p[i]).clamp(lb[i], ub[i])).collect();
            let (ft, gt) = fg(&xt);
            nfev += 1;
            let decrease: f64 = (0..n).map(|i| g[i] * (xt[i] - x[i])).sum();
            if ft.is_finite() && ft <= f + 1e-4 * decrease {
                accepted = Some((xt, ft, gt));
                break;
            }
            step *= 0.5;
        }
        iterations += 1;
        let Some((xt, ft, gt)) = accepted else { break };
        let decrease = f - ft;
        let scale = f.abs().max(ft.abs());
        x = xt;
        f = ft;
        g = gt;
        if decrease <= options.ftol * scale {
            converged = pg_norm(&x, &g) <= options.pgtol;
            break;
        }
    }
    let active = (0..n).map(|i| if x[i] <= lb[i] { -1 } else if x[i] >= ub[i] { 1 } else { 0 }).collect();
    Ok(BoundedResult { x, fun: f, gradient: g, iterations, nfev, converged, active })
}
// ANCHOR_END: tnc

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_with_and_without_bounds() {
        let fg = |x: &[f64]| {
            let n = x.len();
            let mut f = 0.0;
            let mut g = vec![0.0; n];
            for i in 0..n - 1 {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                f += 100.0 * a * a + b * b;
                g[i] += -400.0 * x[i] * a - 2.0 * b;
                g[i + 1] += 200.0 * a;
            }
            (f, g)
        };
        let n = 6;
        let x0 = vec![-1.2; n];
        let inf = vec![f64::INFINITY; n];
        let opts = TncOptions::default();
        let r = minimize_tnc(fg, &x0, &vec![f64::NEG_INFINITY; n], &inf, opts).unwrap();
        assert!(r.converged && r.x.iter().all(|v| (v - 1.0).abs() < 1e-6), "{:?}", r.x);
        // With x₀ ≤ 0.5 the first variable sits on its bound; L-BFGS-B agrees on the rest.
        let mut ub = inf.clone();
        ub[0] = 0.5;
        let lb = vec![f64::NEG_INFINITY; n];
        let r = minimize_tnc(fg, &x0, &lb, &ub, opts).unwrap();
        assert_eq!(r.active[0], 1);
        let l = crate::bounded::minimize_lbfgsb(fg, &x0, &lb, &ub, crate::bounded::LbfgsbOptions { gtol: 1e-10, ftol: 1e-15, ..Default::default() }).unwrap();
        assert!(r.x.iter().zip(&l.x).all(|(a, b)| (a - b).abs() < 1e-5), "{:?} vs {:?}", r.x, l.x);
    }
}
