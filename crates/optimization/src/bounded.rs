//! Optimisation with simple bounds `l ≤ x ≤ u`, following `scipy.optimize`:
//!
//! * [`minimize_lbfgsb`]: the limited-memory quasi-Newton method L-BFGS-B of Byrd, Lu, Nocedal
//!   and Zhu (1995): a generalised Cauchy point along the projected gradient path fixes the active
//!   bounds, a subspace minimisation of the limited-memory model moves the free variables, and a
//!   line search capped at the bounds finishes the step.
//! * [`least_squares_bounded`]: nonlinear least squares with bounds and robust loss functions by
//!   the trust-region reflective method of Branch, Coleman and Li (a port of SciPy's
//!   `least_squares(method='trf')` with the exact trust-region solver).
//! * [`lsq_linear`]: linear least squares with bounds by the bounded-variable least-squares
//!   algorithm of Stark and Parker (SciPy's `lsq_linear(method='bvls')`).
//! * [`nnls`]: non-negative least squares by the active-set method of Lawson and Hanson.
//!
//! See the book entry [Bound-constrained optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/bounded.html).

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// The linear-algebra loops read most clearly with explicit indices.
#![allow(clippy::needless_range_loop)]

use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::{Inverse, LeastSquaresSvd, SVD};
use ndarray::{Array1, Array2, Axis, s};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn check_bounds(x0: &[f64], lb: &[f64], ub: &[f64]) -> Result<(), ErrorsJSL> {
    if lb.len() != x0.len() || ub.len() != x0.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((lb.len(), x0.len())));
    }
    if lb.iter().zip(ub).any(|(l, u)| !(l < u)) {
        return Err(ErrorsJSL::InvalidInputRange("each lower bound must be less than its upper bound"));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------ L-BFGS-B

/// Options for [`minimize_lbfgsb`], with SciPy's defaults.
#[derive(Clone, Copy, Debug)]
pub struct LbfgsbOptions {
    /// The number of correction pairs kept (`maxcor`).
    pub memory: usize,
    /// Stop when `(f_k − f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) ≤ ftol`.
    pub ftol: f64,
    /// Stop when the largest component of the projected gradient is at most `gtol`.
    pub gtol: f64,
    pub max_iterations: usize,
}

impl Default for LbfgsbOptions {
    fn default() -> Self {
        LbfgsbOptions { memory: 10, ftol: 2.220_446_049_250_313e-9, gtol: 1e-5, max_iterations: 15000 }
    }
}

/// The result of a bound-constrained minimisation.
#[derive(Clone, Debug)]
pub struct BoundedResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub gradient: Vec<f64>,
    pub iterations: usize,
    pub function_evaluations: usize,
    /// `true` when a convergence test was met.
    pub converged: bool,
    /// −1, 0, +1 per variable: at its lower bound, free, at its upper bound.
    pub active: Vec<i8>,
}

fn project(x: &mut [f64], lb: &[f64], ub: &[f64]) {
    for i in 0..x.len() {
        x[i] = x[i].clamp(lb[i], ub[i]);
    }
}

/// `‖P(x − g) − x‖∞`, the size of the projected gradient.
fn projected_gradient_norm(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> f64 {
    (0..x.len()).map(|i| ((x[i] - g[i]).clamp(lb[i], ub[i]) - x[i]).abs()).fold(0.0, f64::max)
}

/// The limited-memory matrix `B = θI − W M Wᵀ` in compact form.
struct Compact {
    /// W = [Y, θS], n × 2k, stored by rows.
    w: Vec<Vec<f64>>,
    /// M, 2k × 2k.
    m: Array2<f64>,
    theta: f64,
}

impl Compact {
    fn new(s: &[Vec<f64>], y: &[Vec<f64>], theta: f64, n: usize) -> Result<Self, ErrorsJSL> {
        let k = s.len();
        let w: Vec<Vec<f64>> = (0..n).map(|i| (0..2 * k).map(|j| if j < k { y[j][i] } else { theta * s[j - k][i] }).collect()).collect();
        if k == 0 {
            return Ok(Compact { w, m: Array2::zeros((0, 0)), theta });
        }
        // M⁻¹ = [[−D, Lᵀ], [L, θ SᵀS]].
        let mut minv = Array2::<f64>::zeros((2 * k, 2 * k));
        for i in 0..k {
            for j in 0..k {
                let sy = dot(&s[i], &y[j]);
                if i == j {
                    minv[(i, i)] = -sy;
                }
                if i > j {
                    minv[(k + i, j)] = sy;
                    minv[(j, k + i)] = sy;
                }
                minv[(k + i, k + j)] = theta * dot(&s[i], &s[j]);
            }
        }
        let m = minv.inv().map_err(|_| ErrorsJSL::RuntimeError("L-BFGS-B: singular limited-memory matrix"))?;
        Ok(Compact { w, m, theta })
    }

    fn two_k(&self) -> usize {
        self.m.nrows()
    }

    fn m_times(&self, v: &[f64]) -> Vec<f64> {
        (0..self.two_k()).map(|i| (0..self.two_k()).map(|j| self.m[(i, j)] * v[j]).sum()).collect()
    }
}

// ANCHOR: cauchy
/// The generalised Cauchy point: the first local minimiser of the quadratic model along the
/// projected steepest-descent path `P(x − t g)`, found by visiting the breakpoints where variables
/// reach their bounds in order (Byrd, Lu, Nocedal and Zhu, algorithm CP). Returns the point and
/// `c = Wᵀ(x_cp − x)`.
fn cauchy_point(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64], b: &Compact) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let two_k = b.two_k();
    let mut t = vec![f64::INFINITY; n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        if g[i] < 0.0 {
            t[i] = (x[i] - ub[i]) / g[i];
        } else if g[i] > 0.0 {
            t[i] = (x[i] - lb[i]) / g[i];
        }
        if t[i] > 0.0 {
            d[i] = -g[i];
        }
    }
    let mut order: Vec<usize> = (0..n).filter(|&i| t[i] > 0.0 && t[i].is_finite()).collect();
    order.sort_by(|&a, &c| t[a].total_cmp(&t[c]));
    let mut xcp = x.to_vec();
    for i in 0..n {
        if t[i] == 0.0 {
            xcp[i] = x[i];
        }
    }
    // p = Wᵀ d, c = 0.
    let mut p: Vec<f64> = (0..two_k).map(|j| (0..n).map(|i| b.w[i][j] * d[i]).sum()).collect();
    let mut c = vec![0.0; two_k];
    let mut fp = -dot(&d, &d);
    let mut fpp = -b.theta * fp - dot(&p, &b.m_times(&p));
    let mut dt_min = if fpp > 0.0 { -fp / fpp } else { f64::INFINITY };
    let mut t_old = 0.0;
    for &bi in &order {
        let dt = t[bi] - t_old;
        if dt_min < dt {
            break;
        }
        // Variable bi reaches its bound.
        xcp[bi] = if d[bi] > 0.0 { ub[bi] } else { lb[bi] };
        let zb = xcp[bi] - x[bi];
        for j in 0..two_k {
            c[j] += dt * p[j];
        }
        let gb = g[bi];
        let wb = &b.w[bi];
        let mc = b.m_times(&c);
        let mp = b.m_times(&p);
        let mw = b.m_times(wb);
        fp += dt * fpp + gb * gb + b.theta * gb * zb - gb * dot(wb, &mc);
        fpp += -b.theta * gb * gb - 2.0 * gb * dot(wb, &mp) - gb * gb * dot(wb, &mw);
        for j in 0..two_k {
            p[j] += gb * wb[j];
        }
        d[bi] = 0.0;
        dt_min = if fpp > 0.0 { -fp / fpp } else { f64::INFINITY };
        t_old = t[bi];
    }
    let dt_min = if dt_min.is_finite() { dt_min.max(0.0) } else { 0.0 };
    let t_final = t_old + dt_min;
    for i in 0..n {
        if d[i] != 0.0 {
            xcp[i] = (x[i] + t_final * d[i]).clamp(lb[i], ub[i]);
        }
    }
    for j in 0..two_k {
        c[j] += dt_min * p[j];
    }
    (xcp, c)
}
// ANCHOR_END: cauchy

/// Subspace minimisation (the direct primal method): minimises the model over the variables that
/// are free at the Cauchy point, by the Sherman-Morrison-Woodbury formula for the reduced
/// matrix, and truncates the step at the bounds.
fn subspace_min(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64], xcp: &[f64], c: &[f64], b: &Compact) -> Result<Vec<f64>, ErrorsJSL> {
    let n = x.len();
    let two_k = b.two_k();
    let free: Vec<usize> = (0..n).filter(|&i| xcp[i] > lb[i] && xcp[i] < ub[i]).collect();
    if free.is_empty() || two_k == 0 {
        // With no curvature pairs the model is θ/2 ‖·‖²: minimise it over the free variables directly.
        let mut out = xcp.to_vec();
        if two_k == 0 {
            for &i in &free {
                out[i] = (xcp[i] - (g[i] + b.theta * (xcp[i] - x[i])) / b.theta).clamp(lb[i], ub[i]);
            }
        }
        return Ok(out);
    }
    let mc = b.m_times(c);
    // Reduced gradient r = Zᵀ (g + θ (x_cp − x) − W M c).
    let r: Vec<f64> = free.iter().map(|&i| g[i] + b.theta * (xcp[i] - x[i]) - dot(&b.w[i], &mc)).collect();
    // v = M W_Fᵀ r;  N = I − (1/θ) M W_Fᵀ W_F;  v ← N⁻¹ v;  d = −r/θ − W_F v / θ².
    let wtr: Vec<f64> = (0..two_k).map(|j| free.iter().zip(&r).map(|(&i, ri)| b.w[i][j] * ri).sum()).collect();
    let v = b.m_times(&wtr);
    let mut wtw = Array2::<f64>::zeros((two_k, two_k));
    for &i in &free {
        for a in 0..two_k {
            for c2 in 0..two_k {
                wtw[(a, c2)] += b.w[i][a] * b.w[i][c2];
            }
        }
    }
    let nmat = Array2::<f64>::eye(two_k) - b.m.dot(&wtw) / b.theta;
    let v = nmat.inv().map_err(|_| ErrorsJSL::RuntimeError("L-BFGS-B: singular subspace matrix"))?.dot(&Array1::from(v));
    let du: Vec<f64> = free.iter().zip(&r).map(|(&i, ri)| -ri / b.theta - dot(&b.w[i], v.as_slice().expect("contiguous")) / (b.theta * b.theta)).collect();
    // Truncate the step to stay feasible.
    let mut alpha: f64 = 1.0;
    for (k, &i) in free.iter().enumerate() {
        if du[k] > 0.0 {
            alpha = alpha.min((ub[i] - xcp[i]) / du[k]);
        } else if du[k] < 0.0 {
            alpha = alpha.min((lb[i] - xcp[i]) / du[k]);
        }
    }
    let mut out = xcp.to_vec();
    for (k, &i) in free.iter().enumerate() {
        out[i] = xcp[i] + alpha * du[k];
    }
    project(&mut out, lb, ub);
    Ok(out)
}

/// A line search along `d` for the strong Wolfe conditions (`c₁ = 10⁻³`, `c₂ = 0.9`, as L-BFGS-B),
/// with the step limited to `max_step` so that the iterate stays feasible.
/// An accepted line-search point: step, x, f(x), ∇f(x).
type LinePoint = (f64, Vec<f64>, f64, Vec<f64>);
/// A function returning a value and its gradient.
type ValueAndGradient<'a> = dyn FnMut(&[f64]) -> (f64, Vec<f64>) + 'a;

fn line_search(fg: &mut ValueAndGradient, x: &[f64], f0: f64, g0: &[f64], d: &[f64], first: f64, max_step: f64) -> Option<LinePoint> {
    let (c1, c2) = (1e-3, 0.9);
    let slope0 = dot(g0, d);
    if !(slope0 < 0.0) {
        return None;
    }
    let point = |a: f64| -> Vec<f64> { x.iter().zip(d).map(|(xi, di)| xi + a * di).collect() };
    let (mut lo, mut f_lo, mut slope_lo) = (0.0, f0, slope0);
    let (mut hi, mut f_hi) = (f64::INFINITY, f64::INFINITY);
    let mut a = first.min(max_step);
    for _ in 0..40 {
        let xa = point(a);
        let (fa, ga) = fg(&xa);
        let slope = dot(&ga, d);
        if !fa.is_finite() || fa > f0 + c1 * a * slope0 || fa >= f_lo {
            (hi, f_hi) = (a, fa);
        } else {
            if slope.abs() <= -c2 * slope0 || (a >= max_step && slope < 0.0) {
                return Some((a, xa, fa, ga));
            }
            if slope * (hi - lo) >= 0.0 && hi.is_finite() {
                (hi, f_hi) = (lo, f_lo);
            }
            (lo, f_lo, slope_lo) = (a, fa, slope);
        }
        a = if hi.is_finite() {
            // Minimiser of the quadratic through (lo, f_lo, slope_lo) and (hi, f_hi), kept away from
            // the ends of the bracket.
            let h = hi - lo;
            let curv = f_hi - f_lo - slope_lo * h;
            let trial = if curv > 0.0 && f_hi.is_finite() { lo - 0.5 * slope_lo * h * h / curv } else { lo + 0.5 * h };
            let (l, u) = (lo.min(hi), lo.max(hi));
            trial.clamp(l + 0.1 * (u - l), u - 0.1 * (u - l))
        } else if a >= max_step {
            break;
        } else {
            (4.0 * a).min(max_step)
        };
        if hi.is_finite() && (hi - lo).abs() < 1e-14 * a.abs().max(1.0) {
            break;
        }
    }
    if lo > 0.0 {
        let xa = point(lo);
        let (fa, ga) = fg(&xa);
        return Some((lo, xa, fa, ga));
    }
    None
}

// ANCHOR: lbfgsb
/// Minimises `f` subject to `lb ≤ x ≤ ub` by L-BFGS-B (SciPy's `minimize(method='L-BFGS-B')`).
/// `fg` returns the value and gradient. Infinite bounds are allowed.
pub fn minimize_lbfgsb(mut fg: impl FnMut(&[f64]) -> (f64, Vec<f64>), x0: &[f64], lb: &[f64], ub: &[f64], options: LbfgsbOptions) -> Result<BoundedResult, ErrorsJSL> {
    check_bounds(x0, lb, ub)?;
    let n = x0.len();
    let mut nfev = 0;
    let mut eval = |x: &[f64]| {
        nfev += 1;
        fg(x)
    };
    let mut x = x0.to_vec();
    project(&mut x, lb, ub);
    let (mut f, mut g) = eval(&x);
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut theta = 1.0;
    let mut iterations = 0;
    let mut converged = projected_gradient_norm(&x, &g, lb, ub) <= options.gtol;
    while !converged && iterations < options.max_iterations {
        let b = Compact::new(&s_hist, &y_hist, theta, n)?;
        let (xcp, c) = cauchy_point(&x, &g, lb, ub, &b);
        let xbar = subspace_min(&x, &g, lb, ub, &xcp, &c, &b)?;
        let mut d: Vec<f64> = xbar.iter().zip(&x).map(|(a, b)| a - b).collect();
        if dot(&d, &g) >= 0.0 {
            // Not a descent direction (rounding in the model): restart from steepest descent.
            s_hist.clear();
            y_hist.clear();
            theta = 1.0;
            let b = Compact::new(&s_hist, &y_hist, theta, n)?;
            let (xcp, _) = cauchy_point(&x, &g, lb, ub, &b);
            d = xcp.iter().zip(&x).map(|(a, b)| a - b).collect();
            if dot(&d, &g) >= 0.0 {
                converged = true;
                break;
            }
        }
        // The largest feasible step along d.
        let mut max_step = f64::INFINITY;
        for i in 0..n {
            if d[i] > 0.0 {
                max_step = max_step.min((ub[i] - x[i]) / d[i]);
            } else if d[i] < 0.0 {
                max_step = max_step.min((lb[i] - x[i]) / d[i]);
            }
        }
        let first = if iterations == 0 && s_hist.is_empty() { (1.0 / norm(&d)).min(max_step) } else { 1.0f64.min(max_step) };
        let Some((_, mut x_new, f_new, g_new)) = line_search(&mut eval, &x, f, &g, &d, first, max_step) else { break };
        project(&mut x_new, lb, ub);
        iterations += 1;
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let (f_old, sy, yy) = (f, dot(&s, &y), dot(&y, &y));
        (x, f, g) = (x_new, f_new, g_new);
        if sy > f64::EPSILON * yy {
            if s_hist.len() == options.memory {
                s_hist.remove(0);
                y_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            theta = yy / sy;
        }
        if (f_old - f) / f_old.abs().max(f.abs()).max(1.0) <= options.ftol || projected_gradient_norm(&x, &g, lb, ub) <= options.gtol {
            converged = true;
        }
    }
    let active = (0..n).map(|i| if x[i] <= lb[i] { -1 } else if x[i] >= ub[i] { 1 } else { 0 }).collect();
    Ok(BoundedResult { x, fun: f, gradient: g, iterations, function_evaluations: nfev, converged, active })
}
// ANCHOR_END: lbfgsb

// ------------------------------------------------------------------------- least_squares (TRF)

/// Robust loss functions `ρ(z)` of the squared residual `z = (f / f_scale)²` (SciPy's `loss`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// `ρ(z) = z`: ordinary least squares.
    Linear,
    /// `ρ(z) = 2(√(1 + z) − 1)`.
    SoftL1,
    /// `ρ(z) = z` for `z ≤ 1`, `2√z − 1` beyond.
    Huber,
    /// `ρ(z) = ln(1 + z)`.
    Cauchy,
    /// `ρ(z) = arctan(z)`.
    Arctan,
}

impl Loss {
    /// `(ρ, ρ', ρ'')` at `z`.
    fn rho(self, z: f64) -> [f64; 3] {
        match self {
            Loss::Linear => [z, 1.0, 0.0],
            Loss::SoftL1 => {
                let t = 1.0 + z;
                [2.0 * (t.sqrt() - 1.0), t.powf(-0.5), -0.5 * t.powf(-1.5)]
            }
            Loss::Huber => {
                if z <= 1.0 {
                    [z, 1.0, 0.0]
                } else {
                    [2.0 * z.sqrt() - 1.0, z.powf(-0.5), -0.5 * z.powf(-1.5)]
                }
            }
            Loss::Cauchy => {
                let t = 1.0 + z;
                [z.ln_1p(), 1.0 / t, -1.0 / (t * t)]
            }
            Loss::Arctan => {
                let t = 1.0 + z * z;
                [z.atan(), 1.0 / t, -2.0 * z / (t * t)]
            }
        }
    }
}

/// Options for [`least_squares_bounded`], with SciPy's defaults.
#[derive(Clone, Copy, Debug)]
pub struct LeastSquaresOptions {
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    /// The largest number of residual evaluations; `100 n` when `None`.
    pub max_nfev: Option<usize>,
    pub loss: Loss,
    /// The residual size at which the robust loss starts to differ from ordinary least squares.
    pub f_scale: f64,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        LeastSquaresOptions { ftol: 1e-8, xtol: 1e-8, gtol: 1e-8, max_nfev: None, loss: Loss::Linear, f_scale: 1.0 }
    }
}

/// The result of [`least_squares_bounded`].
#[derive(Clone, Debug)]
pub struct BoundedLeastSquaresResult {
    pub x: Vec<f64>,
    /// `½ Σ ρ(fᵢ²)` (with the loss and scale applied).
    pub cost: f64,
    /// The residuals at the solution.
    pub fun: Vec<f64>,
    /// The gradient of the cost.
    pub grad: Vec<f64>,
    /// The first-order optimality `‖v ∘ g‖∞` with the Coleman-Li scaling `v`.
    pub optimality: f64,
    pub active: Vec<i8>,
    pub nfev: usize,
    pub njev: usize,
    /// SciPy's status: 1 gtol, 2 ftol, 3 xtol, 4 ftol and xtol, 0 the evaluation limit.
    pub status: i32,
}

/// The Coleman-Li scaling vector `v` and its derivative `dv` (SciPy's `CL_scaling_vector`).
fn cl_scaling(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let (mut v, mut dv) = (vec![1.0; n], vec![0.0; n]);
    for i in 0..n {
        if g[i] < 0.0 && ub[i].is_finite() {
            v[i] = ub[i] - x[i];
            dv[i] = -1.0;
        } else if g[i] > 0.0 && lb[i].is_finite() {
            v[i] = x[i] - lb[i];
            dv[i] = 1.0;
        }
    }
    (v, dv)
}

/// The step length to the nearest bound along `s`, and for each variable whether (and in which
/// direction) it is the one that hits (SciPy's `step_size_to_bound`).
fn step_size_to_bound(x: &[f64], s: &[f64], lb: &[f64], ub: &[f64]) -> (f64, Vec<i8>) {
    let steps: Vec<f64> = (0..x.len()).map(|i| if s[i] != 0.0 { ((lb[i] - x[i]) / s[i]).max((ub[i] - x[i]) / s[i]) } else { f64::INFINITY }).collect();
    let min = steps.iter().copied().fold(f64::INFINITY, f64::min);
    let hits = (0..x.len()).map(|i| if steps[i] == min { s[i].signum() as i8 } else { 0 }).collect();
    (min, hits)
}

fn in_bounds(x: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    (0..x.len()).all(|i| x[i] >= lb[i] && x[i] <= ub[i])
}

/// Moves points on or outside the bounds strictly inside (SciPy's `make_strictly_feasible`).
fn make_strictly_feasible(x: &[f64], lb: &[f64], ub: &[f64], rstep: f64) -> Vec<f64> {
    let mut out = x.to_vec();
    for i in 0..x.len() {
        let lower = x[i] <= lb[i] + if rstep == 0.0 { 0.0 } else { rstep * lb[i].abs().max(1.0) };
        let upper = x[i] >= ub[i] - if rstep == 0.0 { 0.0 } else { rstep * ub[i].abs().max(1.0) };
        if lower && lb[i].is_finite() {
            out[i] = if rstep == 0.0 { next_toward(lb[i], ub[i]) } else { lb[i] + rstep * lb[i].abs().max(1.0) };
        } else if upper && ub[i].is_finite() {
            out[i] = if rstep == 0.0 { next_toward(ub[i], lb[i]) } else { ub[i] - rstep * ub[i].abs().max(1.0) };
        }
        if out[i] < lb[i] || out[i] > ub[i] {
            out[i] = 0.5 * (lb[i] + ub[i]);
        }
    }
    out
}

fn next_toward(a: f64, b: f64) -> f64 {
    if b > a { a.next_up() } else { a.next_down() }
}

/// `½ ‖J s‖² + ½ sᵀ diag s + gᵀ s`.
fn evaluate_quadratic(j: &Array2<f64>, g: &[f64], s: &[f64], diag: &[f64]) -> f64 {
    let js = j.dot(&Array1::from(s.to_vec()));
    0.5 * (js.dot(&js) + s.iter().zip(diag).map(|(a, d)| a * a * d).sum::<f64>()) + dot(s, g)
}

/// The coefficients of the quadratic `a t² + b t + c` along `s0 + t s` (SciPy's
/// `build_quadratic_1d`).
fn build_quadratic_1d(j: &Array2<f64>, g: &[f64], s: &[f64], diag: &[f64], s0: Option<&[f64]>) -> (f64, f64, f64) {
    let v = j.dot(&Array1::from(s.to_vec()));
    let a = 0.5 * (v.dot(&v) + s.iter().zip(diag).map(|(a, d)| a * a * d).sum::<f64>());
    let mut b = dot(g, s);
    let mut c = 0.0;
    if let Some(s0) = s0 {
        let u = j.dot(&Array1::from(s0.to_vec()));
        b += u.dot(&v) + s0.iter().zip(diag).zip(s).map(|((a, d), b)| a * d * b).sum::<f64>();
        c = 0.5 * u.dot(&u) + dot(g, s0) + 0.5 * s0.iter().zip(diag).map(|(a, d)| a * a * d).sum::<f64>();
    }
    (a, b, c)
}

/// The minimiser of `a t² + b t + c` on `[lo, hi]` and its value.
fn minimize_quadratic_1d(a: f64, b: f64, lo: f64, hi: f64, c: f64) -> (f64, f64) {
    let mut ts = vec![lo, hi];
    if a != 0.0 {
        let ext = -0.5 * b / a;
        if lo < ext && ext < hi {
            ts.push(ext);
        }
    }
    let mut best = (ts[0], ts[0] * (a * ts[0] + b) + c);
    for &t in &ts[1..] {
        let y = t * (a * t + b) + c;
        if y < best.1 {
            best = (t, y);
        }
    }
    best
}

/// The parameter values where `x + t s` crosses the sphere of radius `Δ` (SciPy's
/// `intersect_trust_region`).
fn intersect_trust_region(x: &[f64], s: &[f64], delta: f64) -> (f64, f64) {
    let a = dot(s, s);
    let b = dot(x, s);
    let c = dot(x, x) - delta * delta;
    let d = (b * b - a * c).max(0.0).sqrt();
    let q = -(b + d.copysign(b));
    let (t1, t2) = (q / a, c / q);
    if t1 < t2 { (t1, t2) } else { (t2, t1) }
}

/// The exact trust-region subproblem `min ‖J p + f‖` with `‖p‖ ≤ Δ` from the SVD of `J`, by
/// Moré's iteration on the Levenberg-Marquardt parameter (SciPy's `solve_lsq_trust_region`).
fn solve_lsq_trust_region(n: usize, m: usize, uf: &[f64], s: &[f64], v: &Array2<f64>, delta: f64, initial_alpha: f64) -> (Vec<f64>, f64) {
    let suf: Vec<f64> = s.iter().zip(uf).map(|(a, b)| a * b).collect();
    let full_rank = m >= n && s[n - 1] > f64::EPSILON * m as f64 * s[0];
    let phi = |alpha: f64| -> (f64, f64) {
        let denom: Vec<f64> = s.iter().map(|si| si * si + alpha).collect();
        let p: Vec<f64> = suf.iter().zip(&denom).map(|(a, d)| a / d).collect();
        let pn = norm(&p);
        (pn - delta, -suf.iter().zip(&denom).map(|(a, d)| a * a / (d * d * d)).sum::<f64>() / pn)
    };
    let vdot = |w: &[f64]| -> Vec<f64> { (0..n).map(|i| (0..n).map(|k| v[(i, k)] * w[k]).sum()).collect() };
    if full_rank {
        let p: Vec<f64> = vdot(&uf.iter().zip(s).map(|(a, b)| -a / b).collect::<Vec<_>>());
        if norm(&p) <= delta {
            return (p, 0.0);
        }
    }
    let mut alpha_upper = norm(&suf) / delta;
    let mut alpha_lower = if full_rank {
        let (ph, dp) = phi(0.0);
        -ph / dp
    } else {
        0.0
    };
    let mut alpha = if !full_rank && initial_alpha == 0.0 { (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt()) } else { initial_alpha };
    for _ in 0..10 {
        if alpha < alpha_lower || alpha > alpha_upper {
            alpha = (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt());
        }
        let (ph, dp) = phi(alpha);
        if ph < 0.0 {
            alpha_upper = alpha;
        }
        let ratio = ph / dp;
        alpha_lower = alpha_lower.max(alpha - ratio);
        alpha -= (ph + delta) * ratio / delta;
        if ph.abs() < 0.01 * delta {
            break;
        }
    }
    let mut p = vdot(&suf.iter().zip(s).map(|(a, si)| -a / (si * si + alpha)).collect::<Vec<_>>());
    let pn = norm(&p);
    p.iter_mut().for_each(|v| *v *= delta / pn);
    (p, alpha)
}

/// Chooses among the trust-region step (reflected or truncated at a bound) and the scaled
/// gradient step, the one with the lowest model value (SciPy's `select_step`).
#[allow(clippy::too_many_arguments)]
fn select_step(x: &[f64], j_h: &Array2<f64>, diag_h: &[f64], g_h: &[f64], p: &[f64], p_h: &[f64], d: &[f64], delta: f64, lb: &[f64], ub: &[f64], theta: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let n = x.len();
    let xp: Vec<f64> = x.iter().zip(p).map(|(a, b)| a + b).collect();
    if in_bounds(&xp, lb, ub) {
        return (p.to_vec(), p_h.to_vec(), -evaluate_quadratic(j_h, g_h, p_h, diag_h));
    }
    let (p_stride, hits) = step_size_to_bound(x, p, lb, ub);
    // The reflected direction.
    let mut r_h = p_h.to_vec();
    for i in 0..n {
        if hits[i] != 0 {
            r_h[i] = -r_h[i];
        }
    }
    let mut r: Vec<f64> = r_h.iter().zip(d).map(|(a, b)| a * b).collect();
    let mut p = p.iter().map(|v| v * p_stride).collect::<Vec<_>>();
    let mut p_h = p_h.iter().map(|v| v * p_stride).collect::<Vec<_>>();
    let x_on_bound: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
    let (_, to_tr) = intersect_trust_region(&p_h, &r_h, delta);
    let (to_bound, _) = step_size_to_bound(&x_on_bound, &r, lb, ub);
    let r_stride = to_bound.min(to_tr);
    let (r_stride_l, r_stride_u) = if r_stride > 0.0 { ((1.0 - theta) * p_stride / r_stride, if r_stride == to_bound { theta * to_bound } else { to_tr }) } else { (0.0, -1.0) };
    let r_value;
    if r_stride_l <= r_stride_u {
        let (a, b, c) = build_quadratic_1d(j_h, g_h, &r_h, diag_h, Some(&p_h));
        let (rs, rv) = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c);
        r_value = rv;
        r_h = r_h.iter().zip(&p_h).map(|(a, b)| a * rs + b).collect();
        r = r_h.iter().zip(d).map(|(a, b)| a * b).collect();
    } else {
        r_value = f64::INFINITY;
    }
    p.iter_mut().for_each(|v| *v *= theta);
    p_h.iter_mut().for_each(|v| *v *= theta);
    let p_value = evaluate_quadratic(j_h, g_h, &p_h, diag_h);
    let mut ag_h: Vec<f64> = g_h.iter().map(|v| -v).collect();
    let mut ag: Vec<f64> = ag_h.iter().zip(d).map(|(a, b)| a * b).collect();
    let to_tr = delta / norm(&ag_h);
    let (to_bound, _) = step_size_to_bound(x, &ag, lb, ub);
    let ag_stride = if to_bound < to_tr { theta * to_bound } else { to_tr };
    let (a, b, _) = build_quadratic_1d(j_h, g_h, &ag_h, diag_h, None);
    let (ag_stride, ag_value) = minimize_quadratic_1d(a, b, 0.0, ag_stride, 0.0);
    ag_h.iter_mut().for_each(|v| *v *= ag_stride);
    ag.iter_mut().for_each(|v| *v *= ag_stride);
    if p_value < r_value && p_value < ag_value {
        (p, p_h, -p_value)
    } else if r_value < p_value && r_value < ag_value {
        (r, r_h, -r_value)
    } else {
        (ag, ag_h, -ag_value)
    }
}

// ANCHOR: trf
/// Minimises `½ Σ ρ(fᵢ(x)²)` subject to `lb ≤ x ≤ ub` by the trust-region reflective method
/// (SciPy's `least_squares(method='trf', tr_solver='exact', x_scale=1)`). `fun` returns the
/// residuals and `jac` their Jacobian (rows = residuals).
///
/// Each iteration scales the variables by the Coleman-Li vector `v` (the distance to the bound
/// that the gradient points at), so that the trust region shrinks near the bounds; solves the
/// scaled trust-region problem exactly through an SVD; and then takes the best of that step
/// truncated at the bound, its reflection off the bound, and a scaled gradient step.
pub fn least_squares_bounded(
    fun: impl Fn(&[f64]) -> Vec<f64>,
    jac: impl Fn(&[f64]) -> Array2<f64>,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    options: LeastSquaresOptions,
) -> Result<BoundedLeastSquaresResult, ErrorsJSL> {
    check_bounds(x0, lb, ub)?;
    if !in_bounds(x0, lb, ub) {
        return Err(ErrorsJSL::InvalidInputRange("the initial guess is outside the bounds"));
    }
    let n = x0.len();
    let mut x = make_strictly_feasible(x0, lb, ub, 1e-10);
    let mut f = fun(&x);
    let m = f.len();
    let mut j = jac(&x);
    if j.dim() != (m, n) {
        return Err(ErrorsJSL::IncompatibleArraySizes((j.nrows(), m)));
    }
    let (mut nfev, mut njev) = (1, 1);
    let max_nfev = options.max_nfev.unwrap_or(100 * n);
    let fs = options.f_scale;
    // Robust loss: cost and the rescaled residuals and Jacobian.
    let loss_cost = |f: &[f64]| 0.5 * fs * fs * f.iter().map(|v| options.loss.rho((v / fs).powi(2))[0]).sum::<f64>();
    let scale_for_loss = |j: &mut Array2<f64>, f: &mut Vec<f64>| {
        if options.loss == Loss::Linear {
            return;
        }
        for i in 0..f.len() {
            let r = options.loss.rho((f[i] / fs).powi(2));
            let (r1, r2) = (r[1], r[2] / (fs * fs));
            let js = (r1 + 2.0 * r2 * f[i] * f[i]).max(f64::EPSILON).sqrt();
            f[i] *= r1 / js;
            j.row_mut(i).mapv_inplace(|v| v * js);
        }
    };
    let mut f_true = f.clone();
    let mut cost = loss_cost(&f);
    scale_for_loss(&mut j, &mut f);
    let grad = |j: &Array2<f64>, f: &[f64]| -> Vec<f64> { j.t().dot(&Array1::from(f.to_vec())).to_vec() };
    let mut g = grad(&j, &f);
    let (v0, _) = cl_scaling(&x, &g, lb, ub);
    let mut delta = norm(&x.iter().zip(&v0).map(|(a, b)| a / b.sqrt()).collect::<Vec<_>>());
    if delta == 0.0 {
        delta = 1.0;
    }
    let mut alpha = 0.0;
    let mut status = None;
    loop {
        let (v, dv) = cl_scaling(&x, &g, lb, ub);
        let g_norm = g.iter().zip(&v).map(|(a, b)| (a * b).abs()).fold(0.0, f64::max);
        if g_norm < options.gtol {
            status = Some(1);
        }
        if status.is_some() || nfev == max_nfev {
            break;
        }
        let d: Vec<f64> = v.iter().map(|vi| vi.sqrt()).collect();
        let diag_h: Vec<f64> = g.iter().zip(&dv).map(|(a, b)| a * b).collect();
        let g_h: Vec<f64> = d.iter().zip(&g).map(|(a, b)| a * b).collect();
        let mut j_h = j.clone();
        for c in 0..n {
            j_h.column_mut(c).mapv_inplace(|x| x * d[c]);
        }
        // Augmented system [J_h; diag(√diag_h)] and its SVD.
        let mut j_aug = Array2::<f64>::zeros((m + n, n));
        j_aug.slice_mut(s![..m, ..]).assign(&j_h);
        for i in 0..n {
            j_aug[(m + i, i)] = diag_h[i].max(0.0).sqrt();
        }
        let mut f_aug = vec![0.0; m + n];
        f_aug[..m].copy_from_slice(&f);
        let (u, sv, vt) = j_aug.svd(true, true).map_err(|_| ErrorsJSL::RuntimeError("least_squares: SVD failed"))?;
        let (u, vt) = (u.expect("requested"), vt.expect("requested"));
        let k = sv.len();
        let uf: Vec<f64> = (0..k).map(|c| (0..m + n).map(|r| u[(r, c)] * f_aug[r]).sum()).collect();
        let vmat = vt.t().slice(s![.., ..k]).to_owned();
        let theta = 0.995f64.max(1.0 - g_norm);
        let mut actual_reduction = -1.0;
        let (mut x_new, mut f_new, mut cost_new) = (x.clone(), f.clone(), cost);
        while actual_reduction <= 0.0 && nfev < max_nfev {
            let (p_h, a) = solve_lsq_trust_region(n, m, &uf, sv.as_slice().expect("contiguous"), &vmat, delta, alpha);
            alpha = a;
            let p: Vec<f64> = d.iter().zip(&p_h).map(|(a, b)| a * b).collect();
            let (step, step_h, predicted) = select_step(&x, &j_h, &diag_h, &g_h, &p, &p_h, &d, delta, lb, ub, theta);
            x_new = make_strictly_feasible(&x.iter().zip(&step).map(|(a, b)| a + b).collect::<Vec<_>>(), lb, ub, 0.0);
            f_new = fun(&x_new);
            nfev += 1;
            let step_h_norm = norm(&step_h);
            if f_new.iter().any(|v| !v.is_finite()) {
                delta = 0.25 * step_h_norm;
                continue;
            }
            cost_new = loss_cost(&f_new);
            actual_reduction = cost - cost_new;
            let ratio = if predicted > 0.0 {
                actual_reduction / predicted
            } else if predicted == 0.0 && actual_reduction == 0.0 {
                1.0
            } else {
                0.0
            };
            let delta_new = if ratio < 0.25 {
                0.25 * step_h_norm
            } else if ratio > 0.75 && step_h_norm > 0.95 * delta {
                2.0 * delta
            } else {
                delta
            };
            let step_norm = norm(&step);
            let ftol_ok = actual_reduction < options.ftol * cost && ratio > 0.25;
            let xtol_ok = step_norm < options.xtol * (options.xtol + norm(&x));
            status = match (ftol_ok, xtol_ok) {
                (true, true) => Some(4),
                (true, false) => Some(2),
                (false, true) => Some(3),
                _ => None,
            };
            if status.is_some() {
                break;
            }
            alpha *= delta / delta_new;
            delta = delta_new;
        }
        if actual_reduction > 0.0 {
            x = x_new;
            f = f_new;
            f_true = f.clone();
            cost = cost_new;
            j = jac(&x);
            njev += 1;
            scale_for_loss(&mut j, &mut f);
            g = grad(&j, &f);
        }
    }
    let (v, _) = cl_scaling(&x, &g, lb, ub);
    let optimality = g.iter().zip(&v).map(|(a, b)| (a * b).abs()).fold(0.0, f64::max);
    let active = (0..n)
        .map(|i| {
            let tol = 1e-10 * lb[i].abs().max(ub[i].abs()).max(1.0);
            if x[i] - lb[i] <= tol { -1 } else if ub[i] - x[i] <= tol { 1 } else { 0 }
        })
        .collect();
    Ok(BoundedLeastSquaresResult { x, cost, fun: f_true, grad: g, optimality, active, nfev, njev, status: status.unwrap_or(0) })
}
// ANCHOR_END: trf

// ------------------------------------------------------------------------------ linear problems

fn lstsq(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, ErrorsJSL> {
    if a.ncols() == 0 {
        return Ok(Array1::zeros(0));
    }
    Ok(a.least_squares(b).map_err(|_| ErrorsJSL::RuntimeError("least-squares solve failed"))?.solution)
}

fn columns(a: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    a.select(Axis(1), idx)
}

/// The result of [`lsq_linear`] and [`nnls`].
#[derive(Clone, Debug)]
pub struct LinearLsqResult {
    pub x: Vec<f64>,
    /// `½ ‖A x − b‖²`.
    pub cost: f64,
    /// The residual norm `‖A x − b‖`.
    pub residual_norm: f64,
    pub active: Vec<i8>,
    pub iterations: usize,
}

// ANCHOR: nnls
/// Non-negative least squares `min ‖A x − b‖` with `x ≥ 0` (SciPy's `nnls`), by Lawson and Hanson's
/// active-set method: repeatedly free the variable whose gradient most wants to grow, solve the
/// unconstrained problem on the free set, and step back to the boundary when a free variable
/// would turn negative.
pub fn nnls(a: &Array2<f64>, b: &[f64]) -> Result<LinearLsqResult, ErrorsJSL> {
    let (m, n) = a.dim();
    if b.len() != m {
        return Err(ErrorsJSL::IncompatibleArraySizes((b.len(), m)));
    }
    let b = Array1::from(b.to_vec());
    let mut x = Array1::<f64>::zeros(n);
    let mut passive = vec![false; n];
    let tol = 10.0 * f64::EPSILON * a.iter().fold(0.0f64, |s, v| s.max(v.abs())) * m.max(n) as f64;
    let mut iterations = 0;
    loop {
        let w = a.t().dot(&(&b - &a.dot(&x)));
        let candidate = (0..n).filter(|&j| !passive[j]).max_by(|&i, &j| w[i].total_cmp(&w[j]));
        let Some(jmax) = candidate else { break };
        if w[jmax] <= tol || iterations >= 3 * n {
            break;
        }
        passive[jmax] = true;
        loop {
            iterations += 1;
            let p: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
            let sp = lstsq(&columns(a, &p), &b)?;
            if sp.iter().all(|&v| v > 0.0) {
                x.fill(0.0);
                for (k, &j) in p.iter().enumerate() {
                    x[j] = sp[k];
                }
                break;
            }
            // Step from x towards s until the first passive variable reaches zero.
            let mut alpha = f64::INFINITY;
            for (k, &j) in p.iter().enumerate() {
                if sp[k] <= 0.0 {
                    alpha = alpha.min(x[j] / (x[j] - sp[k]));
                }
            }
            for (k, &j) in p.iter().enumerate() {
                x[j] += alpha * (sp[k] - x[j]);
            }
            for &j in &p {
                if x[j] <= tol {
                    x[j] = 0.0;
                    passive[j] = false;
                }
            }
            if iterations > 30 * n.max(1) {
                break;
            }
        }
    }
    let r = &a.dot(&x) - &b;
    let rn = r.dot(&r).sqrt();
    let active = (0..n).map(|j| if x[j] == 0.0 { -1 } else { 0 }).collect();
    Ok(LinearLsqResult { x: x.to_vec(), cost: 0.5 * rn * rn, residual_norm: rn, active, iterations })
}
// ANCHOR_END: nnls

/// Linear least squares with bounds, `min ½‖A x − b‖²` subject to `lb ≤ x ≤ ub` (SciPy's
/// `lsq_linear(method='bvls')`), by Stark and Parker's bounded-variable least squares: start from
/// the unconstrained solution clipped to the bounds, then, while the KKT conditions are violated,
/// free the variable with the most violated condition and solve on the free set, stepping back to
/// the first bound crossed.
pub fn lsq_linear(a: &Array2<f64>, b: &[f64], lb: &[f64], ub: &[f64], tol: f64) -> Result<LinearLsqResult, ErrorsJSL> {
    let (m, n) = a.dim();
    if b.len() != m {
        return Err(ErrorsJSL::IncompatibleArraySizes((b.len(), m)));
    }
    check_bounds(&vec![0.0; n], lb, ub)?;
    let bv = Array1::from(b.to_vec());
    let mut x = lstsq(a, &bv)?.to_vec();
    let mut on_bound = vec![0i8; n];
    for i in 0..n {
        if x[i] <= lb[i] {
            x[i] = lb[i];
            on_bound[i] = -1;
        } else if x[i] >= ub[i] {
            x[i] = ub[i];
            on_bound[i] = 1;
        }
    }
    let residual = |x: &[f64]| -> Array1<f64> { a.dot(&Array1::from(x.to_vec())) - &bv };
    let grad = |x: &[f64]| -> Vec<f64> { a.t().dot(&residual(x)).to_vec() };
    let kkt = |g: &[f64], ob: &[i8]| -> f64 { (0..n).map(|i| if ob[i] == 0 { g[i].abs() } else { g[i] * ob[i] as f64 }).fold(f64::NEG_INFINITY, f64::max) };
    // Initial loop: solve on the free set until no free variable violates its bounds.
    let mut free: Vec<usize> = (0..n).filter(|&i| on_bound[i] == 0).collect();
    let mut iterations = 0;
    while !free.is_empty() {
        iterations += 1;
        let fixed_part: Vec<f64> = (0..n).map(|i| if on_bound[i] != 0 { x[i] } else { 0.0 }).collect();
        let b_free = &bv - &a.dot(&Array1::from(fixed_part));
        let z = lstsq(&columns(a, &free), &b_free)?;
        let mut violated = false;
        let mut still_free = Vec::new();
        for (k, &i) in free.iter().enumerate() {
            if z[k] < lb[i] {
                x[i] = lb[i];
                on_bound[i] = -1;
                violated = true;
            } else if z[k] > ub[i] {
                x[i] = ub[i];
                on_bound[i] = 1;
                violated = true;
            } else {
                x[i] = z[k];
                still_free.push(i);
            }
        }
        if !violated {
            break;
        }
        free = still_free;
    }
    let r = residual(&x);
    let mut cost = 0.5 * r.dot(&r);
    let mut g = grad(&x);
    let (start, max_iter) = (iterations, n + iterations);
    for _ in start..max_iter {
        if kkt(&g, &on_bound) < tol {
            break;
        }
        // Free the variable whose bound is most strongly opposed by the gradient.
        let mv = (0..n).max_by(|&i, &j| (g[i] * on_bound[i] as f64).total_cmp(&(g[j] * on_bound[j] as f64))).expect("non-empty");
        on_bound[mv] = 0;
        loop {
            iterations += 1;
            let free: Vec<usize> = (0..n).filter(|&i| on_bound[i] == 0).collect();
            let fixed_part: Vec<f64> = (0..n).map(|i| if on_bound[i] != 0 { x[i] } else { 0.0 }).collect();
            let b_free = &bv - &a.dot(&Array1::from(fixed_part));
            let z = lstsq(&columns(a, &free), &b_free)?;
            let mut best: Option<(f64, usize, i8)> = None;
            for (k, &i) in free.iter().enumerate() {
                let (alpha, side) = if z[k] < lb[i] {
                    ((lb[i] - x[i]) / (z[k] - x[i]), -1)
                } else if z[k] > ub[i] {
                    ((ub[i] - x[i]) / (z[k] - x[i]), 1)
                } else {
                    continue;
                };
                if best.is_none_or(|(a, _, _)| alpha < a) {
                    best = Some((alpha, i, side));
                }
            }
            match best {
                Some((alpha, i, side)) => {
                    for (k, &fi) in free.iter().enumerate() {
                        x[fi] += alpha * (z[k] - x[fi]);
                    }
                    x[i] = if side < 0 { lb[i] } else { ub[i] };
                    on_bound[i] = side;
                }
                None => {
                    for (k, &fi) in free.iter().enumerate() {
                        x[fi] = z[k];
                    }
                    break;
                }
            }
            if iterations > 50 * n.max(1) {
                break;
            }
        }
        let r = residual(&x);
        let cost_new = 0.5 * r.dot(&r);
        let change = cost - cost_new;
        cost = cost_new;
        g = grad(&x);
        if change < tol * cost {
            break;
        }
    }
    let r = residual(&x);
    let rn = r.dot(&r).sqrt();
    Ok(LinearLsqResult { x, cost: 0.5 * rn * rn, residual_norm: rn, active: on_bound, iterations })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn rosen(x: &[f64]) -> (f64, Vec<f64>) {
        let n = x.len();
        let mut f = 0.0;
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            let (a, b) = (x[i + 1] - x[i] * x[i], 1.0 - x[i]);
            f += 100.0 * a * a + b * b;
            g[i] += -400.0 * x[i] * a - 2.0 * b;
            g[i + 1] += 200.0 * a;
        }
        (f, g)
    }

    #[test]
    fn lbfgsb_unconstrained_and_bounded_rosenbrock() {
        let x0 = vec![-1.2, 1.0, -1.2, 1.0];
        let inf = vec![f64::INFINITY; 4];
        let ninf = vec![f64::NEG_INFINITY; 4];
        let r = minimize_lbfgsb(rosen, &x0, &ninf, &inf, LbfgsbOptions::default()).unwrap();
        assert!(r.x.iter().all(|v| (v - 1.0).abs() < 1e-4), "{:?}", r.x);
        // With x ≤ 0.5 the optimum has x₀ on its bound and the KKT conditions hold.
        let ub = vec![0.5; 4];
        let r = minimize_lbfgsb(rosen, &x0, &ninf, &ub, LbfgsbOptions::default()).unwrap();
        assert!(r.converged);
        assert!(projected_gradient_norm(&r.x, &r.gradient, &ninf, &ub) < 1e-4, "{:?} {:?}", r.x, r.gradient);
        assert!(r.active.contains(&1));
    }

    #[test]
    fn nnls_and_lsq_linear_satisfy_kkt() {
        let a = array![[1.0, 2.0, 0.5], [0.3, -1.0, 2.0], [2.0, 0.1, -0.7], [1.5, 1.5, 1.5]];
        let b = [1.0, -2.0, 3.0, 0.5];
        let r = nnls(&a, &b).unwrap();
        let g = a.t().dot(&(a.dot(&Array1::from(r.x.clone())) - Array1::from(b.to_vec())));
        for j in 0..3 {
            assert!(r.x[j] >= 0.0);
            if r.x[j] > 0.0 {
                assert!(g[j].abs() < 1e-10);
            } else {
                assert!(g[j] >= -1e-10);
            }
        }
        let r = lsq_linear(&a, &b, &[-0.2, -0.2, -0.2], &[0.3, 0.3, 0.3], 1e-12).unwrap();
        let g = a.t().dot(&(a.dot(&Array1::from(r.x.clone())) - Array1::from(b.to_vec())));
        for j in 0..3 {
            match r.active[j] {
                0 => assert!(g[j].abs() < 1e-10),
                -1 => assert!(g[j] >= -1e-10),
                _ => assert!(g[j] <= 1e-10),
            }
        }
    }

    #[test]
    fn trf_fits_an_exponential_with_bounds() {
        // y = 2 e^{−0.7 t}, fitted with the rate bounded above by 0.5: the rate sits on its bound.
        let t: Vec<f64> = (0..20).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = t.iter().map(|&ti| 2.0 * (-0.7 * ti).exp()).collect();
        let fun = |p: &[f64]| t.iter().zip(&y).map(|(&ti, &yi)| p[0] * (-p[1] * ti).exp() - yi).collect::<Vec<_>>();
        let jac = |p: &[f64]| Array2::from_shape_fn((t.len(), 2), |(i, k)| if k == 0 { (-p[1] * t[i]).exp() } else { -p[0] * t[i] * (-p[1] * t[i]).exp() });
        let free = least_squares_bounded(fun, jac, &[1.0, 0.1], &[0.0, 0.0], &[10.0, 5.0], LeastSquaresOptions::default()).unwrap();
        assert!((free.x[0] - 2.0).abs() < 1e-6 && (free.x[1] - 0.7).abs() < 1e-6, "{:?}", free.x);
        let bounded = least_squares_bounded(fun, jac, &[1.0, 0.1], &[0.0, 0.0], &[10.0, 0.5], LeastSquaresOptions::default()).unwrap();
        assert_eq!(bounded.active[1], 1);
        assert!(bounded.optimality < 1e-6, "{}", bounded.optimality);
    }
}
