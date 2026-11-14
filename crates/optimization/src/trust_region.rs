//! Trust-region Newton methods, ported from SciPy's `minimize`: `trust-ncg` (the Steihaug
//! conjugate-gradient subproblem, needing only Hessian-vector products), `dogleg` (for positive
//! definite Hessians) and `trust-exact` (Moré and Sorensen's nearly exact subproblem solution).
//!
//! See the book entry [Assignment, trust regions and global optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/assignment-trust-global.html#trust-region-methods).

use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};

/// The subproblem solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustRegionMethod {
    /// Steihaug's truncated conjugate gradients (SciPy's `trust-ncg`).
    Ncg,
    /// Powell's dogleg (SciPy's `dogleg`); the Hessian must be positive definite.
    Dogleg,
    /// Moré-Sorensen iteration on `(H + λI) p = −g` (SciPy's `trust-exact`).
    Exact,
}

/// Options (SciPy's defaults).
#[derive(Clone, Copy, Debug)]
pub struct TrustRegionOptions {
    pub initial_trust_radius: f64,
    pub max_trust_radius: f64,
    /// Accept a step if the actual reduction is more than `eta` of the predicted one.
    pub eta: f64,
    /// Stop when `‖∇f‖ < gtol`.
    pub gtol: f64,
    /// Default `200 n`.
    pub max_iterations: Option<usize>,
}

impl Default for TrustRegionOptions {
    fn default() -> Self {
        TrustRegionOptions { initial_trust_radius: 1.0, max_trust_radius: 1000.0, eta: 0.15, gtol: 1e-4, max_iterations: None }
    }
}

/// The result of [`minimize_trust_region`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TrustRegionResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub jac: Vec<f64>,
    pub iterations: usize,
    pub nfev: usize,
    pub njev: usize,
    pub nhev: usize,
    /// 0 converged, 1 iteration limit, 2 the model failed to predict a decrease, 3 a linear
    /// algebra failure (a non-positive-definite Hessian for `dogleg`).
    pub status: i32,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// The two `t` with `‖z + t d‖ = Δ`, low then high, computed without cancellation.
fn boundaries(z: &[f64], d: &[f64], radius: f64) -> (f64, f64) {
    let a = dot(d, d);
    let b = 2.0 * dot(z, d);
    let c = dot(z, z) - radius * radius;
    let sq = (b * b - 4.0 * a * c).sqrt();
    let aux = b + sq.copysign(b);
    let (ta, tb) = (-aux / (2.0 * a), -2.0 * c / aux);
    if ta <= tb { (ta, tb) } else { (tb, ta) }
}

/// The quadratic model at one point.
struct Model {
    fun: f64,
    jac: Vec<f64>,
    hess: Array2<f64>,
    jac_mag: f64,
    /// State of the exact subproblem kept between solves at the same point.
    lambda_lb: f64,
    previous_radius: f64,
}

impl Model {
    fn value(&self, p: &[f64]) -> f64 {
        let hp = self.hessp(p);
        self.fun + dot(&self.jac, p) + 0.5 * dot(p, &hp)
    }
    fn hessp(&self, p: &[f64]) -> Vec<f64> {
        self.hess.dot(&Array1::from_vec(p.to_vec())).to_vec()
    }

    // ANCHOR: steihaug
    /// Steihaug-CG: conjugate gradients on `H p = −g` from `p = 0`, stopping at the boundary if
    /// a step leaves the region or a direction of negative curvature appears, or when the
    /// residual is below `min(½, √‖g‖) ‖g‖`.
    fn solve_ncg(&self, radius: f64) -> (Vec<f64>, bool) {
        let n = self.jac.len();
        let tolerance = 0.5f64.min(self.jac_mag.sqrt()) * self.jac_mag;
        if self.jac_mag < tolerance {
            return (vec![0.0; n], false);
        }
        let mut z = vec![0.0; n];
        let mut r = self.jac.clone();
        let mut d: Vec<f64> = r.iter().map(|v| -v).collect();
        loop {
            let bd = self.hessp(&d);
            let dbd = dot(&d, &bd);
            if dbd <= 0.0 {
                let (ta, tb) = boundaries(&z, &d, radius);
                let pa: Vec<f64> = z.iter().zip(&d).map(|(a, b)| a + ta * b).collect();
                let pb: Vec<f64> = z.iter().zip(&d).map(|(a, b)| a + tb * b).collect();
                return (if self.value(&pa) < self.value(&pb) { pa } else { pb }, true);
            }
            let r2 = dot(&r, &r);
            let alpha = r2 / dbd;
            let z_next: Vec<f64> = z.iter().zip(&d).map(|(a, b)| a + alpha * b).collect();
            if norm(&z_next) >= radius {
                let (_, tb) = boundaries(&z, &d, radius);
                return (z.iter().zip(&d).map(|(a, b)| a + tb * b).collect(), true);
            }
            let r_next: Vec<f64> = r.iter().zip(&bd).map(|(a, b)| a + alpha * b).collect();
            let rn2 = dot(&r_next, &r_next);
            if rn2.sqrt() < tolerance {
                return (z_next, false);
            }
            let beta = rn2 / r2;
            d = r_next.iter().zip(&d).map(|(a, b)| -a + beta * b).collect();
            z = z_next;
            r = r_next;
        }
    }
    // ANCHOR_END: steihaug

    /// Powell's dogleg: the Newton step if it fits, else the Cauchy point scaled to the boundary
    /// if that is already outside, else the point where the segment between them leaves.
    fn solve_dogleg(&self, radius: f64) -> Option<(Vec<f64>, bool)> {
        let (u, info) = cholesky_upper(&self.hess);
        if info != 0 {
            return None;
        }
        let neg_g: Vec<f64> = self.jac.iter().map(|v| -v).collect();
        let newton = cho_solve(&u, &neg_g);
        if norm(&newton) < radius {
            return Some((newton, false));
        }
        let g = &self.jac;
        let bg = self.hessp(g);
        let cauchy: Vec<f64> = g.iter().map(|v| -(dot(g, g) / dot(g, &bg)) * v).collect();
        let cn = norm(&cauchy);
        if cn >= radius {
            return Some((cauchy.iter().map(|v| v * radius / cn).collect(), true));
        }
        let diff: Vec<f64> = newton.iter().zip(&cauchy).map(|(a, b)| a - b).collect();
        let (_, tb) = boundaries(&cauchy, &diff, radius);
        Some((cauchy.iter().zip(&diff).map(|(a, b)| a + tb * b).collect(), true))
    }

    // ANCHOR: exact
    /// Moré-Sorensen: find `λ ≥ 0` with `H + λI` positive definite and `‖p(λ)‖ = Δ` for
    /// `p(λ) = −(H + λI)⁻¹ g`, by Newton's method on `1/‖p‖ − 1/Δ` safeguarded by bounds on `λ`
    /// (Gershgorin and norm bounds, raised whenever a Cholesky factorisation fails). Inside the
    /// region, a step along an approximate eigenvector of the smallest eigenvalue handles the
    /// "hard case". A port of SciPy's `IterativeSubproblem`, including its reuse of the previous
    /// factorisation after a successful trial one.
    fn solve_exact(&mut self, radius: f64) -> (Vec<f64>, bool) {
        const UPDATE_COEFF: f64 = 0.01;
        const MAXITER: usize = 25;
        let (k_easy, k_hard) = (0.1, 0.2);
        let n = self.jac.len();
        let h = &self.hess;
        let (glb, gub) = gershgorin(h);
        let hinf = h.rows().into_iter().map(|r| r.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
        let hfro = h.iter().map(|v| v * v).sum::<f64>().sqrt();
        let close_to_zero = n as f64 * f64::EPSILON * hinf;
        let mut lambda_ub = (self.jac_mag / radius + (-glb).min(hfro).min(hinf)).max(0.0);
        let min_diag = (0..n).map(|i| h[(i, i)]).fold(f64::INFINITY, f64::min);
        let mut lambda_lb = 0f64.max(-min_diag).max(self.jac_mag / radius - gub.min(hfro).min(hinf));
        if radius < self.previous_radius {
            lambda_lb = lambda_lb.max(self.lambda_lb);
        }
        let mut lambda = if lambda_lb == 0.0 { 0.0 } else { (lambda_lb * lambda_ub).sqrt().max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb)) };
        let shifted = |l: f64| Array2::from_shape_fn((n, n), |(i, j)| h[(i, j)] + if i == j { l } else { 0.0 });
        let mut hits_boundary = true;
        let mut already = false;
        let mut p = vec![0.0; n];
        let (mut u, mut info) = (Array2::zeros((n, n)), 0usize);
        let mut big_h = shifted(lambda);
        for _ in 0..MAXITER {
            if already {
                already = false;
            } else {
                big_h = shifted(lambda);
                let f = cholesky_upper(&big_h);
                u = f.0;
                info = f.1;
            }
            if info == 0 && self.jac_mag > close_to_zero {
                let neg_g: Vec<f64> = self.jac.iter().map(|v| -v).collect();
                p = cho_solve(&u, &neg_g);
                let p_norm = norm(&p);
                if p_norm <= radius && lambda == 0.0 {
                    hits_boundary = false;
                    break;
                }
                let w = solve_upper_transposed(&u, &p);
                let w_norm = norm(&w);
                let delta_lambda = (p_norm / w_norm).powi(2) * (p_norm - radius) / radius;
                let lambda_new = lambda + delta_lambda;
                if p_norm < radius {
                    let (s_min, z_min) = smallest_singular_value(&u);
                    let (ta, tb) = boundaries(&p, &z_min, radius);
                    let step_len = if ta.abs() <= tb.abs() { ta } else { tb };
                    let hp = big_h.dot(&Array1::from_vec(p.clone())).to_vec();
                    let quadratic = dot(&p, &hp);
                    let relative_error = step_len * step_len * s_min * s_min / (quadratic + lambda * radius * radius);
                    if relative_error <= k_hard {
                        p.iter_mut().zip(&z_min).for_each(|(a, b)| *a += step_len * b);
                        break;
                    }
                    lambda_ub = lambda;
                    lambda_lb = lambda_lb.max(lambda - s_min * s_min);
                    big_h = shifted(lambda_new);
                    let (_, info2) = cholesky_upper(&big_h);
                    info = info2;
                    if info2 == 0 {
                        lambda = lambda_new;
                        already = true;
                    } else {
                        lambda_lb = lambda_lb.max(lambda_new);
                        lambda = (lambda_lb * lambda_ub).abs().sqrt().max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb));
                    }
                } else {
                    if (p_norm - radius).abs() / radius <= k_easy {
                        break;
                    }
                    lambda_lb = lambda;
                    lambda = lambda_new;
                }
            } else if info == 0 {
                if lambda == 0.0 {
                    p = vec![0.0; n];
                    hits_boundary = false;
                    break;
                }
                let (s_min, z_min) = smallest_singular_value(&u);
                p = z_min.iter().map(|v| radius * v).collect();
                if radius * radius * s_min * s_min <= k_hard * lambda * radius * radius {
                    break;
                }
                lambda_ub = lambda;
                lambda_lb = lambda_lb.max(lambda - s_min * s_min);
                lambda = (lambda_lb * lambda_ub).sqrt().max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb));
            } else {
                let (delta, v) = singular_leading_submatrix(&big_h, &u, info);
                let vn = norm(&v);
                lambda_lb = lambda_lb.max(lambda + delta / (vn * vn));
                lambda = (lambda_lb * lambda_ub).abs().sqrt().max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb));
            }
        }
        self.lambda_lb = lambda_lb;
        self.previous_radius = radius;
        (p, hits_boundary)
    }
    // ANCHOR_END: exact
}

/// Upper Cholesky `A = UᵀU` row by row (LAPACK's `dpotf2`); `info = k > 0` if the leading minor
/// of order `k` is not positive definite, with rows before `k` computed (and column `k − 1`
/// above the diagonal, which `singular_leading_submatrix` needs).
fn cholesky_upper(a: &Array2<f64>) -> (Array2<f64>, usize) {
    let n = a.nrows();
    let mut u = Array2::zeros((n, n));
    for j in 0..n {
        let s: f64 = (0..j).map(|k| u[(k, j)] * u[(k, j)]).sum();
        let ajj = a[(j, j)] - s;
        if ajj <= 0.0 || ajj.is_nan() {
            return (u, j + 1);
        }
        let ujj = ajj.sqrt();
        u[(j, j)] = ujj;
        for c in j + 1..n {
            let s: f64 = (0..j).map(|k| u[(k, j)] * u[(k, c)]).sum();
            u[(j, c)] = (a[(j, c)] - s) / ujj;
        }
    }
    (u, 0)
}

fn solve_upper(u: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = b.to_vec();
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| u[(i, k)] * x[k]).sum();
        x[i] = (x[i] - s) / u[(i, i)];
    }
    x
}

fn solve_upper_transposed(u: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = b.to_vec();
    for i in 0..n {
        let s: f64 = (0..i).map(|k| u[(k, i)] * x[k]).sum();
        x[i] = (x[i] - s) / u[(i, i)];
    }
    x
}

fn cho_solve(u: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    solve_upper(u, &solve_upper_transposed(u, b))
}

fn gershgorin(h: &Array2<f64>) -> (f64, f64) {
    let n = h.nrows();
    let (mut lb, mut ub) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        let d = h[(i, i)];
        let row: f64 = (0..n).map(|j| h[(i, j)].abs()).sum();
        lb = lb.min(d + d.abs() - row);
        ub = ub.max(d - d.abs() + row);
    }
    (lb, ub)
}

/// An estimate of the smallest singular value of upper-triangular `U` and its right singular
/// vector (Cline, Moler, Stewart and Wilkinson's condition estimator, as in SciPy).
fn smallest_singular_value(u: &Array2<f64>) -> (f64, Vec<f64>) {
    let n = u.nrows();
    let mut p = vec![0.0; n];
    let mut w = vec![0.0; n];
    for k in 0..n {
        let wp = (1.0 - p[k]) / u[(k, k)];
        let wm = (-1.0 - p[k]) / u[(k, k)];
        let pp: Vec<f64> = (k + 1..n).map(|i| p[i] + u[(k, i)] * wp).collect();
        let pm: Vec<f64> = (k + 1..n).map(|i| p[i] + u[(k, i)] * wm).collect();
        let l1 = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>();
        if wp.abs() + l1(&pp) >= wm.abs() + l1(&pm) {
            w[k] = wp;
            p[k + 1..].copy_from_slice(&pp);
        } else {
            w[k] = wm;
            p[k + 1..].copy_from_slice(&pm);
        }
    }
    let v = solve_upper(u, &w);
    let (vn, wn) = (norm(&v), norm(&w));
    (wn / vn, v.iter().map(|x| x / vn).collect())
}

/// `δ ≥ 0` and `v` with `(A + δ e_k e_kᵀ)` singular in its leading `k × k` block, from a failed
/// Cholesky factorisation at step `k`.
fn singular_leading_submatrix(a: &Array2<f64>, u: &Array2<f64>, k: usize) -> (f64, Vec<f64>) {
    let n = a.nrows();
    let delta = (0..k - 1).map(|i| u[(i, k - 1)].powi(2)).sum::<f64>() - a[(k - 1, k - 1)];
    let mut v = vec![0.0; n];
    v[k - 1] = 1.0;
    if k != 1 {
        let sub = Array2::from_shape_fn((k - 1, k - 1), |(i, j)| u[(i, j)]);
        let rhs: Vec<f64> = (0..k - 1).map(|i| -u[(i, k - 1)]).collect();
        let x = solve_upper(&sub, &rhs);
        v[..k - 1].copy_from_slice(&x);
    }
    (delta, v)
}

// ANCHOR: trust_region
/// Minimises `f` with gradient and Hessian by a trust-region Newton method (SciPy's
/// `minimize(method='trust-ncg' | 'dogleg' | 'trust-exact')`). Each iteration minimises the
/// quadratic model within a ball of radius `Δ`, compares the actual reduction with the predicted
/// one, `ρ = (f(x) − f(x + p)) / (m(0) − m(p))`, quarters `Δ` if `ρ < ¼`, doubles it (up to the
/// maximum) if `ρ > ¾` and the step reached the boundary, and accepts the step if `ρ > η`.
pub fn minimize_trust_region(
    mut f: impl FnMut(&[f64]) -> f64,
    mut grad: impl FnMut(&[f64]) -> Vec<f64>,
    mut hess: impl FnMut(&[f64]) -> Array2<f64>,
    x0: &[f64],
    method: TrustRegionMethod,
    options: TrustRegionOptions,
) -> Result<TrustRegionResult, ErrorsJSL> {
    let o = options;
    if !(0.0..0.25).contains(&o.eta) || o.max_trust_radius <= 0.0 || o.initial_trust_radius <= 0.0 || o.initial_trust_radius >= o.max_trust_radius {
        return Err(ErrorsJSL::InvalidInputRange("need 0 ≤ η < ¼ and 0 < initial radius < maximum radius"));
    }
    let max_it = o.max_iterations.unwrap_or(200 * x0.len());
    let (mut nfev, mut njev, mut nhev) = (0, 0, 0);
    let mut model_at = |x: &[f64]| {
        let fun = f(x);
        let jac = grad(x);
        let h = hess(x);
        let jac_mag = norm(&jac);
        Model { fun, jac, hess: h, jac_mag, lambda_lb: 0.0, previous_radius: -1.0 }
    };
    let mut x = x0.to_vec();
    let mut m = model_at(&x);
    nfev += 1;
    njev += 1;
    nhev += 1;
    let mut radius = o.initial_trust_radius;
    let mut k = 0;
    let mut status = 0;
    while m.jac_mag >= o.gtol {
        let (p, hits) = match method {
            TrustRegionMethod::Ncg => m.solve_ncg(radius),
            TrustRegionMethod::Exact => m.solve_exact(radius),
            TrustRegionMethod::Dogleg => match m.solve_dogleg(radius) {
                Some(s) => s,
                None => {
                    status = 3;
                    break;
                }
            },
        };
        let predicted = m.value(&p);
        let xp: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
        let mp = model_at(&xp);
        nfev += 1;
        njev += 1;
        nhev += 1;
        let actual_reduction = m.fun - mp.fun;
        let predicted_reduction = m.fun - predicted;
        if predicted_reduction <= 0.0 {
            status = 2;
            break;
        }
        let rho = actual_reduction / predicted_reduction;
        if rho < 0.25 {
            radius *= 0.25;
        } else if rho > 0.75 && hits {
            radius = (2.0 * radius).min(o.max_trust_radius);
        }
        if rho > o.eta {
            x = xp;
            m = mp;
        }
        k += 1;
        if m.jac_mag < o.gtol {
            status = 0;
            break;
        }
        if k >= max_it {
            status = 1;
            break;
        }
    }
    Ok(TrustRegionResult { x, fun: m.fun, jac: m.jac, iterations: k, nfev, njev, nhev, status })
}
// ANCHOR_END: trust_region

#[cfg(test)]
mod tests {
    use super::*;

    fn rosen(x: &[f64]) -> f64 {
        (0..x.len() - 1).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum()
    }
    fn rosen_grad(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
            g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
        }
        g
    }
    fn rosen_hess(x: &[f64]) -> Array2<f64> {
        let n = x.len();
        let mut h = Array2::zeros((n, n));
        for i in 0..n - 1 {
            h[(i, i)] += 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
            h[(i + 1, i + 1)] += 200.0;
            h[(i, i + 1)] -= 400.0 * x[i];
            h[(i + 1, i)] -= 400.0 * x[i];
        }
        h
    }

    #[test]
    fn rosenbrock_in_four_dimensions() {
        // From this start trust-ncg reaches the global minimum (1, 1, 1, 1), trust-exact the
        // second local minimum near (−0.776, 0.613, 0.382, 0.146), and dogleg stops at once on an
        // indefinite Hessian, all exactly as SciPy does.
        let x0 = [-1.2, 1.0, -0.5, 0.8];
        let o = TrustRegionOptions { gtol: 1e-8, ..Default::default() };
        let r = minimize_trust_region(rosen, rosen_grad, rosen_hess, &x0, TrustRegionMethod::Ncg, o).unwrap();
        assert_eq!(r.status, 0);
        assert!(r.x.iter().all(|v| (v - 1.0).abs() < 1e-8), "{:?}", r.x);
        let r = minimize_trust_region(rosen, rosen_grad, rosen_hess, &x0, TrustRegionMethod::Exact, o).unwrap();
        assert_eq!(r.status, 0);
        assert!((r.x[0] + 0.7756592265653737).abs() < 1e-8 && norm(&rosen_grad(&r.x)) < 1e-8, "{:?}", r.x);
        let r = minimize_trust_region(rosen, rosen_grad, rosen_hess, &x0, TrustRegionMethod::Dogleg, o).unwrap();
        assert_eq!((r.status, r.iterations), (3, 0));
        let r = minimize_trust_region(rosen, rosen_grad, rosen_hess, &[1.2, 1.3, 1.1, 1.0], TrustRegionMethod::Dogleg, o).unwrap();
        assert!(r.status == 0 && r.x.iter().all(|v| (v - 1.0).abs() < 1e-8), "{:?}", r.x);
    }

    #[test]
    fn exact_handles_an_indefinite_start() {
        // f = x⁴ − 2x² + y²: a saddle at the origin, minima at x = ±1.
        let f = |v: &[f64]| v[0].powi(4) - 2.0 * v[0] * v[0] + v[1] * v[1];
        let g = |v: &[f64]| vec![4.0 * v[0].powi(3) - 4.0 * v[0], 2.0 * v[1]];
        let h = |v: &[f64]| Array2::from_shape_vec((2, 2), vec![12.0 * v[0] * v[0] - 4.0, 0.0, 0.0, 2.0]).unwrap();
        for method in [TrustRegionMethod::Exact, TrustRegionMethod::Ncg] {
            let r = minimize_trust_region(f, g, h, &[0.01, 0.5], method, TrustRegionOptions { gtol: 1e-10, ..Default::default() }).unwrap();
            assert!((r.x[0].abs() - 1.0).abs() < 1e-8 && r.x[1].abs() < 1e-8, "{method:?} {:?}", r.x);
        }
    }
}
