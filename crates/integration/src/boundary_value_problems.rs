//! Two-point boundary value problems `y' = f(x, y)` on `[a, b]` with `g(y(a), y(b)) = 0`.
//!
//! * [`solve_bvp_shooting`]: single shooting. The unknown initial state is found by Newton's
//!   method on the boundary residual, each evaluation integrating the initial value problem with
//!   an adaptive Dormand-Prince 5(4) method. Simple and accurate for well-conditioned problems;
//!   it fails when the initial value problem is unstable over `[a, b]` (e.g. boundary layers).
//! * [`solve_bvp_collocation`]: global collocation with the fourth-order Hermite-Simpson
//!   (Lobatto IIIA) scheme, solved by damped Newton on the whole mesh, with mesh refinement where
//!   the defect of the cubic interpolant is large (the method of SciPy's `solve_bvp`). Robust for
//!   stiff and boundary-layer problems.
//!
//! See the book entry [Boundary value problems](https://jwkunz.github.io/lib_jsl/book/integration/boundary-value-problems.html).

use crate::stiff_ode::Lu;
use lib_jsl_core::ErrorsJSL;

/// Options for the boundary value solvers.
#[derive(Debug, Clone, Copy)]
pub struct BvpOptions {
    /// Collocation: bound on the relative defect `|S' − f| / (1 + |f|)` of the interpolant.
    /// Shooting: bound on the boundary residual.
    pub tolerance: f64,
    /// Largest mesh allowed by refinement (collocation).
    pub max_nodes: usize,
    /// The largest number of Newton iterations (per mesh, for collocation).
    pub max_newton_iterations: usize,
}

impl Default for BvpOptions {
    fn default() -> Self {
        BvpOptions { tolerance: 1e-6, max_nodes: 2000, max_newton_iterations: 50 }
    }
}

/// A solution on a mesh, with derivatives for cubic Hermite interpolation.
#[derive(Debug, Clone)]
pub struct BvpSolution {
    /// The mesh (collocation) or the accepted integration steps (shooting).
    pub x: Vec<f64>,
    /// The state at each point of `x`.
    pub y: Vec<Vec<f64>>,
    /// `f(x_i, y_i)` at each mesh point.
    pub derivative: Vec<Vec<f64>>,
    /// Whether the tolerance was met.
    pub converged: bool,
    /// The largest relative defect (collocation) or boundary residual (shooting).
    pub max_residual: f64,
}

/// The cubic Hermite interpolant on one interval at fraction `t`, and its derivative.
fn hermite(t: f64, h: f64, y0: &[f64], f0: &[f64], y1: &[f64], f1: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (t2, t3) = (t * t, t * t * t);
    let (h00, h10, h01, h11) = (2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, -2.0 * t3 + 3.0 * t2, t3 - t2);
    let (d00, d10, d01, d11) = ((6.0 * t2 - 6.0 * t) / h, 3.0 * t2 - 4.0 * t + 1.0, (-6.0 * t2 + 6.0 * t) / h, 3.0 * t2 - 2.0 * t);
    let value = (0..y0.len()).map(|k| h00 * y0[k] + h * h10 * f0[k] + h01 * y1[k] + h * h11 * f1[k]).collect();
    let slope = (0..y0.len()).map(|k| d00 * y0[k] + d10 * f0[k] + d01 * y1[k] + d11 * f1[k]).collect();
    (value, slope)
}

impl BvpSolution {
    /// The solution at `x` by cubic Hermite interpolation (clamped to the interval).
    pub fn evaluate(&self, x: f64) -> Vec<f64> {
        let ascending = self.x[self.x.len() - 1] >= self.x[0];
        let i = self
            .x
            .windows(2)
            .position(|w| if ascending { x <= w[1] } else { x >= w[1] })
            .unwrap_or(self.x.len() - 2);
        let h = self.x[i + 1] - self.x[i];
        let t = ((x - self.x[i]) / h).clamp(0.0, 1.0);
        hermite(t, h, &self.y[i], &self.derivative[i], &self.y[i + 1], &self.derivative[i + 1]).0
    }
}

fn eval<F: Fn(f64, &[f64]) -> Vec<f64>>(f: &F, x: f64, y: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    let v = f(x, y);
    if v.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((v.len(), y.len())));
    }
    Ok(v)
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, a| m.max(a.abs()))
}

/// Mesh points and states of an initial value solve.
type Trajectory = (Vec<f64>, Vec<Vec<f64>>);

/// Integrates `y' = f(x, y)` from `a` to `b` with adaptive Dormand-Prince 5(4), returning every
/// accepted step.
fn dormand_prince_system<F: Fn(f64, &[f64]) -> Vec<f64>>(f: &F, a: f64, y0: &[f64], b: f64, rtol: f64, atol: f64) -> Result<Trajectory, ErrorsJSL> {
    let options = crate::stiff_ode::OdeOptions { relative_tolerance: rtol, absolute_tolerance: atol, initial_step: Some(1e-3 * (b - a).abs()), max_steps: 1_000_000 };
    let s = crate::dormand_prince::dormand_prince_solve(f, a, y0, b, options)?;
    Ok((s.t, s.y))
}

/// Solves the boundary value problem by single shooting from the guess `initial_state` for
/// `y(a)`. `boundary(y(a), y(b))` must return as many residuals as there are state components.
pub fn solve_bvp_shooting<F, G>(f: F, boundary: G, a: f64, b: f64, initial_state: Vec<f64>, options: BvpOptions) -> Result<BvpSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    G: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    let n = initial_state.len();
    if n == 0 || !a.is_finite() || !b.is_finite() || a == b {
        return Err(ErrorsJSL::InvalidInputRange("Shooting needs a non-empty state and a finite, non-empty interval."));
    }
    let (rtol, atol) = (1e-12, 1e-14);
    let residual = |s: &[f64]| -> Result<(Vec<f64>, Trajectory), ErrorsJSL> {
        let trajectory = dormand_prince_system(&f, a, s, b, rtol, atol)?;
        let g = boundary(s, trajectory.1.last().expect("non-empty"));
        if g.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((g.len(), n)));
        }
        Ok((g, trajectory))
    };
    let mut s = initial_state;
    let (mut g, mut trajectory) = residual(&s)?;
    let mut converged = inf_norm(&g) <= options.tolerance;
    for _ in 0..options.max_newton_iterations {
        if converged {
            break;
        }
        // ANCHOR: shooting_newton
        // Finite-difference Jacobian of the boundary residual with respect to y(a).
        let mut jac = vec![vec![0.0; n]; n];
        for k in 0..n {
            let step = 1e-7 * s[k].abs().max(1.0);
            let mut sk = s.clone();
            sk[k] += step;
            let (gk, _) = residual(&sk)?;
            for i in 0..n {
                jac[i][k] = (gk[i] - g[i]) / step;
            }
        }
        let delta = Lu::new(jac)?.solve(&g);
        // Damped update: halve until the residual decreases.
        let norm = inf_norm(&g);
        let mut lambda = 1.0;
        loop {
            let trial: Vec<f64> = s.iter().zip(&delta).map(|(v, d)| v - lambda * d).collect();
            if let Ok((gt, tt)) = residual(&trial)
                && (inf_norm(&gt) < norm || lambda < 1e-3)
            {
                s = trial;
                g = gt;
                trajectory = tt;
                break;
            }
            lambda *= 0.5;
            if lambda < 1e-4 {
                return Err(ErrorsJSL::RuntimeError("Shooting failed: the boundary residual could not be reduced."));
            }
        }
        converged = inf_norm(&g) <= options.tolerance;
        // ANCHOR_END: shooting_newton
    }
    let (x, y) = trajectory;
    let derivative = x.iter().zip(&y).map(|(&xi, yi)| f(xi, yi)).collect();
    Ok(BvpSolution { x, y, derivative, converged, max_residual: inf_norm(&g) })
}

// ANCHOR: hermite_simpson
/// Hermite-Simpson collocation residual `y₁ − y₀ − h/6 (f₀ + 4 f_m + f₁)` of one interval.
fn interval_residual<F: Fn(f64, &[f64]) -> Vec<f64>>(f: &F, x0: f64, h: f64, y0: &[f64], f0: &[f64], y1: &[f64], f1: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    let n = y0.len();
    let ym: Vec<f64> = (0..n).map(|k| 0.5 * (y0[k] + y1[k]) + h / 8.0 * (f0[k] - f1[k])).collect();
    let fm = eval(f, x0 + 0.5 * h, &ym)?;
    Ok((0..n).map(|k| y1[k] - y0[k] - h / 6.0 * (f0[k] + 4.0 * fm[k] + f1[k])).collect())
}
// ANCHOR_END: hermite_simpson

/// Solves the collocation equations on a fixed mesh by damped Newton, from the guess `y`.
fn newton_on_mesh<F, G>(f: &F, boundary: &G, x: &[f64], y: &mut [Vec<f64>], options: &BvpOptions) -> Result<Vec<Vec<f64>>, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    G: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    let (m, n) = (x.len() - 1, y[0].len());
    let size = (m + 1) * n;
    let residuals = |y: &[Vec<f64>], fy: &[Vec<f64>]| -> Result<Vec<f64>, ErrorsJSL> {
        let mut r = Vec::with_capacity(size);
        for i in 0..m {
            r.extend(interval_residual(f, x[i], x[i + 1] - x[i], &y[i], &fy[i], &y[i + 1], &fy[i + 1])?);
        }
        let bc = boundary(&y[0], &y[m]);
        if bc.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((bc.len(), n)));
        }
        r.extend(bc);
        Ok(r)
    };
    let derivatives = |y: &[Vec<f64>]| -> Result<Vec<Vec<f64>>, ErrorsJSL> { x.iter().zip(y).map(|(&xi, yi)| eval(f, xi, yi)).collect() };
    let mut fy = derivatives(y)?;
    let mut r = residuals(y, &fy)?;
    for _ in 0..options.max_newton_iterations {
        // Jacobian: perturbing node j touches intervals j−1 and j and, at the ends, the boundary rows.
        let mut jac = vec![vec![0.0; size]; size];
        for j in 0..=m {
            for k in 0..n {
                let step = 1e-7 * y[j][k].abs().max(1.0);
                let mut yj = y[j].clone();
                yj[k] += step;
                let fj = eval(f, x[j], &yj)?;
                let column = j * n + k;
                let mut write = |row0: usize, values: Vec<f64>, base: &[f64]| {
                    for (i, v) in values.into_iter().enumerate() {
                        jac[row0 + i][column] = (v - base[row0 + i]) / step;
                    }
                };
                if j > 0 {
                    let v = interval_residual(f, x[j - 1], x[j] - x[j - 1], &y[j - 1], &fy[j - 1], &yj, &fj)?;
                    write((j - 1) * n, v, &r);
                }
                if j < m {
                    let v = interval_residual(f, x[j], x[j + 1] - x[j], &yj, &fj, &y[j + 1], &fy[j + 1])?;
                    write(j * n, v, &r);
                }
                if j == 0 {
                    write(m * n, boundary(&yj, &y[m]), &r);
                }
                if j == m {
                    write(m * n, boundary(&y[0], &yj), &r);
                }
            }
        }
        let delta = Lu::new(jac)?.solve(&r);
        let norm = inf_norm(&r);
        let mut lambda = 1.0;
        let mut accepted = false;
        while lambda > 1e-4 {
            let trial: Vec<Vec<f64>> = (0..=m).map(|j| (0..n).map(|k| y[j][k] - lambda * delta[j * n + k]).collect()).collect();
            if let Ok(ft) = derivatives(&trial)
                && let Ok(rt) = residuals(&trial, &ft)
                && inf_norm(&rt) < norm.max(1e-300)
            {
                y.clone_from_slice(&trial);
                fy = ft;
                r = rt;
                accepted = true;
                break;
            }
            lambda *= 0.5;
        }
        let step_size = (0..size).map(|i| (lambda * delta[i]).abs() / (1.0 + y[i / n][i % n].abs())).fold(0.0, f64::max);
        if !accepted || step_size < 1e-11 || inf_norm(&r) < 1e-13 {
            break;
        }
    }
    Ok(fy)
}

/// Solves the boundary value problem by Hermite-Simpson collocation, starting from the mesh
/// `x` (strictly monotonic, at least two points) and the guess `y` at each mesh point. Intervals
/// whose relative defect exceeds the tolerance are split until it is met everywhere or the mesh
/// reaches `max_nodes` (then `converged` is false).
pub fn solve_bvp_collocation<F, G>(f: F, boundary: G, x: Vec<f64>, y: Vec<Vec<f64>>, options: BvpOptions) -> Result<BvpSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    G: Fn(&[f64], &[f64]) -> Vec<f64>,
{
    if x.len() < 2 || x.len() != y.len() || y[0].is_empty() || y.iter().any(|v| v.len() != y[0].len()) {
        return Err(ErrorsJSL::InvalidInputRange("The mesh needs at least two points and a guess of consistent size at each."));
    }
    let increasing = x[1] > x[0];
    if x.windows(2).any(|w| if increasing { w[1] <= w[0] } else { w[1] >= w[0] }) {
        return Err(ErrorsJSL::InvalidInputRange("The mesh must be strictly monotonic."));
    }
    let (mut x, mut y) = (x, y);
    loop {
        let fy = newton_on_mesh(&f, &boundary, &x, &mut y, &options)?;
        // Relative defect of the cubic interpolant at the quarter points of each interval (it
        // vanishes at the nodes and midpoints by construction).
        let mut defects = Vec::with_capacity(x.len() - 1);
        for i in 0..x.len() - 1 {
            let h = x[i + 1] - x[i];
            let mut worst: f64 = 0.0;
            for t in [0.25, 0.75] {
                let (s, ds) = hermite(t, h, &y[i], &fy[i], &y[i + 1], &fy[i + 1]);
                let fs = eval(&f, x[i] + t * h, &s)?;
                worst = worst.max((0..s.len()).map(|k| (ds[k] - fs[k]).abs() / (1.0 + fs[k].abs())).fold(0.0, f64::max));
            }
            defects.push(if worst.is_finite() { worst } else { f64::INFINITY });
        }
        let max_residual = defects.iter().copied().fold(0.0, f64::max);
        let to_split = defects.iter().filter(|&&d| d > options.tolerance).count();
        if to_split == 0 || x.len() + to_split > options.max_nodes {
            let converged = to_split == 0 && boundary(&y[0], &y[y.len() - 1]).iter().all(|v| v.abs() < 1e-6_f64.max(options.tolerance));
            return Ok(BvpSolution { x, y, derivative: fy, converged, max_residual });
        }
        // Split each failing interval in two (three where the defect is very large), seeding the
        // new nodes from the interpolant.
        let mut new_x = vec![x[0]];
        let mut new_y = vec![y[0].clone()];
        for i in 0..x.len() - 1 {
            let h = x[i + 1] - x[i];
            let pieces = if defects[i] > 100.0 * options.tolerance { 3 } else if defects[i] > options.tolerance { 2 } else { 1 };
            for p in 1..pieces {
                let t = p as f64 / pieces as f64;
                new_x.push(x[i] + t * h);
                new_y.push(hermite(t, h, &y[i], &fy[i], &y[i + 1], &fy[i + 1]).0);
            }
            new_x.push(x[i + 1]);
            new_y.push(y[i + 1].clone());
        }
        x = new_x;
        y = new_y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    /// The lower Bratu solution u'' + e^u = 0, u(0) = u(1) = 0:
    /// u = −2 ln(cosh((x − ½)θ/2) / cosh(θ/4)) with θ = √2 cosh(θ/4).
    fn bratu_exact(x: f64) -> f64 {
        let mut theta: f64 = 1.0;
        for _ in 0..100 {
            theta = 2f64.sqrt() * (theta / 4.0).cosh();
        }
        -2.0 * (((x - 0.5) * theta / 2.0).cosh() / (theta / 4.0).cosh()).ln()
    }

    fn bratu(_: f64, y: &[f64]) -> Vec<f64> {
        vec![y[1], -y[0].exp()]
    }

    fn dirichlet_zero(ya: &[f64], yb: &[f64]) -> Vec<f64> {
        vec![ya[0], yb[0]]
    }

    #[test]
    fn shooting_solves_linear_and_bratu_problems() {
        // u'' = −u, u(0) = 0, u(π/2) = 1 → sin x.
        let sol = solve_bvp_shooting(|_, y: &[f64]| vec![y[1], -y[0]], |ya: &[f64], yb: &[f64]| vec![ya[0], yb[0] - 1.0], 0.0, FRAC_PI_2, vec![0.0, 0.3], BvpOptions::default()).unwrap();
        assert!(sol.converged && (sol.y[0][1] - 1.0).abs() < 1e-8, "{:?}", sol.y[0]);
        assert!((sol.evaluate(0.7)[0] - 0.7f64.sin()).abs() < 1e-8);
        let sol = solve_bvp_shooting(bratu, dirichlet_zero, 0.0, 1.0, vec![0.0, 0.1], BvpOptions::default()).unwrap();
        for &x in &[0.1, 0.5, 0.83] {
            assert!((sol.evaluate(x)[0] - bratu_exact(x)).abs() < 1e-8, "x={x}");
        }
    }

    #[test]
    fn collocation_is_fourth_order_and_refines_to_tolerance() {
        let guess = |m: usize| -> (Vec<f64>, Vec<Vec<f64>>) { ((0..=m).map(|i| i as f64 / m as f64).collect(), vec![vec![0.0, 0.0]; m + 1]) };
        // Fixed meshes (refinement disabled): halving h cuts the error about 16×.
        let fixed = BvpOptions { tolerance: f64::INFINITY, ..Default::default() };
        let error = |m: usize| {
            let (x, y) = guess(m);
            let sol = solve_bvp_collocation(bratu, dirichlet_zero, x, y, fixed).unwrap();
            sol.x.iter().zip(&sol.y).map(|(&x, y)| (y[0] - bratu_exact(x)).abs()).fold(0.0, f64::max)
        };
        let (e1, e2) = (error(5), error(10));
        assert!(e1 / e2 > 12.0 && e1 / e2 < 20.0, "{e1} {e2}");
        // With refinement the defect tolerance is met and the solution is accurate.
        let (x, y) = guess(4);
        let sol = solve_bvp_collocation(bratu, dirichlet_zero, x, y, BvpOptions { tolerance: 1e-8, ..Default::default() }).unwrap();
        assert!(sol.converged && sol.max_residual <= 1e-8);
        assert!((sol.evaluate(0.37)[0] - bratu_exact(0.37)).abs() < 1e-9);
    }

    #[test]
    fn collocation_resolves_a_boundary_layer() {
        // ε y'' + y' = 0, y(0) = 0, y(1) = 1: y = (1 − e^{−x/ε}) / (1 − e^{−1/ε}), a layer at x = 0.
        let eps = 1e-3;
        let f = move |_: f64, y: &[f64]| vec![y[1], -y[1] / eps];
        let bc = |ya: &[f64], yb: &[f64]| vec![ya[0], yb[0] - 1.0];
        let x: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|&t| vec![t, 1.0]).collect();
        let sol = solve_bvp_collocation(f, bc, x, y, BvpOptions { tolerance: 1e-5, ..Default::default() }).unwrap();
        assert!(sol.converged, "max residual {}", sol.max_residual);
        let exact = |x: f64| -(-x / eps).exp_m1() / -(-1.0 / eps).exp_m1();
        for &x in &[0.0005, 0.002, 0.01, 0.5] {
            assert!((sol.evaluate(x)[0] - exact(x)).abs() < 1e-4, "x={x}: {} vs {}", sol.evaluate(x)[0], exact(x));
        }
        // Nodes cluster inside the layer: the first 1% of the interval holds over a fifth of them.
        let inside = sol.x.iter().filter(|&&x| x < 0.01).count();
        assert!(inside * 5 > sol.x.len(), "{inside} of {}", sol.x.len());
        assert!(solve_bvp_collocation(f, bc, vec![0.0], vec![vec![0.0, 0.0]], BvpOptions::default()).is_err());
    }
}
