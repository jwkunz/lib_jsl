//! COBYLA: constrained optimisation by linear approximations (Powell, 1994; SciPy's
//! `minimize(method='COBYLA')`), a derivative-free method for problems with nonlinear
//! inequality constraints.
//!
//! See the book entry [Constrained optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/constrained.html#cobyla-and-trust-constr).

use crate::constrained::{ConstrainedProblem, ConstrainedResult, bound_constrained_qp, solve_dense};
use lib_jsl_core::ErrorsJSL;

/// Options for [`cobyla`], with SciPy's defaults.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CobylaOptions {
    /// The initial trust-region radius (the size of the first simplex).
    pub rhobeg: f64,
    /// The final radius, which sets the accuracy (SciPy's `tol`).
    pub rhoend: f64,
    /// The largest number of function evaluations.
    pub max_evaluations: usize,
    /// The constraint violation accepted at the end (SciPy's `catol`).
    pub catol: f64,
}

impl Default for CobylaOptions {
    fn default() -> Self {
        CobylaOptions { rhobeg: 1.0, rhoend: 1e-4, max_evaluations: 1000, catol: 2e-4 }
    }
}

impl CobylaOptions {
    /// The defaults with the given initial and final radii.
    pub fn with_radii(rhobeg: f64, rhoend: f64) -> Self {
        CobylaOptions { rhobeg, rhoend, ..Default::default() }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `min qᵀd + (μ/2)(‖d‖² + w t²) + t·[stage 1]` subject to `A d − [stage 1] t ≤ b`: the minimiser
/// for one μ through the dual (a bound-constrained QP in the multipliers). In stage 1 the extra
/// variable `t ≥ 0` is the largest violation.
fn regularised_lp(q: &[f64], a: &[Vec<f64>], b: &[f64], mu: f64, stage1: bool) -> Option<(Vec<f64>, f64)> {
    let n = q.len();
    let m = a.len();
    // Variables z = (d, t) in stage 1, d otherwise; constraints rows r with r·z ≤ b (plus −t ≤ 0).
    let nz = if stage1 { n + 1 } else { n };
    let mut rows: Vec<Vec<f64>> = a.iter().map(|r| if stage1 { let mut r = r.clone(); r.push(-1.0); r } else { r.clone() }).collect();
    let mut rhs = b.to_vec();
    if stage1 {
        let mut r = vec![0.0; nz];
        r[n] = -1.0;
        rows.push(r);
        rhs.push(0.0);
    }
    let mut c = q.to_vec();
    if stage1 {
        c.push(1.0);
    }
    // Primal: min cᵀz + (μ/2)‖z‖² s.t. R z ≤ rhs. Dual: min ½ λᵀ(R Rᵀ/μ)λ + (R c/μ + rhs)ᵀλ, λ ≥ 0.
    let k = rows.len();
    let mm: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| dot(&rows[i], &rows[j]) / mu).collect()).collect();
    let qq: Vec<f64> = (0..k).map(|i| dot(&rows[i], &c) / mu + rhs[i]).collect();
    let lambda = bound_constrained_qp(&mm, &qq, 0)?;
    let mut z: Vec<f64> = c.iter().map(|v| -v / mu).collect();
    for (r, l) in rows.iter().zip(&lambda) {
        z.iter_mut().zip(r).for_each(|(zi, ri)| *zi -= l * ri / mu);
    }
    let _ = m;
    let t = if stage1 { z[n].max(0.0) } else { 0.0 };
    z.truncate(n);
    Some((z, t))
}

/// The linear trust-region subproblem: minimise `gᵀd` subject to `c + A d ≥ 0` and `‖d‖ ≤ Δ`,
/// or, if the linearised constraints cannot all hold in the ball, first minimise their largest
/// violation and then `gᵀd` without letting it grow (Powell's two stages). Each stage is the
/// limit of `min gᵀd + (μ/2)‖d‖²` as `μ` falls to the value where `‖d‖ = Δ`, found by bisection
/// on `log μ`.
fn subproblem(g: &[f64], c: &[f64], a: &[Vec<f64>], delta: f64) -> Vec<f64> {
    // Constraints −A d ≤ c.
    let neg: Vec<Vec<f64>> = a.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
    let solve = |q: &[f64], b: &[f64], stage1: bool| -> Vec<f64> {
        let scale = (norm(q) + 1.0) / delta;
        let (mut lo, mut hi) = ((scale * 1e-10).ln(), (scale * 1e10).ln());
        let at = |l: f64| regularised_lp(q, &neg, b, l.exp(), stage1).map(|r| r.0);
        match at(lo) {
            Some(d) if norm(&d) <= delta => return d,
            _ => {}
        }
        let mut best = at(hi).unwrap_or_else(|| vec![0.0; q.len()]);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            match at(mid) {
                Some(d) if norm(&d) <= delta => {
                    best = d;
                    hi = mid;
                }
                _ => lo = mid,
            }
            if hi - lo < 1e-9 {
                break;
            }
        }
        best
    };
    let n = g.len();
    if c.iter().all(|&v| v >= 0.0) {
        return solve(g, c, false);
    }
    // Stage 1: the smallest largest violation within the ball.
    let d1 = solve(&vec![0.0; n], c, true);
    let viol = c.iter().zip(a).map(|(ci, r)| -(ci + dot(r, &d1))).fold(0.0f64, f64::max);
    if viol <= 0.0 {
        return solve(g, c, false);
    }
    let relaxed: Vec<f64> = c.iter().map(|v| v + viol * (1.0 + 1e-9)).collect();
    let d2 = solve(g, &relaxed, false);
    let v2 = c.iter().zip(a).map(|(ci, r)| -(ci + dot(r, &d2))).fold(0.0f64, f64::max);
    if v2 <= viol * (1.0 + 1e-6) + 1e-14 { d2 } else { d1 }
}

/// Inverts the `n × n` matrix whose rows are `rows`.
fn invert(rows: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = rows.len();
    let cols: Vec<Vec<f64>> = (0..n).map(|j| solve_dense(rows.to_vec(), (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect())).collect::<Option<_>>()?;
    Some((0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect())
}

// ANCHOR: cobyla
/// Minimises `f` subject to `g(x) ≤ 0` and `h(x) = 0` without derivatives, by Powell's COBYLA
/// (SciPy's `minimize(method='COBYLA')`; equalities become pairs of inequalities).
///
/// The method keeps a simplex of `n + 1` points and the linear functions that interpolate the
/// objective and every constraint on it. Each iteration minimises the linear objective subject
/// to the linear constraints within a ball of radius `ρ` around the best vertex (best by the
/// merit `f + μ·max violation`, with `μ` raised whenever a step would not decrease it), and
/// replaces a vertex by the new point. Steps that would spoil the simplex's shape are replaced
/// by geometry steps, and `ρ` is halved when the trust-region steps become short, until it
/// reaches `rhoend`. The derivatives defaulted by [`ConstrainedProblem`] are never called.
pub fn cobyla<P: ConstrainedProblem>(problem: &P, x0: Vec<f64>, options: CobylaOptions) -> Result<ConstrainedResult, ErrorsJSL> {
    let n = x0.len();
    if n == 0 {
        return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
    }
    if !(options.rhobeg > 0.0 && options.rhoend > 0.0 && options.rhoend <= options.rhobeg) {
        return Err(ErrorsJSL::InvalidInputRange("need 0 < rhoend ≤ rhobeg"));
    }
    // Constraints as c(x) ≥ 0.
    let cons = |x: &[f64]| -> Vec<f64> {
        let mut c: Vec<f64> = problem.inequalities(x).iter().map(|v| -v).collect();
        for h in problem.equalities(x) {
            c.push(h);
            c.push(-h);
        }
        c
    };
    let nfev = std::cell::Cell::new(0usize);
    let eval = |x: &[f64]| -> (f64, Vec<f64>) {
        nfev.set(nfev.get() + 1);
        (problem.objective(x), cons(x))
    };
    let resmax = |c: &[f64]| c.iter().map(|v| -v).fold(0.0f64, f64::max);
    let mut rho = options.rhobeg;
    let mut mu = 0.0f64;
    // Vertex 0 is x0; vertex j is x0 + ρ e_j.
    let mut pts: Vec<Vec<f64>> = vec![x0.clone()];
    let mut vals: Vec<(f64, Vec<f64>)> = vec![eval(&x0)];
    for j in 0..n {
        let mut p = x0.clone();
        p[j] += rho;
        vals.push(eval(&p));
        pts.push(p);
    }
    let phi = |v: &(f64, Vec<f64>), mu: f64| v.0 + mu * resmax(&v.1);
    loop {
        // Make vertex 0 the best by the merit (ties: the smaller violation).
        let best = (0..=n)
            .min_by(|&a, &b| {
                let (pa, pb) = (phi(&vals[a], mu), phi(&vals[b], mu));
                pa.total_cmp(&pb).then(resmax(&vals[a].1).total_cmp(&resmax(&vals[b].1)))
            })
            .unwrap();
        pts.swap(0, best);
        vals.swap(0, best);
        if nfev.get() >= options.max_evaluations {
            break;
        }
        // Linear models: rows of D are the displacements x_j − x_0; gradients solve D ∇ = Δ.
        let d: Vec<Vec<f64>> = (1..=n).map(|j| pts[j].iter().zip(&pts[0]).map(|(a, b)| a - b).collect()).collect();
        let Some(dinv) = invert(&d) else {
            // A degenerate simplex: rebuild it around the best point.
            for j in 0..n {
                let mut p = pts[0].clone();
                p[j] += rho;
                vals[j + 1] = eval(&p);
                pts[j + 1] = p;
            }
            continue;
        };
        let grad_of = |k: Option<usize>| -> Vec<f64> {
            let diffs: Vec<f64> = (1..=n).map(|j| match k {
                None => vals[j].0 - vals[0].0,
                Some(i) => vals[j].1[i] - vals[0].1[i],
            }).collect();
            (0..n).map(|r| (0..n).map(|j| dinv[r][j] * diffs[j]).sum()).collect()
        };
        let g = grad_of(None);
        let mc = vals[0].1.len();
        let a: Vec<Vec<f64>> = (0..mc).map(|i| grad_of(Some(i))).collect();
        // Simplex geometry: Powell's acceptability test with factors 1.1 (lengths) and 0.25
        // (distances of each vertex from the opposite face).
        let lengths: Vec<f64> = d.iter().map(|r| norm(r)).collect();
        let face: Vec<f64> = (0..n).map(|j| 1.0 / norm(&(0..n).map(|r| dinv[r][j]).collect::<Vec<_>>())).collect();
        let worst_len = (0..n).max_by(|&a, &b| lengths[a].total_cmp(&lengths[b])).unwrap();
        let worst_face = (0..n).min_by(|&a, &b| face[a].total_cmp(&face[b])).unwrap();
        let acceptable = lengths[worst_len] <= 1.1 * rho && face[worst_face] >= 0.25 * rho;
        // The trust-region step.
        let step = subproblem(&g, &vals[0].1, &a, 0.5 * rho);
        let short = norm(&step) < 0.25 * rho;
        if short || !acceptable {
            if !acceptable {
                // Geometry step: move the worst vertex to x0 ± ½ρ along the normal of its
                // opposite face, on the side where the linear merit is lower.
                let j = if lengths[worst_len] > 1.1 * rho { worst_len } else { worst_face };
                let v: Vec<f64> = (0..n).map(|r| dinv[r][j]).collect();
                let vn = norm(&v);
                let mut dir: Vec<f64> = v.iter().map(|x| 0.5 * rho * x / vn).collect();
                let lin_merit = |s: &[f64]| dot(&g, s) + mu * vals[0].1.iter().zip(&a).map(|(c, r)| -(c + dot(r, s))).fold(0.0f64, f64::max);
                let neg: Vec<f64> = dir.iter().map(|x| -x).collect();
                if lin_merit(&neg) < lin_merit(&dir) {
                    dir = neg;
                }
                let p: Vec<f64> = pts[0].iter().zip(&dir).map(|(a, b)| a + b).collect();
                vals[j + 1] = eval(&p);
                pts[j + 1] = p;
                continue;
            }
            // A good simplex and a short step: the current radius is exhausted.
            if rho <= options.rhoend {
                break;
            }
            rho = (0.5 * rho).max(options.rhoend);
            if rho <= 1.5 * options.rhoend {
                rho = options.rhoend;
            }
            continue;
        }
        // Raise μ until the step decreases the linearised merit.
        let r0 = resmax(&vals[0].1);
        let r_lin = vals[0].1.iter().zip(&a).map(|(c, r)| -(c + dot(r, &step))).fold(0.0f64, f64::max);
        let fdec = -dot(&g, &step);
        if r_lin < r0 {
            let needed = -fdec / (r0 - r_lin);
            if mu < 1.5 * needed {
                mu = 2.0 * needed;
            }
        }
        let p: Vec<f64> = pts[0].iter().zip(&step).map(|(a, b)| a + b).collect();
        let v = eval(&p);
        // Replace the vertex that the new point's barycentric coordinates weight most, favouring
        // far vertices (Powell's choice), so that the simplex stays well shaped.
        let sigma: Vec<f64> = (0..n).map(|j| (0..n).map(|r| dinv[r][j] * step[r]).sum::<f64>()).collect();
        let improved = phi(&v, mu) < phi(&vals[0], mu);
        let j = (0..n)
            .max_by(|&a, &b| {
                let w = |k: usize| sigma[k].abs() * (norm(&d[k].iter().zip(&step).map(|(x, y)| x - y).collect::<Vec<_>>()) / rho).max(1.0).powi(if improved { 2 } else { 1 });
                w(a).total_cmp(&w(b))
            })
            .unwrap();
        let predicted = fdec + mu * (r0 - r_lin);
        let actual = phi(&vals[0], mu) - phi(&v, mu);
        if improved || sigma[j].abs() > 0.0 {
            vals[j + 1] = v;
            pts[j + 1] = p;
        }
        // A poor step from a well-shaped simplex: the radius is too large for the model.
        if actual <= 0.1 * predicted && acceptable {
            if rho <= options.rhoend {
                break;
            }
            rho = (0.5 * rho).max(options.rhoend);
            if rho <= 1.5 * options.rhoend {
                rho = options.rhoend;
            }
        }
    }
    let best = (0..=n)
        .min_by(|&a, &b| phi(&vals[a], mu).total_cmp(&phi(&vals[b], mu)).then(resmax(&vals[a].1).total_cmp(&resmax(&vals[b].1))))
        .unwrap();
    let x = pts[best].clone();
    let (fx, c) = vals[best].clone();
    let violation = resmax(&c);
    Ok(ConstrainedResult {
        parameters: x,
        objective_value: fx,
        equality_multipliers: Vec::new(),
        inequality_multipliers: Vec::new(),
        constraint_violation: violation,
        iterations: nfev.get(),
        converged: rho <= options.rhoend && violation <= options.catol && nfev.get() < options.max_evaluations,
    })
}
// ANCHOR_END: cobyla

#[cfg(test)]
mod tests {
    use super::*;

    struct Tutorial;
    impl ConstrainedProblem for Tutorial {
        // SciPy's COBYLA example: min x0 x1 s.t. 1 − x0² − x1² ≥ 0, x1 ≥ 0.
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] * x[1]
        }
        fn inequalities(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1] - 1.0, -x[1]]
        }
    }

    struct Rosen;
    impl ConstrainedProblem for Rosen {
        fn objective(&self, x: &[f64]) -> f64 {
            100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
        }
        // A disc that excludes the unconstrained minimum (1, 1).
        fn inequalities(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1] - 1.5]
        }
        fn equalities(&self, _x: &[f64]) -> Vec<f64> {
            Vec::new()
        }
    }

    #[test]
    fn solves_the_scipy_example_and_a_constrained_rosenbrock() {
        let r = cobyla(&Tutorial, vec![1.0, 1.0], CobylaOptions::with_radii(1.0, 1e-8)).unwrap();
        let s = 0.5f64.sqrt();
        assert!(r.converged && (r.parameters[0] + s).abs() < 1e-6 && (r.parameters[1] - s).abs() < 1e-6, "{:?}", r.parameters);
        let r = cobyla(&Rosen, vec![-1.0, 0.0], CobylaOptions { rhoend: 1e-8, max_evaluations: 5000, ..Default::default() }).unwrap();
        // The minimum lies on the circle x² + y² = 1.5.
        let x = &r.parameters;
        assert!(r.converged && (x[0] * x[0] + x[1] * x[1] - 1.5).abs() < 1e-6, "{x:?} {}", r.iterations);
        assert!((x[0] - 0.9072).abs() < 1e-3 && (x[1] - 0.8228).abs() < 1e-3, "{x:?}");
    }
}
