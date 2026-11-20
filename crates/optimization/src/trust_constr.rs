//! `trust-constr`: a trust-region interior-point method for nonlinearly constrained problems
//! (Byrd, Hribar and Nocedal, 1999; SciPy's `minimize(method='trust-constr')`).
//!
//! See the book entry [Constrained optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/constrained.html#cobyla-and-trust-constr).

use crate::constrained::{ConstrainedOptions, ConstrainedProblem, ConstrainedResult, solve_dense};
use lib_jsl_core::ErrorsJSL;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|r| dot(r, v)).collect()
}

fn rmatvec(m: &[Vec<f64>], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (r, vi) in m.iter().zip(v) {
        out.iter_mut().zip(r).for_each(|(o, a)| *o += vi * a);
    }
    out
}

/// Solves `(A Aᵀ + εI) y = r` (the projections of the Byrd-Omojokun step).
fn aat_solve(a: &[Vec<f64>], r: &[f64]) -> Vec<f64> {
    let m = a.len();
    let trace: f64 = a.iter().map(|row| dot(row, row)).sum::<f64>().max(1.0);
    let k: Vec<Vec<f64>> = (0..m).map(|i| (0..m).map(|j| dot(&a[i], &a[j]) + if i == j { 1e-14 * trace } else { 0.0 }).collect()).collect();
    solve_dense(k, r.to_vec()).unwrap_or_else(|| vec![0.0; m])
}

/// `P v = v − Aᵀ(AAᵀ)⁻¹A v`, the projection onto the null space of `A`.
fn project(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return v.to_vec();
    }
    let y = aat_solve(a, &matvec(a, v));
    let aty = rmatvec(a, &y, v.len());
    v.iter().zip(&aty).map(|(p, q)| p - q).collect()
}

/// The two `t` where `‖z + t d‖ = Δ`.
fn to_boundary(z: &[f64], d: &[f64], delta: f64) -> f64 {
    let (a, b, c) = (dot(d, d), 2.0 * dot(z, d), dot(z, z) - delta * delta);
    (-b + (b * b - 4.0 * a * c).max(0.0).sqrt()) / (2.0 * a)
}

/// One Byrd-Omojokun step for `min gᵀd + ½ dᵀHd` subject to `A d + c = 0`, `‖d‖ ≤ Δ` and
/// `d_i ≥ −τ` for the scaled slack components `i ≥ n_x`.
fn byrd_omojokun_step(h: &dyn Fn(&[f64]) -> Vec<f64>, g: &[f64], a: &[Vec<f64>], c: &[f64], delta: f64, n_x: usize) -> Vec<f64> {
    let nz = g.len();
    let tau = 0.995;
    // Normal step: dogleg on min ‖A v + c‖ within 0.8Δ (and half the slack fraction to boundary).
    let mut v = vec![0.0; nz];
    if !a.is_empty() && norm(c) > 0.0 {
        let gn = rmatvec(a, &aat_solve(a, c), nz).iter().map(|t| -t).collect::<Vec<_>>();
        let r = 0.8 * delta;
        v = if norm(&gn) <= r {
            gn
        } else {
            let grad = rmatvec(a, c, nz);
            let ag = matvec(a, &grad);
            let alpha = dot(&grad, &grad) / dot(&ag, &ag).max(f64::MIN_POSITIVE);
            let cauchy: Vec<f64> = grad.iter().map(|t| -alpha * t).collect();
            if norm(&cauchy) >= r {
                cauchy.iter().map(|t| t * r / norm(&cauchy)).collect()
            } else {
                let dir: Vec<f64> = gn.iter().zip(&cauchy).map(|(p, q)| p - q).collect();
                let t = to_boundary(&cauchy, &dir, r);
                cauchy.iter().zip(&dir).map(|(p, q)| p + t * q).collect()
            }
        };
        let scale = v[n_x..].iter().filter(|t| **t < -0.5 * tau).fold(1.0f64, |m, t| m.min(-0.5 * tau / t));
        v.iter_mut().for_each(|t| *t *= scale);
    }
    // Tangential step: projected (Steihaug) CG on the reduced model from v, within Δ.
    let hv = h(&v);
    let mut r: Vec<f64> = g.iter().zip(&hv).map(|(p, q)| p + q).collect();
    let mut d = v.clone();
    let mut pr = project(a, &r);
    let mut p: Vec<f64> = pr.iter().map(|t| -t).collect();
    let tol = 0.01 * norm(&pr).min(1.0) * norm(&pr);
    let mut rpr = dot(&r, &pr);
    for _ in 0..2 * nz + 5 {
        if norm(&pr) <= tol || rpr <= 0.0 {
            break;
        }
        let hp = h(&p);
        let curv = dot(&p, &hp);
        if curv <= 0.0 {
            let t = to_boundary(&d, &p, delta);
            d.iter_mut().zip(&p).for_each(|(x, y)| *x += t * y);
            break;
        }
        let alpha = rpr / curv;
        let trial: Vec<f64> = d.iter().zip(&p).map(|(x, y)| x + alpha * y).collect();
        if norm(&trial) >= delta {
            let t = to_boundary(&d, &p, delta);
            d.iter_mut().zip(&p).for_each(|(x, y)| *x += t * y);
            break;
        }
        d = trial;
        r.iter_mut().zip(&hp).for_each(|(x, y)| *x += alpha * y);
        pr = project(a, &r);
        let rpr_new = dot(&r, &pr);
        let beta = rpr_new / rpr;
        rpr = rpr_new;
        p = p.iter().zip(&pr).map(|(x, y)| -y + beta * x).collect();
    }
    // Fraction to the boundary for the slacks.
    let scale = d[n_x..].iter().filter(|t| **t < -tau).fold(1.0f64, |m, t| m.min(-tau / t));
    d.iter_mut().for_each(|t| *t *= scale);
    d
}

// ANCHOR: trust_constr
/// Minimises `f(x)` subject to `h(x) = 0` and `g(x) ≤ 0` by a trust-region interior-point
/// method (SciPy's `minimize(method='trust-constr')` with a BFGS Hessian).
///
/// Inequalities get slacks, `g(x) + s = 0`, `s > 0`, and the barrier problem
/// `min f(x) − μ Σ ln sᵢ` subject to the equalities is solved by Byrd and Omojokun's
/// trust-region SQP for a decreasing sequence of `μ` (0.1, then a factor 0.2 each time the
/// barrier problem is solved to a tolerance that falls with it). Each SQP step is the sum of a
/// normal step, which reduces the linearised constraint violation within 0.8 of the trust
/// region, and a tangential step, projected conjugate gradients on the quadratic model in the
/// null space of the constraint Jacobian. Slack steps are scaled by `S` and kept a fraction
/// 0.995 from the boundary. Steps are accepted by the ℓ₂ merit function `φ + ν‖c‖`, the
/// Lagrangian Hessian is a damped BFGS approximation, and the multipliers are least-squares
/// estimates. `max_iterations` counts SQP iterations over all barrier problems.
pub fn trust_constr<P: ConstrainedProblem>(problem: &P, x0: Vec<f64>, options: ConstrainedOptions) -> Result<ConstrainedResult, ErrorsJSL> {
    let n = x0.len();
    if n == 0 {
        return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
    }
    let mut x = x0;
    let ng = problem.inequalities(&x).len();
    let mut s: Vec<f64> = problem.inequalities(&x).iter().map(|g| (-1.5 * g).max(1.0)).collect();
    let mut mu = if ng > 0 { 0.1 } else { 0.0 };
    let mut barrier_tol = if ng > 0 { 0.1f64.max(options.tolerance) } else { options.tolerance };
    let mut bfgs: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let mut delta: f64 = 1.0;
    let mut nu: f64 = 1.0;
    let mut iterations = 0;
    // Constraint values c(z) = [h(x); g(x) + s] and the scaled Jacobian [Jh 0; Jg S].
    let eval = |x: &[f64], s: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let h = problem.equalities(x);
        let g = problem.inequalities(x);
        let (jh, jg) = (problem.equality_jacobian(x), problem.inequality_jacobian(x));
        let mut c = h;
        c.extend(g.iter().zip(s).map(|(a, b)| a + b));
        let mut a: Vec<Vec<f64>> = jh.into_iter().map(|mut r| {
            r.extend(std::iter::repeat_n(0.0, s.len()));
            r
        }).collect();
        for (i, mut r) in jg.into_iter().enumerate() {
            r.extend((0..s.len()).map(|k| if k == i { s[i] } else { 0.0 }));
            a.push(r);
        }
        (c, a)
    };
    let converged;
    let mut lambda: Vec<f64>;
    loop {
        // Solve the barrier problem for this μ.
        loop {
            let fx = problem.objective(&x);
            let gx = problem.gradient(&x);
            let (c, a) = eval(&x, &s);
            let mut grad = gx.clone();
            grad.extend(std::iter::repeat_n(-mu, ng));
            // Least-squares multipliers and the optimality of the barrier problem.
            lambda = if a.is_empty() { Vec::new() } else { aat_solve(&a, &matvec(&a, &grad)).iter().map(|t| -t).collect() };
            let lag: Vec<f64> = grad.iter().zip(rmatvec(&a, &lambda, n + ng)).map(|(p, q)| p + q).collect();
            let opt = lag.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let viol = c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if (opt <= barrier_tol && viol <= barrier_tol) || iterations >= options.max_iterations {
                break;
            }
            iterations += 1;
            let hmul = |d: &[f64]| -> Vec<f64> {
                let mut out = matvec(&bfgs, &d[..n]);
                out.extend(d[n..].iter().map(|t| mu * t));
                out
            };
            let d = byrd_omojokun_step(&hmul, &grad, &a, &c, delta, n);
            // Merit: barrier objective plus ν‖c‖, with ν raised to make the step a descent.
            let ad: Vec<f64> = matvec(&a, &d).iter().zip(&c).map(|(p, q)| p + q).collect();
            let quad = dot(&grad, &d) + 0.5 * dot(&d, &hmul(&d));
            let vpred = norm(&c) - norm(&ad);
            if vpred > 1e-14 {
                nu = nu.max(quad / (0.9 * vpred) + 1e-8);
            }
            let predicted = -quad + nu * vpred;
            let barrier = |fx: f64, s: &[f64]| fx - mu * s.iter().map(|v| v.ln()).sum::<f64>();
            let x_new: Vec<f64> = x.iter().zip(&d[..n]).map(|(p, q)| p + q).collect();
            let s_new: Vec<f64> = s.iter().zip(&d[n..]).map(|(p, q)| p * (1.0 + q)).collect();
            let (c_new, _) = eval(&x_new, &s_new);
            let f_new = problem.objective(&x_new);
            let actual = barrier(fx, &s) + nu * norm(&c) - barrier(f_new, &s_new) - nu * norm(&c_new);
            let ratio = if predicted > 0.0 { actual / predicted } else { -1.0 };
            let dn = norm(&d);
            if ratio >= 0.75 {
                delta = delta.max(7.0 * dn);
            } else if ratio >= 0.3 {
                delta = delta.max(2.0 * dn);
            } else if ratio < 0.1 {
                delta = 0.25 * dn.max(1e-16);
            }
            if delta < 1e-14 {
                break;
            }
            if ratio > 1e-4 && f_new.is_finite() && s_new.iter().all(|v| *v > 0.0) {
                // Damped BFGS on the Lagrangian's x-part, with the current multipliers.
                let lag_x = |x: &[f64]| -> Vec<f64> {
                    let mut l = problem.gradient(x);
                    let mut jac = problem.equality_jacobian(x);
                    jac.extend(problem.inequality_jacobian(x));
                    for (row, lm) in jac.iter().zip(&lambda) {
                        l.iter_mut().zip(row).for_each(|(o, r)| *o += lm * r);
                    }
                    l
                };
                let (l0, l1) = (lag_x(&x), lag_x(&x_new));
                let sv: Vec<f64> = d[..n].to_vec();
                let mut yv: Vec<f64> = l1.iter().zip(&l0).map(|(p, q)| p - q).collect();
                let bs = matvec(&bfgs, &sv);
                let sbs = dot(&sv, &bs);
                let sy = dot(&sv, &yv);
                if sbs > 1e-300 {
                    if sy < 0.2 * sbs {
                        let theta = 0.8 * sbs / (sbs - sy);
                        yv = yv.iter().zip(&bs).map(|(y, b)| theta * y + (1.0 - theta) * b).collect();
                    }
                    let sy = dot(&sv, &yv);
                    if sy > 1e-300 {
                        for i in 0..n {
                            for j in 0..n {
                                bfgs[i][j] += yv[i] * yv[j] / sy - bs[i] * bs[j] / sbs;
                            }
                        }
                    }
                }
                x = x_new;
                s = s_new;
            }
        }
        if ng == 0 || iterations >= options.max_iterations {
            converged = iterations < options.max_iterations;
            break;
        }
        if barrier_tol <= options.tolerance {
            converged = true;
            break;
        }
        mu *= 0.2;
        barrier_tol = (0.2 * barrier_tol).max(options.tolerance);
        delta = delta.max(1.0);
    }
    let h = problem.equalities(&x);
    let g = problem.inequalities(&x);
    let violation = h.iter().map(|v| v.abs()).chain(g.iter().map(|v| v.max(0.0))).fold(0.0, f64::max);
    let ne = h.len();
    let (eq, ineq) = if lambda.len() == ne + ng { (lambda[..ne].to_vec(), lambda[ne..].to_vec()) } else { (vec![0.0; ne], vec![0.0; ng]) };
    Ok(ConstrainedResult {
        objective_value: problem.objective(&x),
        parameters: x,
        equality_multipliers: eq,
        inequality_multipliers: ineq,
        constraint_violation: violation,
        iterations,
        converged: converged && violation <= options.tolerance.max(1e-6),
    })
}
// ANCHOR_END: trust_constr

#[cfg(test)]
mod tests {
    use super::*;

    struct Hs071;
    impl ConstrainedProblem for Hs071 {
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
        }
        fn equalities(&self, x: &[f64]) -> Vec<f64> {
            vec![x.iter().map(|v| v * v).sum::<f64>() - 40.0]
        }
        fn inequalities(&self, x: &[f64]) -> Vec<f64> {
            let mut g = vec![25.0 - x.iter().product::<f64>()];
            for &v in x {
                g.push(1.0 - v);
                g.push(v - 5.0);
            }
            g
        }
    }

    struct Circle;
    impl ConstrainedProblem for Circle {
        // min x + y on the unit circle: (−1/√2, −1/√2).
        fn objective(&self, x: &[f64]) -> f64 {
            x[0] + x[1]
        }
        fn equalities(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1] - 1.0]
        }
    }

    #[test]
    fn solves_equality_and_mixed_problems() {
        let o = ConstrainedOptions { tolerance: 1e-8, max_iterations: 1000 };
        let r = trust_constr(&Circle, vec![2.0, 0.5], o).unwrap();
        let s = 0.5f64.sqrt();
        assert!(r.converged && (r.parameters[0] + s).abs() < 1e-7 && (r.parameters[1] + s).abs() < 1e-7, "{:?}", r.parameters);
        assert!((r.equality_multipliers[0] - s).abs() < 1e-6, "{:?}", r.equality_multipliers);
        // HS071: the optimum (1, 4.743, 3.821, 1.379) with f = 17.014.
        let r = trust_constr(&Hs071, vec![1.0, 5.0, 5.0, 1.0], o).unwrap();
        assert!(r.converged && (r.objective_value - 17.0140173).abs() < 1e-5, "{:?} {} {}", r.parameters, r.objective_value, r.iterations);
        assert!((r.parameters[1] - 4.743).abs() < 1e-3, "{:?}", r.parameters);
    }
}
