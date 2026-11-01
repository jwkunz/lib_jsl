//! Solvers for stiff systems of ordinary differential equations `y' = f(t, y)`.
//!
//! Stiff problems mix fast decaying and slow modes; explicit methods must then take steps set
//! by the fastest mode even after it has died out. Both solvers here are implicit (linearly or
//! fully), so their step is limited by accuracy alone:
//!
//! * [`rosenbrock23`]: the L-stable Rosenbrock (linearly implicit) pair of Shampine and Reichelt
//!   (MATLAB's `ode23s`): one LU factorisation and three solves per step, order 2 with an order 3
//!   error estimate. Robust at loose tolerances.
//! * [`bdf`]: variable-step, variable-coefficient backward differentiation formulas of order up
//!   to 5 with Newton iteration, ramping the order up from 1 as history accumulates. More
//!   efficient at tight tolerances.
//!
//! The Jacobian `∂f/∂y` may be supplied; otherwise it is formed by forward differences.
//!
//! See the book entry [Stiff ODE solvers](https://jwkunz.github.io/lib_jsl/book/integration/stiff-ode.html).

use lib_jsl_core::ErrorsJSL;
use std::collections::VecDeque;

/// Tolerances and limits shared by the ODE solvers (stiff and non-stiff). Each step must keep
/// every component's local error below `absolute_tolerance + relative_tolerance · |yᵢ|`.
#[derive(Debug, Clone, Copy)]
pub struct OdeOptions {
    /// Error target relative to the size of each component.
    pub relative_tolerance: f64,
    /// Error target for components near zero.
    pub absolute_tolerance: f64,
    /// Initial step; chosen automatically when `None`.
    pub initial_step: Option<f64>,
    /// The largest number of steps (accepted and rejected).
    pub max_steps: usize,
}

/// The former name of [`OdeOptions`].
pub type StiffOdeOptions = OdeOptions;

impl Default for OdeOptions {
    fn default() -> Self {
        OdeOptions { relative_tolerance: 1e-6, absolute_tolerance: 1e-9, initial_step: None, max_steps: 100_000 }
    }
}

/// The accepted steps of a solve.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    /// The times of the accepted steps, starting with `t0` and ending with `t1`.
    pub t: Vec<f64>,
    /// The state at each time in `t`.
    pub y: Vec<Vec<f64>>,
    /// Steps whose error estimate exceeded the tolerance and were retried with a smaller step.
    pub rejected_steps: usize,
    /// Evaluations of `f`.
    pub function_evaluations: usize,
    /// Evaluations of the Jacobian (zero for explicit methods).
    pub jacobian_evaluations: usize,
}

impl OdeSolution {
    /// The state at the final time.
    pub fn last(&self) -> &[f64] {
        self.y.last().expect("a solution holds at least the initial state")
    }
}

/// Dense LU factorisation with partial pivoting, stored in place.
pub(crate) struct Lu {
    a: Vec<Vec<f64>>,
    pivot: Vec<usize>,
}

impl Lu {
    pub(crate) fn new(mut a: Vec<Vec<f64>>) -> Result<Self, ErrorsJSL> {
        let n = a.len();
        let mut pivot = vec![0; n];
        for k in 0..n {
            let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs())).expect("non-empty");
            if a[p][k] == 0.0 || !a[p][k].is_finite() {
                return Err(ErrorsJSL::RuntimeError("The iteration matrix is singular."));
            }
            a.swap(k, p);
            pivot[k] = p;
            for i in k + 1..n {
                let m = a[i][k] / a[k][k];
                a[i][k] = m;
                let (upper, lower) = a.split_at_mut(i);
                for (x, &pivot_row) in lower[0][k + 1..].iter_mut().zip(&upper[k][k + 1..]) {
                    *x -= m * pivot_row;
                }
            }
        }
        Ok(Lu { a, pivot })
    }

    pub(crate) fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = b.to_vec();
        for k in 0..n {
            x.swap(k, self.pivot[k]);
        }
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.a[i][j] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= self.a[i][j] * x[j];
            }
            x[i] /= self.a[i][i];
        }
        x
    }
}

/// Counts evaluations and supplies the Jacobian (analytic or by forward differences).
struct System<'a, F, J> {
    f: &'a F,
    jacobian: Option<&'a J>,
    evaluations: usize,
    jacobians: usize,
}

impl<F, J> System<'_, F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    fn eval(&mut self, t: f64, y: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
        self.evaluations += 1;
        let v = (self.f)(t, y);
        if v.len() != y.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((v.len(), y.len())));
        }
        Ok(v)
    }

    fn jacobian(&mut self, t: f64, y: &[f64], fy: &[f64]) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
        self.jacobians += 1;
        if let Some(j) = self.jacobian {
            return Ok(j(t, y));
        }
        let n = y.len();
        let mut jac = vec![vec![0.0; n]; n];
        let mut z = y.to_vec();
        for k in 0..n {
            let h = f64::EPSILON.sqrt() * y[k].abs().max(1e-5);
            z[k] = y[k] + h;
            let fz = self.eval(t, &z)?;
            z[k] = y[k];
            for i in 0..n {
                jac[i][k] = (fz[i] - fy[i]) / h;
            }
        }
        Ok(jac)
    }
}

pub(crate) fn weighted_max(v: &[f64], y0: &[f64], y1: &[f64], o: &OdeOptions) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, e)| e.abs() / (o.absolute_tolerance + o.relative_tolerance * y0[i].abs().max(y1[i].abs())))
        .fold(0.0, f64::max)
}

pub(crate) fn validate(y0: &[f64], t0: f64, t1: f64, o: &OdeOptions) -> Result<(), ErrorsJSL> {
    if y0.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("The state must not be empty."));
    }
    if !t0.is_finite() || !t1.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("The time span must be finite."));
    }
    if o.relative_tolerance.is_nan() || o.relative_tolerance <= 0.0 || o.absolute_tolerance.is_nan() || o.absolute_tolerance <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("Tolerances must be positive."));
    }
    Ok(())
}

/// Hairer's starting-step heuristic `0.01 ‖y₀‖ / ‖f₀‖` in the weighted norm, capped by the span.
pub(crate) fn starting_step(y0: &[f64], f0: &[f64], span: f64, o: &OdeOptions) -> f64 {
    if let Some(h) = o.initial_step {
        return h.abs().min(span.abs());
    }
    let d0 = weighted_max(y0, y0, y0, o);
    let d1 = weighted_max(f0, y0, y0, o);
    let h = if d0 < 1e-5 || d1 < 1e-5 { 1e-6 } else { 0.01 * d0 / d1 };
    h.min(span.abs()).max(1e-12 * span.abs())
}

/// Solves `y' = f(t, y)` from `(t0, y0)` to `t1` with the Rosenbrock 2(3) pair of `ode23s`.
/// `∂f/∂t` is estimated by a forward difference.
pub fn rosenbrock23<F, J>(f: F, jacobian: Option<J>, t0: f64, y0: &[f64], t1: f64, options: OdeOptions) -> Result<OdeSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    validate(y0, t0, t1, &options)?;
    let n = y0.len();
    let d = 1.0 / (2.0 + 2f64.sqrt());
    let e32 = 6.0 + 2f64.sqrt();
    let mut sys = System { f: &f, jacobian: jacobian.as_ref(), evaluations: 0, jacobians: 0 };
    let direction = (t1 - t0).signum();
    let (mut t, mut y) = (t0, y0.to_vec());
    let mut f0 = sys.eval(t, &y)?;
    let mut h = starting_step(&y, &f0, t1 - t0, &options);
    let mut solution = OdeSolution { t: vec![t], y: vec![y.clone()], rejected_steps: 0, function_evaluations: 0, jacobian_evaluations: 0 };
    let mut steps = 0;
    while (t1 - t) * direction > 0.0 {
        steps += 1;
        if steps > options.max_steps {
            return Err(ErrorsJSL::RuntimeError("The maximum number of steps was exceeded."));
        }
        h = h.min((t1 - t).abs());
        let hs = h * direction;
        // ANCHOR: rosenbrock_step
        let jac = sys.jacobian(t, &y, &f0)?;
        let dt = f64::EPSILON.sqrt() * t.abs().max(1.0);
        let f_t = sys.eval(t + dt, &y)?;
        let dfdt: Vec<f64> = (0..n).map(|i| (f_t[i] - f0[i]) / dt).collect();
        let w: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|k| if i == k { 1.0 } else { 0.0 } - hs * d * jac[i][k]).collect()).collect();
        let lu = Lu::new(w)?;
        let k1 = lu.solve(&(0..n).map(|i| f0[i] + hs * d * dfdt[i]).collect::<Vec<_>>());
        let y_half: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * hs * k1[i]).collect();
        let f1 = sys.eval(t + 0.5 * hs, &y_half)?;
        let k2: Vec<f64> = lu.solve(&(0..n).map(|i| f1[i] - k1[i]).collect::<Vec<_>>()).iter().zip(&k1).map(|(a, b)| a + b).collect();
        let y_new: Vec<f64> = (0..n).map(|i| y[i] + hs * k2[i]).collect();
        let f2 = sys.eval(t + hs, &y_new)?;
        let rhs: Vec<f64> = (0..n).map(|i| f2[i] - e32 * (k2[i] - f1[i]) - 2.0 * (k1[i] - f0[i]) + hs * d * dfdt[i]).collect();
        let k3 = lu.solve(&rhs);
        let err: Vec<f64> = (0..n).map(|i| hs / 6.0 * (k1[i] - 2.0 * k2[i] + k3[i])).collect();
        let error = weighted_max(&err, &y, &y_new, &options);
        // ANCHOR_END: rosenbrock_step
        if error <= 1.0 && y_new.iter().all(|v| v.is_finite()) {
            // Land exactly on t1 at the last step: t + (t1 − t) need not equal t1 in floating
            // point, and a leftover sliver would trip the step-size underflow check.
            t = if (t1 - t).abs() <= h { t1 } else { t + hs };
            y = y_new;
            f0 = f2;
            solution.t.push(t);
            solution.y.push(y.clone());
        } else {
            solution.rejected_steps += 1;
        }
        let factor = if error.is_finite() { 0.8 * error.max(1e-10).powf(-1.0 / 3.0) } else { 0.1 };
        h *= factor.clamp(0.1, 5.0);
        if h < 1e-14 * t.abs().max(1.0) {
            return Err(ErrorsJSL::RuntimeError("The step size underflowed."));
        }
    }
    solution.function_evaluations = sys.evaluations;
    solution.jacobian_evaluations = sys.jacobians;
    Ok(solution)
}

/// Derivative at `nodes[0]` of the Lagrange basis polynomials on `nodes`.
fn lagrange_derivative_at_first(nodes: &[f64]) -> Vec<f64> {
    let t0 = nodes[0];
    let mut weights = vec![0.0; nodes.len()];
    weights[0] = nodes[1..].iter().map(|&tk| 1.0 / (t0 - tk)).sum();
    for j in 1..nodes.len() {
        let product: f64 = (1..nodes.len()).filter(|&k| k != j).map(|k| (t0 - nodes[k]) / (nodes[j] - nodes[k])).product();
        weights[j] = product / (nodes[j] - t0);
    }
    weights
}

/// Lagrange extrapolation of the points `(times, values)` to `t`.
fn extrapolate(times: &[f64], values: &[&Vec<f64>], t: f64) -> Vec<f64> {
    let n = values[0].len();
    let mut out = vec![0.0; n];
    for (j, (&tj, vj)) in times.iter().zip(values).enumerate() {
        let l: f64 = times.iter().enumerate().filter(|&(k, _)| k != j).map(|(_, &tk)| (t - tk) / (tj - tk)).product();
        out.iter_mut().zip(vj.iter()).for_each(|(o, v)| *o += l * v);
    }
    out
}

/// Solves `y' = f(t, y)` from `(t0, y0)` to `t1` with variable-step BDF formulas of order up to
/// `max_order` (1 to 5). Orders 1 and 2 are A-stable; higher orders are more efficient at tight
/// tolerances but only stable in a wedge about the negative real axis (fine for dissipative
/// problems, less so for lightly damped oscillations). The local error is estimated from the
/// difference between the Newton-corrected solution and the polynomial predictor.
pub fn bdf<F, J>(f: F, jacobian: Option<J>, t0: f64, y0: &[f64], t1: f64, max_order: usize, options: OdeOptions) -> Result<OdeSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    validate(y0, t0, t1, &options)?;
    if !(1..=5).contains(&max_order) {
        return Err(ErrorsJSL::InvalidInputRange("The BDF order must be between 1 and 5."));
    }
    let n = y0.len();
    let mut sys = System { f: &f, jacobian: jacobian.as_ref(), evaluations: 0, jacobians: 0 };
    let direction = (t1 - t0).signum();
    let f0 = sys.eval(t0, y0)?;
    let mut h = starting_step(y0, &f0, t1 - t0, &options);
    let mut history: VecDeque<(f64, Vec<f64>)> = VecDeque::from([(t0, y0.to_vec())]);
    let mut f_last = f0;
    let mut solution = OdeSolution { t: vec![t0], y: vec![y0.to_vec()], rejected_steps: 0, function_evaluations: 0, jacobian_evaluations: 0 };
    let mut steps = 0;
    loop {
        let (t, y) = history.back().cloned().expect("history is never empty");
        if (t1 - t) * direction <= 0.0 {
            break;
        }
        steps += 1;
        if steps > options.max_steps {
            return Err(ErrorsJSL::RuntimeError("The maximum number of steps was exceeded."));
        }
        h = h.min((t1 - t).abs());
        let t_new = if (t1 - t).abs() <= h { t1 } else { t + direction * h };
        let order = max_order.min(history.len().saturating_sub(1)).max(1);
        // Predictor: extrapolate the last order+1 points, or an Euler step at start-up.
        let y_pred = if history.len() >= 2 {
            let start = history.len() - (order + 1).min(history.len());
            let times: Vec<f64> = history.range(start..).map(|p| p.0).collect();
            let values: Vec<&Vec<f64>> = history.range(start..).map(|p| &p.1).collect();
            extrapolate(&times, &values, t_new)
        } else {
            (0..n).map(|i| y[i] + (t_new - t) * f_last[i]).collect()
        };
        // ANCHOR: bdf_corrector
        // Corrector: α₀ y + Σ α_j y_{n+1−j} = f(t_new, y), solved by Newton's method.
        let mut nodes = vec![t_new];
        nodes.extend(history.iter().rev().take(order).map(|p| p.0));
        let alpha = lagrange_derivative_at_first(&nodes);
        let past: Vec<f64> = (0..n).map(|i| history.iter().rev().take(order).enumerate().map(|(j, p)| alpha[j + 1] * p.1[i]).sum()).collect();
        let f_pred = sys.eval(t_new, &y_pred)?;
        let jac = sys.jacobian(t_new, &y_pred, &f_pred)?;
        let lu = Lu::new((0..n).map(|i| (0..n).map(|k| if i == k { alpha[0] } else { 0.0 } - jac[i][k]).collect()).collect())?;
        let mut y_new = y_pred.clone();
        let mut f_new = f_pred;
        let mut converged = false;
        for _ in 0..8 {
            let residual: Vec<f64> = (0..n).map(|i| alpha[0] * y_new[i] + past[i] - f_new[i]).collect();
            let delta = lu.solve(&residual);
            y_new.iter_mut().zip(&delta).for_each(|(v, d)| *v -= d);
            f_new = sys.eval(t_new, &y_new)?;
            if !y_new.iter().all(|v| v.is_finite()) {
                break;
            }
            if weighted_max(&delta, &y, &y_new, &options) < 1e-3 {
                converged = true;
                break;
            }
        }
        if !converged {
            solution.rejected_steps += 1;
            h *= 0.25;
            if h < 1e-14 * t.abs().max(1.0) {
                return Err(ErrorsJSL::RuntimeError("Newton iteration failed at the minimum step size."));
            }
            continue;
        }
        // ANCHOR_END: bdf_corrector
        let scale = if history.len() >= 2 { 1.0 / (order as f64 + 2.0) } else { 0.5 };
        let err: Vec<f64> = (0..n).map(|i| scale * (y_new[i] - y_pred[i])).collect();
        let error = weighted_max(&err, &y, &y_new, &options);
        let growth_limit = if order <= 2 { 2.0 } else { 1.3 };
        let factor = (0.9 * error.max(1e-10).powf(-1.0 / (order as f64 + 1.0))).clamp(0.2, growth_limit);
        if error <= 1.0 {
            history.push_back((t_new, y_new.clone()));
            if history.len() > max_order + 1 {
                history.pop_front();
            }
            f_last = f_new;
            solution.t.push(t_new);
            solution.y.push(y_new);
        } else {
            solution.rejected_steps += 1;
        }
        h *= factor;
        if h < 1e-14 * t.abs().max(1.0) {
            return Err(ErrorsJSL::RuntimeError("The step size underflowed."));
        }
    }
    solution.function_evaluations = sys.evaluations;
    solution.jacobian_evaluations = sys.jacobians;
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    type NoJacobian = fn(f64, &[f64]) -> Vec<Vec<f64>>;

    fn robertson(_: f64, y: &[f64]) -> Vec<f64> {
        vec![
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
            3e7 * y[1] * y[1],
        ]
    }

    fn robertson_jacobian(_: f64, y: &[f64]) -> Vec<Vec<f64>> {
        vec![
            vec![-0.04, 1e4 * y[2], 1e4 * y[1]],
            vec![0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            vec![0.0, 6e7 * y[1], 0.0],
        ]
    }

    /// Robertson's chemical kinetics at t = 40 (Hairer and Wanner's reference solution).
    const ROBERTSON_40: [f64; 3] = [0.715_827_068_7, 9.185_534_765e-6, 0.284_163_746_6];

    #[test]
    fn both_solvers_handle_the_robertson_problem() {
        let options = OdeOptions { relative_tolerance: 1e-6, absolute_tolerance: 1e-10, ..Default::default() };
        let y0 = [1.0, 0.0, 0.0];
        let ros = rosenbrock23(robertson, Some(robertson_jacobian), 0.0, &y0, 40.0, options).unwrap();
        let bdf5 = bdf(robertson, None::<NoJacobian>, 0.0, &y0, 40.0, 5, options).unwrap();
        for (name, sol) in [("rosenbrock", &ros), ("bdf", &bdf5)] {
            let y = sol.last();
            for i in 0..3 {
                assert!((y[i] - ROBERTSON_40[i]).abs() < 1e-4 * ROBERTSON_40[i], "{name} component {i}: {y:?}");
            }
            // Mass is conserved by the exact flow; both methods preserve linear invariants.
            assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-10, "{name}");
            assert!(sol.t.len() < 2000, "{name} took {} steps", sol.t.len());
        }
    }

    #[test]
    fn stiff_linear_problem_follows_the_slow_manifold_cheaply() {
        // y' = −1000 (y − cos t) − sin t has solution cos t; explicit methods need h < 2e-3.
        let f = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos()) - t.sin()];
        let options = OdeOptions { relative_tolerance: 1e-7, absolute_tolerance: 1e-9, ..Default::default() };
        for order in [1, 2, 3, 5] {
            let sol = bdf(f, None::<NoJacobian>, 0.0, &[1.0], 10.0, order, options).unwrap();
            let worst = sol.t.iter().zip(&sol.y).map(|(t, y)| (y[0] - t.cos()).abs()).fold(0.0, f64::max);
            assert!(worst < 2e-5, "order {order}: {worst}");
            if order >= 3 {
                assert!(sol.t.len() < 1500, "order {order}: {} steps", sol.t.len());
            }
        }
        // The second-order Rosenbrock pair suits looser tolerances.
        let loose = OdeOptions { relative_tolerance: 1e-4, absolute_tolerance: 1e-6, ..Default::default() };
        let sol = rosenbrock23(f, None::<NoJacobian>, 0.0, &[1.0], 10.0, loose).unwrap();
        let worst = sol.t.iter().zip(&sol.y).map(|(t, y)| (y[0] - t.cos()).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-4 && sol.t.len() < 1000, "{worst} in {} steps", sol.t.len());
    }

    #[test]
    fn higher_orders_need_fewer_steps_and_backward_integration_works() {
        let f = |_: f64, y: &[f64]| vec![y[1], -y[0]];
        let options = OdeOptions { relative_tolerance: 1e-9, absolute_tolerance: 1e-12, ..Default::default() };
        let low = bdf(f, None::<NoJacobian>, 0.0, &[0.0, 1.0], 2.0, 1, options).unwrap();
        let high = bdf(f, None::<NoJacobian>, 0.0, &[0.0, 1.0], 2.0, 5, options).unwrap();
        assert!(high.t.len() * 10 < low.t.len(), "{} vs {}", high.t.len(), low.t.len());
        assert!((high.last()[0] - 2f64.sin()).abs() < 1e-7);
        // Backwards in time: y' = y from y(1) = e to t = 0.
        let back = rosenbrock23(|_, y: &[f64]| vec![y[0]], None::<NoJacobian>, 1.0, &[std::f64::consts::E], 0.0, OdeOptions::default()).unwrap();
        assert!((back.last()[0] - 1.0).abs() < 1e-4 && *back.t.last().unwrap() == 0.0);
        assert!(bdf(f, None::<NoJacobian>, 0.0, &[0.0, 1.0], 1.0, 6, options).is_err());
        assert!(rosenbrock23(f, None::<NoJacobian>, 0.0, &[], 1.0, options).is_err());
    }
}
