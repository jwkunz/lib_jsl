//! Adaptive Dormand-Prince 5(4) integration of ordinary differential equations.
//!
//! See the book entry [Runge-Kutta methods](https://jwkunz.github.io/lib_jsl/book/integration/runge-kutta.html).

use crate::stiff_ode::{OdeOptions, OdeSolution, starting_step, validate, weighted_max};
use lib_jsl_core::ErrorsJSL;

// ANCHOR: tableau
/// The nodes `c`, coefficients `a` and fifth-order weights (last row of `a`) of the
/// Dormand-Prince 5(4) pair. The seventh stage is evaluated at the new solution, so it is the
/// first stage of the next step ("first same as last").
const C: [f64; 7] = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];
const A: [[f64; 6]; 7] = [
    [0.0; 6],
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
];
/// Fifth-order weights minus the embedded fourth-order weights: `h Σ eⱼ kⱼ` estimates the
/// local error.
const E: [f64; 7] = [
    35.0 / 384.0 - 5179.0 / 57600.0,
    0.0,
    500.0 / 1113.0 - 7571.0 / 16695.0,
    125.0 / 192.0 - 393.0 / 640.0,
    -2187.0 / 6784.0 + 92097.0 / 339200.0,
    11.0 / 84.0 - 187.0 / 2100.0,
    -1.0 / 40.0,
];
// ANCHOR_END: tableau

/// Solves `y' = f(t, y)` from `(t0, y0)` to `t1` (which may be less than `t0`) with the
/// adaptive Dormand-Prince 5(4) method, the method of MATLAB's `ode45` and SciPy's `RK45`.
///
/// Each step costs six new evaluations of `f` (the seventh is reused by the next step). The
/// embedded fourth-order solution estimates the local error, and steps are accepted when every
/// component's error is below `absolute_tolerance + relative_tolerance · |yᵢ|`; the step size
/// then grows or shrinks by `0.9 · error^(−1/5)`, between a fifth and five times. Returns every
/// accepted step. For stiff problems, where this method is forced into tiny steps, use
/// [`rosenbrock23`](crate::stiff_ode::rosenbrock23) or [`bdf`](crate::stiff_ode::bdf).
pub fn dormand_prince_solve<F>(f: F, t0: f64, y0: &[f64], t1: f64, options: OdeOptions) -> Result<OdeSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate(y0, t0, t1, &options)?;
    let n = y0.len();
    let eval = |t: f64, y: &[f64]| -> Result<Vec<f64>, ErrorsJSL> {
        let v = f(t, y);
        if v.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((v.len(), n)));
        }
        Ok(v)
    };
    let span = t1 - t0;
    let direction = span.signum();
    let (mut t, mut y) = (t0, y0.to_vec());
    let mut solution = OdeSolution { t: vec![t], y: vec![y.clone()], rejected_steps: 0, function_evaluations: 1, jacobian_evaluations: 0 };
    let mut k = vec![eval(t, &y)?];
    let mut h = starting_step(&y, &k[0], span, &options);
    for _ in 0..options.max_steps {
        if span == 0.0 || (t1 - t) * direction <= 0.0 {
            return Ok(solution);
        }
        // ANCHOR: dopri_step
        h = h.min((t1 - t).abs());
        let hs = h * direction;
        k.truncate(1);
        for s in 1..7 {
            let stage: Vec<f64> = (0..n).map(|i| y[i] + hs * (0..s).map(|j| A[s][j] * k[j][i]).sum::<f64>()).collect();
            let t_stage = t + C[s] * hs;
            k.push(eval(t_stage, &stage)?);
        }
        solution.function_evaluations += 6;
        // Stage 7 was evaluated at the fifth-order solution itself.
        let y_new: Vec<f64> = (0..n).map(|i| y[i] + hs * (0..6).map(|j| A[6][j] * k[j][i]).sum::<f64>()).collect();
        let local: Vec<f64> = (0..n).map(|i| hs * (0..7).map(|j| E[j] * k[j][i]).sum::<f64>()).collect();
        let error = weighted_max(&local, &y, &y_new, &options);
        if error <= 1.0 && y_new.iter().all(|v| v.is_finite()) {
            t = if (t1 - t).abs() <= h { t1 } else { t + hs };
            y = y_new;
            let last = k.pop().expect("seven stages");
            k = vec![last];
            solution.t.push(t);
            solution.y.push(y.clone());
        } else {
            solution.rejected_steps += 1;
        }
        let factor = if error.is_finite() { 0.9 * error.max(1e-10).powf(-0.2) } else { 0.1 };
        h *= factor.clamp(0.2, 5.0);
        // ANCHOR_END: dopri_step
        if h < 1e-14 * span.abs().max(t.abs()) {
            return Err(ErrorsJSL::RuntimeError("Dormand-Prince: the step size underflowed (the problem may be stiff or singular)."));
        }
    }
    Err(ErrorsJSL::RuntimeError("Dormand-Prince: too many steps (the problem may be stiff)."))
}

#[deprecated(note = "this solves only y' = f(t); use `dormand_prince_solve` for y' = f(t, y)")]
#[allow(deprecated)]
pub fn dormand_prince<F>(f: &F, t0: f64, y0: f64, t1: f64, tolerance : f64) -> Vec<(f64,f64,f64)>
where
    F: Fn(f64) -> f64,
{
    let mut t = t0;
    let mut y = y0;
    let mut h = tolerance;
    let mut result = Vec::<(f64,f64,f64)>::new();
    result.push((t,y0,h));
    while t <  t1{
        let step = dormand_prince_adaptive_step(f, t, y, h, tolerance);
        result.push(step);
        t = step.0;
        y = step.1;
        h = step.2.min(t1-t);
    }
    result
}

/// One Dormand-Prince step for `y' = f(t)` (a derivative independent of `y`): the fifth-order
/// value and the difference from the embedded fourth-order value.
#[deprecated(note = "this solves only y' = f(t); use `dormand_prince_solve` for y' = f(t, y)")]
pub fn dormand_prince_step<F>(
    f: &F,
    t: f64,
    y: f64,
    h: f64,
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let k1 = h * f(t);
    //let k2 = h * f(t + h * 1.0 / 5.0);
    let k3 = h * f(t + h * 3.0 / 10.0);
    let k4 = h * f(t + h * 4.0 / 5.0);
    let k5 = h * f(t + h * 8.0 / 9.0);
    let k6 = h * f(t + h);
    let k7 = h * f(t + h);

    // 5th order
    let y5 = y
        + (35.0 / 384.0) * k1
        + (500.0 / 1113.0) * k3
        + (125.0 / 192.0) * k4
        - (2187.0 / 6784.0) * k5
        + (11.0 / 84.0) * k6;

    // 4th order (embedded)
    let y4 = y
        + (5179.0 / 57600.0) * k1
        + (7571.0 / 16695.0) * k3
        + (393.0 / 640.0) * k4
        - (92097.0 / 339200.0) * k5
        + (187.0 / 2100.0) * k6
        + (1.0 / 40.0) * k7;

    let err = (y5 - y4).abs();
    (y5, err)
}


/// One adaptive Dormand-Prince step for `y' = f(t)` (a derivative independent of `y`).
///
/// Attempts to advance from (t, y) using initial step h.
/// On success, returns (t_next, y_next, h_next).
#[deprecated(note = "this solves only y' = f(t); use `dormand_prince_solve` for y' = f(t, y)")]
#[allow(deprecated)]
pub fn dormand_prince_adaptive_step<F>(
    f: &F,
    t: f64,
    y: f64,
    h: f64,
    tol: f64,
) -> (f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    // Controller parameters
    const SAFETY: f64 = 0.9;
    const MIN_SCALE: f64 = 0.2;
    const MAX_SCALE: f64 = 5.0;
    const ORDER: f64 = 4.0; // lower order of the embedded pair

    let mut h_try = h;

    loop {
        let (y_next, err) = dormand_prince_step(f, t, y, h_try);

        // Scale error relative to tolerance
        let err_norm = err.max(1e-14); // prevent division by zero

        if err_norm <= tol {
            // Accept step
            let scale = (tol / err_norm).powf(1.0 / (ORDER + 1.0));
            let scale = scale.clamp(MIN_SCALE, MAX_SCALE);

            let h_next = SAFETY * h_try * scale;
            return (t + h_try, y_next, h_next);
        } else {
            // Reject step, reduce h
            let scale = (tol / err_norm).powf(1.0 / (ORDER + 1.0));
            let scale = scale.clamp(MIN_SCALE, 1.0);

            h_try *= SAFETY * scale;
        }
    }
}



#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    #[allow(deprecated)]
    fn the_legacy_function_integrates_a_derivative_of_t() {
        let result = dormand_prince(&f64::sin, 0.0, 0.0, PI / 2.0, 1e-6);
        assert!((1.0 - result.last().unwrap().1).abs() < 1e-6);
    }

    #[test]
    fn solves_systems_to_the_requested_accuracy() {
        // The harmonic oscillator over five periods.
        let oscillator = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        for tol in [1e-6, 1e-9, 1e-12] {
            let options = OdeOptions { relative_tolerance: tol, absolute_tolerance: tol, ..Default::default() };
            let s = dormand_prince_solve(oscillator, 0.0, &[1.0, 0.0], 10.0 * PI, options).unwrap();
            let y = s.y.last().unwrap();
            let error = ((y[0] - 1.0).powi(2) + y[1].powi(2)).sqrt();
            assert!(error < 1000.0 * tol, "tol {tol}: error {error}");
            assert_eq!(*s.t.last().unwrap(), 10.0 * PI);
        }
        // Backwards in time, and y' = y: e^{-1} at t = −1.
        let s = dormand_prince_solve(|_, y| vec![y[0]], 0.0, &[1.0], -1.0, OdeOptions { relative_tolerance: 1e-10, absolute_tolerance: 1e-12, ..Default::default() }).unwrap();
        assert!((s.y.last().unwrap()[0] - (-1f64).exp()).abs() < 1e-9);
        // First same as last: six evaluations per attempted step, plus the first.
        assert_eq!(s.function_evaluations, 1 + 6 * (s.t.len() - 1 + s.rejected_steps));
    }

    #[test]
    fn reports_failures() {
        assert!(dormand_prince_solve(|_, y| vec![y[0]], 0.0, &[], 1.0, OdeOptions::default()).is_err());
        // y' = y² from y(0) = 1 blows up at t = 1.
        assert!(dormand_prince_solve(|_, y| vec![y[0] * y[0]], 0.0, &[1.0], 2.0, OdeOptions::default()).is_err());
    }
}
