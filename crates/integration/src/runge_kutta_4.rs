//! The classical fourth-order Runge-Kutta method with a fixed step.
//!
//! See the book entry [Runge-Kutta methods](https://jwkunz.github.io/lib_jsl/book/integration/runge-kutta.html).

use crate::stiff_ode::OdeSolution;
use lib_jsl_core::ErrorsJSL;

/// Solves `y' = f(t, y)` from `(t0, y0)` to `t1` with `n_steps` steps of the classical
/// fourth-order Runge-Kutta method (RK4), returning the state after every step.
///
/// Each step costs four evaluations of `f` and has local error `O(h⁵)`, so the global error is
/// `O(h⁴)`: halving the step divides it by about 16. There is no error control; use
/// [`dormand_prince_solve`](crate::dormand_prince::dormand_prince_solve) when the accuracy must be
/// guaranteed or the solution changes on very different time scales.
pub fn rk4_solve<F>(f: F, t0: f64, y0: &[f64], t1: f64, n_steps: usize) -> Result<OdeSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if y0.is_empty() || n_steps == 0 || !t0.is_finite() || !t1.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("RK4 needs a non-empty state, at least one step and a finite span."));
    }
    let n = y0.len();
    let h = (t1 - t0) / n_steps as f64;
    let mut solution = OdeSolution { t: vec![t0], y: vec![y0.to_vec()], rejected_steps: 0, function_evaluations: 0, jacobian_evaluations: 0 };
    let mut y = y0.to_vec();
    let eval = |t: f64, y: &[f64]| -> Result<Vec<f64>, ErrorsJSL> {
        let v = f(t, y);
        if v.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((v.len(), n)));
        }
        Ok(v)
    };
    let axpy = |y: &[f64], a: f64, k: &[f64]| -> Vec<f64> { y.iter().zip(k).map(|(y, k)| y + a * k).collect() };
    for step in 0..n_steps {
        let t = t0 + step as f64 * h;
        // ANCHOR: rk4_step
        let k1 = eval(t, &y)?;
        let k2 = eval(t + 0.5 * h, &axpy(&y, 0.5 * h, &k1))?;
        let k3 = eval(t + 0.5 * h, &axpy(&y, 0.5 * h, &k2))?;
        let k4 = eval(t + h, &axpy(&y, h, &k3))?;
        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        // ANCHOR_END: rk4_step
        solution.function_evaluations += 4;
        solution.t.push(if step + 1 == n_steps { t1 } else { t + h });
        solution.y.push(y.clone());
    }
    Ok(solution)
}

/// Integrates `y' = f(t)`, a derivative that does not depend on `y`, with RK4 steps.
///
/// Despite its name this is quadrature, not an ODE solver: when `f` ignores `y` the two middle
/// stages coincide and each step is exactly Simpson's rule. Returns `y` after every step.
#[deprecated(note = "this solves only y' = f(t); use `rk4_solve` for y' = f(t, y)")]
pub fn runge_kutta_4<F>(f: F, t0: f64, y0: f64, t1: f64, n_steps: usize) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let dt = (t1 - t0) / n_steps as f64;
    let mut y = y0;
    let mut result = vec![y0];
    for step in 0..n_steps {
        let t = t0 + step as f64 * dt;
        y += dt * (f(t) + 4.0 * f(t + 0.5 * dt) + f(t + dt)) / 6.0;
        result.push(y);
    }
    result
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;

    /// The harmonic oscillator y'' = −y as a first-order system.
    fn oscillator(_t: f64, y: &[f64]) -> Vec<f64> {
        vec![y[1], -y[0]]
    }

    #[test]
    fn fourth_order_convergence_on_a_system() {
        let error = |n| {
            let s = rk4_solve(oscillator, 0.0, &[1.0, 0.0], 2.0 * PI, n).unwrap();
            let y = s.y.last().unwrap();
            ((y[0] - 1.0).powi(2) + y[1].powi(2)).sqrt()
        };
        let ratio = error(50) / error(100);
        assert!((ratio - 16.0).abs() < 0.5, "{ratio}");
        assert!(error(200) < 1e-7);
        let s = rk4_solve(oscillator, 0.0, &[1.0, 0.0], 1.0, 10).unwrap();
        assert_eq!((s.t.len(), s.function_evaluations, *s.t.last().unwrap()), (11, 40, 1.0));
    }

    #[test]
    fn validation() {
        assert!(rk4_solve(oscillator, 0.0, &[], 1.0, 10).is_err());
        assert!(rk4_solve(oscillator, 0.0, &[1.0, 0.0], 1.0, 0).is_err());
        assert!(rk4_solve(|_, _| vec![1.0], 0.0, &[1.0, 0.0], 1.0, 4).is_err());
    }

    #[test]
    #[allow(deprecated)]
    fn the_legacy_function_is_simpsons_rule() {
        let result = runge_kutta_4(|x| x.sin(), 0.0, 0.0, PI / 2.0, 100);
        assert!((1.0 - result.last().unwrap()).abs() < 1e-9);
    }
}
