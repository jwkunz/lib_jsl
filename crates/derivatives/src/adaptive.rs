//! Adaptive derivatives with error estimates by Richardson extrapolation (Ridders' method), the
//! counterpart of `scipy.differentiate.derivative`, `jacobian` and `hessian`.
//!
//! Central differences with a decreasing sequence of steps `h, h/c, h/c², …` are combined in a
//! Neville tableau that cancels the even powers of `h` in their truncation error. The entry whose
//! change from its neighbours is smallest is returned, with that change as the error estimate,
//! and the iteration stops when rounding error starts to dominate.
//!
//! See the book entry [Adaptive derivatives](https://jwkunz.github.io/lib_jsl/book/derivatives/adaptive.html).

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::needless_range_loop)]

use lib_jsl_core::ErrorsJSL;

/// Options for [`derivative`].
#[derive(Clone, Copy, Debug)]
pub struct DerivativeOptions {
    /// The first (largest) step `h`, absolute.
    pub initial_step: f64,
    /// Each step is the previous divided by this factor (> 1).
    pub step_factor: f64,
    /// The largest number of steps (tableau rows).
    pub max_iterations: usize,
    /// Stop once the error estimate is below `atol + rtol |f'|`.
    pub atol: f64,
    pub rtol: f64,
}

impl Default for DerivativeOptions {
    /// SciPy's defaults for the step: 0.5, reduced by factors of 2, at most 10 times.
    fn default() -> Self {
        DerivativeOptions { initial_step: 0.5, step_factor: 2.0, max_iterations: 10, atol: 0.0, rtol: f64::EPSILON.sqrt() * 1e-3 }
    }
}

/// The result of [`derivative`].
#[derive(Clone, Copy, Debug)]
pub struct DerivativeResult {
    pub df: f64,
    /// The estimated absolute error.
    pub error: f64,
    /// Evaluations of the function.
    pub nfev: usize,
    /// `true` when the tolerance was met.
    pub success: bool,
}

// ANCHOR: ridders
/// `f'(x)` with an error estimate, by Ridders' extrapolation of central differences.
///
/// Row `i` of the tableau starts from `D(hᵢ) = (f(x + hᵢ) − f(x − hᵢ)) / 2hᵢ` with `hᵢ = h/cⁱ`; each
/// column `j` removes the next even power of `h`: `T(i, j) = (c^{2j} T(i, j−1) − T(i−1, j−1)) /
/// (c^{2j} − 1)`. The best estimate is the entry whose difference from its two parents is
/// smallest. Extrapolation stops when the diagonal grows by more than twice the best error, the
/// sign that rounding error dominates.
pub fn derivative(f: impl Fn(f64) -> f64, x: f64, options: DerivativeOptions) -> Result<DerivativeResult, ErrorsJSL> {
    if !x.is_finite() || !(options.initial_step > 0.0) || !(options.step_factor > 1.0) || options.max_iterations == 0 {
        return Err(ErrorsJSL::InvalidInputRange("need a finite point, a positive step and a step factor above 1"));
    }
    let c2 = options.step_factor * options.step_factor;
    let mut h = options.initial_step;
    let mut nfev = 0;
    let mut central = |h: f64| {
        nfev += 2;
        (f(x + h) - f(x - h)) / (2.0 * h)
    };
    let mut prev_row = vec![central(h)];
    let (mut best, mut err) = (prev_row[0], f64::INFINITY);
    for iteration in 1..options.max_iterations {
        h /= options.step_factor;
        let mut row = vec![central(h)];
        let mut fac = c2;
        for j in 1..=prev_row.len() {
            let t = (row[j - 1] * fac - prev_row[j - 1]) / (fac - 1.0);
            fac *= c2;
            let errt = (t - row[j - 1]).abs().max((t - prev_row[j - 1]).abs());
            if errt <= err {
                err = errt;
                best = t;
            }
            row.push(t);
        }
        // The divergence test is meaningful only once the steps are small enough for the
        // extrapolation to have converged at least roughly: after a few rows.
        // Stop on divergence only once the estimate is already fairly accurate: early rows from a
        // step that is too large for the function are inaccurate rather than noisy.
        let diverging = iteration >= 3 && err <= f64::EPSILON.sqrt() * best.abs().max(1e-300) && (row[row.len() - 1] - prev_row[prev_row.len() - 1]).abs() >= 2.0 * err;
        prev_row = row;
        if err <= options.atol + options.rtol * best.abs() || diverging {
            break;
        }
    }
    Ok(DerivativeResult { df: best, error: err, nfev, success: err <= options.atol + options.rtol * best.abs() || err.is_finite() && err <= 1e-8 * best.abs().max(1.0) })
}
// ANCHOR_END: ridders

/// The Jacobian `Jᵢⱼ = ∂fᵢ/∂xⱼ` of `f: ℝⁿ → ℝᵐ`, each column by Ridders' method on the
/// coordinate direction (as `scipy.differentiate.jacobian`), with the largest error estimate.
pub fn jacobian(f: impl Fn(&[f64]) -> Vec<f64>, x: &[f64], options: DerivativeOptions) -> Result<(Vec<Vec<f64>>, f64), ErrorsJSL> {
    let m = f(x).len();
    let n = x.len();
    let mut jac = vec![vec![0.0; n]; m];
    let mut worst: f64 = 0.0;
    for j in 0..n {
        for (i, row) in jac.iter_mut().enumerate() {
            let g = |t: f64| {
                let mut z = x.to_vec();
                z[j] = t;
                f(&z)[i]
            };
            let r = derivative(g, x[j], options)?;
            row[j] = r.df;
            worst = worst.max(r.error);
        }
    }
    Ok((jac, worst))
}

/// The Hessian `Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ` of `f: ℝⁿ → ℝ`, as the Jacobian of the gradient, both by
/// Ridders' method (as `scipy.differentiate.hessian`), symmetrised.
pub fn hessian(f: impl Fn(&[f64]) -> f64, x: &[f64], options: DerivativeOptions) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let n = x.len();
    let inner = DerivativeOptions { initial_step: options.initial_step / 4.0, ..options };
    let grad = |z: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                derivative(
                    |t| {
                        let mut w = z.to_vec();
                        w[i] = t;
                        f(&w)
                    },
                    z[i],
                    inner,
                )
                .map(|r| r.df)
                .unwrap_or(f64::NAN)
            })
            .collect()
    };
    let (mut h, _) = jacobian(grad, x, options)?;
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = s;
            h[j][i] = s;
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_of_smooth_functions_are_near_machine_precision() {
        for &x in &[0.3, 1.0, 2.7] {
            let r = derivative(f64::exp, x, DerivativeOptions::default()).unwrap();
            assert!((r.df - x.exp()).abs() < 1e-12 * x.exp(), "{x}: {:?}", r);
            assert!(r.error < 1e-9 * x.exp());
            let r = derivative(|t| (3.0 * t).sin(), x, DerivativeOptions::default()).unwrap();
            assert!((r.df - 3.0 * (3.0 * x).cos()).abs() < 1e-11);
        }
    }

    #[test]
    fn jacobian_and_hessian_of_simple_maps() {
        let f = |v: &[f64]| vec![v[0] * v[1], v[0].sin() + v[1] * v[1]];
        let (j, _) = jacobian(f, &[0.5, 2.0], DerivativeOptions::default()).unwrap();
        let want = [[2.0, 0.5], [0.5f64.cos(), 4.0]];
        for i in 0..2 {
            for k in 0..2 {
                assert!((j[i][k] - want[i][k]).abs() < 1e-11);
            }
        }
        let h = hessian(|v: &[f64]| v[0] * v[0] * v[1] + v[1].exp(), &[1.5, 0.5], DerivativeOptions::default()).unwrap();
        let want = [[2.0 * 0.5, 2.0 * 1.5], [2.0 * 1.5, 0.5f64.exp()]];
        for i in 0..2 {
            for k in 0..2 {
                assert!((h[i][k] - want[i][k]).abs() < 1e-8, "{h:?}");
            }
        }
    }
}
