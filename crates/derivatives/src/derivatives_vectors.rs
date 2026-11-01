//! Finite difference gradients, Hessians and Jacobians of functions of several variables.
//!
//! See the book entry [Finite difference derivatives](https://jwkunz.github.io/lib_jsl/book/derivatives/finite-differences.html).

use crate::derivatives_1d_scalar::{optimal_relative_step, representable_step};
use lib_jsl_core::ErrorsJSL;

/// The gradient `∇f(x)` of a scalar function of several variables by central differences, one
/// coordinate at a time (`2n` evaluations).
///
/// `step_size` is relative: coordinate `i` is perturbed by `h · max(|xᵢ|, 1)`, with default
/// `h = ε^{1/3}` (about ten correct digits for well-scaled functions).
pub fn gradient_2_point<T>(f: &T, x: &[f64], step_size: Option<f64>) -> Result<Vec<f64>, ErrorsJSL>
where
    T: Fn(&[f64]) -> f64,
{
    let relative = step_size.unwrap_or(optimal_relative_step(2, 1));
    let mut gradient = vec![0.0; x.len()];
    let mut point = x.to_vec();
    for i in 0..x.len() {
        let h = representable_step(x[i], relative);
        point[i] = x[i] + h;
        let forward = f(&point);
        point[i] = x[i] - h;
        let backward = f(&point);
        point[i] = x[i];
        gradient[i] = (forward - backward) / (2.0 * h);
    }
    Ok(gradient)
}

/// The Hessian `∇²f(x)` of a scalar function by central differences: second differences on the
/// diagonal and the four-point cross difference off it (`1 + 2n + 2n(n − 1)` evaluations).
///
/// `step_size` is relative, with default `h = ε^{1/4}` (about eight correct digits): the step that
/// balances the `O(h²)` truncation error of a second difference against its `ε/h²` rounding error.
pub fn hessian_2_point<T>(f: &T, x: &[f64], step_size: Option<f64>) -> Result<Vec<Vec<f64>>, ErrorsJSL>
where
    T: Fn(&[f64]) -> f64,
{
    let relative = step_size.unwrap_or(optimal_relative_step(2, 2));
    let n = x.len();
    let steps: Vec<f64> = x.iter().map(|&xi| representable_step(xi, relative)).collect();
    let mut hessian = vec![vec![0.0; n]; n];
    let fx = f(x);
    let mut point = x.to_vec();
    for i in 0..n {
        let hi = steps[i];
        point[i] = x[i] + hi;
        let forward = f(&point);
        point[i] = x[i] - hi;
        let backward = f(&point);
        point[i] = x[i];
        hessian[i][i] = (forward - 2.0 * fx + backward) / (hi * hi);
        for j in i + 1..n {
            let hj = steps[j];
            let mut value = 0.0;
            for (si, sj, sign) in [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)] {
                point[i] = x[i] + si * hi;
                point[j] = x[j] + sj * hj;
                value += sign * f(&point);
            }
            point[i] = x[i];
            point[j] = x[j];
            let value = value / (4.0 * hi * hj);
            hessian[i][j] = value;
            hessian[j][i] = value;
        }
    }
    Ok(hessian)
}

/// The Jacobian `∂fᵢ/∂xⱼ` of a vector-valued function by central differences (`2n + 1`
/// evaluations); row `i` is the gradient of component `i`.
///
/// `step_size` is relative, with default `h = ε^{1/3}`.
pub fn jacobian_2_point<T>(f: &T, x: &[f64], step_size: Option<f64>) -> Result<Vec<Vec<f64>>, ErrorsJSL>
where
    T: Fn(&[f64]) -> Vec<f64>,
{
    let relative = step_size.unwrap_or(optimal_relative_step(2, 1));
    let n = x.len();
    let m = f(x).len();
    let mut jacobian = vec![vec![0.0; n]; m];
    let mut point = x.to_vec();
    for i in 0..n {
        let h = representable_step(x[i], relative);
        point[i] = x[i] + h;
        let forward = f(&point);
        point[i] = x[i] - h;
        let backward = f(&point);
        point[i] = x[i];
        if forward.len() != m || backward.len() != m {
            return Err(ErrorsJSL::IncompatibleArraySizes((forward.len(), m)));
        }
        for j in 0..m {
            jacobian[j][i] = (forward[j] - backward[j]) / (2.0 * h);
        }
    }
    Ok(jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_2_point() {
        let f = |x: &[f64]| (x[0] * x[0] * x[0]) + (x[1] * x[1] * x[1]); // x^3 + y^3
        let result = gradient_2_point(&f, &[2.0, 3.0], Some(1E-4)).unwrap();
        assert!((result[0] - 12.0).abs() < 1E-4);
        assert!((result[1] - 27.0).abs() < 1E-4);
        let result = gradient_2_point(&f, &[2.0, 3.0], None).unwrap();
        assert!((result[0] - 12.0).abs() < 1E-8 && (result[1] - 27.0).abs() < 1E-8);
    }

    #[test]
    fn test_hessian_2_point() {
        let f = |x: &[f64]| (x[0] * x[0] * x[0]) + (x[1] * x[1] * x[1]);
        let result = hessian_2_point(&f, &[2.0, 3.0], Some(1E-4)).unwrap();
        assert!((result[0][0] - 12.0).abs() < 1E-4);
        assert!((result[1][1] - 18.0).abs() < 1E-4);
        assert!((result[0][1]).abs() < 1E-4);
    }

    #[test]
    fn test_jacobian_2_point() {
        let f = |x: &[f64]| vec![x[0] * x[0], x[1] * x[1]];
        let result = jacobian_2_point(&f, &[2.0, 3.0], Some(1E-4)).unwrap();
        assert!((result[0][0] - 4.0).abs() < 1E-4);
        assert!((result[0][1]).abs() < 1E-4);
        assert!((result[1][0]).abs() < 1E-4);
        assert!((result[1][1] - 6.0).abs() < 1E-4);
    }

    #[test]
    fn default_steps_give_accurate_hessians() {
        // f = e^{x + 2y}: Hessian e^{x+2y} [[1, 2], [2, 4]] at (0.3, −0.1).
        let f = |v: &[f64]| (v[0] + 2.0 * v[1]).exp();
        let e = (0.1f64).exp();
        let h = hessian_2_point(&f, &[0.3, -0.1], None).unwrap();
        let want = [[e, 2.0 * e], [2.0 * e, 4.0 * e]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((h[i][j] - want[i][j]).abs() < 1e-6, "({i}, {j}): {}", h[i][j]);
            }
        }
    }
}
