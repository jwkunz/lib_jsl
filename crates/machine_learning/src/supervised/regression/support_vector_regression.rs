//! Support vector regression (ε-SVR) with kernels.

use crate::common::{Regressor, check_xy};
use crate::kernels::Kernel;
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};

/// ε-insensitive support vector regression. Errors smaller than `epsilon` cost nothing, larger
/// ones cost linearly, and `c` trades flatness against fit. Only samples on or outside the ε-tube
/// become support vectors, so the model is sparse.
///
/// The dual problem `min ½ βᵀ(K+1)β - yᵀβ + ε‖β‖₁` with box constraints `|β_i| <= C` (the `+1`
/// folds the bias into the kernel) is solved by cyclic coordinate descent, each coordinate having a
/// closed-form soft-threshold-and-clip update. Predictions are `f(x) = Σ β_i (k(x_i, x) + 1)`.
#[derive(Debug, Clone)]
pub struct SupportVectorRegression {
    kernel: Kernel,
    c: f64,
    epsilon: f64,
    max_iterations: usize,
    tolerance: f64,
    support: R2D,
    beta: R1D,
    fitted: bool,
}

impl SupportVectorRegression {
    pub fn new(kernel: Kernel, c: f64, epsilon: f64) -> Self {
        SupportVectorRegression { kernel, c, epsilon, max_iterations: 5000, tolerance: 1e-8, support: R2D::zeros((0, 0)), beta: R1D::zeros(0), fitted: false }
    }

    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    /// The number of support vectors (non-zero dual coefficients).
    pub fn support_vector_count(&self) -> usize {
        self.beta.len()
    }
}

impl Regressor for SupportVectorRegression {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL> {
        check_xy(x, y.len())?;
        if self.c.is_nan() || self.c <= 0.0 || self.epsilon.is_nan() || self.epsilon < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need C > 0 and epsilon >= 0."));
        }
        let n = x.nrows();
        let k = self.kernel.gram(x) + 1.0; // fold the bias term into the kernel
        let mut beta = vec![0.0; n];
        let mut f = vec![0.0; n]; // f[j] = Σ_i k[i][j] β_i
        for _ in 0..self.max_iterations {
            let mut max_change: f64 = 0.0;
            for i in 0..n {
                let kii = k[[i, i]];
                let z = y[i] - f[i] + kii * beta[i];
                let shrunk = if z > self.epsilon { z - self.epsilon } else if z < -self.epsilon { z + self.epsilon } else { 0.0 };
                let new = (shrunk / kii).clamp(-self.c, self.c);
                let delta = new - beta[i];
                if delta != 0.0 {
                    for j in 0..n {
                        f[j] += k[[i, j]] * delta;
                    }
                    beta[i] = new;
                    max_change = max_change.max(delta.abs());
                }
            }
            if max_change < self.tolerance {
                break;
            }
        }
        let keep: Vec<usize> = (0..n).filter(|&i| beta[i] != 0.0).collect();
        self.support = R2D::from_shape_fn((keep.len(), x.ncols()), |(a, j)| x[[keep[a], j]]);
        self.beta = R1D::from_iter(keep.iter().map(|&i| beta[i]));
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<R1D, ErrorsJSL> {
        if !self.fitted {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        if self.beta.is_empty() {
            return Ok(R1D::zeros(x.nrows()));
        }
        if x.ncols() != self.support.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.support.ncols())));
        }
        Ok((self.kernel.cross(x, self.support.view()) + 1.0).dot(&self.beta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_sine;
    use crate::metrics::{mean_absolute_error, r2_score};
    use ndarray::{arr1, arr2};

    #[test]
    fn linear_kernel_fits_a_line_within_the_tube() {
        let x = R2D::from_shape_fn((40, 1), |(i, _)| i as f64 / 10.0);
        let y = x.column(0).mapv(|v| 2.0 * v + 1.0);
        let mut m = SupportVectorRegression::new(Kernel::Linear, 100.0, 0.05);
        m.fit(x.view(), y.view()).unwrap();
        let p = m.predict(x.view()).unwrap();
        assert!(mean_absolute_error(y.view(), p.view()).unwrap() < 0.06);
        assert!(p.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 0.12));
        assert!(m.support_vector_count() < 40); // sparse: points inside the tube drop out
    }

    #[test]
    fn rbf_kernel_learns_a_sine_wave_and_epsilon_controls_sparsity() {
        let (x, y) = make_sine(150, 0.05, 6);
        let fit = |eps: f64| {
            let mut m = SupportVectorRegression::new(Kernel::Rbf { gamma: 1.0 }, 10.0, eps).with_limits(1500, 1e-6);
            m.fit(x.view(), y.view()).unwrap();
            m
        };
        let tight = fit(0.01);
        let loose = fit(0.3);
        assert!(r2_score(y.view(), tight.predict(x.view()).unwrap().view()).unwrap() > 0.95);
        assert!(loose.support_vector_count() < tight.support_vector_count(), "{} vs {}", loose.support_vector_count(), tight.support_vector_count());
        // Generalises to unseen points.
        let probe = arr2(&[[0.5], [1.5], [3.0], [5.0]]);
        let want = probe.column(0).mapv(f64::sin);
        assert!(mean_absolute_error(want.view(), tight.predict(probe.view()).unwrap().view()).unwrap() < 0.1);
    }

    #[test]
    fn small_c_underfits_and_errors() {
        let x = R2D::from_shape_fn((30, 1), |(i, _)| i as f64);
        let y = x.column(0).mapv(|v| 5.0 * v);
        let mut weak = SupportVectorRegression::new(Kernel::Linear, 0.001, 0.1);
        weak.fit(x.view(), y.view()).unwrap();
        assert!(r2_score(y.view(), weak.predict(x.view()).unwrap().view()).unwrap() < 0.5);
        assert!(SupportVectorRegression::new(Kernel::Linear, 0.0, 0.1).fit(x.view(), y.view()).is_err());
        assert!(SupportVectorRegression::new(Kernel::Linear, 1.0, -0.1).fit(x.view(), y.view()).is_err());
        assert!(SupportVectorRegression::new(Kernel::Linear, 1.0, 0.1).predict(x.view()).is_err());
        let mut m = SupportVectorRegression::new(Kernel::Linear, 1.0, 0.1);
        m.fit(arr2(&[[1.0], [2.0]]).view(), arr1(&[1.0, 2.0]).view()).unwrap();
        assert!(m.predict(arr2(&[[1.0, 2.0]]).view()).is_err());
    }
}
