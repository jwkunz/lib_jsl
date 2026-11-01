//! Weighted and generalised least squares.

use super::least_squares::with_ones;
use crate::common::{Regressor, check_xy};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};
use ndarray::Array1;
use lib_jsl_linalg::{Cholesky, LeastSquaresSvd, SolveTriangular, UPLO};

/// Least squares with a known error structure: minimise `(y - Xw)ᵀ Σ⁻¹ (y - Xw)` where `Σ` is the
/// covariance of the errors (generalised least squares), or the special case of independent errors
/// with per-sample weights (weighted least squares, `Σ = diag(1/weight)`). This is the best linear
/// unbiased estimator when `Σ` is right: noisy or correlated samples count for less.
///
/// Implemented by whitening: factor `Σ = L Lᵀ` and solve the ordinary problem for `L⁻¹ y ~ L⁻¹ X`.
#[derive(Debug, Clone)]
pub struct GeneralLeastSquares {
    covariance: Option<R2D>,
    weights_per_sample: Option<R1D>,
    fit_intercept: bool,
    coef: R1D,
    intercept: f64,
    fitted: bool,
}

impl GeneralLeastSquares {
    /// Errors with the full covariance matrix `Σ` (symmetric positive definite, `n × n`).
    pub fn with_covariance(covariance: R2D) -> Self {
        GeneralLeastSquares { covariance: Some(covariance), weights_per_sample: None, fit_intercept: true, coef: R1D::zeros(0), intercept: 0.0, fitted: false }
    }

    /// Independent errors with the given positive per-sample weights (inverse variances).
    pub fn with_sample_weights(weights: R1D) -> Self {
        GeneralLeastSquares { covariance: None, weights_per_sample: Some(weights), fit_intercept: true, coef: R1D::zeros(0), intercept: 0.0, fitted: false }
    }

    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn weights(&self) -> &R1D {
        &self.coef
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Regressor for GeneralLeastSquares {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL> {
        check_xy(x, y.len())?;
        let n = x.nrows();
        let design = if self.fit_intercept { with_ones(x) } else { x.to_owned() };
        let (wd, wy): (R2D, R1D) = if let Some(cov) = &self.covariance {
            if cov.dim() != (n, n) {
                return Err(ErrorsJSL::IncompatibleArraySizes((cov.nrows(), n)));
            }
            let l = cov.cholesky(UPLO::Lower).map_err(|_| ErrorsJSL::InvalidInputRange("The covariance must be symmetric positive definite."))?;
            let wd = l.solve_triangular(UPLO::Lower, lib_jsl_linalg::Diag::NonUnit, &design).map_err(|_| ErrorsJSL::RuntimeError("Whitening failed."))?;
            let wy = l.solve_triangular(UPLO::Lower, lib_jsl_linalg::Diag::NonUnit, &y.to_owned()).map_err(|_| ErrorsJSL::RuntimeError("Whitening failed."))?;
            (wd, wy)
        } else {
            let w = self.weights_per_sample.as_ref().ok_or(ErrorsJSL::Misconfiguration("No error model was given."))?;
            if w.len() != n {
                return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), n)));
            }
            if w.iter().any(|&v| !v.is_finite() || v <= 0.0) {
                return Err(ErrorsJSL::InvalidInputRange("Sample weights must be positive and finite."));
            }
            let s: Array1<f64> = w.mapv(f64::sqrt);
            (R2D::from_shape_fn(design.dim(), |(i, j)| design[[i, j]] * s[i]), R1D::from_iter(y.iter().zip(&s).map(|(a, b)| a * b)))
        };
        let sol = wd.least_squares(&wy).map_err(|_| ErrorsJSL::RuntimeError("The least-squares solve failed."))?.solution;
        if self.fit_intercept {
            self.intercept = sol[sol.len() - 1];
            self.coef = sol.slice(ndarray::s![..sol.len() - 1]).to_owned();
        } else {
            self.intercept = 0.0;
            self.coef = sol;
        }
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<R1D, ErrorsJSL> {
        if !self.fitted {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        if x.ncols() != self.coef.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.coef.len())));
        }
        Ok(x.dot(&self.coef) + self.intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::MlRng;
    use ndarray::{Array2, arr1, arr2};

    #[test]
    fn heteroscedastic_weights_beat_plain_least_squares() {
        // y = 2x + 1 with a few wildly noisy points that get tiny weights.
        let mut rng = MlRng::new(4);
        let n = 60;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / 10.0);
        let noisy: Vec<bool> = (0..n).map(|i| i % 6 == 0).collect();
        let y = R1D::from_iter((0..n).map(|i| 2.0 * x[[i, 0]] + 1.0 + rng.gaussian(0.0, if noisy[i] { 8.0 } else { 0.05 })));
        let w = R1D::from_iter(noisy.iter().map(|&b| if b { 1.0 / 64.0 } else { 1.0 / 0.0025 }));
        let mut wls = GeneralLeastSquares::with_sample_weights(w);
        wls.fit(x.view(), y.view()).unwrap();
        let mut ols = super::super::least_squares::LinearRegression::new();
        ols.fit(x.view(), y.view()).unwrap();
        let err = |slope: f64, b: f64| (slope - 2.0).abs() + (b - 1.0).abs();
        assert!(err(wls.weights()[0], wls.intercept()) < 0.05, "{:?} {}", wls.weights(), wls.intercept());
        assert!(err(wls.weights()[0], wls.intercept()) < err(ols.weights()[0], ols.intercept()));
    }

    #[test]
    fn full_covariance_matches_weights_when_diagonal() {
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let y = arr1(&[1.1, 2.3, 2.8, 4.4]);
        let var = [1.0, 4.0, 0.25, 2.0];
        let mut gls = GeneralLeastSquares::with_covariance(Array2::from_diag(&arr1(&var)));
        gls.fit(x.view(), y.view()).unwrap();
        let mut wls = GeneralLeastSquares::with_sample_weights(arr1(&var).mapv(|v| 1.0 / v));
        wls.fit(x.view(), y.view()).unwrap();
        assert!((gls.weights()[0] - wls.weights()[0]).abs() < 1e-10 && (gls.intercept() - wls.intercept()).abs() < 1e-10);
        assert!(gls.predict(x.view()).unwrap().len() == 4);
    }

    #[test]
    fn identity_covariance_is_ordinary_least_squares_and_errors() {
        let x = arr2(&[[1.0], [2.0], [3.0]]);
        let y = arr1(&[3.0, 5.0, 7.0]);
        let mut g = GeneralLeastSquares::with_covariance(Array2::eye(3));
        g.fit(x.view(), y.view()).unwrap();
        assert!((g.weights()[0] - 2.0).abs() < 1e-10 && (g.intercept() - 1.0).abs() < 1e-10);
        let mut bad_shape = GeneralLeastSquares::with_covariance(Array2::eye(2));
        assert!(bad_shape.fit(x.view(), y.view()).is_err());
        let mut not_pd = GeneralLeastSquares::with_covariance(arr2(&[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
        assert!(not_pd.fit(x.view(), y.view()).is_err());
        let mut bad_w = GeneralLeastSquares::with_sample_weights(arr1(&[1.0, 0.0, 1.0]));
        assert!(bad_w.fit(x.view(), y.view()).is_err());
        assert!(GeneralLeastSquares::with_sample_weights(arr1(&[1.0])).predict(x.view()).is_err());
    }
}
