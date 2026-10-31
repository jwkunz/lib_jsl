//! Ridge (L2-regularised) regression.

use super::least_squares::column_means;
use crate::common::{Regressor, check_xy};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};
use ndarray::Array2;
use ndarray_linalg::Solve;

/// Minimises `‖y - Xw - b‖² + α ‖w‖²`. The penalty shrinks the weights, trading a little bias for a
/// large drop in variance when features are correlated or samples are few. The intercept is never
/// penalised (data are centred internally). Solved in closed form through the regularised normal
/// equations `(XᵀX + αI) w = Xᵀy`, which are always well posed for `α > 0`.
#[derive(Debug, Clone)]
pub struct Ridge {
    alpha: f64,
    weights: R1D,
    intercept: f64,
    fitted: bool,
}

impl Ridge {
    /// `alpha >= 0` is the penalty strength (0 is ordinary least squares).
    pub fn new(alpha: f64) -> Self {
        Ridge { alpha, weights: R1D::zeros(0), intercept: 0.0, fitted: false }
    }

    pub fn weights(&self) -> &R1D {
        &self.weights
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Regressor for Ridge {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL> {
        check_xy(x, y.len())?;
        if self.alpha.is_nan() || self.alpha < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The penalty alpha must be non-negative."));
        }
        let (x_mean, y_mean) = (column_means(x), y.sum() / y.len() as f64);
        let xc = R2D::from_shape_fn(x.dim(), |(i, j)| x[[i, j]] - x_mean[j]);
        let yc = y.mapv(|v| v - y_mean);
        let mut gram: Array2<f64> = xc.t().dot(&xc);
        for j in 0..gram.nrows() {
            gram[[j, j]] += self.alpha;
        }
        self.weights = gram.solve(&xc.t().dot(&yc)).map_err(|_| ErrorsJSL::RuntimeError("Singular normal equations (try a larger alpha)."))?;
        self.intercept = y_mean - x_mean.dot(&self.weights);
        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<R1D, ErrorsJSL> {
        if !self.fitted {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        if x.ncols() != self.weights.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.weights.len())));
        }
        Ok(x.dot(&self.weights) + self.intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::MlRng;
    use crate::datasets::make_regression;
    use crate::metrics::mean_squared_error;
    use crate::preprocessing::{select, select_rows, train_test_split};
    use ndarray::{arr1, arr2};

    #[test]
    fn zero_penalty_matches_ols_and_penalty_shrinks_weights() {
        let (x, y, w, b) = make_regression(200, 3, 0.1, 5);
        let mut ols_like = Ridge::new(0.0);
        ols_like.fit(x.view(), y.view()).unwrap();
        for (g, t) in ols_like.weights().iter().zip(w.iter()) {
            assert!((g - t).abs() < 0.05);
        }
        assert!((ols_like.intercept() - b).abs() < 0.05);
        let norms: Vec<f64> = [0.0, 10.0, 1000.0, 1e6]
            .iter()
            .map(|&a| {
                let mut r = Ridge::new(a);
                r.fit(x.view(), y.view()).unwrap();
                r.weights().dot(r.weights()).sqrt()
            })
            .collect();
        assert!(norms.windows(2).all(|p| p[1] < p[0]), "{norms:?}");
        assert!(norms[3] < 1e-3);
    }

    #[test]
    fn regularisation_helps_with_correlated_features_and_few_samples() {
        // Ten nearly-duplicate features, only 15 samples: OLS overfits, ridge generalises.
        let mut rng = MlRng::new(8);
        let make = |n: usize, rng: &mut MlRng| {
            let base: Vec<f64> = (0..n).map(|_| rng.gaussian(0.0, 1.0)).collect();
            let x = R2D::from_shape_fn((n, 10), |(i, _)| base[i]);
            let x = x.mapv(|v| v) + R2D::from_shape_fn((n, 10), |_| rng.gaussian(0.0, 0.05));
            let y = R1D::from_iter(base.iter().map(|b| 3.0 * b + rng.gaussian(0.0, 0.3)));
            (x, y)
        };
        let (xtr, ytr) = make(15, &mut rng);
        let (xte, yte) = make(500, &mut rng);
        let test_mse = |alpha: f64| {
            let mut m = Ridge::new(alpha);
            m.fit(xtr.view(), ytr.view()).unwrap();
            mean_squared_error(yte.view(), m.predict(xte.view()).unwrap().view()).unwrap()
        };
        assert!(test_mse(1.0) < test_mse(0.0), "{} vs {}", test_mse(1.0), test_mse(0.0));
        let _ = (select, select_rows, train_test_split);
    }

    #[test]
    fn fits_when_features_exceed_samples_and_errors() {
        let x = arr2(&[[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]);
        let y = arr1(&[1.0, 2.0]);
        let mut r = Ridge::new(0.1);
        r.fit(x.view(), y.view()).unwrap(); // underdetermined: only the penalty makes it solvable
        assert!(r.predict(x.view()).unwrap().iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 0.2));
        assert!(Ridge::new(-1.0).fit(x.view(), y.view()).is_err());
        assert!(Ridge::new(1.0).predict(x.view()).is_err());
        assert!(r.predict(arr2(&[[1.0]]).view()).is_err());
    }
}
