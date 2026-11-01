//! Ordinary least squares linear regression.

use crate::common::{Regressor, check_xy};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};
use ndarray::{Array1, Array2, Axis};
use lib_jsl_linalg::LeastSquaresSvd;

/// `y ≈ X w + b`, fitted by minimum-norm least squares through the SVD, so rank-deficient designs
/// (collinear features, more features than samples) are handled gracefully.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    fit_intercept: bool,
    weights: R1D,
    intercept: f64,
    fitted: bool,
}

impl LinearRegression {
    /// A model that fits an intercept.
    pub fn new() -> Self {
        LinearRegression { fit_intercept: true, weights: R1D::zeros(0), intercept: 0.0, fitted: false }
    }

    /// Sets whether an intercept is fitted (`false` forces the plane through the origin).
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn weights(&self) -> &R1D {
        &self.weights
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Appends a column of ones.
pub(crate) fn with_ones(x: VR2D) -> R2D {
    let mut out = Array2::ones((x.nrows(), x.ncols() + 1));
    out.slice_mut(ndarray::s![.., ..x.ncols()]).assign(&x);
    out
}

impl Regressor for LinearRegression {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL> {
        check_xy(x, y.len())?;
        let design = if self.fit_intercept { with_ones(x) } else { x.to_owned() };
        let sol = design.least_squares(&y.to_owned()).map_err(|_| ErrorsJSL::RuntimeError("The least-squares solve failed."))?;
        let coef: Array1<f64> = sol.solution;
        if self.fit_intercept {
            self.intercept = coef[coef.len() - 1];
            self.weights = coef.slice(ndarray::s![..coef.len() - 1]).to_owned();
        } else {
            self.intercept = 0.0;
            self.weights = coef;
        }
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

/// Column means of a matrix.
pub(crate) fn column_means(x: VR2D) -> R1D {
    x.mean_axis(Axis(0)).unwrap_or_else(|| R1D::zeros(x.ncols()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_regression, make_sine};
    use crate::metrics::r2_score;
    use crate::preprocessing::polynomial_features;
    use ndarray::{arr1, arr2};

    #[test]
    fn recovers_the_true_model() {
        let (x, y, w, b) = make_regression(300, 4, 0.01, 2);
        let mut m = LinearRegression::new();
        m.fit(x.view(), y.view()).unwrap();
        for (got, want) in m.weights().iter().zip(w.iter()) {
            assert!((got - want).abs() < 0.01);
        }
        assert!((m.intercept() - b).abs() < 0.01);
        assert!(r2_score(y.view(), m.predict(x.view()).unwrap().view()).unwrap() > 0.999);
    }

    #[test]
    fn intercept_options_collinearity_and_polynomials() {
        let x = arr2(&[[1.0], [2.0], [3.0]]);
        let y = arr1(&[3.0, 5.0, 7.0]);
        let mut with = LinearRegression::new();
        with.fit(x.view(), y.view()).unwrap();
        assert!((with.weights()[0] - 2.0).abs() < 1e-10 && (with.intercept() - 1.0).abs() < 1e-10);
        let mut without = LinearRegression::new().with_intercept(false);
        without.fit(x.view(), y.view()).unwrap();
        assert_eq!(without.intercept(), 0.0);
        // Perfectly collinear columns still give a fit (minimum-norm weights).
        let dup = arr2(&[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
        let mut c = LinearRegression::new();
        c.fit(dup.view(), y.view()).unwrap();
        assert!(c.predict(dup.view()).unwrap().iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 1e-9));
        // A cubic polynomial fit to a sine wave beats a straight line.
        let (sx, sy) = make_sine(200, 0.05, 3);
        let mut line = LinearRegression::new();
        line.fit(sx.view(), sy.view()).unwrap();
        let px = polynomial_features(sx.view(), 5).unwrap();
        let mut poly = LinearRegression::new();
        poly.fit(px.view(), sy.view()).unwrap();
        let (r_line, r_poly) = (r2_score(sy.view(), line.predict(sx.view()).unwrap().view()).unwrap(), r2_score(sy.view(), poly.predict(px.view()).unwrap().view()).unwrap());
        assert!(r_poly > 0.95 && r_line < 0.7, "{r_line} {r_poly}");
    }

    #[test]
    fn errors() {
        let mut m = LinearRegression::new();
        assert!(m.predict(arr2(&[[1.0]]).view()).is_err());
        assert!(m.fit(arr2(&[[1.0], [2.0]]).view(), arr1(&[1.0]).view()).is_err());
        assert!(m.fit(arr2(&[[f64::NAN]]).view(), arr1(&[1.0]).view()).is_err());
        m.fit(arr2(&[[1.0], [2.0]]).view(), arr1(&[1.0, 2.0]).view()).unwrap();
        assert!(m.predict(arr2(&[[1.0, 2.0]]).view()).is_err());
    }
}
