//! Elastic net regression (lasso and ridge combined), by cyclic coordinate descent.

use super::least_squares::column_means;
use crate::common::{Regressor, check_xy};
use lib_jsl_core::{ErrorsJSL, R1D, VR1D, VR2D};

/// Minimises `1/(2n) ‖y - Xw - b‖² + α (ρ ‖w‖₁ + (1-ρ)/2 ‖w‖²)`.
///
/// The L1 part (`ρ`, the `l1_ratio`) drives some weights exactly to zero, selecting features; the L2
/// part stabilises the solution when features are correlated. `l1_ratio = 1` is the lasso and `0`
/// is (a scaled) ridge. Coordinate descent updates one weight at a time with a closed-form
/// soft-threshold step until the largest change falls below `tolerance`.
#[derive(Debug, Clone)]
pub struct ElasticNet {
    alpha: f64,
    l1_ratio: f64,
    max_iterations: usize,
    tolerance: f64,
    weights: R1D,
    intercept: f64,
    iterations: usize,
    fitted: bool,
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma { z - gamma } else if z < -gamma { z + gamma } else { 0.0 }
}

impl ElasticNet {
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        ElasticNet { alpha, l1_ratio, max_iterations: 10_000, tolerance: 1e-10, weights: R1D::zeros(0), intercept: 0.0, iterations: 0, fitted: false }
    }

    /// The lasso: pure L1 penalty.
    pub fn lasso(alpha: f64) -> Self {
        Self::new(alpha, 1.0)
    }

    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    pub fn weights(&self) -> &R1D {
        &self.weights
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Sweeps used by the last fit.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Number of non-zero weights.
    pub fn nonzero(&self) -> usize {
        self.weights.iter().filter(|&&w| w != 0.0).count()
    }
}

impl Regressor for ElasticNet {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL> {
        check_xy(x, y.len())?;
        if self.alpha.is_nan() || self.alpha < 0.0 || !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(ErrorsJSL::InvalidInputRange("Need alpha >= 0 and l1_ratio in [0, 1]."));
        }
        let (n, p) = x.dim();
        let (x_mean, y_mean) = (column_means(x), y.sum() / n as f64);
        let xc: Vec<Vec<f64>> = (0..p).map(|j| (0..n).map(|i| x[[i, j]] - x_mean[j]).collect()).collect(); // column-major
        let col_sq: Vec<f64> = xc.iter().map(|c| c.iter().map(|v| v * v).sum::<f64>() / n as f64).collect();
        let (l1, l2) = (self.alpha * self.l1_ratio, self.alpha * (1.0 - self.l1_ratio));
        let mut w = vec![0.0; p];
        let mut resid: Vec<f64> = y.iter().map(|v| v - y_mean).collect();
        self.iterations = 0;
        for sweep in 1..=self.max_iterations {
            let mut max_change: f64 = 0.0;
            for j in 0..p {
                if col_sq[j] == 0.0 {
                    continue;
                }
                // rho = (1/n) x_j · (resid + x_j w_j): the correlation with the partial residual.
                let rho = xc[j].iter().zip(&resid).map(|(a, r)| a * r).sum::<f64>() / n as f64 + col_sq[j] * w[j];
                let new = soft_threshold(rho, l1) / (col_sq[j] + l2);
                let delta = new - w[j];
                if delta != 0.0 {
                    resid.iter_mut().zip(&xc[j]).for_each(|(r, a)| *r -= a * delta);
                    w[j] = new;
                }
                max_change = max_change.max(delta.abs());
            }
            self.iterations = sweep;
            if max_change < self.tolerance {
                break;
            }
        }
        self.weights = R1D::from(w);
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
    use crate::metrics::r2_score;
    use ndarray::{Array2, arr1, arr2};

    /// Ten features of which only the first three matter.
    fn sparse_problem(seed: u64) -> (Array2<f64>, R1D) {
        let mut rng = MlRng::new(seed);
        let x = Array2::from_shape_fn((200, 10), |_| rng.gaussian(0.0, 1.0));
        let y = R1D::from_iter((0..200).map(|i| 4.0 * x[[i, 0]] - 3.0 * x[[i, 1]] + 2.0 * x[[i, 2]] + 1.0 + rng.gaussian(0.0, 0.2)));
        (x, y)
    }

    #[test]
    fn lasso_selects_the_relevant_features() {
        let (x, y) = sparse_problem(3);
        let mut m = ElasticNet::lasso(0.2);
        m.fit(x.view(), y.view()).unwrap();
        assert!(m.weights().iter().skip(3).all(|&w| w == 0.0), "{:?}", m.weights());
        assert_eq!(m.nonzero(), 3);
        assert!(m.weights()[0] > 3.5 && m.weights()[1] < -2.5 && m.weights()[2] > 1.5);
        assert!(r2_score(y.view(), m.predict(x.view()).unwrap().view()).unwrap() > 0.97);
    }

    #[test]
    fn penalty_path_and_zero_penalty_limit() {
        let (x, y) = sparse_problem(5);
        let counts: Vec<usize> = [0.01, 0.2, 1.0, 5.0, 50.0]
            .iter()
            .map(|&a| {
                let mut m = ElasticNet::lasso(a);
                m.fit(x.view(), y.view()).unwrap();
                m.nonzero()
            })
            .collect();
        assert!(counts.windows(2).all(|p| p[1] <= p[0]) && counts[0] >= 4 && counts[4] == 0, "{counts:?}");
        // With no penalty it is ordinary least squares.
        let mut free = ElasticNet::new(0.0, 0.5);
        free.fit(x.view(), y.view()).unwrap();
        assert!((free.weights()[0] - 4.0).abs() < 0.1 && (free.intercept() - 1.0).abs() < 0.1);
    }

    #[test]
    fn l2_part_spreads_weight_over_correlated_features() {
        // Two identical columns: the lasso picks one arbitrarily, the elastic net splits the weight.
        let n = 100;
        let mut rng = MlRng::new(2);
        let base: Vec<f64> = (0..n).map(|_| rng.gaussian(0.0, 1.0)).collect();
        let x = Array2::from_shape_fn((n, 2), |(i, _)| base[i]);
        let y = R1D::from_iter(base.iter().map(|b| 2.0 * b));
        let mut en = ElasticNet::new(0.05, 0.3);
        en.fit(x.view(), y.view()).unwrap();
        assert!((en.weights()[0] - en.weights()[1]).abs() < 1e-6 && en.weights()[0] > 0.5);
    }

    #[test]
    fn errors() {
        let (x, y) = (arr2(&[[1.0], [2.0]]), arr1(&[1.0, 2.0]));
        assert!(ElasticNet::new(-1.0, 0.5).fit(x.view(), y.view()).is_err());
        assert!(ElasticNet::new(1.0, 1.5).fit(x.view(), y.view()).is_err());
        assert!(ElasticNet::lasso(0.1).predict(x.view()).is_err());
        let mut m = ElasticNet::lasso(0.0).with_limits(5, 1e-12);
        m.fit(x.view(), y.view()).unwrap();
        assert!(m.iterations() <= 5);
        // A constant column is ignored rather than dividing by zero.
        let mut c = ElasticNet::lasso(0.1);
        c.fit(arr2(&[[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]).view(), arr1(&[2.0, 4.0, 6.0]).view()).unwrap();
        assert_eq!(c.weights()[1], 0.0);
    }
}
