//! Principal component analysis.

use crate::common::{Transformer, check_features};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR2D};
use lib_jsl_linalg::{Eigh, UPLO};

/// PCA: finds orthogonal directions (principal components) of greatest variance and projects the
/// data onto the first `components` of them. Computed from the eigendecomposition of the
/// covariance matrix of the centred data. The leading components capture most of the structure in
/// correlated data, so a few of them often summarise many features.
#[derive(Debug, Clone)]
pub struct Pca {
    components: usize,
    mean: R1D,
    /// Rows are the principal axes (unit vectors), in order of decreasing variance.
    axes: R2D,
    variances: Vec<f64>,
    total_variance: f64,
}

impl Pca {
    /// Keeps `components` principal components.
    pub fn new(components: usize) -> Self {
        Pca { components, mean: R1D::zeros(0), axes: R2D::zeros((0, 0)), variances: Vec::new(), total_variance: 0.0 }
    }

    /// The principal axes, one per row.
    pub fn components(&self) -> &R2D {
        &self.axes
    }

    /// The variance along each kept component.
    pub fn explained_variance(&self) -> &[f64] {
        &self.variances
    }

    /// The fraction of the total variance each kept component explains.
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        self.variances.iter().map(|v| if self.total_variance > 0.0 { v / self.total_variance } else { 0.0 }).collect()
    }

    pub fn mean(&self) -> &R1D {
        &self.mean
    }

    /// Maps component scores back to the original feature space (a lossy reconstruction when
    /// fewer components than features were kept).
    pub fn inverse_transform(&self, z: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.axes.nrows() == 0 {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        if z.ncols() != self.axes.nrows() {
            return Err(ErrorsJSL::IncompatibleArraySizes((z.ncols(), self.axes.nrows())));
        }
        Ok(z.dot(&self.axes) + &self.mean)
    }
}

impl Transformer for Pca {
    fn fit(&mut self, x: VR2D) -> Result<(), ErrorsJSL> {
        check_features(x)?;
        let (n, p) = x.dim();
        if n < 2 || self.components == 0 || self.components > p {
            return Err(ErrorsJSL::InvalidInputRange("Need at least two samples and 1 <= components <= features."));
        }
        self.mean = x.mean_axis(ndarray::Axis(0)).expect("non-empty");
        let centred = &x - &self.mean;
        let cov = centred.t().dot(&centred) / (n as f64 - 1.0);
        let (vals, vecs) = cov.eigh(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("The covariance eigendecomposition failed."))?;
        // eigh returns ascending eigenvalues; take the largest.
        let order: Vec<usize> = (0..p).rev().take(self.components).collect();
        self.variances = order.iter().map(|&i| vals[i].max(0.0)).collect();
        self.total_variance = vals.iter().map(|v| v.max(0.0)).sum();
        self.axes = R2D::from_shape_fn((self.components, p), |(c, j)| vecs[[j, order[c]]]);
        // Sign convention: make the largest-magnitude loading of each axis positive, so results
        // are reproducible.
        for mut axis in self.axes.rows_mut() {
            let big = axis.iter().copied().fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
            if big < 0.0 {
                axis.mapv_inplace(|v| -v);
            }
        }
        Ok(())
    }

    fn transform(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.axes.nrows() == 0 {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.mean.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.mean.len())));
        }
        Ok((&x - &self.mean).dot(&self.axes.t()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::MlRng;
    use ndarray::{Array2, arr2};

    /// Data stretched along (1, 1)/√2 with standard deviation 3, and 0.3 across it.
    fn stretched(n: usize, seed: u64) -> Array2<f64> {
        let mut rng = MlRng::new(seed);
        let pts: Vec<(f64, f64)> = (0..n).map(|_| (rng.gaussian(0.0, 3.0), rng.gaussian(0.0, 0.3))).collect();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { s * (pts[i].0 - pts[i].1) + 5.0 } else { s * (pts[i].0 + pts[i].1) - 2.0 })
    }

    #[test]
    fn finds_the_dominant_axis_and_variances() {
        let x = stretched(2000, 1);
        let mut p = Pca::new(2);
        p.fit(x.view()).unwrap();
        let axis = p.components().row(0);
        assert!((axis[0].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.02 && axis[0] * axis[1] > 0.0);
        assert!((p.explained_variance()[0] - 9.0).abs() < 0.8 && (p.explained_variance()[1] - 0.09).abs() < 0.02);
        let ratio = p.explained_variance_ratio();
        assert!((ratio.iter().sum::<f64>() - 1.0).abs() < 1e-12 && ratio[0] > 0.98);
        assert!((p.mean()[0] - 5.0).abs() < 0.2 && (p.mean()[1] + 2.0).abs() < 0.2);
    }

    #[test]
    fn projection_is_decorrelated_and_reconstruction_is_exact_with_all_components() {
        let x = stretched(500, 2);
        let mut p = Pca::new(2);
        let z = p.fit_transform(x.view()).unwrap();
        let cz = z.t().dot(&z) / 499.0;
        assert!(cz[[0, 1]].abs() < 1e-9); // scores are uncorrelated
        let back = p.inverse_transform(z.view()).unwrap();
        assert!(back.iter().zip(x.iter()).all(|(a, b)| (a - b).abs() < 1e-9));
        // One component: lossy but close, since almost all variance lies on one axis.
        let mut one = Pca::new(1);
        let z1 = one.fit_transform(x.view()).unwrap();
        assert_eq!(z1.dim(), (500, 1));
        let approx = one.inverse_transform(z1.view()).unwrap();
        let err = approx.iter().zip(x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 500.0;
        assert!(err < 0.3, "{err}");
        // Axes are orthonormal.
        let g = p.components().dot(&p.components().t());
        assert!((g[[0, 0]] - 1.0).abs() < 1e-12 && g[[0, 1]].abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]);
        assert!(Pca::new(0).fit(x.view()).is_err() && Pca::new(3).fit(x.view()).is_err());
        assert!(Pca::new(1).fit(arr2(&[[1.0, 2.0]]).view()).is_err());
        assert!(Pca::new(1).transform(x.view()).is_err());
        assert!(Pca::new(1).inverse_transform(x.view()).is_err());
        let mut p = Pca::new(1);
        p.fit(x.view()).unwrap();
        assert!(p.transform(arr2(&[[1.0, 2.0, 3.0]]).view()).is_err());
        assert!(p.inverse_transform(arr2(&[[1.0, 2.0]]).view()).is_err());
    }
}
