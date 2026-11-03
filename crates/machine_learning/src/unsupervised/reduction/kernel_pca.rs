//! Kernel principal component analysis.
//!
//! See the book entry [Principal component analysis and kernel PCA](https://jwkunz.github.io/lib_jsl/book/unsupervised/pca.html#kernel-pca).

use crate::common::{Transformer, check_features};
use crate::kernels::Kernel;
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR2D};
use lib_jsl_linalg::{Eigh, UPLO};

/// PCA in the feature space of a kernel: eigendecomposes the centred Gram matrix, so it finds
/// nonlinear structure (concentric rings become separable along the first component with an RBF
/// kernel) without ever forming the feature map. Component scores of training points are
/// `√λ_i v_i`; new points are projected through the cross-kernel to the training set.
#[derive(Debug, Clone)]
pub struct KernelPca {
    kernel: Kernel,
    components: usize,
    train: R2D,
    /// Eigenvectors of the centred Gram matrix scaled by `1/√λ`, one column per component.
    alphas: R2D,
    eigenvalues: Vec<f64>,
    row_means: R1D,
    total_mean: f64,
}

impl KernelPca {
    pub fn new(kernel: Kernel, components: usize) -> Self {
        KernelPca { kernel, components, train: R2D::zeros((0, 0)), alphas: R2D::zeros((0, 0)), eigenvalues: Vec::new(), row_means: R1D::zeros(0), total_mean: 0.0 }
    }

    /// The kept eigenvalues of the centred Gram matrix, largest first.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
}

impl Transformer for KernelPca {
    fn fit(&mut self, x: VR2D) -> Result<(), ErrorsJSL> {
        check_features(x)?;
        let n = x.nrows();
        if self.components == 0 || self.components > n {
            return Err(ErrorsJSL::InvalidInputRange("Need 1 <= components <= samples."));
        }
        // ANCHOR: kpca_fit
        let k = self.kernel.gram(x);
        self.row_means = k.mean_axis(ndarray::Axis(1)).expect("non-empty");
        self.total_mean = self.row_means.sum() / n as f64;
        // Double centring: Kc = K - 1K - K1 + 1K1.
        let kc = R2D::from_shape_fn((n, n), |(i, j)| k[[i, j]] - self.row_means[i] - self.row_means[j] + self.total_mean);
        let (vals, vecs) = kc.eigh(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("The Gram matrix eigendecomposition failed."))?;
        let order: Vec<usize> = (0..n).rev().take(self.components).collect();
        self.eigenvalues = order.iter().map(|&i| vals[i].max(0.0)).collect();
        self.alphas = R2D::from_shape_fn((n, self.components), |(i, c)| {
            let lambda = vals[order[c]];
            if lambda > 1e-12 { vecs[[i, order[c]]] / lambda.sqrt() } else { 0.0 }
        });
        // ANCHOR_END: kpca_fit
        // Sign convention: first non-negligible loading positive, for reproducibility.
        for mut col in self.alphas.columns_mut() {
            if let Some(&first) = col.iter().find(|v| v.abs() > 1e-12)
                && first < 0.0
            {
                col.mapv_inplace(|v| -v);
            }
        }
        self.train = x.to_owned();
        Ok(())
    }

    fn transform(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.alphas.nrows() == 0 {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.train.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.train.ncols())));
        }
        let k = self.kernel.cross(x, self.train.view());
        let new_means = k.mean_axis(ndarray::Axis(1)).expect("non-empty");
        let kc = R2D::from_shape_fn(k.dim(), |(i, j)| k[[i, j]] - new_means[i] - self.row_means[j] + self.total_mean);
        Ok(kc.dot(&self.alphas))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_circles};
    use crate::unsupervised::reduction::principal_component_analysis::Pca;

    /// The best accuracy of a one-dimensional threshold classifier on the values.
    fn threshold_accuracy(values: &[f64], labels: &[usize]) -> f64 {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let n = values.len();
        let ones = labels.iter().filter(|&&l| l == 1).count();
        let (mut best, mut left_ones) = (0usize, 0usize);
        for (k, &i) in order.iter().enumerate() {
            left_ones += usize::from(labels[i] == 1);
            let left = k + 1;
            let correct = (left - left_ones) + (ones - left_ones); // left predicted 0, right predicted 1
            best = best.max(correct.max(n - correct));
        }
        best as f64 / n as f64
    }

    #[test]
    fn rbf_kernel_pca_unfolds_concentric_circles_that_linear_pca_cannot() {
        let (x, l) = make_circles(100, 0.02, 0.4, 1);
        // The two leading components of a ring are a degenerate pair of angular modes; the third is
        // the radial one, which separates inner from outer ring with a single threshold.
        let mut kp = KernelPca::new(Kernel::Rbf { gamma: 2.0 }, 3);
        let z = kp.fit_transform(x.view()).unwrap();
        assert!(threshold_accuracy(&z.column(2).to_vec(), &l) > 0.98);
        assert!((0..2).all(|c| threshold_accuracy(&z.column(c).to_vec(), &l) < 0.7));
        // Linear PCA cannot separate them along any axis.
        let mut lin = Pca::new(2);
        let zl = lin.fit_transform(x.view()).unwrap();
        assert!((0..2).all(|c| threshold_accuracy(&zl.column(c).to_vec(), &l) < 0.7));
        assert!(kp.eigenvalues().windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn linear_kernel_reproduces_pca_scores_and_new_points_project_consistently() {
        let (x, _) = make_blobs(60, &[vec![0.0, 0.0, 0.0], vec![4.0, 2.0, 1.0]], 1.0, 2);
        let mut kp = KernelPca::new(Kernel::Linear, 2);
        let z = kp.fit_transform(x.view()).unwrap();
        let mut p = Pca::new(2);
        let zp = p.fit_transform(x.view()).unwrap();
        // Identical scores up to sign: both equal sqrt(λ) v, with λ an eigenvalue of the centred Gram
        // matrix (n - 1 times the covariance eigenvalue) and v its unit eigenvector.
        for c in 0..2 {
            let sign = z.column(c).dot(&zp.column(c)).signum();
            assert!((0..x.nrows()).all(|i| (z[[i, c]] * sign - zp[[i, c]]).abs() < 1e-8), "component {c}");
        }
        // Transforming the training data again reproduces the fit-time scores.
        let again = kp.transform(x.view()).unwrap();
        assert!(again.iter().zip(z.iter()).all(|(a, b)| (a - b).abs() < 1e-8));
    }

    #[test]
    fn errors() {
        let (x, _) = make_blobs(5, &[vec![0.0, 0.0]], 1.0, 1);
        assert!(KernelPca::new(Kernel::Linear, 0).fit(x.view()).is_err());
        assert!(KernelPca::new(Kernel::Linear, 6).fit(x.view()).is_err());
        assert!(KernelPca::new(Kernel::Linear, 1).transform(x.view()).is_err());
        let mut k = KernelPca::new(Kernel::Linear, 1);
        k.fit(x.view()).unwrap();
        assert!(k.transform(ndarray::arr2(&[[1.0]]).view()).is_err());
    }
}
