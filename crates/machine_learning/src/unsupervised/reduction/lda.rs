//! Linear discriminant analysis.
//!
//! See the book entry [Linear discriminant analysis](https://jwkunz.github.io/lib_jsl/book/unsupervised/lda.html).

use crate::common::{Classifier, argmax, check_features, check_xy, class_count};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR2D};
use ndarray::Array2;
use lib_jsl_linalg::{Cholesky, Diag, Eigh, SolveTriangular, UPLO};

/// Fisher's linear discriminant analysis: a *supervised* projection that maximises the ratio of
/// between-class to within-class scatter, so classes are pulled apart while each is kept compact.
/// At most `classes - 1` discriminant directions exist. The same model doubles as a classifier
/// (nearest class mean in the projected space, adjusted by log-priors), which is the Gaussian
/// classifier with a shared covariance.
///
/// The generalised eigenproblem `S_b w = λ S_w w` is solved by whitening with the Cholesky factor
/// of the (slightly regularised) within-class scatter.
#[derive(Debug, Clone)]
pub struct Lda {
    components: usize,
    shrinkage: f64,
    mean: R1D,
    /// Columns are discriminant directions, `(features, components)`.
    directions: R2D,
    class_means: R2D, // in projected space
    log_priors: Vec<f64>,
    /// Pooled within-class variance along each discriminant direction.
    projected_variance: Vec<f64>,
    eigenvalues: Vec<f64>,
}

impl Lda {
    /// Keeps `components` discriminant directions (at most `classes - 1`).
    pub fn new(components: usize) -> Self {
        Lda { components, shrinkage: 1e-6, mean: R1D::zeros(0), directions: R2D::zeros((0, 0)), class_means: R2D::zeros((0, 0)), log_priors: Vec::new(), projected_variance: Vec::new(), eigenvalues: Vec::new() }
    }

    /// Sets the ridge added to the within-class scatter, as a fraction of its mean diagonal; use a
    /// larger value (for example `0.1`) when features are nearly collinear or samples are few.
    pub fn with_shrinkage(mut self, shrinkage: f64) -> Self {
        self.shrinkage = shrinkage;
        self
    }

    /// The Fisher criterion value (between/within scatter ratio) of each kept direction.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn directions(&self) -> &R2D {
        &self.directions
    }

    /// Fits on labelled data.
    pub fn fit_labelled(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        let (n, p) = x.dim();
        let classes = class_count(labels);
        if classes < 2 || (0..classes).any(|c| !labels.contains(&c)) {
            return Err(ErrorsJSL::InvalidInputRange("Need at least two classes, each with samples."));
        }
        if self.components == 0 || self.components > (classes - 1).min(p) {
            return Err(ErrorsJSL::InvalidInputRange("components must be between 1 and min(classes - 1, features)."));
        }
        self.mean = x.mean_axis(ndarray::Axis(0)).expect("non-empty");
        let counts: Vec<usize> = (0..classes).map(|c| labels.iter().filter(|&&l| l == c).count()).collect();
        let mut class_mean = Array2::<f64>::zeros((classes, p));
        for (row, &l) in x.rows().into_iter().zip(labels) {
            class_mean.row_mut(l).zip_mut_with(&row, |m, v| *m += v / counts[l] as f64);
        }
        let mut sw = Array2::<f64>::zeros((p, p));
        for (row, &l) in x.rows().into_iter().zip(labels) {
            let d = &row - &class_mean.row(l);
            for a in 0..p {
                for b in 0..p {
                    sw[[a, b]] += d[a] * d[b];
                }
            }
        }
        sw /= n as f64;
        let mut sb = Array2::<f64>::zeros((p, p));
        for (c, &count) in counts.iter().enumerate() {
            let d = &class_mean.row(c) - &self.mean;
            for a in 0..p {
                for b in 0..p {
                    sb[[a, b]] += count as f64 / n as f64 * d[a] * d[b];
                }
            }
        }
        let ridge = self.shrinkage * (0..p).map(|j| sw[[j, j]]).sum::<f64>() / p as f64;
        for j in 0..p {
            sw[[j, j]] += ridge.max(1e-12);
        }
        // ANCHOR: lda_whiten
        // Whiten: Sw = L Lᵀ, M = L⁻¹ Sb L⁻ᵀ (symmetric), directions w = L⁻ᵀ v.
        let l = sw.cholesky(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("The within-class scatter is not positive definite (try more shrinkage)."))?;
        let left = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &sb).map_err(|_| ErrorsJSL::RuntimeError("Whitening failed."))?;
        let m = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &left.t().to_owned()).map_err(|_| ErrorsJSL::RuntimeError("Whitening failed."))?;
        let m = (&m + &m.t()) * 0.5;
        let (vals, vecs) = m.eigh(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("The eigendecomposition failed."))?;
        let order: Vec<usize> = (0..p).rev().take(self.components).collect();
        let lt = l.t().to_owned();
        self.eigenvalues = order.iter().map(|&i| vals[i].max(0.0)).collect();
        let mut dirs = R2D::zeros((p, self.components));
        for (c, &i) in order.iter().enumerate() {
            let v = vecs.column(i).to_owned();
            let w = lt.solve_triangular(UPLO::Upper, Diag::NonUnit, &v).map_err(|_| ErrorsJSL::RuntimeError("Back-substitution failed."))?;
            let norm = w.dot(&w).sqrt().max(1e-300);
            let big = w.iter().copied().fold(0.0f64, |m, x| if x.abs() > m.abs() { x } else { m });
            dirs.column_mut(c).assign(&(w * (big.signum() / norm)));
        }
        // ANCHOR_END: lda_whiten
        self.directions = dirs;
        self.class_means = class_mean.dot(&self.directions) - self.mean.dot(&self.directions);
        self.log_priors = counts.iter().map(|&c| (c as f64 / n as f64).ln()).collect();
        // Pooled within-class variance of the projected training data along each direction.
        let z = (&x - &self.mean).dot(&self.directions);
        self.projected_variance = (0..self.components)
            .map(|k| {
                let ss: f64 = labels.iter().enumerate().map(|(i, &l)| (z[[i, k]] - self.class_means[[l, k]]).powi(2)).sum();
                (ss / n as f64).max(1e-300)
            })
            .collect();
        Ok(())
    }

    /// Projects rows onto the discriminant directions.
    pub fn transform(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.directions.ncols() == 0 {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.mean.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.mean.len())));
        }
        Ok((&x - &self.mean).dot(&self.directions))
    }
}

impl Classifier for Lda {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        self.fit_labelled(x, labels)
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        let z = self.transform(x)?;
        Ok(z.rows()
            .into_iter()
            .map(|row| {
                // Gaussian classifier with a shared covariance: log-prior minus the squared
                // distance to each class mean, measured in units of the within-class spread.
                let scores: Vec<f64> = (0..self.class_means.nrows())
                    .map(|c| self.log_priors[c] - 0.5 * (0..row.len()).map(|k| (row[k] - self.class_means[[c, k]]).powi(2) / self.projected_variance[k]).sum::<f64>())
                    .collect();
                argmax(&scores).unwrap_or(0)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_classification};
    use crate::metrics::accuracy;
    use crate::unsupervised::reduction::principal_component_analysis::Pca;
    use crate::common::{MlRng, Transformer};
    use ndarray::arr2;

    /// Two classes separated along x but with huge, class-independent variance along y.
    fn tricky(n: usize, seed: u64) -> (R2D, Vec<usize>) {
        let mut rng = MlRng::new(seed);
        let labels: Vec<usize> = (0..2 * n).map(|i| usize::from(i >= n)).collect();
        let x = R2D::from_shape_fn((2 * n, 2), |(i, j)| if j == 0 { rng.gaussian(if labels[i] == 0 { -1.5 } else { 1.5 }, 0.5) } else { rng.gaussian(0.0, 10.0) });
        (x, labels)
    }

    fn separation(values: &[f64], labels: &[usize]) -> f64 {
        let mean = |c: usize| {
            let v: Vec<f64> = values.iter().zip(labels).filter(|(_, l)| **l == c).map(|(v, _)| *v).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let var = |c: usize, m: f64| {
            let v: Vec<f64> = values.iter().zip(labels).filter(|(_, l)| **l == c).map(|(v, _)| (v - m).powi(2)).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let (m0, m1) = (mean(0), mean(1));
        (m1 - m0).powi(2) / (var(0, m0) + var(1, m1))
    }

    #[test]
    fn finds_the_discriminating_direction_where_pca_follows_variance() {
        let (x, l) = tricky(300, 1);
        let mut lda = Lda::new(1);
        lda.fit_labelled(x.view(), &l).unwrap();
        let d = lda.directions().column(0);
        assert!(d[0].abs() > 0.99 && d[1].abs() < 0.1, "{d:?}");
        let zl = lda.transform(x.view()).unwrap();
        let mut pca = Pca::new(1);
        let zp = pca.fit_transform(x.view()).unwrap();
        assert!(separation(&zl.column(0).to_vec(), &l) > 10.0 * separation(&zp.column(0).to_vec(), &l));
        assert!(lda.eigenvalues()[0] > 3.0);
    }

    #[test]
    fn classifies_and_reduces_multiclass_data_to_classes_minus_one_dimensions() {
        let (x, l) = make_classification(100, 3);
        let mut lda = Lda::new(2);
        lda.fit(x.view(), &l).unwrap();
        assert_eq!(lda.transform(x.view()).unwrap().dim(), (300, 2));
        assert!(accuracy(&l, &lda.predict(x.view()).unwrap()).unwrap() > 0.85);
        assert!(lda.eigenvalues()[0] >= lda.eigenvalues()[1]);
        let (bx, bl) = make_blobs(40, &[vec![0.0, 0.0], vec![6.0, 0.0]], 0.8, 4);
        let mut two = Lda::new(1).with_shrinkage(0.05);
        two.fit(bx.view(), &bl).unwrap();
        assert_eq!(accuracy(&bl, &two.predict(bx.view()).unwrap()).unwrap(), 1.0);
    }

    #[test]
    fn errors() {
        let x = arr2(&[[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]]);
        let l = [0, 0, 1, 1];
        assert!(Lda::new(0).fit_labelled(x.view(), &l).is_err());
        assert!(Lda::new(2).fit_labelled(x.view(), &l).is_err()); // only one discriminant for two classes
        assert!(Lda::new(1).fit_labelled(x.view(), &[0, 0, 0, 0]).is_err());
        assert!(Lda::new(1).fit_labelled(x.view(), &[0, 0, 1]).is_err());
        assert!(Lda::new(1).transform(x.view()).is_err());
        let mut m = Lda::new(1);
        m.fit_labelled(x.view(), &l).unwrap();
        assert!(m.transform(arr2(&[[1.0, 2.0, 3.0]]).view()).is_err());
    }
}
