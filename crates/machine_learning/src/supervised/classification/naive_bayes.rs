//! Naive Bayes classifiers.
//!
//! See the book entry [Naive Bayes](https://jwkunz.github.io/lib_jsl/book/machine_learning/naive-bayes.html).

use crate::common::{Classifier, argmax, check_features, check_xy, class_count};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// Gaussian naive Bayes: each class models every feature as an independent normal distribution
/// (the "naive" independence assumption), and a point goes to the class with the highest posterior
/// `P(class) Π P(x_j | class)`. Fast, needs little data, and surprisingly strong when features are
/// roughly independent given the class. `var_smoothing` adds a fraction of the largest feature
/// variance to every variance, which keeps constant features from causing division by zero.
#[derive(Debug, Clone)]
pub struct GaussianNaiveBayes {
    var_smoothing: f64,
    log_prior: Vec<f64>,
    mean: Vec<Vec<f64>>,
    var: Vec<Vec<f64>>,
}

impl GaussianNaiveBayes {
    pub fn new() -> Self {
        GaussianNaiveBayes { var_smoothing: 1e-9, log_prior: Vec::new(), mean: Vec::new(), var: Vec::new() }
    }

    pub fn with_var_smoothing(mut self, smoothing: f64) -> Self {
        self.var_smoothing = smoothing;
        self
    }

    /// Per-class feature means (`[class][feature]`).
    pub fn class_means(&self) -> &[Vec<f64>] {
        &self.mean
    }

    // ANCHOR: gaussian_nb
    /// Joint log-likelihood `log P(class) + Σ log P(x_j | class)` as a `(rows, classes)` matrix.
    pub fn joint_log_likelihood(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.mean.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.mean[0].len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.mean[0].len())));
        }
        Ok(R2D::from_shape_fn((x.nrows(), self.mean.len()), |(i, c)| {
            self.log_prior[c]
                - 0.5
                    * (0..x.ncols())
                        .map(|j| (std::f64::consts::TAU * self.var[c][j]).ln() + (x[[i, j]] - self.mean[c][j]).powi(2) / self.var[c][j])
                        .sum::<f64>()
        }))
    }
    // ANCHOR_END: gaussian_nb

    /// Posterior class probabilities.
    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        let mut jll = self.joint_log_likelihood(x)?;
        for mut row in jll.rows_mut() {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|v| (v - max).exp());
            let total = row.sum();
            row.mapv_inplace(|v| v / total);
        }
        Ok(jll)
    }
}

impl Default for GaussianNaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for GaussianNaiveBayes {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        let (n, p) = x.dim();
        let classes = class_count(labels);
        let counts: Vec<usize> = (0..classes).map(|c| labels.iter().filter(|&&l| l == c).count()).collect();
        if counts.contains(&0) {
            return Err(ErrorsJSL::InvalidInputRange("Every class label in 0..classes must occur."));
        }
        let max_var = (0..p)
            .map(|j| {
                let m = x.column(j).sum() / n as f64;
                x.column(j).iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64
            })
            .fold(0.0, f64::max);
        let eps = self.var_smoothing * max_var.max(1e-12);
        self.mean = vec![vec![0.0; p]; classes];
        self.var = vec![vec![0.0; p]; classes];
        for (row, &c) in x.rows().into_iter().zip(labels) {
            for j in 0..p {
                self.mean[c][j] += row[j] / counts[c] as f64;
            }
        }
        for (row, &c) in x.rows().into_iter().zip(labels) {
            for j in 0..p {
                self.var[c][j] += (row[j] - self.mean[c][j]).powi(2) / counts[c] as f64;
            }
        }
        self.var.iter_mut().flatten().for_each(|v| *v += eps);
        self.log_prior = counts.iter().map(|&c| (c as f64 / n as f64).ln()).collect();
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        Ok(self.joint_log_likelihood(x)?.rows().into_iter().map(|r| argmax(&r.to_vec()).unwrap_or(0)).collect())
    }
}

/// Multinomial naive Bayes for count features (word counts, histograms): the class-conditional
/// feature distribution is multinomial with Laplace/Lidstone smoothing `alpha`.
#[derive(Debug, Clone)]
pub struct MultinomialNaiveBayes {
    alpha: f64,
    log_prior: Vec<f64>,
    log_likelihood: Vec<Vec<f64>>,
}

impl MultinomialNaiveBayes {
    pub fn new(alpha: f64) -> Self {
        MultinomialNaiveBayes { alpha, log_prior: Vec::new(), log_likelihood: Vec::new() }
    }
}

impl Classifier for MultinomialNaiveBayes {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        if x.iter().any(|&v| v < 0.0) || self.alpha.is_nan() || self.alpha < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Counts must be non-negative and alpha >= 0."));
        }
        let (n, p) = x.dim();
        let classes = class_count(labels);
        let mut totals = vec![vec![0.0; p]; classes];
        let mut counts = vec![0usize; classes];
        for (row, &c) in x.rows().into_iter().zip(labels) {
            counts[c] += 1;
            for j in 0..p {
                totals[c][j] += row[j];
            }
        }
        if counts.contains(&0) {
            return Err(ErrorsJSL::InvalidInputRange("Every class label in 0..classes must occur."));
        }
        self.log_prior = counts.iter().map(|&c| (c as f64 / n as f64).ln()).collect();
        self.log_likelihood = totals
            .iter()
            .map(|t| {
                let denom = t.iter().sum::<f64>() + self.alpha * p as f64;
                t.iter().map(|&v| ((v + self.alpha) / denom).ln()).collect()
            })
            .collect();
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        if self.log_prior.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.log_likelihood[0].len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.log_likelihood[0].len())));
        }
        Ok(x.rows()
            .into_iter()
            .map(|r| {
                let scores: Vec<f64> = (0..self.log_prior.len()).map(|c| {
                        // Absent features contribute nothing: with alpha = 0 an unseen feature has a
                        // log-likelihood of -inf, and 0 * -inf would be NaN.
                        self.log_prior[c] + r.iter().zip(&self.log_likelihood[c]).filter(|(v, _)| **v != 0.0).map(|(v, l)| v * l).sum::<f64>()
                    })
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
    use crate::preprocessing::{select_labels, select_rows, train_test_split};
    use ndarray::arr2;

    #[test]
    fn gaussian_nb_classifies_overlapping_classes() {
        let (x, l) = make_classification(120, 7);
        let (tr, te) = train_test_split(x.nrows(), 0.3, 2).unwrap();
        let mut m = GaussianNaiveBayes::new();
        m.fit(select_rows(x.view(), &tr).view(), &select_labels(&l, &tr)).unwrap();
        let acc = accuracy(&select_labels(&l, &te), &m.predict(select_rows(x.view(), &te).view()).unwrap()).unwrap();
        assert!(acc > 0.8, "{acc}");
        let p = m.predict_proba(x.view()).unwrap();
        assert!(p.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
        assert_eq!(m.class_means().len(), 3);
    }

    #[test]
    fn priors_matter_and_constant_features_are_safe() {
        // Two identical blobs but 9:1 class imbalance -> the prior dominates at the shared centre.
        let (a, _) = make_blobs(90, &[vec![0.0, 0.0]], 1.0, 1);
        let (b, _) = make_blobs(10, &[vec![0.0, 0.0]], 1.0, 2);
        let x = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        let labels: Vec<usize> = std::iter::repeat_n(0, 90).chain(std::iter::repeat_n(1, 10)).collect();
        let mut m = GaussianNaiveBayes::new();
        m.fit(x.view(), &labels).unwrap();
        assert_eq!(m.predict(arr2(&[[0.0, 0.0]]).view()).unwrap(), vec![0]);
        let with_const = arr2(&[[1.0, 5.0], [1.2, 5.0], [3.0, 5.0], [3.1, 5.0]]);
        let mut c = GaussianNaiveBayes::new();
        c.fit(with_const.view(), &[0, 0, 1, 1]).unwrap();
        assert_eq!(c.predict(with_const.view()).unwrap(), vec![0, 0, 1, 1]);
    }

    #[test]
    fn multinomial_nb_on_word_counts() {
        // Features: counts of ["goal", "match", "vote", "law"]; class 0 = sport, 1 = politics.
        let x = arr2(&[[3.0, 2.0, 0.0, 0.0], [2.0, 4.0, 1.0, 0.0], [0.0, 1.0, 3.0, 2.0], [0.0, 0.0, 4.0, 3.0]]);
        let mut m = MultinomialNaiveBayes::new(1.0);
        m.fit(x.view(), &[0, 0, 1, 1]).unwrap();
        assert_eq!(m.predict(arr2(&[[1.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 5.0]]).view()).unwrap(), vec![0, 1]);
        assert!(MultinomialNaiveBayes::new(1.0).fit(arr2(&[[-1.0]]).view(), &[0]).is_err());
        assert!(MultinomialNaiveBayes::new(1.0).predict(x.view()).is_err());
        // Without smoothing, a word never seen in a class rules that class out only when it occurs.
        let mut unsmoothed = MultinomialNaiveBayes::new(0.0);
        unsmoothed.fit(x.view(), &[0, 0, 1, 1]).unwrap();
        assert_eq!(unsmoothed.predict(arr2(&[[1.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 5.0]]).view()).unwrap(), vec![0, 1]);
    }

    #[test]
    fn errors() {
        let x = arr2(&[[0.0], [1.0]]);
        assert!(GaussianNaiveBayes::new().fit(x.view(), &[0, 2]).is_err()); // class 1 missing
        assert!(GaussianNaiveBayes::new().predict(x.view()).is_err());
        let mut m = GaussianNaiveBayes::new();
        m.fit(x.view(), &[0, 1]).unwrap();
        assert!(m.predict(arr2(&[[1.0, 2.0]]).view()).is_err());
    }
}
