//! Ensembles: gradient-boosted regression trees for regression and classification.
//!
//! Gradient boosting (Friedman) builds an additive model `F(x) = F₀ + ν Σ_m h_m(x)` stage by
//! stage: each small regression tree `h_m` is fitted to the negative gradient of the loss at the
//! current predictions, and its leaves take the Newton step `Σ(−g) / Σh` of the loss restricted to
//! the leaf. The learning rate `ν` (shrinkage) and row subsampling (stochastic boosting)
//! regularise the fit.
//!
//! See the book entry [Gradient boosting](https://jwkunz.github.io/lib_jsl/book/learning/gradient-boosting.html).

use crate::common::{Classifier, MlRng, Regressor, check_xy, class_count, rows};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};

#[derive(Debug, Clone)]
enum Node {
    Leaf(f64),
    Split { feature: usize, threshold: f64, left: usize, right: usize },
}

/// A least-squares regression tree whose leaves hold Newton steps `Σr / Σw` (the mean residual
/// when every weight is one).
#[derive(Debug, Clone)]
pub struct RegressionTree {
    nodes: Vec<Node>,
    max_depth: usize,
    min_samples_leaf: usize,
}

impl RegressionTree {
    pub fn new(max_depth: usize, min_samples_leaf: usize) -> Self {
        RegressionTree { nodes: Vec::new(), max_depth, min_samples_leaf: min_samples_leaf.max(1) }
    }

    /// Fits the tree on rows `samples` of `x` to residuals `r` with Newton weights `w`; adds each
    /// split's squared-error reduction to `importances`.
    pub fn fit(&mut self, x: &[Vec<f64>], r: &[f64], w: &[f64], samples: &[usize], importances: &mut [f64]) {
        self.nodes.clear();
        self.grow(x, r, w, samples.to_vec(), 0, importances);
    }

    fn leaf_value(r: &[f64], w: &[f64], idx: &[usize]) -> f64 {
        let (num, den) = idx.iter().fold((0.0, 0.0), |(a, b), &i| (a + r[i], b + w[i]));
        if den > 1e-12 { num / den } else { 0.0 }
    }

    #[allow(clippy::needless_range_loop)]
    fn grow(&mut self, x: &[Vec<f64>], r: &[f64], w: &[f64], idx: Vec<usize>, depth: usize, importances: &mut [f64]) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::Leaf(Self::leaf_value(r, w, &idx)));
        if depth >= self.max_depth || idx.len() < 2 * self.min_samples_leaf {
            return id;
        }
        let n = idx.len() as f64;
        let total: f64 = idx.iter().map(|&i| r[i]).sum();
        let parent = total * total / n;
        let mut best: Option<(f64, usize, f64)> = None;
        let features = x[0].len();
        let mut order = idx.clone();
        for f in 0..features {
            order.sort_by(|&a, &b| x[a][f].total_cmp(&x[b][f]));
            let mut left_sum = 0.0;
            for k in 0..order.len() - 1 {
                left_sum += r[order[k]];
                let (nl, nr) = ((k + 1) as f64, (order.len() - k - 1) as f64);
                if k + 1 < self.min_samples_leaf || order.len() - k - 1 < self.min_samples_leaf || x[order[k]][f] == x[order[k + 1]][f] {
                    continue;
                }
                // Reduction in the sum of squared residuals: S_L²/n_L + S_R²/n_R − S²/n.
                let gain = left_sum * left_sum / nl + (total - left_sum).powi(2) / nr - parent;
                if best.is_none_or(|(g, _, _)| gain > g) {
                    best = Some((gain, f, 0.5 * (x[order[k]][f] + x[order[k + 1]][f])));
                }
            }
        }
        let Some((gain, feature, threshold)) = best else { return id };
        if gain <= 1e-12 {
            return id;
        }
        importances[feature] += gain;
        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[i][feature] <= threshold);
        let left = self.grow(x, r, w, left_idx, depth + 1, importances);
        let right = self.grow(x, r, w, right_idx, depth + 1, importances);
        self.nodes[id] = Node::Split { feature, threshold, left, right };
        id
    }

    pub fn predict_row(&self, row: &[f64]) -> f64 {
        let mut i = 0;
        loop {
            match self.nodes[i] {
                Node::Leaf(v) => return v,
                Node::Split { feature, threshold, left, right } => i = if row[feature] <= threshold { left } else { right },
            }
        }
    }
}

/// Shared boosting settings.
#[derive(Debug, Clone)]
pub struct BoostingParameters {
    pub stages: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    /// Fraction of rows used per stage (1 for deterministic boosting).
    pub subsample: f64,
    pub seed: u64,
}

impl Default for BoostingParameters {
    fn default() -> Self {
        BoostingParameters { stages: 200, learning_rate: 0.1, max_depth: 3, min_samples_leaf: 1, subsample: 1.0, seed: 0 }
    }
}

impl BoostingParameters {
    fn validate(&self) -> Result<(), ErrorsJSL> {
        if self.stages == 0 || self.learning_rate.is_nan() || self.learning_rate <= 0.0 || !(0.0..=1.0).contains(&self.subsample) || self.subsample == 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need stages > 0, a positive learning rate and 0 < subsample <= 1."));
        }
        Ok(())
    }

    fn sample(&self, n: usize, rng: &mut MlRng) -> Vec<usize> {
        if self.subsample >= 1.0 { (0..n).collect() } else { rng.sample_indices(n, ((self.subsample * n as f64).round() as usize).max(1)) }
    }
}

/// Gradient boosting for regression with the squared-error loss.
#[derive(Debug, Clone)]
pub struct GradientBoostingRegressor {
    pub parameters: BoostingParameters,
    features: usize,
    base: f64,
    trees: Vec<RegressionTree>,
    importances: Vec<f64>,
    /// Training mean squared error after each stage.
    pub training_loss: Vec<f64>,
}

impl GradientBoostingRegressor {
    pub fn new(parameters: BoostingParameters) -> Self {
        GradientBoostingRegressor { parameters, features: 0, base: 0.0, trees: Vec::new(), importances: Vec::new(), training_loss: Vec::new() }
    }

    /// Relative feature importances (normalised squared-error reductions).
    pub fn feature_importances(&self) -> Vec<f64> {
        let total: f64 = self.importances.iter().sum();
        self.importances.iter().map(|v| if total > 0.0 { v / total } else { 0.0 }).collect()
    }
}

impl Regressor for GradientBoostingRegressor {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL> {
        check_xy(x, y.len())?;
        self.parameters.validate()?;
        let p = &self.parameters;
        let data = rows(x);
        let n = data.len();
        self.features = x.ncols();
        self.base = y.iter().sum::<f64>() / n as f64;
        let mut prediction = vec![self.base; n];
        let ones = vec![1.0; n];
        let mut rng = MlRng::new(p.seed);
        self.importances = vec![0.0; x.ncols()];
        self.trees.clear();
        self.training_loss.clear();
        // ANCHOR: boost_regression
        for _ in 0..p.stages {
            let residual: Vec<f64> = (0..n).map(|i| y[i] - prediction[i]).collect();
            let mut tree = RegressionTree::new(p.max_depth, p.min_samples_leaf);
            tree.fit(&data, &residual, &ones, &p.sample(n, &mut rng), &mut self.importances);
            for (i, row) in data.iter().enumerate() {
                prediction[i] += p.learning_rate * tree.predict_row(row);
            }
            self.trees.push(tree);
            self.training_loss.push((0..n).map(|i| (y[i] - prediction[i]).powi(2)).sum::<f64>() / n as f64);
        }
        // ANCHOR_END: boost_regression
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<R1D, ErrorsJSL> {
        check_fitted(self.trees.is_empty(), self.features, x)?;
        Ok(x.rows().into_iter().map(|row| {
            let r = row.to_vec();
            self.base + self.parameters.learning_rate * self.trees.iter().map(|t| t.predict_row(&r)).sum::<f64>()
        }).collect())
    }
}

/// Gradient boosting for classification with the multinomial deviance (softmax cross-entropy):
/// one tree per class per stage, leaves taking the Newton step `Σ(y − p) / Σ p(1 − p)`.
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier {
    pub parameters: BoostingParameters,
    features: usize,
    base: Vec<f64>,
    trees: Vec<Vec<RegressionTree>>,
    /// Mean training deviance (cross-entropy) after each stage.
    pub training_loss: Vec<f64>,
}

/// Errors for an unfitted model or a feature count different from training.
fn check_fitted(unfitted: bool, features: usize, x: VR2D) -> Result<(), ErrorsJSL> {
    if unfitted {
        return Err(ErrorsJSL::RuntimeError("The model has not been fitted."));
    }
    if x.ncols() != features {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), features)));
    }
    Ok(())
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

impl GradientBoostingClassifier {
    pub fn new(parameters: BoostingParameters) -> Self {
        GradientBoostingClassifier { parameters, features: 0, base: Vec::new(), trees: Vec::new(), training_loss: Vec::new() }
    }

    fn scores(&self, row: &[f64]) -> Vec<f64> {
        let mut s = self.base.clone();
        for stage in &self.trees {
            for (k, tree) in stage.iter().enumerate() {
                s[k] += self.parameters.learning_rate * tree.predict_row(row);
            }
        }
        s
    }

    /// Class probabilities, one row per sample.
    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        check_fitted(self.trees.is_empty(), self.features, x)?;
        let k = self.base.len();
        let mut out = R2D::zeros((x.nrows(), k));
        for (i, row) in x.rows().into_iter().enumerate() {
            for (c, p) in softmax(&self.scores(&row.to_vec())).into_iter().enumerate() {
                out[[i, c]] = p;
            }
        }
        Ok(out)
    }
}

impl Classifier for GradientBoostingClassifier {
    #[allow(clippy::needless_range_loop)]
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        self.parameters.validate()?;
        let p = self.parameters.clone();
        let k = class_count(labels).max(2);
        let data = rows(x);
        let n = data.len();
        self.features = x.ncols();
        // Start from the log class priors (smoothed so absent classes stay finite).
        self.base = (0..k).map(|c| ((labels.iter().filter(|&&l| l == c).count() as f64 + 0.5) / (n as f64 + 0.5 * k as f64)).ln()).collect();
        let mut scores: Vec<Vec<f64>> = vec![self.base.clone(); n];
        let mut rng = MlRng::new(p.seed);
        let mut importances = vec![0.0; x.ncols()];
        self.trees.clear();
        self.training_loss.clear();
        for _ in 0..p.stages {
            let probabilities: Vec<Vec<f64>> = scores.iter().map(|s| softmax(s)).collect();
            let sample = p.sample(n, &mut rng);
            let mut stage = Vec::with_capacity(k);
            // ANCHOR: boost_classification
            for c in 0..k {
                let residual: Vec<f64> = (0..n).map(|i| f64::from(u8::from(labels[i] == c)) - probabilities[i][c]).collect();
                let hessian: Vec<f64> = (0..n).map(|i| probabilities[i][c] * (1.0 - probabilities[i][c])).collect();
                let mut tree = RegressionTree::new(p.max_depth, p.min_samples_leaf);
                tree.fit(&data, &residual, &hessian, &sample, &mut importances);
                // Friedman's (K − 1)/K factor for the multinomial Newton step.
                let factor = (k - 1) as f64 / k as f64;
                for node in tree.nodes.iter_mut() {
                    if let Node::Leaf(v) = node {
                        *v = (*v * factor).clamp(-20.0, 20.0);
                    }
                }
                stage.push(tree);
            }
            // ANCHOR_END: boost_classification
            for (i, row) in data.iter().enumerate() {
                for (c, tree) in stage.iter().enumerate() {
                    scores[i][c] += p.learning_rate * tree.predict_row(row);
                }
            }
            self.trees.push(stage);
            self.training_loss.push(-(0..n).map(|i| softmax(&scores[i])[labels[i]].max(1e-300).ln()).sum::<f64>() / n as f64);
        }
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        let proba = self.predict_proba(x)?;
        Ok(proba.rows().into_iter().map(|r| crate::common::argmax(&r.to_vec()).unwrap_or(0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_circles, make_moons, make_sine};
    use crate::metrics::{accuracy, mean_squared_error};
    use crate::preprocessing::train_test_split;
    use ndarray::Axis;

    #[test]
    fn boosting_fits_a_sine_far_better_than_one_tree() {
        let (x, y) = make_sine(400, 0.1, 1);
        let (train, test) = train_test_split(400, 0.25, 2).unwrap();
        let (xt, yt) = (x.select(Axis(0), &train), y.select(Axis(0), &train));
        let (xs, ys) = (x.select(Axis(0), &test), y.select(Axis(0), &test));
        let mut gb = GradientBoostingRegressor::new(BoostingParameters { stages: 300, learning_rate: 0.05, ..Default::default() });
        gb.fit(xt.view(), yt.view()).unwrap();
        let boosted = mean_squared_error(ys.view(), gb.predict(xs.view()).unwrap().view()).unwrap();
        let mut stump = GradientBoostingRegressor::new(BoostingParameters { stages: 1, learning_rate: 1.0, ..Default::default() });
        stump.fit(xt.view(), yt.view()).unwrap();
        let single = mean_squared_error(ys.view(), stump.predict(xs.view()).unwrap().view()).unwrap();
        // Compare the error above the noise floor (noise variance 0.01).
        assert!(boosted < 0.02 && boosted - 0.01 < (single - 0.01) / 3.0, "boosted {boosted} single {single}");
        // Training loss decreases monotonically for full-sample boosting.
        assert!(gb.training_loss.windows(2).all(|w| w[1] <= w[0] + 1e-12));
        assert!((gb.feature_importances().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn boosted_classifiers_separate_nonlinear_classes() {
        for (name, (x, labels)) in [("moons", make_moons(400, 0.15, 3)), ("circles", make_circles(400, 0.08, 0.5, 4))] {
            let (train, test) = train_test_split(400, 0.25, 5).unwrap();
            let lt: Vec<usize> = train.iter().map(|&i| labels[i]).collect();
            let ls: Vec<usize> = test.iter().map(|&i| labels[i]).collect();
            let mut gb = GradientBoostingClassifier::new(BoostingParameters { stages: 100, subsample: 0.8, seed: 6, ..Default::default() });
            gb.fit(x.select(Axis(0), &train).view(), &lt).unwrap();
            let predicted = gb.predict(x.select(Axis(0), &test).view()).unwrap();
            assert!(accuracy(&ls, &predicted).unwrap() > 0.93, "{name}");
            let proba = gb.predict_proba(x.select(Axis(0), &test).view()).unwrap();
            assert!(proba.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
            assert!(gb.training_loss.last().unwrap() < &gb.training_loss[0]);
        }
        let mut bad = GradientBoostingClassifier::new(BoostingParameters { learning_rate: 0.0, ..Default::default() });
        let (x, l) = make_moons(20, 0.1, 1);
        assert!(bad.fit(x.view(), &l).is_err());
        // Predicting with the wrong number of features is an error, not a panic.
        let mut gb = GradientBoostingClassifier::new(BoostingParameters { stages: 3, ..Default::default() });
        gb.fit(x.view(), &l).unwrap();
        assert!(gb.predict(R2D::zeros((2, 1)).view()).is_err());
        let mut gr = GradientBoostingRegressor::new(BoostingParameters { stages: 3, ..Default::default() });
        gr.fit(x.view(), x.column(0)).unwrap();
        assert!(gr.predict(R2D::zeros((2, 3)).view()).is_err());
    }
}
