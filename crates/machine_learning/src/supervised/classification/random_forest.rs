//! Decision trees and random forests.
//!
//! See the book entry [Decision trees and random forests](https://jwkunz.github.io/lib_jsl/book/machine_learning/decision-trees.html).

use crate::common::{Classifier, MlRng, argmax, check_features, check_xy, class_count};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

#[derive(Debug, Clone)]
enum Node {
    Leaf { distribution: Vec<f64> },
    Split { feature: usize, threshold: f64, left: Box<Node>, right: Box<Node> },
}

/// A CART classification tree grown greedily by Gini impurity.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    max_depth: usize,
    min_samples_split: usize,
    /// Features examined at each split (`None` = all); used by forests for decorrelation.
    max_features: Option<usize>,
    seed: u64,
    root: Option<Node>,
    classes: usize,
    features: usize,
}

fn gini(counts: &[f64], total: f64) -> f64 {
    if total == 0.0 { 0.0 } else { 1.0 - counts.iter().map(|c| (c / total).powi(2)).sum::<f64>() }
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        DecisionTree { max_depth, min_samples_split: 2, max_features: None, seed: 0, root: None, classes: 0, features: 0 }
    }

    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n.max(2);
        self
    }

    pub fn with_max_features(mut self, m: usize, seed: u64) -> Self {
        self.max_features = Some(m);
        self.seed = seed;
        self
    }

    /// Depth of the grown tree (0 for a single leaf).
    pub fn depth(&self) -> usize {
        fn go(n: &Node) -> usize {
            match n {
                Node::Leaf { .. } => 0,
                Node::Split { left, right, .. } => 1 + go(left).max(go(right)),
            }
        }
        self.root.as_ref().map_or(0, go)
    }

    fn grow(&self, x: VR2D, labels: &[usize], idx: &[usize], depth: usize, rng: &mut MlRng) -> Node {
        let mut counts = vec![0.0; self.classes];
        idx.iter().for_each(|&i| counts[labels[i]] += 1.0);
        let n = idx.len() as f64;
        let leaf = |counts: &[f64]| Node::Leaf { distribution: counts.iter().map(|c| c / n).collect() };
        let parent = gini(&counts, n);
        if depth >= self.max_depth || idx.len() < self.min_samples_split || parent == 0.0 {
            return leaf(&counts);
        }
        let candidates: Vec<usize> = match self.max_features {
            Some(m) => rng.sample_indices(self.features, m.clamp(1, self.features)),
            None => (0..self.features).collect(),
        };
        // ANCHOR: cart_split
        let mut best: Option<(f64, usize, f64)> = None; // (impurity decrease, feature, threshold)
        for &f in &candidates {
            let mut order: Vec<usize> = idx.to_vec();
            order.sort_by(|&a, &b| x[[a, f]].total_cmp(&x[[b, f]]));
            let mut left = vec![0.0; self.classes];
            for k in 0..order.len() - 1 {
                left[labels[order[k]]] += 1.0;
                let (a, b) = (x[[order[k], f]], x[[order[k + 1], f]]);
                if a == b {
                    continue;
                }
                let nl = (k + 1) as f64;
                let right: Vec<f64> = counts.iter().zip(&left).map(|(c, l)| c - l).collect();
                let weighted = (nl * gini(&left, nl) + (n - nl) * gini(&right, n - nl)) / n;
                let gain = parent - weighted;
                if best.is_none_or(|(g, _, _)| gain > g + 1e-15) {
                    best = Some((gain, f, 0.5 * (a + b)));
                }
            }
        }
        match best {
            // Zero-gain splits are allowed: structure such as XOR has no single split that improves
            // purity, yet splitting twice separates it. Depth and node-size limits ensure termination.
            Some((gain, feature, threshold)) if gain > -1e-12 => {
                let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[[i, feature]] <= threshold);
                Node::Split { feature, threshold, left: Box::new(self.grow(x, labels, &l, depth + 1, rng)), right: Box::new(self.grow(x, labels, &r, depth + 1, rng)) }
            }
            _ => leaf(&counts),
        }
        // ANCHOR_END: cart_split
    }

    fn distribution(&self, row: &[f64]) -> &[f64] {
        let mut node = self.root.as_ref().expect("fitted");
        loop {
            match node {
                Node::Leaf { distribution } => return distribution,
                Node::Split { feature, threshold, left, right } => node = if row[*feature] <= *threshold { left } else { right },
            }
        }
    }

    /// Class probabilities (the leaf's class shares) for each row.
    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.root.is_none() {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.features {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.features)));
        }
        Ok(R2D::from_shape_fn((x.nrows(), self.classes), |(i, c)| self.distribution(&x.row(i).to_vec())[c]))
    }

    fn fit_subset(&mut self, x: VR2D, labels: &[usize], idx: &[usize], classes: usize) {
        self.classes = classes;
        self.features = x.ncols();
        let mut rng = MlRng::new(self.seed);
        self.root = Some(self.grow(x, labels, idx, 0, &mut rng));
    }
}

impl Classifier for DecisionTree {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        let idx: Vec<usize> = (0..x.nrows()).collect();
        self.fit_subset(x, labels, &idx, class_count(labels));
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        Ok(self.predict_proba(x)?.rows().into_iter().map(|r| argmax(&r.to_vec()).unwrap_or(0)).collect())
    }
}

/// A random forest: many decorrelated decision trees, each grown on a bootstrap resample of the
/// data and restricted to a random subset of features at every split (default `√features`), with
/// predictions made by averaging the trees' class probabilities. Averaging cancels the variance
/// of the individual deep trees, so forests resist overfitting. The out-of-bag error (each tree
/// scored on the samples it never saw) gives a free generalisation estimate.
#[derive(Debug, Clone)]
pub struct RandomForest {
    trees: usize,
    max_depth: usize,
    max_features: Option<usize>,
    seed: u64,
    forest: Vec<DecisionTree>,
    classes: usize,
    oob_accuracy: Option<f64>,
    importances: Vec<f64>,
}

impl RandomForest {
    pub fn new(trees: usize, max_depth: usize, seed: u64) -> Self {
        RandomForest { trees, max_depth, max_features: None, seed, forest: Vec::new(), classes: 0, oob_accuracy: None, importances: Vec::new() }
    }

    pub fn with_max_features(mut self, m: usize) -> Self {
        self.max_features = Some(m);
        self
    }

    /// Accuracy on out-of-bag samples from the last fit, if every sample was out of bag at least once.
    pub fn oob_accuracy(&self) -> Option<f64> {
        self.oob_accuracy
    }

    /// How often each feature is used for a split, normalised to sum to one.
    pub fn feature_importances(&self) -> &[f64] {
        &self.importances
    }

    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.forest.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        let mut total = self.forest[0].predict_proba(x)?;
        for t in &self.forest[1..] {
            total += &t.predict_proba(x)?;
        }
        Ok(total / self.forest.len() as f64)
    }
}

fn count_splits(node: &Node, counts: &mut [f64]) {
    if let Node::Split { feature, left, right, .. } = node {
        counts[*feature] += 1.0;
        count_splits(left, counts);
        count_splits(right, counts);
    }
}

impl Classifier for RandomForest {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        if self.trees == 0 {
            return Err(ErrorsJSL::InvalidInputRange("A forest needs at least one tree."));
        }
        let (n, p) = x.dim();
        let classes = class_count(labels);
        let m = self.max_features.unwrap_or(((p as f64).sqrt().round() as usize).max(1));
        let mut rng = MlRng::new(self.seed);
        self.forest.clear();
        let mut oob_votes = vec![vec![0.0; classes]; n];
        let mut oob_seen = vec![false; n];
        // ANCHOR: forest
        for t in 0..self.trees {
            let sample: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
            let mut in_bag = vec![false; n];
            sample.iter().for_each(|&i| in_bag[i] = true);
            let mut tree = DecisionTree::new(self.max_depth).with_max_features(m, self.seed.wrapping_add(1 + t as u64));
            tree.fit_subset(x, labels, &sample, classes);
            for i in (0..n).filter(|&i| !in_bag[i]) {
                let d = tree.distribution(&x.row(i).to_vec());
                oob_votes[i].iter_mut().zip(d).for_each(|(v, p)| *v += p);
                oob_seen[i] = true;
            }
            self.forest.push(tree);
        }
        // ANCHOR_END: forest
        self.classes = classes;
        self.oob_accuracy = oob_seen.iter().all(|&s| s).then(|| (0..n).filter(|&i| argmax(&oob_votes[i]) == Some(labels[i])).count() as f64 / n as f64);
        let mut counts = vec![0.0; p];
        self.forest.iter().filter_map(|t| t.root.as_ref()).for_each(|r| count_splits(r, &mut counts));
        let total: f64 = counts.iter().sum();
        self.importances = counts.iter().map(|c| if total > 0.0 { c / total } else { 0.0 }).collect();
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        Ok(self.predict_proba(x)?.rows().into_iter().map(|r| argmax(&r.to_vec()).unwrap_or(0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_classification, make_moons};
    use crate::metrics::accuracy;
    use crate::preprocessing::{select_labels, select_rows, train_test_split};
    use ndarray::arr2;

    #[test]
    fn a_tree_learns_axis_aligned_rules_exactly() {
        // XOR on a grid: class = (x0 > 0) xor (x1 > 0). No single axis split separates it, but a
        // depth-limited tree still builds it from two levels of splits.
        let pts: Vec<(f64, f64)> = (-4..=4).filter(|&v| v != 0).flat_map(|i| (-4..=4).filter(|&v| v != 0).map(move |j| (i as f64 - 0.3, j as f64 + 0.2))).collect();
        let x = R2D::from_shape_fn((pts.len(), 2), |(i, j)| if j == 0 { pts[i].0 } else { pts[i].1 });
        let l: Vec<usize> = pts.iter().map(|&(a, b)| usize::from((a > 0.0) != (b > 0.0))).collect();
        let mut t = DecisionTree::new(8);
        t.fit(x.view(), &l).unwrap();
        assert_eq!(t.predict(x.view()).unwrap(), l);
        assert!(t.depth() >= 2);
        let mut stump = DecisionTree::new(1);
        stump.fit(x.view(), &l).unwrap();
        assert_eq!(stump.depth(), 1);
        assert!(accuracy(&l, &stump.predict(x.view()).unwrap()).unwrap() < 1.0);
    }

    #[test]
    fn forest_generalises_better_than_one_deep_tree_and_reports_oob() {
        let (x, l) = make_moons(200, 0.3, 4);
        let (tr, te) = train_test_split(x.nrows(), 0.3, 3).unwrap();
        let (xtr, ltr) = (select_rows(x.view(), &tr), select_labels(&l, &tr));
        let (xte, lte) = (select_rows(x.view(), &te), select_labels(&l, &te));
        let mut tree = DecisionTree::new(30);
        tree.fit(xtr.view(), &ltr).unwrap();
        let mut forest = RandomForest::new(60, 30, 5);
        forest.fit(xtr.view(), &ltr).unwrap();
        let (a_tree, a_forest) = (accuracy(&lte, &tree.predict(xte.view()).unwrap()).unwrap(), accuracy(&lte, &forest.predict(xte.view()).unwrap()).unwrap());
        assert!(a_forest >= a_tree && a_forest > 0.85, "{a_tree} {a_forest}");
        let oob = forest.oob_accuracy().unwrap();
        assert!((oob - a_forest).abs() < 0.1, "oob {oob} vs test {a_forest}");
        let p = forest.predict_proba(xte.view()).unwrap();
        assert!(p.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
    }

    #[test]
    fn feature_importance_finds_the_informative_features() {
        // Feature 0 determines the class; features 1-3 are noise.
        let (x, _) = make_classification(100, 2);
        let l: Vec<usize> = (0..x.nrows()).map(|i| usize::from(x[[i, 0]] > 0.5)).collect();
        let mut f = RandomForest::new(40, 6, 1).with_max_features(4);
        f.fit(x.view(), &l).unwrap();
        let imp = f.feature_importances();
        assert!((imp.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(imp[0] > imp[1] && imp[0] > imp[2] && imp[0] > imp[3], "{imp:?}");
        // Same seed, same forest.
        let mut g = RandomForest::new(40, 6, 1).with_max_features(4);
        g.fit(x.view(), &l).unwrap();
        assert_eq!(f.predict(x.view()).unwrap(), g.predict(x.view()).unwrap());
    }

    #[test]
    fn errors() {
        let x = arr2(&[[0.0], [1.0]]);
        assert!(RandomForest::new(0, 3, 1).fit(x.view(), &[0, 1]).is_err());
        assert!(RandomForest::new(3, 3, 1).predict(x.view()).is_err());
        assert!(DecisionTree::new(3).predict(x.view()).is_err());
        let mut t = DecisionTree::new(3);
        t.fit(x.view(), &[0, 1]).unwrap();
        assert!(t.predict(arr2(&[[1.0, 2.0]]).view()).is_err());
    }
}
