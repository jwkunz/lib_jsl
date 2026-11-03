//! k-nearest-neighbours classification.
//!
//! See the book entry [k-nearest neighbours](https://jwkunz.github.io/lib_jsl/book/machine_learning/nearest-neighbours.html).

use crate::common::{Classifier, argmax, check_features, check_xy, class_count, squared_distance};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// Classifies a point by a vote among its `k` closest training points (Euclidean distance). The
/// model is "lazy": fitting just stores the data. With `distance_weighted` each neighbour's vote
/// counts `1/distance` so nearer points matter more (an exact match decides alone). Prediction is
/// brute force, `O(n·d)` per query; use a k-d tree from the geometry crate for large 2D/3D sets.
#[derive(Debug, Clone)]
pub struct KNearestNeighbors {
    k: usize,
    distance_weighted: bool,
    train: R2D,
    labels: Vec<usize>,
    classes: usize,
}

impl KNearestNeighbors {
    pub fn new(k: usize) -> Self {
        KNearestNeighbors { k, distance_weighted: false, train: R2D::zeros((0, 0)), labels: Vec::new(), classes: 0 }
    }

    pub fn distance_weighted(mut self, weighted: bool) -> Self {
        self.distance_weighted = weighted;
        self
    }

    fn check_fitted(&self, x: VR2D) -> Result<(), ErrorsJSL> {
        if self.labels.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.train.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.train.ncols())));
        }
        Ok(())
    }

    // ANCHOR: knn
    /// The `(index, distance)` of the `k` nearest training points to `query`, nearest first.
    pub fn neighbours(&self, query: &[f64]) -> Vec<(usize, f64)> {
        let mut d: Vec<(usize, f64)> = self.train.rows().into_iter().enumerate().map(|(i, r)| (i, squared_distance(&r.to_vec(), query))).collect();
        if d.is_empty() || self.k == 0 {
            return Vec::new();
        }
        let k = self.k.min(d.len());
        d.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        d.truncate(k);
        d.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        d.into_iter().map(|(i, d2)| (i, d2.sqrt())).collect()
    }

    /// Class probabilities (vote shares) for each row of `x`, as a `(rows, classes)` matrix.
    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        self.check_fitted(x)?;
        let mut out = R2D::zeros((x.nrows(), self.classes));
        for (i, row) in x.rows().into_iter().enumerate() {
            let near = self.neighbours(&row.to_vec());
            let exact: Vec<&(usize, f64)> = near.iter().filter(|n| n.1 == 0.0).collect();
            let votes: Vec<(usize, f64)> = if self.distance_weighted && !exact.is_empty() {
                exact.iter().map(|n| (n.0, 1.0)).collect()
            } else {
                near.iter().map(|&(j, d)| (j, if self.distance_weighted { 1.0 / d } else { 1.0 })).collect()
            };
            let total: f64 = votes.iter().map(|v| v.1).sum();
            for (j, w) in votes {
                out[[i, self.labels[j]]] += w / total;
            }
        }
        Ok(out)
    }
    // ANCHOR_END: knn
}

impl Classifier for KNearestNeighbors {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        if self.k == 0 {
            return Err(ErrorsJSL::InvalidInputRange("k must be at least 1."));
        }
        self.train = x.to_owned();
        self.labels = labels.to_vec();
        self.classes = class_count(labels);
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        let proba = self.predict_proba(x)?;
        Ok(proba.rows().into_iter().map(|r| argmax(&r.to_vec()).unwrap_or(0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_moons};
    use crate::metrics::accuracy;
    use crate::preprocessing::{select_labels, select_rows, train_test_split};
    use ndarray::arr2;

    #[test]
    fn separates_blobs_and_handles_nonlinear_moons() {
        let (x, l) = make_moons(150, 0.15, 3);
        let (tr, te) = train_test_split(x.nrows(), 0.3, 1).unwrap();
        let mut m = KNearestNeighbors::new(5);
        m.fit(select_rows(x.view(), &tr).view(), &select_labels(&l, &tr)).unwrap();
        let acc = accuracy(&select_labels(&l, &te), &m.predict(select_rows(x.view(), &te).view()).unwrap()).unwrap();
        assert!(acc > 0.93, "{acc}");
        let (bx, bl) = make_blobs(30, &[vec![0.0, 0.0], vec![6.0, 6.0], vec![-6.0, 6.0]], 1.0, 2);
        let mut b = KNearestNeighbors::new(3);
        b.fit(bx.view(), &bl).unwrap();
        assert_eq!(accuracy(&bl, &b.predict(bx.view()).unwrap()).unwrap(), 1.0);
    }

    #[test]
    fn k_and_weighting_change_the_vote() {
        // One class-1 point right next to the query, three class-0 points farther away.
        let x = arr2(&[[0.1, 0.0], [2.0, 0.0], [2.1, 0.0], [2.2, 0.0]]);
        let l = [1, 0, 0, 0];
        let mut plain = KNearestNeighbors::new(4);
        plain.fit(x.view(), &l).unwrap();
        assert_eq!(plain.predict(arr2(&[[0.0, 0.0]]).view()).unwrap(), vec![0]);
        let mut weighted = KNearestNeighbors::new(4).distance_weighted(true);
        weighted.fit(x.view(), &l).unwrap();
        assert_eq!(weighted.predict(arr2(&[[0.0, 0.0]]).view()).unwrap(), vec![1]);
        let mut one = KNearestNeighbors::new(1);
        one.fit(x.view(), &l).unwrap();
        assert_eq!(one.predict(arr2(&[[0.0, 0.0]]).view()).unwrap(), vec![1]);
        let p = plain.predict_proba(arr2(&[[0.0, 0.0]]).view()).unwrap();
        assert_eq!((p[[0, 0]], p[[0, 1]]), (0.75, 0.25));
        // An exact hit decides alone under distance weighting.
        assert_eq!(weighted.predict(arr2(&[[2.0, 0.0]]).view()).unwrap(), vec![0]);
        assert_eq!(one.neighbours(&[0.0, 0.0]).len(), 1);
        assert_eq!(KNearestNeighbors::new(10).neighbours(&[0.0, 0.0]).len(), 0); // unfitted: no data
    }

    #[test]
    fn errors() {
        let x = arr2(&[[0.0], [1.0]]);
        assert!(KNearestNeighbors::new(0).fit(x.view(), &[0, 1]).is_err());
        assert!(KNearestNeighbors::new(1).fit(x.view(), &[0]).is_err());
        assert!(KNearestNeighbors::new(1).predict(x.view()).is_err());
        let mut m = KNearestNeighbors::new(5);
        m.fit(x.view(), &[0, 1]).unwrap(); // k larger than the data is clamped
        assert_eq!(m.predict(x.view()).unwrap().len(), 2);
        assert!(m.predict(arr2(&[[1.0, 2.0]]).view()).is_err());
    }
}
