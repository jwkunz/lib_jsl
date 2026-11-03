//! k-medoids clustering (Partitioning Around Medoids).
//!
//! See the book entry [k-means and k-medoids](https://jwkunz.github.io/lib_jsl/book/unsupervised/k-means.html#k-medoids).

use crate::common::{Clusterer, check_features, squared_distance};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// PAM: like k-means, but each cluster is represented by an actual data point (its medoid)
/// instead of a mean, and the objective is the total *distance* (not squared), which makes the
/// result robust to outliers and lets any dissimilarity be used (see [`KMedoids::fit_distances`]).
/// A greedy BUILD phase picks the initial medoids, then SWAP repeatedly exchanges a medoid for a
/// non-medoid whenever that lowers the total cost.
#[derive(Debug, Clone)]
pub struct KMedoids {
    k: usize,
    max_iterations: usize,
    medoids: Vec<usize>,
    cost: f64,
    labels: Vec<usize>,
}

impl KMedoids {
    pub fn new(k: usize) -> Self {
        KMedoids { k, max_iterations: 100, medoids: Vec::new(), cost: f64::NAN, labels: Vec::new() }
    }

    /// Indices (into the training data) of the chosen medoids, one per cluster.
    pub fn medoids(&self) -> &[usize] {
        &self.medoids
    }

    /// The total distance from every point to its medoid.
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Clusters from a precomputed symmetric dissimilarity matrix; returns cluster labels.
    pub fn fit_distances(&mut self, d: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        let n = d.nrows();
        if d.ncols() != n || n == 0 || self.k == 0 || self.k > n {
            return Err(ErrorsJSL::InvalidInputRange("Need a square distance matrix and 1 <= k <= n."));
        }
        if d.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Distances must be finite and non-negative."));
        }
        // ANCHOR: pam
        // BUILD: first the point with the smallest total distance, then greedily the point that
        // most reduces the total cost.
        let mut medoids = vec![(0..n).min_by(|&a, &b| d.row(a).sum().total_cmp(&d.row(b).sum())).expect("n > 0")];
        let mut nearest: Vec<f64> = (0..n).map(|i| d[[i, medoids[0]]]).collect();
        while medoids.len() < self.k {
            let gain = |c: usize| (0..n).map(|i| (nearest[i] - d[[i, c]]).max(0.0)).sum::<f64>();
            let c = (0..n).filter(|c| !medoids.contains(c)).max_by(|&a, &b| gain(a).total_cmp(&gain(b))).expect("k <= n");
            medoids.push(c);
            for i in 0..n {
                nearest[i] = nearest[i].min(d[[i, c]]);
            }
        }
        let total = |m: &[usize]| (0..n).map(|i| m.iter().map(|&c| d[[i, c]]).fold(f64::INFINITY, f64::min)).sum::<f64>();
        let mut cost = total(&medoids);
        // SWAP until no exchange improves the cost.
        for _ in 0..self.max_iterations {
            let mut best = (cost, None);
            for slot in 0..self.k {
                for candidate in (0..n).filter(|c| !medoids.contains(c)) {
                    let mut trial = medoids.clone();
                    trial[slot] = candidate;
                    let c = total(&trial);
                    if c < best.0 - 1e-12 {
                        best = (c, Some((slot, candidate)));
                    }
                }
            }
            match best.1 {
                Some((slot, candidate)) => {
                    medoids[slot] = candidate;
                    cost = best.0;
                }
                None => break,
            }
        }
        // ANCHOR_END: pam
        self.labels = (0..n).map(|i| (0..self.k).min_by(|&a, &b| d[[i, medoids[a]]].total_cmp(&d[[i, medoids[b]]])).expect("k > 0")).collect();
        self.medoids = medoids;
        self.cost = cost;
        Ok(self.labels.clone())
    }
}

impl Clusterer for KMedoids {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL> {
        check_features(x)?;
        let rows: Vec<Vec<f64>> = x.rows().into_iter().map(|r| r.to_vec()).collect();
        let d = R2D::from_shape_fn((rows.len(), rows.len()), |(i, j)| squared_distance(&rows[i], &rows[j]).sqrt());
        Ok(self.fit_distances(d.view())?.into_iter().map(Some).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_blobs;
    use crate::metrics::adjusted_rand_index;
    use ndarray::{Axis, arr2, concatenate};

    #[test]
    fn recovers_blobs_and_returns_real_data_points_as_medoids() {
        let (x, l) = make_blobs(30, &[vec![0.0, 0.0], vec![7.0, 0.0], vec![3.0, 6.0]], 0.7, 2);
        let mut km = KMedoids::new(3);
        let got = km.fit_predict(x.view()).unwrap();
        let truth: Vec<Option<usize>> = l.iter().map(|&c| Some(c)).collect();
        assert!((adjusted_rand_index(&got, &truth).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(km.medoids().len(), 3);
        assert!(km.medoids().iter().all(|&m| m < x.nrows()));
        assert!(km.cost() > 0.0);
    }

    #[test]
    fn robust_to_an_outlier_where_means_are_pulled() {
        let (cluster, _) = make_blobs(40, &[vec![0.0, 0.0]], 0.5, 1);
        let far = arr2(&[[100.0, 100.0]]);
        let x = concatenate(Axis(0), &[cluster.view(), far.view()]).unwrap();
        let mut km = KMedoids::new(1);
        km.fit_predict(x.view()).unwrap();
        let m = km.medoids()[0];
        assert!(x[[m, 0]].abs() < 1.0, "medoid should stay in the cluster, got {:?}", x.row(m));
        // The mean of all points is dragged far from the cluster.
        assert!(x.column(0).sum() / 41.0 > 2.0);
    }

    #[test]
    fn works_with_a_custom_dissimilarity_and_validates() {
        // Two groups defined purely by a distance matrix (no coordinates).
        let mut d = R2D::from_elem((6, 6), 10.0);
        for i in 0..6 {
            d[[i, i]] = 0.0;
        }
        for g in [0, 3] {
            for i in g..g + 3 {
                for j in g..g + 3 {
                    if i != j {
                        d[[i, j]] = 1.0;
                    }
                }
            }
        }
        let labels = KMedoids::new(2).fit_distances(d.view()).unwrap();
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
        assert!(KMedoids::new(0).fit_distances(d.view()).is_err());
        assert!(KMedoids::new(7).fit_distances(d.view()).is_err());
        assert!(KMedoids::new(2).fit_distances(arr2(&[[0.0, -1.0], [-1.0, 0.0]]).view()).is_err());
        assert!(KMedoids::new(2).fit_distances(R2D::zeros((2, 3)).view()).is_err());
    }
}
