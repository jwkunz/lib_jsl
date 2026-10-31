//! DBSCAN density-based clustering.

use crate::common::{Clusterer, check_features, pairwise_distances};
use lib_jsl_core::{ErrorsJSL, VR2D};

/// Density-Based Spatial Clustering of Applications with Noise. A point is a *core* point if at
/// least `min_samples` points (itself included) lie within `eps` of it; clusters are the
/// connected regions of core points together with the *border* points within `eps` of a core
/// point, and everything else is noise (`None`). Unlike k-means it needs no cluster count, finds
/// arbitrarily shaped clusters and flags outliers, but a single `eps` cannot suit clusters of
/// very different density (see HDBSCAN).
#[derive(Debug, Clone)]
pub struct Dbscan {
    eps: f64,
    min_samples: usize,
    core_points: Vec<usize>,
}

impl Dbscan {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Dbscan { eps, min_samples, core_points: Vec::new() }
    }

    /// Indices of the core points found by the last fit.
    pub fn core_points(&self) -> &[usize] {
        &self.core_points
    }
}

impl Clusterer for Dbscan {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL> {
        check_features(x)?;
        if self.eps.is_nan() || self.eps <= 0.0 || self.min_samples == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Need eps > 0 and min_samples >= 1."));
        }
        let n = x.nrows();
        let d = pairwise_distances(x);
        let neighbours: Vec<Vec<usize>> = (0..n).map(|i| (0..n).filter(|&j| d[[i, j]] <= self.eps).collect()).collect();
        let is_core: Vec<bool> = neighbours.iter().map(|nb| nb.len() >= self.min_samples).collect();
        let mut labels: Vec<Option<usize>> = vec![None; n];
        let mut next = 0;
        for start in (0..n).filter(|&i| is_core[i]) {
            if labels[start].is_some() {
                continue;
            }
            labels[start] = Some(next);
            let mut queue = vec![start];
            while let Some(p) = queue.pop() {
                for &q in &neighbours[p] {
                    if labels[q].is_none() {
                        labels[q] = Some(next);
                        if is_core[q] {
                            queue.push(q); // only core points propagate the cluster
                        }
                    }
                }
            }
            next += 1;
        }
        self.core_points = (0..n).filter(|&i| is_core[i]).collect();
        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_circles, make_moons};
    use crate::metrics::adjusted_rand_index;
    use ndarray::{Axis, arr2, concatenate};

    #[test]
    fn separates_nonconvex_shapes_that_defeat_k_means() {
        let (x, l) = make_moons(150, 0.05, 1);
        let got = Dbscan::new(0.2, 5).fit_predict(x.view()).unwrap();
        let truth: Vec<Option<usize>> = l.iter().map(|&c| Some(c)).collect();
        assert!(adjusted_rand_index(&got, &truth).unwrap() > 0.97);
        let (cx, cl) = make_circles(120, 0.03, 0.4, 2);
        let got = Dbscan::new(0.15, 4).fit_predict(cx.view()).unwrap();
        assert!(adjusted_rand_index(&got, &cl.iter().map(|&c| Some(c)).collect::<Vec<_>>()).unwrap() > 0.95);
    }

    #[test]
    fn noise_border_and_core_points_are_distinguished() {
        // A dense line of five points, one border point near the end, and one far outlier.
        let x = arr2(&[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0], [2.9, 0.0], [50.0, 50.0]]);
        let mut db = Dbscan::new(0.6, 3);
        let labels = db.fit_predict(x.view()).unwrap();
        assert!(labels[..5].iter().all(|l| *l == Some(0)));
        assert_eq!(labels[5], None); // 0.9 from the nearest point: out of reach
        assert_eq!(labels[6], None);
        assert_eq!(db.core_points(), &[1, 2, 3]);
        // Raising eps lets the straggler join as a border point (it is not core itself).
        let mut wide = Dbscan::new(1.0, 3);
        let l2 = wide.fit_predict(x.view()).unwrap();
        assert_eq!(l2[5], Some(0));
        assert!(!wide.core_points().contains(&5));
    }

    #[test]
    fn parameters_control_the_outcome() {
        let (a, _) = make_blobs(40, &[vec![0.0, 0.0], vec![6.0, 0.0]], 0.4, 3);
        let tiny = Dbscan::new(0.001, 3).fit_predict(a.view()).unwrap();
        assert!(tiny.iter().all(Option::is_none)); // eps too small: everything is noise
        let huge = Dbscan::new(100.0, 3).fit_predict(a.view()).unwrap();
        assert!(huge.iter().all(|l| *l == Some(0))); // eps too large: one blob
        let with_outliers = concatenate(Axis(0), &[a.view(), arr2(&[[20.0, 20.0], [-20.0, 15.0]]).view()]).unwrap();
        let l = Dbscan::new(0.8, 4).fit_predict(with_outliers.view()).unwrap();
        assert_eq!(l.iter().flatten().collect::<std::collections::HashSet<_>>().len(), 2);
        assert!(l[80].is_none() && l[81].is_none());
        assert!(Dbscan::new(0.0, 3).fit_predict(a.view()).is_err());
        assert!(Dbscan::new(1.0, 0).fit_predict(a.view()).is_err());
    }
}
