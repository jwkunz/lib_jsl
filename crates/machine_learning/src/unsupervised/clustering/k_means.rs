//! k-means clustering.

use crate::common::{Clusterer, MlRng, check_features, squared_distance};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// One Lloyd run: final centres, labels, inertia and iteration count.
type LloydRun = (Vec<Vec<f64>>, Vec<usize>, f64, usize);

/// Lloyd's k-means with k-means++ seeding: alternately assign each point to its nearest centroid
/// and move each centroid to the mean of its points, minimising the within-cluster sum of squared
/// distances (the inertia). The run is repeated `n_init` times from different seedings and the
/// lowest-inertia result kept, because the algorithm only finds a local optimum. A cluster that
/// loses all its points is re-seeded at the point farthest from its centroid.
#[derive(Debug, Clone)]
pub struct KMeans {
    k: usize,
    n_init: usize,
    max_iterations: usize,
    tolerance: f64,
    seed: u64,
    centroids: R2D,
    inertia: f64,
    iterations: usize,
}

impl KMeans {
    pub fn new(k: usize, seed: u64) -> Self {
        KMeans { k, n_init: 5, max_iterations: 300, tolerance: 1e-8, seed, centroids: R2D::zeros((0, 0)), inertia: f64::NAN, iterations: 0 }
    }

    pub fn with_restarts(mut self, n_init: usize) -> Self {
        self.n_init = n_init.max(1);
        self
    }

    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    pub fn centroids(&self) -> &R2D {
        &self.centroids
    }

    /// The sum of squared distances from each point to its centroid, for the best restart.
    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    /// Lloyd iterations used by the best restart.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The nearest centroid of each row.
    pub fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        if self.centroids.nrows() == 0 {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.centroids.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.centroids.ncols())));
        }
        let c: Vec<Vec<f64>> = self.centroids.rows().into_iter().map(|r| r.to_vec()).collect();
        Ok(x.rows().into_iter().map(|r| nearest(&r.to_vec(), &c).0).collect())
    }

    /// k-means++: the first centre uniformly, each next with probability proportional to the
    /// squared distance to the closest chosen centre.
    fn seed_centres(&self, pts: &[Vec<f64>], rng: &mut MlRng) -> Vec<Vec<f64>> {
        let mut centres = vec![pts[rng.below(pts.len())].clone()];
        let mut d2: Vec<f64> = pts.iter().map(|p| squared_distance(p, &centres[0])).collect();
        while centres.len() < self.k {
            let total: f64 = d2.iter().sum();
            let pick = if total <= 0.0 {
                rng.below(pts.len())
            } else {
                let mut target = rng.uniform() * total;
                let mut chosen = pts.len() - 1;
                for (i, &w) in d2.iter().enumerate() {
                    if target < w {
                        chosen = i;
                        break;
                    }
                    target -= w;
                }
                chosen
            };
            centres.push(pts[pick].clone());
            for (i, p) in pts.iter().enumerate() {
                d2[i] = d2[i].min(squared_distance(p, centres.last().expect("non-empty")));
            }
        }
        centres
    }

    fn lloyd(&self, pts: &[Vec<f64>], mut centres: Vec<Vec<f64>>) -> LloydRun {
        let (n, dims) = (pts.len(), pts[0].len());
        let mut labels = vec![0usize; n];
        let mut iterations = 0;
        for it in 1..=self.max_iterations {
            iterations = it;
            let mut dist = vec![0.0; n];
            for (i, p) in pts.iter().enumerate() {
                let (c, d) = nearest(p, &centres);
                labels[i] = c;
                dist[i] = d;
            }
            let mut sums = vec![vec![0.0; dims]; self.k];
            let mut counts = vec![0usize; self.k];
            for (p, &l) in pts.iter().zip(&labels) {
                counts[l] += 1;
                sums[l].iter_mut().zip(p).for_each(|(s, v)| *s += v);
            }
            let mut shift: f64 = 0.0;
            for c in 0..self.k {
                let new: Vec<f64> = if counts[c] == 0 {
                    // Re-seed an empty cluster at the worst-fitted point.
                    let far = dist.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
                    dist[far] = 0.0;
                    pts[far].clone()
                } else {
                    sums[c].iter().map(|s| s / counts[c] as f64).collect()
                };
                shift = shift.max(squared_distance(&new, &centres[c]));
                centres[c] = new;
            }
            if shift <= self.tolerance * self.tolerance {
                break;
            }
        }
        let mut inertia = 0.0;
        for (i, p) in pts.iter().enumerate() {
            let (c, d) = nearest(p, &centres);
            labels[i] = c;
            inertia += d;
        }
        (centres, labels, inertia, iterations)
    }
}

/// The nearest centre's index and squared distance.
fn nearest(p: &[f64], centres: &[Vec<f64>]) -> (usize, f64) {
    centres.iter().enumerate().map(|(i, c)| (i, squared_distance(p, c))).fold((0, f64::INFINITY), |b, c| if c.1 < b.1 { c } else { b })
}

impl Clusterer for KMeans {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL> {
        check_features(x)?;
        if self.k == 0 || self.k > x.nrows() {
            return Err(ErrorsJSL::InvalidInputRange("k must be between 1 and the number of samples."));
        }
        let pts: Vec<Vec<f64>> = x.rows().into_iter().map(|r| r.to_vec()).collect();
        let mut rng = MlRng::new(self.seed);
        let mut best: Option<LloydRun> = None;
        for _ in 0..self.n_init {
            let seeds = self.seed_centres(&pts, &mut rng);
            let run = self.lloyd(&pts, seeds);
            if best.as_ref().is_none_or(|b| run.2 < b.2) {
                best = Some(run);
            }
        }
        let (centres, labels, inertia, iterations) = best.expect("at least one restart");
        self.centroids = R2D::from_shape_fn((self.k, x.ncols()), |(i, j)| centres[i][j]);
        self.inertia = inertia;
        self.iterations = iterations;
        Ok(labels.into_iter().map(Some).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_moons};
    use crate::metrics::adjusted_rand_index;
    use ndarray::arr2;

    fn truth(l: &[usize]) -> Vec<Option<usize>> {
        l.iter().map(|&c| Some(c)).collect()
    }

    #[test]
    fn recovers_well_separated_blobs() {
        let (x, l) = make_blobs(50, &[vec![0.0, 0.0], vec![8.0, 0.0], vec![4.0, 7.0], vec![-6.0, 5.0]], 0.8, 3);
        let mut km = KMeans::new(4, 1);
        let got = km.fit_predict(x.view()).unwrap();
        assert!((adjusted_rand_index(&got, &truth(&l)).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(km.centroids().dim(), (4, 2));
        assert_eq!(km.predict(x.view()).unwrap(), got.iter().map(|g| g.unwrap()).collect::<Vec<_>>());
        assert!(km.iterations() >= 1 && km.inertia() > 0.0);
    }

    #[test]
    fn inertia_decreases_with_k_and_restarts_never_hurt() {
        let (x, _) = make_blobs(40, &[vec![0.0, 0.0], vec![6.0, 6.0], vec![0.0, 9.0]], 1.0, 5);
        let inertia = |k: usize, n_init: usize| {
            let mut km = KMeans::new(k, 2).with_restarts(n_init);
            km.fit_predict(x.view()).unwrap();
            km.inertia()
        };
        assert!(inertia(1, 3) > inertia(2, 3) && inertia(2, 3) > inertia(3, 3) && inertia(3, 3) > inertia(6, 3));
        assert!(inertia(3, 10) <= inertia(3, 1) + 1e-9);
        // k-means cannot separate interleaved moons: the grouping disagrees with the truth.
        let (mx, ml) = make_moons(100, 0.05, 1);
        let mut km = KMeans::new(2, 1);
        let got = km.fit_predict(mx.view()).unwrap();
        assert!(adjusted_rand_index(&got, &truth(&ml)).unwrap() < 0.6);
    }

    #[test]
    fn edge_cases() {
        let x = arr2(&[[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]]);
        let mut km = KMeans::new(2, 7);
        let got = km.fit_predict(x.view()).unwrap();
        assert_eq!(got[0], got[1]);
        assert_ne!(got[0], got[2]);
        assert!(km.inertia() < 1e-12);
        // As many clusters as points, with duplicates, still yields k non-empty centroids.
        let mut all = KMeans::new(4, 1);
        all.fit_predict(x.view()).unwrap();
        assert_eq!(all.centroids().nrows(), 4);
        assert!(KMeans::new(0, 1).fit_predict(x.view()).is_err());
        assert!(KMeans::new(5, 1).fit_predict(x.view()).is_err());
        assert!(KMeans::new(2, 1).predict(x.view()).is_err());
        assert!(km.predict(arr2(&[[1.0]]).view()).is_err());
    }
}
