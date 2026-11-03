//! Spectral clustering.
//!
//! See the book entry [Spectral clustering](https://jwkunz.github.io/lib_jsl/book/unsupervised/spectral.html).

use crate::common::{Clusterer, check_features, pairwise_distances};
use crate::unsupervised::clustering::k_means::KMeans;
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};
use lib_jsl_linalg::{Eigh, UPLO};

/// How the similarity graph between points is built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Affinity {
    /// A dense graph with weights `exp(-γ d²)`.
    Rbf { gamma: f64 },
    /// A symmetrised `k`-nearest-neighbour graph with unit weights: good for manifold-shaped
    /// clusters (moons, rings) because it only connects genuinely nearby points.
    NearestNeighbors { k: usize },
}

/// Normalised spectral clustering (Ng, Jordan and Weiss). The points become the vertices of a
/// similarity graph; the `k` eigenvectors of the normalised Laplacian `I - D^{-1/2} W D^{-1/2}`
/// with the smallest eigenvalues embed the points so that connected regions of the graph become
/// compact, well-separated groups, which k-means then splits. This clusters by *connectivity*, so
/// it separates shapes (rings, moons) that centroid methods cannot.
#[derive(Debug, Clone)]
pub struct SpectralClustering {
    k: usize,
    affinity: Affinity,
    seed: u64,
    eigenvalues: Vec<f64>,
}

impl SpectralClustering {
    pub fn new(k: usize, affinity: Affinity, seed: u64) -> Self {
        SpectralClustering { k, affinity, seed, eigenvalues: Vec::new() }
    }

    /// The `k` smallest Laplacian eigenvalues from the last fit. A cluster count `k` is natural
    /// where the eigenvalues jump (the eigengap).
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    fn affinity_matrix(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        let n = x.nrows();
        let d = pairwise_distances(x);
        match self.affinity {
            Affinity::Rbf { gamma } => {
                if gamma.is_nan() || gamma <= 0.0 {
                    return Err(ErrorsJSL::InvalidInputRange("gamma must be positive."));
                }
                Ok(R2D::from_shape_fn((n, n), |(i, j)| if i == j { 0.0 } else { (-gamma * d[[i, j]] * d[[i, j]]).exp() }))
            }
            Affinity::NearestNeighbors { k } => {
                if k == 0 || k >= n {
                    return Err(ErrorsJSL::InvalidInputRange("Need 1 <= neighbours < number of samples."));
                }
                let mut w = R2D::zeros((n, n));
                for i in 0..n {
                    let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                    order.sort_by(|&a, &b| d[[i, a]].total_cmp(&d[[i, b]]));
                    for &j in order.iter().take(k) {
                        w[[i, j]] = 1.0;
                        w[[j, i]] = 1.0;
                    }
                }
                Ok(w)
            }
        }
    }
}

impl Clusterer for SpectralClustering {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL> {
        check_features(x)?;
        let n = x.nrows();
        if self.k == 0 || self.k > n {
            return Err(ErrorsJSL::InvalidInputRange("k must be between 1 and the number of samples."));
        }
        let w = self.affinity_matrix(x)?;
        // ANCHOR: spectral
        let deg: Vec<f64> = (0..n).map(|i| w.row(i).sum()).collect();
        let inv_sqrt: Vec<f64> = deg.iter().map(|&d| if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 }).collect();
        let lap = R2D::from_shape_fn((n, n), |(i, j)| f64::from(i == j) - inv_sqrt[i] * w[[i, j]] * inv_sqrt[j]);
        let (vals, vecs) = lap.eigh(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("The Laplacian eigendecomposition failed."))?;
        self.eigenvalues = vals.iter().take(self.k).copied().collect();
        // Embed with the k smallest eigenvectors, normalising each row to unit length.
        let mut embedding = R2D::from_shape_fn((n, self.k), |(i, j)| vecs[[i, j]]);
        for mut row in embedding.rows_mut() {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|v| v / norm);
            }
        }
        KMeans::new(self.k, self.seed).with_restarts(10).fit_predict(embedding.view())
        // ANCHOR_END: spectral
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_circles, make_moons};
    use crate::metrics::adjusted_rand_index;

    fn truth(l: &[usize]) -> Vec<Option<usize>> {
        l.iter().map(|&c| Some(c)).collect()
    }

    #[test]
    fn separates_concentric_rings_and_moons() {
        let (c, cl) = make_circles(100, 0.04, 0.4, 1);
        let got = SpectralClustering::new(2, Affinity::NearestNeighbors { k: 8 }, 1).fit_predict(c.view()).unwrap();
        assert!(adjusted_rand_index(&got, &truth(&cl)).unwrap() > 0.98);
        let (m, ml) = make_moons(100, 0.06, 2);
        let got = SpectralClustering::new(2, Affinity::NearestNeighbors { k: 10 }, 1).fit_predict(m.view()).unwrap();
        assert!(adjusted_rand_index(&got, &truth(&ml)).unwrap() > 0.98);
        let rbf = SpectralClustering::new(2, Affinity::Rbf { gamma: 25.0 }, 1).fit_predict(m.view()).unwrap();
        assert!(adjusted_rand_index(&rbf, &truth(&ml)).unwrap() > 0.9);
    }

    #[test]
    fn eigengap_reveals_the_cluster_count() {
        let (x, l) = make_blobs(40, &[vec![0.0, 0.0], vec![10.0, 0.0], vec![5.0, 9.0]], 0.5, 3);
        let mut sc = SpectralClustering::new(4, Affinity::NearestNeighbors { k: 6 }, 1);
        sc.fit_predict(x.view()).unwrap();
        let ev = sc.eigenvalues();
        assert!(ev[0].abs() < 1e-6 && ev[1] < 0.05 && ev[2] < 0.05, "{ev:?}");
        assert!(ev[3] > 10.0 * ev[2].max(1e-3), "gap after the third eigenvalue: {ev:?}");
        let got = SpectralClustering::new(3, Affinity::NearestNeighbors { k: 6 }, 1).fit_predict(x.view()).unwrap();
        assert!((adjusted_rand_index(&got, &truth(&l)).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn errors() {
        let (x, _) = make_blobs(5, &[vec![0.0, 0.0]], 1.0, 1);
        assert!(SpectralClustering::new(0, Affinity::Rbf { gamma: 1.0 }, 1).fit_predict(x.view()).is_err());
        assert!(SpectralClustering::new(2, Affinity::Rbf { gamma: 0.0 }, 1).fit_predict(x.view()).is_err());
        assert!(SpectralClustering::new(2, Affinity::NearestNeighbors { k: 0 }, 1).fit_predict(x.view()).is_err());
        assert!(SpectralClustering::new(2, Affinity::NearestNeighbors { k: 5 }, 1).fit_predict(x.view()).is_err());
        assert_eq!(SpectralClustering::new(1, Affinity::Rbf { gamma: 1.0 }, 1).fit_predict(x.view()).unwrap(), vec![Some(0); 5]);
    }
}
