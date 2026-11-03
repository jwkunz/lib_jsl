//! Agglomerative hierarchical clustering and dendrograms.
//!
//! See the book entry [Hierarchical clustering and phylogenetic trees](https://jwkunz.github.io/lib_jsl/book/unsupervised/hierarchical.html).

use crate::common::{Clusterer, check_features, pairwise_distances};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// How the distance between two clusters is defined from the distances between their points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Closest pair: finds elongated, chained clusters, sensitive to noise bridges.
    Single,
    /// Farthest pair: compact, similar-diameter clusters.
    Complete,
    /// Mean of all pairs (UPGMA): a compromise between single and complete.
    Average,
    /// Minimum increase in within-cluster variance (Euclidean data only); tends to give compact,
    /// evenly sized clusters.
    Ward,
}

/// One merge of two clusters into a new one. Cluster ids `0..n` are the original points; the
/// merge at position `i` creates cluster `n + i` (the same convention as SciPy's linkage matrix).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Merge {
    pub a: usize,
    pub b: usize,
    pub distance: f64,
    pub size: usize,
}

/// The full merge tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Dendrogram {
    pub points: usize,
    pub merges: Vec<Merge>,
}

/// Builds the dendrogram from a symmetric distance matrix with the Lance-Williams update, in
/// `O(n³)` time and `O(n²)` memory. For [`Linkage::Ward`] the distances must be Euclidean.
#[allow(clippy::needless_range_loop)]
pub fn agglomerate(distances: VR2D, linkage: Linkage) -> Result<Dendrogram, ErrorsJSL> {
    let n = distances.nrows();
    if n == 0 || distances.ncols() != n {
        return Err(ErrorsJSL::InvalidInputRange("Need a non-empty square distance matrix."));
    }
    if distances.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Distances must be finite and non-negative."));
    }
    let mut d: Vec<Vec<f64>> = distances.rows().into_iter().map(|r| r.iter().map(|&v| if linkage == Linkage::Ward { v * v } else { v }).collect()).collect();
    let mut ids: Vec<usize> = (0..n).collect(); // cluster id of each active row
    let mut sizes = vec![1usize; n];
    let mut merges = Vec::with_capacity(n.saturating_sub(1));
    // ANCHOR: lance_williams
    while ids.len() > 1 {
        let m = ids.len();
        let (mut bi, mut bj, mut best) = (0, 1, f64::INFINITY);
        for i in 0..m {
            for j in i + 1..m {
                if d[i][j] < best {
                    (bi, bj, best) = (i, j, d[i][j]);
                }
            }
        }
        let (si, sj) = (sizes[bi] as f64, sizes[bj] as f64);
        let new_row: Vec<f64> = (0..m)
            .map(|k| {
                let sk = sizes[k] as f64;
                match linkage {
                    Linkage::Single => d[bi][k].min(d[bj][k]),
                    Linkage::Complete => d[bi][k].max(d[bj][k]),
                    Linkage::Average => (si * d[bi][k] + sj * d[bj][k]) / (si + sj),
                    Linkage::Ward => ((si + sk) * d[bi][k] + (sj + sk) * d[bj][k] - sk * d[bi][bj]) / (si + sj + sk),
                }
            })
            .collect();
        merges.push(Merge { a: ids[bi], b: ids[bj], distance: if linkage == Linkage::Ward { best.sqrt() } else { best }, size: sizes[bi] + sizes[bj] });
        for k in 0..m {
            d[bi][k] = new_row[k];
            d[k][bi] = new_row[k];
        }
        d[bi][bi] = 0.0;
        d.remove(bj);
        d.iter_mut().for_each(|r| {
            r.remove(bj);
        });
        sizes[bi] += sizes[bj];
        sizes.remove(bj);
        ids[bi] = n + merges.len() - 1;
        ids.remove(bj);
    }
    // ANCHOR_END: lance_williams
    Ok(Dendrogram { points: n, merges })
}

impl Dendrogram {
    /// Cuts the tree into exactly `k` clusters (`1 <= k <= points`), returning a label per point.
    pub fn cut_clusters(&self, k: usize) -> Result<Vec<usize>, ErrorsJSL> {
        if k == 0 || k > self.points {
            return Err(ErrorsJSL::InvalidInputRange("k must be between 1 and the number of points."));
        }
        Ok(self.labels_after(self.points - k))
    }

    /// Cuts the tree at `height`: merges at distance `<= height` are applied.
    pub fn cut_distance(&self, height: f64) -> Vec<usize> {
        self.labels_after(self.merges.iter().take_while(|m| m.distance <= height).count())
    }

    /// Labels after applying the first `applied` merges.
    fn labels_after(&self, applied: usize) -> Vec<usize> {
        let mut parent: Vec<usize> = (0..self.points + self.merges.len()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for (i, m) in self.merges.iter().take(applied).enumerate() {
            let new = self.points + i;
            let (ra, rb) = (find(&mut parent, m.a), find(&mut parent, m.b));
            parent[ra] = new;
            parent[rb] = new;
        }
        let mut label_of = std::collections::HashMap::new();
        (0..self.points)
            .map(|p| {
                let root = find(&mut parent, p);
                let next = label_of.len();
                *label_of.entry(root).or_insert(next)
            })
            .collect()
    }

    /// The cophenetic distance matrix: the height at which each pair of points first joins.
    pub fn cophenetic_distances(&self) -> R2D {
        let mut members: Vec<Vec<usize>> = (0..self.points).map(|p| vec![p]).collect();
        let mut out = R2D::zeros((self.points, self.points));
        for m in &self.merges {
            for &i in &members[m.a] {
                for &j in &members[m.b] {
                    out[[i, j]] = m.distance;
                    out[[j, i]] = m.distance;
                }
            }
            let joined: Vec<usize> = members[m.a].iter().chain(&members[m.b]).copied().collect();
            members.push(joined);
        }
        out
    }
}

/// Agglomerative clustering of points into `k` clusters by cutting the dendrogram.
#[derive(Debug, Clone)]
pub struct HierarchicalClustering {
    k: usize,
    linkage: Linkage,
    dendrogram: Option<Dendrogram>,
}

impl HierarchicalClustering {
    pub fn new(k: usize, linkage: Linkage) -> Self {
        HierarchicalClustering { k, linkage, dendrogram: None }
    }

    /// The dendrogram from the last fit.
    pub fn dendrogram(&self) -> Option<&Dendrogram> {
        self.dendrogram.as_ref()
    }
}

impl Clusterer for HierarchicalClustering {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL> {
        check_features(x)?;
        let tree = agglomerate(pairwise_distances(x).view(), self.linkage)?;
        let labels = tree.cut_clusters(self.k)?;
        self.dendrogram = Some(tree);
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
    fn small_worked_example_matches_hand_computation() {
        // Points on a line at 0, 1, 5, 6, 20.
        let x = arr2(&[[0.0], [1.0], [5.0], [6.0], [20.0]]);
        let single = agglomerate(pairwise_distances(x.view()).view(), Linkage::Single).unwrap();
        let heights: Vec<f64> = single.merges.iter().map(|m| m.distance).collect();
        assert_eq!(heights, vec![1.0, 1.0, 4.0, 14.0]);
        let complete = agglomerate(pairwise_distances(x.view()).view(), Linkage::Complete).unwrap();
        assert_eq!(complete.merges.iter().map(|m| m.distance).collect::<Vec<_>>(), vec![1.0, 1.0, 6.0, 20.0]);
        let average = agglomerate(pairwise_distances(x.view()).view(), Linkage::Average).unwrap();
        assert!((average.merges[2].distance - 5.0).abs() < 1e-12 && (average.merges[3].distance - (20.0 + 19.0 + 15.0 + 14.0) / 4.0).abs() < 1e-12);
        assert_eq!(single.merges[3].size, 5);
        assert_eq!(single.cut_clusters(3).unwrap(), vec![0, 0, 1, 1, 2]);
        assert_eq!(single.cut_distance(1.5), vec![0, 0, 1, 1, 2]);
        assert_eq!(single.cut_clusters(1).unwrap(), vec![0; 5]);
        assert_eq!(single.cut_clusters(5).unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn ward_matches_known_height_and_merge_heights_are_monotone() {
        let x = arr2(&[[0.0, 0.0], [2.0, 0.0]]);
        let w = agglomerate(pairwise_distances(x.view()).view(), Linkage::Ward).unwrap();
        assert!((w.merges[0].distance - 2.0).abs() < 1e-12); // sqrt(2·ΔSS) with ΔSS = 2
        let (b, _) = make_blobs(20, &[vec![0.0, 0.0], vec![5.0, 5.0], vec![0.0, 8.0]], 1.0, 4);
        for linkage in [Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward] {
            let t = agglomerate(pairwise_distances(b.view()).view(), linkage).unwrap();
            assert!(t.merges.windows(2).all(|p| p[1].distance >= p[0].distance - 1e-9), "{linkage:?}");
            // Cophenetic distances are ultrametric.
            let c = t.cophenetic_distances();
            assert!((0..5).all(|i| (0..5).all(|j| (0..5).all(|k| c[[i, j]] <= c[[i, k]].max(c[[k, j]]) + 1e-9))));
        }
    }

    #[test]
    fn linkage_choice_matters_single_follows_chains_complete_does_not() {
        let (x, l) = make_moons(80, 0.05, 2);
        let ari = |lk| {
            let mut h = HierarchicalClustering::new(2, lk);
            adjusted_rand_index(&h.fit_predict(x.view()).unwrap(), &truth(&l)).unwrap()
        };
        assert!(ari(Linkage::Single) > 0.99);
        assert!(ari(Linkage::Complete) < 0.7);
        let (bx, bl) = make_blobs(30, &[vec![0.0, 0.0], vec![9.0, 0.0], vec![4.0, 8.0]], 0.8, 3);
        for lk in [Linkage::Complete, Linkage::Average, Linkage::Ward] {
            let mut h = HierarchicalClustering::new(3, lk);
            assert!((adjusted_rand_index(&h.fit_predict(bx.view()).unwrap(), &truth(&bl)).unwrap() - 1.0).abs() < 1e-12, "{lk:?}");
        }
    }

    #[test]
    fn errors() {
        assert!(agglomerate(R2D::zeros((0, 0)).view(), Linkage::Single).is_err());
        assert!(agglomerate(R2D::zeros((2, 3)).view(), Linkage::Single).is_err());
        assert!(agglomerate(arr2(&[[0.0, -1.0], [-1.0, 0.0]]).view(), Linkage::Single).is_err());
        let one = agglomerate(arr2(&[[0.0]]).view(), Linkage::Single).unwrap();
        assert!(one.merges.is_empty() && one.cut_clusters(1).unwrap() == vec![0]);
        assert!(one.cut_clusters(2).is_err() && one.cut_clusters(0).is_err());
        assert!(HierarchicalClustering::new(5, Linkage::Ward).fit_predict(arr2(&[[0.0], [1.0]]).view()).is_err());
    }
}
