//! HDBSCAN hierarchical density-based clustering.
//!
//! See the book entry [Density-based clustering: DBSCAN and HDBSCAN](https://jwkunz.github.io/lib_jsl/book/unsupervised/density.html#hdbscan-every-density-at-once).

use crate::common::{Clusterer, check_features, pairwise_distances};
use lib_jsl_core::{ErrorsJSL, VR2D};

/// Hierarchical DBSCAN (Campello, Moulavi and Sander). Instead of one density threshold `eps` it
/// considers *all* of them at once:
///
/// 1. the *mutual reachability* distance `max(core(a), core(b), d(a, b))` (with `core` the
///    distance to the `min_samples`-th nearest neighbour) spreads sparse points apart;
/// 2. a minimum spanning tree of that metric gives the single-linkage hierarchy;
/// 3. the hierarchy is condensed: a split only counts if both parts have at least
///    `min_cluster_size` points, smaller pieces are just points falling out of their cluster;
/// 4. each condensed cluster's *stability* `Σ (λ_leave - λ_birth)` (with `λ = 1/distance`) measures
///    how long and how densely it persists, and the set of non-overlapping clusters with the
///    greatest total stability is chosen ("excess of mass").
///
/// The result finds clusters of different densities without choosing `eps`, and labels the rest as
/// noise (`None`). The implementation uses the dense `O(n²)` distance matrix.
#[derive(Debug, Clone)]
pub struct Hdbscan {
    min_cluster_size: usize,
    min_samples: usize,
    stabilities: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CondensedCluster {
    parent: Option<usize>,
    birth_lambda: f64,
    children: Vec<usize>,
    stability: f64,
    /// Points that fell out of this cluster, with the λ at which they did.
    fallen: Vec<usize>,
}

const LAMBDA_CAP: f64 = 1e12;

fn lambda_of(distance: f64) -> f64 {
    if distance > 0.0 { (1.0 / distance).min(LAMBDA_CAP) } else { LAMBDA_CAP }
}

impl Hdbscan {
    pub fn new(min_cluster_size: usize) -> Self {
        Hdbscan { min_cluster_size, min_samples: min_cluster_size, stabilities: Vec::new() }
    }

    /// Sets `min_samples`, the neighbour count defining local density (default: `min_cluster_size`);
    /// larger values make the result more conservative about calling points part of clusters.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// The stability of each selected cluster, in label order.
    pub fn cluster_stabilities(&self) -> &[f64] {
        &self.stabilities
    }
}

impl Clusterer for Hdbscan {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL> {
        check_features(x)?;
        let n = x.nrows();
        if self.min_cluster_size < 2 || self.min_samples == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Need min_cluster_size >= 2 and min_samples >= 1."));
        }
        if n < self.min_cluster_size {
            self.stabilities.clear();
            return Ok(vec![None; n]);
        }
        let d = pairwise_distances(x);
        // ANCHOR: mutual_reachability
        // 1. Core distances and mutual reachability.
        let core: Vec<f64> = (0..n)
            .map(|i| {
                let mut row = d.row(i).to_vec();
                row.sort_by(f64::total_cmp);
                row[(self.min_samples - 1).min(n - 1)]
            })
            .collect();
        let mr = |i: usize, j: usize| d[[i, j]].max(core[i]).max(core[j]);
        // 2. Minimum spanning tree (Prim, O(n²)).
        let mut in_tree = vec![false; n];
        let mut best = vec![(f64::INFINITY, 0usize); n];
        in_tree[0] = true;
        for (j, slot) in best.iter_mut().enumerate().skip(1) {
            *slot = (mr(0, j), 0);
        }
        let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n - 1);
        for _ in 1..n {
            let v = (0..n).filter(|&j| !in_tree[j]).min_by(|&a, &b| best[a].0.total_cmp(&best[b].0)).expect("a vertex remains");
            edges.push((best[v].0, best[v].1, v));
            in_tree[v] = true;
            for j in (0..n).filter(|&j| !in_tree[j]) {
                let w = mr(v, j);
                if w < best[j].0 {
                    best[j] = (w, v);
                }
            }
        }
        edges.sort_by(|a, b| a.0.total_cmp(&b.0));
        // ANCHOR_END: mutual_reachability
        // 3. Single-linkage tree: node ids 0..n are points, n.. are merges.
        let mut uf: Vec<usize> = (0..2 * n).collect();
        fn find(uf: &mut [usize], mut x: usize) -> usize {
            while uf[x] != x {
                uf[x] = uf[uf[x]];
                x = uf[x];
            }
            x
        }
        let mut size = vec![1usize; 2 * n];
        let mut children = vec![(0usize, 0usize); 2 * n];
        let mut height = vec![0.0; 2 * n];
        for (step, &(w, a, b)) in edges.iter().enumerate() {
            let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
            let id = n + step;
            children[id] = (ra, rb);
            height[id] = w;
            size[id] = size[ra] + size[rb];
            uf[ra] = id;
            uf[rb] = id;
        }
        let root = 2 * n - 2;
        // 4. Condense the tree.
        let mcs = self.min_cluster_size;
        let mut clusters = vec![CondensedCluster { parent: None, birth_lambda: 0.0, children: Vec::new(), stability: 0.0, fallen: Vec::new() }];
        let mut point_cluster = vec![0usize; n];
        let mut stack = vec![(root, 0usize)];
        let leaves_of = |node: usize, children: &[(usize, usize)]| {
            let mut out = Vec::new();
            let mut s = vec![node];
            while let Some(u) = s.pop() {
                if u < n {
                    out.push(u);
                } else {
                    s.push(children[u].0);
                    s.push(children[u].1);
                }
            }
            out
        };
        while let Some((node, cid)) = stack.pop() {
            if node < n {
                // A lone point reached the bottom of the condensed cluster.
                point_cluster[node] = cid;
                clusters[cid].fallen.push(node);
                continue;
            }
            let lambda = lambda_of(height[node]);
            let (l, r) = children[node];
            let birth = clusters[cid].birth_lambda;
            match (size[l] >= mcs, size[r] >= mcs) {
                (true, true) => {
                    clusters[cid].stability += (lambda - birth) * size[node] as f64;
                    let mut kids = Vec::new();
                    for side in [l, r] {
                        let id = clusters.len();
                        clusters.push(CondensedCluster { parent: Some(cid), birth_lambda: lambda, children: Vec::new(), stability: 0.0, fallen: Vec::new() });
                        kids.push(id);
                        stack.push((side, id));
                    }
                    clusters[cid].children = kids;
                }
                (false, false) => {
                    clusters[cid].stability += (lambda - birth) * size[node] as f64;
                    for p in leaves_of(node, &children) {
                        point_cluster[p] = cid;
                        clusters[cid].fallen.push(p);
                    }
                }
                (big_left, _) => {
                    let (big, small) = if big_left { (l, r) } else { (r, l) };
                    clusters[cid].stability += (lambda - birth) * size[small] as f64;
                    for p in leaves_of(small, &children) {
                        point_cluster[p] = cid;
                        clusters[cid].fallen.push(p);
                    }
                    stack.push((big, cid));
                }
            }
        }
        // ANCHOR: excess_of_mass
        // 5. Excess-of-mass selection, bottom-up (children have larger ids than their parents).
        let m = clusters.len();
        let mut selected = vec![false; m];
        let mut best_total = vec![0.0; m];
        for c in (0..m).rev() {
            if clusters[c].children.is_empty() {
                selected[c] = true;
                best_total[c] = clusters[c].stability;
            } else {
                let child_sum: f64 = clusters[c].children.iter().map(|&k| best_total[k]).sum();
                if c != 0 && clusters[c].stability >= child_sum {
                    selected[c] = true;
                    best_total[c] = clusters[c].stability;
                    let mut s = clusters[c].children.clone();
                    while let Some(k) = s.pop() {
                        selected[k] = false;
                        s.extend(clusters[k].children.iter().copied());
                    }
                } else {
                    best_total[c] = child_sum;
                }
            }
        }
        selected[0] = false; // the root is the whole data set, never a cluster by itself
        // ANCHOR_END: excess_of_mass
        // 6. Label the points by the selected cluster above (or at) where they fell out.
        let mut label_of = vec![None; m];
        self.stabilities.clear();
        for (next, c) in (0..m).filter(|&c| selected[c]).enumerate() {
            label_of[c] = Some(next);
            self.stabilities.push(clusters[c].stability);
        }
        Ok((0..n)
            .map(|p| {
                let mut c = Some(point_cluster[p]);
                while let Some(id) = c {
                    if selected[id] {
                        return label_of[id];
                    }
                    c = clusters[id].parent;
                }
                None
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_moons};
    use crate::metrics::adjusted_rand_index;
    use ndarray::{Axis, arr2, concatenate};

    #[test]
    fn finds_clusters_of_very_different_density_that_dbscan_cannot() {
        // A tight blob and a diffuse blob: no single eps suits both.
        let (dense, _) = make_blobs(60, &[vec![0.0, 0.0]], 0.2, 1);
        let (sparse, _) = make_blobs(60, &[vec![10.0, 10.0]], 2.0, 2);
        let x = concatenate(Axis(0), &[dense.view(), sparse.view()]).unwrap();
        let truth: Vec<Option<usize>> = (0..120).map(|i| Some(usize::from(i >= 60))).collect();
        let got = Hdbscan::new(10).fit_predict(x.view()).unwrap();
        let clusters: std::collections::HashSet<_> = got.iter().flatten().collect();
        assert_eq!(clusters.len(), 2);
        // Most points land in the right cluster (some sparse-tail points may be noise).
        let right = (0..120).filter(|&i| got[i].is_some()).count();
        assert!(right > 100, "{right}");
        let (kept, kept_truth): (Vec<_>, Vec<_>) = (0..120).filter(|&i| got[i].is_some()).map(|i| (got[i], truth[i])).unzip();
        assert!(adjusted_rand_index(&kept, &kept_truth).unwrap() > 0.95);
    }

    #[test]
    fn recovers_moons_and_flags_outliers_as_noise() {
        let (m, l) = make_moons(120, 0.06, 5);
        let got = Hdbscan::new(15).fit_predict(m.view()).unwrap();
        let (kept, truth): (Vec<_>, Vec<_>) = (0..240).filter(|&i| got[i].is_some()).map(|i| (got[i], Some(l[i]))).unzip();
        assert!(kept.len() > 200 && adjusted_rand_index(&kept, &truth).unwrap() > 0.9);
        let with_noise = concatenate(Axis(0), &[m.view(), arr2(&[[8.0, 8.0], [-7.0, 9.0], [9.0, -6.0]]).view()]).unwrap();
        let g = Hdbscan::new(15).fit_predict(with_noise.view()).unwrap();
        assert!(g[240..].iter().all(Option::is_none));
    }

    #[test]
    fn three_blobs_stabilities_and_degenerate_inputs() {
        let (x, l) = make_blobs(40, &[vec![0.0, 0.0], vec![8.0, 0.0], vec![4.0, 7.0]], 0.6, 7);
        let mut h = Hdbscan::new(8);
        let got = h.fit_predict(x.view()).unwrap();
        let truth: Vec<Option<usize>> = l.iter().map(|&c| Some(c)).collect();
        assert!((adjusted_rand_index(&got, &truth).unwrap() - 1.0).abs() < 1e-9, "{got:?}");
        assert_eq!(h.cluster_stabilities().len(), 3);
        assert!(h.cluster_stabilities().iter().all(|&s| s > 0.0));
        // Fewer points than the minimum cluster size: all noise, no panic.
        assert_eq!(Hdbscan::new(10).fit_predict(arr2(&[[0.0], [1.0]]).view()).unwrap(), vec![None, None]);
        // Identical points do not break the λ computation.
        let same = lib_jsl_core::R2D::ones((30, 2));
        assert_eq!(Hdbscan::new(5).fit_predict(same.view()).unwrap().len(), 30);
        assert!(Hdbscan::new(1).fit_predict(x.view()).is_err());
        assert!(Hdbscan::new(5).with_min_samples(0).fit_predict(x.view()).is_err());
        // Larger min_samples is more conservative: no more clustered points than before.
        let loose = Hdbscan::new(8).fit_predict(x.view()).unwrap().iter().flatten().count();
        let strict = Hdbscan::new(8).with_min_samples(25).fit_predict(x.view()).unwrap().iter().flatten().count();
        assert!(strict <= loose);
    }
}
