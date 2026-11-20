//! Tools on finished dendrograms, following `scipy.cluster`: SciPy's linkage-matrix format,
//! the leaf order, inconsistency coefficients, optimal leaf ordering, and `whiten`.
//!
//! See the book entry [Hierarchical clustering and phylogenetic trees](https://jwkunz.github.io/lib_jsl/book/unsupervised/hierarchical.html#inconsistency-and-leaf-order).

use super::hierarchical::{Dendrogram, Merge};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

impl Dendrogram {
    /// The linkage matrix in SciPy's format: row `i` is `[a, b, distance, size]` for the merge
    /// that creates cluster `points + i`. The two ids are in the dendrogram's left-to-right
    /// order, which for trees from [`agglomerate`](super::agglomerate) need not have the smaller
    /// id first as SciPy's `linkage` does; the tree is the same.
    pub fn to_linkage_matrix(&self) -> R2D {
        let mut z = R2D::zeros((self.merges.len(), 4));
        for (i, m) in self.merges.iter().enumerate() {
            z[[i, 0]] = m.a as f64;
            z[[i, 1]] = m.b as f64;
            z[[i, 2]] = m.distance;
            z[[i, 3]] = m.size as f64;
        }
        z
    }

    /// A dendrogram from a linkage matrix in SciPy's format (`n − 1` rows of
    /// `[a, b, distance, size]`), checking that each row joins two existing, unused clusters
    /// and that the sizes add up.
    pub fn from_linkage_matrix(z: VR2D) -> Result<Dendrogram, ErrorsJSL> {
        if z.ncols() != 4 {
            return Err(ErrorsJSL::InvalidInputRange("A linkage matrix has four columns."));
        }
        let n = z.nrows() + 1;
        let mut sizes = vec![1usize; n];
        let mut used = vec![false; 2 * n - 1];
        let mut merges = Vec::with_capacity(n - 1);
        for (i, row) in z.rows().into_iter().enumerate() {
            let id = |v: f64| -> Result<usize, ErrorsJSL> {
                if v.fract() != 0.0 || v < 0.0 || v as usize >= n + i {
                    return Err(ErrorsJSL::InvalidInputRange("A linkage matrix row refers to a cluster that does not exist yet."));
                }
                Ok(v as usize)
            };
            let (a, b) = (id(row[0])?, id(row[1])?);
            if a == b || used[a] || used[b] {
                return Err(ErrorsJSL::InvalidInputRange("A linkage matrix uses a cluster twice."));
            }
            if !row[2].is_finite() || row[2] < 0.0 {
                return Err(ErrorsJSL::InvalidInputRange("Linkage distances must be finite and non-negative."));
            }
            let size = sizes[a] + sizes[b];
            if row[3] != size as f64 {
                return Err(ErrorsJSL::InvalidInputRange("A linkage matrix row has the wrong cluster size."));
            }
            used[a] = true;
            used[b] = true;
            sizes.push(size);
            merges.push(Merge { a, b, distance: row[2], size });
        }
        Ok(Dendrogram { points: n, merges })
    }

    /// The points in the order the dendrogram draws them, left to right (SciPy's `leaves_list`).
    pub fn leaves(&self) -> Vec<usize> {
        let n = self.points;
        if self.merges.is_empty() {
            return (0..n).collect();
        }
        let mut out = Vec::with_capacity(n);
        let mut stack = vec![n + self.merges.len() - 1];
        while let Some(c) = stack.pop() {
            if c < n {
                out.push(c);
            } else {
                let m = &self.merges[c - n];
                stack.push(m.b);
                stack.push(m.a);
            }
        }
        out
    }

    // ANCHOR: inconsistent
    /// The inconsistency statistics of every merge (SciPy's `inconsistent(Z, d)`): row `i` holds
    /// the mean and standard deviation of the heights of the merges at most `depth − 1` levels
    /// below merge `i` (itself included), their number, and the inconsistency coefficient
    /// `(h_i − mean) / std` (0 when the deviation is 0). A large coefficient marks a merge much
    /// higher than those beneath it: a natural place to cut.
    pub fn inconsistent(&self, depth: usize) -> Result<R2D, ErrorsJSL> {
        if depth == 0 {
            return Err(ErrorsJSL::InvalidInputRange("The depth must be at least 1."));
        }
        let n = self.points;
        let mut r = R2D::zeros((self.merges.len(), 4));
        let mut level = Vec::new();
        let mut next = Vec::new();
        for (i, m) in self.merges.iter().enumerate() {
            let (mut sum, mut sq, mut count) = (0.0, 0.0, 0usize);
            level.clear();
            level.push(i);
            for d in 0..depth {
                next.clear();
                for &j in &level {
                    let h = self.merges[j].distance;
                    sum += h;
                    sq += h * h;
                    count += 1;
                    if d + 1 < depth {
                        for c in [self.merges[j].a, self.merges[j].b] {
                            if c >= n {
                                next.push(c - n);
                            }
                        }
                    }
                }
                std::mem::swap(&mut level, &mut next);
                if level.is_empty() {
                    break;
                }
            }
            let k = count as f64;
            let mean = sum / k;
            let var = (sq - sum * sum / k) / if count < 2 { k } else { k - 1.0 };
            r[[i, 0]] = mean;
            r[[i, 2]] = k;
            if var > 0.0 {
                let s = var.sqrt();
                r[[i, 1]] = s;
                r[[i, 3]] = (m.distance - mean) / s;
            }
        }
        Ok(r)
    }
    // ANCHOR_END: inconsistent

    // ANCHOR: optimal_leaf_ordering
    /// The same tree with the two children of each merge swapped where needed so that the sum
    /// of the distances between neighbouring leaves is as small as possible (SciPy's
    /// `optimal_leaf_ordering`; Bar-Joseph, Gifford and Jaakkola's dynamic programme, `O(n³)`
    /// time and `O(n²)` memory). `distances` is the full matrix the tree was built from.
    pub fn optimal_leaf_ordering(&self, distances: VR2D) -> Result<Dendrogram, ErrorsJSL> {
        let n = self.points;
        if distances.nrows() != n || distances.ncols() != n {
            return Err(ErrorsJSL::InvalidInputRange("The distance matrix must be n × n for a tree on n points."));
        }
        if self.merges.len() + 1 != n {
            return Err(ErrorsJSL::InvalidInputRange("The tree must join all its points."));
        }
        if n <= 2 {
            return Ok(self.clone());
        }
        // Leaves of every cluster.
        let mut members: Vec<Vec<usize>> = (0..n).map(|p| vec![p]).collect();
        for m in &self.merges {
            let joined: Vec<usize> = members[m.a].iter().chain(&members[m.b]).copied().collect();
            members.push(joined);
        }
        // best[u][w]: the smallest path length through the leaves of the cluster where u and w
        // first meet, starting at u and ending at w.
        let mut best = vec![vec![0.0f64; n]; n];
        for m in &self.merges {
            let (l, r) = (&members[m.a], &members[m.b]);
            // Leaves of a child that can end an ordering of that child starting at `u`: those in
            // the other grandchild (or u itself for a single leaf).
            let ends = |c: usize, u: usize| -> Vec<usize> {
                if c < n {
                    return vec![u];
                }
                let mc = &self.merges[c - n];
                if members[mc.a].contains(&u) { members[mc.b].clone() } else { members[mc.a].clone() }
            };
            // t[u][k] = min over m (end of the left part starting at u) of best[u][m] + D[m][k].
            let mut t = vec![vec![f64::INFINITY; r.len()]; l.len()];
            for (iu, &u) in l.iter().enumerate() {
                let em = ends(m.a, u);
                for (ik, &k) in r.iter().enumerate() {
                    t[iu][ik] = em.iter().map(|&mm| best[u][mm] + distances[[mm, k]]).fold(f64::INFINITY, f64::min);
                }
            }
            let r_index: std::collections::HashMap<usize, usize> = r.iter().enumerate().map(|(i, &k)| (k, i)).collect();
            for (iu, &u) in l.iter().enumerate() {
                for &w in r.iter() {
                    let ek = ends(m.b, w);
                    let v = ek.iter().map(|&k| t[iu][r_index[&k]] + best[k][w]).fold(f64::INFINITY, f64::min);
                    best[u][w] = v;
                    best[w][u] = v;
                }
            }
        }
        // Reconstruct from the root: choose the ends, then recurse.
        let root = n + self.merges.len() - 1;
        let mr = &self.merges[root - n];
        let (mut bu, mut bw, mut bv) = (0, 0, f64::INFINITY);
        for &u in &members[mr.a] {
            for &w in &members[mr.b] {
                if best[u][w] < bv {
                    (bu, bw, bv) = (u, w, best[u][w]);
                }
            }
        }
        let mut out = self.clone();
        // Work list of (cluster, leaf that must come first, leaf that must come last).
        let mut work = vec![(root, bu, bw)];
        while let Some((c, u, w)) = work.pop() {
            if c < n {
                continue;
            }
            let idx = c - n;
            let mut m = self.merges[idx];
            if !members[m.a].contains(&u) {
                std::mem::swap(&mut m.a, &mut m.b);
            }
            out.merges[idx].a = m.a;
            out.merges[idx].b = m.b;
            // Find the inner ends m_end (in a) and k (in b) achieving best[u][w].
            let ends = |c: usize, x: usize| -> Vec<usize> {
                if c < n {
                    return vec![x];
                }
                let mc = &self.merges[c - n];
                if members[mc.a].contains(&x) { members[mc.b].clone() } else { members[mc.a].clone() }
            };
            let (ea, eb) = (ends(m.a, u), ends(m.b, w));
            let (mut bm, mut bk, mut bv) = (u, w, f64::INFINITY);
            for &mm in &ea {
                for &k in &eb {
                    let v = best[u][mm] + distances[[mm, k]] + best[k][w];
                    if v < bv {
                        (bm, bk, bv) = (mm, k, v);
                    }
                }
            }
            work.push((m.a, u, bm));
            work.push((m.b, bk, w));
        }
        Ok(out)
    }
    // ANCHOR_END: optimal_leaf_ordering

    /// The sum of the distances between neighbouring leaves in [`leaves`](Self::leaves) order,
    /// the quantity [`optimal_leaf_ordering`](Self::optimal_leaf_ordering) minimises.
    pub fn leaf_path_length(&self, distances: VR2D) -> f64 {
        self.leaves().windows(2).map(|p| distances[[p[0], p[1]]]).sum()
    }
}

/// Each feature (column) divided by its standard deviation over the observations (SciPy's
/// `whiten`, population standard deviation). Columns with zero deviation are left unchanged.
pub fn whiten(obs: VR2D) -> R2D {
    let mut out = obs.to_owned();
    let m = obs.nrows() as f64;
    for mut col in out.columns_mut() {
        let mean = col.sum() / m;
        let sd = (col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / m).sqrt();
        if sd > 0.0 {
            col.mapv_inplace(|v| v / sd);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::pairwise_distances;
    use crate::unsupervised::clustering::{Linkage, agglomerate};
    use ndarray::arr2;

    fn brute_best(t: &Dendrogram, d: VR2D) -> f64 {
        // Try all 2^(n−1) child swaps.
        let k = t.merges.len();
        (0..1u32 << k)
            .map(|mask| {
                let mut s = t.clone();
                for i in 0..k {
                    if mask >> i & 1 == 1 {
                        let m = &mut s.merges[i];
                        std::mem::swap(&mut m.a, &mut m.b);
                    }
                }
                s.leaf_path_length(d)
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn optimal_leaf_ordering_is_optimal() {
        let x = arr2(&[[0.0, 0.0], [3.1, 0.2], [0.4, 2.2], [5.0, 4.0], [1.0, 1.0], [6.2, 0.3], [2.5, 3.3], [4.1, 1.7], [0.2, 5.1]]);
        let d = pairwise_distances(x.view());
        for lk in [Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward] {
            let t = agglomerate(d.view(), lk).unwrap();
            let o = t.optimal_leaf_ordering(d.view()).unwrap();
            assert!((o.leaf_path_length(d.view()) - brute_best(&t, d.view())).abs() < 1e-12, "{lk:?}");
            assert_eq!(o.cophenetic_distances(), t.cophenetic_distances());
            let mut l = o.leaves();
            l.sort_unstable();
            assert_eq!(l, (0..9).collect::<Vec<_>>());
        }
    }

    #[test]
    fn optimal_leaf_ordering_matches_brute_force_on_random_trees() {
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..40 {
            let x = R2D::from_shape_fn((8, 2), |_| next());
            let d = pairwise_distances(x.view());
            let t = agglomerate(d.view(), Linkage::Average).unwrap();
            let o = t.optimal_leaf_ordering(d.view()).unwrap();
            assert!((o.leaf_path_length(d.view()) - brute_best(&t, d.view())).abs() < 1e-12);
        }
    }

    #[test]
    fn linkage_matrix_round_trip_and_inconsistency() {
        let x = arr2(&[[0.0], [1.0], [5.0], [6.0], [20.0]]);
        let t = agglomerate(pairwise_distances(x.view()).view(), Linkage::Single).unwrap();
        let z = t.to_linkage_matrix();
        assert_eq!(Dendrogram::from_linkage_matrix(z.view()).unwrap(), t);
        let mut bad = z.clone();
        bad[[1, 3]] = 3.0;
        assert!(Dendrogram::from_linkage_matrix(bad.view()).is_err());
        // Heights 1, 1, 4, 14; the top merge sits on the merges at 4 and 1 (depth 2).
        let r = t.inconsistent(2).unwrap();
        assert_eq!(r[[0, 2]], 1.0);
        assert_eq!(r[[0, 1]], 0.0);
        let (mean, sd) = ((4.0 + 14.0) / 2.0, ((16.0 + 196.0 - 18.0f64 * 18.0 / 2.0) / 1.0f64).sqrt());
        assert!((r[[3, 0]] - mean).abs() < 1e-12 && (r[[3, 1]] - sd).abs() < 1e-12);
        assert!((r[[3, 3]] - (14.0 - mean) / sd).abs() < 1e-12);
        assert!(t.inconsistent(0).is_err());
    }

    #[test]
    fn whiten_gives_unit_deviation() {
        let w = whiten(arr2(&[[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [8.0, 5.0, 9.0]]).view());
        for j in [0, 2] {
            let c = w.column(j);
            let mean = c.sum() / 3.0;
            assert!(((c.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 3.0) - 1.0).abs() < 1e-12);
        }
        assert_eq!(w.column(1).to_vec(), vec![5.0; 3]);
    }
}
