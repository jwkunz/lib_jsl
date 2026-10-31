//! Phylogenetic tree construction from a distance matrix: UPGMA and neighbor joining.

use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// A weighted tree over `leaves` labelled leaves (ids `0..leaves`) and internal nodes (ids
/// `leaves..`). UPGMA trees are rooted; neighbour-joining trees are unrooted and are drawn rooted
/// at the final join for printing.
#[derive(Debug, Clone, PartialEq)]
pub struct PhylogeneticTree {
    pub labels: Vec<String>,
    /// Undirected edges `(a, b, length)`.
    pub edges: Vec<(usize, usize, f64)>,
    /// The root node, for rooted (UPGMA) trees.
    pub root: Option<usize>,
    nodes: usize,
}

fn validate(distances: VR2D, labels: &[&str]) -> Result<usize, ErrorsJSL> {
    let n = distances.nrows();
    if n < 2 || distances.ncols() != n || labels.len() != n {
        return Err(ErrorsJSL::IncompatibleArraySizes((n, labels.len())));
    }
    for i in 0..n {
        if distances[[i, i]].abs() > 1e-12 {
            return Err(ErrorsJSL::InvalidInputRange("The distance matrix must have a zero diagonal."));
        }
        for j in 0..n {
            if !distances[[i, j]].is_finite() || distances[[i, j]] < 0.0 || (distances[[i, j]] - distances[[j, i]]).abs() > 1e-9 * (1.0 + distances[[i, j]].abs()) {
                return Err(ErrorsJSL::InvalidInputRange("Distances must be finite, non-negative and symmetric."));
            }
        }
    }
    Ok(n)
}

impl PhylogeneticTree {
    pub fn leaves(&self) -> usize {
        self.labels.len()
    }

    pub fn nodes(&self) -> usize {
        self.nodes
    }

    fn adjacency(&self) -> Vec<Vec<(usize, f64)>> {
        let mut adj = vec![Vec::new(); self.nodes];
        for &(a, b, w) in &self.edges {
            adj[a].push((b, w));
            adj[b].push((a, w));
        }
        adj
    }

    /// The path-length (patristic) distance between every pair of leaves along the tree.
    pub fn patristic_distances(&self) -> R2D {
        let n = self.leaves();
        let adj = self.adjacency();
        let mut out = R2D::zeros((n, n));
        for s in 0..n {
            let mut dist = vec![f64::NAN; self.nodes];
            dist[s] = 0.0;
            let mut stack = vec![s];
            while let Some(u) = stack.pop() {
                for &(v, w) in &adj[u] {
                    if dist[v].is_nan() {
                        dist[v] = dist[u] + w;
                        stack.push(v);
                    }
                }
            }
            for t in 0..n {
                out[[s, t]] = dist[t];
            }
        }
        out
    }

    /// The tree in Newick format, e.g. `((A:1.00000,B:1.00000):0.50000,C:1.50000);`.
    pub fn to_newick(&self) -> String {
        let adj = self.adjacency();
        let root = self.root.unwrap_or(self.nodes - 1);
        fn go(u: usize, parent: Option<usize>, adj: &[Vec<(usize, f64)>], t: &PhylogeneticTree, length: Option<f64>) -> String {
            let kids: Vec<(usize, f64)> = adj[u].iter().copied().filter(|&(v, _)| Some(v) != parent).collect();
            let name = if u < t.leaves() { t.labels[u].clone() } else { String::new() };
            let body = if kids.is_empty() { name } else { format!("({})", kids.iter().map(|&(v, w)| go(v, Some(u), adj, t, Some(w))).collect::<Vec<_>>().join(",")) };
            match length {
                Some(w) => format!("{body}:{w:.5}"),
                None => body,
            }
        }
        format!("{};", go(root, None, &adj, self, None))
    }

    /// For rooted trees: the distance from the root to every leaf (equal for an ultrametric tree).
    pub fn root_to_leaf_distances(&self) -> Option<Vec<f64>> {
        let root = self.root?;
        let adj = self.adjacency();
        let mut dist = vec![f64::NAN; self.nodes];
        dist[root] = 0.0;
        let mut stack = vec![root];
        while let Some(u) = stack.pop() {
            for &(v, w) in &adj[u] {
                if dist[v].is_nan() {
                    dist[v] = dist[u] + w;
                    stack.push(v);
                }
            }
        }
        Some(dist[..self.leaves()].to_vec())
    }
}

/// UPGMA (unweighted pair group method with arithmetic mean): repeatedly merges the two closest
/// clusters, placing the new node at half their distance, and averages distances weighted by
/// cluster size. Assumes a constant evolutionary rate (a molecular clock) and returns a rooted,
/// ultrametric tree; it is exact when the input is ultrametric.
#[allow(clippy::needless_range_loop)]
pub fn upgma(distances: VR2D, labels: &[&str]) -> Result<PhylogeneticTree, ErrorsJSL> {
    let n = validate(distances, labels)?;
    let mut d: Vec<Vec<f64>> = distances.rows().into_iter().map(|r| r.to_vec()).collect();
    let mut active: Vec<usize> = (0..n).collect(); // node ids of the current clusters
    let mut size = vec![1.0; n];
    let mut height = vec![0.0; n];
    let mut edges = Vec::new();
    let mut next = n;
    while active.len() > 1 {
        let (mut bi, mut bj, mut best) = (0, 1, f64::INFINITY);
        for a in 0..active.len() {
            for b in a + 1..active.len() {
                if d[a][b] < best {
                    (bi, bj, best) = (a, b, d[a][b]);
                }
            }
        }
        let (ni, nj) = (active[bi], active[bj]);
        let h = best / 2.0;
        edges.push((next, ni, h - height[ni]));
        edges.push((next, nj, h - height[nj]));
        let (si, sj) = (size[ni], size[nj]);
        let merged: Vec<f64> = (0..active.len()).map(|k| (d[bi][k] * si + d[bj][k] * sj) / (si + sj)).collect();
        // Replace row/column bi by the merged cluster, drop bj.
        for k in 0..active.len() {
            d[bi][k] = merged[k];
            d[k][bi] = merged[k];
        }
        d[bi][bi] = 0.0;
        d.remove(bj);
        d.iter_mut().for_each(|r| {
            r.remove(bj);
        });
        active.remove(bj);
        size.push(si + sj);
        height.push(h);
        active[bi] = next;
        next += 1;
    }
    Ok(PhylogeneticTree { labels: labels.iter().map(|s| s.to_string()).collect(), edges, root: Some(next - 1), nodes: next })
}

/// Neighbour joining (Saitou and Nei): repeatedly joins the pair that minimises the total tree
/// length, using the net divergence of each taxon to correct for unequal rates. Returns an
/// unrooted tree and recovers the true tree exactly from additive distances.
#[allow(clippy::needless_range_loop)]
pub fn neighbor_joining(distances: VR2D, labels: &[&str]) -> Result<PhylogeneticTree, ErrorsJSL> {
    let n = validate(distances, labels)?;
    let mut d: Vec<Vec<f64>> = distances.rows().into_iter().map(|r| r.to_vec()).collect();
    let mut active: Vec<usize> = (0..n).collect();
    let mut edges = Vec::new();
    let mut next = n;
    while active.len() > 2 {
        let m = active.len();
        let r: Vec<f64> = (0..m).map(|i| (0..m).map(|j| d[i][j]).sum()).collect();
        let (mut bi, mut bj, mut best) = (0, 1, f64::INFINITY);
        for i in 0..m {
            for j in i + 1..m {
                let q = (m as f64 - 2.0) * d[i][j] - r[i] - r[j];
                if q < best {
                    (bi, bj, best) = (i, j, q);
                }
            }
        }
        let li = 0.5 * d[bi][bj] + (r[bi] - r[bj]) / (2.0 * (m as f64 - 2.0));
        let lj = d[bi][bj] - li;
        edges.push((next, active[bi], li));
        edges.push((next, active[bj], lj));
        let merged: Vec<f64> = (0..m).map(|k| 0.5 * (d[bi][k] + d[bj][k] - d[bi][bj])).collect();
        for k in 0..m {
            d[bi][k] = merged[k];
            d[k][bi] = merged[k];
        }
        d[bi][bi] = 0.0;
        d.remove(bj);
        d.iter_mut().for_each(|row| {
            row.remove(bj);
        });
        active.remove(bj);
        active[bi] = next;
        next += 1;
    }
    edges.push((active[0], active[1], d[0][1]));
    Ok(PhylogeneticTree { labels: labels.iter().map(|s| s.to_string()).collect(), edges, root: None, nodes: next })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn close(a: &R2D, b: &R2D, tol: f64) -> bool {
        a.dim() == b.dim() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn upgma_recovers_an_ultrametric_tree() {
        // ((A,B),C) with A-B = 2, others 6: heights 1 and 3.
        let d = arr2(&[[0.0, 2.0, 6.0], [2.0, 0.0, 6.0], [6.0, 6.0, 0.0]]);
        let t = upgma(d.view(), &["A", "B", "C"]).unwrap();
        assert!(close(&t.patristic_distances(), &d, 1e-12));
        assert_eq!(t.to_newick(), "((A:1.00000,B:1.00000):2.00000,C:3.00000);");
        let heights = t.root_to_leaf_distances().unwrap();
        assert!(heights.iter().all(|h| (h - 3.0).abs() < 1e-12));
    }

    #[test]
    fn neighbor_joining_recovers_additive_distances_exactly() {
        // Known additive tree: leaves a..e on a caterpillar with unequal branch lengths.
        let d = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]); // the textbook (Saitou-Nei / Wikipedia) example
        let t = neighbor_joining(d.view(), &["a", "b", "c", "d", "e"]).unwrap();
        assert!(close(&t.patristic_distances(), &d, 1e-9));
        assert_eq!(t.edges.len(), 2 * 5 - 3); // unrooted binary tree on 5 leaves
        assert!(t.edges.iter().all(|e| e.2 > -1e-9));
        assert!(t.root.is_none());
        // Cherries (a,b) and (d,e) must be sister groups.
        let newick = t.to_newick();
        assert!(newick.contains('a') && newick.ends_with(';'));
    }

    #[test]
    fn nj_beats_upgma_when_rates_differ() {
        // Additive but strongly non-clock-like: A has a very long branch.
        let d = arr2(&[[0.0, 11.0, 12.0, 13.0], [11.0, 0.0, 3.0, 4.0], [12.0, 3.0, 0.0, 3.0], [13.0, 4.0, 3.0, 0.0]]);
        let labels = ["A", "B", "C", "D"];
        let nj = neighbor_joining(d.view(), &labels).unwrap();
        let up = upgma(d.view(), &labels).unwrap();
        let err = |t: &PhylogeneticTree| t.patristic_distances().iter().zip(d.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>();
        assert!(err(&nj) < 1e-9 && err(&up) > 1.0);
    }

    #[test]
    fn two_taxa_and_validation() {
        let d = arr2(&[[0.0, 4.0], [4.0, 0.0]]);
        let nj = neighbor_joining(d.view(), &["x", "y"]).unwrap();
        assert_eq!(nj.edges, vec![(0, 1, 4.0)]);
        let up = upgma(d.view(), &["x", "y"]).unwrap();
        assert_eq!(up.to_newick(), "(x:2.00000,y:2.00000);");
        assert!(upgma(arr2(&[[0.0]]).view(), &["x"]).is_err());
        assert!(upgma(d.view(), &["x"]).is_err());
        assert!(neighbor_joining(arr2(&[[0.0, 1.0], [2.0, 0.0]]).view(), &["x", "y"]).is_err());
        assert!(neighbor_joining(arr2(&[[1.0, 1.0], [1.0, 0.0]]).view(), &["x", "y"]).is_err());
        assert!(neighbor_joining(arr2(&[[0.0, -1.0], [-1.0, 0.0]]).view(), &["x", "y"]).is_err());
    }
}
