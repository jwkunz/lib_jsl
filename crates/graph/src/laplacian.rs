//! Graph Laplacians (SciPy's `scipy.sparse.csgraph.laplacian`).
//!
//! See the book entry [Matchings, k shortest paths and Laplacians](https://jwkunz.github.io/lib_jsl/book/graph/matching-laplacian.html#the-laplacian).

// Dense matrices read best as index loops.
#![allow(clippy::needless_range_loop)]

use crate::graph::Graph;

// ANCHOR: laplacian
/// The Laplacian `L = D − A` of a weighted graph, as a dense matrix, and the degrees (SciPy's
/// `laplacian`). `A` is the weighted adjacency matrix (parallel edges summed, loops ignored) and
/// `D` the diagonal of degrees: the in-degrees (column sums) by default, the out-degrees (row
/// sums) with `use_out_degree`; the two agree for undirected graphs. With `normed`, the symmetric
/// normalisation `I − D^{−1/2} A D^{−1/2}`, with isolated vertices left at 0 on the diagonal; the
/// returned degrees are then their square roots (1 for isolated vertices), as SciPy returns them.
pub fn laplacian(g: &Graph, normed: bool, use_out_degree: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = g.vertex_count();
    let mut a = vec![vec![0.0; n]; n];
    for u in 0..n {
        for &(v, w) in g.neighbors(u) {
            if u != v {
                a[u][v] += w;
            }
        }
    }
    let deg: Vec<f64> = if use_out_degree { a.iter().map(|r| r.iter().sum()).collect() } else { (0..n).map(|j| a.iter().map(|r| r[j]).sum()).collect() };
    if !normed {
        let mut l = a;
        for i in 0..n {
            for j in 0..n {
                l[i][j] = if i == j { deg[i] } else { -l[i][j] };
            }
        }
        return (l, deg);
    }
    let isolated: Vec<bool> = deg.iter().map(|&d| d == 0.0).collect();
    let s: Vec<f64> = deg.iter().zip(&isolated).map(|(&d, &iso)| if iso { 1.0 } else { d.sqrt() }).collect();
    let mut l = a;
    for i in 0..n {
        for j in 0..n {
            l[i][j] = if i == j { if isolated[i] { 0.0 } else { 1.0 } } else { -l[i][j] / s[j] / s[i] };
        }
    }
    (l, s)
}
// ANCHOR_END: laplacian

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_sum_to_zero_and_normalised_spectrum_is_bounded() {
        let g = Graph::from_edges(4, &[(0, 1, 1.0), (1, 2, 2.0), (2, 0, 0.5), (2, 3, 1.5)], false).unwrap();
        let (l, d) = laplacian(&g, false, false);
        assert_eq!(d, vec![1.5, 3.0, 4.0, 1.5]);
        for row in &l {
            assert!(row.iter().sum::<f64>().abs() < 1e-15);
        }
        let (n, _) = laplacian(&g, true, false);
        assert!((n[0][1] + 1.0 / (1.5f64 * 3.0).sqrt()).abs() < 1e-15 && n[2][2] == 1.0);
        // An isolated vertex.
        let g = Graph::from_edges(3, &[(0, 1, 2.0)], true).unwrap();
        let (n, d) = laplacian(&g, true, false);
        assert_eq!(n[0][0], 0.0);
        assert_eq!(d[0], 1.0);
    }
}
