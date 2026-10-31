//! UMAP: uniform manifold approximation and projection.

use crate::common::{MlRng, Transformer, check_features, pairwise_distances};
use crate::unsupervised::reduction::principal_component_analysis::Pca;
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// UMAP (McInnes, Healy and Melville) builds a weighted *fuzzy graph* of the data, in which each
/// point is connected to its `neighbors` nearest points with membership strengths that fall off
/// with distance (measured beyond the nearest neighbour, which locally always connects), and the
/// directed graph is symmetrised by fuzzy union. It then finds a low-dimensional layout whose
/// fuzzy graph is as similar as possible, by stochastic gradient descent: connected points attract
/// each other and randomly sampled points repel. Compared with t-SNE it is faster and keeps more
/// global structure.
///
/// `min_dist` sets how tightly points may pack in the embedding (small values give dense clumps),
/// and `neighbors` trades local detail against global structure. This implementation builds the
/// exact neighbour graph, so it suits up to several thousand points, and initialises the layout
/// with PCA.
#[derive(Debug, Clone)]
pub struct Umap {
    components: usize,
    neighbors: usize,
    min_dist: f64,
    epochs: usize,
    learning_rate: f64,
    negative_rate: usize,
    seed: u64,
}

/// Fits the smooth curve `1 / (1 + a x^{2b})` to the target similarity profile: 1 up to `min_dist`
/// and an exponential decay of scale `spread` beyond it. Levenberg-Marquardt on `(a, b)`.
pub fn fit_ab(min_dist: f64, spread: f64) -> (f64, f64) {
    let xs: Vec<f64> = (1..=300).map(|i| 3.0 * spread * i as f64 / 300.0).collect();
    let target: Vec<f64> = xs.iter().map(|&x| if x <= min_dist { 1.0 } else { (-(x - min_dist) / spread).exp() }).collect();
    let model = |a: f64, b: f64, x: f64| 1.0 / (1.0 + a * x.powf(2.0 * b));
    let cost = |a: f64, b: f64| xs.iter().zip(&target).map(|(&x, &t)| (model(a, b, x) - t).powi(2)).sum::<f64>();
    let (mut a, mut b, mut lambda) = (1.5, 0.9, 1e-2);
    let mut current = cost(a, b);
    for _ in 0..200 {
        // Normal equations with a numerical Jacobian.
        let (mut jtj, mut jtr) = ([[0.0; 2]; 2], [0.0; 2]);
        for (&x, &t) in xs.iter().zip(&target) {
            let r = model(a, b, x) - t;
            let ja = (model(a + 1e-6, b, x) - model(a, b, x)) / 1e-6;
            let jb = (model(a, b + 1e-6, x) - model(a, b, x)) / 1e-6;
            jtj[0][0] += ja * ja;
            jtj[0][1] += ja * jb;
            jtj[1][1] += jb * jb;
            jtr[0] += ja * r;
            jtr[1] += jb * r;
        }
        jtj[1][0] = jtj[0][1];
        let mut improved = false;
        for _ in 0..20 {
            let (m00, m11) = (jtj[0][0] * (1.0 + lambda), jtj[1][1] * (1.0 + lambda));
            let det = m00 * m11 - jtj[0][1] * jtj[1][0];
            if det.abs() < 1e-300 {
                lambda *= 10.0;
                continue;
            }
            let da = -(m11 * jtr[0] - jtj[0][1] * jtr[1]) / det;
            let db = -(-jtj[1][0] * jtr[0] + m00 * jtr[1]) / det;
            let (na, nb) = ((a + da).max(1e-3), (b + db).max(1e-3));
            let c = cost(na, nb);
            if c < current {
                (a, b, current, lambda) = (na, nb, c, (lambda * 0.5).max(1e-12));
                improved = true;
                break;
            }
            lambda *= 4.0;
        }
        if !improved || current < 1e-12 {
            break;
        }
    }
    (a, b)
}

impl Umap {
    pub fn new(components: usize, neighbors: usize, min_dist: f64, seed: u64) -> Self {
        Umap { components, neighbors, min_dist, epochs: 300, learning_rate: 1.0, negative_rate: 5, seed }
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// The symmetrised fuzzy-graph edge weights `(i, j, weight)` with `i < j`.
    #[allow(clippy::needless_range_loop)]
    pub fn fuzzy_graph(&self, x: VR2D) -> Result<Vec<(usize, usize, f64)>, ErrorsJSL> {
        check_features(x)?;
        let n = x.nrows();
        if self.neighbors < 2 || self.neighbors >= n {
            return Err(ErrorsJSL::InvalidInputRange("Need 2 <= neighbors < samples."));
        }
        let d = pairwise_distances(x);
        let k = self.neighbors;
        let target = (k as f64).log2();
        let mut directed = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            order.sort_by(|&a, &b| d[[i, a]].total_cmp(&d[[i, b]]));
            let nn = &order[..k];
            let rho = nn.iter().map(|&j| d[[i, j]]).find(|&v| v > 0.0).unwrap_or(0.0);
            // Bandwidth sigma so that the memberships sum to log2(k).
            let (mut lo, mut hi, mut sigma) = (0.0f64, f64::INFINITY, 1.0f64);
            for _ in 0..64 {
                let total: f64 = nn.iter().map(|&j| (-(d[[i, j]] - rho).max(0.0) / sigma).exp()).sum();
                if (total - target).abs() < 1e-5 {
                    break;
                }
                if total > target {
                    hi = sigma;
                    sigma = 0.5 * (lo + hi);
                } else {
                    lo = sigma;
                    sigma = if hi.is_finite() { 0.5 * (lo + hi) } else { sigma * 2.0 };
                }
            }
            for &j in nn {
                directed[i][j] = (-(d[[i, j]] - rho).max(0.0) / sigma.max(1e-12)).exp();
            }
        }
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = (directed[i][j], directed[j][i]);
                let w = a + b - a * b; // fuzzy set union
                if w > 0.0 {
                    edges.push((i, j, w));
                }
            }
        }
        Ok(edges)
    }

    /// Embeds `x`, one row of coordinates per point.
    pub fn fit_transform(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        let edges = self.fuzzy_graph(x)?;
        let n = x.nrows();
        if self.components == 0 || self.components > x.ncols() || self.min_dist.is_nan() || self.min_dist < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need 1 <= components <= features and min_dist >= 0."));
        }
        let dims = self.components;
        let (a, b) = fit_ab(self.min_dist, 1.0);
        let mut rng = MlRng::new(self.seed);
        // PCA initialisation scaled to a span of ±10, with a little jitter to break ties.
        let mut y = Pca::new(dims).fit_transform(x)?;
        let span = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        y.mapv_inplace(|v| v / span * 10.0 + rng.gaussian(0.0, 1e-4));
        let max_w = edges.iter().map(|e| e.2).fold(0.0, f64::max);
        let clip = |v: f64| v.clamp(-4.0, 4.0);
        for epoch in 0..self.epochs {
            let alpha = self.learning_rate * (1.0 - epoch as f64 / self.epochs as f64);
            for &(i, j, w) in &edges {
                if rng.uniform() * max_w > w {
                    continue; // sample edges in proportion to their weight
                }
                let d2: f64 = (0..dims).map(|k| (y[[i, k]] - y[[j, k]]).powi(2)).sum();
                if d2 > 0.0 {
                    let coeff = -2.0 * a * b * d2.powf(b - 1.0) / (1.0 + a * d2.powf(b));
                    for k in 0..dims {
                        let g = clip(coeff * (y[[i, k]] - y[[j, k]]));
                        y[[i, k]] += alpha * g;
                        y[[j, k]] -= alpha * g;
                    }
                }
                for _ in 0..self.negative_rate {
                    let r = rng.below(n);
                    if r == i {
                        continue;
                    }
                    let d2: f64 = (0..dims).map(|k| (y[[i, k]] - y[[r, k]]).powi(2)).sum();
                    let coeff = 2.0 * b / ((0.001 + d2) * (1.0 + a * d2.powf(b)));
                    for k in 0..dims {
                        let g = if d2 > 0.0 { clip(coeff * (y[[i, k]] - y[[r, k]])) } else { 4.0 };
                        y[[i, k]] += alpha * g;
                    }
                }
            }
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::squared_distance;
    use crate::datasets::{make_blobs, make_swiss_roll};

    fn neighbour_purity(y: &R2D, labels: &[usize], k: usize) -> f64 {
        let n = y.nrows();
        let mut agree = 0usize;
        for i in 0..n {
            let mut d: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| (j, squared_distance(&y.row(i).to_vec(), &y.row(j).to_vec()))).collect();
            d.sort_by(|a, b| a.1.total_cmp(&b.1));
            agree += d.iter().take(k).filter(|(j, _)| labels[*j] == labels[i]).count();
        }
        agree as f64 / (n * k) as f64
    }

    #[test]
    fn curve_fit_matches_the_reference_parameters() {
        let (a, b) = fit_ab(0.1, 1.0);
        assert!((a - 1.577).abs() < 0.1 && (b - 0.895).abs() < 0.05, "{a} {b}");
        // A larger min_dist keeps points further apart: the fitted curve stays high for longer.
        let (a2, _) = fit_ab(0.5, 1.0);
        assert!(a2 < a);
    }

    #[test]
    fn graph_weights_are_valid_fuzzy_memberships() {
        let (x, _) = make_blobs(20, &[vec![0.0, 0.0], vec![8.0, 8.0]], 0.8, 1);
        let g = Umap::new(2, 6, 0.1, 1).fuzzy_graph(x.view()).unwrap();
        assert!(g.iter().all(|&(i, j, w)| i < j && w > 0.0 && w <= 1.0 + 1e-12));
        // The two blobs are far apart, so no edge crosses between them.
        assert!(g.iter().all(|&(i, j, _)| (i < 20) == (j < 20)));
        assert!(g.len() >= 20 * 6 / 2); // symmetrisation never loses directed edges
    }

    #[test]
    fn separates_clusters_and_keeps_manifold_neighbourhoods() {
        let centres: Vec<Vec<f64>> = (0..3).map(|c| (0..8).map(|d| if d % 3 == c { 7.0 } else { 0.0 }).collect()).collect();
        let (x, l) = make_blobs(40, &centres, 1.0, 3);
        let y = Umap::new(2, 10, 0.1, 1).fit_transform(x.view()).unwrap();
        assert!(neighbour_purity(&y, &l, 10) > 0.97, "{}", neighbour_purity(&y, &l, 10));
        // Swiss roll: embedded neighbours should be close on the roll's own coordinate (angle t).
        let (roll, t) = make_swiss_roll(250, 0.0, 2);
        let e = Umap::new(2, 12, 0.2, 1).fit_transform(roll.view()).unwrap();
        let mut close = 0.0;
        for i in 0..250 {
            let mut d: Vec<(usize, f64)> = (0..250).filter(|&j| j != i).map(|j| (j, squared_distance(&e.row(i).to_vec(), &e.row(j).to_vec()))).collect();
            d.sort_by(|a, b| a.1.total_cmp(&b.1));
            close += d.iter().take(5).filter(|(j, _)| (t[*j] - t[i]).abs() < 1.2).count() as f64 / 5.0;
        }
        assert!(close / 250.0 > 0.8, "{}", close / 250.0);
        // Same seed, same embedding.
        assert_eq!(Umap::new(2, 10, 0.1, 1).fit_transform(x.view()).unwrap(), y);
    }

    #[test]
    fn errors() {
        let (x, _) = make_blobs(10, &[vec![0.0, 0.0]], 1.0, 1);
        assert!(Umap::new(2, 1, 0.1, 1).fit_transform(x.view()).is_err());
        assert!(Umap::new(2, 10, 0.1, 1).fit_transform(x.view()).is_err()); // neighbors >= samples
        assert!(Umap::new(0, 5, 0.1, 1).fit_transform(x.view()).is_err());
        assert!(Umap::new(2, 5, -1.0, 1).fit_transform(x.view()).is_err());
        assert!(Umap::new(1, 5, 0.1, 1).with_epochs(20).fit_transform(x.view()).is_ok());
    }
}
