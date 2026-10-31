//! t-distributed stochastic neighbour embedding.

use crate::common::{Transformer, check_features, squared_distance};
use crate::unsupervised::reduction::principal_component_analysis::Pca;
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// Exact t-SNE (van der Maaten and Hinton): embeds high-dimensional points in 2D or 3D so that
/// neighbours stay neighbours. Pairwise similarities in the input are Gaussian with a per-point
/// bandwidth chosen to match a target *perplexity* (the effective number of neighbours); in the
/// embedding they are Student-t distributed (heavy tails let dissimilar clusters separate). The
/// embedding minimises the KL divergence between the two by gradient descent with momentum and
/// adaptive gains, using "early exaggeration" of the input similarities at the start to form
/// clusters.
///
/// The exact method costs `O(n²)` per iteration, fine for up to a few thousand points. Only the
/// layout of the *fitted* points is produced: t-SNE has no transform for new data, and distances
/// between clusters in the picture are not meaningful.
#[derive(Debug, Clone)]
pub struct Tsne {
    components: usize,
    perplexity: f64,
    iterations: usize,
    learning_rate: f64,
    early_exaggeration: f64,
    kl_divergence: f64,
}

impl Tsne {
    /// `components` output dimensions (2 or 3 typical); `perplexity` roughly 5 to 50.
    pub fn new(components: usize, perplexity: f64) -> Self {
        Tsne { components, perplexity, iterations: 750, learning_rate: 200.0, early_exaggeration: 12.0, kl_divergence: f64::NAN }
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// The KL divergence of the final embedding (lower is a better match to the input).
    pub fn kl_divergence(&self) -> f64 {
        self.kl_divergence
    }

    /// Joint input probabilities `P` (symmetrised, summing to one).
    fn input_affinities(&self, x: VR2D) -> R2D {
        let n = x.nrows();
        let rows: Vec<Vec<f64>> = x.rows().into_iter().map(|r| r.to_vec()).collect();
        let d2 = R2D::from_shape_fn((n, n), |(i, j)| squared_distance(&rows[i], &rows[j]));
        let target = self.perplexity.ln();
        let mut p = R2D::zeros((n, n));
        for i in 0..n {
            // Binary search for the precision beta = 1/(2σ²) that yields the target entropy.
            let (mut lo, mut hi, mut beta) = (0.0f64, f64::INFINITY, 1.0f64);
            let mut row = vec![0.0; n];
            for _ in 0..60 {
                let mut sum = 0.0;
                for j in 0..n {
                    row[j] = if j == i { 0.0 } else { (-beta * d2[[i, j]]).exp() };
                    sum += row[j];
                }
                let sum = sum.max(1e-300);
                let entropy = sum.ln() + beta * (0..n).map(|j| d2[[i, j]] * row[j]).sum::<f64>() / sum;
                if (entropy - target).abs() < 1e-5 {
                    break;
                }
                if entropy > target {
                    lo = beta;
                    beta = if hi.is_finite() { 0.5 * (beta + hi) } else { beta * 2.0 };
                } else {
                    hi = beta;
                    beta = 0.5 * (beta + lo);
                }
            }
            let sum: f64 = row.iter().sum::<f64>().max(1e-300);
            for j in 0..n {
                p[[i, j]] = row[j] / sum;
            }
        }
        R2D::from_shape_fn((n, n), |(i, j)| ((p[[i, j]] + p[[j, i]]) / (2.0 * n as f64)).max(1e-12))
    }

    /// Embeds `x`, returning one row of coordinates per point.
    pub fn fit_transform(&mut self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        check_features(x)?;
        let n = x.nrows();
        if n < 4 || self.components == 0 || self.components > x.ncols() || self.perplexity.is_nan() || self.perplexity < 1.0 || self.perplexity >= (n - 1) as f64 {
            return Err(ErrorsJSL::InvalidInputRange("Need at least 4 samples, 1 <= components <= features and 1 <= perplexity < samples."));
        }
        let dims = self.components;
        let mut p = self.input_affinities(x);
        // PCA initialisation (rescaled to a tiny spread) is deterministic and keeps global structure.
        let mut y = Pca::new(dims).fit_transform(x)?;
        let spread = y.column(0).iter().map(|v| v * v).sum::<f64>().sqrt() / (n as f64).sqrt();
        y.mapv_inplace(|v| v / spread.max(1e-12) * 1e-4);
        let mut velocity = R2D::zeros((n, dims));
        let mut gains = R2D::ones((n, dims));
        let exaggeration_steps = (self.iterations / 4).min(250);
        p *= self.early_exaggeration;
        for it in 0..self.iterations {
            if it == exaggeration_steps {
                p /= self.early_exaggeration;
            }
            // Student-t similarities in the embedding.
            let mut num = R2D::zeros((n, n));
            let mut total = 0.0;
            for i in 0..n {
                for j in i + 1..n {
                    let d2: f64 = (0..dims).map(|k| (y[[i, k]] - y[[j, k]]).powi(2)).sum();
                    let v = 1.0 / (1.0 + d2);
                    num[[i, j]] = v;
                    num[[j, i]] = v;
                    total += 2.0 * v;
                }
            }
            let total = total.max(1e-300);
            let mut grad = R2D::zeros((n, dims));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let q = (num[[i, j]] / total).max(1e-12);
                        let mult = 4.0 * (p[[i, j]] - q) * num[[i, j]];
                        for k in 0..dims {
                            grad[[i, k]] += mult * (y[[i, k]] - y[[j, k]]);
                        }
                    }
                }
            }
            let momentum = if it < exaggeration_steps { 0.5 } else { 0.8 };
            for i in 0..n {
                for k in 0..dims {
                    gains[[i, k]] = if (grad[[i, k]] > 0.0) != (velocity[[i, k]] > 0.0) { gains[[i, k]] + 0.2 } else { gains[[i, k]] * 0.8 }.max(0.01);
                    velocity[[i, k]] = momentum * velocity[[i, k]] - self.learning_rate * gains[[i, k]] * grad[[i, k]];
                    y[[i, k]] += velocity[[i, k]];
                }
            }
            let mean = y.mean_axis(ndarray::Axis(0)).expect("non-empty");
            y -= &mean;
        }
        // Final KL divergence against the un-exaggerated P.
        if self.iterations <= exaggeration_steps {
            p /= self.early_exaggeration;
        }
        let mut total = 0.0;
        let mut num = R2D::zeros((n, n));
        for i in 0..n {
            for j in i + 1..n {
                let v = 1.0 / (1.0 + (0..dims).map(|k| (y[[i, k]] - y[[j, k]]).powi(2)).sum::<f64>());
                num[[i, j]] = v;
                num[[j, i]] = v;
                total += 2.0 * v;
            }
        }
        self.kl_divergence = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).filter(|(i, j)| i != j).map(|(i, j)| p[[i, j]] * (p[[i, j]] / (num[[i, j]] / total).max(1e-12)).ln()).sum();
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_blobs;

    /// Fraction of points whose nearest embedded neighbours share their label.
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
    fn separates_clusters_from_ten_dimensions_down_to_two() {
        let centres: Vec<Vec<f64>> = (0..3).map(|c| (0..10).map(|d| if d % 3 == c { 6.0 } else { 0.0 }).collect()).collect();
        let (x, l) = make_blobs(25, &centres, 1.0, 5);
        let mut t = Tsne::new(2, 12.0).with_iterations(300);
        let y = t.fit_transform(x.view()).unwrap();
        assert_eq!(y.dim(), (75, 2));
        assert!(neighbour_purity(&y, &l, 10) > 0.97, "{}", neighbour_purity(&y, &l, 10));
        assert!(t.kl_divergence().is_finite() && t.kl_divergence() > 0.0 && t.kl_divergence() < 1.0, "{}", t.kl_divergence());
    }

    #[test]
    fn more_iterations_lower_the_divergence_and_runs_are_reproducible() {
        let (x, _) = make_blobs(15, &[vec![0.0, 0.0, 0.0], vec![5.0, 5.0, 0.0], vec![0.0, 5.0, 5.0]], 0.7, 2);
        let kl = |iters: usize| {
            let mut t = Tsne::new(2, 10.0).with_iterations(iters);
            let y = t.fit_transform(x.view()).unwrap();
            (t.kl_divergence(), y)
        };
        let (short, y1) = kl(80);
        let (long, y2) = kl(400);
        assert!(long < short, "{long} vs {short}");
        assert_eq!(kl(80).1, y1); // deterministic: no random numbers are used
        assert_ne!(y1, y2);
        // The embedding is centred.
        assert!(y2.mean_axis(ndarray::Axis(0)).unwrap().iter().all(|m| m.abs() < 1e-9));
    }

    #[test]
    fn errors() {
        let (x, _) = make_blobs(10, &[vec![0.0, 0.0]], 1.0, 1);
        assert!(Tsne::new(2, 30.0).fit_transform(x.view()).is_err()); // perplexity >= samples
        assert!(Tsne::new(0, 5.0).fit_transform(x.view()).is_err());
        assert!(Tsne::new(3, 5.0).fit_transform(x.view()).is_err()); // more output dims than input
        assert!(Tsne::new(2, 0.5).fit_transform(x.view()).is_err());
        assert!(Tsne::new(1, 3.0).with_iterations(20).fit_transform(x.view()).is_ok());
    }
}
