//! Deterministic synthetic datasets for demonstrating and testing the learning algorithms.
//!
//! Every generator takes a seed, so results are reproducible. Feature matrices have one sample per
//! row.
//!
//! See the book entry [Foundations](https://jwkunz.github.io/lib_jsl/book/machine_learning/foundations.html#synthetic-data-sets).

use crate::common::MlRng;
use lib_jsl_core::{R1D, R2D};
use std::f64::consts::{PI, TAU};

/// Gaussian blobs: `per_cluster` points around each centre with standard deviation `spread`.
/// Returns the points and the true cluster of each.
pub fn make_blobs(per_cluster: usize, centres: &[Vec<f64>], spread: f64, seed: u64) -> (R2D, Vec<usize>) {
    let mut rng = MlRng::new(seed);
    let dims = centres.first().map_or(0, Vec::len);
    let mut x = R2D::zeros((per_cluster * centres.len(), dims));
    let mut labels = Vec::with_capacity(x.nrows());
    for (c, centre) in centres.iter().enumerate() {
        for i in 0..per_cluster {
            for (d, &mu) in centre.iter().enumerate() {
                x[[c * per_cluster + i, d]] = rng.gaussian(mu, spread);
            }
            labels.push(c);
        }
    }
    (x, labels)
}

/// Two interleaving half circles ("moons"): `n` points per moon with Gaussian noise.
pub fn make_moons(n: usize, noise: f64, seed: u64) -> (R2D, Vec<usize>) {
    let mut rng = MlRng::new(seed);
    let mut x = R2D::zeros((2 * n, 2));
    let mut labels = Vec::with_capacity(2 * n);
    for i in 0..n {
        let t = PI * i as f64 / (n.max(2) - 1) as f64;
        x[[i, 0]] = t.cos() + rng.gaussian(0.0, noise);
        x[[i, 1]] = t.sin() + rng.gaussian(0.0, noise);
        x[[n + i, 0]] = 1.0 - t.cos() + rng.gaussian(0.0, noise);
        x[[n + i, 1]] = 0.5 - t.sin() + rng.gaussian(0.0, noise);
        labels.push(0);
    }
    labels.extend(std::iter::repeat_n(1, n));
    (x, labels)
}

/// Two concentric circles: the inner has radius `factor` times the outer's (radius 1).
pub fn make_circles(n: usize, noise: f64, factor: f64, seed: u64) -> (R2D, Vec<usize>) {
    let mut rng = MlRng::new(seed);
    let mut x = R2D::zeros((2 * n, 2));
    let mut labels = Vec::with_capacity(2 * n);
    for i in 0..n {
        let t = TAU * i as f64 / n as f64;
        x[[i, 0]] = t.cos() + rng.gaussian(0.0, noise);
        x[[i, 1]] = t.sin() + rng.gaussian(0.0, noise);
        x[[n + i, 0]] = factor * t.cos() + rng.gaussian(0.0, noise);
        x[[n + i, 1]] = factor * t.sin() + rng.gaussian(0.0, noise);
        labels.push(0);
    }
    labels.extend(std::iter::repeat_n(1, n));
    (x, labels)
}

/// A linear regression problem `y = Xw + b + noise` with standard-normal features. Returns
/// `(x, y, true weights, true intercept)`.
pub fn make_regression(samples: usize, features: usize, noise: f64, seed: u64) -> (R2D, R1D, R1D, f64) {
    let mut rng = MlRng::new(seed);
    let w = R1D::from_iter((0..features).map(|_| rng.range(-3.0, 3.0)));
    let intercept = rng.range(-2.0, 2.0);
    let x = R2D::from_shape_fn((samples, features), |_| rng.gaussian(0.0, 1.0));
    let y = x.dot(&w) + intercept + R1D::from_iter((0..samples).map(|_| rng.gaussian(0.0, noise)));
    (x, y, w, intercept)
}

/// A noisy sine curve `y = sin(x)` sampled at `samples` random points in `[0, 2π]`, as a one-column
/// feature matrix; a classic nonlinear regression demonstration.
pub fn make_sine(samples: usize, noise: f64, seed: u64) -> (R2D, R1D) {
    let mut rng = MlRng::new(seed);
    let xs: Vec<f64> = (0..samples).map(|_| rng.range(0.0, TAU)).collect();
    let y = R1D::from_iter(xs.iter().map(|&v| v.sin() + rng.gaussian(0.0, noise)));
    (R2D::from_shape_fn((samples, 1), |(i, _)| xs[i]), y)
}

/// The "Swiss roll": a 2D sheet rolled up in 3D, a standard test for manifold learning. Returns the
/// 3D points and each point's position along the roll.
pub fn make_swiss_roll(samples: usize, noise: f64, seed: u64) -> (R2D, Vec<f64>) {
    let mut rng = MlRng::new(seed);
    let mut x = R2D::zeros((samples, 3));
    let mut t = Vec::with_capacity(samples);
    for i in 0..samples {
        let angle = 1.5 * PI * (1.0 + 2.0 * rng.uniform());
        x[[i, 0]] = angle * angle.cos() + rng.gaussian(0.0, noise);
        x[[i, 1]] = 21.0 * rng.uniform() + rng.gaussian(0.0, noise);
        x[[i, 2]] = angle * angle.sin() + rng.gaussian(0.0, noise);
        t.push(angle);
    }
    (x, t)
}

/// Three Gaussian classes in four dimensions with deliberately overlapping, correlated features,
/// a small stand-in for a tabular classification benchmark.
pub fn make_classification(per_class: usize, seed: u64) -> (R2D, Vec<usize>) {
    let centres = vec![vec![0.0, 0.0, 0.0, 0.0], vec![2.5, 1.0, -1.0, 0.5], vec![-1.0, 2.5, 1.5, -1.5]];
    let (mut x, labels) = make_blobs(per_class, &centres, 1.0, seed);
    for i in 0..x.nrows() {
        let leak = 0.5 * x[[i, 0]];
        x[[i, 1]] += leak; // correlate features 0 and 1
    }
    (x, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_labels_and_determinism() {
        let (x, l) = make_blobs(20, &[vec![0.0, 0.0], vec![10.0, 10.0], vec![-10.0, 5.0]], 0.5, 1);
        assert_eq!((x.dim(), l.len()), ((60, 2), 60));
        assert_eq!(make_blobs(5, &[vec![1.0]], 1.0, 7).0, make_blobs(5, &[vec![1.0]], 1.0, 7).0);
        assert_ne!(make_blobs(5, &[vec![1.0]], 1.0, 7).0, make_blobs(5, &[vec![1.0]], 1.0, 8).0);
        // Blob means land near their centres.
        let mean0: f64 = (0..20).map(|i| x[[i, 0]]).sum::<f64>() / 20.0;
        assert!(mean0.abs() < 0.5);
        let (m, ml) = make_moons(50, 0.05, 3);
        assert_eq!((m.dim(), ml.iter().sum::<usize>()), ((100, 2), 50));
        let (c, _) = make_circles(40, 0.0, 0.5, 4);
        assert!((c[[0, 0]].hypot(c[[0, 1]]) - 1.0).abs() < 1e-12 && (c[[40, 0]].hypot(c[[40, 1]]) - 0.5).abs() < 1e-12);
        assert_eq!(make_swiss_roll(30, 0.0, 1).0.dim(), (30, 3));
        assert_eq!(make_classification(10, 1).0.dim(), (30, 4));
        let (sx, sy) = make_sine(25, 0.1, 5);
        assert_eq!((sx.dim(), sy.len()), ((25, 1), 25));
    }

    #[test]
    fn regression_data_follows_its_true_model() {
        let (x, y, w, b) = make_regression(200, 3, 0.0, 9);
        let resid = &y - &(x.dot(&w) + b);
        assert!(resid.iter().all(|r| r.abs() < 1e-12));
    }
}
