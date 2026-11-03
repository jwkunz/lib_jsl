//! Shared building blocks: model traits, a seeded random helper and small numeric utilities.
//!
//! Data is stored as `ndarray` arrays: a feature matrix `x` of shape `(samples, features)` (a
//! [`VR2D`] view) and targets as a [`VR1D`] (regression) or `&[usize]` class labels (classification).
//!
//! See the book entry [Foundations](https://jwkunz.github.io/lib_jsl/book/machine_learning/foundations.html).

use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};
use lib_jsl_random::distributions::gaussian::gaussian_distribution_box_muller;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;

// ANCHOR: traits
/// A model mapping feature rows to real-valued predictions.
pub trait Regressor {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL>;
    fn predict(&self, x: VR2D) -> Result<R1D, ErrorsJSL>;
}

/// A model mapping feature rows to class labels `0..classes`.
pub trait Classifier {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL>;
    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL>;
}

/// A model that groups rows without labels. Noise points (density-based methods) get `None`.
pub trait Clusterer {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL>;
}

/// A learned transformation of feature rows (scaling, projection, embedding).
pub trait Transformer {
    fn fit(&mut self, x: VR2D) -> Result<(), ErrorsJSL>;
    fn transform(&self, x: VR2D) -> Result<R2D, ErrorsJSL>;
    fn fit_transform(&mut self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        self.fit(x)?;
        self.transform(x)
    }
}
// ANCHOR_END: traits

/// A small seeded random-number helper used by the stochastic algorithms.
#[derive(Debug, Clone)]
pub struct MlRng {
    rng: Xoshiro256PlusPlus,
}

impl MlRng {
    pub fn new(seed: u64) -> Self {
        MlRng { rng: Xoshiro256PlusPlus::from_seed(seed) }
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.rng.next_f64()
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform integer in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        ((self.uniform() * n as f64) as usize).min(n - 1)
    }

    /// A normal sample with the given mean and standard deviation.
    pub fn gaussian(&mut self, mean: f64, sd: f64) -> f64 {
        // Box-Muller needs u0 in (0, 1]; next_f64 returns [0, 1).
        loop {
            let (z, _) = gaussian_distribution_box_muller(&mut self.rng, mean, sd);
            if z.is_finite() {
                return z;
            }
        }
    }

    /// Fisher-Yates shuffle in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }

    /// `k` distinct indices from `0..n` (all of them, shuffled, if `k >= n`).
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut all: Vec<usize> = (0..n).collect();
        self.shuffle(&mut all);
        all.truncate(k.min(n));
        all
    }
}

/// Squared Euclidean distance between two equal-length slices.
pub fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The rows of a matrix as owned vectors.
pub fn rows(x: VR2D) -> Vec<Vec<f64>> {
    x.rows().into_iter().map(|r| r.to_vec()).collect()
}

/// Checks a non-empty feature matrix with finite values.
pub fn check_features(x: VR2D) -> Result<(), ErrorsJSL> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(ErrorsJSL::InvalidInputRange("The feature matrix must have at least one row and column."));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("The feature matrix contains non-finite values."));
    }
    Ok(())
}

/// Checks features and a matching-length target.
pub fn check_xy(x: VR2D, len: usize) -> Result<(), ErrorsJSL> {
    check_features(x)?;
    if x.nrows() != len {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.nrows(), len)));
    }
    Ok(())
}

/// The number of classes implied by labels (max label + 1).
pub fn class_count(labels: &[usize]) -> usize {
    labels.iter().max().map_or(0, |m| m + 1)
}

/// The index of the largest value (first on ties); `None` if empty or all NaN.
pub fn argmax(values: &[f64]) -> Option<usize> {
    values.iter().enumerate().filter(|(_, v)| !v.is_nan()).fold(None, |best: Option<(usize, f64)>, (i, &v)| match best {
        Some((_, b)) if b >= v => best,
        _ => Some((i, v)),
    }).map(|(i, _)| i)
}

/// The matrix of Euclidean distances between all pairs of rows.
pub fn pairwise_distances(x: VR2D) -> R2D {
    let r: Vec<Vec<f64>> = rows(x);
    let mut d = R2D::zeros((r.len(), r.len()));
    for i in 0..r.len() {
        for j in i + 1..r.len() {
            let v = squared_distance(&r[i], &r[j]).sqrt();
            d[[i, j]] = v;
            d[[j, i]] = v;
        }
    }
    d
}
