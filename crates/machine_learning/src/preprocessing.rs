//! Feature scaling, splitting and expansion.
//!
//! See the book entry [Foundations](https://jwkunz.github.io/lib_jsl/book/machine_learning/foundations.html#scaling).

use crate::common::{MlRng, Transformer, check_features};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};

/// Standardises each column to zero mean and unit variance (constant columns are only centred).
#[derive(Debug, Clone, Default)]
pub struct StandardScaler {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }
    /// Undoes [`Transformer::transform`].
    pub fn inverse_transform(&self, z: VR2D) -> Result<R2D, ErrorsJSL> {
        self.check_width(z)?;
        Ok(R2D::from_shape_fn(z.dim(), |(i, j)| z[[i, j]] * self.scale[j] + self.mean[j]))
    }
    fn check_width(&self, x: VR2D) -> Result<(), ErrorsJSL> {
        if self.mean.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The scaler has not been fitted."));
        }
        if x.ncols() != self.mean.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.mean.len())));
        }
        Ok(())
    }
}

impl Transformer for StandardScaler {
    fn fit(&mut self, x: VR2D) -> Result<(), ErrorsJSL> {
        check_features(x)?;
        let n = x.nrows() as f64;
        self.mean = x.columns().into_iter().map(|c| c.sum() / n).collect();
        self.scale = x
            .columns()
            .into_iter()
            .zip(&self.mean)
            .map(|(c, &m)| {
                let sd = (c.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt();
                if sd > 1e-12 { sd } else { 1.0 }
            })
            .collect();
        Ok(())
    }
    fn transform(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        self.check_width(x)?;
        Ok(R2D::from_shape_fn(x.dim(), |(i, j)| (x[[i, j]] - self.mean[j]) / self.scale[j]))
    }
}

/// Rescales each column linearly onto `[0, 1]` (constant columns map to 0).
#[derive(Debug, Clone, Default)]
pub struct MinMaxScaler {
    min: Vec<f64>,
    range: Vec<f64>,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Transformer for MinMaxScaler {
    fn fit(&mut self, x: VR2D) -> Result<(), ErrorsJSL> {
        check_features(x)?;
        self.min = x.columns().into_iter().map(|c| c.iter().copied().fold(f64::INFINITY, f64::min)).collect();
        self.range = x
            .columns()
            .into_iter()
            .zip(&self.min)
            .map(|(c, &lo)| {
                let r = c.iter().copied().fold(f64::NEG_INFINITY, f64::max) - lo;
                if r > 1e-12 { r } else { 1.0 }
            })
            .collect();
        Ok(())
    }
    fn transform(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.min.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The scaler has not been fitted."));
        }
        if x.ncols() != self.min.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.min.len())));
        }
        Ok(R2D::from_shape_fn(x.dim(), |(i, j)| (x[[i, j]] - self.min[j]) / self.range[j]))
    }
}

/// Shuffles sample indices and splits them into `(train, test)` index lists, with
/// `test_fraction` of the samples (at least one, and leaving at least one) in the test set.
pub fn train_test_split(samples: usize, test_fraction: f64, seed: u64) -> Result<(Vec<usize>, Vec<usize>), ErrorsJSL> {
    if samples < 2 || !(0.0..1.0).contains(&test_fraction) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least two samples and a test fraction in [0, 1)."));
    }
    let mut idx: Vec<usize> = (0..samples).collect();
    MlRng::new(seed).shuffle(&mut idx);
    let n_test = ((samples as f64 * test_fraction).round() as usize).clamp(1, samples - 1);
    let test = idx.split_off(samples - n_test);
    Ok((idx, test))
}

/// Selects rows of a matrix.
pub fn select_rows(x: VR2D, indices: &[usize]) -> R2D {
    R2D::from_shape_fn((indices.len(), x.ncols()), |(i, j)| x[[indices[i], j]])
}

/// Selects entries of a vector.
pub fn select(y: VR1D, indices: &[usize]) -> R1D {
    R1D::from_iter(indices.iter().map(|&i| y[i]))
}

/// Selects entries of a label slice.
pub fn select_labels(labels: &[usize], indices: &[usize]) -> Vec<usize> {
    indices.iter().map(|&i| labels[i]).collect()
}

/// Polynomial feature expansion of a single column up to `degree` (columns `x, x², ..., x^degree`).
pub fn polynomial_features(x: VR2D, degree: usize) -> Result<R2D, ErrorsJSL> {
    if x.ncols() != 1 || degree == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Polynomial expansion needs one input column and degree >= 1."));
    }
    Ok(R2D::from_shape_fn((x.nrows(), degree), |(i, j)| x[[i, 0]].powi(j as i32 + 1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn scalers_round_trip_and_normalise() {
        let x = arr2(&[[1.0, 10.0, 5.0], [2.0, 20.0, 5.0], [3.0, 60.0, 5.0]]);
        let mut s = StandardScaler::new();
        let z = s.fit_transform(x.view()).unwrap();
        for c in 0..2 {
            let col = z.column(c);
            assert!(col.sum().abs() < 1e-12 && (col.iter().map(|v| v * v).sum::<f64>() / 3.0 - 1.0).abs() < 1e-12);
        }
        assert!(z.column(2).iter().all(|v| *v == 0.0)); // constant column only centred
        let back = s.inverse_transform(z.view()).unwrap();
        assert!(back.iter().zip(x.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
        let mut m = MinMaxScaler::new();
        let u = m.fit_transform(x.view()).unwrap();
        assert_eq!((u[[0, 1]], u[[2, 1]], u[[1, 2]]), (0.0, 1.0, 0.0));
        assert!(StandardScaler::new().transform(x.view()).is_err());
        assert!(s.transform(arr2(&[[1.0]]).view()).is_err());
    }

    #[test]
    fn splitting_and_selection() {
        let (train, test) = train_test_split(10, 0.3, 1).unwrap();
        assert_eq!((train.len(), test.len()), (7, 3));
        let mut all: Vec<usize> = train.iter().chain(&test).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        assert_eq!(train_test_split(10, 0.3, 1).unwrap(), (train, test)); // deterministic
        assert!(train_test_split(1, 0.5, 1).is_err() && train_test_split(10, 1.0, 1).is_err());
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        assert_eq!(select_rows(x.view(), &[2, 0]), arr2(&[[5.0, 6.0], [1.0, 2.0]]));
        assert_eq!(select_labels(&[7, 8, 9], &[2, 2, 0]), vec![9, 9, 7]);
        let p = polynomial_features(arr2(&[[2.0], [3.0]]).view(), 3).unwrap();
        assert_eq!(p, arr2(&[[2.0, 4.0, 8.0], [3.0, 9.0, 27.0]]));
        assert!(polynomial_features(x.view(), 2).is_err());
    }
}
