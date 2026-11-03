//! Kernel functions and Gram matrices, shared by the kernel methods (SVR, SVM, kernel PCA,
//! spectral clustering).
//!
//! See the book entry [Support vector machines and kernels](https://jwkunz.github.io/lib_jsl/book/machine_learning/support-vector-machines.html).

use crate::common::squared_distance;
use lib_jsl_core::{R2D, VR2D};

/// A positive-definite similarity `k(a, b)` between two feature vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// `a · b`
    Linear,
    /// `(γ a·b + coef0)^degree`
    Polynomial { degree: u32, gamma: f64, coef0: f64 },
    /// `exp(-γ ‖a - b‖²)`
    Rbf { gamma: f64 },
    /// `tanh(γ a·b + coef0)` (not positive definite for all parameters)
    Sigmoid { gamma: f64, coef0: f64 },
}

impl Kernel {
    pub fn evaluate(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot = || a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        match *self {
            Kernel::Linear => dot(),
            Kernel::Polynomial { degree, gamma, coef0 } => (gamma * dot() + coef0).powi(degree as i32),
            Kernel::Rbf { gamma } => (-gamma * squared_distance(a, b)).exp(),
            Kernel::Sigmoid { gamma, coef0 } => (gamma * dot() + coef0).tanh(),
        }
    }

    /// The Gram matrix `K[i][j] = k(x_i, x_j)` of the rows of `x`.
    pub fn gram(&self, x: VR2D) -> R2D {
        let rows: Vec<Vec<f64>> = x.rows().into_iter().map(|r| r.to_vec()).collect();
        let mut k = R2D::zeros((rows.len(), rows.len()));
        for i in 0..rows.len() {
            for j in i..rows.len() {
                let v = self.evaluate(&rows[i], &rows[j]);
                k[[i, j]] = v;
                k[[j, i]] = v;
            }
        }
        k
    }

    /// The cross-kernel `K[i][j] = k(a_i, b_j)` between the rows of `a` and `b`.
    pub fn cross(&self, a: VR2D, b: VR2D) -> R2D {
        R2D::from_shape_fn((a.nrows(), b.nrows()), |(i, j)| self.evaluate(&a.row(i).to_vec(), &b.row(j).to_vec()))
    }

    /// An RBF kernel with the common heuristic `γ = 1 / (features · variance)` of the data.
    pub fn rbf_scaled(x: VR2D) -> Kernel {
        let n = (x.nrows() * x.ncols()).max(1) as f64;
        let mean = x.sum() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        Kernel::Rbf { gamma: 1.0 / (x.ncols() as f64 * var.max(1e-12)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn values_and_gram_properties() {
        let (a, b) = ([1.0, 2.0], [3.0, -1.0]);
        assert_eq!(Kernel::Linear.evaluate(&a, &b), 1.0);
        assert_eq!(Kernel::Polynomial { degree: 2, gamma: 1.0, coef0: 1.0 }.evaluate(&a, &b), 4.0);
        assert!((Kernel::Rbf { gamma: 0.5 }.evaluate(&a, &b) - (-0.5f64 * 13.0).exp()).abs() < 1e-15);
        assert_eq!(Kernel::Rbf { gamma: 2.0 }.evaluate(&a, &a), 1.0);
        assert!((Kernel::Sigmoid { gamma: 1.0, coef0: 0.0 }.evaluate(&a, &b) - 1.0f64.tanh()).abs() < 1e-15);
        let x = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]);
        let g = Kernel::Rbf { gamma: 1.0 }.gram(x.view());
        assert!((0..3).all(|i| g[[i, i]] == 1.0) && (0..3).all(|i| (0..3).all(|j| g[[i, j]] == g[[j, i]])));
        assert_eq!(Kernel::Linear.cross(x.view(), x.view()), Kernel::Linear.gram(x.view()));
        assert!(matches!(Kernel::rbf_scaled(x.view()), Kernel::Rbf { gamma } if gamma > 0.0));
    }
}
