//! Readable reference implementations of the classical dense linear algebra algorithms, kept for
//! study and as benchmark baselines.
//!
//! **If you just want to solve a system or decompose a matrix, use `lib_jsl_linalg` instead.**
//! The routines here work on real `f64` matrices, favour the shape of the textbook formulas over
//! speed, and several are included precisely to show *why* production code does something else:
//!
//! | Module | Algorithms | The lesson |
//! |---|---|---|
//! | [`gaussian_elimination`] | Elimination without and with partial pivoting | Pivoting is what makes elimination stable |
//! | [`lu`] | Doolittle LU, triangular solves, determinant, inverse | Factor once, solve many times |
//! | [`cholesky`] | Cholesky-Banachiewicz | Symmetry halves the work and needs no pivoting |
//! | [`qr`] | Classical and modified Gram-Schmidt, Householder | Orthogonality is lost by Gram-Schmidt and kept by reflections |
//! | [`power_iteration`] | Power, inverse and Rayleigh-quotient iteration | Repeated multiplication finds dominant eigenvectors; shifts make it fast |
//! | [`jacobi_eigenvalue`] | Cyclic Jacobi rotations | Annihilating off-diagonal entries one at a time converges quadratically |
//! | [`qr_algorithm`] | Unshifted and shifted QR iteration | The QR algorithm is subspace iteration in disguise; shifts accelerate it |
//! | [`naive_svd`] | SVD from the eigenvalues of `AᵀA` | Forming `AᵀA` squares the condition number |
//! | [`least_squares`] | Normal equations versus QR | The same trade-off for least squares |
//!
//! Run `cargo bench -p lib_jsl_linalg_reference` to compare them with the production crate.

pub mod cholesky;
pub mod gaussian_elimination;
pub mod jacobi_eigenvalue;
pub mod least_squares;
pub mod lu;
pub mod naive_svd;
pub mod power_iteration;
pub mod qr;
pub mod qr_algorithm;

use ndarray::Array2;

/// Why a reference routine failed.
pub type Error = &'static str;

/// The identity matrix of order `n`.
pub(crate) fn identity(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { 1.0 } else { 0.0 })
}

/// A deterministic pseudo-random matrix with entries in [−1, 1), for tests and benchmarks.
pub fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    })
}

/// A random symmetric matrix `G + Gᵀ`.
pub fn random_symmetric(n: usize, seed: u64) -> Array2<f64> {
    let g = random_matrix(n, n, seed);
    &g + &g.t()
}

/// A random symmetric positive-definite matrix `G Gᵀ + n I`.
pub fn random_positive_definite(n: usize, seed: u64) -> Array2<f64> {
    let g = random_matrix(n, n, seed);
    g.dot(&g.t()) + identity(n) * n as f64
}

/// The off-diagonal Frobenius norm, used to measure how far a matrix is from diagonal.
pub fn off_diagonal_norm(a: &Array2<f64>) -> f64 {
    a.indexed_iter().filter(|((i, j), _)| i != j).map(|(_, v)| v * v).sum::<f64>().sqrt()
}
