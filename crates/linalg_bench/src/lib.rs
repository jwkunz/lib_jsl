//! Shared test matrices for the `lib_jsl_linalg` versus `ndarray-linalg` benchmarks.
//!
//! This crate is excluded from the workspace's default members because `ndarray-linalg` needs the
//! system OpenBLAS library. Run the comparison with
//! `cargo bench -p lib_jsl_linalg_bench` (append `-- --quick` for a fast pass).

// Link the BLAS/LAPACK libraries that ndarray-linalg depends on.
use ndarray_linalg as _;

use ndarray::{Array1, Array2};

/// A deterministic pseudo-random matrix with entries in [−1, 1).
pub fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    })
}

/// A random right-hand side.
pub fn random_vector(n: usize, seed: u64) -> Array1<f64> {
    random_matrix(n, 1, seed).column(0).to_owned()
}

/// A symmetric matrix `G + Gᵀ`.
pub fn symmetric(n: usize, seed: u64) -> Array2<f64> {
    let g = random_matrix(n, n, seed);
    &g + &g.t()
}

/// A symmetric positive-definite matrix `G Gᵀ + n I`.
pub fn positive_definite(n: usize, seed: u64) -> Array2<f64> {
    let g = random_matrix(n, n, seed);
    g.dot(&g.t()) + Array2::<f64>::eye(n) * n as f64
}


/// Defines the benchmarked operations for one library with ordinary method-call syntax, so that
/// both libraries' traits resolve the same way.
macro_rules! operations {
    ($name:ident, $crate_path:ident) => {
        pub mod $name {
            use ndarray::{Array1, Array2};
            use $crate_path::{Cholesky, Eig, Eigh, Inverse, LeastSquaresSvd, SVD, Solve, UPLO};
            use num_complex::Complex;

            pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
                a.solve(b).unwrap()
            }
            pub fn inverse(a: &Array2<f64>) -> Array2<f64> {
                a.inv().unwrap()
            }
            pub fn cholesky(a: &Array2<f64>) -> Array2<f64> {
                a.cholesky(UPLO::Lower).unwrap()
            }
            pub fn eigh(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
                a.eigh(UPLO::Lower).unwrap()
            }
            pub fn eig(a: &Array2<f64>) -> (Array1<Complex<f64>>, Array2<Complex<f64>>) {
                a.eig().unwrap()
            }
            pub fn singular_values(a: &Array2<f64>) -> Array1<f64> {
                a.svd(false, false).unwrap().1
            }
            pub fn svd(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
                let (u, s, vt) = a.svd(true, true).unwrap();
                (u.unwrap(), s, vt.unwrap())
            }
            pub fn least_squares(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
                a.least_squares(b).unwrap().solution
            }
        }
    };
}

operations!(ours, lib_jsl_linalg);
operations!(lapack, ndarray_linalg);

#[cfg(test)]
mod tests {
    use super::*;

    /// The two libraries agree, so the benchmarks compare like with like.
    #[test]
    fn implementations_agree() {
        let n = 40;
        let (a, b) = (random_matrix(n, n, 1), random_vector(n, 2));
        assert!((ours::solve(&a, &b) - lapack::solve(&a, &b)).iter().all(|v| v.abs() < 1e-9));
        let s = symmetric(n, 3);
        assert!((ours::eigh(&s).0 - lapack::eigh(&s).0).iter().all(|v| v.abs() < 1e-10));
        let r = random_matrix(60, n, 4);
        assert!((ours::singular_values(&r) - lapack::singular_values(&r)).iter().all(|v| v.abs() < 1e-10));
        let p = positive_definite(n, 5);
        assert!((ours::cholesky(&p) - lapack::cholesky(&p)).iter().all(|v| v.abs() < 1e-10));
        let rb = random_vector(60, 6);
        assert!((ours::least_squares(&r, &rb) - lapack::least_squares(&r, &rb)).iter().all(|v| v.abs() < 1e-9));
        let mut e1: Vec<f64> = ours::eig(&a).0.iter().map(|z| z.norm()).collect();
        let mut e2: Vec<f64> = lapack::eig(&a).0.iter().map(|z| z.norm()).collect();
        e1.sort_by(f64::total_cmp);
        e2.sort_by(f64::total_cmp);
        assert!(e1.iter().zip(&e2).all(|(x, y)| (x - y).abs() < 1e-9));
    }
}
