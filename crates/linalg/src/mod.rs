//! Dense linear algebra in pure Rust for real (`f64`) and complex (`Complex<f64>`) matrices.
//!
//! | Operation | Trait | Algorithm |
//! |---|---|---|
//! | Solve `A x = b`, inverse, determinant | [`Solve`], [`Inverse`], [`Factorize`], [`Determinant`] | LU with partial pivoting |
//! | Hermitian positive-definite factorisation | [`Cholesky`] | Cholesky `A = L Lᴴ` |
//! | Triangular systems | [`SolveTriangular`] | Forward and back substitution |
//! | Hermitian / symmetric eigenproblem | [`Eigh`], [`EigValsh`] | Householder tridiagonalisation and implicit QL |
//! | General eigenproblem | [`Eig`], [`EigVals`] | Hessenberg reduction and shifted QR (complex Schur form) |
//! | Singular value decomposition | [`SVD`] | One-sided Jacobi |
//! | Least squares, minimum-norm for rank-deficient systems | [`LeastSquaresSvd`] | SVD |
//!
//! The traits follow the names and signatures of the `ndarray-linalg` crate, so code written
//! against it ports by changing the import. They are implemented for every ndarray
//! `ArrayBase<S, Ix2>` (owned arrays and views). Everything is written for clarity and moderate
//! sizes (up to a few hundred rows); for large problems an optimised LAPACK will be faster. The
//! benchmarks in `crates/linalg_bench` compare the two.

// Matrix algorithms read most clearly with explicit row and column indices.
#![allow(clippy::needless_range_loop)]

mod cholesky;
mod dense;
mod eig;
mod eigh;
mod error;
mod lu;
mod scalar;
mod svd;

pub use error::{Diag, LinalgError, Result, UPLO};
pub use lu::LuFactorization;
pub use scalar::Scalar;

use dense::{Mat, require_square, to_array1, vector};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use num_complex::Complex;

/// Solves `A x = b` for square `A` by LU with partial pivoting.
pub trait Solve<T> {
    fn solve<S: Data<Elem = T>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<T>>;
    /// As [`Solve::solve`], consuming the right-hand side.
    fn solve_into(&self, b: Array1<T>) -> Result<Array1<T>> {
        self.solve(&b)
    }
}

/// The inverse of a square matrix.
pub trait Inverse<T> {
    fn inv(&self) -> Result<Array2<T>>;
}

/// The determinant of a square matrix.
pub trait Determinant<T> {
    fn det(&self) -> Result<T>;
}

/// An LU factorisation to solve many systems with the same matrix.
pub trait Factorize<T> {
    fn factorize(&self) -> Result<LuFactorization<T>>;
}

/// The Cholesky factor of a Hermitian positive-definite matrix whose `uplo` triangle is read:
/// lower `L` with `A = L Lᴴ` for [`UPLO::Lower`], upper `U = Lᴴ` for [`UPLO::Upper`].
pub trait Cholesky<T> {
    fn cholesky(&self, uplo: UPLO) -> Result<Array2<T>>;
}

/// Solves `T X = B` for a triangular `T` (only the `uplo` triangle is read) and a right-hand side
/// that is a vector or a matrix.
pub trait SolveTriangular<T, Rhs> {
    type Output;
    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &Rhs) -> Result<Self::Output>;
}

/// Eigenvalues (ascending) and orthonormal eigenvectors (columns) of a Hermitian matrix whose
/// `uplo` triangle is read.
pub trait Eigh<T> {
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<f64>, Array2<T>)>;
}

/// Eigenvalues (ascending) of a Hermitian matrix.
pub trait EigValsh {
    fn eigvalsh(&self, uplo: UPLO) -> Result<Array1<f64>>;
}

/// Complex eigenvalues and eigenvectors (as columns).
pub type ComplexEigen = (Array1<Complex<f64>>, Array2<Complex<f64>>);

/// Eigenvalues and unit eigenvectors (columns) of a general square matrix, both complex.
pub trait Eig {
    fn eig(&self) -> Result<ComplexEigen>;
}

/// Eigenvalues of a general square matrix.
pub trait EigVals {
    fn eigvals(&self) -> Result<Array1<Complex<f64>>>;
}

/// The singular value decomposition `A = U Σ Vᴴ`: `(U, σ, Vᴴ)` with the full square unitary
/// factors when requested and the `min(m, n)` singular values in descending order.
#[allow(clippy::upper_case_acronyms)]
pub trait SVD<T> {
    #[allow(clippy::type_complexity)]
    fn svd(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Array2<T>>, Array1<f64>, Option<Array2<T>>)>;
}

/// The result of [`LeastSquaresSvd::least_squares`].
#[derive(Debug, Clone)]
pub struct LeastSquaresResult<T> {
    /// The minimum-norm minimiser of `‖A x − b‖₂`.
    pub solution: Array1<T>,
    pub singular_values: Array1<f64>,
    /// The numerical rank (singular values above `max(m, n) · ε · σ_max`).
    pub rank: usize,
    /// `‖A x − b‖²` when the system is overdetermined and of full rank.
    pub residual_sum_of_squares: Option<f64>,
}

/// The minimum-norm least-squares solution of `A x ≈ b` via the SVD (robust to rank
/// deficiency).
pub trait LeastSquaresSvd<T> {
    fn least_squares<S: Data<Elem = T>>(&self, b: &ArrayBase<S, Ix1>) -> Result<LeastSquaresResult<T>>;
}

impl<T: Scalar, S: Data<Elem = T>> Solve<T> for ArrayBase<S, Ix2> {
    fn solve<S2: Data<Elem = T>>(&self, b: &ArrayBase<S2, Ix1>) -> Result<Array1<T>> {
        let lu = self.factorize()?;
        lu.solve(b)
    }
}

impl<T: Scalar> LuFactorization<T> {
    /// Solves `A x = b` with the stored factorisation.
    pub fn solve<S: Data<Elem = T>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<T>> {
        if b.len() != self.dimension() {
            return Err(LinalgError::Shape(format!("matrix of order {} and vector of length {}", self.dimension(), b.len())));
        }
        Ok(to_array1(self.solve_vec(&vector(b)?)))
    }

    /// The inverse of the factorised matrix.
    pub fn inv(&self) -> Array2<T> {
        self.inverse().into_array()
    }
}

impl<T: Scalar, S: Data<Elem = T>> Factorize<T> for ArrayBase<S, Ix2> {
    fn factorize(&self) -> Result<LuFactorization<T>> {
        LuFactorization::new(Mat::from_array(self)?)
    }
}

impl<T: Scalar, S: Data<Elem = T>> Inverse<T> for ArrayBase<S, Ix2> {
    fn inv(&self) -> Result<Array2<T>> {
        Ok(self.factorize()?.inv())
    }
}

impl<T: Scalar, S: Data<Elem = T>> Determinant<T> for ArrayBase<S, Ix2> {
    fn det(&self) -> Result<T> {
        require_square(self.nrows(), self.ncols())?;
        match self.factorize() {
            Ok(lu) => Ok(lu.determinant()),
            Err(LinalgError::Singular(_)) => Ok(T::zero()),
            Err(e) => Err(e),
        }
    }
}

impl<T: Scalar, S: Data<Elem = T>> Cholesky<T> for ArrayBase<S, Ix2> {
    fn cholesky(&self, uplo: UPLO) -> Result<Array2<T>> {
        let l = cholesky::cholesky_lower(&Mat::from_array(self)?, uplo)?;
        Ok(match uplo {
            UPLO::Lower => l.into_array(),
            UPLO::Upper => l.adjoint().into_array(),
        })
    }
}

impl<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>> SolveTriangular<T, ArrayBase<S2, Ix2>> for ArrayBase<S, Ix2> {
    type Output = Array2<T>;
    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &ArrayBase<S2, Ix2>) -> Result<Array2<T>> {
        Ok(cholesky::solve_triangular(&Mat::from_array(self)?, uplo, diag, &Mat::from_array(b)?)?.into_array())
    }
}

impl<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>> SolveTriangular<T, ArrayBase<S2, Ix1>> for ArrayBase<S, Ix2> {
    type Output = Array1<T>;
    fn solve_triangular(&self, uplo: UPLO, diag: Diag, b: &ArrayBase<S2, Ix1>) -> Result<Array1<T>> {
        let rhs = Mat { rows: b.len(), cols: 1, data: vector(b)? };
        Ok(to_array1(cholesky::solve_triangular(&Mat::from_array(self)?, uplo, diag, &rhs)?.data))
    }
}

impl<T: Scalar, S: Data<Elem = T>> Eigh<T> for ArrayBase<S, Ix2> {
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<f64>, Array2<T>)> {
        let (values, vectors) = eigh::eigh(&Mat::from_array(self)?, uplo)?;
        Ok((Array1::from_vec(values), vectors.into_array()))
    }
}

impl<T: Scalar, S: Data<Elem = T>> EigValsh for ArrayBase<S, Ix2> {
    fn eigvalsh(&self, uplo: UPLO) -> Result<Array1<f64>> {
        Ok(self.eigh(uplo)?.0)
    }
}

impl<T: Scalar, S: Data<Elem = T>> Eig for ArrayBase<S, Ix2> {
    fn eig(&self) -> Result<ComplexEigen> {
        let (values, vectors) = eig::eig(&Mat::from_array(self)?)?;
        Ok((Array1::from_vec(values), vectors.into_array()))
    }
}

impl<T: Scalar, S: Data<Elem = T>> EigVals for ArrayBase<S, Ix2> {
    fn eigvals(&self) -> Result<Array1<Complex<f64>>> {
        let a = Mat::from_array(self)?;
        require_square(a.rows, a.cols)?;
        let ac = Mat { rows: a.rows, cols: a.cols, data: a.data.iter().map(|v| v.to_complex()).collect() };
        let (t, _) = eig::schur(&ac)?;
        Ok(Array1::from_iter((0..t.rows).map(|i| t.at(i, i))))
    }
}

impl<T: Scalar, S: Data<Elem = T>> SVD<T> for ArrayBase<S, Ix2> {
    fn svd(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Array2<T>>, Array1<f64>, Option<Array2<T>>)> {
        let a = Mat::from_array(self)?;
        let thin = svd::thin_svd(&a)?;
        let u = calc_u.then(|| svd::complete_square(&thin.u).into_array());
        let vt = calc_vt.then(|| svd::complete_square(&thin.v).adjoint().into_array());
        Ok((u, Array1::from_vec(thin.sigma), vt))
    }
}

impl<T: Scalar, S: Data<Elem = T>> LeastSquaresSvd<T> for ArrayBase<S, Ix2> {
    fn least_squares<S2: Data<Elem = T>>(&self, b: &ArrayBase<S2, Ix1>) -> Result<LeastSquaresResult<T>> {
        let (m, n) = self.dim();
        if b.len() != m {
            return Err(LinalgError::Shape(format!("matrix {m} × {n} and right-hand side of length {}", b.len())));
        }
        let a = Mat::from_array(self)?;
        let rhs = vector(b)?;
        let thin = svd::thin_svd(&a)?;
        let threshold = thin.sigma.first().copied().unwrap_or(0.0) * f64::EPSILON * m.max(n) as f64;
        let rank = thin.sigma.iter().filter(|&&s| s > threshold).count();
        // x = V Σ⁺ Uᴴ b over the numerically non-zero singular values.
        let mut x = vec![T::zero(); n];
        for j in 0..rank {
            let coefficient = (0..m).fold(T::zero(), |s, i| s + thin.u.at(i, j).conj() * rhs[i]).scale(1.0 / thin.sigma[j]);
            for (i, xi) in x.iter_mut().enumerate() {
                *xi += thin.v.at(i, j) * coefficient;
            }
        }
        let residual_sum_of_squares = (m > n && rank == n).then(|| {
            (0..m).map(|i| (rhs[i] - (0..n).fold(T::zero(), |s, j| s + a.at(i, j) * x[j])).abs_sqr()).sum::<f64>()
        });
        Ok(LeastSquaresResult { solution: to_array1(x), singular_values: Array1::from_vec(thin.sigma), rank, residual_sum_of_squares })
    }
}

#[cfg(test)]
mod tests;
