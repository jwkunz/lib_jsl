//! Dense linear algebra in pure Rust for real (`f64`) and complex (`Complex<f64>`) matrices.
//!
//! | Operation | Trait | Algorithm |
//! |---|---|---|
//! | Solve `A x = b`, inverse, determinant | [`Solve`], [`Inverse`], [`Factorize`], [`Determinant`] | LU with partial pivoting |
//! | Hermitian positive-definite factorisation | [`Cholesky`] | Cholesky `A = L Lᴴ` |
//! | Triangular systems | [`SolveTriangular`] | Forward and back substitution |
//! | Hermitian / symmetric eigenproblem | [`Eigh`], [`EigValsh`] | Householder tridiagonalisation and implicit QL |
//! | General eigenproblem | [`Eig`], [`EigVals`] | Hessenberg reduction and shifted QR (complex Schur form) |
//! | Singular value decomposition | [`SVD`], [`SvdJacobi`] | Golub-Kahan bidiagonalisation; one-sided Jacobi for high relative accuracy |
//! | Least squares, minimum-norm for rank-deficient systems | [`LeastSquaresSvd`] | Householder QR with column pivoting; SVD when rank-deficient |
//!
//! The traits follow the names and signatures of the `ndarray-linalg` crate, so code written
//! against it ports by changing the import. They are implemented for every ndarray
//! `ArrayBase<S, Ix2>` (owned arrays and views). Everything is written for clarity and moderate
//! sizes (up to a few hundred rows); for large problems an optimised LAPACK will be faster. The
//! benchmarks in `crates/linalg_bench` compare the two.
//!
//! See the book chapter [Dense linear algebra](https://jwkunz.github.io/lib_jsl/book/linear_algebra/index.html).

// Matrix algorithms read most clearly with explicit row and column indices.
#![allow(clippy::needless_range_loop)]

mod cholesky;
mod dense;
mod eig;
mod eigh;
mod error;
mod lu;
mod qr;
mod reflectors;
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
    /// Solves `Aᵀ x = b` (transpose, without conjugation).
    fn solve_t<S: Data<Elem = T>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<T>>;
    /// As [`Solve::solve_t`], consuming the right-hand side.
    fn solve_t_into(&self, b: Array1<T>) -> Result<Array1<T>> {
        self.solve_t(&b)
    }
    /// Solves `Aᴴ x = b` (conjugate transpose).
    fn solve_h<S: Data<Elem = T>>(&self, b: &ArrayBase<S, Ix1>) -> Result<Array1<T>>;
    /// As [`Solve::solve_h`], consuming the right-hand side.
    fn solve_h_into(&self, b: Array1<T>) -> Result<Array1<T>> {
        self.solve_h(&b)
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

/// The SVD by the one-sided Jacobi method, with the same output as [`SVD::svd`]. It is slower than
/// the default (Golub-Kahan) method but determines small singular values to high *relative*
/// accuracy, which matters for graded or badly scaled matrices.
pub trait SvdJacobi<T> {
    #[allow(clippy::type_complexity)]
    fn svd_jacobi(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Array2<T>>, Array1<f64>, Option<Array2<T>>)>;
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

/// The minimum-norm least-squares solution of `A x ≈ b`, robust to rank deficiency.
///
/// Overdetermined systems are first factorised by Householder QR with column pivoting; when that
/// shows full column rank, the solution follows by back-substitution and the singular values from
/// the eigenvalues of `RᴴR`. Rank-deficient and underdetermined systems use the SVD, which gives
/// the minimum-norm solution.
pub trait LeastSquaresSvd<T> {
    fn least_squares<S: Data<Elem = T>>(&self, b: &ArrayBase<S, Ix1>) -> Result<LeastSquaresResult<T>>;
}

impl<T: Scalar, S: Data<Elem = T>> Solve<T> for ArrayBase<S, Ix2> {
    fn solve<S2: Data<Elem = T>>(&self, b: &ArrayBase<S2, Ix1>) -> Result<Array1<T>> {
        let lu = self.factorize()?;
        lu.solve(b)
    }
    fn solve_t<S2: Data<Elem = T>>(&self, b: &ArrayBase<S2, Ix1>) -> Result<Array1<T>> {
        self.t().solve(b)
    }
    fn solve_h<S2: Data<Elem = T>>(&self, b: &ArrayBase<S2, Ix1>) -> Result<Array1<T>> {
        self.t().mapv(|v| v.conj()).solve(b)
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
        Ok(Array1::from_vec(eigh::eigvalsh(&Mat::from_array(self)?, uplo)?))
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
        Ok(Array1::from_vec(eig::eigvals(&Mat::from_array(self)?)?))
    }
}

impl<T: Scalar, S: Data<Elem = T>> SVD<T> for ArrayBase<S, Ix2> {
    fn svd(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Array2<T>>, Array1<f64>, Option<Array2<T>>)> {
        let a = Mat::from_array(self)?;
        if !calc_u && !calc_vt {
            return Ok((None, Array1::from_vec(svd::singular_values(&a)?), None));
        }
        Ok(full_factors(svd::thin_svd(&a)?, calc_u, calc_vt))
    }
}

impl<T: Scalar, S: Data<Elem = T>> SvdJacobi<T> for ArrayBase<S, Ix2> {
    fn svd_jacobi(&self, calc_u: bool, calc_vt: bool) -> Result<(Option<Array2<T>>, Array1<f64>, Option<Array2<T>>)> {
        Ok(full_factors(svd::thin_svd_jacobi(&Mat::from_array(self)?)?, calc_u, calc_vt))
    }
}

/// Completes a thin SVD to square `U` and `Vᴴ` as requested.
#[allow(clippy::type_complexity)]
fn full_factors<T: Scalar>(thin: svd::ThinSvd<T>, calc_u: bool, calc_vt: bool) -> (Option<Array2<T>>, Array1<f64>, Option<Array2<T>>) {
    let u = calc_u.then(|| svd::complete_square(&thin.u).into_array());
    let vt = calc_vt.then(|| svd::complete_square(&thin.v).adjoint().into_array());
    (u, Array1::from_vec(thin.sigma), vt)
}

impl<T: Scalar, S: Data<Elem = T>> LeastSquaresSvd<T> for ArrayBase<S, Ix2> {
    fn least_squares<S2: Data<Elem = T>>(&self, b: &ArrayBase<S2, Ix1>) -> Result<LeastSquaresResult<T>> {
        let (m, n) = self.dim();
        if b.len() != m {
            return Err(LinalgError::Shape(format!("matrix {m} × {n} and right-hand side of length {}", b.len())));
        }
        let a = Mat::from_array(self)?;
        let rhs = vector(b)?;
        // ANCHOR: least_squares
        if m >= n && n > 0 {
            let qr = qr::householder_pivoted(&a);
            let mut qtb = rhs.clone();
            qr.apply_adjoint(&mut qtb);
            let r00 = qr.r.at(0, 0).abs();
            let tolerance = r00 * f64::EPSILON * m.max(n) as f64;
            if r00 > 0.0 && (0..n).all(|i| qr.r.at(i, i).abs() > tolerance * 1e3) {
                // Full column rank: back-substitute R z = (Qᴴ b)[..n], then undo the pivoting.
                let mut z = qtb[..n].to_vec();
                for i in (0..n).rev() {
                    let s = (i + 1..n).fold(T::zero(), |s, k| s + qr.r.at(i, k) * z[k]);
                    z[i] = (z[i] - s) / qr.r.at(i, i);
                }
                let mut x = vec![T::zero(); n];
                for (j, &p) in qr.permutation.iter().enumerate() {
                    x[p] = z[j];
                }
                // The singular values of A are those of R. (Taking them as the square roots of
                // eig(RᴴR) squared the condition number: σ = 1e-10 next to σ = 1 came out 3.5e-9.)
                let singular_values = Array1::from_vec(svd::thin_svd(&qr.upper())?.sigma);
                let residual = (m > n).then(|| qtb[n..].iter().map(|v| v.abs_sqr()).sum::<f64>());
                return Ok(LeastSquaresResult { solution: to_array1(x), singular_values, rank: n, residual_sum_of_squares: residual });
            }
        }
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
        // ANCHOR_END: least_squares
    }
}

#[cfg(test)]
mod tests;
