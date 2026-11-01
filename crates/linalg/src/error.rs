//! Errors and the triangle and diagonal selectors shared by the decompositions.

use std::fmt;

/// Why a decomposition or solve failed.
#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {
    /// The operation needs a square matrix but got `rows × cols`.
    NotSquare { rows: usize, cols: usize },
    /// Operand shapes do not fit together.
    Shape(String),
    /// The matrix is singular (an exact or numerically zero pivot at this step).
    Singular(usize),
    /// Cholesky found a non-positive pivot: the matrix is not positive definite.
    NotPositiveDefinite(usize),
    /// An iterative algorithm did not converge within its iteration budget.
    NoConvergence(&'static str),
    /// The matrix contains NaN or infinity.
    NonFinite,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::NotSquare { rows, cols } => write!(f, "expected a square matrix, got {rows} × {cols}"),
            LinalgError::Shape(s) => write!(f, "incompatible shapes: {s}"),
            LinalgError::Singular(k) => write!(f, "the matrix is singular (zero pivot at step {k})"),
            LinalgError::NotPositiveDefinite(k) => write!(f, "the matrix is not positive definite (pivot {k})"),
            LinalgError::NoConvergence(what) => write!(f, "{what} did not converge"),
            LinalgError::NonFinite => write!(f, "the matrix contains NaN or infinite entries"),
        }
    }
}

impl std::error::Error for LinalgError {}

/// Which triangle of a matrix to use (or produce).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UPLO {
    Upper,
    Lower,
}

/// Whether a triangular matrix has an implicit unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    Unit,
    NonUnit,
}

pub type Result<T> = std::result::Result<T, LinalgError>;
