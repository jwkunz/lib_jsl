//! Pseudoinverses, rank, orthonormal bases of the range and null space, matrix norms and condition
//! numbers, all from the singular value decomposition (SciPy's `pinv`, `null_space`, `orth`;
//! NumPy's `matrix_rank`, `norm` and `cond`).
//!
//! See the book entry [Matrix functions, subspaces and norms](https://jwkunz.github.io/lib_jsl/book/linear_algebra/subspaces-norms.html).

use crate::error::{LinalgError, Result};
use crate::scalar::Scalar;
use crate::SVD;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, s};

fn svd_full<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<(Array2<T>, Array1<f64>, Array2<T>)> {
    let (u, sv, vt) = a.svd(true, true)?;
    Ok((u.expect("requested"), sv, vt.expect("requested")))
}

fn rank_of(sv: &Array1<f64>, rtol: f64) -> usize {
    let smax = sv.iter().copied().fold(0.0, f64::max);
    sv.iter().filter(|&&s| s > rtol * smax).count()
}

// ANCHOR: pinv
/// The Moore-Penrose pseudoinverse `A⁺ = V Σ⁺ Uᴴ` (SciPy's `pinv`): singular values at most
/// `rtol · σ_max` are treated as zero, with `rtol = max(m, n) ε` by default.
pub fn pinv<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, rtol: Option<f64>) -> Result<Array2<T>> {
    let (m, n) = a.dim();
    let (u, sv, vt) = svd_full(a)?;
    let rtol = rtol.unwrap_or(m.max(n) as f64 * f64::EPSILON);
    let r = rank_of(&sv, rtol);
    let mut out = Array2::<T>::from_elem((n, m), T::zero());
    for k in 0..r {
        let inv = 1.0 / sv[k];
        for i in 0..n {
            let vik = vt[(k, i)].conj().scale(inv);
            for j in 0..m {
                out[(i, j)] += vik * u[(j, k)].conj();
            }
        }
    }
    Ok(out)
}
// ANCHOR_END: pinv

/// The numerical rank: the number of singular values above `tol` (NumPy's default
/// `σ_max · max(m, n) · ε` when `None`).
pub fn matrix_rank<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, tol: Option<f64>) -> Result<usize> {
    let (_, sv, _) = a.svd(false, false)?;
    let (m, n) = a.dim();
    let smax = sv.iter().copied().fold(0.0, f64::max);
    let tol = tol.unwrap_or(smax * m.max(n) as f64 * f64::EPSILON);
    Ok(sv.iter().filter(|&&s| s > tol).count())
}

/// An orthonormal basis of the null space of `A`, as columns (SciPy's `null_space`): the right
/// singular vectors of the singular values at most `rcond · σ_max` (default `max(m, n) ε`).
pub fn null_space<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, rcond: Option<f64>) -> Result<Array2<T>> {
    let (m, n) = a.dim();
    let (_, sv, vt) = svd_full(a)?;
    let r = rank_of(&sv, rcond.unwrap_or(m.max(n) as f64 * f64::EPSILON));
    Ok(vt.slice(s![r.., ..]).t().mapv(|v| v.conj()))
}

/// An orthonormal basis of the range (column space) of `A`, as columns (SciPy's `orth`).
pub fn orth<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, rcond: Option<f64>) -> Result<Array2<T>> {
    let (m, n) = a.dim();
    let (u, sv, _) = svd_full(a)?;
    let r = rank_of(&sv, rcond.unwrap_or(m.max(n) as f64 * f64::EPSILON));
    Ok(u.slice(s![.., ..r]).to_owned())
}

/// Which matrix norm [`norm`] computes (NumPy's `ord`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// `√Σ|aᵢⱼ|²`.
    Frobenius,
    /// The sum of the singular values.
    Nuclear,
    /// The largest column sum of absolute values.
    One,
    /// The smallest column sum.
    MinusOne,
    /// The largest singular value.
    Two,
    /// The smallest singular value.
    MinusTwo,
    /// The largest row sum.
    Infinity,
    /// The smallest row sum.
    MinusInfinity,
}

// ANCHOR: norm
/// A matrix norm (or, for the `Minus` variants, the corresponding minimum), as NumPy's `norm`.
pub fn norm<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, ord: Norm) -> Result<f64> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err(LinalgError::Shape("empty matrix".into()));
    }
    let col_sums = || (0..n).map(|j| (0..m).map(|i| a[(i, j)].abs()).sum::<f64>());
    let row_sums = || (0..m).map(|i| (0..n).map(|j| a[(i, j)].abs()).sum::<f64>());
    Ok(match ord {
        Norm::Frobenius => a.iter().map(|v| v.abs_sqr()).sum::<f64>().sqrt(),
        Norm::One => col_sums().fold(0.0, f64::max),
        Norm::MinusOne => col_sums().fold(f64::INFINITY, f64::min),
        Norm::Infinity => row_sums().fold(0.0, f64::max),
        Norm::MinusInfinity => row_sums().fold(f64::INFINITY, f64::min),
        Norm::Two | Norm::MinusTwo | Norm::Nuclear => {
            let (_, sv, _) = a.svd(false, false)?;
            match ord {
                Norm::Two => sv.iter().copied().fold(0.0, f64::max),
                Norm::MinusTwo => sv.iter().copied().fold(f64::INFINITY, f64::min),
                _ => sv.sum(),
            }
        }
    })
}
// ANCHOR_END: norm

/// The condition number `‖A‖ ‖A⁻¹‖` in the given norm (NumPy's `cond`); for the 2-norm,
/// `σ_max / σ_min`, which is also defined for rectangular matrices. Infinite for singular
/// matrices.
pub fn cond<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, ord: Norm) -> Result<f64> {
    match ord {
        Norm::Two | Norm::MinusTwo => {
            let (_, sv, _) = a.svd(false, false)?;
            let (big, small) = (sv.iter().copied().fold(0.0, f64::max), sv.iter().copied().fold(f64::INFINITY, f64::min));
            Ok(if ord == Norm::Two { big / small } else { small / big })
        }
        _ => {
            let inv = match crate::Inverse::inv(a) {
                Ok(i) => i,
                Err(LinalgError::Singular(_)) => return Ok(f64::INFINITY),
                Err(e) => return Err(e),
            };
            Ok(norm(a, ord)? * norm(&inv, ord)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pinv_and_null_space_of_a_rank_two_matrix() {
        let a = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        assert_eq!(matrix_rank(&a, None).unwrap(), 2);
        let p = pinv(&a, None).unwrap();
        // Penrose conditions: A A⁺ A = A and A⁺ A A⁺ = A⁺.
        let apa = a.dot(&p).dot(&a);
        assert!(apa.iter().zip(a.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
        let ns = null_space(&a, None).unwrap();
        assert_eq!(ns.ncols(), 1);
        assert!(a.dot(&ns).iter().all(|v| v.abs() < 1e-12));
        assert_eq!(orth(&a, None).unwrap().ncols(), 2);
        let id = array![[2.0, 0.0], [0.0, 0.5]];
        assert!((cond(&id, Norm::Two).unwrap() - 4.0).abs() < 1e-14);
        assert!((cond(&id, Norm::One).unwrap() - 4.0).abs() < 1e-14);
        assert!((norm(&id, Norm::Nuclear).unwrap() - 2.5).abs() < 1e-14);
    }
}
