//! A small row-major dense matrix used internally by the algorithms (contiguous storage and
//! unchecked-shape indexing keep the inner loops fast), with conversions to and from ndarray.

use crate::error::{LinalgError, Result};
use crate::scalar::Scalar;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mat<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Scalar> Mat<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![T::zero(); rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::one();
        }
        m
    }

    #[inline(always)]
    pub fn at(&self, i: usize, j: usize) -> T {
        self.data[i * self.cols + j]
    }

    #[inline(always)]
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.cols + j]
    }

    pub fn from_array<S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<Self> {
        let (rows, cols) = a.dim();
        let data: Vec<T> = a.iter().copied().collect();
        if data.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NonFinite);
        }
        Ok(Mat { rows, cols, data })
    }

    pub fn into_array(self) -> Array2<T> {
        Array2::from_shape_vec((self.rows, self.cols), self.data).expect("consistent shape")
    }

    /// The conjugate transpose.
    pub fn adjoint(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.data[j * self.rows + i] = self.at(i, j).conj();
            }
        }
        t
    }
}


/// The block size of the blocked factorisations: large enough for the matrix-multiply kernel to
/// run efficiently, small enough for the unblocked panel work to stay in cache.
pub(crate) const BLOCK: usize = 32;

/// `C ← C − A B` for row-major blocks (`A` is `m × k`, `B` is `k × n`, `C` is `m × n`), each given
/// by a pointer to its first element and its row stride.
///
/// # Safety
/// The blocks must lie inside live allocations, and `C` must not overlap `A` or `B`. They may lie
/// in the same buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sub_matmul<T: Scalar>(m: usize, k: usize, n: usize, a: *const T, lda: usize, b: *const T, ldb: usize, c: *mut T, ldc: usize) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    unsafe { T::gemm_sub(m, k, n, a, lda as isize, 1, b, ldb as isize, 1, c, ldc as isize, 1) }
}

/// `Σ x_i conj(y_i)` with four independent accumulators, so the additions pipeline.
#[inline]
pub(crate) fn dot_conj<T: Scalar>(x: &[T], y: &[T]) -> T {
    let n = x.len().min(y.len());
    let (x, y) = (&x[..n], &y[..n]);
    let mut acc = [T::zero(); 4];
    let chunks = n / 4;
    for c in 0..chunks {
        let (xs, ys) = (&x[4 * c..4 * c + 4], &y[4 * c..4 * c + 4]);
        for l in 0..4 {
            acc[l] += xs[l] * ys[l].conj();
        }
    }
    let mut tail = T::zero();
    for i in 4 * chunks..n {
        tail += x[i] * y[i].conj();
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `y ← y − a x`.
#[inline]
pub(crate) fn axpy_sub<T: Scalar>(y: &mut [T], a: T, x: &[T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi -= a * xi;
    }
}

pub(crate) fn require_square(rows: usize, cols: usize) -> Result<()> {
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols });
    }
    Ok(())
}

pub(crate) fn vector<T: Scalar, S: Data<Elem = T>>(b: &ArrayBase<S, Ix1>) -> Result<Vec<T>> {
    let v: Vec<T> = b.iter().copied().collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(LinalgError::NonFinite);
    }
    Ok(v)
}

pub(crate) fn to_array1<T: Scalar>(v: Vec<T>) -> Array1<T> {
    Array1::from_vec(v)
}
