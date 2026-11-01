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

    /// `self · other`.
    pub fn matmul(&self, other: &Self) -> Self {
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.at(i, k);
                if a == T::zero() {
                    continue;
                }
                let row = &other.data[k * other.cols..(k + 1) * other.cols];
                let dst = &mut out.data[i * other.cols..(i + 1) * other.cols];
                for (d, &b) in dst.iter_mut().zip(row) {
                    *d += a * b;
                }
            }
        }
        out
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
