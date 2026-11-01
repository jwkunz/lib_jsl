//! LU factorisation with partial pivoting: `P A = L U`.

use crate::dense::{Mat, require_square};
use crate::error::{LinalgError, Result};
use crate::scalar::Scalar;

/// A factorised square matrix, reusable for many right-hand sides.
#[derive(Debug, Clone)]
pub struct LuFactorization<T> {
    /// `L` (unit lower, below the diagonal) and `U` (on and above) packed together.
    pub(crate) lu: Mat<T>,
    /// Row `i` of `P A` is row `pivots[i]` of `A`.
    pub(crate) pivots: Vec<usize>,
}

impl<T: Scalar> LuFactorization<T> {
    pub(crate) fn new(mut a: Mat<T>) -> Result<Self> {
        require_square(a.rows, a.cols)?;
        let n = a.rows;
        let mut pivots: Vec<usize> = (0..n).collect();
        let scale = a.data.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        for k in 0..n {
            // Partial pivoting: the largest entry in column k at or below the diagonal.
            let (p, big) = (k..n).map(|i| (i, a.at(i, k).abs())).fold((k, -1.0), |best, c| if c.1 > best.1 { c } else { best });
            if big <= f64::EPSILON * scale * 1e-3 || big == 0.0 {
                return Err(LinalgError::Singular(k));
            }
            if p != k {
                for j in 0..n {
                    a.data.swap(k * n + j, p * n + j);
                }
                pivots.swap(k, p);
            }
            let pivot = a.at(k, k);
            for i in k + 1..n {
                let m = a.at(i, k) / pivot;
                *a.at_mut(i, k) = m;
                if m == T::zero() {
                    continue;
                }
                let (top, bottom) = a.data.split_at_mut(i * n);
                let src = &top[k * n + k + 1..k * n + n];
                let dst = &mut bottom[k + 1..n];
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d -= m * s;
                }
            }
        }
        Ok(LuFactorization { lu: a, pivots })
    }

    pub fn dimension(&self) -> usize {
        self.lu.rows
    }

    /// Solves `A x = b` in place.
    pub(crate) fn solve_vec(&self, b: &[T]) -> Vec<T> {
        let n = self.lu.rows;
        let mut x: Vec<T> = self.pivots.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let row = &self.lu.data[i * n..i * n + i];
            let s = row.iter().zip(&x[..i]).fold(T::zero(), |acc, (&l, &v)| acc + l * v);
            x[i] -= s;
        }
        for i in (0..n).rev() {
            let row = &self.lu.data[i * n + i + 1..(i + 1) * n];
            let s = row.iter().zip(&x[i + 1..]).fold(T::zero(), |acc, (&u, &v)| acc + u * v);
            x[i] = (x[i] - s) / self.lu.at(i, i);
        }
        x
    }

    /// The inverse, column by column.
    pub(crate) fn inverse(&self) -> Mat<T> {
        let n = self.lu.rows;
        let mut inv = Mat::zeros(n, n);
        let mut e = vec![T::zero(); n];
        for j in 0..n {
            e.iter_mut().for_each(|v| *v = T::zero());
            e[j] = T::one();
            let col = self.solve_vec(&e);
            for i in 0..n {
                *inv.at_mut(i, j) = col[i];
            }
        }
        inv
    }

    /// The determinant.
    pub fn determinant(&self) -> T {
        let n = self.lu.rows;
        // The sign of the permutation from its cycle structure.
        let mut seen = vec![false; n];
        let mut sign = 1.0;
        for i in 0..n {
            if seen[i] {
                continue;
            }
            let (mut j, mut length) = (i, 0);
            while !seen[j] {
                seen[j] = true;
                j = self.pivots[j];
                length += 1;
            }
            if length % 2 == 0 {
                sign = -sign;
            }
        }
        (0..n).fold(T::from_real(sign), |acc, i| acc * self.lu.at(i, i))
    }
}
