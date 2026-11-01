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

    /// Solves `A X = B` for all columns of `B` at once. The substitutions run along contiguous
    /// rows of `X`, which is far more cache-friendly than solving column by column.
    pub(crate) fn solve_matrix(&self, b: &Mat<T>) -> Mat<T> {
        let (n, m) = (self.lu.rows, b.cols);
        let mut x = Mat::zeros(n, m);
        for (i, &p) in self.pivots.iter().enumerate() {
            x.data[i * m..(i + 1) * m].copy_from_slice(&b.data[p * m..(p + 1) * m]);
        }
        for i in 0..n {
            let (done, rest) = x.data.split_at_mut(i * m);
            let row = &mut rest[..m];
            for k in 0..i {
                let l = self.lu.at(i, k);
                if l != T::zero() {
                    for (r, &v) in row.iter_mut().zip(&done[k * m..(k + 1) * m]) {
                        *r -= l * v;
                    }
                }
            }
        }
        for i in (0..n).rev() {
            let (head, tail) = x.data.split_at_mut((i + 1) * m);
            let row = &mut head[i * m..];
            for k in i + 1..n {
                let u = self.lu.at(i, k);
                if u != T::zero() {
                    for (r, &v) in row.iter_mut().zip(&tail[(k - i - 1) * m..(k - i) * m]) {
                        *r -= u * v;
                    }
                }
            }
            let d = self.lu.at(i, i);
            row.iter_mut().for_each(|r| *r /= d);
        }
        x
    }

    /// The inverse.
    pub(crate) fn inverse(&self) -> Mat<T> {
        self.solve_matrix(&Mat::identity(self.lu.rows))
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
