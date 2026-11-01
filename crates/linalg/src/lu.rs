//! LU factorisation with partial pivoting: `P A = L U`.

use crate::dense::{BLOCK, Mat, axpy_sub, require_square, sub_matmul};
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
    /// Right-looking blocked LU: factor a panel of [`BLOCK`] columns with partial pivoting,
    /// solve for the corresponding block row of `U`, then update the trailing matrix with the
    /// matrix-multiply kernel.
    pub(crate) fn new(mut a: Mat<T>) -> Result<Self> {
        require_square(a.rows, a.cols)?;
        let n = a.rows;
        let mut pivots: Vec<usize> = (0..n).collect();
        let scale = a.data.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        for k in (0..n).step_by(BLOCK) {
            let kb = BLOCK.min(n - k);
            let end = k + kb;
            for c in k..end {
                // Partial pivoting: the largest entry in column c at or below the diagonal.
                let (p, big) = (c..n).map(|i| (i, a.data[i * n + c].abs())).fold((c, -1.0), |best, x| if x.1 > best.1 { x } else { best });
                if big <= f64::EPSILON * scale * 1e-3 || big == 0.0 {
                    return Err(LinalgError::Singular(c));
                }
                if p != c {
                    let (top, bottom) = a.data.split_at_mut(p * n);
                    top[c * n..(c + 1) * n].swap_with_slice(&mut bottom[..n]);
                    pivots.swap(c, p);
                }
                // Eliminate below the pivot, within the panel only.
                let (top, bottom) = a.data.split_at_mut((c + 1) * n);
                let pivot_row = &top[c * n + c..c * n + end];
                let inverse_pivot = T::one() / pivot_row[0];
                for row in bottom.chunks_exact_mut(n) {
                    let m = row[c] * inverse_pivot;
                    row[c] = m;
                    if m != T::zero() {
                        axpy_sub(&mut row[c + 1..end], m, &pivot_row[1..]);
                    }
                }
            }
            if end == n {
                break;
            }
            // U₁₂ ← L₁₁⁻¹ A₁₂ (unit lower forward substitution on the block row).
            for r in k + 1..end {
                let (top, bottom) = a.data.split_at_mut(r * n);
                let row = &mut bottom[..n];
                for p in k..r {
                    let l = row[p];
                    if l != T::zero() {
                        axpy_sub(&mut row[end..], l, &top[p * n + end..(p + 1) * n]);
                    }
                }
            }
            // A₂₂ ← A₂₂ − L₂₁ U₁₂.
            let ptr = a.data.as_mut_ptr();
            let rest = n - end;
            // SAFETY: L₂₁ (rows ≥ end, columns k..end), U₁₂ (rows k..end, columns ≥ end) and
            // A₂₂ (rows and columns ≥ end) are disjoint blocks of the n × n buffer.
            unsafe { sub_matmul(rest, kb, rest, ptr.add(end * n + k), n, ptr.add(k * n + end), n, ptr.add(end * n + end), n) };
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

    /// Solves `A X = B` for all columns of `B` at once. Substitution proceeds a block of rows at
    /// a time: the contribution of all earlier blocks is subtracted with one matrix multiply, and
    /// only the small triangle within the block is done row by row.
    pub(crate) fn solve_matrix(&self, b: &Mat<T>) -> Mat<T> {
        let (n, m) = (self.lu.rows, b.cols);
        let mut x = Mat::zeros(n, m);
        for (i, &p) in self.pivots.iter().enumerate() {
            x.data[i * m..(i + 1) * m].copy_from_slice(&b.data[p * m..(p + 1) * m]);
        }
        let lu = &self.lu.data;
        // Forward: L Y = P B.
        for i0 in (0..n).step_by(BLOCK) {
            let i1 = (i0 + BLOCK).min(n);
            let ptr = x.data.as_mut_ptr();
            // SAFETY: rows i0..i1 of X are disjoint from rows 0..i0; L lives in another buffer.
            unsafe { sub_matmul(i1 - i0, i0, m, lu.as_ptr().add(i0 * n), n, ptr, m, ptr.add(i0 * m), m) };
            for i in i0 + 1..i1 {
                let (done, rest) = x.data.split_at_mut(i * m);
                let row = &mut rest[..m];
                for k in i0..i {
                    let l = lu[i * n + k];
                    if l != T::zero() {
                        axpy_sub(row, l, &done[k * m..(k + 1) * m]);
                    }
                }
            }
        }
        // Backward: U X = Y.
        let mut i1 = n;
        while i1 > 0 {
            let i0 = i1.saturating_sub(BLOCK);
            let ptr = x.data.as_mut_ptr();
            // SAFETY: rows i0..i1 of X are disjoint from rows i1..n.
            unsafe { sub_matmul(i1 - i0, n - i1, m, lu.as_ptr().add(i0 * n + i1), n, ptr.add(i1 * m), m, ptr.add(i0 * m), m) };
            for i in (i0..i1).rev() {
                let (head, tail) = x.data.split_at_mut((i + 1) * m);
                let row = &mut head[i * m..];
                for k in i + 1..i1 {
                    let u = lu[i * n + k];
                    if u != T::zero() {
                        axpy_sub(row, u, &tail[(k - i - 1) * m..(k - i) * m]);
                    }
                }
                let d = T::one() / lu[i * n + i];
                row.iter_mut().for_each(|r| *r *= d);
            }
            i1 = i0;
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
