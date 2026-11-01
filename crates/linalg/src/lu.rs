//! LU factorisation with partial pivoting: `P A = L U`.
//!
//! See the book entry [Gaussian elimination and LU factorisation](https://jwkunz.github.io/lib_jsl/book/linear_algebra/lu.html#blocked-and-recursive-factorisation).

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
    /// Recursive LU (Toledo's algorithm, LAPACK's `getrf2`): factor the left half of the columns,
    /// solve for the top-right block, update the bottom-right block with one matrix multiply,
    /// and recurse on it. Nearly all the arithmetic lands in the matrix-multiply kernel, at every
    /// level of the cache hierarchy.
    pub(crate) fn new(mut a: Mat<T>) -> Result<Self> {
        require_square(a.rows, a.cols)?;
        let n = a.rows;
        let mut pivots: Vec<usize> = (0..n).collect();
        let scale = a.data.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        recursive_lu(&mut a, &mut pivots, 0, n, f64::EPSILON * scale * 1e-3)?;
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

    /// `X ← L⁻¹ X`, a block of rows at a time: the contribution of all earlier blocks is
    /// subtracted with one matrix multiply, and only the small triangle within the block is done
    /// row by row. With `lower_rhs`, `X` is known to be lower triangular (as when inverting), and
    /// the columns that are still zero are skipped.
    fn forward(&self, x: &mut Mat<T>, lower_rhs: bool) {
        let (n, m) = (self.lu.rows, x.cols);
        let lu = &self.lu.data;
        for i0 in (0..n).step_by(BLOCK) {
            let i1 = (i0 + BLOCK).min(n);
            let cols = if lower_rhs { i1 } else { m };
            let ptr = x.data.as_mut_ptr();
            // SAFETY: rows i0..i1 of X are disjoint from rows 0..i0; L lives in another buffer.
            unsafe { sub_matmul(i1 - i0, i0, cols, lu.as_ptr().add(i0 * n), n, ptr, m, ptr.add(i0 * m), m) };
            for i in i0 + 1..i1 {
                let (done, rest) = x.data.split_at_mut(i * m);
                let row = &mut rest[..cols];
                for k in i0..i {
                    let l = lu[i * n + k];
                    if l != T::zero() {
                        axpy_sub(row, l, &done[k * m..k * m + cols]);
                    }
                }
            }
        }
    }

    /// `X ← U⁻¹ X`, blocked like [`Self::forward`].
    fn backward(&self, x: &mut Mat<T>) {
        let (n, m) = (self.lu.rows, x.cols);
        let lu = &self.lu.data;
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
    }

    /// The inverse `A⁻¹ = U⁻¹ L⁻¹ P`: `L⁻¹` costs `n³/3` because it stays lower triangular, then
    /// one full back substitution and a column permutation (`2n³` with the factorisation, as in
    /// LAPACK's `getri`).
    pub(crate) fn inverse(&self) -> Mat<T> {
        let n = self.lu.rows;
        let mut x = Mat::identity(n);
        self.forward(&mut x, true);
        self.backward(&mut x);
        // (M P)[:, pivots[i]] = M[:, i].
        let mut inv = Mat::zeros(n, n);
        for (row_in, row_out) in x.data.chunks_exact(n.max(1)).zip(inv.data.chunks_exact_mut(n.max(1))) {
            for (i, &p) in self.pivots.iter().enumerate() {
                row_out[p] = row_in[i];
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

/// Panels up to this width are factorised in column-major order.
const LEAF: usize = 32;

/// Triangular blocks up to this order are solved row by row.
const SOLVE_LEAF: usize = 16;

// ANCHOR: recursive_lu
/// Factorises the panel of columns `k..k + w` (rows `k..n`) of the square buffer in place,
/// swapping whole rows so that every column sees the same permutation.
fn recursive_lu<T: Scalar>(a: &mut Mat<T>, pivots: &mut [usize], k: usize, w: usize, threshold: f64) -> Result<()> {
    let n = a.cols;
    if w <= LEAF {
        return unblocked_panel(a, pivots, k, w, threshold);
    }
    let w1 = w / 2;
    recursive_lu(a, pivots, k, w1, threshold)?;
    // A₁₂ ← L₁₁⁻¹ A₁₂ (rows k..k+w1, columns k+w1..k+w).
    unit_lower_solve(a, k, w1, k + w1, w - w1);
    // A₂₂ ← A₂₂ − A₂₁ A₁₂ (rows k+w1..n, columns k+w1..k+w).
    let ptr = a.data.as_mut_ptr();
    // SAFETY: A₂₁ (columns k..k+w1, rows ≥ k+w1), A₁₂ (rows k..k+w1) and A₂₂ are disjoint.
    unsafe { sub_matmul(n - k - w1, w1, w - w1, ptr.add((k + w1) * n + k), n, ptr.add(k * n + k + w1), n, ptr.add((k + w1) * n + k + w1), n) };
    recursive_lu(a, pivots, k + w1, w - w1, threshold)
}
// ANCHOR_END: recursive_lu

/// Factorises the panel of columns `k..k + w` (rows `k..n`). The narrow panel is copied to
/// column-major order, where pivot searches and eliminations run along long contiguous columns,
/// and factorised there recursively; the row interchanges are then applied to the rest of the
/// matrix.
fn unblocked_panel<T: Scalar>(a: &mut Mat<T>, pivots: &mut [usize], k: usize, w: usize, threshold: f64) -> Result<()> {
    let n = a.cols;
    let m = n - k;
    let mut panel = vec![T::zero(); w * m];
    for i in 0..m {
        for j in 0..w {
            panel[j * m + i] = a.data[(k + i) * n + k + j];
        }
    }
    let mut local = vec![0usize; w];
    column_major_lu(&mut panel, m, w, 0, w, &mut local, threshold).map_err(|c| LinalgError::Singular(k + c))?;
    for (c, &r) in local.iter().enumerate() {
        if r != c {
            let (top, bottom) = a.data.split_at_mut((k + r) * n);
            let (upper, lower) = (&mut top[(k + c) * n..(k + c + 1) * n], &mut bottom[..n]);
            upper[..k].swap_with_slice(&mut lower[..k]);
            upper[k + w..].swap_with_slice(&mut lower[k + w..]);
            pivots.swap(k + c, k + r);
        }
    }
    for i in 0..m {
        for j in 0..w {
            a.data[(k + i) * n + k + j] = panel[j * m + i];
        }
    }
    Ok(())
}

/// Recursive LU of columns `c0..c0 + w` (rows `c0..m`) of an `m × total` column-major panel,
/// recording in `local[c]` the row swapped with row `c`. Swaps act on all `total` columns.
/// Returns the failing column if a pivot is negligible.
fn column_major_lu<T: Scalar>(p: &mut [T], m: usize, total: usize, c0: usize, w: usize, local: &mut [usize], threshold: f64) -> std::result::Result<(), usize> {
    if w <= 8 {
        for c in c0..c0 + w {
            let column = &p[c * m..(c + 1) * m];
            let (piv, big) = (c..m).map(|i| (i, column[i].abs())).fold((c, -1.0), |best, x| if x.1 > best.1 { x } else { best });
            if big <= threshold || big == 0.0 {
                return Err(c);
            }
            local[c] = piv;
            if piv != c {
                for j in 0..total {
                    p.swap(j * m + c, j * m + piv);
                }
            }
            let (left, right) = p.split_at_mut((c + 1) * m);
            let column = &mut left[c * m..];
            let inverse_pivot = T::one() / column[c];
            column[c + 1..].iter_mut().for_each(|x| *x *= inverse_pivot);
            let multipliers = &column[c + 1..];
            for later in right.chunks_exact_mut(m).take(c0 + w - c - 1) {
                let f = later[c];
                if f != T::zero() {
                    axpy_sub(&mut later[c + 1..], f, multipliers);
                }
            }
        }
        return Ok(());
    }
    let w1 = w / 2;
    let (mid, end) = (c0 + w1, c0 + w);
    column_major_lu(p, m, total, c0, w1, local, threshold)?;
    // A₁₂ ← L₁₁⁻¹ A₁₂, column by column.
    for j in mid..end {
        let (left, right) = p.split_at_mut(j * m);
        let x = &mut right[c0..mid];
        for q in c0..mid {
            let f = x[q - c0];
            if f != T::zero() {
                axpy_sub(&mut x[q - c0 + 1..], f, &left[q * m + q + 1..q * m + mid]);
            }
        }
    }
    // A₂₂ ← A₂₂ − A₂₁ A₁₂ with column-major strides (row stride 1, column stride m).
    let ptr = p.as_mut_ptr();
    let ms = m as isize;
    // SAFETY: A₂₁ (columns c0..mid), A₁₂ (rows c0..mid of columns mid..end) and A₂₂ (rows ≥ mid of
    // columns mid..end) are disjoint parts of the panel.
    unsafe { T::gemm_sub(m - mid, w1, w - w1, ptr.add(c0 * m + mid), 1, ms, ptr.add(mid * m + c0), 1, ms, ptr.add(mid * m + mid), 1, ms) };
    column_major_lu(p, m, total, mid, w - w1, local, threshold)
}

/// `X ← L⁻¹ X` for the unit lower triangle `L` of order `w` at `(k, k)` and the block `X` in
/// rows `k..k + w`, columns `c..c + nc`, recursively so that most of the work is a multiply.
fn unit_lower_solve<T: Scalar>(a: &mut Mat<T>, k: usize, w: usize, c: usize, nc: usize) {
    let n = a.cols;
    if w <= SOLVE_LEAF {
        for r in k + 1..k + w {
            let (top, bottom) = a.data.split_at_mut(r * n);
            // Row r holds its multipliers (columns k..r) to the left of its block of X (c ≥ k + w).
            let (multipliers, right) = bottom.split_at_mut(c);
            let row = &mut right[..nc];
            for p in k..r {
                let l = multipliers[p];
                if l != T::zero() {
                    axpy_sub(row, l, &top[p * n + c..p * n + c + nc]);
                }
            }
        }
        return;
    }
    let h = w / 2;
    unit_lower_solve(a, k, h, c, nc);
    let ptr = a.data.as_mut_ptr();
    // SAFETY: L₂₁ (rows k+h..k+w, columns k..k+h), X₁ (rows k..k+h, columns c..) and
    // X₂ (rows k+h..k+w, columns c..) are disjoint because c ≥ k + w.
    unsafe { sub_matmul(w - h, h, nc, ptr.add((k + h) * n + k), n, ptr.add(k * n + c), n, ptr.add((k + h) * n + c), n) };
    unit_lower_solve(a, k + h, w - h, c, nc);
}
