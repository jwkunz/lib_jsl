//! Householder QR with column pivoting, `A P = Q R`, used for least squares and to precondition
//! the Jacobi SVD.
//!
//! See the book entry [QR factorisation](https://jwkunz.github.io/lib_jsl/book/linear_algebra/qr.html#column-pivoting).

use crate::dense::{Mat, axpy_sub, sub_matmul};
use crate::scalar::Scalar;

/// A pivoted Householder factorisation of an `m × n` matrix (`m ≥ n`).
pub(crate) struct Householder<T> {
    /// `R` in the upper `n × n` triangle (the rest is overwritten).
    pub r: Mat<T>,
    /// Column `j` of `A P` is column `permutation[j]` of `A`.
    pub permutation: Vec<usize>,
    /// The unit vectors `v_k` (acting on rows `k..m`) of `Q = H₀ H₁ ⋯` with `H_k = I − 2 v_k v_kᴴ`;
    /// an empty vector stands for the identity.
    pub reflectors: Vec<Vec<T>>,
}

// ANCHOR: householder_pivoted
/// Factorises `A P = Q R`, choosing at each step the remaining column of largest norm. Column
/// norms are downdated rather than recomputed, with a recomputation when cancellation would make
/// the downdated value unreliable (the safeguard of LAPACK's `geqp3`).
///
/// This is the unblocked form: every reflector updates the whole trailing matrix, two passes over
/// memory per column. [`householder_pivoted`] gives the same factorisation with most of the work
/// in matrix multiplications.
pub(crate) fn householder_pivoted_unblocked<T: Scalar>(a: &Mat<T>) -> Householder<T> {
    let (m, n) = (a.rows, a.cols);
    let mut r = a.clone();
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut reflectors = Vec::with_capacity(n);
    let mut norms2 = vec![0.0f64; n];
    for row in r.data.chunks_exact(n.max(1)) {
        for (s, v) in norms2.iter_mut().zip(row) {
            *s += v.abs_sqr();
        }
    }
    let mut reference = norms2.clone();
    let mut w = vec![T::zero(); n];
    let recompute_below = f64::EPSILON.sqrt();
    for k in 0..n.min(m) {
        let p = (k..n).fold(k, |best, j| if norms2[j] > norms2[best] { j } else { best });
        if p != k {
            for row in r.data.chunks_exact_mut(n) {
                row.swap(k, p);
            }
            permutation.swap(k, p);
            norms2.swap(k, p);
            reference.swap(k, p);
        }
        let norm = (k..m).map(|i| r.data[i * n + k].abs_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 {
            break;
        }
        let x0 = r.at(k, k);
        let phase = if x0.abs() > 0.0 { x0.scale(1.0 / x0.abs()) } else { T::one() };
        let alpha = phase.scale(-norm);
        // v = x − α e₁, normalised.
        let mut v: Vec<T> = (k..m).map(|i| r.at(i, k)).collect();
        v[0] -= alpha;
        let vnorm = v.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
        if vnorm == 0.0 {
            reflectors.push(Vec::new());
            continue;
        }
        v.iter_mut().for_each(|z| *z = z.scale(1.0 / vnorm));
        // R ← (I − 2vvᴴ) R on rows k.., columns k.., row by row for contiguous access.
        let w = &mut w[k..n];
        w.fill(T::zero());
        for (ii, &vi) in v.iter().enumerate() {
            axpy_sub(w, -vi.conj(), &r.data[(k + ii) * n + k..(k + ii + 1) * n]);
        }
        for (ii, &vi) in v.iter().enumerate() {
            axpy_sub(&mut r.data[(k + ii) * n + k..(k + ii + 1) * n], vi.scale(2.0), w);
        }
        reflectors.push(v);
        for j in k + 1..n {
            norms2[j] -= r.data[k * n + j].abs_sqr();
            if norms2[j] <= recompute_below * reference[j] {
                norms2[j] = (k + 1..m).map(|i| r.data[i * n + j].abs_sqr()).sum();
                reference[j] = norms2[j];
            }
        }
    }
    Householder { r, permutation, reflectors }
}
// ANCHOR_END: householder_pivoted

/// Columns factorised per block by [`householder_pivoted`].
const NB: usize = 32;

// ANCHOR: householder_pivoted_blocked
/// Factorises `A P = Q R` with column pivoting, as [`householder_pivoted_unblocked`], but in blocks
/// of `NB` columns (LAPACK's `geqp3` with `laqps`).
///
/// Pivoting needs the norms of the trailing columns after every step, so the reflectors of a block
/// cannot simply be applied together at its end. The trick is to keep the trailing matrix in the
/// factored form `A − V Fᴴ`, where `V` holds the block's reflectors and `F = Aᴴ V T` is built one
/// column per step. Within the block only the column about to become the pivot and the current row
/// of `R` are brought up to date (the row is what the norm downdate needs); the rest of the update,
/// `A ← A − V Fᴴ`, is one matrix multiplication at the end of the block. If a downdated norm becomes
/// unreliable the block ends early, so that the norm can be recomputed from the updated matrix.
///
/// Below about 200 columns the matrix stays in cache and the unblocked form is as fast, so it is
/// used there.
pub(crate) fn householder_pivoted<T: Scalar>(a: &Mat<T>) -> Householder<T> {
    if a.cols < BLOCKED_FROM {
        return householder_pivoted_unblocked(a);
    }
    householder_pivoted_blocked(a)
}

/// The number of columns from which [`householder_pivoted`] uses the blocked form.
const BLOCKED_FROM: usize = 192;

/// The blocked form of [`householder_pivoted`], for any size.
pub(crate) fn householder_pivoted_blocked<T: Scalar>(a: &Mat<T>) -> Householder<T> {
    let (m, n) = (a.rows, a.cols);
    let mut r = a.clone();
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut reflectors: Vec<Vec<T>> = Vec::with_capacity(n);
    let mut norms2 = vec![0.0f64; n];
    for row in r.data.chunks_exact(n.max(1)) {
        for (s, v) in norms2.iter_mut().zip(row) {
            *s += v.abs_sqr();
        }
    }
    let mut reference = norms2.clone();
    let recompute_below = f64::EPSILON.sqrt();
    let steps = n.min(m);
    let mut k0 = 0;
    let mut finished = false;
    // Block workspaces: V is (m − k0) × NB, row-major with stride NB, with reflector l starting at
    // row k0 + l; F is stored transposed, NB × (n − k0) with stride n, so that every update of it
    // runs along contiguous memory (`ft[l * n + c - k0]` is F(c, l)).
    let mut vbuf = vec![T::zero(); m * NB];
    let mut ft = vec![T::zero(); NB * n];
    let mut aux = [T::zero(); NB];
    while k0 < steps && !finished {
        let nb = NB.min(steps - k0);
        let rows = m - k0;
        vbuf[..rows * NB].fill(T::zero());
        ft.fill(T::zero());
        let mut recompute = Vec::new();
        let mut done = 0;
        for j in 0..nb {
            let k = k0 + j;
            // 1. Pivot: the remaining column of largest (downdated) norm.
            let p = (k..n).fold(k, |best, c| if norms2[c] > norms2[best] { c } else { best });
            if p != k {
                for row in r.data.chunks_exact_mut(n) {
                    row.swap(k, p);
                }
                permutation.swap(k, p);
                norms2.swap(k, p);
                reference.swap(k, p);
                for l in 0..j {
                    ft.swap(l * n + k - k0, l * n + p - k0);
                }
            }
            // 2. Bring column k up to date on rows k..m: A(i, k) −= Σ_l V(i, l) conj(F(k, l)).
            if j > 0 {
                let fk: Vec<T> = (0..j).map(|l| ft[l * n + k - k0].conj()).collect();
                for i in k..m {
                    let vrow = &vbuf[(i - k0) * NB..(i - k0) * NB + j];
                    let s = vrow.iter().zip(&fk).fold(T::zero(), |s, (&v, &f)| s + v * f);
                    r.data[i * n + k] -= s;
                }
            }
            // 3. The reflector that maps column k (rows k..m) to α e₁.
            let norm = (k..m).map(|i| r.data[i * n + k].abs_sqr()).sum::<f64>().sqrt();
            if norm == 0.0 {
                finished = true;
                break;
            }
            let x0 = r.at(k, k);
            let phase = if x0.abs() > 0.0 { x0.scale(1.0 / x0.abs()) } else { T::one() };
            let alpha = phase.scale(-norm);
            let mut v: Vec<T> = (k..m).map(|i| r.data[i * n + k]).collect();
            v[0] -= alpha;
            let vnorm = v.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
            if vnorm == 0.0 {
                // The column is already α e₁: the reflector is the identity.
                v.clear();
            } else {
                v.iter_mut().for_each(|z| *z = z.scale(1.0 / vnorm));
                *r.at_mut(k, k) = alpha;
                for i in k + 1..m {
                    r.data[i * n + k] = T::zero();
                }
                for (ii, &vi) in v.iter().enumerate() {
                    vbuf[(k + ii - k0) * NB + j] = vi;
                }
                // 4. F(c, j) = 2 Σ_i conj(A(i, c)) v(i) over the block-start trailing matrix, minus
                //    the correction F(:, ..j) (2 V(:, ..j)ᴴ v) for the earlier reflectors.
                let (before, rest) = ft.split_at_mut(j * n);
                let fj = &mut rest[k + 1 - k0..n - k0];
                // Four rows per pass, so F's row is loaded and stored a quarter as often.
                let width = n - k - 1;
                let mut ii = 0;
                while ii + 4 <= v.len() {
                    let base = (k + ii) * n + k + 1;
                    let (r0, r1, r2, r3) = (
                        &r.data[base..base + width],
                        &r.data[base + n..base + n + width],
                        &r.data[base + 2 * n..base + 2 * n + width],
                        &r.data[base + 3 * n..base + 3 * n + width],
                    );
                    let (v0, v1, v2, v3) = (v[ii].scale(2.0), v[ii + 1].scale(2.0), v[ii + 2].scale(2.0), v[ii + 3].scale(2.0));
                    for ((((f, &a0), &a1), &a2), &a3) in fj.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3) {
                        *f += a0.conj() * v0 + a1.conj() * v1 + a2.conj() * v2 + a3.conj() * v3;
                    }
                    ii += 4;
                }
                for (ii, &vi) in v.iter().enumerate().skip(ii) {
                    let row = &r.data[(k + ii) * n + k + 1..(k + ii + 1) * n];
                    let vi2 = vi.scale(2.0);
                    for (f, &x) in fj.iter_mut().zip(row) {
                        *f += x.conj() * vi2;
                    }
                }
                if j > 0 {
                    for (l, a) in aux[..j].iter_mut().enumerate() {
                        *a = v.iter().enumerate().fold(T::zero(), |s, (ii, &vi)| s + vbuf[(k + ii - k0) * NB + l].conj() * vi).scale(2.0);
                    }
                    for (l, &a) in aux[..j].iter().enumerate() {
                        axpy_sub(fj, a, &before[l * n + k + 1 - k0..l * n + n - k0]);
                    }
                }
            }
            // 5. Bring row k of R up to date: A(k, c) −= Σ_{l ≤ j} V(k, l) conj(F(c, l)).
            for l in 0..=j {
                let vkl = vbuf[(k - k0) * NB + l];
                if vkl == T::zero() {
                    continue;
                }
                let row = &mut r.data[k * n + k + 1..(k + 1) * n];
                for (x, &f) in row.iter_mut().zip(&ft[l * n + k + 1 - k0..l * n + n - k0]) {
                    *x -= vkl * f.conj();
                }
            }
            reflectors.push(v);
            done = j + 1;
            // 6. Downdate the norms; an unreliable one ends the block.
            for c in k + 1..n {
                norms2[c] -= r.data[k * n + c].abs_sqr();
                if norms2[c] <= recompute_below * reference[c] {
                    recompute.push(c);
                }
            }
            if !recompute.is_empty() {
                break;
            }
        }
        // 7. The deferred trailing update A ← A − V Fᴴ on rows and columns k0 + done...
        let lo = k0 + done;
        if done > 0 && lo < m && lo < n {
            let tc = n - lo;
            let mut fh = vec![T::zero(); done * tc];
            for l in 0..done {
                for (d, &f) in fh[l * tc..(l + 1) * tc].iter_mut().zip(&ft[l * n + lo - k0..l * n + n - k0]) {
                    *d = f.conj();
                }
            }
            // SAFETY: V and Fᴴ are separate buffers from R.
            unsafe {
                sub_matmul(m - lo, done, tc, vbuf.as_ptr().add((lo - k0) * NB), NB, fh.as_ptr(), tc, r.data.as_mut_ptr().add(lo * n + lo), n);
            }
        }
        for c in recompute {
            norms2[c] = (lo..m).map(|i| r.data[i * n + c].abs_sqr()).sum();
            reference[c] = norms2[c];
        }
        k0 = lo;
        if done == 0 {
            break;
        }
    }
    Householder { r, permutation, reflectors }
}
// ANCHOR_END: householder_pivoted_blocked

impl<T: Scalar> Householder<T> {
    /// `b ← Qᴴ b`.
    pub fn apply_adjoint(&self, b: &mut [T]) {
        for (k, v) in self.reflectors.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
            let s = v.iter().zip(&b[k..]).fold(T::zero(), |s, (vi, &bi)| s + vi.conj() * bi).scale(2.0);
            axpy_sub(&mut b[k..], s, v);
        }
    }

    /// `Q [W; 0]` for an `n × c` matrix `W` (the first `n` columns of `Q` times `W`).
    pub fn apply_to_top(&self, top: &Mat<T>) -> Mat<T> {
        let (m, c) = (self.r.rows, top.cols);
        let mut x = Mat::zeros(m, c);
        x.data[..top.data.len()].copy_from_slice(&top.data);
        let mut t = vec![T::zero(); c];
        for (k, v) in self.reflectors.iter().enumerate().rev().filter(|(_, v)| !v.is_empty()) {
            t.fill(T::zero());
            for (ii, &vi) in v.iter().enumerate() {
                axpy_sub(&mut t, -vi.conj(), &x.data[(k + ii) * c..(k + ii + 1) * c]);
            }
            for (ii, &vi) in v.iter().enumerate() {
                axpy_sub(&mut x.data[(k + ii) * c..(k + ii + 1) * c], vi.scale(2.0), &t);
            }
        }
        x
    }

    /// The upper triangular `n × n` factor.
    pub fn upper(&self) -> Mat<T> {
        let n = self.r.cols;
        let mut u = Mat::zeros(n, n);
        for i in 0..n.min(self.r.rows) {
            u.data[i * n + i..(i + 1) * n].copy_from_slice(&self.r.data[i * n + i..(i + 1) * n]);
        }
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn random(m: usize, n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..m * n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    /// The largest entry of `A P − Q R`, relative to the largest entry of `A`.
    fn residual<T: Scalar>(a: &Mat<T>, f: &Householder<T>) -> f64 {
        let (m, n) = (a.rows, a.cols);
        let qr = f.apply_to_top(&f.upper());
        let scale = a.data.iter().fold(0.0f64, |s, x| s.max(x.abs()));
        let mut worst = 0.0f64;
        for i in 0..m {
            for (j, &p) in f.permutation.iter().enumerate() {
                worst = worst.max((a.at(i, p) - qr.at(i, j)).abs());
            }
        }
        let _ = n;
        worst / scale
    }

    fn check<T: Scalar>(a: &Mat<T>) {
        let blocked = householder_pivoted_blocked(a);
        let plain = householder_pivoted_unblocked(a);
        let (rb, ru) = (residual(a, &blocked), residual(a, &plain));
        assert!(rb < 1e-13, "blocked residual {rb:e} ({} × {})", a.rows, a.cols);
        assert!(ru < 1e-13, "unblocked residual {ru:e}");
        // The diagonal of R is non-increasing in magnitude and agrees with the unblocked one.
        let n = a.cols.min(a.rows);
        let scale = blocked.r.at(0, 0).abs().max(1e-300);
        for i in 0..n {
            let (b, u) = (blocked.r.at(i, i).abs(), plain.r.at(i, i).abs());
            assert!((b - u).abs() <= 1e-10 * scale, "|r_ii| differs at {i}: {b} vs {u}");
            if i > 0 {
                assert!(b <= blocked.r.at(i - 1, i - 1).abs() * (1.0 + 1e-12) + 1e-14 * scale);
            }
        }
    }

    #[test]
    fn blocked_pivoted_qr_matches_the_unblocked_one() {
        for &(m, n) in &[(1, 1), (5, 3), (40, 40), (70, 33), (100, 70), (130, 129), (200, 97)] {
            let a = Mat { rows: m, cols: n, data: random(m, n, (m * 31 + n) as u64) };
            check(&a);
            let re = random(m, n, 7 + m as u64);
            let im = random(m, n, 11 + n as u64);
            let c = Mat { rows: m, cols: n, data: re.iter().zip(&im).map(|(&x, &y)| Complex::new(x, y)).collect() };
            check(&c);
        }
    }

    #[test]
    fn blocked_pivoted_qr_handles_rank_deficiency() {
        // Rank 20 in a 120 × 80 matrix, with columns of very different sizes, so the downdated
        // norms collapse and must be recomputed, and the factorisation stops on zero columns.
        let (m, n, k) = (120, 80, 20);
        let b = random(m, k, 3);
        let c = random(k, n, 5);
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let s: f64 = (0..k).map(|l| b[i * k + l] * c[l * n + j]).sum();
                data[i * n + j] = s * 10f64.powi((j % 7) as i32 - 3);
            }
        }
        let a = Mat { rows: m, cols: n, data };
        let f = householder_pivoted_blocked(&a);
        assert!(residual(&a, &f) < 1e-12, "{:e}", residual(&a, &f));
        let r00 = f.r.at(0, 0).abs();
        assert!(f.r.at(k - 1, k - 1).abs() > 1e-8 * r00);
        assert!(f.r.at(k, k).abs() < 1e-10 * r00, "{:e}", f.r.at(k, k).abs() / r00);
    }
}
