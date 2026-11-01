//! Householder QR with column pivoting, `A P = Q R`, used for least squares and to precondition
//! the Jacobi SVD.

use crate::dense::{Mat, axpy_sub};
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

/// Factorises `A P = Q R`, choosing at each step the remaining column of largest norm. Column
/// norms are downdated rather than recomputed, with a recomputation when cancellation would make
/// the downdated value unreliable (the safeguard of LAPACK's `geqp3`).
pub(crate) fn householder_pivoted<T: Scalar>(a: &Mat<T>) -> Householder<T> {
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
