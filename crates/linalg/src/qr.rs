//! Householder QR with column pivoting, `A P = Q R`, used for fast least squares.

use crate::dense::Mat;
use crate::scalar::Scalar;

/// The pivoted QR factorisation of an `m × n` matrix (`m ≥ n`), applied on the fly to one
/// right-hand side.
pub(crate) struct PivotedQr<T> {
    /// `R` in the upper `n × n` triangle (the rest is overwritten).
    pub r: Mat<T>,
    /// Column `j` of `A P` is column `permutation[j]` of `A`.
    pub permutation: Vec<usize>,
    /// `Qᴴ b`.
    pub qtb: Vec<T>,
}

pub(crate) fn pivoted_qr<T: Scalar>(a: &Mat<T>, b: &[T]) -> PivotedQr<T> {
    let (m, n) = (a.rows, a.cols);
    let mut r = a.clone();
    let mut qtb = b.to_vec();
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut w = vec![T::zero(); n];
    for k in 0..n.min(m) {
        // Pivot: the remaining column with the largest norm below row k.
        let norms: Vec<f64> = (k..n).map(|j| (k..m).map(|i| r.at(i, j).abs_sqr()).sum::<f64>()).collect();
        let p = k + (0..norms.len()).fold(0, |best, i| if norms[i] > norms[best] { i } else { best });
        if p != k {
            for i in 0..m {
                r.data.swap(i * n + k, i * n + p);
            }
            permutation.swap(k, p);
        }
        let norm = norms[p - k].sqrt();
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
            continue;
        }
        v.iter_mut().for_each(|z| *z = z.scale(1.0 / vnorm));
        // R ← (I − 2vvᴴ) R on rows k.., columns k.., row by row for contiguous access.
        w[k..n].iter_mut().for_each(|x| *x = T::zero());
        for (ii, vi) in v.iter().enumerate() {
            let row = &r.data[(k + ii) * n + k..(k + ii + 1) * n];
            let vc = vi.conj();
            for (wj, &x) in w[k..n].iter_mut().zip(row) {
                *wj += vc * x;
            }
        }
        for (ii, vi) in v.iter().enumerate() {
            let row = &mut r.data[(k + ii) * n + k..(k + ii + 1) * n];
            let factor = vi.scale(2.0);
            for (x, &wj) in row.iter_mut().zip(&w[k..n]) {
                *x -= factor * wj;
            }
        }
        let s = v.iter().zip(&qtb[k..]).fold(T::zero(), |s, (vi, &bi)| s + vi.conj() * bi).scale(2.0);
        for (bi, vi) in qtb[k..].iter_mut().zip(&v) {
            *bi -= *vi * s;
        }
    }
    PivotedQr { r, permutation, qtb }
}
