//! Accumulating Householder reflectors into an explicit unitary matrix, in blocks.
//!
//! A product of `nb` reflectors `H_j = I − 2 v_j v_jᴴ` equals `I − V T Vᴴ` for `V = [v_0 … v_{nb−1}]`
//! and an upper triangular `T` (the compact WY representation; LAPACK's `larft`). Applying the
//! product then costs three matrix multiplies (`larfb`) instead of `nb` rank-one updates.
//!
//! See the book entry [QR factorisation](https://jwkunz.github.io/lib_jsl/book/linear_algebra/qr.html#householder-reflections).

use crate::dense::{Mat, sub_matmul};
use crate::scalar::Scalar;

/// Reflectors per block.
const NB: usize = 32;

/// `Q = H_0 H_1 ⋯` as an `n × n` matrix, where reflector `k` is the unit vector
/// `reflectors[k]`, acting on rows (and columns) `k + offset..n`; a zero vector stands for the
/// identity. Blocks are applied from the last to the first, each to the trailing block of `Q`
/// that it touches.
pub(crate) fn accumulate<T: Scalar>(n: usize, reflectors: &[Vec<T>], offset: usize) -> Mat<T> {
    let mut q = Mat::<T>::identity(n);
    let count = reflectors.len();
    let mut end = count;
    while end > 0 {
        let start = end.saturating_sub(NB);
        let nb = end - start;
        let lo = start + offset; // first row touched by the block
        let rows = n - lo;
        // V (rows × nb) and Vᴴ (nb × rows), with reflector j starting j rows down.
        let mut v = vec![T::zero(); rows * nb];
        let mut vh = vec![T::zero(); nb * rows];
        for (j, r) in reflectors[start..end].iter().enumerate() {
            for (i, &x) in r.iter().enumerate() {
                v[(j + i) * nb + j] = x;
                vh[j * rows + j + i] = x.conj();
            }
        }
        // T: T_jj = 2, T[..j, j] = −2 T[..j, ..j] (V[:, ..j]ᴴ v_j).
        let mut t = vec![T::zero(); nb * nb];
        for j in 0..nb {
            t[j * nb + j] = T::from_real(2.0);
            if j == 0 {
                continue;
            }
            let z: Vec<T> = (0..j)
                .map(|i| (0..rows).fold(T::zero(), |s, r| s + vh[i * rows + r] * v[r * nb + j]))
                .collect();
            for i in 0..j {
                let s = (i..j).fold(T::zero(), |s, l| s + t[i * nb + l] * z[l]);
                t[i * nb + j] = s.scale(-2.0);
            }
        }
        // Q₂₂ ← (I − V T Vᴴ) Q₂₂ via W = −Vᴴ Q₂₂, W' = −T W = T Vᴴ Q₂₂, Q₂₂ −= V W'
        // (`sub_matmul` computes C ← C − A B, starting from zero for W and W').
        let cols = n - lo;
        let mut w = vec![T::zero(); nb * cols];
        let mut w2 = vec![T::zero(); nb * cols];
        let qp = q.data.as_mut_ptr();
        // SAFETY: every product reads and writes distinct buffers, apart from the last, which
        // reads V and W' and writes Q.
        unsafe {
            sub_matmul(nb, rows, cols, vh.as_ptr(), rows, qp.add(lo * n + lo), n, w.as_mut_ptr(), cols);
            sub_matmul(nb, nb, cols, t.as_ptr(), nb, w.as_ptr(), cols, w2.as_mut_ptr(), cols);
            sub_matmul(rows, nb, cols, v.as_ptr(), nb, w2.as_ptr(), cols, qp.add(lo * n + lo), n);
        }
        end = start;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_accumulation_matches_the_plain_product() {
        let n = 75;
        let mut s = 7u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let reflectors: Vec<Vec<f64>> = (0..n - 2)
            .map(|k| {
                let mut v: Vec<f64> = (0..n - k - 1).map(|_| next()).collect();
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect();
        let mut plain = Mat::<f64>::identity(n);
        for (k, v) in reflectors.iter().enumerate().rev() {
            let lo = k + 1;
            for c in 0..n {
                let d: f64 = v.iter().enumerate().map(|(i, x)| x * plain.at(lo + i, c)).sum();
                for (i, x) in v.iter().enumerate() {
                    *plain.at_mut(lo + i, c) -= 2.0 * d * x;
                }
            }
        }
        let blocked = accumulate(n, &reflectors, 1);
        let err = plain.data.iter().zip(&blocked.data).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-12, "{err:e}");
    }
}
