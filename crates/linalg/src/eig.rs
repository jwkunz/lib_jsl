//! Eigendecomposition of general square matrices via the complex Schur form.
//!
//! 1. Householder reflections reduce `A` to upper Hessenberg form `H = Qᴴ A Q`.
//! 2. The shifted QR algorithm (Givens rotations, Wilkinson shifts, deflation of negligible
//!    sub-diagonal entries) drives `H` to upper triangular `T = Zᴴ A Z`, the complex Schur form,
//!    whose diagonal holds the eigenvalues.
//! 3. Eigenvectors of `T` follow by back-substitution and are mapped back through `Z`.
//!
//! Real matrices are handled in complex arithmetic, since their eigenvalues are complex in
//! general.

use crate::dense::{Mat, require_square};
use crate::error::{LinalgError, Result};
use crate::scalar::Scalar;
use num_complex::Complex;

type C = Complex<f64>;

/// A complex Givens rotation `G = [[c, s], [−s̄, c]]` with `G [a; b] = [r; 0]`.
#[inline]
fn givens(a: C, b: C) -> (f64, C) {
    let (na, nb) = (a.norm(), b.norm());
    if nb == 0.0 {
        return (1.0, C::new(0.0, 0.0));
    }
    if na == 0.0 {
        return (0.0, b.conj() / nb);
    }
    let r = na.hypot(nb);
    (na / r, (a / na) * b.conj() / r)
}

/// Reduces `h` to upper Hessenberg form in place, accumulating the reflections into `q`.
fn hessenberg(h: &mut Mat<C>, q: &mut Mat<C>) {
    let n = h.rows;
    for k in 0..n.saturating_sub(2) {
        let x: Vec<C> = (k + 1..n).map(|i| h.at(i, k)).collect();
        let norm = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let phase = if x[0].norm() > 0.0 { x[0] / x[0].norm() } else { C::new(1.0, 0.0) };
        let alpha = -phase * norm;
        let mut v = x;
        v[0] -= alpha;
        let vnorm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if vnorm == 0.0 {
            continue;
        }
        v.iter_mut().for_each(|z| *z /= vnorm);
        // H ← (I − 2vvᴴ) H on rows k+1..n.
        for c in 0..n {
            let s = (0..v.len()).fold(C::new(0.0, 0.0), |s, r| s + v[r].conj() * h.at(k + 1 + r, c)) * 2.0;
            for (r, vr) in v.iter().enumerate() {
                *h.at_mut(k + 1 + r, c) -= vr * s;
            }
        }
        // H ← H (I − 2vvᴴ) and Q ← Q (I − 2vvᴴ) on columns k+1..n.
        for m in [&mut *h, &mut *q] {
            for row in 0..n {
                let base = row * n + k + 1;
                let s = (0..v.len()).fold(C::new(0.0, 0.0), |s, c| s + m.data[base + c] * v[c]) * 2.0;
                for (c, vc) in v.iter().enumerate() {
                    m.data[base + c] -= s * vc.conj();
                }
            }
        }
        for r in k + 2..n {
            *h.at_mut(r, k) = C::new(0.0, 0.0);
        }
    }
}

/// The eigenvalue of the trailing 2 × 2 block `[[a, b], [c, d]]` closer to `d`.
fn wilkinson_shift(a: C, b: C, c: C, d: C) -> C {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = (tr * tr * 0.25 - det).sqrt();
    let (l1, l2) = (tr * 0.5 + disc, tr * 0.5 - disc);
    if (l1 - d).norm() < (l2 - d).norm() { l1 } else { l2 }
}

/// The complex Schur form `T = Zᴴ A Z` (upper triangular) and `Z`.
pub(crate) fn schur(a: &Mat<C>) -> Result<(Mat<C>, Mat<C>)> {
    require_square(a.rows, a.cols)?;
    let n = a.rows;
    let mut t = a.clone();
    let mut z = Mat::<C>::identity(n);
    hessenberg(&mut t, &mut z);
    let mut hi = n;
    let mut iterations = 0usize;
    let mut since_deflation = 0usize;
    let zero = C::new(0.0, 0.0);
    while hi > 1 {
        // Deflate negligible sub-diagonal entries; find the active window [lo, hi).
        let mut lo = hi - 1;
        while lo > 0 {
            let s = t.at(lo - 1, lo - 1).norm() + t.at(lo, lo).norm();
            let scale = if s == 0.0 { 1.0 } else { s };
            if t.at(lo, lo - 1).norm() <= f64::EPSILON * scale {
                *t.at_mut(lo, lo - 1) = zero;
                break;
            }
            lo -= 1;
        }
        if lo == hi - 1 {
            hi -= 1;
            since_deflation = 0;
            continue;
        }
        iterations += 1;
        since_deflation += 1;
        if iterations > 100 * n.max(10) {
            return Err(LinalgError::NoConvergence("the QR eigenvalue iteration"));
        }
        let m = hi - 1;
        let mut mu = wilkinson_shift(t.at(m - 1, m - 1), t.at(m - 1, m), t.at(m, m - 1), t.at(m, m));
        if since_deflation % 11 == 10 {
            // Exceptional shift to break cycles.
            mu = t.at(m, m) + C::new(t.at(m, m - 1).norm() * 0.75, 0.0);
        }
        // One QR step on the window: rotations from H − μI, applied as similarity transforms.
        for k in lo..hi {
            *t.at_mut(k, k) -= mu;
        }
        let mut rotations = Vec::with_capacity(hi - lo - 1);
        for k in lo..hi - 1 {
            let (c, s) = givens(t.at(k, k), t.at(k + 1, k));
            rotations.push((c, s));
            for col in k..n {
                let (x, y) = (t.at(k, col), t.at(k + 1, col));
                *t.at_mut(k, col) = x * c + s * y;
                *t.at_mut(k + 1, col) = -s.conj() * x + y * c;
            }
        }
        for (i, &(c, s)) in rotations.iter().enumerate() {
            let k = lo + i;
            let last = (k + 2).min(hi);
            for row in 0..last {
                let (p, q) = (t.at(row, k), t.at(row, k + 1));
                *t.at_mut(row, k) = p * c + q * s.conj();
                *t.at_mut(row, k + 1) = -p * s + q * c;
            }
            for row in 0..n {
                let (p, q) = (z.at(row, k), z.at(row, k + 1));
                *z.at_mut(row, k) = p * c + q * s.conj();
                *z.at_mut(row, k + 1) = -p * s + q * c;
            }
        }
        for k in lo..hi {
            *t.at_mut(k, k) += mu;
        }
    }
    // Clear the strictly lower part left by rounding.
    for i in 1..n {
        for j in 0..i {
            *t.at_mut(i, j) = zero;
        }
    }
    Ok((t, z))
}

/// Eigenvalues (in Schur order) and unit-norm eigenvectors as columns.
pub(crate) fn eig<T: Scalar>(a: &Mat<T>) -> Result<(Vec<C>, Mat<C>)> {
    require_square(a.rows, a.cols)?;
    let n = a.rows;
    let ac = Mat { rows: n, cols: n, data: a.data.iter().map(|v| v.to_complex()).collect() };
    let (t, z) = schur(&ac)?;
    let values: Vec<C> = (0..n).map(|i| t.at(i, i)).collect();
    let norm = t.data.iter().fold(0.0f64, |m, v| m.max(v.norm())).max(f64::MIN_POSITIVE);
    let small = f64::EPSILON * norm;
    // Eigenvectors of T by back-substitution: y_k = 1, y_j = 0 for j > k.
    let mut y = Mat::<C>::zeros(n, n);
    for k in 0..n {
        let lambda = values[k];
        *y.at_mut(k, k) = C::new(1.0, 0.0);
        for j in (0..k).rev() {
            let s = (j + 1..=k).fold(C::new(0.0, 0.0), |s, i| s + t.at(j, i) * y.at(i, k));
            let mut denom = t.at(j, j) - lambda;
            if denom.norm() < small {
                denom = C::new(small, 0.0);
            }
            *y.at_mut(j, k) = -s / denom;
        }
    }
    let mut vectors = z.matmul(&y);
    for k in 0..n {
        let norm: f64 = (0..n).map(|i| vectors.at(i, k).norm_sqr()).sum::<f64>().sqrt();
        if norm > 0.0 {
            for i in 0..n {
                *vectors.at_mut(i, k) /= norm;
            }
        }
    }
    Ok((values, vectors))
}
