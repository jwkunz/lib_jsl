//! Singular value decomposition, by two methods.
//!
//! * [`thin_svd`] (the default, behind `SVD::svd`): Golub-Kahan bidiagonalisation followed by
//!   implicit-shift QR on the bidiagonal, as in LAPACK's `gesvd`. Fast and backward stable: the
//!   singular values are accurate to about `ε σ_max` in absolute terms.
//! * [`thin_svd_jacobi`] (behind `SvdJacobi::svd_jacobi`): one-sided Jacobi rotations (Hestenes),
//!   applied to pairs of columns until every pair is orthogonal; the column norms are then the
//!   singular values, the normalised columns the left singular vectors, and the accumulated
//!   rotations the right singular vectors. Slower, but small singular values of graded matrices
//!   are found to high *relative* accuracy. Pivoted QR preconditions it.

use crate::dense::{Mat, axpy_sub, dot_conj};
use crate::error::{LinalgError, Result};
use crate::qr::householder_pivoted;
use crate::scalar::Scalar;

/// A thin SVD `A = U Σ Vᴴ` with `U` of size `m × k`, `V` of size `n × k`, `k = min(m, n)` and the
/// singular values in descending order.
pub(crate) struct ThinSvd<T> {
    pub u: Mat<T>,
    pub sigma: Vec<f64>,
    pub v: Mat<T>,
}

/// One-sided Jacobi on a tall (`m ≥ n`) matrix.
fn jacobi_tall<T: Scalar>(a: &Mat<T>) -> Result<ThinSvd<T>> {
    let (m, n) = (a.rows, a.cols);
    // Work on columns: store A transposed so each column is contiguous.
    let mut cols: Vec<Vec<T>> = (0..n).map(|j| (0..m).map(|i| a.at(i, j)).collect()).collect();
    let mut v: Vec<Vec<T>> = (0..n).map(|j| (0..n).map(|i| if i == j { T::one() } else { T::zero() }).collect()).collect();
    let tol = f64::EPSILON * (m as f64).sqrt();
    let mut converged = false;
    for _sweep in 0..60 {
        let mut rotated = false;
        // Squared column norms, recomputed each sweep and updated exactly through each rotation.
        let mut norms2: Vec<f64> = cols.iter().map(|c| dot_conj(c, c).re()).collect();
        for p in 0..n {
            for q in p + 1..n {
                let (alpha, beta) = (norms2[p], norms2[q]);
                let gamma = dot_conj(&cols[q], &cols[p]); // Σ conj(a_p) a_q
                let g = gamma.abs();
                if g == 0.0 || g <= tol * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                // Remove the phase of γ from column q, then apply a real rotation.
                let phase = gamma.scale(1.0 / g);
                let zeta = (beta - alpha) / (2.0 * g);
                // The smaller root of t² + 2ζt − 1 = 0 (the rotation angle below π/4).
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                norms2[p] = alpha - t * g;
                norms2[q] = beta + t * g;
                let phase_conj = phase.conj();
                for x in [&mut cols, &mut v] {
                    let (left, right) = x.split_at_mut(q);
                    let (cp, cq) = (&mut left[p], &mut right[0]);
                    if T::IS_COMPLEX {
                        for (u, w) in cp.iter_mut().zip(cq.iter_mut()) {
                            let wq = *w * phase_conj;
                            let up = *u;
                            *u = up.scale(c) - wq.scale(s);
                            *w = up.scale(s) + wq.scale(c);
                        }
                    } else {
                        // Real data: the phase φ is ±1, folded into the coefficients of w.
                        let (s_phi, c_phi) = (s * phase.re(), c * phase.re());
                        for (u, w) in cp.iter_mut().zip(cq.iter_mut()) {
                            let (up, wp) = (*u, *w);
                            *u = up.scale(c) - wp.scale(s_phi);
                            *w = up.scale(s) + wp.scale(c_phi);
                        }
                    }
                }
            }
        }
        if !rotated {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence("the one-sided Jacobi SVD"));
    }
    let norms: Vec<f64> = cols.iter().map(|c| c.iter().map(|x| x.abs_sqr()).sum::<f64>().sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    let mut u = Mat::<T>::zeros(m, n);
    let mut vm = Mat::<T>::zeros(n, n);
    let mut sigma = Vec::with_capacity(n);
    for (k, &j) in order.iter().enumerate() {
        sigma.push(norms[j]);
        for i in 0..m {
            *u.at_mut(i, k) = if norms[j] > 0.0 { cols[j][i].scale(1.0 / norms[j]) } else { T::zero() };
        }
        for i in 0..n {
            *vm.at_mut(i, k) = v[j][i];
        }
    }
    // Columns of U for zero singular values are completed to an orthonormal set.
    complete_columns(&mut u, sigma.iter().filter(|&&s| s > 0.0).count());
    Ok(ThinSvd { u, sigma, v: vm })
}

/// Replaces columns `valid..` of `q` (whose first `valid` columns are orthonormal) with further
/// orthonormal columns, by Gram-Schmidt on the standard basis.
pub(crate) fn complete_columns<T: Scalar>(q: &mut Mat<T>, valid: usize) {
    let (m, k) = (q.rows, q.cols);
    let mut filled = valid;
    let mut candidate = 0;
    while filled < k && candidate < m {
        let mut w: Vec<T> = (0..m).map(|i| if i == candidate { T::one() } else { T::zero() }).collect();
        for _ in 0..2 {
            for c in 0..filled {
                let proj = (0..m).fold(T::zero(), |s, i| s + q.at(i, c).conj() * w[i]);
                for (i, wi) in w.iter_mut().enumerate() {
                    *wi -= q.at(i, c) * proj;
                }
            }
        }
        let norm = w.iter().map(|x| x.abs_sqr()).sum::<f64>().sqrt();
        candidate += 1;
        if norm > 1e-8 {
            for (i, wi) in w.iter().enumerate() {
                *q.at_mut(i, filled) = wi.scale(1.0 / norm);
            }
            filled += 1;
        }
    }
}

/// Below this many columns the QR preconditioning costs more than it saves.
const PRECONDITION_FROM: usize = 16;

/// The thin SVD of any matrix by Golub-Kahan (wide matrices through their adjoint).
pub(crate) fn thin_svd<T: Scalar>(a: &Mat<T>) -> Result<ThinSvd<T>> {
    if a.rows < a.cols {
        let ThinSvd { u, sigma, v } = thin_svd(&a.adjoint())?;
        return Ok(ThinSvd { u: v, sigma, v: u });
    }
    golub_kahan(a, true)
}

/// The singular values alone, in descending order.
pub(crate) fn singular_values<T: Scalar>(a: &Mat<T>) -> Result<Vec<f64>> {
    let sigma = if a.rows < a.cols { golub_kahan(&a.adjoint(), false)?.sigma } else { golub_kahan(a, false)?.sigma };
    Ok(sigma)
}

/// The thin SVD by one-sided Jacobi (wide matrices through their adjoint): slower than
/// Golub-Kahan, but it finds small singular values to high *relative* accuracy.
///
/// Larger matrices are first factorised as `A P = Q R` with column pivoting; one-sided Jacobi
/// then runs on `Rᴴ`, whose columns are already close to orthogonal, so it converges in a few
/// sweeps (the preconditioning of Drmač and Veselić). With `Rᴴ = U₁ Σ W₁ᴴ`,
/// `A = (Q W₁) Σ (P U₁)ᴴ`.
pub(crate) fn thin_svd_jacobi<T: Scalar>(a: &Mat<T>) -> Result<ThinSvd<T>> {
    if a.rows < a.cols {
        let ThinSvd { u, sigma, v } = thin_svd_jacobi(&a.adjoint())?;
        return Ok(ThinSvd { u: v, sigma, v: u });
    }
    let n = a.cols;
    if n < PRECONDITION_FROM {
        return jacobi_tall(a);
    }
    let qr = householder_pivoted(a);
    let ThinSvd { u: u1, sigma, v: w1 } = jacobi_tall(&qr.upper().adjoint())?;
    let u = qr.apply_to_top(&w1);
    let mut v = Mat::<T>::zeros(n, n);
    for (j, &p) in qr.permutation.iter().enumerate() {
        v.data[p * n..(p + 1) * n].copy_from_slice(&u1.data[j * n..(j + 1) * n]);
    }
    Ok(ThinSvd { u, sigma, v })
}

/// Extends an `r × k` matrix with orthonormal columns to a square unitary `r × r` matrix.
pub(crate) fn complete_square<T: Scalar>(q: &Mat<T>) -> Mat<T> {
    let (r, k) = (q.rows, q.cols);
    let mut full = Mat::<T>::zeros(r, r);
    for i in 0..r {
        for j in 0..k {
            *full.at_mut(i, j) = q.at(i, j);
        }
    }
    complete_columns(&mut full, k);
    full
}

/// The SVD by Golub-Kahan bidiagonalisation and implicit-shift QR on the bidiagonal (the
/// method of LAPACK's `gesvd`) for a tall `a` (`m ≥ n`); with `want_vectors == false` only the
/// singular values are computed and `u`, `v` are empty.
///
/// 1. Householder reflections from the left and right reduce `A` to an upper bidiagonal
///    `B = U₀ᴴ A V₀`; diagonal unitary scalings make `B` real and non-negative.
/// 2. Each QR step chases a bulge down `B` with Givens rotations from the right and left,
///    implicitly applying the QR algorithm to `BᵀB` with a Wilkinson shift; negligible
///    super-diagonal entries split the problem, and zero diagonal entries are chased out.
/// 3. The rotations accumulate into the rows of `Uᵀ` and `Vᵀ` (contiguous in row-major storage).
pub(crate) fn golub_kahan<T: Scalar>(a: &Mat<T>, want_vectors: bool) -> Result<ThinSvd<T>> {
    let (m, n) = (a.rows, a.cols);
    debug_assert!(m >= n);
    let mut w = a.clone();
    let (mut left, mut right): (Vec<Vec<T>>, Vec<Vec<T>>) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut dc = vec![T::zero(); n];
    let mut ec = vec![T::zero(); n.saturating_sub(1)];
    let mut t = vec![T::zero(); n];
    // A unit reflector v with (I − 2vvᴴ) x = α e₁, or a zero vector when x = 0.
    let reflector = |x: Vec<T>| -> (Vec<T>, T) {
        let norm = x.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 {
            return (x, T::zero());
        }
        let phase = if x[0].abs() > 0.0 { x[0].scale(1.0 / x[0].abs()) } else { T::one() };
        let alpha = phase.scale(-norm);
        let mut v = x;
        v[0] -= alpha;
        let vnorm = v.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
        v.iter_mut().for_each(|z| *z = z.scale(1.0 / vnorm));
        (v, alpha)
    };
    for k in 0..n {
        // Left: zero column k below the diagonal; rows k.., columns k.. .
        let (v, alpha) = reflector((k..m).map(|i| w.data[i * n + k]).collect());
        let t = &mut t[..n - k];
        t.fill(T::zero());
        for (r, &vr) in v.iter().enumerate() {
            axpy_sub(t, -vr.conj(), &w.data[(k + r) * n + k..(k + r + 1) * n]);
        }
        for (r, &vr) in v.iter().enumerate() {
            axpy_sub(&mut w.data[(k + r) * n + k..(k + r + 1) * n], vr.scale(2.0), t);
        }
        dc[k] = alpha;
        left.push(v);
        if k + 1 < n {
            // Right: zero row k beyond the super-diagonal. With y = conj(row), (I − 2uuᴴ) y = β e₁
            // makes row (I − 2uuᴴ) = conj(β) e₁ᵀ; rows k+1.. are updated as row ← row − 2 (row·u) uᴴ.
            let (u, beta) = reflector((k + 1..n).map(|j| w.data[k * n + j].conj()).collect());
            let u_conj: Vec<T> = u.iter().map(|z| z.conj()).collect();
            for row in w.data[(k + 1) * n..].chunks_exact_mut(n) {
                let r = &mut row[k + 1..];
                let s = dot_conj(r, &u_conj).scale(2.0);
                axpy_sub(r, s, &u_conj);
            }
            ec[k] = beta.conj();
            right.push(u);
        }
    }
    // Real non-negative bidiagonal: B = diag(φ)ᴴ B_c diag(ψ), U ← U diag(φ), V ← V diag(ψ).
    let phase = |z: T| if z.abs() > 0.0 { z.scale(1.0 / z.abs()) } else { T::one() };
    let (mut phi, mut psi) = (vec![T::one(); n], vec![T::one(); n]);
    let (mut d, mut e) = (vec![0.0; n], vec![0.0; n.saturating_sub(1)]);
    for k in 0..n {
        if k > 0 {
            psi[k] = phase(phi[k - 1].conj() * ec[k - 1]).conj();
            e[k - 1] = ec[k - 1].abs();
        }
        phi[k] = phase(dc[k] * psi[k]);
        d[k] = dc[k].abs();
    }
    // Uᵀ (n × m) and Vᵀ (n × n), rows scaled by the phases.
    let (mut ut, mut vt) = (Mat::<T>::zeros(0, 0), Mat::<T>::zeros(0, 0));
    if want_vectors {
        let q = crate::reflectors::accumulate(m, &left, 0);
        ut = Mat::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                ut.data[j * m + i] = q.data[i * m + j] * phi[j];
            }
        }
        let p = crate::reflectors::accumulate(n, &right, 1);
        vt = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                vt.data[j * n + i] = p.data[i * n + j] * psi[j];
            }
        }
    }
    bidiagonal_qr(&mut d, &mut e, want_vectors.then_some(&mut ut), want_vectors.then_some(&mut vt))?;
    // Non-negative singular values, in descending order.
    for k in 0..n {
        if d[k] < 0.0 {
            d[k] = -d[k];
            if want_vectors {
                vt.data[k * n..(k + 1) * n].iter_mut().for_each(|x| *x = -*x);
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].total_cmp(&d[i]));
    let sigma: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    if !want_vectors {
        return Ok(ThinSvd { u: Mat::zeros(0, 0), sigma, v: Mat::zeros(0, 0) });
    }
    let mut u = Mat::<T>::zeros(m, n);
    let mut v = Mat::<T>::zeros(n, n);
    for (out, &k) in order.iter().enumerate() {
        for i in 0..m {
            u.data[i * n + out] = ut.data[k * m + i];
        }
        for i in 0..n {
            v.data[i * n + out] = vt.data[k * n + i];
        }
    }
    Ok(ThinSvd { u, sigma, v })
}

/// `(c, s, r)` with `c = f / r`, `s = g / r`, `r = √(f² + g²)` (so `c f + s g = r`, `−s f + c g = 0`).
#[inline]
fn rotation(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    let r = f.hypot(g);
    (f / r, g / r, r)
}

/// Rotates rows `i` and `j` of `x`: `(x_i, x_j) ← (c x_i + s x_j, −s x_i + c x_j)`.
#[inline]
fn rotate<T: Scalar>(x: &mut Mat<T>, i: usize, j: usize, c: f64, s: f64) {
    let n = x.cols;
    let (lo, hi) = (i.min(j), i.max(j));
    let (head, tail) = x.data.split_at_mut(hi * n);
    let (a, b) = (&mut head[lo * n..(lo + 1) * n], &mut tail[..n]);
    let (xi, xj) = if i < j { (a, b) } else { (b, a) };
    for (p, q) in xi.iter_mut().zip(xj.iter_mut()) {
        let (u, w) = (*p, *q);
        *p = u.scale(c) + w.scale(s);
        *q = w.scale(c) - u.scale(s);
    }
}

/// Implicit-shift QR on the real upper bidiagonal with diagonal `d` and super-diagonal `e`,
/// accumulating the left rotations into the rows of `ut` and the right ones into the rows of `vt`.
fn bidiagonal_qr<T: Scalar>(d: &mut [f64], e: &mut [f64], mut ut: Option<&mut Mat<T>>, mut vt: Option<&mut Mat<T>>) -> Result<()> {
    let n = d.len();
    if n < 2 {
        return Ok(());
    }
    let scale = d.iter().chain(e.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
    let tiny = f64::EPSILON * scale;
    let mut hi = n - 1;
    let mut iterations = 0usize;
    while hi > 0 {
        for i in 0..hi {
            if e[i].abs() <= f64::EPSILON * (d[i].abs() + d[i + 1].abs()) || e[i].abs() <= f64::MIN_POSITIVE {
                e[i] = 0.0;
            }
        }
        if e[hi - 1] == 0.0 {
            hi -= 1;
            continue;
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != 0.0 {
            lo -= 1;
        }
        iterations += 1;
        if iterations > 75 * n {
            return Err(LinalgError::NoConvergence("the bidiagonal QR iteration"));
        }
        // A negligible diagonal entry: rotate its row's super-diagonal entry out to the right
        // (or, at the bottom, its column's entry out upwards), splitting the block.
        if let Some(i) = (lo..hi).find(|&i| d[i].abs() <= tiny) {
            d[i] = 0.0;
            let mut f = e[i];
            e[i] = 0.0;
            for j in i + 1..=hi {
                let (c, s, r) = rotation(d[j], f);
                d[j] = r;
                if j < hi {
                    f = -s * e[j];
                    e[j] *= c;
                }
                if let Some(ut) = ut.as_deref_mut() {
                    rotate(ut, j, i, c, s);
                }
            }
            continue;
        }
        if d[hi].abs() <= tiny {
            d[hi] = 0.0;
            let mut f = e[hi - 1];
            e[hi - 1] = 0.0;
            for j in (lo..hi).rev() {
                let (c, s, r) = rotation(d[j], f);
                d[j] = r;
                if j > lo {
                    f = -s * e[j - 1];
                    e[j - 1] *= c;
                }
                if let Some(vt) = vt.as_deref_mut() {
                    rotate(vt, j, hi, c, s);
                }
            }
            continue;
        }
        // Wilkinson shift from the trailing 2 × 2 block of BᵀB.
        let t11 = d[hi - 1] * d[hi - 1] + if hi - 1 > lo { e[hi - 2] * e[hi - 2] } else { 0.0 };
        let t12 = d[hi - 1] * e[hi - 1];
        let t22 = d[hi] * d[hi] + e[hi - 1] * e[hi - 1];
        let half = (t11 - t22) / 2.0;
        let sign = if half >= 0.0 { 1.0 } else { -1.0 };
        let mu = t22 - t12 * t12 / (half + sign * half.hypot(t12));
        let (mut y, mut z) = (d[lo] * d[lo] - mu, d[lo] * e[lo]);
        for k in lo..hi {
            // From the right on columns k, k+1: zero z against y.
            let (c, s, r) = rotation(y, z);
            if k > lo {
                e[k - 1] = r;
            }
            let (dk, ek, dk1) = (d[k], e[k], d[k + 1]);
            let (b_kk, b_kk1, bulge, b_k1k1) = (c * dk + s * ek, -s * dk + c * ek, s * dk1, c * dk1);
            if let Some(vt) = vt.as_deref_mut() {
                rotate(vt, k, k + 1, c, s);
            }
            // From the left on rows k, k+1: zero the bulge below the diagonal.
            let (c2, s2, r2) = rotation(b_kk, bulge);
            d[k] = r2;
            e[k] = c2 * b_kk1 + s2 * b_k1k1;
            d[k + 1] = -s2 * b_kk1 + c2 * b_k1k1;
            if k + 1 < hi {
                y = e[k];
                z = s2 * e[k + 1];
                e[k + 1] *= c2;
            }
            if let Some(ut) = ut.as_deref_mut() {
                rotate(ut, k, k + 1, c2, s2);
            }
        }
    }
    Ok(())
}
