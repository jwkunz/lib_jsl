//! Singular value decomposition by one-sided Jacobi rotations (Hestenes), and least squares.
//!
//! Plane rotations are applied to pairs of columns of `A` until every pair is orthogonal; the
//! column norms are then the singular values, the normalised columns the left singular vectors,
//! and the accumulated rotations the right singular vectors. The method is simple and highly
//! accurate (small singular values are found to high relative accuracy).

use crate::dense::{Mat, dot_conj};
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

/// The thin SVD of any matrix (wide matrices are handled through their adjoint).
///
/// Larger matrices are first factorised as `A P = Q R` with column pivoting; one-sided Jacobi
/// then runs on `Rᴴ`, whose columns are already close to orthogonal, so it converges in a few
/// sweeps (the preconditioning of Drmač and Veselić). With `Rᴴ = U₁ Σ W₁ᴴ`,
/// `A = (Q W₁) Σ (P U₁)ᴴ`.
pub(crate) fn thin_svd<T: Scalar>(a: &Mat<T>) -> Result<ThinSvd<T>> {
    if a.rows < a.cols {
        let ThinSvd { u, sigma, v } = thin_svd(&a.adjoint())?;
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
