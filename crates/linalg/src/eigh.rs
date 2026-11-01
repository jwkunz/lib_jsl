//! Eigendecomposition of Hermitian (real symmetric) matrices.
//!
//! 1. Householder reflections reduce `A` to a Hermitian tridiagonal `T = Qᴴ A Q`.
//! 2. A diagonal unitary scaling `D` makes the off-diagonal real and non-negative.
//! 3. The implicit QL algorithm with Wilkinson-like shifts diagonalises the real symmetric
//!    tridiagonal matrix, accumulating rotations `Z`.
//!
//! The eigenvectors are the columns of `Q D Z`. Cost is about `9n³` flops with vectors.

use crate::dense::{Mat, require_square};
use crate::error::{LinalgError, Result, UPLO};
use crate::scalar::Scalar;

/// Eigenvalues in ascending order and the matching orthonormal eigenvectors as columns.
pub(crate) fn eigh<T: Scalar>(input: &Mat<T>, uplo: UPLO) -> Result<(Vec<f64>, Mat<T>)> {
    let (values, vectors) = eigh_impl(input, uplo, true)?;
    Ok((values, vectors.expect("vectors requested")))
}

/// Eigenvalues only, in ascending order (skips accumulating the transformations).
pub(crate) fn eigvalsh<T: Scalar>(input: &Mat<T>, uplo: UPLO) -> Result<Vec<f64>> {
    Ok(eigh_impl(input, uplo, false)?.0)
}

fn eigh_impl<T: Scalar>(input: &Mat<T>, uplo: UPLO, want_vectors: bool) -> Result<(Vec<f64>, Option<Mat<T>>)> {
    require_square(input.rows, input.cols)?;
    let n = input.rows;
    if n == 0 {
        return Ok((Vec::new(), want_vectors.then(|| Mat::zeros(0, 0))));
    }
    // Build the full Hermitian matrix from the requested triangle.
    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let lower = i >= j;
            let v = if (uplo == UPLO::Lower) == lower { input.at(i, j) } else { input.at(j, i).conj() };
            *a.at_mut(i, j) = if i == j { T::from_real(v.re()) } else { v };
        }
    }
    let mut q = Mat::<T>::identity(n);
    // Householder tridiagonalisation.
    for k in 0..n.saturating_sub(2) {
        let x: Vec<T> = (k + 1..n).map(|i| a.at(i, k)).collect();
        let norm = x.iter().map(|v| v.abs_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        // α = −e^{i arg x₀} ‖x‖ avoids cancellation in v = x − α e₁.
        let phase = if x[0].abs() > 0.0 { x[0].scale(1.0 / x[0].abs()) } else { T::one() };
        let alpha = phase.scale(-norm);
        let mut v = x;
        v[0] -= alpha;
        let vnorm = v.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
        if vnorm == 0.0 {
            continue;
        }
        v.iter_mut().for_each(|z| *z = z.scale(1.0 / vnorm));
        let m = n - k - 1;
        // p = A₂₂ v, μ = vᴴ p, w = p − μ v; A₂₂ ← A₂₂ − 2 v wᴴ − 2 w vᴴ.
        let mut p = vec![T::zero(); m];
        for (r, pr) in p.iter_mut().enumerate() {
            let row = &a.data[(k + 1 + r) * n + k + 1..(k + 2 + r) * n];
            *pr = row.iter().zip(&v).fold(T::zero(), |s, (&ar, &vv)| s + ar * vv);
        }
        let mu = v.iter().zip(&p).fold(T::zero(), |s, (&vv, &pp)| s + vv.conj() * pp);
        let w: Vec<T> = p.iter().zip(&v).map(|(&pp, &vv)| pp - mu * vv).collect();
        for r in 0..m {
            for c in 0..m {
                let delta = (v[r] * w[c].conj() + w[r] * v[c].conj()).scale(2.0);
                *a.at_mut(k + 1 + r, k + 1 + c) -= delta;
            }
        }
        *a.at_mut(k + 1, k) = alpha;
        *a.at_mut(k, k + 1) = alpha.conj();
        for r in k + 2..n {
            *a.at_mut(r, k) = T::zero();
            *a.at_mut(k, r) = T::zero();
        }
        // Q ← Q (I − 2 v vᴴ) on columns k+1..n.
        if !want_vectors {
            continue;
        }
        for row in 0..n {
            let base = row * n + k + 1;
            let s = (0..m).fold(T::zero(), |s, c| s + q.data[base + c] * v[c]).scale(2.0);
            for c in 0..m {
                q.data[base + c] -= s * v[c].conj();
            }
        }
    }
    // Real tridiagonal via the phase scaling D.
    let mut d: Vec<f64> = (0..n).map(|i| a.at(i, i).re()).collect();
    let mut e = vec![0.0; n];
    let mut phases = vec![T::one(); n];
    for k in 0..n - 1 {
        let sub = a.at(k + 1, k);
        let m = sub.abs();
        e[k] = m;
        phases[k + 1] = if m > 0.0 { phases[k] * sub.scale(1.0 / m) } else { phases[k] };
    }
    for row in 0..n {
        for (c, ph) in phases.iter().enumerate() {
            q.data[row * n + c] *= *ph;
        }
    }
    if !want_vectors {
        tql(&mut d, &mut e, None)?;
        d.sort_by(f64::total_cmp);
        return Ok((d, None));
    }
    let mut z = Mat::<f64>::identity(n);
    tql(&mut d, &mut e, Some(&mut z))?;
    // Sort ascending and form Q D Z.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let values: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let mut vectors = Mat::<T>::zeros(n, n);
    for row in 0..n {
        for (out_col, &col) in order.iter().enumerate() {
            let mut s = T::zero();
            for k in 0..n {
                s += q.at(row, k).scale(z.at(k, col));
            }
            *vectors.at_mut(row, out_col) = s;
        }
    }
    Ok((values, Some(vectors)))
}

/// Implicit QL with shifts on the symmetric tridiagonal matrix with diagonal `d` and
/// sub-diagonal `e` (`e[i]` couples `i` and `i+1`; `e[n−1]` is unused), accumulating the
/// rotations into the columns of `z` (Numerical Recipes `tqli`).
fn tql(d: &mut [f64], e: &mut [f64], mut z: Option<&mut Mat<f64>>) -> Result<()> {
    let n = d.len();
    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            if iterations > 60 {
                return Err(LinalgError::NoConvergence("the symmetric QL iteration"));
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut i = m;
            let mut underflow = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                if let Some(z) = z.as_deref_mut() {
                    for k in 0..n {
                        let f = z.at(k, i + 1);
                        let zi = z.at(k, i);
                        *z.at_mut(k, i + 1) = s * zi + c * f;
                        *z.at_mut(k, i) = c * zi - s * f;
                    }
                }
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    Ok(())
}
