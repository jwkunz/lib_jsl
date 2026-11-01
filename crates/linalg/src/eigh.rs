//! Eigendecomposition of Hermitian (real symmetric) matrices.
//!
//! 1. Householder reflections reduce `A` to a Hermitian tridiagonal `T = Qᴴ A Q`.
//! 2. A diagonal unitary scaling `D` makes the off-diagonal real and non-negative.
//! 3. The implicit QL algorithm with Wilkinson-like shifts diagonalises the real symmetric
//!    tridiagonal matrix, accumulating rotations `Z`.
//!
//! The eigenvectors are the columns of `Q D Z`. Cost is about `9n³` flops with vectors.
//!
//! See the book entry [Symmetric and general eigensolvers](https://jwkunz.github.io/lib_jsl/book/linear_algebra/eigensolvers.html#hermitian-matrices-tridiagonalisation-and-ql).

use crate::dense::{Mat, axpy_sub, dot_conj, require_square, sub_matmul};
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
    // Householder tridiagonalisation; the reflectors are kept to form Q afterwards.
    let mut reflectors: Vec<Vec<T>> = Vec::new();
    for k in 0..n.saturating_sub(2) {
        let x: Vec<T> = (k + 1..n).map(|i| a.at(i, k)).collect();
        let norm = x.iter().map(|v| v.abs_sqr()).sum::<f64>().sqrt();
        let m = n - k - 1;
        if norm == 0.0 {
            reflectors.push(vec![T::zero(); m]);
            continue;
        }
        // α = −e^{i arg x₀} ‖x‖ avoids cancellation in v = x − α e₁.
        let phase = if x[0].abs() > 0.0 { x[0].scale(1.0 / x[0].abs()) } else { T::one() };
        let alpha = phase.scale(-norm);
        let mut v = x;
        v[0] -= alpha;
        let vnorm = v.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
        if vnorm == 0.0 {
            reflectors.push(vec![T::zero(); m]);
            continue;
        }
        v.iter_mut().for_each(|z| *z = z.scale(1.0 / vnorm));
        // p = A₂₂ v, μ = vᴴ p, w = p − μ v; A₂₂ ← A₂₂ − 2 v wᴴ − 2 w vᴴ. Only the lower triangle
        // is read and updated: one pass over it gives the whole Hermitian product, since each
        // stored a_rj (j < r) contributes a_rj v_j to p_r and conj(a_rj) v_r to p_j.
        let v_conj: Vec<T> = v.iter().map(|z| z.conj()).collect();
        let mut p = vec![T::zero(); m];
        for r in 0..m {
            let row = &a.data[(k + 1 + r) * n + k + 1..(k + 1 + r) * n + k + 2 + r];
            p[r] += dot_conj(row, &v_conj[..=r]);
            let vr = v[r];
            for (pj, &arj) in p[..r].iter_mut().zip(&row[..r]) {
                *pj += arj.conj() * vr;
            }
        }
        let mu = dot_conj(&p, &v);
        let w2: Vec<T> = p.iter().zip(&v).map(|(&pp, &vv)| (pp - mu * vv).scale(2.0)).collect();
        let w2_conj: Vec<T> = w2.iter().map(|z| z.conj()).collect();
        for r in 0..m {
            let row = &mut a.data[(k + 1 + r) * n + k + 1..(k + 1 + r) * n + k + 2 + r];
            axpy_sub(row, v[r], &w2_conj[..=r]);
            axpy_sub(row, w2[r], &v_conj[..=r]);
        }
        *a.at_mut(k + 1, k) = alpha;
        *a.at_mut(k, k + 1) = alpha.conj();
        for r in k + 2..n {
            *a.at_mut(r, k) = T::zero();
            *a.at_mut(k, r) = T::zero();
        }
        reflectors.push(v);
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
    if !want_vectors {
        tql(&mut d, &mut e, None)?;
        d.sort_by(f64::total_cmp);
        return Ok((d, None));
    }
    // Q = H₀ H₁ ⋯ H_{n−3}, accumulated backwards in blocks.
    let mut q = crate::reflectors::accumulate(n, &reflectors, 1);
    for row in 0..n {
        for (c, ph) in phases.iter().enumerate() {
            q.data[row * n + c] *= *ph;
        }
    }
    // The QL rotations act on columns of Z; storing Zᵀ makes each rotation a pair of rows.
    let mut zt = Mat::<f64>::identity(n);
    tql(&mut d, &mut e, Some(&mut zt))?;
    // Sort ascending and form Q D Z with the matrix-multiply kernel.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let values: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let mut minus_z = Mat::<T>::zeros(n, n);
    for (out_col, &col) in order.iter().enumerate() {
        for k in 0..n {
            minus_z.data[k * n + out_col] = T::from_real(-zt.data[col * n + k]);
        }
    }
    let mut vectors = Mat::<T>::zeros(n, n);
    // SAFETY: three distinct n × n buffers.
    unsafe { sub_matmul(n, n, n, q.data.as_ptr(), n, minus_z.data.as_ptr(), n, vectors.data.as_mut_ptr(), n) };
    Ok((values, Some(vectors)))
}

// ANCHOR: tql
/// Implicit QL with shifts on the symmetric tridiagonal matrix with diagonal `d` and
/// sub-diagonal `e` (`e[i]` couples `i` and `i+1`; `e[n−1]` is unused), accumulating the
/// rotations into the *rows* of `zt` (the transpose of the eigenvector matrix; Numerical Recipes
/// `tqli`).
fn tql(d: &mut [f64], e: &mut [f64], mut zt: Option<&mut Mat<f64>>) -> Result<()> {
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
                if let Some(zt) = zt.as_deref_mut() {
                    let (row_i, row_next) = zt.data[i * n..(i + 2) * n].split_at_mut(n);
                    for (zi, f) in row_i.iter_mut().zip(row_next.iter_mut()) {
                        let (a, b) = (*zi, *f);
                        *f = s * a + c * b;
                        *zi = c * a - s * b;
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
// ANCHOR_END: tql
