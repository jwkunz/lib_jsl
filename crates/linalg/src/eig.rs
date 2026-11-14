//! Eigendecomposition of general square matrices via the Schur form.
//!
//! 0. Balancing by a diagonal similarity with power-of-two scale factors equalises the row and
//!    column norms (as LAPACK's `geev` does by default); the eigenvectors are scaled back at the end.
//! 1. Householder reflections reduce `A` to upper Hessenberg form `H = Qᴴ A Q`.
//! 2. The shifted QR algorithm drives `H` to the Schur form `T = Zᴴ A Z`:
//!    - real matrices use Francis's implicit double-shift step, in real arithmetic, reaching the
//!      real Schur form (2 × 2 diagonal blocks for complex-conjugate pairs). Each block is then
//!      triangularised by one unitary rotation, giving the complex Schur form;
//!    - complex matrices use single shifts (Wilkinson, with exceptional shifts) and Givens
//!      rotations.
//! 3. Eigenvectors of the triangular `T` follow by back-substitution and are mapped back
//!    through `Z`.
//!
//! The accumulated transformation is stored transposed (`Zᵀ`), so that the rotations, which
//! combine columns of `Z`, update contiguous rows.
//!
//! See the book entry [Symmetric and general eigensolvers](https://jwkunz.github.io/lib_jsl/book/linear_algebra/eigensolvers.html#general-matrices-hessenberg-reduction-and-franciss-algorithm).

use crate::dense::{Mat, axpy_sub, dot_conj, require_square};
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

/// Reduces `h` to upper Hessenberg form in place and returns `Qᵀ` if requested.
fn hessenberg<T: Scalar>(h: &mut Mat<T>, want_q: bool) -> Option<Mat<T>> {
    let n = h.rows;
    let mut reflectors: Vec<Vec<T>> = Vec::new();
    let mut t = vec![T::zero(); n];
    for k in 0..n.saturating_sub(2) {
        let m = n - k - 1;
        let x: Vec<T> = (k + 1..n).map(|i| h.at(i, k)).collect();
        let norm = x.iter().map(|v| v.abs_sqr()).sum::<f64>().sqrt();
        let phase = if x[0].abs() > 0.0 { x[0].scale(1.0 / x[0].abs()) } else { T::one() };
        let alpha = phase.scale(-norm);
        let mut v = x;
        v[0] -= alpha;
        let vnorm = v.iter().map(|z| z.abs_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 || vnorm == 0.0 {
            reflectors.push(vec![T::zero(); m]);
            continue;
        }
        v.iter_mut().for_each(|z| *z = z.scale(1.0 / vnorm));
        // H ← (I − 2vvᴴ) H on rows k+1..n (columns k..n): t = vᴴ H, then rows −= 2 v_r t.
        let t = &mut t[..n - k];
        t.fill(T::zero());
        for (r, &vr) in v.iter().enumerate() {
            axpy_sub(t, -vr.conj(), &h.data[(k + 1 + r) * n + k..(k + 2 + r) * n]);
        }
        for (r, &vr) in v.iter().enumerate() {
            axpy_sub(&mut h.data[(k + 1 + r) * n + k..(k + 2 + r) * n], vr.scale(2.0), t);
        }
        // H ← H (I − 2vvᴴ) on columns k+1..n: each row loses 2 (row · v) vᴴ.
        let v_conj: Vec<T> = v.iter().map(|z| z.conj()).collect();
        for row in 0..n {
            let r = &mut h.data[row * n + k + 1..(row + 1) * n];
            let s = dot_conj(r, &v_conj).scale(2.0);
            axpy_sub(r, s, &v_conj);
        }
        *h.at_mut(k + 1, k) = alpha;
        for r in k + 2..n {
            *h.at_mut(r, k) = T::zero();
        }
        reflectors.push(v);
    }
    if !want_q {
        return None;
    }
    let q = crate::reflectors::accumulate(n, &reflectors, 1);
    let mut qt = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            qt.data[j * n + i] = q.data[i * n + j];
        }
    }
    Some(qt)
}

/// Rotates rows `i` and `j` of `x`: `(a, b) ← (f(a, b), g(a, b))` elementwise.
#[inline]
fn rotate_rows<T: Copy>(x: &mut Mat<T>, i: usize, j: usize, f: impl Fn(T, T) -> (T, T)) {
    let n = x.cols;
    debug_assert!(i < j);
    let (head, tail) = x.data.split_at_mut(j * n);
    for (a, b) in head[i * n..(i + 1) * n].iter_mut().zip(&mut tail[..n]) {
        (*a, *b) = f(*a, *b);
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

/// Single-shift complex QR on the Hessenberg `t`, to upper triangular form; rotations are
/// accumulated into the rows of `zt`.
fn complex_schur(t: &mut Mat<C>, mut zt: Option<&mut Mat<C>>) -> Result<()> {
    let n = t.rows;
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
            let (head, tail) = t.data.split_at_mut((k + 1) * n);
            for (x, y) in head[k * n + k..].iter_mut().zip(&mut tail[k..n]) {
                (*x, *y) = (*x * c + s * *y, -s.conj() * *x + *y * c);
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
            if let Some(zt) = zt.as_deref_mut() {
                rotate_rows(zt, k, k + 1, |p, q| (p * c + q * s.conj(), -p * s + q * c));
            }
        }
        for k in lo..hi {
            *t.at_mut(k, k) += mu;
        }
    }
    Ok(())
}

/// Francis double-shift QR on the real Hessenberg `a`, to real Schur form (after the EISPACK
/// routine `hqr2`, as presented in *Numerical Recipes*). Returns, for each `k`, whether a
/// complex-conjugate 2 × 2 block starts at `(k, k)`.
fn real_schur(a: &mut Mat<f64>, mut zt: Option<&mut Mat<f64>>) -> Result<Vec<bool>> {
    let n = a.rows;
    let mut pair = vec![false; n];
    let anorm: f64 = (0..n).map(|i| a.data[i * n + i.saturating_sub(1)..(i + 1) * n].iter().map(|v| v.abs()).sum::<f64>()).sum();
    let mut shift_total = 0.0;
    let mut hi = n; // the active matrix is rows/columns 0..hi
    let mut its = 0;
    // Without Z only the active block need be kept up to date (EISPACK's hqr rather than hqr2):
    // entries outside it are never read again.
    let full = zt.is_some();
    let col_end = |nn: usize| if full { n } else { nn + 1 };
    let row_start = |l: usize| if full { 0 } else { l };
    macro_rules! at {
        ($i:expr, $j:expr) => {
            a.data[($i) * n + ($j)]
        };
    }
    while hi > 0 {
        let nn = hi - 1;
        // The lowest negligible sub-diagonal entry splits off the active block l..=nn.
        let mut l = nn;
        while l > 0 {
            let mut s = at!(l - 1, l - 1).abs() + at!(l, l).abs();
            if s == 0.0 {
                s = anorm;
            }
            if at!(l, l - 1).abs() <= f64::EPSILON * s {
                at!(l, l - 1) = 0.0;
                break;
            }
            l -= 1;
        }
        let mut x = at!(nn, nn);
        if l == nn {
            // A single real eigenvalue.
            at!(nn, nn) = x + shift_total;
            hi -= 1;
            its = 0;
            continue;
        }
        let mut y = at!(nn - 1, nn - 1);
        let mut w = at!(nn, nn - 1) * at!(nn - 1, nn);
        if l == nn - 1 {
            // A 2 × 2 block: a real pair (rotated to triangular form) or a complex pair.
            let p = 0.5 * (y - x);
            let q = p * p + w;
            let z = q.abs().sqrt();
            x += shift_total;
            at!(nn, nn) = x;
            at!(nn - 1, nn - 1) = y + shift_total;
            if q >= 0.0 {
                let z = p + z.copysign(p);
                let x = at!(nn, nn - 1);
                let s = x.abs() + z.abs();
                let (p, q) = (x / s, z / s);
                let r = p.hypot(q);
                let (p, q) = (p / r, q / r);
                for j in nn - 1..col_end(nn) {
                    let (u, v) = (at!(nn - 1, j), at!(nn, j));
                    at!(nn - 1, j) = q * u + p * v;
                    at!(nn, j) = q * v - p * u;
                }
                for i in row_start(l)..=nn {
                    let (u, v) = (at!(i, nn - 1), at!(i, nn));
                    at!(i, nn - 1) = q * u + p * v;
                    at!(i, nn) = q * v - p * u;
                }
                if let Some(zt) = zt.as_deref_mut() {
                    rotate_rows(zt, nn - 1, nn, |u, v| (q * u + p * v, q * v - p * u));
                }
                at!(nn, nn - 1) = 0.0;
            } else {
                pair[nn - 1] = true;
            }
            hi -= 2;
            its = 0;
            continue;
        }
        if its == 60 {
            return Err(LinalgError::NoConvergence("the Francis QR iteration"));
        }
        if its == 10 || its == 20 {
            // Exceptional shift.
            shift_total += x;
            for i in 0..=nn {
                at!(i, i) -= x;
            }
            let s = at!(nn, nn - 1).abs() + at!(nn - 1, nn - 2).abs();
            x = 0.75 * s;
            y = x;
            w = -0.4375 * s * s;
        }
        its += 1;
        // Look for two consecutive small sub-diagonal entries to start the bulge lower.
        let mut m = nn - 2;
        let (mut p, mut q, mut r);
        loop {
            let z = at!(m, m);
            let rr = x - z;
            let s = y - z;
            p = (rr * s - w) / at!(m + 1, m) + at!(m, m + 1);
            q = at!(m + 1, m + 1) - z - rr - s;
            r = at!(m + 2, m + 1);
            let s = p.abs() + q.abs() + r.abs();
            p /= s;
            q /= s;
            r /= s;
            if m == l {
                break;
            }
            let u = at!(m, m - 1).abs() * (q.abs() + r.abs());
            let v = p.abs() * (at!(m - 1, m - 1).abs() + z.abs() + at!(m + 1, m + 1).abs());
            if u <= f64::EPSILON * v {
                break;
            }
            m -= 1;
        }
        for i in m..nn - 1 {
            at!(i + 2, i) = 0.0;
            if i != m {
                at!(i + 2, i - 1) = 0.0;
            }
        }
        // Chase the bulge with 3 × 3 Householder reflections.
        for k in m..nn {
            let last = k + 1 == nn;
            if k != m {
                p = at!(k, k - 1);
                q = at!(k + 1, k - 1);
                r = if last { 0.0 } else { at!(k + 2, k - 1) };
                x = p.abs() + q.abs() + r.abs();
                if x != 0.0 {
                    p /= x;
                    q /= x;
                    r /= x;
                }
            }
            let s = (p * p + q * q + r * r).sqrt().copysign(p);
            if s == 0.0 {
                continue;
            }
            if k == m {
                if l != m {
                    at!(k, k - 1) = -at!(k, k - 1);
                }
            } else {
                at!(k, k - 1) = -s * x;
            }
            p += s;
            let (hx, hy, hz) = (p / s, q / s, r / s);
            q /= p;
            r /= p;
            for j in k..col_end(nn) {
                let mut t = at!(k, j) + q * at!(k + 1, j);
                if !last {
                    t += r * at!(k + 2, j);
                    at!(k + 2, j) -= t * hz;
                }
                at!(k + 1, j) -= t * hy;
                at!(k, j) -= t * hx;
            }
            for i in row_start(l)..=nn.min(k + 3) {
                let mut t = hx * at!(i, k) + hy * at!(i, k + 1);
                if !last {
                    t += hz * at!(i, k + 2);
                    at!(i, k + 2) -= t * r;
                }
                at!(i, k + 1) -= t * q;
                at!(i, k) -= t;
            }
            if let Some(zt) = zt.as_deref_mut() {
                let rows = if last { 2 } else { 3 };
                let block = &mut zt.data[k * n..(k + rows) * n];
                let (r0, rest) = block.split_at_mut(n);
                let (r1, r2) = rest.split_at_mut(n);
                for i in 0..n {
                    let mut t = hx * r0[i] + hy * r1[i];
                    if !last {
                        t += hz * r2[i];
                        r2[i] -= t * r;
                    }
                    r1[i] -= t * q;
                    r0[i] -= t;
                }
            }
        }
    }
    // Clear everything below the sub-diagonal, and the sub-diagonal outside 2 × 2 blocks.
    for i in 1..n {
        for j in 0..i {
            if !(j + 1 == i && pair[j]) {
                at!(i, j) = 0.0;
            }
        }
    }
    Ok(pair)
}

/// Triangularises each complex-conjugate 2 × 2 block of a real Schur form with one unitary
/// rotation `G = [x, x^⊥]`, `x` the unit eigenvector of the block, turning it into the complex
/// Schur form.
fn complexify(a: &Mat<f64>, pair: &[bool], zt: Option<&Mat<f64>>) -> (Mat<C>, Option<Mat<C>>) {
    let n = a.rows;
    let to_c = |m: &Mat<f64>| Mat { rows: m.rows, cols: m.cols, data: m.data.iter().map(|&v| C::new(v, 0.0)).collect() };
    let mut t = to_c(a);
    let mut zt = zt.map(to_c);
    for k in (0..n).filter(|&k| pair[k]) {
        let (a, b, c, d) = (t.at(k, k), t.at(k, k + 1), t.at(k + 1, k), t.at(k + 1, k + 1));
        let half = (a - d) * 0.5;
        let lambda = (a + d) * 0.5 + (half * half + b * c).sqrt();
        let (x0, x1) = (b, lambda - a);
        let norm = (x0.norm_sqr() + x1.norm_sqr()).sqrt();
        let (x0, x1) = (x0 / norm, x1 / norm);
        // T ← Gᴴ T on rows k, k+1; T ← T G and Z ← Z G on columns k, k+1.
        let (h0, h1) = (x0.conj(), x1.conj());
        let right = move |u: C, v: C| (u * x0 + v * x1, v * h0 - u * h1);
        let (head, tail) = t.data.split_at_mut((k + 1) * n);
        for (u, v) in head[k * n + k..].iter_mut().zip(&mut tail[k..n]) {
            (*u, *v) = (h0 * *u + h1 * *v, x0 * *v - x1 * *u);
        }
        for i in 0..=k + 1 {
            let (u, v) = right(t.at(i, k), t.at(i, k + 1));
            *t.at_mut(i, k) = u;
            *t.at_mut(i, k + 1) = v;
        }
        if let Some(zt) = zt.as_mut() {
            rotate_rows(zt, k, k + 1, right);
        }
        *t.at_mut(k + 1, k) = C::new(0.0, 0.0);
    }
    (t, zt)
}

/// The Schur form, `Zᵀ` if requested, and the balancing scale factors.
type BalancedSchur = (Mat<C>, Option<Mat<C>>, Vec<f64>);

// ANCHOR: balance
/// Balances `a` in place by a diagonal similarity `D⁻¹ A D` (Parlett and Reinsch, 1969, as in
/// LAPACK's `gebal`), so that each row and the matching column have similar norms; returns `D`.
/// The scale factors are powers of two, so the eigenvalues are unchanged exactly, while the
/// rounding errors of the QR iteration, proportional to `‖A‖`, can shrink by many orders of
/// magnitude (companion matrices of polynomials with widely ranging coefficients are the classic
/// case). An eigenvector `v` of the balanced matrix is `D v` for the original.
fn balance<T: Scalar>(a: &mut Mat<T>) -> Vec<f64> {
    let n = a.rows;
    let mut d = vec![1.0f64; n];
    let mut converged = false;
    while !converged {
        converged = true;
        for i in 0..n {
            let (mut c, mut r) = (0.0f64, 0.0f64);
            for j in (0..n).filter(|&j| j != i) {
                c += a.at(j, i).abs();
                r += a.at(i, j).abs();
            }
            if c == 0.0 || r == 0.0 {
                continue;
            }
            let (s, mut f) = (c + r, 1.0f64);
            let (mut c2, mut r2) = (c, r);
            while c2 < r2 / 2.0 {
                (c2, r2, f) = (c2 * 2.0, r2 / 2.0, f * 2.0);
            }
            while c2 >= r2 * 2.0 {
                (c2, r2, f) = (c2 / 2.0, r2 * 2.0, f / 2.0);
            }
            // Rescale only if it reduces the row-plus-column norm noticeably.
            if c2 + r2 < 0.95 * s {
                converged = false;
                d[i] *= f;
                for j in 0..n {
                    *a.at_mut(i, j) = a.at(i, j).scale(1.0 / f);
                    *a.at_mut(j, i) = a.at(j, i).scale(f);
                }
            }
        }
    }
    d
}
// ANCHOR_END: balance

/// The complex Schur form `T` (upper triangular) of the balanced matrix, `Zᵀ` if requested, and
/// the balancing scale factors.
fn schur<T: Scalar>(a: &Mat<T>, want_z: bool) -> Result<BalancedSchur> {
    require_square(a.rows, a.cols)?;
    let mut a = Mat { rows: a.rows, cols: a.cols, data: a.data.clone() };
    let d = balance(&mut a);
    let (t, zt) = schur_balanced(&a, want_z)?;
    Ok((t, zt, d))
}

fn schur_balanced<T: Scalar>(a: &Mat<T>, want_z: bool) -> Result<(Mat<C>, Option<Mat<C>>)> {
    if T::IS_COMPLEX {
        let mut t = Mat { rows: a.rows, cols: a.cols, data: a.data.iter().map(|v| v.to_complex()).collect() };
        let mut zt = hessenberg(&mut t, want_z);
        complex_schur(&mut t, zt.as_mut())?;
        Ok((t, zt))
    } else {
        let mut h = Mat { rows: a.rows, cols: a.cols, data: a.data.iter().map(|v| v.re()).collect() };
        let mut zt = hessenberg(&mut h, want_z);
        let pair = real_schur(&mut h, zt.as_mut())?;
        Ok(complexify(&h, &pair, zt.as_ref()))
    }
}

/// The complex Schur form `A = Z T Zᴴ` of the matrix itself (no balancing, so that `Z` is
/// unitary): `T` upper triangular with the eigenvalues on its diagonal.
pub(crate) fn schur_unbalanced<T: Scalar>(a: &Mat<T>) -> Result<(Mat<C>, Mat<C>)> {
    require_square(a.rows, a.cols)?;
    let (t, zt) = schur_balanced(a, true)?;
    let zt = zt.expect("Z requested");
    let n = t.rows;
    let mut z = Mat::<C>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *z.at_mut(i, j) = zt.at(j, i);
        }
    }
    Ok((t, z))
}

/// The eigenvalues, in Schur order.
pub(crate) fn eigvals<T: Scalar>(a: &Mat<T>) -> Result<Vec<C>> {
    let (t, _, _) = schur(a, false)?;
    Ok((0..t.rows).map(|i| t.at(i, i)).collect())
}

/// Eigenvalues (in Schur order) and unit-norm eigenvectors as columns.
pub(crate) fn eig<T: Scalar>(a: &Mat<T>) -> Result<(Vec<C>, Mat<C>)> {
    let (t, zt, d) = schur(a, true)?;
    let zt = zt.expect("Z requested");
    let n = t.rows;
    let values: Vec<C> = (0..n).map(|i| t.at(i, i)).collect();
    let norm = t.data.iter().fold(0.0f64, |m, v| m.max(v.norm())).max(f64::MIN_POSITIVE);
    let small = f64::EPSILON * norm;
    // Eigenvectors of T by back-substitution (y_k = 1, y_j = 0 for j > k), stored as rows of Yᵀ.
    let mut yt = Mat::<C>::zeros(n, n);
    for k in 0..n {
        let lambda = values[k];
        let y = &mut yt.data[k * n..(k + 1) * n];
        y[k] = C::new(1.0, 0.0);
        for j in (0..k).rev() {
            let row = &t.data[j * n + j + 1..j * n + k + 1];
            let s = row.iter().zip(&y[j + 1..=k]).fold(C::new(0.0, 0.0), |s, (&a, &b)| s + a * b);
            let mut denom = t.at(j, j) - lambda;
            if denom.norm() < small {
                denom = C::new(small, 0.0);
            }
            y[j] = -s / denom;
        }
    }
    // V = Z Y with the matrix-multiply kernel, reading both factors through transposed strides.
    let mut vectors = Mat::<C>::zeros(n, n);
    if n > 0 {
        let n_ = n as isize;
        // SAFETY: three distinct n × n buffers; (rs, cs) = (1, n) reads Zᵀ as Z and Yᵀ as Y.
        // C ← C − Z (−Y): the sign is folded into the result below.
        unsafe { C::gemm_sub(n, n, n, zt.data.as_ptr(), 1, n_, yt.data.as_ptr(), 1, n_, vectors.data.as_mut_ptr(), n_, 1) };
    }
    // Undo the balancing: the eigenvectors of A are D times those of D⁻¹ A D.
    for (row, &di) in vectors.data.chunks_exact_mut(n.max(1)).zip(&d) {
        row.iter_mut().for_each(|v| *v = v.scale(di));
    }
    let mut norms = vec![0.0f64; n];
    for row in vectors.data.chunks_exact(n.max(1)) {
        for (s, v) in norms.iter_mut().zip(row) {
            *s += v.norm_sqr();
        }
    }
    // The product above is −Z Y; flip the sign while normalising.
    let scales: Vec<f64> = norms.iter().map(|&s| if s > 0.0 { -1.0 / s.sqrt() } else { -1.0 }).collect();
    for row in vectors.data.chunks_exact_mut(n.max(1)) {
        for (v, &s) in row.iter_mut().zip(&scales) {
            *v = v.scale(s);
        }
    }
    Ok((values, vectors))
}
