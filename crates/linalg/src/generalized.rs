//! Generalised eigenvalue problems `A x = λ B x`: the symmetric-definite case by Cholesky
//! reduction (SciPy's `eigh(a, b)`), and the general case by the QZ algorithm (Moler and Stewart,
//! 1973), which gives the generalised Schur form `A = Q S Zᴴ`, `B = Q T Zᴴ` (SciPy's
//! `qz(output='complex')` and `eig(a, b)`).
//!
//! See the book entry [Generalised eigenproblems and structured matrices](https://jwkunz.github.io/lib_jsl/book/linear_algebra/generalized-structured.html).

use crate::error::{LinalgError, Result};
use crate::scalar::Scalar;
use crate::{Cholesky, Diag, Eigh, QrMode, SolveTriangular, UPLO, qr};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_complex::Complex;

type C = Complex<f64>;

fn cz() -> C {
    C::new(0.0, 0.0)
}

// ANCHOR: eigh_generalized
/// The eigenvalues (ascending) and eigenvectors of the symmetric-definite problem `A x = λ B x`,
/// `A` Hermitian and `B` Hermitian positive definite (SciPy's `eigh(a, b)`, LAPACK's `sygv`
/// type 1). With `B = L Lᴴ`, the problem becomes the standard `C y = λ y` for
/// `C = L⁻¹ A L⁻ᴴ`, and `x = L⁻ᴴ y`; the eigenvectors are `B`-orthonormal, `Xᴴ B X = I`.
pub fn eigh_generalized<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix2>, uplo: UPLO) -> Result<(Array1<f64>, Array2<T>)> {
    let n = a.nrows();
    if a.ncols() != n || b.dim() != (n, n) {
        return Err(LinalgError::Shape(format!("A ({:?}) and B ({:?}) must be square and of one size", a.dim(), b.dim())));
    }
    // Both triangles of A from the `uplo` one.
    let full = Array2::from_shape_fn((n, n), |(i, j)| {
        let read_lower = matches!(uplo, UPLO::Lower);
        if (i >= j) == read_lower || i == j { a[(i, j)] } else { a[(j, i)].conj() }
    });
    let l = b.cholesky(uplo).map(|f| if matches!(uplo, UPLO::Lower) { f } else { f.t().mapv(|v| v.conj()) })?;
    // C = L⁻¹ A L⁻ᴴ: W = L⁻¹ A, then C = (L⁻¹ Wᴴ)ᴴ.
    let w = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &full)?;
    let wh = w.t().mapv(|v| v.conj());
    let c = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &wh)?.t().mapv(|v| v.conj());
    let c = Array2::from_shape_fn((n, n), |(i, j)| (c[(i, j)] + c[(j, i)].conj()).scale(0.5));
    let (values, y) = c.eigh(UPLO::Lower)?;
    let lh = l.t().mapv(|v| v.conj());
    let x = lh.solve_triangular(UPLO::Upper, Diag::NonUnit, &y)?;
    Ok((values, x))
}
// ANCHOR_END: eigh_generalized

/// `(c, s)` with `[c s; −s̄ c] [f; g] = [r; 0]`, `c` real (LAPACK's `lartg`).
fn rotation(f: C, g: C) -> (f64, C) {
    if g == cz() {
        return (1.0, cz());
    }
    if f == cz() {
        return (0.0, g.conj() / g.norm());
    }
    let r = f.norm().hypot(g.norm());
    (f.norm() / r, f / f.norm() * g.conj() / r)
}

/// Rows `p`, `q` of `m` (columns `cols`) ← `[c s; −s̄ c]` applied to them.
fn rotate_rows(m: &mut Array2<C>, p: usize, q: usize, c: f64, s: C, cols: std::ops::Range<usize>) {
    for j in cols {
        let (x, y) = (m[(p, j)], m[(q, j)]);
        m[(p, j)] = x * c + s * y;
        m[(q, j)] = y * c - s.conj() * x;
    }
}

/// Columns `q`, `p` of `m` (rows `rows`) ← `(c q + s p, −s̄ q + c p)`: a right multiplication by a
/// unitary matrix, chosen so that it zeros an entry of column `p`.
fn rotate_cols(m: &mut Array2<C>, q: usize, p: usize, c: f64, s: C, rows: std::ops::Range<usize>) {
    for i in rows {
        let (x, y) = (m[(i, q)], m[(i, p)]);
        m[(i, q)] = x * c + s * y;
        m[(i, p)] = y * c - s.conj() * x;
    }
}

/// The generalised complex Schur form `A = Q S Zᴴ`, `B = Q T Zᴴ`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QzFactors {
    /// Upper triangular.
    pub s: Array2<C>,
    /// Upper triangular, with a real non-negative diagonal.
    pub t: Array2<C>,
    pub q: Array2<C>,
    pub z: Array2<C>,
}

/// A row rotation on rows `p`, `q` of `S` and `T` (columns from `from`), with `Q ← Q Gᴴ`.
fn left_rot(f: &mut QzFactors, p: usize, q: usize, c: f64, s: C, from: usize) {
    let n = f.s.ncols();
    rotate_rows(&mut f.s, p, q, c, s, from..n);
    rotate_rows(&mut f.t, p, q, c, s, from..n);
    // Q ← Q Gᴴ: columns (p, q) ← (c Q_p + s̄ Q_q, −s Q_p + c Q_q).
    for i in 0..n {
        let (x, y) = (f.q[(i, p)], f.q[(i, q)]);
        f.q[(i, p)] = x * c + s.conj() * y;
        f.q[(i, q)] = y * c - s * x;
    }
}

/// A column rotation on columns `q`, `p` of `S` (rows `..s_rows`) and `T` (rows `..t_rows`), with
/// the same rotation applied to `Z`.
fn right_rot(f: &mut QzFactors, q: usize, p: usize, c: f64, s: C, s_rows: usize, t_rows: usize) {
    let n = f.s.ncols();
    rotate_cols(&mut f.s, q, p, c, s, 0..s_rows);
    rotate_cols(&mut f.t, q, p, c, s, 0..t_rows);
    rotate_cols(&mut f.z, q, p, c, s, 0..n);
}

// ANCHOR: qz
/// The generalised complex Schur form of the pencil `(A, B)` (SciPy's `qz(a, b, output='complex')`):
/// unitary `Q`, `Z` with `Qᴴ A Z = S`, `Qᴴ B Z = T` upper triangular. The generalised eigenvalues
/// are `S_ii / T_ii` (infinite where `T_ii = 0`). `B` is first made triangular by a QR
/// factorisation and `A` Hessenberg by Givens rotations that keep `B` triangular; then QZ sweeps
/// with Wilkinson-type shifts chase a bulge down the pencil until the subdiagonal of `S`
/// vanishes. A zero on the diagonal of `T` (an infinite eigenvalue) is chased to the bottom and
/// deflated.
pub fn qz<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix2>) -> Result<QzFactors> {
    let n = a.nrows();
    if a.ncols() != n || b.dim() != (n, n) {
        return Err(LinalgError::Shape(format!("A ({:?}) and B ({:?}) must be square and of one size", a.dim(), b.dim())));
    }
    let ac = a.mapv(|v| v.to_complex());
    let bc = b.mapv(|v| v.to_complex());
    if ac.iter().chain(bc.iter()).any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        return Err(LinalgError::NonFinite);
    }
    // B = Q₀ R₀; S = Q₀ᴴ A.
    let f0 = qr(&bc, QrMode::Full, false)?;
    let q0h = f0.q.t().mapv(|v| v.conj());
    let mut f = QzFactors { s: q0h.dot(&ac), t: f0.r, q: f0.q, z: Array2::from_shape_fn((n, n), |(i, j)| if i == j { C::new(1.0, 0.0) } else { cz() }) };
    // Hessenberg-triangular reduction.
    for j in 0..n.saturating_sub(2) {
        for i in (j + 2..n).rev() {
            let (c, s) = rotation(f.s[(i - 1, j)], f.s[(i, j)]);
            left_rot(&mut f, i - 1, i, c, s, 0);
            f.s[(i, j)] = cz();
            // T[i, i−1] filled in; zero it with columns (i, i − 1).
            let (c, s) = rotation(f.t[(i, i)], f.t[(i, i - 1)]);
            right_rot(&mut f, i, i - 1, c, s, n, i + 1);
            f.t[(i, i - 1)] = cz();
        }
    }
    let norm = |m: &Array2<C>| m.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    let (snorm, tnorm) = (norm(&f.s).max(f64::MIN_POSITIVE), norm(&f.t).max(f64::MIN_POSITIVE));
    let eps = f64::EPSILON;
    let mut hi = n;
    let mut its = 0;
    let mut total = 0;
    while hi > 0 {
        let ihi = hi - 1;
        // The start of the active block: the last negligible subdiagonal entry.
        let mut l = ihi;
        while l > 0 {
            let sub = f.s[(l, l - 1)].norm();
            if sub <= eps * (f.s[(l, l)].norm() + f.s[(l - 1, l - 1)].norm()).max(eps * snorm) {
                f.s[(l, l - 1)] = cz();
                break;
            }
            l -= 1;
        }
        if l == ihi {
            hi -= 1;
            its = 0;
            continue;
        }
        total += 1;
        if total > 30 * n.max(1) * 10 {
            return Err(LinalgError::Shape("qz: the QZ iteration did not converge".into()));
        }
        // An infinite eigenvalue: chase the zero on T's diagonal to the bottom and deflate.
        if let Some(j) = (l..=ihi).find(|&j| f.t[(j, j)].norm() <= eps * tnorm) {
            f.t[(j, j)] = cz();
            for k in j..ihi {
                let (c, s) = rotation(f.t[(k, k + 1)], f.t[(k + 1, k + 1)]);
                left_rot(&mut f, k, k + 1, c, s, k.saturating_sub(1));
                f.t[(k + 1, k + 1)] = cz();
                if k > l {
                    let (c, s) = rotation(f.s[(k + 1, k)], f.s[(k + 1, k - 1)]);
                    right_rot(&mut f, k, k - 1, c, s, (k + 2).min(n), k + 1);
                    f.s[(k + 1, k - 1)] = cz();
                }
            }
            let (c, s) = rotation(f.s[(ihi, ihi)], f.s[(ihi, ihi - 1)]);
            right_rot(&mut f, ihi, ihi - 1, c, s, ihi + 1, ihi + 1);
            f.s[(ihi, ihi - 1)] = cz();
            continue;
        }
        its += 1;
        // Shift: the eigenvalue of the trailing 2 × 2 pencil nearer S_mm / T_mm.
        let m = ihi;
        let shift = if its % 10 == 0 {
            // An exceptional shift breaks cycles.
            f.s[(m, m - 1)] / f.t[(m - 1, m - 1)] + f.s[(m, m)] / f.t[(m, m)]
        } else {
            let (a11, a12, a21, a22) = (f.s[(m - 1, m - 1)], f.s[(m - 1, m)], f.s[(m, m - 1)], f.s[(m, m)]);
            let (b11, b12, b22) = (f.t[(m - 1, m - 1)], f.t[(m - 1, m)], f.t[(m, m)]);
            // det(A₂ − λ B₂) = 0: (a11 − λb11)(a22 − λb22) − (a12 − λb12) a21 = 0.
            let qa = b11 * b22;
            let qb = -(a11 * b22 + a22 * b11 - b12 * a21);
            let qc = a11 * a22 - a12 * a21;
            let disc = (qb * qb - qa * qc * 4.0).sqrt();
            let (r1, r2) = ((-qb + disc) / (qa * 2.0), (-qb - disc) / (qa * 2.0));
            let target = a22 / b22;
            if (r1 - target).norm() <= (r2 - target).norm() { r1 } else { r2 }
        };
        // The QZ sweep from l to ihi.
        let (c, s) = rotation(f.s[(l, l)] - shift * f.t[(l, l)], f.s[(l + 1, l)]);
        left_rot(&mut f, l, l + 1, c, s, l);
        for k in l..ihi {
            let (c, s) = rotation(f.t[(k + 1, k + 1)], f.t[(k + 1, k)]);
            right_rot(&mut f, k + 1, k, c, s, (k + 3).min(ihi + 1), k + 2);
            f.t[(k + 1, k)] = cz();
            if k + 1 < ihi {
                let (c, s) = rotation(f.s[(k + 1, k)], f.s[(k + 2, k)]);
                left_rot(&mut f, k + 1, k + 2, c, s, k);
                f.s[(k + 2, k)] = cz();
            }
        }
    }
    // Make T's diagonal real and non-negative, as LAPACK does.
    for i in 0..n {
        let b = f.t[(i, i)];
        if b.norm() > 0.0 && (b.im != 0.0 || b.re < 0.0) {
            let phase = b.conj() / b.norm();
            for j in i..n {
                f.t[(i, j)] *= phase;
                f.s[(i, j)] *= phase;
            }
            for r in 0..n {
                f.q[(r, i)] *= phase.conj();
            }
            f.t[(i, i)] = C::new(f.t[(i, i)].norm(), 0.0);
        }
    }
    for i in 0..n {
        for j in 0..i {
            f.s[(i, j)] = cz();
            f.t[(i, j)] = cz();
        }
    }
    Ok(f)
}
// ANCHOR_END: qz

/// The solution of the generalised eigenproblem `A x = λ B x`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GeneralizedEigen {
    /// `λ = α / β`: infinite (`∞ + 0i`) where `β = 0`, NaN where both vanish.
    pub values: Array1<C>,
    pub alpha: Array1<C>,
    /// Real and non-negative.
    pub beta: Array1<f64>,
    /// Right eigenvectors as unit columns (as SciPy normalises them).
    pub vectors: Array2<C>,
}

// ANCHOR: eig_generalized
/// The generalised eigenvalues and right eigenvectors of `(A, B)` (SciPy's `eig(a, b)`), from
/// the QZ form: for each `k`, `(β_k S − α_k T) y = 0` is solved by back substitution with
/// `y_k = 1`, and `x = Z y` is normalised to unit length.
pub fn eig_generalized<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix2>) -> Result<GeneralizedEigen> {
    let f = qz(a, b)?;
    let n = f.s.nrows();
    let alpha = Array1::from_shape_fn(n, |i| f.s[(i, i)]);
    let beta = Array1::from_shape_fn(n, |i| f.t[(i, i)].re);
    let values = Array1::from_shape_fn(n, |i| {
        if beta[i] != 0.0 {
            alpha[i] / beta[i]
        } else if alpha[i] != cz() {
            C::new(f64::INFINITY, 0.0)
        } else {
            C::new(f64::NAN, if alpha.iter().all(|v| v.im == 0.0) { 0.0 } else { f64::NAN })
        }
    });
    let scale = f.s.iter().chain(f.t.iter()).fold(0.0f64, |m, v| m.max(v.norm())).max(f64::MIN_POSITIVE);
    let mut vectors = Array2::from_elem((n, n), cz());
    for k in 0..n {
        let (ak, bk) = (alpha[k], C::new(beta[k], 0.0));
        let mut y = vec![cz(); n];
        y[k] = C::new(1.0, 0.0);
        for i in (0..k).rev() {
            let sum: C = (i + 1..=k).map(|j| (bk * f.s[(i, j)] - ak * f.t[(i, j)]) * y[j]).sum();
            let mut d = bk * f.s[(i, i)] - ak * f.t[(i, i)];
            if d.norm() < f64::EPSILON * scale {
                d = C::new(f64::EPSILON * scale, 0.0);
            }
            y[i] = -sum / d;
        }
        let x: Vec<C> = (0..n).map(|r| (0..=k).map(|j| f.z[(r, j)] * y[j]).sum()).collect();
        let nx = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        for (r, v) in x.iter().enumerate() {
            vectors[(r, k)] = v / nx;
        }
    }
    Ok(GeneralizedEigen { values, alpha, beta, vectors })
}
// ANCHOR_END: eig_generalized

#[cfg(test)]
mod tests {
    use super::*;

    fn pencil(n: usize, seed: f64) -> (Array2<f64>, Array2<f64>) {
        let a = Array2::from_shape_fn((n, n), |(i, j)| (seed + 1.3 * i as f64 + 0.7 * (j * j) as f64 + 0.37 * (i * j) as f64).sin());
        let b = Array2::from_shape_fn((n, n), |(i, j)| (seed + 0.5 * i as f64 - 0.9 * j as f64 + 0.21 * (i * j) as f64).cos() + if i == j { 3.0 } else { 0.0 });
        (a, b)
    }

    fn residual(a: &Array2<C>, b: &Array2<C>, e: &GeneralizedEigen) -> f64 {
        let n = a.nrows();
        let mut worst = 0.0f64;
        for k in 0..n {
            let x = e.vectors.column(k);
            let (al, be) = (e.alpha[k], C::new(e.beta[k], 0.0));
            for i in 0..n {
                let r: C = (0..n).map(|j| (be * a[(i, j)] - al * b[(i, j)]) * x[j]).sum();
                worst = worst.max(r.norm());
            }
        }
        worst
    }

    #[test]
    fn qz_form_and_eigenpairs() {
        for (n, seed) in [(1, 0.0), (2, 1.0), (6, 2.0), (15, 3.0)] {
            let (a, b) = pencil(n, seed);
            let f = qz(&a, &b).unwrap();
            let (ac, bc) = (a.mapv(|v| C::new(v, 0.0)), b.mapv(|v| C::new(v, 0.0)));
            let zh = f.z.t().mapv(|v| v.conj());
            let ra = f.q.dot(&f.s).dot(&zh) - &ac;
            let rb = f.q.dot(&f.t).dot(&zh) - &bc;
            assert!(ra.iter().chain(rb.iter()).all(|v| v.norm() < 1e-12), "n = {n}");
            assert!((0..n).all(|i| (0..i).all(|j| f.s[(i, j)] == cz() && f.t[(i, j)] == cz())));
            assert!((0..n).all(|i| f.t[(i, i)].im == 0.0 && f.t[(i, i)].re >= 0.0));
            let e = eig_generalized(&a, &b).unwrap();
            assert!(residual(&ac, &bc, &e) < 1e-11, "n = {n}");
        }
        // Complex data.
        let (a, b) = pencil(7, 4.0);
        let ac = Array2::from_shape_fn((7, 7), |(i, j)| C::new(a[(i, j)], b[(j, i)] - 3.0 * (i == j) as u8 as f64));
        let bc = Array2::from_shape_fn((7, 7), |(i, j)| C::new(b[(i, j)], 0.3 * a[(j, i)]));
        let e = eig_generalized(&ac, &bc).unwrap();
        assert!(residual(&ac, &bc, &e) < 1e-11);
    }

    #[test]
    fn singular_b_gives_infinite_eigenvalues() {
        // B = diag(1, 1, 0): one infinite eigenvalue; the finite ones are those of A's leading block.
        let a = Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.5, 0.0, 3.0, 1.0, 1.0, 0.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let e = eig_generalized(&a, &b).unwrap();
        assert_eq!(e.values.iter().filter(|v| v.re.is_infinite()).count(), 1, "{:?}", e.values);
        // det(A − λB) = 4λ² − ... : the finite eigenvalues solve det of the Schur complement.
        let finite: Vec<C> = e.values.iter().copied().filter(|v| v.re.is_finite()).collect();
        for l in finite {
            let m = Array2::from_shape_fn((3, 3), |(i, j)| C::new(a[(i, j)] - if i == j && i < 2 { l.re } else { 0.0 }, if i == j && i < 2 { -l.im } else { 0.0 }));
            let det = m[(0, 0)] * (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)]) - m[(0, 1)] * (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)]) + m[(0, 2)] * (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)]);
            assert!(det.norm() < 1e-10, "{l}: {det}");
        }
    }

    #[test]
    fn symmetric_definite_pencils() {
        let n = 6;
        let (a, b) = pencil(n, 5.0);
        let sa = Array2::from_shape_fn((n, n), |(i, j)| a[(i, j)] + a[(j, i)]);
        let sb = b.t().dot(&b);
        for uplo in [UPLO::Lower, UPLO::Upper] {
            let (w, x) = eigh_generalized(&sa, &sb, uplo).unwrap();
            assert!(w.windows(2).into_iter().all(|p| p[0] <= p[1]));
            let xbx = x.t().dot(&sb).dot(&x);
            let ax = sa.dot(&x);
            let bx = sb.dot(&x);
            for k in 0..n {
                assert!((0..n).all(|i| (ax[(i, k)] - w[k] * bx[(i, k)]).abs() < 1e-10));
                assert!((0..n).all(|i| (xbx[(i, k)] - if i == k { 1.0 } else { 0.0 }).abs() < 1e-10));
            }
        }
        assert!(eigh_generalized(&sa, &(-&sb), UPLO::Lower).is_err(), "B must be positive definite");
    }
}
