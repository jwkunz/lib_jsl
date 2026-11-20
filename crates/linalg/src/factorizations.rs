//! Further matrix decompositions with SciPy's conventions: QR with optional column pivoting,
//! RQ, the Hessenberg and real Schur forms, the Bunch-Kaufman `LDLᵀ` (or `LDLᴴ`) factorisation,
//! the polar decomposition, principal angles between subspaces, and general matrix functions by
//! the Schur-Parlett method. The Householder reflectors follow LAPACK's `larfg` sign convention,
//! so the factors agree with SciPy's entry by entry, not merely up to signs.
//!
//! See the book entry [Further factorisations](https://jwkunz.github.io/lib_jsl/book/linear_algebra/factorizations.html).

use crate::dense::Mat;
use crate::error::{LinalgError, Result};
use crate::scalar::Scalar;
use crate::{SVD, orth, schur_complex};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2, s};
use num_complex::Complex;

type C = Complex<f64>;

fn zeros<T: Scalar>(rows: usize, cols: usize) -> Array2<T> {
    Array2::from_elem((rows, cols), T::zero())
}

fn eye<T: Scalar>(n: usize) -> Array2<T> {
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { T::one() } else { T::zero() })
}

fn matmul<T: Scalar>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
    Array2::from_shape_fn((a.nrows(), b.ncols()), |(i, j)| (0..a.ncols()).fold(T::zero(), |acc, k| acc + a[(i, k)] * b[(k, j)]))
}

fn adjoint<T: Scalar>(a: &Array2<T>) -> Array2<T> {
    Array2::from_shape_fn((a.ncols(), a.nrows()), |(i, j)| a[(j, i)].conj())
}

fn check_finite<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<Array2<T>> {
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite);
    }
    Ok(a.to_owned())
}

// ANCHOR: larfg
/// LAPACK's `larfg`: the reflector `H = I − τ v vᴴ`, `v = (1, x̂)`, with `Hᴴ (α, x) = (β, 0)` and
/// `β = −sign(Re α) ‖(α, x)‖` real. Overwrites `x` with `x̂` and returns `(β, τ)`; `τ = 0` (no
/// reflection) when `x = 0` and `α` is real.
fn larfg<T: Scalar>(alpha: T, x: &mut [T]) -> (T, T) {
    let xnorm = x.iter().map(|v| v.abs_sqr()).sum::<f64>().sqrt();
    if xnorm == 0.0 && alpha.im() == 0.0 {
        return (alpha, T::zero());
    }
    let norm = (alpha.abs_sqr() + xnorm * xnorm).sqrt();
    let beta = T::from_real(if alpha.re() >= 0.0 { -norm } else { norm });
    let tau = (beta - alpha) / beta;
    let scale = T::one() / (alpha - beta);
    x.iter_mut().for_each(|v| *v *= scale);
    (beta, tau)
}
// ANCHOR_END: larfg

/// `A[rows, cols] ← (I − τ v vᴴ) A[rows, cols]`, with `v` laid along `rows`.
fn reflect_left<T: Scalar>(a: &mut Array2<T>, v: &[T], tau: T, row0: usize, cols: std::ops::Range<usize>) {
    if tau == T::zero() {
        return;
    }
    for j in cols {
        let w = v.iter().enumerate().fold(T::zero(), |acc, (i, &vi)| acc + vi.conj() * a[(row0 + i, j)]);
        let tw = tau * w;
        for (i, &vi) in v.iter().enumerate() {
            a[(row0 + i, j)] -= vi * tw;
        }
    }
}

/// `A[rows, cols] ← A[rows, cols] (I − τ v vᴴ)`, with `v` laid along `cols`.
fn reflect_right<T: Scalar>(a: &mut Array2<T>, v: &[T], tau: T, rows: std::ops::Range<usize>, col0: usize) {
    if tau == T::zero() {
        return;
    }
    for i in rows {
        let w = v.iter().enumerate().fold(T::zero(), |acc, (j, &vj)| acc + a[(i, col0 + j)] * vj);
        let tw = tau * w;
        for (j, &vj) in v.iter().enumerate() {
            a[(i, col0 + j)] -= tw * vj.conj();
        }
    }
}

/// How much of a QR (or RQ) factorisation to return (SciPy's `mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum QrMode {
    /// `Q` square (`m × m` for QR, `n × n` for RQ).
    Full,
    /// `Q` with `min(m, n)` columns (QR) or rows (RQ), and `R` to match.
    Economic,
}

/// A QR factorisation `A P = Q R` (`P` the identity without pivoting).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QrFactors<T> {
    pub q: Array2<T>,
    pub r: Array2<T>,
    /// The column permutation: column `j` of `A P` is column `permutation[j]` of `A`.
    pub permutation: Vec<usize>,
}

/// The Householder QR of `a`, optionally with column pivoting, as LAPACK's `geqr2` and `laqp2`:
/// returns the overwritten matrix (`R` above the diagonal, reflectors below), the `τ`s and the
/// permutation.
fn householder_qr<T: Scalar>(mut a: Array2<T>, pivoting: bool) -> (Array2<T>, Vec<T>, Vec<usize>) {
    let (m, n) = a.dim();
    let k = m.min(n);
    let mut jpvt: Vec<usize> = (0..n).collect();
    let col_norm = |a: &Array2<T>, j: usize, from: usize| (from..m).map(|i| a[(i, j)].abs_sqr()).sum::<f64>().sqrt();
    let mut vn1: Vec<f64> = (0..n).map(|j| col_norm(&a, j, 0)).collect();
    let mut vn2 = vn1.clone();
    let tol3z = f64::EPSILON.sqrt();
    let mut taus = Vec::with_capacity(k);
    for i in 0..k {
        if pivoting {
            // The first column of largest remaining norm (LAPACK's idamax).
            let mut pvt = i;
            for j in i + 1..n {
                if vn1[j] > vn1[pvt] {
                    pvt = j;
                }
            }
            if pvt != i {
                for r in 0..m {
                    a.swap((r, pvt), (r, i));
                }
                jpvt.swap(pvt, i);
                vn1[pvt] = vn1[i];
                vn2[pvt] = vn2[i];
            }
        }
        let alpha = a[(i, i)];
        let mut x: Vec<T> = (i + 1..m).map(|r| a[(r, i)]).collect();
        let (beta, tau) = larfg(alpha, &mut x);
        let mut v = vec![T::one()];
        v.extend_from_slice(&x);
        a[(i, i)] = beta;
        for (r, &xv) in x.iter().enumerate() {
            a[(i + 1 + r, i)] = xv;
        }
        // Apply Hᴴ from the left to the trailing columns.
        reflect_left(&mut a, &v, tau.conj(), i, i + 1..n);
        taus.push(tau);
        if pivoting {
            // Downdate the partial column norms, recomputing them when cancellation threatens.
            for j in i + 1..n {
                if vn1[j] != 0.0 {
                    let t = (1.0 - (a[(i, j)].abs() / vn1[j]).powi(2)).max(0.0);
                    let t2 = t * (vn1[j] / vn2[j]).powi(2);
                    if t2 <= tol3z {
                        if i + 1 < m {
                            vn1[j] = col_norm(&a, j, i + 1);
                            vn2[j] = vn1[j];
                        } else {
                            vn1[j] = 0.0;
                            vn2[j] = 0.0;
                        }
                    } else {
                        vn1[j] *= t.sqrt();
                    }
                }
            }
        }
    }
    (a, taus, jpvt)
}

// ANCHOR: qr
/// The QR factorisation `A P = Q R` (SciPy's `qr`, with `pivoting` its `pivoting=True`): `Q` has
/// orthonormal columns, `R` is upper triangular (trapezoidal), and with pivoting the columns are
/// chosen in order of largest remaining norm, so that `|R₁₁| ≥ |R₂₂| ≥ …`, which reveals the rank.
pub fn qr<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, mode: QrMode, pivoting: bool) -> Result<QrFactors<T>> {
    let a = check_finite(a)?;
    let (m, n) = a.dim();
    let k = m.min(n);
    let (f, taus, permutation) = householder_qr(a, pivoting);
    let qcols = if mode == QrMode::Full { m } else { k };
    // Q = H₁ H₂ ⋯ H_k applied to the first columns of the identity (LAPACK's orgqr).
    let mut q = zeros::<T>(m, qcols);
    for j in 0..qcols.min(m) {
        q[(j, j)] = T::one();
    }
    for i in (0..k).rev() {
        let mut v = vec![T::one()];
        v.extend((i + 1..m).map(|r| f[(r, i)]));
        reflect_left(&mut q, &v, taus[i], i, 0..qcols);
    }
    let rrows = if mode == QrMode::Full { m } else { k };
    let r = Array2::from_shape_fn((rrows, n), |(i, j)| if j >= i { f[(i, j)] } else { T::zero() });
    Ok(QrFactors { q, r, permutation })
}
// ANCHOR_END: qr

// ANCHOR: rq
/// The RQ factorisation `A = R Q` (SciPy's `rq`, LAPACK's `gerq2`): `R` upper triangular
/// (trapezoidal, aligned to the right), `Q` with orthonormal rows. Each step reflects a row,
/// from the last upwards, onto its own last allowed column. Returns `(R, Q)`.
pub fn rq<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, mode: QrMode) -> Result<(Array2<T>, Array2<T>)> {
    let mut a = check_finite(a)?;
    let (m, n) = a.dim();
    let k = m.min(n);
    let mut reflectors: Vec<(Vec<T>, T)> = Vec::with_capacity(k);
    for i in (0..k).rev() {
        let (r, c) = (m - k + i, n - k + i);
        // The conjugated row segment is reflected onto its last entry.
        let alpha = a[(r, c)].conj();
        let mut x: Vec<T> = (0..c).map(|j| a[(r, j)].conj()).collect();
        let (beta, tau) = larfg(alpha, &mut x);
        let mut v = x.clone();
        v.push(T::one());
        // A[0..r, 0..=c] ← A[0..r, 0..=c] H.
        reflect_right(&mut a, &v, tau, 0..r, 0);
        a[(r, c)] = beta;
        for j in 0..c {
            a[(r, j)] = T::zero();
        }
        reflectors.push((v, tau));
    }
    reflectors.reverse(); // reflectors[i] is H(i), acting on columns 0..=n−k+i
    // Q = H(1)ᴴ H(2)ᴴ ⋯ H(k)ᴴ as an n × n matrix (LAPACK's orgrq).
    let mut q = eye::<T>(n);
    for (v, tau) in &reflectors {
        reflect_right(&mut q, v, tau.conj(), 0..n, 0);
    }
    // R occupies the last min(m, n) "diagonals": R[i, j] = 0 for j < i + n − m.
    let r_full = Array2::from_shape_fn((m, n), |(i, j)| if j as isize >= i as isize + n as isize - m as isize { a[(i, j)] } else { T::zero() });
    if mode == QrMode::Economic && m <= n {
        return Ok((r_full.slice(s![.., n - m..]).to_owned(), q.slice(s![n - m.., ..]).to_owned()));
    }
    Ok((r_full, q))
}
// ANCHOR_END: rq

// ANCHOR: hessenberg
/// The Hessenberg form `A = Q H Qᴴ` (SciPy's `hessenberg` with `calc_q=True`; LAPACK's `gehd2`
/// and `orghr`): `H` is zero below its first subdiagonal, `Q` unitary. Reflector `i` zeros
/// column `i` below row `i + 1` and is applied from both sides. Returns `(H, Q)`.
pub fn hessenberg<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<(Array2<T>, Array2<T>)> {
    let mut a = check_finite(a)?;
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::NotSquare { rows: n, cols: a.ncols() });
    }
    let mut reflectors = Vec::new();
    // As LAPACK, the last step has no entries to zero; for a complex matrix it still reflects,
    // to make the last subdiagonal entry real.
    for i in 0..n.saturating_sub(1) {
        let alpha = a[(i + 1, i)];
        let mut x: Vec<T> = (i + 2..n).map(|r| a[(r, i)]).collect();
        let (beta, tau) = larfg(alpha, &mut x);
        let mut v = vec![T::one()];
        v.extend_from_slice(&x);
        a[(i + 1, i)] = beta;
        for r in i + 2..n {
            a[(r, i)] = T::zero();
        }
        reflect_right(&mut a, &v, tau, 0..n, i + 1);
        reflect_left(&mut a, &v, tau.conj(), i + 1, i + 1..n);
        reflectors.push((i + 1, v, tau));
    }
    let mut q = eye::<T>(n);
    for (row0, v, tau) in reflectors.iter().rev() {
        reflect_left(&mut q, v, *tau, *row0, *row0..n);
    }
    Ok((a, q))
}
// ANCHOR_END: hessenberg

// ANCHOR: lanv2
/// LAPACK's `lanv2`: the rotation `[c −s; s c]` that brings a real 2 × 2 block to standard form,
/// `[a b; c d] = R [a' b'; c' d'] Rᵀ`, where the new block is upper triangular when its eigenvalues
/// are real, and otherwise has `a' = d'` and `b' c' < 0`. Returns `(a', b', c', d', cs, sn)`.
fn lanv2(mut a: f64, mut b: f64, mut c: f64, mut d: f64) -> (f64, f64, f64, f64, f64, f64) {
    let eps = f64::EPSILON;
    let sign = |x: f64, y: f64| if y >= 0.0 { x.abs() } else { -x.abs() };
    let (mut cs, mut sn);
    if c == 0.0 {
        cs = 1.0;
        sn = 0.0;
    } else if b == 0.0 {
        cs = 0.0;
        sn = 1.0;
        std::mem::swap(&mut a, &mut d);
        b = -c;
        c = 0.0;
    } else if a - d == 0.0 && b.signum() != c.signum() {
        cs = 1.0;
        sn = 0.0;
    } else {
        let temp = a - d;
        let mut p = 0.5 * temp;
        let bcmax = b.abs().max(c.abs());
        let bcmis = b.abs().min(c.abs()) * b.signum() * c.signum();
        let scale = p.abs().max(bcmax);
        let mut z = p / scale * p + bcmax / scale * bcmis;
        if z >= 4.0 * eps {
            // Real eigenvalues: make the block upper triangular.
            z = p + sign(scale.sqrt() * z.sqrt(), p);
            a = d + z;
            d -= bcmax / z * bcmis;
            let tau = c.hypot(z);
            cs = z / tau;
            sn = c / tau;
            b -= c;
            c = 0.0;
        } else {
            // Complex or nearly equal real eigenvalues: equalise the diagonal.
            let sigma = b + c;
            let tau = sigma.hypot(temp);
            cs = (0.5 * (1.0 + sigma.abs() / tau)).sqrt();
            sn = -(p / (tau * cs)) * sign(1.0, sigma);
            let (aa, bb, cc, dd) = (a * cs + b * sn, -a * sn + b * cs, c * cs + d * sn, -c * sn + d * cs);
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;
            let temp = 0.5 * (a + d);
            a = temp;
            d = temp;
            if c != 0.0 {
                if b != 0.0 {
                    if b.signum() == c.signum() {
                        // Real eigenvalues after all: triangularise.
                        let sab = b.abs().sqrt();
                        let sac = c.abs().sqrt();
                        p = sign(sab * sac, c);
                        let tau = 1.0 / (b + c).abs().sqrt();
                        a = temp + p;
                        d = temp - p;
                        b -= c;
                        c = 0.0;
                        let (cs1, sn1) = (sab * tau, sac * tau);
                        let t = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = t;
                    }
                } else {
                    b = -c;
                    c = 0.0;
                    let t = cs;
                    cs = -sn;
                    sn = t;
                }
            }
        }
    }
    (a, b, c, d, cs, sn)
}
// ANCHOR_END: lanv2

/// [`lanv2`] for the reordering module.
pub(crate) fn lanv2_pub(a: f64, b: f64, c: f64, d: f64) -> (f64, f64, f64, f64, f64, f64) {
    lanv2(a, b, c, d)
}

// ANCHOR: schur_real
/// The real Schur form `A = Z T Zᵀ` of a real matrix (SciPy's `schur(a, output='real')`): `Z`
/// orthogonal and `T` quasi upper triangular, with 1 × 1 blocks for real eigenvalues and 2 × 2
/// blocks `[a b; c a]`, `b c < 0`, for each complex conjugate pair `a ± i√(−bc)`. Returns `(T, Z)`.
pub fn schur_real<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)> {
    let (mut t, mut z, pair) = crate::eig::real_schur_unbalanced(&Mat::from_array(a)?)?;
    let n = t.rows;
    let rot = |x: f64, y: f64, cs: f64, sn: f64| (cs * x + sn * y, cs * y - sn * x);
    for k in (0..n).filter(|&k| pair[k]) {
        let (a0, b0, c0, d0, cs, sn) = lanv2(t.at(k, k), t.at(k, k + 1), t.at(k + 1, k), t.at(k + 1, k + 1));
        *t.at_mut(k, k) = a0;
        *t.at_mut(k, k + 1) = b0;
        *t.at_mut(k + 1, k) = c0;
        *t.at_mut(k + 1, k + 1) = d0;
        for j in k + 2..n {
            let (x, y) = rot(t.at(k, j), t.at(k + 1, j), cs, sn);
            *t.at_mut(k, j) = x;
            *t.at_mut(k + 1, j) = y;
        }
        for i in 0..k {
            let (x, y) = rot(t.at(i, k), t.at(i, k + 1), cs, sn);
            *t.at_mut(i, k) = x;
            *t.at_mut(i, k + 1) = y;
        }
        for i in 0..n {
            let (x, y) = rot(z.at(i, k), z.at(i, k + 1), cs, sn);
            *z.at_mut(i, k) = x;
            *z.at_mut(i, k + 1) = y;
        }
    }
    Ok((t.into_array(), z.into_array()))
}
// ANCHOR_END: schur_real

/// An `LDLᵀ` (or `LDLᴴ`) factorisation `A = L D Lᴴ` with symmetric pivoting.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LdlFactors<T> {
    /// The triangular factor, with its rows permuted: `L[perm, :]` is unit triangular.
    pub l: Array2<T>,
    /// Block diagonal, with 1 × 1 and 2 × 2 blocks.
    pub d: Array2<T>,
    /// The row order that makes `l` triangular.
    pub perm: Vec<usize>,
}

/// `|x|` as LAPACK compares complex entries when choosing pivots, `|Re x| + |Im x|`.
fn cabs1<T: Scalar>(x: T) -> f64 {
    if T::IS_COMPLEX { x.re().abs() + x.im().abs() } else { x.abs() }
}

// ANCHOR: ldl
/// The Bunch-Kaufman factorisation of a symmetric or Hermitian matrix (SciPy's `ldl`, LAPACK's
/// `sytf2`/`hetf2`): `A = L D Lᴴ` (or `U D Uᴴ` with `lower = false`), `D` block diagonal with 1 × 1
/// and 2 × 2 blocks, and `L` a row permutation of a unit triangular matrix. A 2 × 2 pivot is
/// taken when no diagonal entry is large enough relative to its column, `α = (1 + √17)/8`,
/// which bounds the growth of the entries for indefinite matrices. Only the `lower` (or upper)
/// triangle is read; `hermitian = false` factorises a complex symmetric matrix, `A = L D Lᵀ`.
pub fn ldl<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, lower: bool, hermitian: bool) -> Result<LdlFactors<T>> {
    let mut a = check_finite(a)?;
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::NotSquare { rows: n, cols: a.ncols() });
    }
    let herm = hermitian && T::IS_COMPLEX;
    let cj = |x: T| if herm { x.conj() } else { x };
    let re = |x: T| if herm { T::from_real(x.re()) } else { x };
    let dabs = |x: T| if herm { x.re().abs() } else { cabs1(x) };
    let alpha = (1.0 + 17f64.sqrt()) / 8.0;
    // Work on the lower triangle; the upper case reads the transpose (conjugated if Hermitian),
    // which is the lower triangle of the same matrix, and mirrors the indices.
    let mut ipiv = vec![0isize; n];
    if lower {
        let mut k = 0;
        while k < n {
            let mut kstep = 1;
            let absakk = dabs(a[(k, k)]);
            let (imax, colmax) = if k + 1 < n {
                let mut im = k + 1;
                for i in k + 2..n {
                    if cabs1(a[(i, k)]) > cabs1(a[(im, k)]) {
                        im = i;
                    }
                }
                (im, cabs1(a[(im, k)]))
            } else {
                (k, 0.0)
            };
            let kp;
            if absakk.max(colmax) == 0.0 || absakk >= alpha * colmax {
                kp = k;
            } else {
                let mut rowmax = 0.0f64;
                for j in k..imax {
                    rowmax = rowmax.max(cabs1(a[(imax, j)]));
                }
                for j in imax + 1..n {
                    rowmax = rowmax.max(cabs1(a[(j, imax)]));
                }
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kp = k;
                } else if dabs(a[(imax, imax)]) >= alpha * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + kstep - 1;
            if kp != kk {
                for i in kp + 1..n {
                    a.swap((i, kk), (i, kp));
                }
                for j in kk + 1..kp {
                    let t = cj(a[(j, kk)]);
                    a[(j, kk)] = cj(a[(kp, j)]);
                    a[(kp, j)] = t;
                }
                if herm {
                    a[(kp, kk)] = a[(kp, kk)].conj();
                }
                let t = re(a[(kk, kk)]);
                a[(kk, kk)] = re(a[(kp, kp)]);
                a[(kp, kp)] = t;
                if kstep == 2 {
                    if herm {
                        a[(k, k)] = re(a[(k, k)]);
                    }
                    a.swap((k + 1, k), (kp, k));
                }
            } else if herm {
                a[(k, k)] = re(a[(k, k)]);
                if kstep == 2 {
                    a[(k + 1, k + 1)] = re(a[(k + 1, k + 1)]);
                }
            }
            if kstep == 1 {
                if k + 1 < n {
                    let d11 = T::one() / re(a[(k, k)]);
                    for j in k + 1..n {
                        let ajk = a[(j, k)];
                        for i in j..n {
                            let aik = a[(i, k)];
                            a[(i, j)] -= d11 * aik * cj(ajk);
                        }
                        if herm {
                            a[(j, j)] = re(a[(j, j)]);
                        }
                    }
                    for i in k + 1..n {
                        a[(i, k)] *= d11;
                    }
                }
                ipiv[k] = kp as isize + 1;
            } else {
                if k + 2 < n {
                    if herm {
                        let d = a[(k + 1, k)].abs();
                        let d11 = a[(k + 1, k + 1)].re() / d;
                        let d22 = a[(k, k)].re() / d;
                        let tt = 1.0 / (d11 * d22 - 1.0);
                        let d21 = a[(k + 1, k)].scale(1.0 / d);
                        let dd = tt / d;
                        for j in k + 2..n {
                            let wk = (a[(j, k)].scale(d11) - d21 * a[(j, k + 1)]).scale(dd);
                            let wkp1 = (a[(j, k + 1)].scale(d22) - d21.conj() * a[(j, k)]).scale(dd);
                            for i in j..n {
                                let (aik, aik1) = (a[(i, k)], a[(i, k + 1)]);
                                a[(i, j)] -= aik * wk.conj() + aik1 * wkp1.conj();
                            }
                            a[(j, k)] = wk;
                            a[(j, k + 1)] = wkp1;
                            a[(j, j)] = re(a[(j, j)]);
                        }
                    } else {
                        let mut d21 = a[(k + 1, k)];
                        let d11 = a[(k + 1, k + 1)] / d21;
                        let d22 = a[(k, k)] / d21;
                        let t = T::one() / (d11 * d22 - T::one());
                        d21 = t / d21;
                        for j in k + 2..n {
                            let wk = d21 * (d11 * a[(j, k)] - a[(j, k + 1)]);
                            let wkp1 = d21 * (d22 * a[(j, k + 1)] - a[(j, k)]);
                            for i in j..n {
                                let (aik, aik1) = (a[(i, k)], a[(i, k + 1)]);
                                a[(i, j)] -= aik * wk + aik1 * wkp1;
                            }
                            a[(j, k)] = wk;
                            a[(j, k + 1)] = wkp1;
                        }
                    }
                }
                ipiv[k] = -(kp as isize + 1);
                ipiv[k + 1] = -(kp as isize + 1);
            }
            k += kstep;
        }
    } else {
        let mut k = n;
        while k > 0 {
            let kc = k - 1; // the current column, 0-based
            let mut kstep = 1;
            let absakk = dabs(a[(kc, kc)]);
            let (imax, colmax) = if kc > 0 {
                let mut im = 0;
                for i in 1..kc {
                    if cabs1(a[(i, kc)]) > cabs1(a[(im, kc)]) {
                        im = i;
                    }
                }
                (im, cabs1(a[(im, kc)]))
            } else {
                (kc, 0.0)
            };
            let kp;
            if absakk.max(colmax) == 0.0 || absakk >= alpha * colmax {
                kp = kc;
            } else {
                let mut rowmax = 0.0f64;
                for j in imax + 1..=kc {
                    rowmax = rowmax.max(cabs1(a[(imax, j)]));
                }
                for i in 0..imax {
                    rowmax = rowmax.max(cabs1(a[(i, imax)]));
                }
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kp = kc;
                } else if dabs(a[(imax, imax)]) >= alpha * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = kc + 1 - kstep;
            if kp != kk {
                for i in 0..kp {
                    a.swap((i, kk), (i, kp));
                }
                for j in kp + 1..kk {
                    let t = cj(a[(j, kk)]);
                    a[(j, kk)] = cj(a[(kp, j)]);
                    a[(kp, j)] = t;
                }
                if herm {
                    a[(kp, kk)] = a[(kp, kk)].conj();
                }
                let t = re(a[(kk, kk)]);
                a[(kk, kk)] = re(a[(kp, kp)]);
                a[(kp, kp)] = t;
                if kstep == 2 {
                    if herm {
                        a[(kc, kc)] = re(a[(kc, kc)]);
                    }
                    a.swap((kc - 1, kc), (kp, kc));
                }
            } else if herm {
                a[(kc, kc)] = re(a[(kc, kc)]);
                if kstep == 2 {
                    a[(kc - 1, kc - 1)] = re(a[(kc - 1, kc - 1)]);
                }
            }
            if kstep == 1 {
                let r1 = T::one() / re(a[(kc, kc)]);
                for j in (0..kc).rev() {
                    let ajk = a[(j, kc)];
                    for i in 0..=j {
                        let aik = a[(i, kc)];
                        a[(i, j)] -= r1 * aik * cj(ajk);
                    }
                    if herm {
                        a[(j, j)] = re(a[(j, j)]);
                    }
                }
                for i in 0..kc {
                    a[(i, kc)] *= r1;
                }
                ipiv[kc] = kp as isize + 1;
            } else {
                if kc > 1 {
                    if herm {
                        let d = a[(kc - 1, kc)].abs();
                        let d22 = a[(kc - 1, kc - 1)].re() / d;
                        let d11 = a[(kc, kc)].re() / d;
                        let tt = 1.0 / (d11 * d22 - 1.0);
                        let d12 = a[(kc - 1, kc)].scale(1.0 / d);
                        let dd = tt / d;
                        for j in (0..kc - 1).rev() {
                            let wkm1 = (a[(j, kc - 1)].scale(d11) - d12.conj() * a[(j, kc)]).scale(dd);
                            let wk = (a[(j, kc)].scale(d22) - d12 * a[(j, kc - 1)]).scale(dd);
                            for i in (0..=j).rev() {
                                let (aik, aikm1) = (a[(i, kc)], a[(i, kc - 1)]);
                                a[(i, j)] -= aik * wk.conj() + aikm1 * wkm1.conj();
                            }
                            a[(j, kc)] = wk;
                            a[(j, kc - 1)] = wkm1;
                            a[(j, j)] = re(a[(j, j)]);
                        }
                    } else {
                        let mut d12 = a[(kc - 1, kc)];
                        let d22 = a[(kc - 1, kc - 1)] / d12;
                        let d11 = a[(kc, kc)] / d12;
                        let t = T::one() / (d11 * d22 - T::one());
                        d12 = t / d12;
                        for j in (0..kc - 1).rev() {
                            let wkm1 = d12 * (d11 * a[(j, kc - 1)] - a[(j, kc)]);
                            let wk = d12 * (d22 * a[(j, kc)] - a[(j, kc - 1)]);
                            for i in (0..=j).rev() {
                                let (aik, aikm1) = (a[(i, kc)], a[(i, kc - 1)]);
                                a[(i, j)] -= aik * wk + aikm1 * wkm1;
                            }
                            a[(j, kc)] = wk;
                            a[(j, kc - 1)] = wkm1;
                        }
                    }
                }
                ipiv[kc] = -(kp as isize + 1);
                ipiv[kc - 1] = -(kp as isize + 1);
            }
            k -= kstep;
        }
    }
    Ok(ldl_assemble(a, &ipiv, lower, herm))
}
// ANCHOR_END: ldl

/// SciPy's post-processing of `sytrf`/`hetrf` (`_ldl_sanitize_ipiv`, `_ldl_get_d_and_l`,
/// `_ldl_construct_tri_factor`): the swap list and block sizes from the pivot indices, `D` and the
/// unit triangle from the packed factor, and the row interchanges applied to the triangle.
fn ldl_assemble<T: Scalar>(ldu: Array2<T>, ipiv: &[isize], lower: bool, herm: bool) -> LdlFactors<T> {
    let n = ldu.nrows();
    let mut swap: Vec<usize> = (0..n).collect();
    let mut pivots = vec![0usize; n];
    let indices: Vec<usize> = if lower { (0..n).collect() } else { (0..n).rev().collect() };
    let (x, y): (isize, isize) = if lower { (1, 0) } else { (-1, -1) };
    let mut skip = false;
    for &ind in &indices {
        if skip {
            skip = false;
            continue;
        }
        let cur = ipiv[ind];
        if cur > 0 {
            if cur as usize != ind + 1 {
                swap[ind] = swap[cur as usize - 1];
            }
            pivots[ind] = 1;
        } else {
            let other = (ind as isize + x) as usize;
            // As SciPy: record the interchange unless −ipiv = ind + 2 (1-based ind + 1).
            if (-cur) as usize != ind + 2 {
                swap[other] = swap[(-cur) as usize - 1];
            }
            pivots[(ind as isize + y) as usize] = 2;
            skip = true;
        }
    }
    let mut d = zeros::<T>(n, n);
    for i in 0..n {
        d[(i, i)] = ldu[(i, i)];
    }
    let mut lu = Array2::from_shape_fn((n, n), |(i, j)| if (lower && i > j) || (!lower && i < j) { ldu[(i, j)] } else if i == j { T::one() } else { T::zero() });
    let (bx, by) = if lower { (1usize, 0usize) } else { (0, 1) };
    let mut blk = 0;
    for &p in pivots.iter().filter(|&&p| p != 0) {
        if p == 2 {
            let v = ldu[(blk + bx, blk + by)];
            d[(blk + bx, blk + by)] = v;
            d[(blk + by, blk + bx)] = if herm { v.conj() } else { v };
            lu[(blk + bx, blk + by)] = T::zero();
        }
        blk += p;
    }
    let mut perm: Vec<usize> = (0..n).collect();
    let order: Vec<usize> = if lower { (0..n).rev().collect() } else { (0..n).collect() };
    for ind in order {
        let s_ind = swap[ind];
        if s_ind != ind {
            let (mut col_s, mut col_e) = if lower { (ind, n) } else { (0, ind + 1) };
            if pivots[ind] == if lower { 0 } else { 2 } {
                if lower {
                    col_s -= 1;
                } else {
                    col_e += 1;
                }
            }
            for c in col_s..col_e {
                let t = lu[(s_ind, c)];
                lu[(s_ind, c)] = lu[(ind, c)];
                lu[(ind, c)] = t;
            }
            perm.swap(s_ind, ind);
        }
    }
    // perm ← argsort(perm).
    let mut inverse = vec![0; n];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    LdlFactors { l: lu, d, perm: inverse }
}

// ANCHOR: polar
/// The polar decomposition `A = U P` (`side = right`) or `A = P U` (left) (SciPy's `polar`): `U`
/// with orthonormal columns (or rows) and `P` Hermitian positive semi-definite, from the SVD
/// `A = W Σ Vᴴ`: `U = W Vᴴ`, and `P = V Σ Vᴴ` or `W Σ Wᴴ`. Returns `(U, P)`.
pub fn polar<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, right: bool) -> Result<(Array2<T>, Array2<T>)> {
    let (m, n) = a.dim();
    let k = m.min(n);
    let (w, sv, vh) = a.svd(true, true)?;
    let (w, vh) = (w.expect("requested"), vh.expect("requested"));
    let w = w.slice(s![.., ..k]).to_owned();
    let vh = vh.slice(s![..k, ..]).to_owned();
    let u = matmul(&w, &vh);
    let p = if right {
        Array2::from_shape_fn((n, n), |(i, j)| (0..k).fold(T::zero(), |acc, l| acc + vh[(l, i)].conj().scale(sv[l]) * vh[(l, j)]))
    } else {
        Array2::from_shape_fn((m, m), |(i, j)| (0..k).fold(T::zero(), |acc, l| acc + w[(i, l)].scale(sv[l]) * w[(j, l)].conj()))
    };
    Ok((u, p))
}
// ANCHOR_END: polar

// ANCHOR: subspace_angles
/// The principal angles between the column spaces of `A` and `B` (SciPy's `subspace_angles`),
/// largest first. With orthonormal bases `Q_A`, `Q_B`, the cosines are the singular values of
/// `Q_Aᴴ Q_B`; angles whose cosine exceeds `1/√2` are taken instead from the sines, the singular
/// values of `Q_B − Q_A Q_Aᴴ Q_B`, which keeps small angles accurate (Knyazev and Argentati, 2002).
pub fn subspace_angles<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix2>) -> Result<Array1<f64>> {
    if a.nrows() != b.nrows() {
        return Err(LinalgError::Shape(format!("A and B must have the same number of rows, got {} and {}", a.nrows(), b.nrows())));
    }
    let qa = orth(a, None)?;
    let qb = orth(b, None)?;
    let qa_h_qb = matmul(&adjoint(&qa), &qb);
    let (_, sigma, _) = qa_h_qb.svd(false, false)?;
    let residual = if qa.ncols() >= qb.ncols() {
        let p = matmul(&qa, &qa_h_qb);
        Array2::from_shape_fn(qb.dim(), |(i, j)| qb[(i, j)] - p[(i, j)])
    } else {
        let p = matmul(&qb, &adjoint(&qa_h_qb));
        Array2::from_shape_fn(qa.dim(), |(i, j)| qa[(i, j)] - p[(i, j)])
    };
    let use_sines = sigma.iter().any(|s| s * s >= 0.5);
    let mu_arcsin: Vec<f64> = if use_sines {
        let (_, sv, _) = residual.svd(false, false)?;
        sv.iter().map(|v| v.clamp(-1.0, 1.0).asin()).collect()
    } else {
        Vec::new()
    };
    let len = sigma.len();
    Ok(Array1::from_shape_fn(len, |i| {
        if sigma[i] * sigma[i] >= 0.5 { mu_arcsin.get(i).copied().unwrap_or(0.0) } else { sigma[len - 1 - i].clamp(-1.0, 1.0).acos() }
    }))
}
// ANCHOR_END: subspace_angles

// ANCHOR: funm
/// `f(A)` for a scalar function `f` (SciPy's `funm`), by the Schur-Parlett method: with the complex
/// Schur form `A = Z T Zᴴ`, `F = f(T)` is upper triangular with `F_ii = f(T_ii)`, and the
/// commutation `T F = F T` determines the rest, superdiagonal by superdiagonal:
/// `F_ij = (T_ij (F_jj − F_ii) + Σ_{i<k<j} (T_ik F_kj − F_ik T_kj)) / (T_jj − T_ii)`.
/// Returns `(Z F Zᴴ, error estimate)`; the estimate is large when eigenvalues nearly coincide,
/// where this recurrence divides by small differences.
pub fn funm<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, f: impl Fn(C) -> C) -> Result<(Array2<C>, f64)> {
    let (t, z) = schur_complex(a)?;
    let n = t.nrows();
    let mut fm = Array2::<C>::zeros((n, n));
    for i in 0..n {
        fm[(i, i)] = f(t[(i, i)]);
    }
    let mut minden = if n > 0 { t[(0, 0)].norm() } else { 0.0 };
    for p in 1..n {
        for i in 0..n - p {
            let j = i + p;
            let mut s = t[(i, j)] * (fm[(j, j)] - fm[(i, i)]);
            for k in i + 1..j {
                s += t[(i, k)] * fm[(k, j)] - fm[(i, k)] * t[(k, j)];
            }
            let den = t[(j, j)] - t[(i, i)];
            if den != C::new(0.0, 0.0) {
                s /= den;
            }
            fm[(i, j)] = s;
            minden = minden.min(den.norm());
        }
    }
    let zh = z.t().mapv(|v| v.conj());
    let result = z.dot(&fm).dot(&zh);
    let tol = f64::EPSILON;
    let minden = if minden == 0.0 { tol } else { minden };
    // ‖triu(T, 1)‖₁, the largest column sum of the strictly upper part.
    let upper_norm = (0..n).map(|j| (0..j).map(|i| t[(i, j)].norm()).sum::<f64>()).fold(0.0, f64::max);
    let mut err = 1f64.min(tol.max(tol / minden * upper_norm));
    if result.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        err = f64::INFINITY;
    }
    Ok((result, err))
}
// ANCHOR_END: funm

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expm;

    fn max_diff<T: Scalar>(a: &Array2<T>, b: &Array2<T>) -> f64 {
        a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| m.max((*x - *y).abs()))
    }

    fn test_matrix(m: usize, n: usize, seed: f64) -> Array2<f64> {
        Array2::from_shape_fn((m, n), |(i, j)| (seed + 1.3 * i as f64 + 0.7 * (j * j) as f64 + 0.37 * (i * j) as f64).sin() + if i == j { 2.0 } else { 0.0 })
    }

    fn complex_matrix(m: usize, n: usize, seed: f64) -> Array2<C> {
        let r = test_matrix(m, n, seed);
        Array2::from_shape_fn((m, n), |(i, j)| C::new(r[(i, j)], (seed + 0.4 * (i * i) as f64 - 0.9 * j as f64).cos()))
    }

    #[test]
    fn qr_and_rq_reconstruct_and_are_orthogonal() {
        for (m, n) in [(6, 6), (7, 4), (4, 7)] {
            let a = complex_matrix(m, n, 1.0);
            for mode in [QrMode::Full, QrMode::Economic] {
                for pivoting in [false, true] {
                    let f = qr(&a, mode, pivoting).unwrap();
                    let ap = Array2::from_shape_fn((m, n), |(i, j)| a[(i, f.permutation[j])]);
                    assert!(max_diff(&matmul(&f.q, &f.r), &ap) < 1e-13);
                    let qhq = matmul(&adjoint(&f.q), &f.q);
                    assert!(max_diff(&qhq, &eye(f.q.ncols())) < 1e-13);
                    assert!((0..f.r.nrows()).all(|i| (0..i.min(n)).all(|j| f.r[(i, j)] == C::new(0.0, 0.0))));
                    if pivoting {
                        let d: Vec<f64> = (0..m.min(n)).map(|i| f.r[(i, i)].norm()).collect();
                        assert!(d.windows(2).all(|w| w[0] >= w[1] - 1e-12), "{d:?}");
                    }
                }
                let (r, q) = rq(&a, mode).unwrap();
                assert!(max_diff(&matmul(&r, &q), &a) < 1e-13);
                assert!(max_diff(&matmul(&q, &adjoint(&q)), &eye(q.nrows())) < 1e-13);
            }
        }
    }

    #[test]
    fn hessenberg_and_real_schur_forms() {
        let a = test_matrix(9, 9, 2.0);
        let (h, q) = hessenberg(&a).unwrap();
        assert!((0..9usize).all(|i| (0..i.saturating_sub(1)).all(|j| h[(i, j)] == 0.0)));
        assert!(max_diff(&matmul(&matmul(&q, &h), &adjoint(&q)), &a) < 1e-13);
        let c = complex_matrix(6, 6, 3.0);
        let (h, q) = hessenberg(&c).unwrap();
        assert!((1..6).all(|i| h[(i, i - 1)].im == 0.0), "subdiagonal made real, as LAPACK");
        assert!(max_diff(&matmul(&matmul(&q, &h), &adjoint(&q)), &c) < 1e-13);
        // A rotation-like matrix has complex eigenvalues: 2 × 2 blocks in standard form.
        let (t, z) = schur_real(&a).unwrap();
        assert!(max_diff(&matmul(&matmul(&z, &t), &adjoint(&z)), &a) < 1e-12);
        let mut blocks = 0;
        for i in 1..9 {
            if t[(i, i - 1)] != 0.0 {
                blocks += 1;
                assert!(t[(i, i)] == t[(i - 1, i - 1)] && t[(i, i - 1)] * t[(i - 1, i)] < 0.0);
            }
        }
        assert!(blocks > 0);
        assert!(hessenberg(&Array2::<f64>::zeros((2, 3))).is_err());
    }

    #[test]
    fn ldl_reconstructs_with_two_by_two_pivots() {
        // A zero diagonal forces 2 × 2 pivots.
        let n = 6;
        let s = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 0.0 } else { (0.9 * (i + j) as f64 + 0.3 * (i * j) as f64).sin() * 2.0 });
        for lower in [true, false] {
            let f = ldl(&s, lower, true).unwrap();
            let rec = matmul(&matmul(&f.l, &f.d), &adjoint(&f.l));
            assert!(max_diff(&rec, &s) < 1e-13, "lower = {lower}");
            assert!((0..n - 1).any(|i| f.d[(i + 1, i)] != 0.0));
            // L[perm] is unit triangular.
            for (r, &p) in f.perm.iter().enumerate() {
                assert!((f.l[(p, r)] - 1.0).abs() < 1e-15);
            }
        }
        let h = complex_matrix(5, 5, 4.0);
        let herm = Array2::from_shape_fn((5, 5), |(i, j)| if i == j { C::new(h[(i, i)].re, 0.0) } else { (h[(i, j)] + h[(j, i)].conj()) / 2.0 });
        let f = ldl(&herm, true, true).unwrap();
        assert!(max_diff(&matmul(&matmul(&f.l, &f.d), &adjoint(&f.l)), &herm) < 1e-13);
        let sym = Array2::from_shape_fn((5, 5), |(i, j)| h[(i.min(j), i.max(j))]);
        let f = ldl(&sym, false, false).unwrap();
        let lt = f.l.t().to_owned();
        assert!(max_diff(&matmul(&matmul(&f.l, &f.d), &lt), &sym) < 1e-13);
    }

    #[test]
    fn polar_angles_and_matrix_functions() {
        let a = test_matrix(5, 3, 5.0);
        let (u, p) = polar(&a, true).unwrap();
        assert!(max_diff(&matmul(&u, &p), &a) < 1e-13 && max_diff(&matmul(&adjoint(&u), &u), &eye(3)) < 1e-13);
        let (u, p) = polar(&a, false).unwrap();
        assert!(max_diff(&matmul(&p, &u), &a) < 1e-13);
        // The x axis and the line at 30° in the xy plane of ℝ³.
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 0.0, 0.0]).unwrap();
        let l = Array2::from_shape_vec((3, 1), vec![3f64.sqrt() / 2.0, 0.5, 0.0]).unwrap();
        let th = subspace_angles(&x, &l).unwrap();
        assert!((th[0] - std::f64::consts::PI / 6.0).abs() < 1e-15);
        let tiny = Array2::from_shape_vec((3, 1), vec![1.0, 1e-9, 0.0]).unwrap();
        assert!((subspace_angles(&x, &tiny).unwrap()[0] - 1e-9).abs() < 1e-22, "small angles from sines");
        let m = test_matrix(5, 5, 6.0) * 0.5;
        let (f, err) = funm(&m, |z| z.exp()).unwrap();
        let e = expm(&m).unwrap();
        assert!(f.iter().zip(e.iter()).all(|(x, y)| (x.re - y).abs() < 1e-12 && x.im.abs() < 1e-12), "err {err:e}");
        assert!(err < 1e-10);
    }
}
