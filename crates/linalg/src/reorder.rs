//! The real generalised Schur form (SciPy's `qz(a, b, output='real')`), and reordering of Schur
//! and generalised Schur forms so that chosen eigenvalues come first (SciPy's
//! `schur(a, sort=...)` and `ordqz`).
//!
//! The real QZ algorithm reduces `B` to triangular and `A` to Hessenberg form, then chases
//! Moler and Stewart's implicit double-shift bulges down the pencil until `A` is quasi-triangular.
//! Reordering swaps adjacent diagonal blocks: the invariant (or deflating) subspace belonging to
//! the lower block is found from a small Sylvester equation, and orthogonal transformations
//! built from it bring that block to the top.
//!
//! See the book entry [Real QZ, reordering and Hankel transforms](https://jwkunz.github.io/lib_jsl/book/linear_algebra/qz-reordering.html).

// Dense small-matrix kernels read best as index loops.
#![allow(clippy::needless_range_loop)]

use crate::error::{LinalgError, Result};
use crate::generalized::qz;
use crate::{schur_complex, schur_real};
use ndarray::{Array2, ArrayBase, Data, Ix2};
use num_complex::Complex;

type C = Complex<f64>;

/// Which eigenvalues [`schur_sorted`], [`schur_sorted_complex`], [`ordqz`] and
/// [`ordqz_complex`] move to the top left, as SciPy's `sort` argument.
#[non_exhaustive]
pub enum EigenSort<'a> {
    /// `Re λ < 0` (SciPy's `'lhp'`).
    LeftHalfPlane,
    /// `Re λ ≥ 0` for `schur`, `Re λ > 0` for `ordqz` (SciPy's `'rhp'`).
    RightHalfPlane,
    /// `|λ| ≤ 1` (`'iuc'`).
    InsideUnitCircle,
    /// `|λ| > 1` (`'ouc'`; for `ordqz`, infinite eigenvalues too).
    OutsideUnitCircle,
    /// A predicate on the eigenvalue `λ` (infinite eigenvalues of pencils are passed as
    /// `∞ + 0i`, and the indeterminate `0/0` as NaN).
    Custom(&'a dyn Fn(C) -> bool),
}

impl EigenSort<'_> {
    /// SciPy's `schur` predicates on `λ`.
    fn schur(&self, l: C) -> bool {
        match self {
            EigenSort::LeftHalfPlane => l.re < 0.0,
            EigenSort::RightHalfPlane => l.re >= 0.0,
            EigenSort::InsideUnitCircle => l.norm() <= 1.0,
            EigenSort::OutsideUnitCircle => l.norm() > 1.0,
            EigenSort::Custom(f) => f(l),
        }
    }

    /// SciPy's `ordqz` predicates on `(α, β)`.
    fn pencil(&self, alpha: C, beta: C) -> bool {
        let finite = beta != C::new(0.0, 0.0);
        let l = if finite { alpha / beta } else if alpha == C::new(0.0, 0.0) { C::new(f64::NAN, 0.0) } else { C::new(f64::INFINITY, 0.0) };
        match self {
            EigenSort::LeftHalfPlane => finite && l.re < 0.0,
            EigenSort::RightHalfPlane => finite && l.re > 0.0,
            EigenSort::InsideUnitCircle => finite && l.norm() <= 1.0,
            EigenSort::OutsideUnitCircle => !(alpha == C::new(0.0, 0.0) && !finite) && (!finite || l.norm() > 1.0),
            EigenSort::Custom(f) => f(l),
        }
    }
}

// ------------------------------------------------------------------------------ small kernels

/// A Householder reflector `I − β u uᵀ` with `u₀ = 1` taking `x` to `±‖x‖ e₁`.
pub(crate) fn house(x: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let alpha = x[0];
    let sigma: f64 = x[1..].iter().map(|v| v * v).sum();
    let mut u = x.to_vec();
    u[0] = 1.0;
    if sigma == 0.0 {
        return (u, 0.0);
    }
    let mu = (alpha * alpha + sigma).sqrt();
    let v0 = if alpha <= 0.0 { alpha - mu } else { -sigma / (alpha + mu) };
    let beta = 2.0 * v0 * v0 / (sigma + v0 * v0);
    for k in 1..n {
        u[k] = x[k] / v0;
    }
    (u, beta)
}

/// `M[rows, cols] ← (I − β u uᵀ) M[rows, cols]` with the reflector acting on rows `r0..r0+len`.
pub(crate) fn reflect_rows(m: &mut Array2<f64>, u: &[f64], beta: f64, r0: usize) {
    if beta == 0.0 {
        return;
    }
    for j in 0..m.ncols() {
        let s: f64 = u.iter().enumerate().map(|(k, uk)| uk * m[(r0 + k, j)]).sum();
        for (k, uk) in u.iter().enumerate() {
            m[(r0 + k, j)] -= beta * uk * s;
        }
    }
}

/// `M ← M (I − β u uᵀ)` on columns `c0..c0+len`.
pub(crate) fn reflect_cols(m: &mut Array2<f64>, u: &[f64], beta: f64, c0: usize) {
    if beta == 0.0 {
        return;
    }
    for i in 0..m.nrows() {
        let s: f64 = u.iter().enumerate().map(|(k, uk)| uk * m[(i, c0 + k)]).sum();
        for (k, uk) in u.iter().enumerate() {
            m[(i, c0 + k)] -= beta * uk * s;
        }
    }
}

/// The rotation `(c, s)` with `[c s; −s c] [f; g] = [r; 0]`.
fn givens(f: f64, g: f64) -> (f64, f64) {
    if g == 0.0 {
        return (1.0, 0.0);
    }
    let r = f.hypot(g);
    (f / r, g / r)
}

fn rot_rows(m: &mut Array2<f64>, p: usize, q: usize, c: f64, s: f64) {
    for j in 0..m.ncols() {
        let (x, y) = (m[(p, j)], m[(q, j)]);
        m[(p, j)] = c * x + s * y;
        m[(q, j)] = c * y - s * x;
    }
}

fn rot_cols(m: &mut Array2<f64>, p: usize, q: usize, c: f64, s: f64) {
    for i in 0..m.nrows() {
        let (x, y) = (m[(i, p)], m[(i, q)]);
        m[(i, p)] = c * x + s * y;
        m[(i, q)] = c * y - s * x;
    }
}

/// Solves a small dense system by Gaussian elimination with partial pivoting; `None` if
/// singular to working precision.
fn solve_small(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let scale = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() <= 1e-14 * scale {
            return None;
        }
        a.swap(p, c);
        b.swap(p, c);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    for c in (0..n).rev() {
        let s: f64 = (c + 1..n).map(|k| a[c][k] * b[k]).sum();
        b[c] = (b[c] - s) / a[c][c];
    }
    Some(b)
}

/// An orthogonal `m × m` matrix whose leading `p` columns span the columns of `x` (`m × p`, full
/// rank), from a Householder QR factorisation.
fn orthonormal_completion(x: &Array2<f64>) -> Array2<f64> {
    let (m, p) = x.dim();
    let mut a = x.clone();
    let mut q = Array2::<f64>::eye(m);
    for j in 0..p {
        let col: Vec<f64> = (j..m).map(|i| a[(i, j)]).collect();
        let (u, beta) = house(&col);
        reflect_rows(&mut a, &u, beta, j);
        reflect_cols(&mut q, &u, beta, j);
    }
    q
}

/// The diagonal block sizes of a quasi-triangular matrix.
fn blocks(t: &Array2<f64>) -> Vec<(usize, usize)> {
    let n = t.nrows();
    let mut out = Vec::new();
    let mut k = 0;
    while k < n {
        let s = if k + 1 < n && t[(k + 1, k)] != 0.0 { 2 } else { 1 };
        out.push((k, s));
        k += s;
    }
    out
}

// ------------------------------------------------------------------------------- real Schur

/// The 2 × 2 block at `k` in the standard form `[a b; c a]`, `bc < 0` (LAPACK's `dlanv2`), or
/// split into two 1 × 1 blocks when its eigenvalues are real.
fn standardize_schur_block(t: &mut Array2<f64>, z: &mut Array2<f64>, k: usize) {
    let (_, _, _, _, cs, sn) = crate::factorizations::lanv2_pub(t[(k, k)], t[(k, k + 1)], t[(k + 1, k)], t[(k + 1, k + 1)]);
    rot_rows(t, k, k + 1, cs, sn);
    rot_cols(t, k, k + 1, cs, sn);
    rot_cols(z, k, k + 1, cs, sn);
    let (a, b, c, d) = (t[(k, k)], t[(k, k + 1)], t[(k + 1, k)], t[(k + 1, k + 1)]);
    // Clean the block to exactly the form dlanv2 produces.
    let (a2, b2, c2, d2, _, _) = crate::factorizations::lanv2_pub(a, b, c, d);
    t[(k, k)] = a2;
    t[(k, k + 1)] = b2;
    t[(k + 1, k)] = c2;
    t[(k + 1, k + 1)] = d2;
}

/// Swaps the adjacent diagonal blocks at `k` (size `p1`) and `k + p1` (size `p2`) of the real
/// Schur form `T`, accumulating into `Z`.
pub(crate) fn swap_schur(t: &mut Array2<f64>, z: &mut Array2<f64>, k: usize, p1: usize, p2: usize) -> bool {
    let m = p1 + p2;
    // A₁₁ X − X A₂₂ = −A₁₂ by Kronecker products: (I ⊗ A₁₁ − A₂₂ᵀ ⊗ I) vec X = −vec A₁₂.
    let a11 = |i: usize, j: usize| t[(k + i, k + j)];
    let a22 = |i: usize, j: usize| t[(k + p1 + i, k + p1 + j)];
    let nx = p1 * p2;
    let mut sys = vec![vec![0.0; nx]; nx];
    let mut rhs = vec![0.0; nx];
    for c in 0..p2 {
        for r in 0..p1 {
            let row = r + c * p1;
            rhs[row] = -t[(k + r, k + p1 + c)];
            for rr in 0..p1 {
                sys[row][rr + c * p1] += a11(r, rr);
            }
            for cc in 0..p2 {
                sys[row][r + cc * p1] -= a22(cc, c);
            }
        }
    }
    let Some(x) = solve_small(sys, rhs) else { return false };
    let mut xi = Array2::<f64>::zeros((m, p2));
    for c in 0..p2 {
        for r in 0..p1 {
            xi[(r, c)] = x[r + c * p1];
        }
        xi[(p1 + c, c)] = 1.0;
    }
    let q = orthonormal_completion(&xi);
    let n = t.nrows();
    // Reject the swap if the transformed block is not block triangular to working accuracy
    // (LAPACK's dlaexc test): forcing the off-diagonal block to zero would then not be a
    // similarity.
    let dnorm = (0..m).flat_map(|i| (0..m).map(move |j| (i, j))).map(|(i, j)| t[(k + i, k + j)].abs()).fold(0.0f64, f64::max);
    let thresh = (10.0 * f64::EPSILON * dnorm).max(f64::MIN_POSITIVE);
    for i in p2..m {
        for j in 0..p2 {
            let v: f64 = (0..m).map(|r| q[(r, i)] * (0..m).map(|c| t[(k + r, k + c)] * q[(c, j)]).sum::<f64>()).sum();
            if v.abs() > thresh {
                return false;
            }
        }
    }
    // T ← Qᵀ T Q on rows/columns k..k+m.
    let mut rows = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            rows[(i, j)] = (0..m).map(|l| q[(l, i)] * t[(k + l, j)]).sum();
        }
    }
    for i in 0..m {
        for j in 0..n {
            t[(k + i, j)] = rows[(i, j)];
        }
    }
    let mut cols = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            cols[(i, j)] = (0..m).map(|l| t[(i, k + l)] * q[(l, j)]).sum();
        }
    }
    for i in 0..n {
        for j in 0..m {
            t[(i, k + j)] = cols[(i, j)];
        }
    }
    let mut zc = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            zc[(i, j)] = (0..m).map(|l| z[(i, k + l)] * q[(l, j)]).sum();
        }
    }
    for i in 0..n {
        for j in 0..m {
            z[(i, k + j)] = zc[(i, j)];
        }
    }
    for i in p2..m {
        for j in 0..p2 {
            t[(k + i, k + j)] = 0.0;
        }
    }
    if p2 == 2 {
        standardize_schur_block(t, z, k);
    }
    if p1 == 2 {
        standardize_schur_block(t, z, k + p2);
    }
    true
}

/// The eigenvalue(s) of the block at `k` of size `s`.
pub(crate) fn block_eigs(t: &Array2<f64>, k: usize, s: usize) -> Vec<C> {
    if s == 1 {
        return vec![C::new(t[(k, k)], 0.0)];
    }
    let (a, b, c, d) = (t[(k, k)], t[(k, k + 1)], t[(k + 1, k)], t[(k + 1, k + 1)]);
    let tr = 0.5 * (a + d);
    let disc = 0.25 * (a - d) * (a - d) + b * c;
    if disc >= 0.0 {
        vec![C::new(tr + disc.sqrt(), 0.0), C::new(tr - disc.sqrt(), 0.0)]
    } else {
        vec![C::new(tr, (-disc).sqrt()), C::new(tr, -(-disc).sqrt())]
    }
}

// ANCHOR: schur_sorted
/// The real Schur form `A = Z T Zᵀ` with the eigenvalues selected by `sort` in the leading
/// `sdim` positions (SciPy's `schur(a, output='real', sort=...)`). A complex pair is selected
/// when either member is. The selected blocks are moved up one adjacent swap at a time
/// (LAPACK's `dtrsen`/`dtrexc`). Returns `(T, Z, sdim)`.
pub fn schur_sorted<S: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>, sort: &EigenSort) -> Result<(Array2<f64>, Array2<f64>, usize)> {
    let (mut t, mut z) = schur_real(a)?;
    let mut ks = 0;
    loop {
        // Find the first selected block at or after ks.
        let bl = blocks(&t);
        let next = bl.iter().find(|&&(k, s)| k >= ks && block_eigs(&t, k, s).iter().any(|&l| sort.schur(l)));
        let Some(&(mut k, s)) = next else { break };
        // Move it up to ks.
        while k > ks {
            let above = blocks(&t).into_iter().rfind(|&(kk, _)| kk < k).expect("a block above");
            if !swap_schur(&mut t, &mut z, above.0, above.1, s) {
                return Err(LinalgError::NoConvergence("schur_sorted: the blocks are too close to swap"));
            }
            k = above.0;
        }
        ks += s;
    }
    Ok((t, z, ks))
}
// ANCHOR_END: schur_sorted

/// `zrot`: `(x, y) ← (c x + s y, c y − s̄ x)`.
fn zrot(x: C, y: C, c: f64, s: C) -> (C, C) {
    (x * c + s * y, y * c - s.conj() * x)
}

/// The complex Givens rotation `(c, s)` with `[c s; −s̄ c][f; g] = [r; 0]` (LAPACK's `zlartg`).
fn zlartg(f: C, g: C) -> (f64, C) {
    if g == C::new(0.0, 0.0) {
        return (1.0, C::new(0.0, 0.0));
    }
    if f == C::new(0.0, 0.0) {
        return (0.0, g.conj() / g.norm());
    }
    let (fa, ga) = (f.norm(), g.norm());
    let r = fa.hypot(ga);
    let c = fa / r;
    let s = (f / fa) * g.conj() / r;
    (c, s)
}

/// The complex Schur form `A = Z T Zᴴ` with the eigenvalues selected by `sort` first (SciPy's
/// `schur(a, output='complex', sort=...)`), by adjacent swaps of 1 × 1 blocks (LAPACK's
/// `ztrexc`). Returns `(T, Z, sdim)`.
pub fn schur_sorted_complex<T: crate::Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, sort: &EigenSort) -> Result<(Array2<C>, Array2<C>, usize)> {
    let (mut t, mut z) = schur_complex(a)?;
    let n = t.nrows();
    let mut ks = 0;
    for k in 0..n {
        if !sort.schur(t[(k, k)]) {
            continue;
        }
        for j in (ks..k).rev() {
            // Swap the diagonal entries at j and j + 1.
            let (t11, t22) = (t[(j, j)], t[(j + 1, j + 1)]);
            let (cs, sn) = zlartg(t[(j, j + 1)], t22 - t11);
            for c in j + 2..n {
                let (x, y) = zrot(t[(j, c)], t[(j + 1, c)], cs, sn);
                t[(j, c)] = x;
                t[(j + 1, c)] = y;
            }
            for r in 0..j {
                let (x, y) = zrot(t[(r, j)], t[(r, j + 1)], cs, sn.conj());
                t[(r, j)] = x;
                t[(r, j + 1)] = y;
            }
            t[(j, j)] = t22;
            t[(j + 1, j + 1)] = t11;
            for r in 0..n {
                let (x, y) = zrot(z[(r, j)], z[(r, j + 1)], cs, sn.conj());
                z[(r, j)] = x;
                z[(r, j + 1)] = y;
            }
        }
        ks += 1;
    }
    Ok((t, z, ks))
}

// ------------------------------------------------------------------------------- real QZ

/// The real generalised Schur form `A = Q AA Zᵀ`, `B = Q BB Zᵀ`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RealQzFactors {
    /// Quasi upper triangular: 1 × 1 blocks for real eigenvalues, 2 × 2 for complex pairs.
    pub aa: Array2<f64>,
    /// Upper triangular, with a non-negative diagonal.
    pub bb: Array2<f64>,
    pub q: Array2<f64>,
    pub z: Array2<f64>,
}

struct Pencil {
    a: Array2<f64>,
    b: Array2<f64>,
    q: Array2<f64>,
    z: Array2<f64>,
}

impl Pencil {
    fn rows(&mut self, p: usize, q: usize, c: f64, s: f64) {
        rot_rows(&mut self.a, p, q, c, s);
        rot_rows(&mut self.b, p, q, c, s);
        rot_cols(&mut self.q, p, q, c, s);
    }
    fn cols(&mut self, p: usize, q: usize, c: f64, s: f64) {
        rot_cols(&mut self.a, p, q, c, s);
        rot_cols(&mut self.b, p, q, c, s);
        rot_cols(&mut self.z, p, q, c, s);
    }
    fn house_rows(&mut self, u: &[f64], beta: f64, r0: usize) {
        reflect_rows(&mut self.a, u, beta, r0);
        reflect_rows(&mut self.b, u, beta, r0);
        reflect_cols(&mut self.q, u, beta, r0);
    }
    fn house_cols(&mut self, u: &[f64], beta: f64, c0: usize) {
        reflect_cols(&mut self.a, u, beta, c0);
        reflect_cols(&mut self.b, u, beta, c0);
        reflect_cols(&mut self.z, u, beta, c0);
    }

    /// A reflector from the right on columns `c0..c0+len` taking the row vector `r` to a
    /// multiple of its last unit vector.
    fn house_cols_to_last(&mut self, r: &[f64], c0: usize) {
        let rev: Vec<f64> = r.iter().rev().copied().collect();
        let (u, beta) = house(&rev);
        let u: Vec<f64> = u.into_iter().rev().collect();
        self.house_cols(&u, beta, c0);
    }

    /// `B[j][j] = 0` in the window `lo..=hi`: chase the zero to `hi` and deflate the infinite
    /// eigenvalue there by zeroing `A[hi][hi−1]`.
    fn chase_zero(&mut self, j: usize, lo: usize, hi: usize) {
        for k in j..hi {
            let (c, s) = givens(self.b[(k, k + 1)], self.b[(k + 1, k + 1)]);
            self.rows(k, k + 1, c, s);
            self.b[(k + 1, k + 1)] = 0.0;
            if k > lo {
                let (c, s) = givens(self.a[(k + 1, k)], self.a[(k + 1, k - 1)]);
                self.cols(k, k - 1, c, s);
                self.a[(k + 1, k - 1)] = 0.0;
            }
        }
        if hi > lo {
            let (c, s) = givens(self.a[(hi, hi)], self.a[(hi, hi - 1)]);
            self.cols(hi, hi - 1, c, s);
            self.a[(hi, hi - 1)] = 0.0;
        }
    }

    /// The 2 × 2 block at `k`: split it into two 1 × 1 blocks if its eigenvalues are real.
    fn standardize(&mut self, k: usize) {
        let (a11, a12, a21, a22) = (self.a[(k, k)], self.a[(k, k + 1)], self.a[(k + 1, k)], self.a[(k + 1, k + 1)]);
        let (b11, b12, b22) = (self.b[(k, k)], self.b[(k, k + 1)], self.b[(k + 1, k + 1)]);
        if a21 == 0.0 {
            return;
        }
        // det(A − λB) = qa λ² + qb λ + qc.
        let qa = b11 * b22;
        let qb = -(a11 * b22 + a22 * b11 - a21 * b12);
        let qc = a11 * a22 - a12 * a21;
        let lambda = if qa == 0.0 {
            if qb == 0.0 {
                return;
            }
            -qc / qb
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                return;
            }
            let sq = disc.sqrt();
            let r1 = (-qb + if qb >= 0.0 { -sq } else { sq }) / (2.0 * qa);
            if r1 == 0.0 { 0.0 } else { qc / (qa * r1) }
        };
        // A right null vector of A − λB.
        let (m11, m12, m21, m22) = (a11 - lambda * b11, a12 - lambda * b12, a21, a22 - lambda * b22);
        let (v1, v2) = if m11.hypot(m12) >= m21.hypot(m22) { (-m12, m11) } else { (-m22, m21) };
        let nv = v1.hypot(v2);
        if nv == 0.0 {
            return;
        }
        // Columns (k, k+1) ← [v, v⊥].
        let (c, s) = (v1 / nv, v2 / nv);
        // rot_cols with (c, s) maps column k to c·col_k + s·col_{k+1}.
        self.cols(k, k + 1, c, s);
        let (c2, s2) = givens(self.b[(k, k)], self.b[(k + 1, k)]);
        self.rows(k, k + 1, c2, s2);
        self.b[(k + 1, k)] = 0.0;
        self.a[(k + 1, k)] = 0.0;
    }

    /// One Moler-Stewart double-shift sweep on the window `l..=h`.
    fn double_shift(&mut self, l: usize, h: usize, exceptional: bool) {
        // Trailing 2 × 2 of A B⁻¹: rows h−1, h; B⁻¹ from the trailing 3 × 3 of B.
        let b0 = h - 2;
        let mut binv = [[0.0f64; 3]; 3];
        for c in 0..3 {
            // Back substitution for column c of the inverse of the upper triangular 3 × 3.
            for r in (0..3).rev() {
                let mut s = if r == c { 1.0 } else { 0.0 };
                for k in r + 1..3 {
                    s -= self.b[(b0 + r, b0 + k)] * binv[k][c];
                }
                binv[r][c] = s / self.b[(b0 + r, b0 + r)];
            }
        }
        let cm = |i: usize, j: usize| -> f64 { (0..3).map(|k| self.a[(h - 1 + i, b0 + k)] * binv[k][1 + j]).sum() };
        let (c11, c12, c21, c22) = (cm(0, 0), cm(0, 1), cm(1, 0), cm(1, 1));
        let (mut tr, mut det) = (c11 + c22, c11 * c22 - c12 * c21);
        if exceptional {
            // An ad hoc shift to break cycles.
            let s = self.a[(h, h - 1)].abs() / self.b[(h - 1, h - 1)].abs().max(f64::MIN_POSITIVE) + self.a[(h - 1, h - 2)].abs() / self.b[(h - 2, h - 2)].abs().max(f64::MIN_POSITIVE);
            tr = 1.5 * s;
            det = s * s;
        }
        // First column of (C² − tr C + det I) e_l, C = A B⁻¹.
        let bll = self.b[(l, l)];
        let (u0, u1) = (self.a[(l, l)] / bll, self.a[(l + 1, l)] / bll);
        let w1 = u1 / self.b[(l + 1, l + 1)];
        let w0 = (u0 - self.b[(l, l + 1)] * w1) / bll;
        let mut x = self.a[(l, l)] * w0 + self.a[(l, l + 1)] * w1 - tr * u0 + det;
        let mut y = self.a[(l + 1, l)] * w0 + self.a[(l + 1, l + 1)] * w1 - tr * u1;
        let mut z = self.a[(l + 2, l + 1)] * w1;
        for k in l..h - 1 {
            let (u, beta) = house(&[x, y, z]);
            self.house_rows(&u, beta, k);
            if k > l {
                self.a[(k + 1, k - 1)] = 0.0;
                self.a[(k + 2, k - 1)] = 0.0;
            }
            // Restore B: zero B[k+2][k], B[k+2][k+1], then B[k+1][k].
            let r = [self.b[(k + 2, k)], self.b[(k + 2, k + 1)], self.b[(k + 2, k + 2)]];
            self.house_cols_to_last(&r, k);
            self.b[(k + 2, k)] = 0.0;
            self.b[(k + 2, k + 1)] = 0.0;
            let (c, s) = givens(self.b[(k + 1, k + 1)], self.b[(k + 1, k)]);
            self.cols(k + 1, k, c, s);
            self.b[(k + 1, k)] = 0.0;
            x = self.a[(k + 1, k)];
            y = self.a[(k + 2, k)];
            z = if k + 3 <= h { self.a[(k + 3, k)] } else { 0.0 };
        }
        // The final 2-vector.
        let (c, s) = givens(x, y);
        self.rows(h - 1, h, c, s);
        if h >= 2 {
            self.a[(h, h - 2)] = 0.0;
        }
        let (c, s) = givens(self.b[(h, h)], self.b[(h, h - 1)]);
        self.cols(h, h - 1, c, s);
        self.b[(h, h - 1)] = 0.0;
    }
}

// ANCHOR: qz_real
/// The real generalised Schur form of the pencil `(A, B)` (SciPy's `qz(a, b, output='real')`):
/// orthogonal `Q`, `Z` with `Qᵀ A Z = AA` quasi upper triangular and `Qᵀ B Z = BB` upper
/// triangular. `B` is made triangular by a QR factorisation and `A` Hessenberg by Givens
/// rotations that keep `B` triangular; then Moler and Stewart's implicit double-shift QZ sweeps
/// — the Francis step applied to `A B⁻¹` without forming it — chase bulges down the pencil until
/// `A` is quasi-triangular. A zero on the diagonal of `B` (an infinite eigenvalue) is chased to
/// the bottom of the active window and deflated. 2 × 2 blocks with real eigenvalues are split.
pub fn qz_real<S: Data<Elem = f64>, S2: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix2>) -> Result<RealQzFactors> {
    let n = a.nrows();
    if a.ncols() != n || b.dim() != (n, n) {
        return Err(LinalgError::Shape(format!("A ({:?}) and B ({:?}) must be square and of one size", a.dim(), b.dim())));
    }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite);
    }
    let mut p = Pencil { a: a.to_owned(), b: b.to_owned(), q: Array2::eye(n), z: Array2::eye(n) };
    if n == 0 {
        return Ok(RealQzFactors { aa: p.a, bb: p.b, q: p.q, z: p.z });
    }
    // B = QR.
    for j in 0..n.saturating_sub(1) {
        let col: Vec<f64> = (j..n).map(|i| p.b[(i, j)]).collect();
        let (u, beta) = house(&col);
        p.house_rows(&u, beta, j);
        for i in j + 1..n {
            p.b[(i, j)] = 0.0;
        }
    }
    // A to Hessenberg form, keeping B triangular.
    for j in 0..n.saturating_sub(2) {
        for i in (j + 2..n).rev() {
            let (c, s) = givens(p.a[(i - 1, j)], p.a[(i, j)]);
            p.rows(i - 1, i, c, s);
            p.a[(i, j)] = 0.0;
            let (c, s) = givens(p.b[(i, i)], p.b[(i, i - 1)]);
            p.cols(i, i - 1, c, s);
            p.b[(i, i - 1)] = 0.0;
        }
    }
    let anorm = p.a.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
    let bnorm = p.b.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
    let eps = f64::EPSILON;
    let mut hi = n as isize - 1;
    let mut iterations = 0;
    let mut since_deflation = 0;
    while hi >= 0 {
        let h = hi as usize;
        // Find the start of the active window.
        let mut l = 0;
        for k in (1..=h).rev() {
            if p.a[(k, k - 1)].abs() <= eps * (p.a[(k - 1, k - 1)].abs() + p.a[(k, k)].abs()).max(eps * anorm) {
                p.a[(k, k - 1)] = 0.0;
                l = k;
                break;
            }
        }
        // A negligible diagonal entry of B: an infinite eigenvalue.
        if let Some(j) = (l..=h).find(|&j| p.b[(j, j)].abs() <= eps * bnorm) {
            p.b[(j, j)] = 0.0;
            p.chase_zero(j, l, h);
            hi -= 1;
            since_deflation = 0;
            continue;
        }
        match h - l + 1 {
            1 => {
                hi -= 1;
                since_deflation = 0;
            }
            2 => {
                p.standardize(l);
                hi -= 2;
                since_deflation = 0;
            }
            _ => {
                iterations += 1;
                since_deflation += 1;
                if iterations > 30 * n.max(10) {
                    return Err(LinalgError::NoConvergence("qz_real: the QZ iteration did not converge"));
                }
                p.double_shift(l, h, since_deflation % 10 == 0);
            }
        }
    }
    // A non-negative diagonal for B.
    for j in 0..n {
        if p.b[(j, j)] < 0.0 {
            for c in 0..n {
                p.a[(j, c)] = -p.a[(j, c)];
                p.b[(j, c)] = -p.b[(j, c)];
            }
            for r in 0..n {
                p.q[(r, j)] = -p.q[(r, j)];
            }
        }
    }
    Ok(RealQzFactors { aa: p.a, bb: p.b, q: p.q, z: p.z })
}
// ANCHOR_END: qz_real

/// The generalised eigenvalues `(α, β)` of the real pencil's blocks, `λ = α/β`; for a complex
/// pair `β` is taken from `B`'s diagonal so that `α = λ β`.
fn pencil_eigs(a: &Array2<f64>, b: &Array2<f64>) -> (Vec<C>, Vec<f64>) {
    let mut alpha = Vec::new();
    let mut beta = Vec::new();
    for (k, s) in blocks(a) {
        if s == 1 {
            alpha.push(C::new(a[(k, k)], 0.0));
            beta.push(b[(k, k)]);
        } else {
            let (a11, a12, a21, a22) = (a[(k, k)], a[(k, k + 1)], a[(k + 1, k)], a[(k + 1, k + 1)]);
            let (b11, b12, b22) = (b[(k, k)], b[(k, k + 1)], b[(k + 1, k + 1)]);
            let qa = b11 * b22;
            let qb = -(a11 * b22 + a22 * b11 - a21 * b12);
            let qc = a11 * a22 - a12 * a21;
            let disc = qb * qb - 4.0 * qa * qc;
            let (re, im) = (-qb / (2.0 * qa), (-disc).max(0.0).sqrt() / (2.0 * qa).abs());
            alpha.push(C::new(re, im) * b11);
            beta.push(b11);
            alpha.push(C::new(re, -im) * b22);
            beta.push(b22);
        }
    }
    (alpha, beta)
}

/// Swaps the adjacent diagonal blocks of a real generalised Schur form at `k` (sizes `p1`,
/// `p2`) by solving the generalised Sylvester equation for the deflating subspaces.
fn swap_pencil(p: &mut Pencil, k: usize, p1: usize, p2: usize) -> bool {
    let m = p1 + p2;
    let nx = p1 * p2;
    // Unknowns vec R then vec L: A₁₁R − L A₂₂ = −A₁₂, B₁₁R − L B₂₂ = −B₁₂.
    let mut sys = vec![vec![0.0; 2 * nx]; 2 * nx];
    let mut rhs = vec![0.0; 2 * nx];
    for (eq, mat) in [(0usize, &p.a), (nx, &p.b)] {
        for c in 0..p2 {
            for r in 0..p1 {
                let row = eq + r + c * p1;
                rhs[row] = -mat[(k + r, k + p1 + c)];
                for rr in 0..p1 {
                    sys[row][rr + c * p1] += mat[(k + r, k + rr)];
                }
                for cc in 0..p2 {
                    sys[row][nx + r + cc * p1] -= mat[(k + p1 + cc, k + p1 + c)];
                }
            }
        }
    }
    let Some(sol) = solve_small(sys, rhs) else { return false };
    let build = |off: usize| {
        let mut x = Array2::<f64>::zeros((m, p2));
        for c in 0..p2 {
            for r in 0..p1 {
                x[(r, c)] = sol[off + r + c * p1];
            }
            x[(p1 + c, c)] = 1.0;
        }
        orthonormal_completion(&x)
    };
    let (zl, ql) = (build(0), build(nx));
    let n = p.a.nrows();
    let apply_left = |m_: &mut Array2<f64>| {
        let mut rows = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                rows[(i, j)] = (0..m).map(|l| ql[(l, i)] * m_[(k + l, j)]).sum();
            }
        }
        for i in 0..m {
            for j in 0..n {
                m_[(k + i, j)] = rows[(i, j)];
            }
        }
    };
    let apply_right = |m_: &mut Array2<f64>, w: &Array2<f64>| {
        let nr = m_.nrows();
        let mut cols = Array2::<f64>::zeros((nr, m));
        for i in 0..nr {
            for j in 0..m {
                cols[(i, j)] = (0..m).map(|l| m_[(i, k + l)] * w[(l, j)]).sum();
            }
        }
        for i in 0..nr {
            for j in 0..m {
                m_[(i, k + j)] = cols[(i, j)];
            }
        }
    };
    apply_left(&mut p.a);
    apply_left(&mut p.b);
    apply_right(&mut p.a, &zl);
    apply_right(&mut p.b, &zl);
    apply_right(&mut p.z, &zl);
    apply_right(&mut p.q, &ql);
    for i in p2..m {
        for j in 0..p2 {
            p.a[(k + i, k + j)] = 0.0;
            p.b[(k + i, k + j)] = 0.0;
        }
    }
    // Make the diagonal blocks of B triangular again.
    for (start, size) in [(k, p2), (k + p2, p1)] {
        if size == 2 {
            let (c, s) = givens(p.b[(start, start)], p.b[(start + 1, start)]);
            p.rows(start, start + 1, c, s);
            p.b[(start + 1, start)] = 0.0;
        }
    }
    true
}

/// The result of [`ordqz`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OrdQz {
    pub aa: Array2<f64>,
    pub bb: Array2<f64>,
    /// The generalised eigenvalues are `α/β`, in the order of the reordered form.
    pub alpha: Vec<C>,
    pub beta: Vec<f64>,
    pub q: Array2<f64>,
    pub z: Array2<f64>,
    /// The number of selected eigenvalues (now leading).
    pub sdim: usize,
}

// ANCHOR: ordqz
/// The real generalised Schur form with the eigenvalues selected by `sort` leading (SciPy's
/// `ordqz(a, b, sort=..., output='real')`): [`qz_real`], then adjacent block swaps (LAPACK's
/// `dtgsen`/`dtgexc`) built from the generalised Sylvester equation
/// `A₁₁R − L A₂₂ = −A₁₂`, `B₁₁R − L B₂₂ = −B₁₂`, whose solution spans the deflating subspaces of
/// the lower block.
pub fn ordqz<S: Data<Elem = f64>, S2: Data<Elem = f64>>(a: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix2>, sort: &EigenSort) -> Result<OrdQz> {
    let f = qz_real(a, b)?;
    let mut p = Pencil { a: f.aa, b: f.bb, q: f.q, z: f.z };
    let mut ks = 0;
    loop {
        let bl = blocks(&p.a);
        let (alpha, beta) = pencil_eigs(&p.a, &p.b);
        let mut idx = 0;
        let mut next = None;
        for &(k, s) in &bl {
            if k >= ks && (0..s).any(|t| sort.pencil(alpha[idx + t], C::new(beta[idx + t], 0.0))) {
                next = Some((k, s));
                break;
            }
            idx += s;
        }
        let Some((mut k, s)) = next else { break };
        while k > ks {
            let above = blocks(&p.a).into_iter().rfind(|&(kk, _)| kk < k).expect("a block above");
            if !swap_pencil(&mut p, above.0, above.1, s) {
                return Err(LinalgError::NoConvergence("ordqz: the blocks are too close to swap"));
            }
            k = above.0;
        }
        ks += s;
    }
    for j in 0..p.a.nrows() {
        if p.b[(j, j)] < 0.0 && (j + 1 >= p.a.nrows() || p.a[(j + 1, j)] == 0.0) && (j == 0 || p.a[(j, j - 1)] == 0.0) {
            for c in 0..p.a.ncols() {
                p.a[(j, c)] = -p.a[(j, c)];
                p.b[(j, c)] = -p.b[(j, c)];
            }
            for r in 0..p.q.nrows() {
                p.q[(r, j)] = -p.q[(r, j)];
            }
        }
    }
    let (alpha, beta) = pencil_eigs(&p.a, &p.b);
    Ok(OrdQz { aa: p.a, bb: p.b, alpha, beta, q: p.q, z: p.z, sdim: ks })
}
// ANCHOR_END: ordqz

/// The result of [`ordqz_complex`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OrdQzComplex {
    pub aa: Array2<C>,
    pub bb: Array2<C>,
    pub alpha: Vec<C>,
    pub beta: Vec<C>,
    pub q: Array2<C>,
    pub z: Array2<C>,
    pub sdim: usize,
}

/// The complex generalised Schur form with the selected eigenvalues leading (SciPy's
/// `ordqz(..., output='complex')`): [`qz`], then swaps of adjacent 1 × 1 blocks from the
/// generalised Sylvester equation (LAPACK's `ztgexc`).
pub fn ordqz_complex<T: crate::Scalar, S: Data<Elem = T>, S2: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix2>, sort: &EigenSort) -> Result<OrdQzComplex> {
    let f = qz(a, b)?;
    let (mut s, mut t, mut q, mut z) = (f.s, f.t, f.q, f.z);
    let n = s.nrows();
    let mut ks = 0;
    for k in 0..n {
        if !sort.pencil(s[(k, k)], t[(k, k)]) {
            continue;
        }
        for j in (ks..k).rev() {
            // a11 r − l a22 = −a12, b11 r − l b22 = −b12.
            let (a11, a12, a22) = (s[(j, j)], s[(j, j + 1)], s[(j + 1, j + 1)]);
            let (b11, b12, b22) = (t[(j, j)], t[(j, j + 1)], t[(j + 1, j + 1)]);
            let det = a11 * (-b22) - (-a22) * b11;
            if det.norm() <= 1e-300 {
                continue;
            }
            let r = ((-a12) * (-b22) - (-a22) * (-b12)) / det;
            let l = (a11 * (-b12) - b11 * (-a12)) / det;
            let zm = [[r, C::new(-1.0, 0.0)], [C::new(1.0, 0.0), r.conj()]];
            let qm = [[l, C::new(-1.0, 0.0)], [C::new(1.0, 0.0), l.conj()]];
            let (nz, nq) = ((1.0 + r.norm_sqr()).sqrt(), (1.0 + l.norm_sqr()).sqrt());
            for m_ in [&mut s, &mut t] {
                for c in 0..n {
                    let (x, y) = (m_[(j, c)], m_[(j + 1, c)]);
                    m_[(j, c)] = (qm[0][0].conj() * x + qm[1][0].conj() * y) / nq;
                    m_[(j + 1, c)] = (qm[0][1].conj() * x + qm[1][1].conj() * y) / nq;
                }
            }
            for m_ in [&mut s, &mut t, &mut z] {
                for r_ in 0..n {
                    let (x, y) = (m_[(r_, j)], m_[(r_, j + 1)]);
                    m_[(r_, j)] = (x * zm[0][0] + y * zm[1][0]) / nz;
                    m_[(r_, j + 1)] = (x * zm[0][1] + y * zm[1][1]) / nz;
                }
            }
            for r_ in 0..n {
                let (x, y) = (q[(r_, j)], q[(r_, j + 1)]);
                q[(r_, j)] = (x * qm[0][0] + y * qm[1][0]) / nq;
                q[(r_, j + 1)] = (x * qm[0][1] + y * qm[1][1]) / nq;
            }
            s[(j + 1, j)] = C::new(0.0, 0.0);
            t[(j + 1, j)] = C::new(0.0, 0.0);
            // Real non-negative diagonal of T.
            for d in [j, j + 1] {
                let tn = t[(d, d)].norm();
                if tn > 0.0 {
                    let ph = t[(d, d)].conj() / tn;
                    for r_ in 0..n {
                        s[(r_, d)] *= ph;
                        t[(r_, d)] *= ph;
                        z[(r_, d)] *= ph;
                    }
                    t[(d, d)] = C::new(tn, 0.0);
                }
            }
        }
        ks += 1;
    }
    let alpha = (0..n).map(|k| s[(k, k)]).collect();
    let beta = (0..n).map(|k| t[(k, k)]).collect();
    Ok(OrdQzComplex { aa: s, bb: t, alpha, beta, q, z, sdim: ks })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(n: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        Array2::from_shape_fn((n, n), |_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
    }

    fn check_pencil(a: &Array2<f64>, b: &Array2<f64>, f_a: &Array2<f64>, f_b: &Array2<f64>, q: &Array2<f64>, z: &Array2<f64>) {
        let ra = q.dot(f_a).dot(&z.t());
        let rb = q.dot(f_b).dot(&z.t());
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!((ra[(i, j)] - a[(i, j)]).abs() < 1e-12, "A");
                assert!((rb[(i, j)] - b[(i, j)]).abs() < 1e-12, "B");
                if i > j {
                    assert_eq!(f_b[(i, j)], 0.0);
                }
                if i > j + 1 {
                    assert_eq!(f_a[(i, j)], 0.0);
                }
            }
        }
        let qq = q.t().dot(q);
        assert!((0..n).all(|i| (0..n).all(|j| (qq[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13)));
    }

    #[test]
    fn real_qz_and_reordering() {
        for n in [1usize, 2, 5, 9, 16] {
            let (a, b) = (mat(n, 3 + n as u64), mat(n, 101 + n as u64));
            let f = qz_real(&a, &b).unwrap();
            check_pencil(&a, &b, &f.aa, &f.bb, &f.q, &f.z);
            // The eigenvalues agree with the complex QZ.
            let o = ordqz(&a, &b, &EigenSort::InsideUnitCircle).unwrap();
            check_pencil(&a, &b, &o.aa, &o.bb, &o.q, &o.z);
            let lam: Vec<C> = o.alpha.iter().zip(&o.beta).map(|(al, be)| al / be).collect();
            assert!(lam[..o.sdim].iter().all(|l| l.norm() <= 1.0 + 1e-10));
            assert!(lam[o.sdim..].iter().all(|l| l.norm() > 1.0 - 1e-10));
            let (t, z, sdim) = schur_sorted(&a, &EigenSort::LeftHalfPlane).unwrap();
            let r = z.dot(&t).dot(&z.t());
            assert!(r.iter().zip(a.iter()).all(|(x, y)| (x - y).abs() < 1e-12));
            let mut idx = 0;
            for (k, s) in blocks(&t) {
                let e = block_eigs(&t, k, s);
                assert_eq!(e[0].re < 0.0, idx < sdim, "n = {n}");
                idx += s;
            }
            let oc = ordqz_complex(&a, &b, &EigenSort::LeftHalfPlane).unwrap();
            let lc: Vec<C> = oc.alpha.iter().zip(&oc.beta).map(|(al, be)| al / be).collect();
            assert!(lc[..oc.sdim].iter().all(|l| l.re < 0.0) && lc[oc.sdim..].iter().all(|l| l.re >= 0.0));
            let (tc, _, sc) = schur_sorted_complex(&a, &EigenSort::OutsideUnitCircle).unwrap();
            assert!((0..n).all(|k| (tc[(k, k)].norm() > 1.0) == (k < sc)));
        }
        // A singular B: infinite eigenvalues are deflated.
        let a = mat(6, 7);
        let mut b = mat(6, 8);
        for j in 0..6 {
            b[(2, j)] = 0.0;
            b[(4, j)] = 0.0;
        }
        let f = qz_real(&a, &b).unwrap();
        check_pencil(&a, &b, &f.aa, &f.bb, &f.q, &f.z);
        assert_eq!((0..6).filter(|&k| f.bb[(k, k)].abs() < 1e-12).count(), 2);
    }
}
