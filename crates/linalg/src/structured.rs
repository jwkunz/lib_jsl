//! Structured matrices (SciPy's `scipy.linalg` constructors: `toeplitz`, `hankel`, `circulant`,
//! `hadamard`, `block_diag`, `companion`, `pascal`, `hilbert`, `leslie`, `dft`,
//! `convolution_matrix`, `fiedler`) and fast solvers that exploit structure: Levinson recursion
//! for Toeplitz systems, the FFT for circulant ones, and LU and Cholesky in band storage.
//!
//! See the book entry [Generalised eigenproblems and structured matrices](https://jwkunz.github.io/lib_jsl/book/linear_algebra/generalized-structured.html).

use crate::error::{LinalgError, Result};
use crate::scalar::Scalar;
use lib_jsl_ffts::{FftDirection, fft_axis};
use ndarray::{Array1, Array2, ArrayBase, ArrayD, Data, Ix1, Ix2, IxDyn};
use num_complex::Complex;

type C = Complex<f64>;

fn filled<T: Scalar>(rows: usize, cols: usize) -> Array2<T> {
    Array2::from_elem((rows, cols), T::zero())
}

/// The Toeplitz matrix with first column `c` and first row `r` (SciPy's `toeplitz`); `r[0]` is
/// ignored. Without `r`, the matrix is Hermitian: `r = conj(c)`.
pub fn toeplitz<T: Scalar>(c: &[T], r: Option<&[T]>) -> Array2<T> {
    let conj: Vec<T> = c.iter().map(|v| v.conj()).collect();
    let r = r.unwrap_or(&conj);
    Array2::from_shape_fn((c.len(), r.len()), |(i, j)| if i >= j { c[i - j] } else { r[j - i] })
}

/// The Hankel matrix with first column `c` and last row `r` (SciPy's `hankel`); `r[0]` is ignored.
/// Without `r`, the entries below the anti-diagonal are zero.
pub fn hankel<T: Scalar>(c: &[T], r: Option<&[T]>) -> Array2<T> {
    let zeros = vec![T::zero(); c.len()];
    let r = r.unwrap_or(&zeros);
    let m = c.len();
    Array2::from_shape_fn((m, r.len()), |(i, j)| if i + j < m { c[i + j] } else { r[i + j - m + 1] })
}

/// The circulant matrix with first column `c` (SciPy's `circulant`): `C[i, j] = c[(i − j) mod n]`.
pub fn circulant<T: Scalar>(c: &[T]) -> Array2<T> {
    let n = c.len();
    Array2::from_shape_fn((n, n), |(i, j)| c[(i + n - j) % n])
}

/// The `n × n` Hadamard matrix of ±1 entries by Sylvester's construction (SciPy's `hadamard`); `n`
/// must be a power of two. Its rows are orthogonal: `H Hᵀ = n I`.
pub fn hadamard(n: usize) -> Result<Array2<f64>> {
    if n == 0 || !n.is_power_of_two() {
        return Err(LinalgError::Shape(format!("hadamard: n must be a power of 2, got {n}")));
    }
    Ok(Array2::from_shape_fn((n, n), |(i, j)| if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 }))
}

/// The block diagonal matrix with the given blocks (SciPy's `block_diag`).
pub fn block_diag<T: Scalar>(blocks: &[Array2<T>]) -> Array2<T> {
    let rows = blocks.iter().map(|b| b.nrows()).sum();
    let cols = blocks.iter().map(|b| b.ncols()).sum();
    let mut out = filled(rows, cols);
    let (mut r0, mut c0) = (0, 0);
    for b in blocks {
        for ((i, j), &v) in b.indexed_iter() {
            out[(r0 + i, c0 + j)] = v;
        }
        r0 += b.nrows();
        c0 += b.ncols();
    }
    out
}

/// The companion matrix of the polynomial `a[0] xⁿ + a[1] xⁿ⁻¹ + … + a[n]` (SciPy's `companion`):
/// first row `−a[1..]/a[0]`, ones on the subdiagonal. Its eigenvalues are the polynomial's roots.
pub fn companion<T: Scalar>(a: &[T]) -> Result<Array2<T>> {
    if a.len() < 2 || a[0] == T::zero() {
        return Err(LinalgError::Shape("companion: need at least two coefficients and a[0] ≠ 0".into()));
    }
    let n = a.len() - 1;
    Ok(Array2::from_shape_fn((n, n), |(i, j)| if i == 0 { -a[j + 1] / a[0] } else if i == j + 1 { T::one() } else { T::zero() }))
}

/// Which Pascal matrix (SciPy's `kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PascalKind {
    /// `binom(i + j, i)`.
    Symmetric,
    /// `binom(i, j)`.
    Lower,
    /// `binom(j, i)`.
    Upper,
}

fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64).round()
}

/// The Pascal matrix (SciPy's `pascal`, `exact=False`): binomial coefficients, exact while they
/// are below 2⁵³.
pub fn pascal(n: usize, kind: PascalKind) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| match kind {
        PascalKind::Symmetric => binom(i + j, i),
        PascalKind::Lower => binom(i, j),
        PascalKind::Upper => binom(j, i),
    })
}

/// The inverse of the Pascal matrix (SciPy's `invpascal`): integer entries,
/// `(−1)^{i+j} binom(i, j)` for the triangular kinds, and their product for the symmetric one.
pub fn invpascal(n: usize, kind: PascalKind) -> Array2<f64> {
    let lower_inv = Array2::from_shape_fn((n, n), |(i, j)| if (i + j) % 2 == 0 { binom(i, j) } else { -binom(i, j) });
    match kind {
        PascalKind::Lower => lower_inv,
        PascalKind::Upper => lower_inv.t().to_owned(),
        // P = L Lᵀ, so P⁻¹ = L⁻ᵀ L⁻¹.
        PascalKind::Symmetric => lower_inv.t().dot(&lower_inv),
    }
}

/// The Hilbert matrix `1/(i + j + 1)` (SciPy's `hilbert`), a classic ill-conditioned test matrix.
pub fn hilbert(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64)
}

/// The exact inverse of the Hilbert matrix (SciPy's `invhilbert`), from its closed form
/// `(−1)^{i+j} (i + j + 1) binom(n+i, n−j−1) binom(n+j, n−i−1) binom(i+j, i)²`; the integer entries
/// are exact for `n ≤ 14`.
pub fn invhilbert(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        let s = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
        let b = binom(i + j, i);
        s * (i + j + 1) as f64 * binom(n + i, n - j - 1) * binom(n + j, n - i - 1) * b * b
    })
}

/// The Leslie matrix of a population model (SciPy's `leslie`): fecundities `f` (length `n`) on
/// the first row, survival rates `s` (length `n − 1`) on the subdiagonal.
pub fn leslie(f: &[f64], s: &[f64]) -> Result<Array2<f64>> {
    let n = f.len();
    if n == 0 || s.len() + 1 != n {
        return Err(LinalgError::Shape(format!("leslie: need len(s) = len(f) − 1, got {} and {}", s.len(), n)));
    }
    Ok(Array2::from_shape_fn((n, n), |(i, j)| if i == 0 { f[j] } else if i == j + 1 { s[j] } else { 0.0 }))
}

/// How [`dft`] scales its matrix (SciPy's `scale`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DftScale {
    /// No scaling.
    None,
    /// `1/√n`: unitary.
    SqrtN,
    /// `1/n`.
    N,
}

/// The discrete Fourier transform matrix `W[j, k] = ω^{jk}`, `ω = e^{−2πi/n}` (SciPy's `dft`).
pub fn dft(n: usize, scale: DftScale) -> Array2<C> {
    let s = match scale {
        DftScale::None => 1.0,
        DftScale::SqrtN => 1.0 / (n as f64).sqrt(),
        DftScale::N => 1.0 / n as f64,
    };
    Array2::from_shape_fn((n, n), |(j, k)| {
        let phase = -2.0 * std::f64::consts::PI * ((j * k) % n.max(1)) as f64 / n as f64;
        C::new(phase.cos(), phase.sin()) * s
    })
}

/// Which part of a convolution [`convolution_matrix`] produces (as `numpy.convolve`'s `mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvolutionMode {
    /// All `m + n − 1` outputs.
    Full,
    /// `max(m, n)` outputs, centred.
    Same,
    /// The `max(m, n) − min(m, n) + 1` outputs computed without padding.
    Valid,
}

/// The matrix `M` with `M v = convolve(a, v)` for vectors `v` of length `n` (SciPy's
/// `convolution_matrix`).
pub fn convolution_matrix<T: Scalar>(a: &[T], n: usize, mode: ConvolutionMode) -> Result<Array2<T>> {
    let m = a.len();
    if m == 0 || n == 0 {
        return Err(LinalgError::Shape("convolution_matrix: a and n must be non-empty".into()));
    }
    let full = Array2::from_shape_fn((m + n - 1, n), |(i, j)| if i >= j && i - j < m { a[i - j] } else { T::zero() });
    let (start, len) = match mode {
        ConvolutionMode::Full => (0, m + n - 1),
        ConvolutionMode::Same => {
            let len = m.max(n);
            ((m + n - 1 - len) / 2, len)
        }
        ConvolutionMode::Valid => (m.min(n) - 1, m.max(n) - m.min(n) + 1),
    };
    Ok(full.slice(ndarray::s![start..start + len, ..]).to_owned())
}

/// The Fiedler matrix `|a_i − a_j|` (SciPy's `fiedler`).
pub fn fiedler(a: &[f64]) -> Array2<f64> {
    let n = a.len();
    Array2::from_shape_fn((n, n), |(i, j)| (a[i] - a[j]).abs())
}

// ANCHOR: solve_toeplitz
/// Solves `T x = b` for the Toeplitz matrix `T` with first column `c` and first row `r` (default
/// `conj(c)`) by Levinson recursion in `O(n²)` operations (SciPy's `solve_toeplitz`, after the
/// classic `toeplz`). The recursion builds the solutions for the leading `m × m` sections one size at
/// a time, together with those of two auxiliary systems for the reversed sections; it needs every
/// leading principal minor to be non-singular, and is not stable for indefinite or badly
/// conditioned `T` (then use `solve`).
pub fn solve_toeplitz<T: Scalar>(c: &[T], r: Option<&[T]>, b: &[T]) -> Result<Array1<T>> {
    let n = b.len();
    let conj: Vec<T> = c.iter().map(|v| v.conj()).collect();
    let r = r.unwrap_or(&conj);
    if c.len() != n || r.len() != n || n == 0 {
        return Err(LinalgError::Shape(format!("solve_toeplitz: c, r and b must have one length, got {}, {}, {n}", c.len(), r.len())));
    }
    if c.iter().chain(r).chain(b).any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite);
    }
    // R[k] (1-based, k = 1..2n−1) with T_ij = R[n + i − j].
    let mut rr = vec![T::zero(); 2 * n];
    for k in 1..2 * n {
        let d = k as isize - n as isize; // i − j
        rr[k] = if d >= 0 { c[d as usize] } else { r[(-d) as usize] };
    }
    let singular = || LinalgError::Shape("solve_toeplitz: a leading principal minor is singular".into());
    let y: Vec<T> = std::iter::once(T::zero()).chain(b.iter().copied()).collect();
    let mut x = vec![T::zero(); n + 1];
    let mut g = vec![T::zero(); n + 1];
    let mut h = vec![T::zero(); n + 1];
    if rr[n] == T::zero() {
        return Err(singular());
    }
    x[1] = y[1] / rr[n];
    if n == 1 {
        return Ok(Array1::from_vec(x[1..].to_vec()));
    }
    g[1] = rr[n - 1] / rr[n];
    h[1] = rr[n + 1] / rr[n];
    for m in 1..=n {
        let m1 = m + 1;
        let mut sxn = -y[m1];
        let mut sd = -rr[n];
        for j in 1..=m {
            sxn += rr[n + m1 - j] * x[j];
            sd += rr[n + m1 - j] * g[m - j + 1];
        }
        if sd == T::zero() {
            return Err(singular());
        }
        x[m1] = sxn / sd;
        for j in 1..=m {
            let t = x[m1] * g[m - j + 1];
            x[j] -= t;
        }
        if m1 == n {
            break;
        }
        let mut sgn = -rr[n - m1];
        let mut shn = -rr[n + m1];
        let mut sgd = -rr[n];
        for j in 1..=m {
            sgn += rr[n + j - m1] * g[j];
            shn += rr[n + m1 - j] * h[j];
            sgd += rr[n + j - m1] * h[m - j + 1];
        }
        if sgd == T::zero() {
            return Err(singular());
        }
        g[m1] = sgn / sgd;
        h[m1] = shn / sd;
        let (pp, qq) = (g[m1], h[m1]);
        let mut k = m;
        for j in 1..=m.div_ceil(2) {
            let (pt1, pt2, qt1, qt2) = (g[j], g[k], h[j], h[k]);
            g[j] = pt1 - pp * qt2;
            g[k] = pt2 - pp * qt1;
            h[j] = qt1 - qq * pt2;
            h[k] = qt2 - qq * pt1;
            k -= 1;
        }
    }
    Ok(Array1::from_vec(x[1..].to_vec()))
}
// ANCHOR_END: solve_toeplitz

// ANCHOR: solve_circulant
/// Solves `C x = b` for the circulant matrix `C` with first column `c` in `O(n log n)` (SciPy's
/// `solve_circulant`): the DFT diagonalises every circulant matrix, `C = F⁻¹ diag(F c) F`, so
/// `x = F⁻¹ (F b / F c)`. An error is returned when some `|F c|` is below `tol` (default
/// `n ε max |F c|`), where `C` is numerically singular.
pub fn solve_circulant<T: Scalar>(c: &[T], b: &[T], tol: Option<f64>) -> Result<Array1<T>> {
    let n = c.len();
    if b.len() != n || n == 0 {
        return Err(LinalgError::Shape(format!("solve_circulant: c and b must have one length, got {n} and {}", b.len())));
    }
    let transform = |v: Vec<C>, direction: FftDirection| -> Result<Vec<C>> {
        let mut data = ArrayD::from_shape_vec(IxDyn(&[n]), v).expect("shape");
        fft_axis(&mut data, 0, direction).map_err(|e| LinalgError::Shape(format!("fft: {e:?}")))?;
        Ok(data.into_iter().collect())
    };
    let fc = transform(c.iter().map(|z| z.to_complex()).collect(), FftDirection::Forward)?;
    let fb = transform(b.iter().map(|z| z.to_complex()).collect(), FftDirection::Forward)?;
    let largest = fc.iter().fold(0.0f64, |m, v| m.max(v.norm()));
    let tol = tol.unwrap_or(largest * n as f64 * f64::EPSILON);
    if fc.iter().any(|v| v.norm() <= tol) {
        return Err(LinalgError::Shape("solve_circulant: the circulant matrix is singular".into()));
    }
    // The quotient stays complex until after the inverse transform.
    let q: Vec<C> = fb.iter().zip(&fc).map(|(p, d)| p / d).collect();
    let x = transform(q, FftDirection::Inverse)?;
    Ok(Array1::from_iter(x.into_iter().map(T::from_complex)))
}
// ANCHOR_END: solve_circulant

/// A row of a banded matrix, stored by absolute column from `start`.
struct BandRow<T> {
    start: isize,
    vals: Vec<T>,
}

impl<T: Scalar> BandRow<T> {
    fn get(&self, j: usize) -> T {
        let k = j as isize - self.start;
        if k >= 0 && (k as usize) < self.vals.len() { self.vals[k as usize] } else { T::zero() }
    }
    fn set(&mut self, j: usize, v: T) {
        let k = (j as isize - self.start) as usize;
        self.vals[k] = v;
    }
}

// ANCHOR: solve_banded
/// Solves `A x = b` for a banded `A` with `l` subdiagonals and `u` superdiagonals, given in
/// LAPACK band storage `ab[u + i − j, j] = A[i, j]` (SciPy's `solve_banded((l, u), ab, b)`), by
/// Gaussian elimination with partial pivoting within the band (LAPACK's `gbsv`): `O(n l (l + u))`
/// operations. Row interchanges widen the upper band of `U` to `l + u`.
pub fn solve_banded<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>>(lu: (usize, usize), ab: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix1>) -> Result<Array1<T>> {
    let (l, u) = lu;
    let n = ab.ncols();
    if ab.nrows() != l + u + 1 || b.len() != n {
        return Err(LinalgError::Shape(format!("solve_banded: ab must be {} × n and b of length n = {n}", l + u + 1)));
    }
    if ab.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite);
    }
    // The row at position i is stored over columns i − l ..= i + u + l. At step k every candidate
    // row is non-zero only in columns k ..= k + l + u, so after an interchange each row still fits
    // the window of its new position, and is moved into it.
    let width = 2 * l + u + 1;
    let mut rows: Vec<BandRow<T>> = (0..n)
        .map(|i| {
            let start = i as isize - l as isize;
            let mut row = BandRow { start, vals: vec![T::zero(); width] };
            for j in i.saturating_sub(l)..=(i + u).min(n - 1) {
                row.set(j, ab[(u + i - j, j)]);
            }
            row
        })
        .collect();
    let mut rhs: Vec<T> = b.to_vec();
    for k in 0..n {
        let last = (k + l).min(n - 1);
        let p = (k..=last).max_by(|&i, &j| rows[i].get(k).abs().total_cmp(&rows[j].get(k).abs()).then(j.cmp(&i))).expect("non-empty");
        if rows[p].get(k) == T::zero() {
            return Err(LinalgError::Shape(format!("solve_banded: the matrix is singular (zero pivot in column {k})")));
        }
        if p != k {
            let rewindow = |row: &BandRow<T>, pos: usize| {
                let mut out = BandRow { start: pos as isize - l as isize, vals: vec![T::zero(); width] };
                for j in k..=(k + l + u).min(n - 1) {
                    out.set(j, row.get(j));
                }
                out
            };
            let (to_k, to_p) = (rewindow(&rows[p], k), rewindow(&rows[k], p));
            rows[k] = to_k;
            rows[p] = to_p;
            rhs.swap(p, k);
        }
        let pivot = rows[k].get(k);
        let end = (k + u + l).min(n - 1);
        for i in k + 1..=last {
            let m = rows[i].get(k) / pivot;
            if m == T::zero() {
                continue;
            }
            for j in k..=end {
                let v = rows[i].get(j) - m * rows[k].get(j);
                rows[i].set(j, v);
            }
            let t = m * rhs[k];
            rhs[i] -= t;
        }
    }
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let end = (i + u + l).min(n - 1);
        let s = (i + 1..=end).fold(T::zero(), |acc, j| acc + rows[i].get(j) * x[j]);
        x[i] = (rhs[i] - s) / rows[i].get(i);
    }
    Ok(Array1::from_vec(x))
}
// ANCHOR_END: solve_banded

// ANCHOR: solveh_banded
/// Solves `A x = b` for a Hermitian positive definite banded `A` (SciPy's `solveh_banded`, LAPACK's
/// `pbsv`) by the band Cholesky factorisation `A = L Lᴴ`, which keeps the bandwidth `w`:
/// `O(n w²)` operations. `ab` has `w + 1` rows; with `lower = false` (SciPy's default) it holds the
/// upper triangle, `ab[w + i − j, j] = A[i, j]` for `i ≤ j`, and with `lower = true` the lower one,
/// `ab[i − j, j] = A[i, j]` for `i ≥ j`.
pub fn solveh_banded<T: Scalar, S: Data<Elem = T>, S2: Data<Elem = T>>(ab: &ArrayBase<S, Ix2>, b: &ArrayBase<S2, Ix1>, lower: bool) -> Result<Array1<T>> {
    let n = ab.ncols();
    let w = ab.nrows().saturating_sub(1);
    if ab.nrows() == 0 || b.len() != n {
        return Err(LinalgError::Shape(format!("solveh_banded: ab must have at least one row and b length n = {n}")));
    }
    if ab.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite);
    }
    // a(i, j) for i ≥ j within the band, from either storage.
    let a = |i: usize, j: usize| -> T { if lower { ab[(i - j, j)] } else { ab[(w + j - i, i)].conj() } };
    // L by rows within the band: lband[i][k] = L[i, i − w + k].
    let mut lmat = vec![vec![T::zero(); w + 1]; n];
    let get = |lmat: &Vec<Vec<T>>, i: usize, j: usize| -> T { if j + w >= i && j <= i { lmat[i][j + w - i] } else { T::zero() } };
    for j in 0..n {
        let start = j.saturating_sub(w);
        let d = a(j, j).re() - (start..j).map(|k| get(&lmat, j, k).abs_sqr()).sum::<f64>();
        if d <= 0.0 || !d.is_finite() {
            return Err(LinalgError::Shape(format!("solveh_banded: the matrix is not positive definite (leading minor {})", j + 1)));
        }
        let ljj = d.sqrt();
        lmat[j][w] = T::from_real(ljj);
        for i in j + 1..=(j + w).min(n - 1) {
            let s = (i.saturating_sub(w).max(start)..j).fold(T::zero(), |acc, k| acc + get(&lmat, i, k) * get(&lmat, j, k).conj());
            lmat[i][j + w - i] = (a(i, j) - s).scale(1.0 / ljj);
        }
    }
    // L y = b, then Lᴴ x = y.
    let mut y: Vec<T> = b.to_vec();
    for i in 0..n {
        let s = (i.saturating_sub(w)..i).fold(T::zero(), |acc, k| acc + get(&lmat, i, k) * y[k]);
        y[i] = (y[i] - s) / get(&lmat, i, i);
    }
    let mut x = y;
    for i in (0..n).rev() {
        let s = (i + 1..=(i + w).min(n - 1)).fold(T::zero(), |acc, k| acc + get(&lmat, k, i).conj() * x[k]);
        x[i] = (x[i] - s) / get(&lmat, i, i);
    }
    Ok(Array1::from_vec(x))
}
// ANCHOR_END: solveh_banded

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Eig, Solve};

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
    }

    #[test]
    fn fast_solvers_agree_with_dense_solves() {
        let n = 9;
        let c: Vec<f64> = (0..n).map(|k| 4.0 / (1.0 + k as f64) + if k == 0 { 2.0 } else { 0.0 }).collect();
        let r: Vec<f64> = (0..n).map(|k| if k == 0 { c[0] } else { -0.7 / (1.0 + k as f64 * k as f64) }).collect();
        let b: Vec<f64> = (0..n).map(|k| (k as f64 * 0.7).sin() + 1.0).collect();
        let t = toeplitz(&c, Some(&r));
        let dense = t.solve(&Array1::from_vec(b.clone())).unwrap();
        let fast = solve_toeplitz(&c, Some(&r), &b).unwrap();
        assert!(close(fast.as_slice().unwrap(), dense.as_slice().unwrap(), 1e-12));
        // Complex Hermitian Toeplitz (r defaults to conj(c)).
        let cc: Vec<C> = c.iter().enumerate().map(|(k, &v)| C::new(v, if k == 0 { 0.0 } else { 0.3 / k as f64 })).collect();
        let bc: Vec<C> = b.iter().map(|&v| C::new(v, -v)).collect();
        let xc = solve_toeplitz(&cc, None, &bc).unwrap();
        let tc = toeplitz(&cc, None);
        assert!((tc.dot(&xc) - Array1::from_vec(bc.clone())).iter().all(|v| v.norm() < 1e-12));
        // Circulant.
        let x = solve_circulant(&c, &b, None).unwrap();
        let dense = circulant(&c).solve(&Array1::from_vec(b.clone())).unwrap();
        assert!(close(x.as_slice().unwrap(), dense.as_slice().unwrap(), 1e-12));
        assert!(solve_circulant(&[1.0, 1.0], &[1.0, 2.0], None).is_err(), "singular");
        // Banded: l = 2, u = 1, with a small diagonal so that pivoting happens.
        let (l, u) = (2, 1);
        let a = Array2::from_shape_fn((n, n), |(i, j)| if i <= j + l && j <= i + u { ((i * 7 + j * 3) as f64 * 0.37).sin() + if i == j { 0.01 } else { 0.0 } } else { 0.0 });
        let mut ab = Array2::<f64>::zeros((l + u + 1, n));
        for i in 0..n {
            for j in 0..n {
                if i <= j + l && j <= i + u {
                    ab[(u + i - j, j)] = a[(i, j)];
                }
            }
        }
        let bb = Array1::from_vec(b.clone());
        let xb = solve_banded((l, u), &ab, &bb).unwrap();
        assert!(close(xb.as_slice().unwrap(), a.solve(&bb).unwrap().as_slice().unwrap(), 1e-11));
        // Repeated interchanges that push a row down several times (l = 3, u = 2, n = 40).
        let (l, u, n2) = (3, 2, 40);
        let ab = Array2::from_shape_fn((l + u + 1, n2), |(i, j)| (0.37 * (7 * (i + 1) + 3 * j) as f64).sin() + if i == u { 0.01 } else { 0.0 });
        let dense = Array2::from_shape_fn((n2, n2), |(i, j)| if i <= j + l && j <= i + u { ab[(u + i - j, j)] } else { 0.0 });
        let rhs = Array1::from_shape_fn(n2, |k| (0.7 * k as f64).sin() + 1.0);
        let xb = solve_banded((l, u), &ab, &rhs).unwrap();
        assert!((dense.dot(&xb) - &rhs).iter().all(|v| v.abs() < 1e-10));
        // Symmetric positive definite band, both storages.
        let w = 2;
        let spd = Array2::from_shape_fn((n, n), |(i, j)| if i.abs_diff(j) <= w { if i == j { 6.0 } else { 1.0 / (1 + i + j) as f64 } } else { 0.0 });
        let mut upper = Array2::<f64>::zeros((w + 1, n));
        let mut lower = Array2::<f64>::zeros((w + 1, n));
        for j in 0..n {
            for i in j.saturating_sub(w)..=j {
                upper[(w + i - j, j)] = spd[(i, j)];
            }
            for i in j..=(j + w).min(n - 1) {
                lower[(i - j, j)] = spd[(i, j)];
            }
        }
        let want = spd.solve(&bb).unwrap();
        for (storage, low) in [(&upper, false), (&lower, true)] {
            let x = solveh_banded(storage, &bb, low).unwrap();
            assert!(close(x.as_slice().unwrap(), want.as_slice().unwrap(), 1e-12));
        }
    }

    #[test]
    fn special_matrices() {
        let h = hadamard(8).unwrap();
        assert!((h.dot(&h.t()) - Array2::<f64>::eye(8) * 8.0).iter().all(|v| *v == 0.0));
        assert!(hadamard(6).is_err());
        let ih = invhilbert(6).dot(&hilbert(6));
        assert!((ih - Array2::<f64>::eye(6)).iter().all(|v| v.abs() < 1e-8));
        for kind in [PascalKind::Symmetric, PascalKind::Lower, PascalKind::Upper] {
            let p = pascal(7, kind).dot(&invpascal(7, kind));
            assert!((p - Array2::<f64>::eye(7)).iter().all(|v| *v == 0.0), "{kind:?}");
        }
        // Roots of (x − 1)(x − 2)(x − 3) = x³ − 6x² + 11x − 6.
        let (roots, _) = companion(&[1.0, -6.0, 11.0, -6.0]).unwrap().eig().unwrap();
        let mut re: Vec<f64> = roots.iter().map(|z| z.re).collect();
        re.sort_by(f64::total_cmp);
        assert!(close(&re, &[1.0, 2.0, 3.0], 1e-12));
        // Convolution matrices reproduce numpy.convolve.
        let a = [1.0, 2.0, 3.0];
        let v = Array1::from_vec(vec![1.0, -1.0, 0.5, 2.0]);
        assert_eq!(convolution_matrix(&a, 4, ConvolutionMode::Full).unwrap().dot(&v).to_vec(), vec![1.0, 1.0, 1.5, 0.0, 5.5, 6.0]);
        assert_eq!(convolution_matrix(&a, 4, ConvolutionMode::Same).unwrap().dot(&v).to_vec(), vec![1.0, 1.5, 0.0, 5.5]);
        assert_eq!(convolution_matrix(&a, 4, ConvolutionMode::Valid).unwrap().dot(&v).to_vec(), vec![1.5, 0.0]);
        assert_eq!(hankel(&[1.0, 2.0, 3.0], Some(&[9.0, 4.0, 5.0])), Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]).unwrap());
        let bd = block_diag(&[Array2::from_elem((1, 2), 1.0), Array2::from_elem((2, 1), 2.0)]);
        assert_eq!(bd.dim(), (3, 3));
        assert_eq!(bd[(1, 2)], 2.0);
        let f = dft(4, DftScale::SqrtN);
        let fh = f.t().mapv(|z| z.conj());
        assert!((f.dot(&fh) - Array2::<C>::eye(4)).iter().all(|v| v.norm() < 1e-15));
        assert!(leslie(&[0.1, 2.0, 1.0], &[0.2, 0.8]).unwrap()[(2, 1)] == 0.8);
        assert_eq!(fiedler(&[1.0, 4.0, 2.0])[(1, 2)], 2.0);
    }
}
