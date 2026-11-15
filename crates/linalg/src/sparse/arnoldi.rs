//! A few eigenvalues of a large nonsymmetric operator (SciPy's `eigs`), by the Arnoldi process
//! with Krylov-Schur restarts: the wanted part of the Ritz subspace is kept, in a real orthonormal
//! basis, and the rest discarded, until the wanted Ritz pairs converge.

use super::direct::{Ordering, SparseLu};
use super::eigen::{ShiftInvert, start_vector};
use super::formats::CsrMatrix;
use super::iterative::LinearOperator;
use crate::error::{LinalgError, Result};
use crate::{SVD, schur_complex};
use ndarray::{Array2, s};
use num_complex::Complex;

type C = Complex<f64>;

/// Which eigenvalues [`eigs`] returns (SciPy's `which`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EigsWhich {
    /// Largest magnitude `|λ|` (`'LM'`, SciPy's default).
    LargestMagnitude,
    /// Smallest magnitude (`'SM'`); [`eigs_shift_invert`] with `σ = 0` is usually far faster.
    SmallestMagnitude,
    /// Largest real part (`'LR'`).
    LargestReal,
    /// Smallest real part (`'SR'`).
    SmallestReal,
    /// Largest imaginary part in magnitude, `|Im λ|` (`'LI'`; for a real matrix the eigenvalues
    /// come in conjugate pairs, and ARPACK compares `|Im λ|`).
    LargestImaginary,
    /// Smallest `|Im λ|` (`'SI'`).
    SmallestImaginary,
}

/// Options for [`eigs`] (SciPy's keyword arguments).
#[derive(Clone, Debug, Default)]
pub struct EigsOptions {
    /// The largest dimension of the Krylov subspace (`ncv`); default `max(2k + 1, 30)`, at most `n`
    /// (SciPy's is `max(2k + 1, 20)`; the larger space costs fewer products in the book's tests).
    pub ncv: Option<usize>,
    /// Relative accuracy of the Ritz residuals; 0 (the default) means machine precision, taken
    /// as 8ε, the level at which the residual estimates of converged pairs settle.
    pub tol: f64,
    /// The most restarts (`maxiter`); default `10 n`.
    pub max_restarts: Option<usize>,
    /// The start vector; default a fixed vector with all components non-zero.
    pub v0: Option<Vec<f64>>,
}

/// The result of [`eigs`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EigsResult {
    /// The eigenvalues, most wanted first; a real eigenvalue has a zero imaginary part.
    pub values: Vec<C>,
    /// Unit eigenvectors (columns), each scaled so that its largest component is real and
    /// positive, which makes the vectors of a conjugate pair conjugates and real eigenvalues'
    /// vectors real.
    pub vectors: Array2<C>,
    /// Whether all `k` pairs met the tolerance.
    pub converged: bool,
    /// How many of the `k` pairs met it.
    pub nconv: usize,
    /// Restarts made.
    pub restarts: usize,
    /// Products with the operator.
    pub matvecs: usize,
}

/// How strongly `which` prefers `z` (larger is preferred).
fn preference(which: EigsWhich, z: C) -> f64 {
    match which {
        EigsWhich::LargestMagnitude => z.norm(),
        EigsWhich::SmallestMagnitude => -z.norm(),
        EigsWhich::LargestReal => z.re,
        EigsWhich::SmallestReal => -z.re,
        EigsWhich::LargestImaginary => z.im.abs(),
        EigsWhich::SmallestImaginary => -z.im.abs(),
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `[c s; −s̄ c] [f; g] = [r; 0]` with real `c` (LAPACK's `zlartg`).
fn rotation(f: C, g: C) -> (f64, C) {
    if g == C::new(0.0, 0.0) {
        return (1.0, C::new(0.0, 0.0));
    }
    if f == C::new(0.0, 0.0) {
        return (0.0, g.conj() / g.norm());
    }
    let r = f.norm().hypot(g.norm());
    (f.norm() / r, f / f.norm() * g.conj() / r)
}

// ANCHOR: swap
/// Swaps the adjacent diagonal entries `k`, `k + 1` of the upper triangular `T` in
/// `H = Z T Zᴴ` by a plane rotation (LAPACK's `ztrexc`): the rotation that maps the eigenvector
/// `(t₁₂, t₂₂ − t₁₁)` of the 2 × 2 block onto the first axis makes the second eigenvalue lead.
fn swap(t: &mut Array2<C>, z: &mut Array2<C>, k: usize) {
    let m = t.nrows();
    let (t11, t22) = (t[(k, k)], t[(k + 1, k + 1)]);
    let (c, s) = rotation(t[(k, k + 1)], t22 - t11);
    for j in k + 2..m {
        let (x, y) = (t[(k, j)], t[(k + 1, j)]);
        t[(k, j)] = x * c + s * y;
        t[(k + 1, j)] = y * c - s.conj() * x;
    }
    for i in 0..k {
        let (x, y) = (t[(i, k)], t[(i, k + 1)]);
        t[(i, k)] = x * c + s.conj() * y;
        t[(i, k + 1)] = y * c - s * x;
    }
    t[(k, k)] = t22;
    t[(k + 1, k + 1)] = t11;
    for i in 0..m {
        let (x, y) = (z[(i, k)], z[(i, k + 1)]);
        z[(i, k)] = x * c + s.conj() * y;
        z[(i, k + 1)] = y * c - s * x;
    }
}
// ANCHOR_END: swap

/// Reorders the Schur form so that the eigenvalues come in order of preference, a conjugate pair
/// (whose two values a real matrix always has) kept together. Returns the group of each position:
/// equal numbers mark the two halves of a pair.
fn reorder(t: &mut Array2<C>, z: &mut Array2<C>, which: EigsWhich) -> Vec<usize> {
    let m = t.nrows();
    let vals: Vec<C> = (0..m).map(|i| t[(i, i)]).collect();
    let scale = vals.iter().fold(0.0f64, |acc, v| acc.max(v.norm())).max(f64::MIN_POSITIVE);
    let small = 1e3 * f64::EPSILON * scale;
    // Pair each value having positive imaginary part with the nearest conjugate.
    let mut group = vec![usize::MAX; m];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..m {
        if group[i] != usize::MAX {
            continue;
        }
        group[i] = groups.len();
        let mut members = vec![i];
        if vals[i].im.abs() > small {
            let partner = (0..m)
                .filter(|&j| group[j] == usize::MAX && vals[j].im * vals[i].im < 0.0)
                .min_by(|&x, &y| (vals[x] - vals[i].conj()).norm().total_cmp(&(vals[y] - vals[i].conj()).norm()));
            if let Some(j) = partner {
                group[j] = groups.len();
                members.push(j);
                members.sort_by(|&x, &y| vals[y].im.total_cmp(&vals[x].im));
            }
        }
        groups.push(members);
    }
    let key = |g: &Vec<usize>| g.iter().map(|&i| preference(which, vals[i])).fold(f64::NEG_INFINITY, f64::max);
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by(|&x, &y| key(&groups[y]).total_cmp(&key(&groups[x])));
    // Move the values into place one by one, tracking each position's original index.
    let mut label: Vec<usize> = (0..m).collect();
    let mut target = 0;
    for g in &order {
        for &orig in &groups[*g] {
            let mut j = label.iter().position(|&l| l == orig).expect("present");
            while j > target {
                swap(t, z, j - 1);
                label.swap(j - 1, j);
                j -= 1;
            }
            target += 1;
        }
    }
    label.iter().map(|&l| group[l]).collect()
}

/// The eigenvector of the upper triangular `T` for its `i`-th diagonal value, by back
/// substitution in the leading `(i + 1) × (i + 1)` block.
fn triangular_eigenvector(t: &Array2<C>, i: usize) -> Vec<C> {
    let lambda = t[(i, i)];
    let floor = f64::EPSILON * t.iter().fold(0.0f64, |acc, v| acc.max(v.norm())).max(f64::MIN_POSITIVE);
    let mut s = vec![C::new(0.0, 0.0); t.nrows()];
    s[i] = C::new(1.0, 0.0);
    for r in (0..i).rev() {
        let sum: C = (r + 1..=i).map(|c| t[(r, c)] * s[c]).sum();
        let mut d = t[(r, r)] - lambda;
        if d.norm() < floor {
            d = C::new(floor, 0.0);
        }
        s[r] = -sum / d;
    }
    s
}

// ANCHOR: eigs
/// `k` eigenvalues and eigenvectors of a square operator (SciPy's `eigs`), by the Krylov-Schur
/// method. The Arnoldi process builds an orthonormal basis `V` of a Krylov subspace with
/// `A V = V H + f eₘᵀ`; the eigenvalues of the small `H` (Ritz values) approximate those of `A`,
/// with residuals `‖f‖ |eₘᵀ y|` known without forming the vectors. When the subspace reaches
/// `ncv` vectors, the Schur form of `H` is reordered to put the wanted Ritz values first, and
/// the factorisation is cut back to their invariant subspace: a real orthonormal basis of it, so
/// that the work stays in real arithmetic. The Arnoldi process then extends it again.
pub fn eigs(a: &dyn LinearOperator, k: usize, which: EigsWhich, options: &EigsOptions) -> Result<EigsResult> {
    let (n, n2) = a.shape();
    if n != n2 || k == 0 || k > n {
        return Err(LinalgError::Shape(format!("need a square operator and 1 ≤ k ≤ n, got {n} × {n2}, k = {k}")));
    }
    let m = options.ncv.unwrap_or((2 * k + 1).max(30)).clamp((k + 1).min(n), n);
    // Residual estimates of converged pairs settle at a few ε|θ|, which may lie just above ε|θ|.
    let tol = options.tol.max(8.0 * f64::EPSILON);
    let max_restarts = options.max_restarts.unwrap_or(10 * n);
    let mut v0 = match &options.v0 {
        Some(v) if v.len() == n && v.iter().any(|x| *x != 0.0) => v.clone(),
        Some(_) => return Err(LinalgError::Shape(format!("v0 must be a non-zero vector of length {n}"))),
        None => start_vector(n, &[]),
    };
    let nv = dot(&v0, &v0).sqrt();
    v0.iter_mut().for_each(|x| *x /= nv);
    let mut basis: Vec<Vec<f64>> = vec![v0];
    // H̄ is (m + 1) × m: A V[:, :j] = V[:, :j+1] H̄[:j+1, :j] for the columns built so far.
    let mut hbar = Array2::<f64>::zeros((m + 1, m));
    let (mut kept, mut restarts, mut matvecs) = (0, 0, 0);
    loop {
        // Extend the Arnoldi factorisation from `kept` to `m` columns.
        for j in kept..m {
            let mut w = a.matvec(&basis[j]);
            matvecs += 1;
            let wnorm = dot(&w, &w).sqrt();
            // Classical Gram-Schmidt, twice (the second pass restores orthogonality).
            for _ in 0..2 {
                let c: Vec<f64> = basis.iter().map(|v| dot(&w, v)).collect();
                for (v, ci) in basis.iter().zip(&c) {
                    w.iter_mut().zip(v).for_each(|(wi, vi)| *wi -= ci * vi);
                }
                for (i, ci) in c.iter().enumerate() {
                    hbar[(i, j)] += ci;
                }
            }
            let beta = dot(&w, &w).sqrt();
            if beta > 1e3 * f64::EPSILON * wnorm.max(f64::MIN_POSITIVE) {
                hbar[(j + 1, j)] = beta;
                basis.push(w.iter().map(|x| x / beta).collect());
            } else {
                // An invariant subspace: continue with a new direction, or a zero vector if
                // the basis already spans the whole space.
                hbar[(j + 1, j)] = 0.0;
                basis.push(if basis.len() < n { start_vector(n, &basis) } else { vec![0.0; n] });
            }
        }
        let beta_m = hbar[(m, m - 1)];
        let h = hbar.slice(s![..m, ..]).to_owned();
        let (mut t, mut z) = schur_complex(&h)?;
        let group = reorder(&mut t, &mut z, which);
        let hnorm = h.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        let eps23 = f64::EPSILON.powf(2.0 / 3.0) * hnorm;
        let wanted = k.min(m);
        let ritz: Vec<Vec<C>> = (0..wanted).map(|i| {
            let s = triangular_eigenvector(&t, i);
            (0..m).map(|r| (0..=i).map(|c| z[(r, c)] * s[c]).sum()).collect()
        }).collect();
        let nconv = (0..wanted)
            .filter(|&i| {
                let y = &ritz[i];
                let ynorm = y.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                beta_m * y[m - 1].norm() / ynorm <= tol * t[(i, i)].norm().max(eps23)
            })
            .count();
        if nconv >= wanted || restarts >= max_restarts || m == n {
            return Ok(finish(&basis, &t, &group, &ritz, n, m, nconv, wanted, restarts, matvecs));
        }
        restarts += 1;
        // Keep the wanted values and half of the rest, without splitting a conjugate pair.
        let mut p = (k + (m - k) / 2).min(m - 1);
        if p < m && p > 0 && group[p] == group[p - 1] {
            if p + 1 < m { p += 1 } else { p -= 1 }
        }
        // A real orthonormal basis of the invariant subspace spanned by the leading p Schur
        // vectors: it is closed under conjugation, so [Re Z_p, Im Z_p] has rank p.
        let w = Array2::from_shape_fn((m, 2 * p), |(r, c)| if c < p { z[(r, c)].re } else { z[(r, c - p)].im });
        let (u, _, _) = w.svd(true, false)?;
        let u = u.expect("requested").slice(s![.., ..p]).to_owned();
        let small = u.t().dot(&h).dot(&u);
        let mut new_basis: Vec<Vec<f64>> = (0..p).map(|c| {
            let mut v = vec![0.0; n];
            for (j, bj) in basis.iter().take(m).enumerate() {
                let coef = u[(j, c)];
                v.iter_mut().zip(bj).for_each(|(vi, b)| *vi += coef * b);
            }
            v
        }).collect();
        new_basis.push(basis[m].clone());
        basis = new_basis;
        hbar.fill(0.0);
        hbar.slice_mut(s![..p, ..p]).assign(&small);
        for c in 0..p {
            hbar[(p, c)] = beta_m * u[(m - 1, c)];
        }
        kept = p;
    }
}
// ANCHOR_END: eigs

/// The Ritz pairs as eigenvalues and unit eigenvectors of `A`.
#[allow(clippy::too_many_arguments)]
fn finish(basis: &[Vec<f64>], t: &Array2<C>, group: &[usize], ritz: &[Vec<C>], n: usize, m: usize, nconv: usize, wanted: usize, restarts: usize, matvecs: usize) -> EigsResult {
    let mut values = Vec::with_capacity(wanted);
    let mut vectors = Array2::<C>::zeros((n, wanted));
    for i in 0..wanted {
        let paired = group.iter().filter(|&&g| g == group[i]).count() == 2;
        let lambda = t[(i, i)];
        values.push(if paired { lambda } else { C::new(lambda.re, 0.0) });
        let mut x = vec![C::new(0.0, 0.0); n];
        for (j, bj) in basis.iter().take(m).enumerate() {
            let y = ritz[i][j];
            x.iter_mut().zip(bj).for_each(|(xi, b)| *xi += y * b);
        }
        let big = x.iter().copied().max_by(|p, q| p.norm().total_cmp(&q.norm())).unwrap_or(C::new(1.0, 0.0));
        let norm = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let phase = if big.norm() > 0.0 { big.conj() / big.norm() } else { C::new(1.0, 0.0) };
        for (r, xi) in x.iter().enumerate() {
            let v = xi * phase / norm;
            vectors[(r, i)] = if paired { v } else { C::new(v.re, 0.0) };
        }
    }
    EigsResult { values, vectors, converged: nconv >= wanted, nconv, restarts, matvecs }
}

// ANCHOR: eigs_shift_invert
/// The `k` eigenvalues of a sparse matrix nearest the real shift `σ`, with eigenvectors
/// (SciPy's `eigs(A, k, sigma=σ)`). The eigenvalues `λ` near `σ` are the largest in magnitude of
/// `(A − σI)⁻¹`, `θ = 1/(λ − σ)`, which Arnoldi finds quickly; each product with the inverse is
/// one solve with a sparse LU factorisation of `A − σI`. The values are returned nearest first.
pub fn eigs_shift_invert(a: &CsrMatrix, k: usize, sigma: f64, options: &EigsOptions) -> Result<EigsResult> {
    let n = a.nrows;
    if a.ncols != n {
        return Err(LinalgError::NotSquare { rows: a.nrows, cols: a.ncols });
    }
    let shifted = a.add(1.0, &CsrMatrix::identity(n), -sigma)?;
    let op = ShiftInvert { lu: SparseLu::factor(&shifted, Ordering::MinimumDegree)?, n };
    let mut r = eigs(&op, k, EigsWhich::LargestMagnitude, options)?;
    for v in r.values.iter_mut() {
        let lambda = C::new(sigma, 0.0) + C::new(1.0, 0.0) / *v;
        *v = if v.im == 0.0 { C::new(lambda.re, 0.0) } else { lambda };
    }
    Ok(r)
}
// ANCHOR_END: eigs_shift_invert

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Eig;
    use crate::sparse::CooMatrix;

    /// A pseudo-random sparse matrix with complex eigenvalues: four off-diagonal entries per row.
    fn scattered(n: usize) -> CsrMatrix {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, (i as f64 * 0.37).sin());
            for t in 1..=4 {
                let j = (i * 37 + 11 * t + 7 * t * t) % n;
                coo.push(i, j, (i as f64 + 3.0 * t as f64 + 1.0).sin());
            }
        }
        coo.to_csr().unwrap()
    }

    /// The 5-point convection-diffusion operator on an `m × m` grid; complex eigenvalues when `γ > 1`.
    fn convection(m: usize, gamma: f64) -> CsrMatrix {
        let mut coo = CooMatrix::new(m * m, m * m);
        for i in 0..m {
            for j in 0..m {
                let r = i * m + j;
                coo.push(r, r, 4.0);
                if i > 0 { coo.push(r, r - m, -1.0 - gamma); }
                if i + 1 < m { coo.push(r, r + m, -1.0 + gamma); }
                if j > 0 { coo.push(r, r - 1, -1.0 - gamma); }
                if j + 1 < m { coo.push(r, r + 1, -1.0 + gamma); }
            }
        }
        coo.to_csr().unwrap()
    }

    fn check(a: &CsrMatrix, k: usize, which: EigsWhich) {
        let r = eigs(a, k, which, &EigsOptions::default()).unwrap();
        assert!(r.converged, "{which:?}: {} of {k} after {} restarts", r.nconv, r.restarts);
        let (all, _) = a.to_dense().eig().unwrap();
        let mut all: Vec<C> = all.to_vec();
        all.sort_by(|x, y| preference(which, *y).total_cmp(&preference(which, *x)));
        let scale = all.iter().fold(0.0f64, |acc, v| acc.max(v.norm()));
        // Each value found is an eigenvalue, and the k-th most wanted is matched in preference.
        for (i, v) in r.values.iter().enumerate() {
            let nearest = all.iter().map(|w| (w - v).norm()).fold(f64::INFINITY, f64::min);
            assert!(nearest < 1e-10 * scale, "{which:?} value {v} is {nearest:e} from the spectrum");
            assert!((preference(which, *v) - preference(which, all[i])).abs() < 1e-10 * scale, "{which:?}: {:?} vs {:?}", r.values, &all[..k]);
            let x: Vec<C> = r.vectors.column(i).to_vec();
            let re: Vec<f64> = x.iter().map(|z| z.re).collect();
            let im: Vec<f64> = x.iter().map(|z| z.im).collect();
            let (are, aim) = (a.matvec(&re), a.matvec(&im));
            let res = (0..x.len()).map(|j| (C::new(are[j], aim[j]) - v * x[j]).norm_sqr()).sum::<f64>().sqrt();
            assert!(res < 1e-9 * scale, "{which:?} residual {res:e}");
        }
    }

    #[test]
    fn every_which_on_a_scattered_matrix() {
        let a = scattered(300);
        for which in [EigsWhich::LargestMagnitude, EigsWhich::LargestReal, EigsWhich::SmallestReal, EigsWhich::LargestImaginary] {
            check(&a, 6, which);
        }
    }

    /// A diagonal ramp plus a skew-symmetric tridiagonal part: nearly normal, with complex
    /// eigenvalues spread along the real axis.
    fn ramp(n: usize) -> CsrMatrix {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0 * i as f64 / n as f64);
            if i + 1 < n {
                coo.push(i, i + 1, 1.0);
                coo.push(i + 1, i, -1.0);
            }
        }
        coo.to_csr().unwrap()
    }

    #[test]
    fn convection_diffusion_and_a_complex_spectrum() {
        check(&convection(15, 0.5), 5, EigsWhich::LargestMagnitude);
        check(&convection(15, 0.5), 4, EigsWhich::SmallestReal);
        let a = ramp(400);
        for which in [EigsWhich::LargestReal, EigsWhich::LargestImaginary, EigsWhich::SmallestReal] {
            check(&a, 6, which);
        }
    }

    #[test]
    fn shift_invert_and_small_problems() {
        let a = scattered(300);
        let sigma = 0.3;
        let r = eigs_shift_invert(&a, 5, sigma, &EigsOptions::default()).unwrap();
        assert!(r.converged);
        let (all, _) = a.to_dense().eig().unwrap();
        let mut d: Vec<f64> = all.iter().map(|w| (w - sigma).norm()).collect();
        d.sort_by(f64::total_cmp);
        for (v, want) in r.values.iter().zip(&d) {
            assert!(((v - sigma).norm() - want).abs() < 1e-9, "{:?}", r.values);
        }
        // k = n, and an operator smaller than the default subspace.
        let small = scattered(7);
        check(&small, 7, EigsWhich::LargestMagnitude);
        check(&small, 3, EigsWhich::SmallestMagnitude);
        assert!(eigs(&small, 0, EigsWhich::LargestMagnitude, &EigsOptions::default()).is_err());
    }
}
