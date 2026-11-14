//! Partial fractions of rational transfer functions and related polynomial tools, ported from
//! `scipy.signal`: `residue`/`invres` (positive powers of `s` or `z`), `residuez`/`invresz`
//! (negative powers of `z`), `unique_roots`, `deconvolve`, and the zero-pole-gain conversions
//! `tf2zpk`/`zpk2tf`.
//!
//! See the book entry [Partial fractions, zoom FFT and waveforms](https://jwkunz.github.io/lib_jsl/book/spectral/partial-fractions-waveforms.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::EigVals;
use ndarray::Array2;
use num_complex::Complex;

type C = Complex<f64>;

/// How [`unique_roots`] represents a group of nearly equal roots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootType {
    /// The mean of the group (SciPy's `'avg'`, the default of `residue`).
    Average,
    /// The smallest, comparing real parts then imaginary parts.
    Min,
    /// The largest.
    Max,
}

fn reduce(group: &[C], rtype: RootType) -> C {
    let cmp = |a: &&C, b: &&C| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im));
    match rtype {
        RootType::Average => group.iter().sum::<C>() / group.len() as f64,
        RootType::Min => *group.iter().min_by(cmp).expect("non-empty"),
        RootType::Max => *group.iter().max_by(cmp).expect("non-empty"),
    }
}

/// The roots of a polynomial, highest power first (NumPy's `roots`): leading zeros are dropped,
/// trailing zeros give roots at 0, and the rest are the eigenvalues of the companion matrix.
/// Complex-conjugate pairs are listed with the positive imaginary part first, as LAPACK does.
pub fn roots(p: &[f64]) -> Result<Vec<C>, ErrorsJSL> {
    roots_complex(&p.iter().map(|&v| C::new(v, 0.0)).collect::<Vec<_>>())
}

fn roots_complex(p: &[C]) -> Result<Vec<C>, ErrorsJSL> {
    let nz: Vec<usize> = (0..p.len()).filter(|&i| p[i] != C::new(0.0, 0.0)).collect();
    let (Some(&first), Some(&last)) = (nz.first(), nz.last()) else { return Ok(Vec::new()) };
    let trailing = p.len() - 1 - last;
    let q = &p[first..=last];
    let n = q.len() - 1;
    let mut out = Vec::with_capacity(n + trailing);
    if n > 0 {
        let real = q.iter().all(|c| c.im == 0.0);
        let mut ev: Vec<C> = if real {
            // A real companion matrix: the real Schur form gives exact conjugate pairs, as LAPACK.
            let comp = Array2::from_shape_fn((n, n), |(r, c)| if r == 0 { -q[c + 1].re / q[0].re } else if r == c + 1 { 1.0 } else { 0.0 });
            comp.eigvals().map_err(|_| ErrorsJSL::RuntimeError("the eigenvalue iteration failed"))?.to_vec()
        } else {
            let comp = Array2::from_shape_fn((n, n), |(r, c)| if r == 0 { -q[c + 1] / q[0] } else if r == c + 1 { C::new(1.0, 0.0) } else { C::new(0.0, 0.0) });
            comp.eigvals().map_err(|_| ErrorsJSL::RuntimeError("the eigenvalue iteration failed"))?.to_vec()
        };
        if real {
            // Make conjugate pairs exact and put the positive imaginary part first.
            for i in 0..ev.len() {
                if ev[i].im < 0.0 {
                    let target = ev[i].conj();
                    let tol = 1e-8 * ev[i].norm().max(1.0);
                    if let Some(j) = (i + 1..ev.len()).find(|&j| (ev[j] - target).norm() <= tol) {
                        let z = ev[j];
                        ev[i] = z;
                        ev[j] = z.conj();
                    }
                }
            }
        }
        out.extend(ev);
    }
    out.extend(std::iter::repeat_n(C::new(0.0, 0.0), trailing));
    Ok(out)
}

// ANCHOR: unique_roots
/// Groups nearly equal roots (SciPy's `unique_roots`): going through `p` in order, each root not
/// yet used takes every unused root within distance `tol` of it into its group; the group is
/// represented by its mean, minimum or maximum. Returns the representatives and multiplicities.
pub fn unique_roots(p: &[C], tol: f64, rtype: RootType) -> (Vec<C>, Vec<usize>) {
    let mut used = vec![false; p.len()];
    let (mut unique, mut mult) = (Vec::new(), Vec::new());
    for i in 0..p.len() {
        if used[i] {
            continue;
        }
        let group: Vec<usize> = (0..p.len()).filter(|&j| !used[j] && (p[j] - p[i]).norm() <= tol).collect();
        let values: Vec<C> = group.iter().map(|&j| p[j]).collect();
        unique.push(reduce(&values, rtype));
        mult.push(group.len());
        for j in group {
            used[j] = true;
        }
    }
    (unique, mult)
}
// ANCHOR_END: unique_roots

// --------------------------------------------------------------- polynomial helpers (NumPy's)

fn polymul(a: &[C], b: &[C]) -> Vec<C> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![C::new(0.0, 0.0); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

fn polyadd(a: &[C], b: &[C]) -> Vec<C> {
    let n = a.len().max(b.len());
    let mut out = vec![C::new(0.0, 0.0); n];
    for (i, x) in a.iter().enumerate() {
        out[n - a.len() + i] += x;
    }
    for (i, x) in b.iter().enumerate() {
        out[n - b.len() + i] += x;
    }
    out
}

fn polyval(p: &[C], x: C) -> C {
    p.iter().fold(C::new(0.0, 0.0), |acc, &c| acc * x + c)
}

/// NumPy's `polydiv`: quotient and remainder, with leading remainder terms within an absolute
/// `10⁻⁸` of zero stripped (NumPy's `allclose(r[0], 0, rtol=1e-14)`).
fn polydiv(u: &[C], v: &[C]) -> (Vec<C>, Vec<C>) {
    let m = u.len() - 1;
    let n = v.len() - 1;
    let scale = C::new(1.0, 0.0) / v[0];
    let mut q = vec![C::new(0.0, 0.0); (m + 1).saturating_sub(n).max(1)];
    let mut r = u.to_vec();
    if m >= n {
        for k in 0..=m - n {
            let d = scale * r[k];
            q[k] = d;
            for (i, vi) in v.iter().enumerate() {
                r[k + i] -= d * vi;
            }
        }
    }
    let mut start = 0;
    while start + 1 < r.len() && r[start].norm() <= 1e-8 {
        start += 1;
    }
    (q, r[start..].to_vec())
}

fn to_c(p: &[f64]) -> Vec<C> {
    p.iter().map(|&v| C::new(v, 0.0)).collect()
}

fn trim_front(p: &[f64]) -> Vec<f64> {
    p.iter().copied().skip_while(|&v| v == 0.0).collect()
}

fn trim_back(p: &[f64]) -> Vec<f64> {
    let mut v = p.to_vec();
    while v.last() == Some(&0.0) {
        v.pop();
    }
    v
}

/// Sorts by magnitude, stably (SciPy's `_cmplx_sort`).
fn cmplx_sort(p: &[C]) -> (Vec<C>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[a].norm().total_cmp(&p[b].norm()));
    (idx.iter().map(|&i| p[i]).collect(), idx)
}

/// The total product divided by each root's factor (SciPy's `_compute_factors`); with
/// `include_powers` one entry per power of each repeated root.
fn compute_factors(roots: &[C], mult: &[usize], include_powers: bool) -> (Vec<Vec<C>>, Vec<C>) {
    let one = vec![C::new(1.0, 0.0)];
    let mut current = one.clone();
    let mut suffixes = vec![current.clone()];
    for (pole, &m) in roots.iter().zip(mult).skip(1).rev() {
        let mono = vec![C::new(1.0, 0.0), -pole];
        for _ in 0..m {
            current = polymul(&current, &mono);
        }
        suffixes.push(current.clone());
    }
    suffixes.reverse();
    let mut factors = Vec::new();
    let mut current = one;
    for ((pole, &m), suffix) in roots.iter().zip(mult).zip(&suffixes) {
        let mono = vec![C::new(1.0, 0.0), -pole];
        let mut block = Vec::new();
        for i in 0..m {
            if i == 0 || include_powers {
                block.push(polymul(&current, suffix));
            }
            current = polymul(&current, &mono);
        }
        block.reverse();
        factors.extend(block);
    }
    (factors, current)
}

// ANCHOR: compute_residues
/// The residues for poles of given multiplicity (SciPy's `_compute_residues`, "deflation through
/// subtraction", method 6 of Mahoney and Sivazlian): a simple pole's residue is
/// `b(p)/∏_{q≠p}(p − q)`; for a repeated pole, divide numerator and the cofactor by `(x − p)`
/// repeatedly, peeling off one residue per power and subtracting its contribution.
fn compute_residues(poles: &[C], mult: &[usize], numerator: &[C]) -> Vec<C> {
    let (factors, _) = compute_factors(poles, mult, false);
    let mut residues = Vec::new();
    for ((pole, &m), factor) in poles.iter().zip(mult).zip(&factors) {
        if m == 1 {
            residues.push(polyval(numerator, *pole) / polyval(factor, *pole));
        } else {
            let mono = vec![C::new(1.0, 0.0), -pole];
            let (factor, d) = polydiv(factor, &mono);
            let mut numer = numerator.to_vec();
            let mut block = Vec::new();
            for _ in 0..m {
                let (q, n) = polydiv(&numer, &mono);
                let r = n[0] / d[0];
                numer = polyadd(&q, &factor.iter().map(|f| -r * f).collect::<Vec<_>>());
                block.push(r);
            }
            block.reverse();
            residues.extend(block);
        }
    }
    residues
}
// ANCHOR_END: compute_residues

/// The partial-fraction expansion: residues `r`, poles `p` (repeated poles adjacent, ordered by
/// magnitude) and the direct polynomial term `k` (highest power first for `residue`, increasing
/// powers of `z⁻¹` for `residuez`).
#[derive(Clone, Debug, PartialEq)]
pub struct PartialFractions {
    pub r: Vec<C>,
    pub p: Vec<C>,
    pub k: Vec<f64>,
}

// ANCHOR: residue
/// The partial-fraction expansion of `b(s)/a(s)` in positive powers (SciPy's `residue`):
/// `Σ rᵢ/(s − pᵢ) + k(s)`, with terms `r/(s − p)^j` for poles repeated within `tol`.
pub fn residue(b: &[f64], a: &[f64], tol: f64, rtype: RootType) -> Result<PartialFractions, ErrorsJSL> {
    let b = trim_front(b);
    let a = trim_front(a);
    if a.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("the denominator is zero"));
    }
    let mut poles = roots(&a)?;
    if b.is_empty() {
        return Ok(PartialFractions { r: vec![C::new(0.0, 0.0); poles.len()], p: cmplx_sort(&poles).0, k: Vec::new() });
    }
    let (k, bnum) = if b.len() < a.len() { (Vec::new(), to_c(&b)) } else { polydiv(&to_c(&b), &to_c(&a)) };
    let (unique, mult) = unique_roots(&poles, tol, rtype);
    let (unique, order) = cmplx_sort(&unique);
    let mult: Vec<usize> = order.iter().map(|&i| mult[i]).collect();
    let residues = compute_residues(&unique, &mult, &bnum);
    let mut index = 0;
    for (pole, &m) in unique.iter().zip(&mult) {
        for slot in poles.iter_mut().skip(index).take(m) {
            *slot = *pole;
        }
        index += m;
    }
    Ok(PartialFractions { r: residues.iter().map(|r| r / a[0]).collect(), p: poles, k: k.iter().map(|c| c.re).collect() })
}
// ANCHOR_END: residue

/// The partial-fraction expansion of `b(z)/a(z)` in negative powers of `z` (SciPy's `residuez`):
/// `Σ rᵢ/(1 − pᵢz⁻¹) + k₀ + k₁z⁻¹ + …`.
pub fn residuez(b: &[f64], a: &[f64], tol: f64, rtype: RootType) -> Result<PartialFractions, ErrorsJSL> {
    let b = trim_back(b);
    let a = trim_back(a);
    if a.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("the denominator is zero"));
    }
    if a[0] == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("the first denominator coefficient must be non-zero"));
    }
    let mut poles = roots(&a)?;
    if b.is_empty() {
        return Ok(PartialFractions { r: vec![C::new(0.0, 0.0); poles.len()], p: cmplx_sort(&poles).0, k: Vec::new() });
    }
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    let a_rev: Vec<f64> = a.iter().rev().copied().collect();
    let (k_rev, b_rev) = if b_rev.len() < a_rev.len() { (Vec::new(), to_c(&b_rev)) } else { polydiv(&to_c(&b_rev), &to_c(&a_rev)) };
    let (unique, mult) = unique_roots(&poles, tol, rtype);
    let (unique, order) = cmplx_sort(&unique);
    let mult: Vec<usize> = order.iter().map(|&i| mult[i]).collect();
    let inv: Vec<C> = unique.iter().map(|p| C::new(1.0, 0.0) / p).collect();
    let mut residues = compute_residues(&inv, &mult, &b_rev);
    let mut index = 0;
    let mut powers = vec![0i32; residues.len()];
    for (pole, &m) in unique.iter().zip(&mult) {
        for j in 0..m {
            poles[index + j] = *pole;
            powers[index + j] = 1 + j as i32;
        }
        index += m;
    }
    for (i, r) in residues.iter_mut().enumerate() {
        *r *= (-poles[i]).powi(powers[i]) / a_rev[0];
    }
    Ok(PartialFractions { r: residues, p: poles, k: k_rev.iter().rev().map(|c| c.re).collect() })
}

/// Groups consecutive poles within `tol` of the first of the group (SciPy's `_group_poles`).
fn group_poles(p: &[C], tol: f64, rtype: RootType) -> (Vec<C>, Vec<usize>) {
    let (mut unique, mut mult) = (Vec::new(), Vec::new());
    let mut pole = p[0];
    let mut block = vec![pole];
    for &q in &p[1..] {
        if (q - pole).norm() <= tol {
            block.push(pole);
        } else {
            unique.push(reduce(&block, rtype));
            mult.push(block.len());
            pole = q;
            block = vec![pole];
        }
    }
    unique.push(reduce(&block, rtype));
    mult.push(block.len());
    (unique, mult)
}

/// `b(s)` and `a(s)` from a partial-fraction expansion (SciPy's `invres`); repeated poles must
/// be adjacent. Complex results whose imaginary parts cancel to rounding are returned as real
/// parts together with the largest imaginary part dropped.
pub fn invres(r: &[C], p: &[C], k: &[f64], tol: f64, rtype: RootType) -> (Vec<C>, Vec<C>) {
    let k = trim_front(k);
    let (unique, mult) = group_poles(p, tol, rtype);
    let (factors, denominator) = compute_factors(&unique, &mult, true);
    let mut numerator = if k.is_empty() { vec![C::new(0.0, 0.0)] } else { polymul(&to_c(&k), &denominator) };
    for (res, factor) in r.iter().zip(&factors) {
        numerator = polyadd(&numerator, &factor.iter().map(|f| res * f).collect::<Vec<_>>());
    }
    (numerator, denominator)
}

/// `b(z)` and `a(z)` in negative powers of `z` from a partial-fraction expansion (SciPy's
/// `invresz`).
pub fn invresz(r: &[C], p: &[C], k: &[f64], tol: f64, rtype: RootType) -> (Vec<C>, Vec<C>) {
    let k = trim_back(k);
    let (unique, mult) = group_poles(p, tol, rtype);
    let (factors, denominator) = compute_factors(&unique, &mult, true);
    let rev = |v: &[C]| v.iter().rev().copied().collect::<Vec<_>>();
    let mut numerator = if k.is_empty() { vec![C::new(0.0, 0.0)] } else { polymul(&rev(&to_c(&k)), &rev(&denominator)) };
    for (res, factor) in r.iter().zip(&factors) {
        numerator = polyadd(&numerator, &rev(factor).iter().map(|f| res * f).collect::<Vec<_>>());
    }
    (rev(&numerator), denominator)
}

// ANCHOR: deconvolve
/// Inverse filtering (SciPy's `deconvolve`): the quotient `q` and remainder `r` with
/// `signal = divisor ∗ q + r`, `q` being the impulse response of `signal(z)/divisor(z)` for
/// `len(signal) − len(divisor) + 1` samples. It is polynomial division, so it is unstable when the
/// divisor has zeros outside the unit circle.
pub fn deconvolve(signal: &[f64], divisor: &[f64]) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
    let (n, d) = (signal.len(), divisor.len());
    if n == 0 || d == 0 {
        return Err(ErrorsJSL::InvalidInputRange("signal and divisor must be non-empty"));
    }
    if divisor[0] == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("the divisor's first coefficient must be non-zero"));
    }
    if d > n {
        return Ok((Vec::new(), signal.to_vec()));
    }
    let len = n - d + 1;
    let mut q = vec![0.0; len];
    for i in 0..len {
        let s: f64 = (1..d.min(i + 1)).map(|k| divisor[k] * q[i - k]).sum();
        q[i] = (signal[i] - s) / divisor[0];
    }
    let mut rem = signal.to_vec();
    for (i, qi) in q.iter().enumerate() {
        for (k, dk) in divisor.iter().enumerate() {
            rem[i + k] -= dk * qi;
        }
    }
    Ok((q, rem))
}
// ANCHOR_END: deconvolve

/// Zeros, poles and gain of `b(s)/a(s)` (SciPy's `tf2zpk`): both are normalised by the leading
/// denominator coefficient, the gain is the leading numerator coefficient, and the zeros and poles
/// are the roots of the normalised polynomials.
pub fn tf2zpk(b: &[f64], a: &[f64]) -> Result<(Vec<C>, Vec<C>, f64), ErrorsJSL> {
    let a = trim_front(a);
    if a.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("the denominator is zero"));
    }
    let b: Vec<f64> = trim_front(b).iter().map(|v| v / a[0]).collect();
    let a: Vec<f64> = a.iter().map(|v| v / a[0]).collect();
    let k = b.first().copied().unwrap_or(0.0);
    let bn: Vec<f64> = if k == 0.0 { Vec::new() } else { b.iter().map(|v| v / k).collect() };
    Ok((roots(&bn)?, roots(&a)?, k))
}

/// `b(s)` and `a(s)` from zeros, poles and gain (SciPy's `zpk2tf`), real when the zeros and
/// poles come in conjugate pairs.
pub fn zpk2tf(z: &[C], p: &[C], k: f64) -> (Vec<f64>, Vec<f64>) {
    let b = crate::continuous::lti::poly_from_roots(z).iter().map(|v| v * k).collect();
    (b, crate::continuous::lti::poly_from_roots(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: C, b: C, tol: f64) -> bool {
        (a - b).norm() <= tol * b.norm().max(1.0)
    }

    #[test]
    fn residue_of_a_simple_rational_function() {
        // (s + 2)/((s + 1)(s + 3)) = 0.5/(s + 1) + 0.5/(s + 3).
        let pf = residue(&[1.0, 2.0], &[1.0, 4.0, 3.0], 1e-3, RootType::Average).unwrap();
        assert!(pf.k.is_empty());
        for (r, p) in pf.r.iter().zip(&pf.p) {
            assert!(close(*r, C::new(0.5, 0.0), 1e-14));
            assert!(close(*p, C::new(-1.0, 0.0), 1e-14) || close(*p, C::new(-3.0, 0.0), 1e-14));
        }
        // A double pole: 1/(s + 1)² has r = [0, 1] for powers 1 and 2.
        let pf = residue(&[1.0], &[1.0, 2.0, 1.0], 1e-3, RootType::Average).unwrap();
        assert!(close(pf.r[0], C::new(0.0, 0.0), 1e-7) && close(pf.r[1], C::new(1.0, 0.0), 1e-7), "{:?}", pf.r);
        let (b, a) = invres(&pf.r, &pf.p, &pf.k, 1e-3, RootType::Average);
        assert!(close(b[b.len() - 1], C::new(1.0, 0.0), 1e-7) && close(a[1], C::new(2.0, 0.0), 1e-7));
    }

    #[test]
    fn residuez_round_trips() {
        let b = [1.0, -0.5, 0.25];
        let a = [1.0, -0.9, 0.2];
        let pf = residuez(&b, &a, 1e-3, RootType::Average).unwrap();
        let (b2, a2) = invresz(&pf.r, &pf.p, &pf.k, 1e-3, RootType::Average);
        for (x, y) in b2.iter().zip(&b) {
            assert!(close(*x, C::new(*y, 0.0), 1e-12), "{b2:?}");
        }
        for (x, y) in a2.iter().zip(&a) {
            assert!(close(*x, C::new(*y, 0.0), 1e-12), "{a2:?}");
        }
    }

    #[test]
    fn deconvolve_recovers_the_signal() {
        let (q, r) = deconvolve(&[0.0, 2.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 0.0], &[2.0, 1.0]).unwrap();
        assert_eq!(q, vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        assert!(r.iter().all(|v| v.abs() < 1e-15));
    }
}
