//! The cubic smoothing spline (SciPy's `make_smoothing_spline`): the function `g` minimising
//!
//! `Σ wᵢ (yᵢ − g(xᵢ))² + λ ∫ g''(x)² dx`,
//!
//! which is a natural cubic spline with knots at the data. Large `λ` approaches the straight-line
//! least-squares fit, `λ → 0` the interpolating natural spline. When `λ` is not given, it is
//! chosen by generalised cross-validation (GCV).
//!
//! See the book entry [B-splines and smoothing splines](https://jwkunz.github.io/lib_jsl/book/interpolation/b-splines.html#smoothing-splines).

use lib_jsl_core::ErrorsJSL;

use super::b_spline::{BSpline, nonzero_basis, nonzero_basis_derivatives};

/// The result of [`make_smoothing_spline`]: the spline and the smoothing parameter used.
#[derive(Clone, Debug)]
pub struct SmoothingSpline {
    pub spline: BSpline,
    /// `λ`, as given or as chosen by GCV.
    pub lam: f64,
    /// The trace of the influence matrix `A(λ)` (`ŷ = A y`): the effective number of parameters.
    pub degrees_of_freedom: f64,
}

/// A symmetric positive-definite banded matrix with half-bandwidth 3, row `i` storing the
/// entries `(i, i), (i, i+1), (i, i+2), (i, i+3)`.
#[derive(Clone)]
struct Band4 {
    d: Vec<[f64; 4]>,
}

impl Band4 {
    fn zeros(n: usize) -> Self {
        Band4 { d: vec![[0.0; 4]; n] }
    }
    fn add(&mut self, i: usize, j: usize, v: f64) {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        self.d[a][b - a] += v;
    }
    fn get(&self, i: usize, j: usize) -> f64 {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        if b - a > 3 { 0.0 } else { self.d[a][b - a] }
    }
    fn lincomb(&self, lam: f64, other: &Band4) -> Band4 {
        Band4 { d: self.d.iter().zip(&other.d).map(|(a, b)| std::array::from_fn(|k| a[k] + lam * b[k])).collect() }
    }

    // ANCHOR: ldl
    /// `LDLᵀ` factorisation: returns the unit lower factor (`l[i][k]` = `L[i+k+1][i]`) and `D`.
    fn ldl(&self) -> Result<(Vec<[f64; 3]>, Vec<f64>), ErrorsJSL> {
        let n = self.d.len();
        let mut l = vec![[0.0f64; 3]; n];
        let mut dd = vec![0.0; n];
        for j in 0..n {
            let mut djj = self.get(j, j);
            for k in j.saturating_sub(3)..j {
                djj -= l[k][j - k - 1].powi(2) * dd[k];
            }
            if djj <= 0.0 || !djj.is_finite() {
                return Err(ErrorsJSL::RuntimeError("the smoothing system is not positive definite"));
            }
            dd[j] = djj;
            for i in j + 1..(j + 4).min(n) {
                let mut v = self.get(i, j);
                for k in i.saturating_sub(3)..j {
                    v -= l[k][i - k - 1] * l[k][j - k - 1] * dd[k];
                }
                l[j][i - j - 1] = v / djj;
            }
        }
        Ok((l, dd))
    }
    // ANCHOR_END: ldl
}

fn ldl_solve(l: &[[f64; 3]], d: &[f64], b: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut x = b.to_vec();
    for i in 0..n {
        for k in i.saturating_sub(3)..i {
            x[i] -= l[k][i - k - 1] * x[k];
        }
    }
    for i in 0..n {
        x[i] /= d[i];
    }
    for i in (0..n).rev() {
        for k in i + 1..(i + 4).min(n) {
            x[i] -= l[i][k - i - 1] * x[k];
        }
    }
    x
}

// ANCHOR: band_inverse
/// The entries of `M⁻¹` within the band, from `M = L D Lᵀ`, by the recurrence of Hutchinson and
/// de Hoog (1985): `Z = D⁻¹ L⁻¹ + (I − Lᵀ) Z`, run from the last row upwards, needs only entries of
/// `Z` that are themselves in the band.
fn band_of_inverse(l: &[[f64; 3]], d: &[f64]) -> Band4 {
    let n = d.len();
    let mut z = Band4::zeros(n);
    for i in (0..n).rev() {
        let reach = (i + 4).min(n);
        for j in (i..reach).rev() {
            let mut s = if i == j { 1.0 / d[i] } else { 0.0 };
            for k in i + 1..reach {
                s -= l[i][k - i - 1] * z.get(k, j);
            }
            z.d[i][j - i] = s;
        }
    }
    z
}
// ANCHOR_END: band_inverse

/// Bounded Brent minimisation, step for step as SciPy's `minimize_scalar(method='bounded')`
/// (`fminbound`) with `xatol = 1e-5`.
fn fminbound(f: impl Fn(f64) -> f64, x1: f64, x2: f64) -> f64 {
    let xatol = 1e-5;
    let sqrt_eps = 2.2e-16f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let (mut rat, mut e) = (0.0f64, 0.0f64);
    let mut fx = f(xf);
    let mut num = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    let sign = |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };
    while (xf - xm).abs() > tol2 - 0.5 * (b - a) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        let x = xf + si * rat.abs().max(tol1);
        let fu = f(x);
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            (fulc, ffulc) = (nfc, fnfc);
            (nfc, fnfc) = (xf, fx);
            (xf, fx) = (x, fu);
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                (fulc, ffulc) = (nfc, fnfc);
                (nfc, fnfc) = (x, fu);
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                (fulc, ffulc) = (x, fu);
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= 500 {
            break;
        }
    }
    xf
}

// ANCHOR: smoothing
/// The cubic smoothing spline of `(x, y)` with weights `w` (default 1) and smoothing parameter
/// `lam` (SciPy's `make_smoothing_spline`). Without `lam`, the parameter minimising the GCV
/// score `n ‖y − ŷ‖² / (n − tr A(λ))²` over `0 < λ < n` is used, found by the same bounded Brent
/// search as SciPy.
///
/// In the cubic B-spline basis with knots `x` (ends repeated four times), the coefficients solve
/// `(Bᵀ W B + λ Ω) c = Bᵀ W y`, where `Bᵢⱼ = Bⱼ(xᵢ)` and `Ωᵢⱼ = ∫ Bᵢ'' Bⱼ''`, restricted to natural
/// splines by eliminating the two end coefficients. The matrices have half-bandwidth 3, so each
/// solve costs `O(n)`, as does the trace of the influence matrix, from the band of the inverse.
pub fn make_smoothing_spline(x: &[f64], y: &[f64], w: Option<&[f64]>, lam: Option<f64>) -> Result<SmoothingSpline, ErrorsJSL> {
    let n = x.len();
    if y.len() != n {
        return Err(ErrorsJSL::IncompatibleArraySizes((y.len(), n)));
    }
    if n < 5 {
        return Err(ErrorsJSL::InvalidInputRange("need at least five points"));
    }
    if x.windows(2).any(|p| p[1] <= p[0]) || x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("x must be strictly increasing and the data finite"));
    }
    let ones = vec![1.0; n];
    let w = w.unwrap_or(&ones);
    if w.len() != n || w.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("weights must be positive, one per point"));
    }
    if let Some(l) = lam
        && (l < 0.0 || !l.is_finite())
    {
        return Err(ErrorsJSL::InvalidInputRange("the smoothing parameter must be non-negative"));
    }
    let mut t = vec![x[0]; 3];
    t.extend_from_slice(x);
    t.extend([x[n - 1]; 3]);
    let m = n + 2;
    // Bᵀ W B and Bᵀ W y: data point i lies in knot interval l = i + 3 (the last in l = n + 1).
    let mut btwb = Band4::zeros(m);
    let mut btwy = vec![0.0; m];
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let l = (i + 3).min(n + 1);
        let b = nonzero_basis(&t, 3, l, x[i]);
        for r in 0..4 {
            btwy[l - 3 + r] += w[i] * b[r] * y[i];
            for s in r..4 {
                btwb.add(l - 3 + r, l - 3 + s, w[i] * b[r] * b[s]);
            }
        }
        rows.push((l - 3, b));
    }
    // Ω: B'' is linear on each interval, so two-point Gauss-Legendre integrates the products exactly.
    let mut omega = Band4::zeros(m);
    let g = 1.0 / 3f64.sqrt();
    for i in 0..n - 1 {
        let (a, b) = (x[i], x[i + 1]);
        let l = i + 3;
        for s in [-g, g] {
            let u = 0.5 * (a + b) + 0.5 * s * (b - a);
            let d2 = nonzero_basis_derivatives(&t, 3, l, u, 2);
            for r in 0..4 {
                for q in r..4 {
                    omega.add(l - 3 + r, l - 3 + q, d2[r] * d2[q] * 0.5 * (b - a));
                }
            }
        }
    }
    // The minimiser is a natural spline, s'' = 0 at both ends. Eliminating c₀ and c_{m−1} with
    // these two conditions, c = T a, leaves n unknowns a = (c₁, …, c_{m−2}) and a system that is
    // positive definite even for λ = 0.
    let d0 = nonzero_basis_derivatives(&t, 3, 3, x[0], 2);
    let dn = nonzero_basis_derivatives(&t, 3, n + 1, x[n - 1], 2);
    let (alpha1, alpha2) = (-d0[1] / d0[0], -d0[2] / d0[0]);
    let (beta1, beta2) = (-dn[2] / dn[3], -dn[1] / dn[3]);
    // Column p of T: (row, value) pairs.
    let tcol = |p: usize| -> Vec<(usize, f64)> {
        let mut v = vec![(p + 1, 1.0)];
        if p == 0 {
            v.push((0, alpha1));
        }
        if p == 1 {
            v.push((0, alpha2));
        }
        if p == n - 1 {
            v.push((m - 1, beta1));
        }
        if p == n - 2 {
            v.push((m - 1, beta2));
        }
        v
    };
    let reduce = |xm: &Band4| -> Band4 {
        let mut r = Band4::zeros(n);
        for p in 0..n {
            for q in p..(p + 4).min(n) {
                let mut v = 0.0;
                for &(rr, tp) in &tcol(p) {
                    for &(ss, tq) in &tcol(q) {
                        v += tp * tq * xm.get(rr, ss);
                    }
                }
                r.d[p][q - p] = v;
            }
        }
        r
    };
    let (btwb_r, omega_r) = (reduce(&btwb), reduce(&omega));
    let btwy_r: Vec<f64> = (0..n).map(|p| tcol(p).iter().map(|&(r, v)| v * btwy[r]).sum()).collect();
    let expand = |a: &[f64]| -> Vec<f64> {
        let mut c = vec![0.0; m];
        c[1..m - 1].copy_from_slice(a);
        c[0] = alpha1 * a[0] + alpha2 * a[1];
        c[m - 1] = beta1 * a[n - 1] + beta2 * a[n - 2];
        c
    };
    let solve = |lam: f64| -> Result<(Vec<f64>, f64), ErrorsJSL> {
        let mat = btwb_r.lincomb(lam, &omega_r);
        let (l, d) = mat.ldl()?;
        let c = expand(&ldl_solve(&l, &d, &btwy_r));
        // tr A = tr(M⁻¹ Tᵀ Bᵀ W B T), using only the band of M⁻¹.
        let z = band_of_inverse(&l, &d);
        let mut trace = 0.0;
        for i in 0..n {
            for j in i..(i + 4).min(n) {
                let v = z.get(i, j) * btwb_r.get(j, i);
                trace += if i == j { v } else { 2.0 * v };
            }
        }
        Ok((c, trace))
    };
    let lam = match lam {
        Some(l) => l,
        None => {
            let gcv = |lam: f64| -> f64 {
                let Ok((c, trace)) = solve(lam) else { return f64::INFINITY };
                let rss: f64 = rows.iter().zip(y).map(|((c0, b), &yi)| (yi - b.iter().enumerate().map(|(r, v)| c[c0 + r] * v).sum::<f64>()).powi(2)).sum();
                (rss / n as f64) / (1.0 - trace / n as f64).powi(2)
            };
            fminbound(gcv, 0.0, n as f64)
        }
    };
    let (c, trace) = solve(lam)?;
    Ok(SmoothingSpline { spline: BSpline::new(t, c, 3)?, lam, degrees_of_freedom: trace })
}
// ANCHOR_END: smoothing

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_smoothing_gives_the_regression_line_and_none_interpolates() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.3 + (i as f64 * 0.7).sin() * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin() + 0.5 * v).collect();
        let interp = make_smoothing_spline(&x, &y, None, Some(0.0)).unwrap();
        for (xi, yi) in x.iter().zip(&y) {
            assert!((interp.spline.evaluate(*xi) - yi).abs() < 1e-9);
        }
        assert!((interp.degrees_of_freedom - 20.0).abs() < 1e-6);
        let line = make_smoothing_spline(&x, &y, None, Some(1e8)).unwrap();
        assert!((line.degrees_of_freedom - 2.0).abs() < 1e-3);
        assert!(line.spline.evaluate_derivative(1.7, 2).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn band_of_inverse_matches_a_dense_inverse() {
        let n = 9;
        let mut m = Band4::zeros(n);
        for i in 0..n {
            m.add(i, i, 4.0 + i as f64 * 0.1);
            for k in 1..4 {
                if i + k < n {
                    m.add(i, i + k, 0.5 / k as f64);
                }
            }
        }
        let (l, d) = m.ldl().unwrap();
        let z = band_of_inverse(&l, &d);
        // Column j of the inverse by solving with e_j.
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = ldl_solve(&l, &d, &e);
            for i in j.saturating_sub(3)..(j + 4).min(n) {
                assert!((z.get(i, j) - col[i]).abs() < 1e-14);
            }
        }
    }
}
