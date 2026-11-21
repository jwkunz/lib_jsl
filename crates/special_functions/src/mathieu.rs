//! Mathieu functions (SciPy's `mathieu_a`, `mathieu_b`, `mathieu_cem`, `mathieu_sem`,
//! `mathieu_modcem1`, `mathieu_modcem2`, `mathieu_modsem1`, `mathieu_modsem2`): solutions of
//! `y'' + (a − 2q cos 2x) y = 0` periodic in `x`, and of the modified equation
//! `y'' − (a − 2q cosh 2z) y = 0`.
//!
//! See the book entry [Mathieu, parabolic cylinder and spheroidal wave functions](https://jwkunz.github.io/lib_jsl/book/special_functions/wave-functions.html).

use crate::bessel_functions_integer::bessel_jy_integer;
use crate::tridiagonal::{eigenvalues, eigenvector};

/// The four families of periodic Mathieu functions, by parity and period.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    /// `ce_{2n}`: `Σ A_{2k} cos 2kx`.
    CeEven,
    /// `ce_{2n+1}`: `Σ A_{2k+1} cos (2k+1)x`.
    CeOdd,
    /// `se_{2n+1}`: `Σ B_{2k+1} sin (2k+1)x`.
    SeOdd,
    /// `se_{2n+2}`: `Σ B_{2k+2} sin (2k+2)x`.
    SeEven,
}

impl Family {
    fn of(m: usize, sine: bool) -> Family {
        match (sine, m.is_multiple_of(2)) {
            (false, true) => Family::CeEven,
            (false, false) => Family::CeOdd,
            (true, false) => Family::SeOdd,
            (true, true) => Family::SeEven,
        }
    }
    /// The harmonic of coefficient index `k`.
    fn harmonic(self, k: usize) -> f64 {
        match self {
            Family::CeEven => (2 * k) as f64,
            Family::CeOdd | Family::SeOdd => (2 * k + 1) as f64,
            Family::SeEven => (2 * k + 2) as f64,
        }
    }
    /// The position of order `m` in its family (the eigenvalue index).
    fn index(self, m: usize) -> usize {
        match self {
            Family::CeEven => m / 2,
            Family::CeOdd | Family::SeOdd => (m - 1) / 2,
            Family::SeEven => (m - 2) / 2,
        }
    }
}

// ANCHOR: mathieu_matrix
/// The tridiagonal matrix of the Fourier-coefficient recurrence of a family (`q ≥ 0`):
/// diagonal `h_k²` (with `±q` in the first entry of the odd families) and off-diagonals `q`,
/// except the coupling of `A₀` and `A₂`, which is `2q` below the diagonal.
fn matrix(f: Family, q: f64, size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut d: Vec<f64> = (0..size).map(|k| f.harmonic(k).powi(2)).collect();
    let sup = vec![q; size - 1];
    let mut sub = vec![q; size - 1];
    match f {
        Family::CeEven => sub[0] = 2.0 * q,
        Family::CeOdd => d[0] += q,
        Family::SeOdd => d[0] -= q,
        Family::SeEven => {}
    }
    (d, sup, sub)
}
// ANCHOR_END: mathieu_matrix

fn size_for(m: usize, q: f64) -> usize {
    m / 2 + 30 + (4.0 * q.abs().sqrt()) as usize
}

/// The characteristic value and the normalised Fourier coefficients for `q ≥ 0`.
fn solve(f: Family, m: usize, q: f64) -> (f64, Vec<f64>) {
    let n = size_for(m, q);
    let (d, sup, sub) = matrix(f, q, n);
    let lam = eigenvalues(&d, &sup, &sub)[f.index(m)];
    let mut c = eigenvector(&d, &sup, &sub, lam);
    // ∫₀^{2π} y² = π: 2A₀² + ΣA² = 1 for ce_{2n}, ΣA² = 1 otherwise.
    let norm = (c.iter().map(|v| v * v).sum::<f64>() + if f == Family::CeEven { c[0] * c[0] } else { 0.0 }).sqrt();
    c.iter_mut().for_each(|v| *v /= norm);
    // Sign (DLMF 28.2.30): ce_m(0, q) > 0 and se_m'(0, q) > 0.
    let at_zero: f64 = match f {
        Family::CeEven | Family::CeOdd => c.iter().sum(),
        Family::SeOdd | Family::SeEven => c.iter().enumerate().map(|(k, v)| f.harmonic(k) * v).sum(),
    };
    if at_zero < 0.0 {
        c.iter_mut().for_each(|v| *v = -*v);
    }
    (lam, c)
}

/// The characteristic value `a_m(q)` of the even solutions `ce_m` (SciPy's `mathieu_a`).
pub fn mathieu_a(m: usize, q: f64) -> f64 {
    if q < 0.0 {
        // a_{2n}(−q) = a_{2n}(q), a_{2n+1}(−q) = b_{2n+1}(q).
        return if m.is_multiple_of(2) { mathieu_a(m, -q) } else { mathieu_b(m, -q) };
    }
    solve(Family::of(m, false), m, q).0
}

/// The characteristic value `b_m(q)` of the odd solutions `se_m`, `m ≥ 1` (SciPy's `mathieu_b`).
pub fn mathieu_b(m: usize, q: f64) -> f64 {
    if m == 0 {
        return f64::NAN;
    }
    if q < 0.0 {
        return if m.is_multiple_of(2) { mathieu_b(m, -q) } else { mathieu_a(m, -q) };
    }
    solve(Family::of(m, true), m, q).0
}

fn series(f: Family, c: &[f64], x: f64) -> (f64, f64) {
    let sine = matches!(f, Family::SeOdd | Family::SeEven);
    let (mut y, mut dy) = (0.0, 0.0);
    for (k, ck) in c.iter().enumerate() {
        let h = f.harmonic(k);
        if sine {
            y += ck * (h * x).sin();
            dy += ck * h * (h * x).cos();
        } else {
            y += ck * (h * x).cos();
            dy -= ck * h * (h * x).sin();
        }
    }
    (y, dy)
}

// ANCHOR: mathieu_cem
/// The even Mathieu function `ce_m(x, q)` and its derivative `d/dx` (SciPy's `mathieu_cem`;
/// as in SciPy, `x` is in **degrees** and the derivative is with respect to radians).
/// Normalised so that `∫₀^{2π} ce_m² dx = π`, with `ce_m(0, q) > 0` (DLMF 28.2.30).
pub fn mathieu_cem(m: usize, q: f64, x_deg: f64) -> (f64, f64) {
    let x = x_deg.to_radians();
    if q < 0.0 {
        // ce_{2n}(x, −q) = (−1)ⁿ ce_{2n}(π/2 − x, q); ce_{2n+1}(x, −q) = (−1)ⁿ se_{2n+1}(π/2 − x, q).
        let sign = if (m / 2).is_multiple_of(2) { 1.0 } else { -1.0 };
        let (y, dy) = if m.is_multiple_of(2) { mathieu_cem(m, -q, 90.0 - x_deg) } else { mathieu_sem(m, -q, 90.0 - x_deg) };
        return (sign * y, -sign * dy);
    }
    let f = Family::of(m, false);
    let (_, c) = solve(f, m, q);
    series(f, &c, x)
}
// ANCHOR_END: mathieu_cem

/// The odd Mathieu function `se_m(x, q)`, `m ≥ 1`, and its derivative (SciPy's `mathieu_sem`;
/// `x` in degrees).
pub fn mathieu_sem(m: usize, q: f64, x_deg: f64) -> (f64, f64) {
    if m == 0 {
        return (0.0, 0.0);
    }
    let x = x_deg.to_radians();
    if q < 0.0 {
        // se_{2n+1}(x, −q) = (−1)ⁿ ce_{2n+1}(π/2 − x, q); se_{2n+2}(x, −q) = (−1)ⁿ se_{2n+2}(π/2 − x, q).
        let n = if m % 2 == 1 { (m - 1) / 2 } else { (m - 2) / 2 };
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let (y, dy) = if m % 2 == 1 { mathieu_cem(m, -q, 90.0 - x_deg) } else { mathieu_sem(m, -q, 90.0 - x_deg) };
        return (sign * y, -sign * dy);
    }
    let f = Family::of(m, true);
    let (_, c) = solve(f, m, q);
    series(f, &c, x)
}

/// `J_n` and `C_n` (`J` or `Y`) of integer order, any sign, with derivatives.
fn bessel(kind: usize, n: i64, x: f64) -> (f64, f64) {
    let val = |n: i64| -> f64 {
        let (j, y) = bessel_jy_integer(n.unsigned_abs() as i32, x);
        let v = if kind == 1 { j } else { y };
        if n < 0 && n % 2 != 0 { -v } else { v }
    };
    (val(n), 0.5 * (val(n - 1) - val(n + 1)))
}

// ANCHOR: modified
/// The modified (radial) Mathieu functions of the first (`kind = 1`) and second (`kind = 2`)
/// kind by Bessel-function product series (DLMF 28.24): with `u₁ = √q e^{−z}`, `u₂ = √q e^{z}`,
/// e.g. `Mc_{2n}^{(j)}(z) = Σ_k (−1)^{k+n} A_{2k}/A_{2s} [J_{k−s}(u₁) C_{k+s}(u₂) + J_{k+s}(u₁) C_{k−s}(u₂)] / ε_s`,
/// where `s` is the index of the largest coefficient (for accuracy), `ε₀ = 2`, `ε_s = 1`
/// otherwise, and `C = J` or `Y`. Returns the value and the derivative in `z`; `q > 0`.
fn modified(m: usize, q: f64, z: f64, kind: usize, sine: bool) -> (f64, f64) {
    if q <= 0.0 || (sine && m == 0) {
        return (f64::NAN, f64::NAN);
    }
    let f = Family::of(m, sine);
    let (_, c) = solve(f, m, q);
    let h = q.sqrt();
    let (u1, u2) = (h * (-z).exp(), h * z.exp());
    let s = (0..c.len()).max_by(|&a, &b| c[a].abs().total_cmp(&c[b].abs())).unwrap();
    // Offsets: Mc_{2n}: (k−s, k+s); Mc_{2n+1}, Ms_{2n+1}: (k−s, k+s+1); Ms_{2n+2}: (k−s, k+s+2).
    let off = match f {
        Family::CeEven => 0,
        Family::CeOdd | Family::SeOdd => 1,
        Family::SeEven => 2,
    };
    let minus = sine;
    let n = f.index(m) as i64;
    let (mut val, mut der) = (0.0, 0.0);
    for (k, ck) in c.iter().enumerate() {
        let k = k as i64;
        let sk = s as i64;
        let sign = if (k + n) % 2 == 0 { 1.0 } else { -1.0 };
        let w = sign * ck / c[s];
        let (j1a, dj1a) = bessel(1, k - sk, u1);
        let (c2a, dc2a) = bessel(kind, k + sk + off, u2);
        let (j1b, dj1b) = bessel(1, k + sk + off, u1);
        let (c2b, dc2b) = bessel(kind, k - sk, u2);
        let pm = if minus { -1.0 } else { 1.0 };
        val += w * (j1a * c2a + pm * j1b * c2b);
        // d/dz: J(u₁)' brings −u₁, C(u₂)' brings +u₂.
        der += w * ((-u1 * dj1a) * c2a + j1a * (u2 * dc2a) + pm * ((-u1 * dj1b) * c2b + j1b * (u2 * dc2b)));
    }
    let eps = if f == Family::CeEven && s == 0 { 2.0 } else { 1.0 };
    (val / eps, der / eps)
}
// ANCHOR_END: modified

/// `Mc_m^{(1)}(z, q)` and its derivative (SciPy's `mathieu_modcem1`).
pub fn mathieu_modcem1(m: usize, q: f64, z: f64) -> (f64, f64) {
    modified(m, q, z, 1, false)
}

/// `Mc_m^{(2)}(z, q)` and its derivative (SciPy's `mathieu_modcem2`).
pub fn mathieu_modcem2(m: usize, q: f64, z: f64) -> (f64, f64) {
    modified(m, q, z, 2, false)
}

/// `Ms_m^{(1)}(z, q)` and its derivative, `m ≥ 1` (SciPy's `mathieu_modsem1`).
pub fn mathieu_modsem1(m: usize, q: f64, z: f64) -> (f64, f64) {
    modified(m, q, z, 1, true)
}

/// `Ms_m^{(2)}(z, q)` and its derivative, `m ≥ 1` (SciPy's `mathieu_modsem2`).
pub fn mathieu_modsem2(m: usize, q: f64, z: f64) -> (f64, f64) {
    modified(m, q, z, 2, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_q_limits_and_symmetries() {
        // a_m(0) = b_m(0) = m², ce_m(x, 0) = cos mx (ce₀ = 1/√2).
        for m in 1..5 {
            assert!((mathieu_a(m, 0.0) - (m * m) as f64).abs() < 1e-12 && (mathieu_b(m, 0.0) - (m * m) as f64).abs() < 1e-12);
        }
        assert!((mathieu_cem(0, 0.0, 33.0).0 - 0.5f64.sqrt()).abs() < 1e-14);
        assert!((mathieu_cem(3, 0.0, 33.0).0 - (99.0f64).to_radians().cos()).abs() < 1e-14);
        // Known value: a₀(1) = −0.4551386041…
        assert!((mathieu_a(0, 1.0) + 0.455_138_604_107_414_7).abs() < 1e-12);
        // a_{2n+1}(−q) = b_{2n+1}(q).
        assert!((mathieu_a(3, -2.5) - mathieu_b(3, 2.5)).abs() < 1e-12);
        // The Wronskian of the modified functions: Mc⁽¹⁾ Mc⁽²⁾' − Mc⁽¹⁾' Mc⁽²⁾ = 2/π.
        let (j, dj) = mathieu_modcem1(2, 3.0, 0.7);
        let (y, dy) = mathieu_modcem2(2, 3.0, 0.7);
        assert!((j * dy - dj * y - 2.0 / std::f64::consts::PI).abs() < 1e-10);
    }
}
