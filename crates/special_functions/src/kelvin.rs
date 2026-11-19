//! The Kelvin functions `ber`, `bei`, `ker`, `kei` and their derivatives (SciPy's `ber`, `bei`,
//! `ker`, `kei`, `berp`, `beip`, `kerp`, `keip` and `kelvin`).
//!
//! See the book entry [Faddeeva, Struve, Kelvin and Hankel functions](https://jwkunz.github.io/lib_jsl/book/special_functions/faddeeva-struve-kelvin.html#kelvin-functions).

use num_complex::Complex;
use std::f64::consts::{FRAC_PI_4, PI};

type C = Complex<f64>;

fn rotation() -> C {
    C::from_polar(1.0, FRAC_PI_4)
}

/// `I₀(z)` and `I₁(z)` by the periodic trapezoid rule on `(1/2π) ∮ e^{z cos θ} (1, cos θ) dθ`,
/// which converges geometrically once the number of points exceeds about `e|z|/2`.
fn i01(z: C) -> (C, C) {
    let n = (1.5 * z.norm()) as usize + 40;
    let (mut i0, mut i1) = (C::new(0.0, 0.0), C::new(0.0, 0.0));
    for j in 0..n {
        let c = (2.0 * PI * j as f64 / n as f64).cos();
        let e = (z * c).exp();
        i0 += e;
        i1 += e * c;
    }
    (i0 / n as f64, i1 / n as f64)
}

/// `K₀(z)` and `K₁(z)` for `Re z > 0` by the trapezoid rule on `∫₀^∞ e^{−z cosh t} (1, cosh t) dt`.
/// The integrand is analytic in a strip of half-width π/4 about the real axis when `arg z = π/4`,
/// so a step of 0.04 gives full accuracy; it is cut off where `Re z cosh t` passes 745.
fn k01(z: C) -> (C, C) {
    let h = 0.04;
    let end = (2.0 * 745.0 / z.re).ln() + 1.0;
    let (mut k0, mut k1) = (C::new(0.0, 0.0), C::new(0.0, 0.0));
    let mut t = 0.0;
    let mut first = true;
    while t < end {
        let ch = t.cosh();
        let w = if first { 0.5 } else { 1.0 };
        let e = (-z * ch).exp() * w;
        k0 += e;
        k1 += e * ch;
        first = false;
        t += h;
    }
    (k0 * h, k1 * h)
}

// ANCHOR: kelvin
/// All four Kelvin functions and their derivatives at `x` (SciPy's `kelvin`): returns
/// `(ber + i bei, ker + i kei, ber' + i bei', ker' + i kei')`, from
/// `ber x + i bei x = I₀(x e^{iπ/4})` and `ker x + i kei x = K₀(x e^{iπ/4})`, with derivatives
/// `e^{iπ/4} I₁` and `−e^{iπ/4} K₁`. The `I` integrals use the periodic trapezoid rule, the `K`
/// integrals the trapezoid rule on the real half line; both converge exponentially. For `x < 0`
/// `ber` and `bei` are even and their derivatives odd, while `ker`, `kei` and theirs are `NaN`.
pub fn kelvin(x: f64) -> (C, C, C, C) {
    let nan = C::new(f64::NAN, f64::NAN);
    if x.is_nan() {
        return (nan, nan, nan, nan);
    }
    let r = rotation();
    let (i0, i1) = i01(r * x.abs());
    let (be, bep) = (i0, if x < 0.0 { -(r * i1) } else { r * i1 });
    if x < 0.0 {
        return (be, nan, bep, nan);
    }
    if x == 0.0 {
        return (be, C::new(f64::INFINITY, -FRAC_PI_4), bep, C::new(f64::NEG_INFINITY, 0.0));
    }
    let (k0, k1) = k01(r * x);
    (be, k0, bep, -(r * k1))
}
// ANCHOR_END: kelvin

/// The Kelvin function `ber x`, the real part of `J₀(x e^{3iπ/4})`.
pub fn ber(x: f64) -> f64 {
    kelvin_i(x).0.re
}

/// The Kelvin function `bei x`, the imaginary part of `J₀(x e^{3iπ/4})`.
pub fn bei(x: f64) -> f64 {
    kelvin_i(x).0.im
}

/// The derivative of `ber`.
pub fn berp(x: f64) -> f64 {
    kelvin_i(x).1.re
}

/// The derivative of `bei`.
pub fn beip(x: f64) -> f64 {
    kelvin_i(x).1.im
}

/// The Kelvin function `ker x`, the real part of `K₀(x e^{iπ/4})` (`x ≥ 0`).
pub fn ker(x: f64) -> f64 {
    kelvin_k(x).0.re
}

/// The Kelvin function `kei x`, the imaginary part of `K₀(x e^{iπ/4})` (`x ≥ 0`).
pub fn kei(x: f64) -> f64 {
    kelvin_k(x).0.im
}

/// The derivative of `ker`.
pub fn kerp(x: f64) -> f64 {
    kelvin_k(x).1.re
}

/// The derivative of `kei`.
pub fn keip(x: f64) -> f64 {
    kelvin_k(x).1.im
}

/// `(ber + i bei, ber' + i bei')` without the `K` integrals.
fn kelvin_i(x: f64) -> (C, C) {
    if x.is_nan() {
        return (C::new(f64::NAN, f64::NAN), C::new(f64::NAN, f64::NAN));
    }
    let r = rotation();
    let (i0, i1) = i01(r * x.abs());
    (i0, if x < 0.0 { -(r * i1) } else { r * i1 })
}

/// `(ker + i kei, ker' + i kei')` without the `I` integrals.
fn kelvin_k(x: f64) -> (C, C) {
    let nan = C::new(f64::NAN, f64::NAN);
    if !(x >= 0.0) {
        return (nan, nan);
    }
    if x == 0.0 {
        return (C::new(f64::INFINITY, -FRAC_PI_4), C::new(f64::NEG_INFINITY, 0.0));
    }
    let r = rotation();
    let (k0, k1) = k01(r * x);
    (k0, -(r * k1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_wronskian_and_large_x() {
        // ber x = Σ (−1)^k (x/2)^{4k}/((2k)!)², bei x = Σ (−1)^k (x/2)^{4k+2}/((2k+1)!)².
        let x: f64 = 3.7;
        let (mut b, mut bi, mut f) = (0.0, 0.0, 1.0);
        for k in 0..30 {
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            b += s * (x / 2.0).powi(4 * k) / (f * f);
            f *= (2 * k + 1) as f64;
            bi += s * (x / 2.0).powi(4 * k + 2) / (f * f);
            f *= (2 * k + 2) as f64;
        }
        assert!((ber(x) - b).abs() < 1e-14 * b.abs().max(1.0));
        assert!((bei(x) - bi).abs() < 1e-14 * bi.abs().max(1.0));
        // The Wronskian ber·kei' − bei·ker' − (ber'·kei − bei'·ker) = 1/x … in complex form:
        // I₀K₁ + I₁K₀ = 1/z with z = x e^{iπ/4}.
        for &x in &[0.2, 1.5, 6.0, 15.0] {
            let (be, ke, bep, kep) = kelvin(x);
            let r = rotation();
            let w = be * (-kep / r) + (bep / r) * ke;
            assert!((w * (r * x) - 1.0).norm() < 1e-13, "{x}: {w}");
        }
        assert!(ker(-1.0).is_nan());
        assert_eq!(kei(0.0), -FRAC_PI_4);
        assert!((ber(-2.0) - ber(2.0)).abs() == 0.0 && (berp(-2.0) + berp(2.0)).abs() == 0.0);
    }
}
