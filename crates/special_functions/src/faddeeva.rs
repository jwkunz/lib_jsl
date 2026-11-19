//! The Faddeeva function `w(z) = e^{−z²} erfc(−iz)` (SciPy's `wofz`) and the Voigt profile, the
//! convolution of a Gaussian and a Lorentzian (SciPy's `voigt_profile`).
//!
//! See the book entry [Faddeeva, Struve, Kelvin and Hankel functions](https://jwkunz.github.io/lib_jsl/book/special_functions/faddeeva-struve-kelvin.html).

use crate::dawsons_integral::dawson_integral;
use num_complex::Complex;
use std::f64::consts::PI;
use std::sync::OnceLock;

type C = Complex<f64>;

/// Terms of Weideman's rational approximation.
const N: usize = 40;

/// The coefficients of Weideman's (1994) expansion of `w` in powers of `Z = (L + iz)/(L − iz)`,
/// from a discrete Fourier transform of `e^{−t²}(L² + t²)` sampled at `t = L tan(θ/2)`.
fn weideman_coefficients() -> &'static (f64, Vec<f64>) {
    static COEF: OnceLock<(f64, Vec<f64>)> = OnceLock::new();
    COEF.get_or_init(|| {
        let m = 2 * N;
        let m2 = 2 * m;
        let l = (N as f64 / 2f64.sqrt()).sqrt();
        // f over k = −M+1..M−1, preceded by a zero: 2M samples.
        let mut f = vec![0.0; 2 * m];
        for (idx, k) in (-(m as isize) + 1..m as isize).enumerate() {
            let theta = k as f64 * PI / m as f64;
            let t = l * (theta / 2.0).tan();
            f[idx + 1] = (-t * t).exp() * (l * l + t * t);
        }
        // fftshift, then the real part of the DFT, divided by 2M·2 = M2.
        let len = f.len();
        let shifted: Vec<f64> = (0..len).map(|i| f[(i + len / 2) % len]).collect();
        let a: Vec<f64> = (0..len)
            .map(|j| shifted.iter().enumerate().map(|(n, v)| v * (2.0 * PI * (j * n) as f64 / len as f64).cos()).sum::<f64>() / m2 as f64)
            .collect();
        // a(2..N+1), reversed: the polynomial's coefficients from the highest power.
        let coef: Vec<f64> = a[1..=N].iter().rev().copied().collect();
        (l, coef)
    })
}

fn weideman(z: C) -> C {
    let (l, coef) = weideman_coefficients();
    let i = C::new(0.0, 1.0);
    let zz = (*l + i * z) / (*l - i * z);
    let p = coef.iter().fold(C::new(0.0, 0.0), |acc, &c| acc * zz + c);
    p * 2.0 / ((*l - i * z) * (*l - i * z)) + 1.0 / (PI.sqrt() * (*l - i * z))
}

/// Laplace's continued fraction `w(z) = (i/√π) / (z − ½/(z − 1/(z − (3/2)/(z − …))))`, which
/// converges quickly for large `|z|` in the upper half plane.
fn continued_fraction(z: C) -> C {
    let mut t = z;
    for k in (1..=40).rev() {
        t = z - (k as f64 / 2.0) / t;
    }
    C::new(0.0, 1.0 / PI.sqrt()) / t
}

/// The Taylor series about the real axis, `w(x + iy) = Σ w⁽ⁿ⁾(x) (iy)ⁿ/n!`, with
/// `w(x) = e^{−x²} + (2i/√π) D(x)`, `w' = −2zw + 2i/√π` and `w⁽ⁿ⁺¹⁾ = −2x w⁽ⁿ⁾ − 2n w⁽ⁿ⁻¹⁾`. Near the
/// real axis `Re w` is much smaller than `|w|`, so the other methods, accurate relative to `|w|`,
/// lose its digits; the series keeps both parts to about 10⁻¹⁴ for `0 < y ≤ ½`, `|x| < 9`.
fn taylor_near_real_axis(x: f64, y: f64) -> C {
    let c = 2.0 / PI.sqrt();
    let w0 = C::new((-x * x).exp(), c * dawson_integral(x));
    let mut prev = w0;
    let mut cur = -2.0 * x * w0 + C::new(0.0, c);
    let mut sum = w0;
    let mut t = C::new(0.0, y);
    for n in 1..60 {
        sum += cur * t;
        let next = -2.0 * x * cur - 2.0 * n as f64 * prev;
        prev = cur;
        cur = next;
        t *= C::new(0.0, y / (n + 1) as f64);
        if t.norm() == 0.0 {
            break;
        }
    }
    sum
}

// ANCHOR: wofz
/// The Faddeeva function `w(z) = e^{−z²} erfc(−iz)` (SciPy's `wofz`). On the real axis it is
/// `e^{−x²} + (2i/√π) D(x)` with Dawson's integral `D`; just above it (`0 < y ≤ ½`, `|x| < 9`) the
/// Taylor series in `y` about that, which keeps the small real part accurate; elsewhere in the
/// upper half plane, Weideman's rational approximation with 40 terms for `|z| < 6` and Laplace's
/// continued fraction beyond;
/// in the lower half plane, the reflection `w(z) = 2 e^{−z²} − w(−z)`. Relative accuracy about
/// 10⁻¹⁴ in the upper half plane, for the real and imaginary parts separately.
pub fn wofz(z: C) -> C {
    if z.re.is_nan() || z.im.is_nan() {
        return C::new(f64::NAN, f64::NAN);
    }
    if z.im == 0.0 {
        let x = z.re;
        return C::new((-x * x).exp(), 2.0 / PI.sqrt() * dawson_integral(x));
    }
    if z.im < 0.0 {
        return (-z * z).exp() * 2.0 - wofz(-z);
    }
    if z.im <= 0.5 && z.re.abs() < 9.0 {
        taylor_near_real_axis(z.re, z.im)
    } else if z.norm() >= 6.0 {
        continued_fraction(z)
    } else {
        weideman(z)
    }
}
// ANCHOR_END: wofz

// ANCHOR: voigt
/// The Voigt profile (SciPy's `voigt_profile`): the density of the sum of a normal variable with
/// standard deviation `σ` and a Cauchy variable with half-width `γ`,
/// `V(x) = Re w((x + iγ)/(σ√2)) / (σ√(2π))`. With `σ = 0` it is the Lorentzian, with `γ = 0` the
/// Gaussian.
pub fn voigt_profile(x: f64, sigma: f64, gamma: f64) -> f64 {
    if !(sigma >= 0.0) || !(gamma >= 0.0) {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return if gamma == 0.0 { if x == 0.0 { f64::INFINITY } else { 0.0 } } else { gamma / (PI * (x * x + gamma * gamma)) };
    }
    let s = sigma * 2f64.sqrt();
    wofz(C::new(x / s, gamma / s)).re / (sigma * (2.0 * PI).sqrt())
}
// ANCHOR_END: voigt

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_and_limits() {
        // w(iy) = e^{y²} erfc(y), real.
        for &y in &[0.1, 0.8, 3.0, 10.0] {
            let w = wofz(C::new(0.0, y));
            let want = (y * y).exp() * crate::error_function::erfc(y);
            assert!((w.re - want).abs() < 1e-13 * want && w.im.abs() < 1e-14 * want, "{y}");
        }
        // w(−z̄) = conj(w(z)), across all regions.
        for &(x, y) in &[(0.3, 0.2), (2.0, 1.0), (7.0, 0.1), (3.0, 4.0), (-5.0, 9.0), (1.0, -0.7)] {
            let (a, b) = (wofz(C::new(x, y)), wofz(C::new(-x, y)));
            assert!((a - b.conj()).norm() < 1e-14 * a.norm(), "{x} {y}");
        }
        // Continuity across the switch between methods.
        for &(x, y) in &[(8.999, 0.3), (9.001, 0.3), (2.0, 0.4999), (2.0, 0.5001), (4.2, 4.3)] {
            let (a, b) = (wofz(C::new(x, y)), wofz(C::new(x + 1e-9, y + 1e-9)));
            assert!((a - b).norm() < 1e-8 * a.norm(), "{x} {y}");
        }
        // The Voigt profile integrates to one and has the right limits.
        let h = 0.01;
        let area: f64 = (-40000..=40000).map(|i| voigt_profile(i as f64 * h, 0.7, 0.3)).sum::<f64>() * h;
        assert!((area - 1.0).abs() < 1e-3, "{area}");
        assert!((voigt_profile(0.4, 1.0, 0.0) - (-0.08f64).exp() / (2.0 * PI).sqrt()).abs() < 1e-15);
    }
}
