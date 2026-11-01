//! Associated Legendre functions and spherical harmonics.
//!
//! Conventions: the Condon-Shortley phase `(−1)^m` is included in `P_l^m`, and
//! `Y_l^m(θ, φ) = √((2l+1)/(4π) · (l−m)!/(l+m)!) P_l^m(cos θ) e^{imφ}` is orthonormal on the unit
//! sphere, with `Y_l^{−m} = (−1)^m conj(Y_l^m)` (the physics and quantum-mechanics convention).
//!
//! See the book entry [Orthogonal polynomials and spherical harmonics](https://jwkunz.github.io/lib_jsl/book/special_functions/orthogonal-polynomials.html#associated-legendre-functions-and-spherical-harmonics).

use num_complex::Complex;
use std::f64::consts::PI;

/// The associated Legendre function `P_l^m(x)` for `|x| ≤ 1` and `|m| ≤ l` (zero otherwise),
/// with the Condon-Shortley phase. Negative orders use `P_l^{−m} = (−1)^m (l−m)!/(l+m)! P_l^m`.
/// Unnormalized values grow like `(2m)!` and overflow for orders beyond about 150; use
/// [`normalized_associated_legendre`] for high degrees.
pub fn associated_legendre(l: u32, m: i32, x: f64) -> f64 {
    let am = m.unsigned_abs();
    if am > l {
        return 0.0;
    }
    let s = (1.0 - x * x).max(0.0).sqrt();
    // P_m^m = (−1)^m (2m−1)!! (1−x²)^{m/2}.
    let mut pmm = 1.0;
    for k in 1..=am {
        pmm *= -((2 * k - 1) as f64) * s;
    }
    let mut value = pmm;
    if l > am {
        let mut prev = pmm;
        let mut cur = x * (2 * am + 1) as f64 * pmm;
        for ll in am + 2..=l {
            let next = ((2 * ll - 1) as f64 * x * cur - (ll + am - 1) as f64 * prev) / (ll - am) as f64;
            prev = cur;
            cur = next;
        }
        value = cur;
    }
    if m < 0 {
        // (l−m)!/(l+m)! for the positive order, as a product to avoid overflow.
        let ratio: f64 = (l - am + 1..=l + am).map(|k| 1.0 / k as f64).product();
        let sign = if am.is_multiple_of(2) { 1.0 } else { -1.0 };
        value *= sign * ratio;
    }
    value
}

/// The fully normalized `P̄_l^m(x) = √((2l+1)/(4π) · (l−m)!/(l+m)!) P_l^m(x)` for `m ≥ 0`, so that
/// `Y_l^m(θ, φ) = P̄_l^m(cos θ) e^{imφ}`. Computed by the normalized recurrence, which is stable
/// and free of overflow for degrees into the thousands.
pub fn normalized_associated_legendre(l: u32, m: u32, x: f64) -> f64 {
    if m > l {
        return 0.0;
    }
    let s = (1.0 - x * x).max(0.0).sqrt();
    let mut pmm = (1.0 / (4.0 * PI)).sqrt();
    for k in 1..=m {
        let k = k as f64;
        pmm *= -((2.0 * k + 1.0) / (2.0 * k)).sqrt() * s;
    }
    if l == m {
        return pmm;
    }
    let mf = m as f64;
    let a = |ll: f64| ((4.0 * ll * ll - 1.0) / (ll * ll - mf * mf)).sqrt();
    let mut prev = pmm;
    let mut cur = x * (2.0 * mf + 3.0).sqrt() * pmm;
    for ll in m + 2..=l {
        let llf = ll as f64;
        let next = a(llf) * (x * cur - prev / a(llf - 1.0));
        prev = cur;
        cur = next;
    }
    cur
}

/// The complex spherical harmonic `Y_l^m(θ, φ)` (θ polar from +z, φ azimuth from +x), zero for
/// `|m| > l`.
pub fn spherical_harmonic(l: u32, m: i32, theta: f64, phi: f64) -> Complex<f64> {
    let am = m.unsigned_abs();
    let p = normalized_associated_legendre(l, am, theta.cos());
    let y = Complex::from_polar(p, am as f64 * phi);
    if m >= 0 {
        y
    } else if am.is_multiple_of(2) {
        y.conj()
    } else {
        -y.conj()
    }
}

/// The real spherical harmonic: `√2 (−1)^m Re Y_l^m` for `m > 0`, `Y_l^0` for `m = 0` and
/// `√2 (−1)^m Im Y_l^{|m|}` for `m < 0`, i.e. `√2 P̄_l^{|m|} (−1)^m cos(mφ)` or `sin(|m|φ)`. These
/// form a real orthonormal basis.
pub fn real_spherical_harmonic(l: u32, m: i32, theta: f64, phi: f64) -> f64 {
    let am = m.unsigned_abs();
    let p = normalized_associated_legendre(l, am, theta.cos());
    let sign = if am.is_multiple_of(2) { 1.0 } else { -1.0 };
    match m.signum() {
        0 => p,
        1 => 2f64.sqrt() * sign * p * (am as f64 * phi).cos(),
        _ => 2f64.sqrt() * sign * p * (am as f64 * phi).sin(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orthogonal_polynomials::legendre;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    fn factorial(n: u32) -> f64 {
        (1..=n).map(|k| k as f64).product()
    }

    #[test]
    fn low_orders_match_closed_forms() {
        for &t in &[0.2f64, 1.0, 2.3] {
            let (c, s) = (t.cos(), t.sin());
            assert!(close(associated_legendre(1, 1, c), -s, 1e-14));
            assert!(close(associated_legendre(2, 1, c), -3.0 * c * s, 1e-14));
            assert!(close(associated_legendre(2, 2, c), 3.0 * s * s, 1e-14));
            assert!(close(associated_legendre(3, 2, c), 15.0 * c * s * s, 1e-14));
            assert!(close(associated_legendre(2, -1, c), 0.5 * c * s, 1e-14));
            let phi = 0.7;
            let y10 = spherical_harmonic(1, 0, t, phi);
            assert!(close(y10.re, (3.0 / (4.0 * PI)).sqrt() * c, 1e-14) && y10.im == 0.0);
            let y11 = spherical_harmonic(1, 1, t, phi);
            let want = Complex::from_polar(-(3.0 / (8.0 * PI)).sqrt() * s, phi);
            assert!((y11 - want).norm() < 1e-14);
            let y22 = spherical_harmonic(2, 2, t, phi);
            assert!((y22 - Complex::from_polar(0.25 * (15.0 / (2.0 * PI)).sqrt() * s * s, 2.0 * phi)).norm() < 1e-14);
            // Y_l^{−m} = (−1)^m conj(Y_l^m).
            assert!((spherical_harmonic(3, -1, t, phi) + spherical_harmonic(3, 1, t, phi).conj()).norm() < 1e-14);
        }
        for l in 0..10 {
            assert!(close(associated_legendre(l, 0, 0.37), legendre(l, 0.37), 1e-13));
        }
        assert_eq!(associated_legendre(2, 3, 0.5), 0.0);
        assert_eq!(spherical_harmonic(2, -3, 0.5, 0.1), Complex::new(0.0, 0.0));
    }

    #[test]
    fn normalized_values_agree_and_stay_stable_at_high_degree() {
        for l in 0..20u32 {
            for m in 0..=l {
                let norm = ((2 * l + 1) as f64 / (4.0 * PI) * factorial(l - m) / factorial(l + m)).sqrt();
                let x = -0.63;
                assert!(close(normalized_associated_legendre(l, m, x), norm * associated_legendre(l, m as i32, x), 1e-11), "{l} {m}");
            }
        }
        // Unsöld's theorem Σ_m |Y_l^m|² = (2l+1)/(4π) at a degree where P_l^m itself overflows.
        let l = 400;
        let theta = 1.1;
        let sum: f64 = (-(l as i32)..=l as i32).map(|m| spherical_harmonic(l, m, theta, 0.3).norm_sqr()).sum();
        assert!(close(sum, (2 * l + 1) as f64 / (4.0 * PI), 1e-10), "{sum}");
    }

    #[test]
    fn harmonics_are_orthonormal_on_the_sphere() {
        // Simpson in θ (smooth integrands) and the trapezoid rule in φ, which is
        // exact for these trigonometric polynomials.
        let (nt, np) = (400, 16);
        let integrate = |f: &dyn Fn(f64, f64) -> Complex<f64>| -> Complex<f64> {
            let h = PI / nt as f64;
            let mut total = Complex::new(0.0, 0.0);
            for i in 0..=nt {
                let t = i as f64 * h;
                let w = if i == 0 || i == nt { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 } * h / 3.0 * t.sin();
                for j in 0..np {
                    let p = 2.0 * PI * j as f64 / np as f64;
                    total += f(t, p) * w * (2.0 * PI / np as f64);
                }
            }
            total
        };
        let modes = [(0u32, 0i32), (1, -1), (1, 1), (2, 0), (3, 2), (3, -2), (4, 1)];
        for &(l1, m1) in &modes {
            for &(l2, m2) in &modes {
                let ip = integrate(&|t, p| spherical_harmonic(l1, m1, t, p) * spherical_harmonic(l2, m2, t, p).conj());
                let want = if (l1, m1) == (l2, m2) { 1.0 } else { 0.0 };
                assert!((ip - want).norm() < 1e-8, "complex ({l1},{m1}) ({l2},{m2}): {ip}");
                let rp = integrate(&|t, p| Complex::new(real_spherical_harmonic(l1, m1, t, p) * real_spherical_harmonic(l2, m2, t, p), 0.0));
                assert!((rp - want).norm() < 1e-8, "real ({l1},{m1}) ({l2},{m2}): {rp}");
            }
        }
    }
}
