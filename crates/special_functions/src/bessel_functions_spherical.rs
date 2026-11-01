//! Spherical Bessel functions.
//!
//! See the book entry [Bessel and Airy functions](https://jwkunz.github.io/lib_jsl/book/special_functions/bessel.html#spherical-bessel-functions).

use crate::bessel_functions_fractional::bessel_jy;
use std::f64::consts::PI;

/// Returns `(j_n(x), y_n(x))` for `n >= 0` and `x > 0`.
fn spherical_jy(n: u32, x: f64) -> (f64, f64) {
    let (j, y) = bessel_jy(n as f64 + 0.5, x);
    let pref = (PI / (2.0 * x)).sqrt();
    (pref * j, pref * y)
}

/// The spherical Bessel function of the first kind j_n(x) = √(π/2x) J_{n+1/2}(x).
///
/// Defined for all real `x` and integer `n` (negative `n` use `j_{-n-1} = (-1)^(n+1) y_n`).
/// `j_n(-x) = (-1)^n j_n(x)`.
pub fn bessel_spherical(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if n < 0 {
        let m = (-n - 1) as u32;
        let y = bessel_spherical_y(m as i32, x);
        return if m.is_multiple_of(2) { -y } else { y };
    }
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x < 0.0 {
        let r = bessel_spherical(n, -x);
        return if n % 2 == 0 { r } else { -r };
    }
    let n = n as u32;
    if n == 0 {
        // Closed form avoids needless loss near the origin.
        return x.sin() / x;
    }
    spherical_jy(n, x).0
}

/// The spherical Bessel function of the second kind y_n(x) = √(π/2x) Y_{n+1/2}(x) for `x > 0`
/// (`-inf` at 0, `NaN` for negative `x`).
pub fn bessel_spherical_y(n: i32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if n < 0 {
        let m = (-n - 1) as u32;
        let j = bessel_spherical(m as i32, x);
        return if m.is_multiple_of(2) { j } else { -j };
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if n == 0 {
        return -x.cos() / x;
    }
    spherical_jy(n as u32, x).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms() {
        for &x in &[0.5f64, 1.0, 3.0, 12.0] {
            let j1 = x.sin() / (x * x) - x.cos() / x;
            let j2 = (3.0 / (x * x) - 1.0) * x.sin() / x - 3.0 * x.cos() / (x * x);
            assert!((bessel_spherical(1, x) - j1).abs() < 1e-12 * (1.0 + j1.abs()) , "j1 x={x}");
            assert!((bessel_spherical(2, x) - j2).abs() < 1e-12 * (1.0 + j2.abs()), "j2 x={x}");
            let y1 = -x.cos() / (x * x) - x.sin() / x;
            assert!((bessel_spherical_y(1, x) - y1).abs() < 1e-12 * (1.0 + y1.abs()), "y1 x={x}");
        }
    }

    #[test]
    fn recurrence() {
        // j_{n-1} + j_{n+1} = (2n+1)/x j_n
        let x: f64 = 2.3;
        for n in 1..10 {
            let lhs = bessel_spherical(n - 1, x) + bessel_spherical(n + 1, x);
            let rhs = (2 * n + 1) as f64 / x * bessel_spherical(n, x);
            assert!((lhs - rhs).abs() < 1e-13);
        }
    }

    #[test]
    fn symmetry_and_edges() {
        assert_eq!(bessel_spherical(0, 0.0), 1.0);
        assert_eq!(bessel_spherical(3, 0.0), 0.0);
        assert!((bessel_spherical(1, -2.0) + bessel_spherical(1, 2.0)).abs() < 1e-15);
        assert!((bessel_spherical(2, -2.0) - bessel_spherical(2, 2.0)).abs() < 1e-15);
        // j_{-1} = cos x / x, y_{-1} = sin x / x
        let x: f64 = 1.3;
        assert!((bessel_spherical(-1, x) - x.cos() / x).abs() < 1e-14);
        assert!((bessel_spherical_y(-1, x) - x.sin() / x).abs() < 1e-14);
        assert!(bessel_spherical_y(0, -1.0).is_nan());
    }

    #[test]
    fn small_argument_limit() {
        // j_n(x) ~ x^n / (2n+1)!!
        let x = 1e-3;
        assert!((bessel_spherical(3, x) / (x.powi(3) / 105.0) - 1.0).abs() < 1e-5);
    }
}
