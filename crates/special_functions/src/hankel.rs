//! The Hankel functions `H⁽¹⁾_ν = J_ν + iY_ν` and `H⁽²⁾_ν = J_ν − iY_ν` (SciPy's `hankel1`,
//! `hankel2`, `hankel1e` and `hankel2e`).
//!
//! See the book entry [Faddeeva, Struve, Kelvin and Hankel functions](https://jwkunz.github.io/lib_jsl/book/special_functions/faddeeva-struve-kelvin.html#hankel-functions-and-bessel-zeros).

use crate::bessel_functions_fractional::bessel_jy;
use num_complex::Complex;

type C = Complex<f64>;

// ANCHOR: hankel
/// The Hankel function of the first kind `H⁽¹⁾_ν(x) = J_ν(x) + i Y_ν(x)` for real order and
/// `x > 0` (`NaN` otherwise); an outgoing cylindrical wave, `≈ √(2/(πx)) e^{i(x − νπ/2 − π/4)}`.
pub fn hankel1(v: f64, x: f64) -> C {
    let (j, y) = bessel_jy(v, x);
    C::new(j, y)
}

/// The Hankel function of the second kind `H⁽²⁾_ν(x) = J_ν(x) − i Y_ν(x)`.
pub fn hankel2(v: f64, x: f64) -> C {
    let (j, y) = bessel_jy(v, x);
    C::new(j, -y)
}
// ANCHOR_END: hankel

/// The scaled Hankel function `H⁽¹⁾_ν(x) e^{−ix}`, which removes the oscillation of the phase.
pub fn hankel1e(v: f64, x: f64) -> C {
    hankel1(v, x) * C::from_polar(1.0, -x)
}

/// The scaled Hankel function `H⁽²⁾_ν(x) e^{ix}`.
pub fn hankel2e(v: f64, x: f64) -> C {
    hankel2(v, x) * C::from_polar(1.0, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn half_order_closed_form_and_conjugacy() {
        // H⁽¹⁾_{1/2}(x) = −i √(2/(πx)) e^{ix}.
        for &x in &[0.5, 2.0, 9.0, 40.0] {
            let h = hankel1(0.5, x);
            let want = C::new(0.0, -(2.0 / (PI * x)).sqrt()) * C::from_polar(1.0, x);
            assert!((h - want).norm() < 1e-13 * want.norm(), "{x}");
            assert_eq!(hankel2(0.5, x), h.conj());
            assert!((hankel1e(0.5, x).norm() - want.norm()).abs() < 1e-14);
        }
        assert!(hankel1(0.0, -1.0).re.is_nan());
    }
}
