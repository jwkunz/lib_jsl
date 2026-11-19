//! The Struve functions `H_ν` and modified Struve functions `L_ν` (SciPy's `struve` and
//! `modstruve`).
//!
//! See the book entry [Faddeeva, Struve, Kelvin and Hankel functions](https://jwkunz.github.io/lib_jsl/book/special_functions/faddeeva-struve-kelvin.html#struve-functions).

use crate::bessel_functions_fractional::{bessel_ik, bessel_jy};
use crate::gamma::ln_gamma;
use std::f64::consts::PI;

const EPS: f64 = f64::EPSILON;

/// `1/Γ(a)` as `(ln|1/Γ(a)|, sign)`, with sign 0 at the poles.
fn ln_rgamma(a: f64) -> (f64, f64) {
    if a <= 0.0 && a == a.floor() {
        return (f64::NEG_INFINITY, 0.0);
    }
    let sign = if a > 0.0 || (a.floor() as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    (-ln_gamma(a), sign)
}

/// An estimate of a value together with a bound on its absolute error.
type Estimate = (f64, f64);

/// The defining power series `Σ (∓1)^k (x/2)^{2k+ν+1} / (Γ(k + 3/2) Γ(k + ν + 3/2))`, accurate
/// while its terms do not cancel, i.e. for `x` up to about `ν + 20`.
fn power_series(v: f64, x: f64, modified: bool) -> Estimate {
    let (mut sum, mut largest) = (0.0f64, 0.0f64);
    let lx = (x / 2.0).ln();
    for k in 0..400 {
        let kf = k as f64;
        let (lr, sign) = ln_rgamma(kf + v + 1.5);
        let alt = if modified || k % 2 == 0 { 1.0 } else { -1.0 };
        let term = alt * sign * ((2.0 * kf + v + 1.0) * lx - ln_gamma(kf + 1.5) + lr).exp();
        sum += term;
        largest = largest.max(term.abs());
        if k > 5 && term.abs() <= 1e-17 * sum.abs() {
            break;
        }
    }
    (sum, (largest + sum.abs()) * EPS * 2.0)
}

/// The Bessel-function series `√(x/(2π)) Σ (x/2)^k/(k! (k + ½)) J_{k+ν+½}(x)`, with
/// `(−1)^k I_{k+ν+½}` for `L_ν`; for `H_ν` it converges at all `x` without cancellation.
fn bessel_series(v: f64, x: f64, modified: bool) -> Estimate {
    let (mut sum, mut largest) = (0.0f64, 0.0f64);
    let lx = (x / 2.0).ln();
    for k in 0..400 {
        let kf = k as f64;
        let c = (kf * lx - ln_gamma(kf + 1.0)).exp() / (kf + 0.5);
        let order = kf + v + 0.5;
        let b = if modified {
            let i = bessel_ik(order, x).0;
            if k % 2 == 0 { i } else { -i }
        } else {
            bessel_jy(order, x).0
        };
        let term = c * b;
        if !term.is_finite() {
            return (f64::NAN, f64::INFINITY);
        }
        sum += term;
        largest = largest.max(term.abs());
        if k > 5 && term.abs() <= 1e-17 * sum.abs() {
            break;
        }
    }
    let pre = (x / (2.0 * PI)).sqrt();
    (pre * sum, pre * (largest * 10.0 + sum.abs()) * EPS)
}

/// The large-`x` expansion `H_ν = Y_ν + (1/π) Σ Γ(k + ½) (x/2)^{ν−2k−1} / Γ(ν + ½ − k)` (and
/// `L_ν = I_{−ν} − (1/π) Σ (−1)^k …`), summed until its terms stop decreasing; its error is
/// about the last term.
fn asymptotic(v: f64, x: f64, modified: bool) -> Estimate {
    let lx = (x / 2.0).ln();
    let (mut sum, mut last) = (0.0f64, f64::INFINITY);
    for k in 0..80 {
        let kf = k as f64;
        let (lr, sign) = ln_rgamma(v + 0.5 - kf);
        let mut a = sign * (ln_gamma(kf + 0.5) + (v - 2.0 * kf - 1.0) * lx + lr).exp();
        if modified && k % 2 == 0 {
            a = -a;
        }
        if a.abs() > last {
            break;
        }
        sum += a;
        last = a.abs();
        if last == 0.0 {
            break;
        }
    }
    let base = if modified { bessel_ik(-v, x).0 } else { bessel_jy(v, x).1 };
    let value = base + sum / PI;
    (value, last / PI + (base.abs() + value.abs()) * EPS)
}

/// For `ν > −½`, `H_ν = Y_ν + (2(x/2)^ν/(√π Γ(ν + ½))) ∫₀^∞ e^{−xt}(1 + t²)^{ν−½} dt`: the integral
/// is smooth and positive, and with `xt = exp(s − e^{−s})` (a double-exponential substitution)
/// the trapezoid rule converges to full accuracy in 160 points. Accurate unless `H_ν` is much
/// smaller than `Y_ν` (large orders at small `x`).
fn integral_about_y(v: f64, x: f64) -> Estimate {
    let h = 1.0 / 16.0;
    let mut sum = 0.0;
    for k in 0..160 {
        let s = -5.0 + k as f64 * h;
        let es = (-s).exp();
        let u = (s - es).exp();
        sum += (-u).exp() * (1.0 + (u / x).powi(2)).powf(v - 0.5) * (1.0 + es) * u;
    }
    let part = (v * (x / 2.0).ln() - ln_gamma(v + 0.5)).exp() * 2.0 / PI.sqrt() * sum * h / x;
    let y = bessel_jy(v, x).1;
    (y + part, (y.abs() + part.abs()) * 4.0 * EPS)
}

/// The most accurate of the methods, judging by their error estimates; the costly
/// Bessel series only when neither of the others reaches full accuracy.
fn struve_positive(v: f64, x: f64, modified: bool) -> f64 {
    let consider = |best: &mut Estimate, e: Estimate| {
        if e.0.is_finite() && e.1 < best.1 {
            *best = e;
        }
    };
    let mut best: Estimate = (f64::NAN, f64::INFINITY);
    if x > 2.0 {
        consider(&mut best, asymptotic(v, x, modified));
        if !modified && v > -0.5 {
            consider(&mut best, integral_about_y(v, x));
        }
    }
    if x < 30.0 + v.abs() {
        consider(&mut best, power_series(v, x, modified));
    }
    if !(best.1 <= 4.0 * EPS * best.0.abs()) {
        consider(&mut best, bessel_series(v, x, modified));
    }
    best.0
}

/// `H_ν(0)` and `L_ν(0)`: 0 for `ν > −1`, `2/π` at `ν = −1`, infinite below.
fn at_zero(v: f64) -> f64 {
    if v > -1.0 {
        0.0
    } else if v == -1.0 {
        2.0 / PI
    } else {
        f64::INFINITY
    }
}

/// Negative `x`: `F_n(−x) = (−1)^{n+1} F_n(x)` for integer orders, `NaN` otherwise.
fn reflect(v: f64, x: f64, modified: bool) -> f64 {
    if v != v.round() {
        return f64::NAN;
    }
    let r = struve_positive(v, -x, modified);
    if (v.round() as i64).rem_euclid(2) == 0 { -r } else { r }
}

// ANCHOR: struve
/// The Struve function `H_ν(x)` (SciPy's `struve`), a particular solution of the inhomogeneous
/// Bessel equation `x²y'' + xy' + (x² − ν²)y = 4(x/2)^{ν+1}/(√π Γ(ν + ½))`. Of the defining
/// power series, the Bessel-function series, the large-`x` expansion about `Y_ν` and (for
/// `ν > −½`) the integral of `H_ν − Y_ν`, the one with the smallest error estimate is used.
/// Negative `x` only for integer orders.
pub fn struve_h(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return at_zero(v);
    }
    if x < 0.0 {
        return reflect(v, x, false);
    }
    struve_positive(v, x, false)
}
// ANCHOR_END: struve

// ANCHOR: modstruve
/// The modified Struve function `L_ν(x) = −i e^{−iνπ/2} H_ν(ix)` (SciPy's `modstruve`), by the
/// same three methods with `I` in place of `J` and `Y`.
pub fn struve_l(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return at_zero(v);
    }
    if x < 0.0 {
        return reflect(v, x, true);
    }
    struve_positive(v, x, true)
}
// ANCHOR_END: modstruve

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms_of_half_integer_order() {
        // H_{1/2}(x) = √(2/(πx)) (1 − cos x), L_{−1/2}(x) = √(2/(πx)) sinh x.
        for &x in &[0.3, 1.0, 4.0, 12.0, 25.0, 60.0] {
            let c = (2.0 / (PI * x)).sqrt();
            let h = c * (1.0 - x.cos());
            assert!((struve_h(0.5, x) - h).abs() < 1e-13 * h.abs().max(c), "H {x}");
            let l = c * x.sinh();
            assert!((struve_l(-0.5, x) - l).abs() < 1e-13 * l, "L {x}");
        }
        // Integer reflection and the value at zero.
        assert!((struve_h(1.0, -2.0) - struve_h(1.0, 2.0)).abs() < 1e-15);
        assert!((struve_h(0.0, -2.0) + struve_h(0.0, 2.0)).abs() < 1e-15);
        assert_eq!(struve_h(0.0, 0.0), 0.0);
        assert!(struve_h(0.5, -1.0).is_nan());
    }
}
