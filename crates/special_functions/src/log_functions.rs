//! Functions for computing in the log domain without overflow or cancellation (SciPy's
//! `logsumexp`, `softmax`, `log_softmax`, `logit`, `expit`, `log_expit`, `log_ndtr`), and the
//! Box-Cox power transforms (`boxcox`, `boxcox1p`, `inv_boxcox`).
//!
//! See the book entry [Log-domain functions and inverse incomplete integrals](https://jwkunz.github.io/lib_jsl/book/special_functions/log-domain.html).

use crate::error_function::erfc;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

// ANCHOR: logsumexp
/// `ln Σ exp(aᵢ)` without overflow (SciPy's `logsumexp`): with `m = max aᵢ`, the sum is
/// `m + ln Σ exp(aᵢ − m)`, whose terms are at most 1. Returns `−∞` for an empty slice or when
/// every term is `−∞`, and `+∞` if any term is `+∞`.
pub fn logsumexp(a: &[f64]) -> f64 {
    let m = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return if a.iter().any(|v| v.is_nan()) { f64::NAN } else { m };
    }
    m + a.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}
// ANCHOR_END: logsumexp

/// `ln |Σ bᵢ exp(aᵢ)|` and the sign of the sum, for weights of either sign (SciPy's `logsumexp`
/// with `b` and `return_sign=True`). The log is `−∞` with sign 0 when the sum is zero.
pub fn logsumexp_weighted(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "values and weights must have the same length");
    let m = a.iter().zip(b).filter(|(_, w)| **w != 0.0).map(|(v, _)| *v).fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return (m, if m.is_infinite() && m > 0.0 { 1.0 } else { 0.0 });
    }
    let s: f64 = a.iter().zip(b).map(|(&v, &w)| w * (v - m).exp()).sum();
    if s == 0.0 { (f64::NEG_INFINITY, 0.0) } else { (m + s.abs().ln(), s.signum()) }
}

/// `exp(aᵢ) / Σ exp(aⱼ)`, computed as `exp(aᵢ − logsumexp(a))`.
pub fn softmax(a: &[f64]) -> Vec<f64> {
    let l = logsumexp(a);
    a.iter().map(|v| (v - l).exp()).collect()
}

/// `aᵢ − logsumexp(a)`, the logarithm of [`softmax`] without underflow.
pub fn log_softmax(a: &[f64]) -> Vec<f64> {
    let l = logsumexp(a);
    a.iter().map(|v| v - l).collect()
}

/// The log-odds `ln(p / (1 − p))`, the inverse of [`expit`].
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// The logistic sigmoid `1 / (1 + e^{−x})`, evaluated so that neither branch overflows.
pub fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `ln expit(x) = −ln(1 + e^{−x})`, accurate for large `|x|`.
pub fn log_expit(x: f64) -> f64 {
    if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() }
}

// ANCHOR: log_ndtr
/// The logarithm of the standard normal CDF, `ln Φ(x)` (SciPy's `log_ndtr`), accurate in both
/// tails: for `x > −1` as `ln(1 − Φ(−x))` with `ln_1p`; for `−20 < x ≤ −1` from `erfc`, which is
/// accurate relative to its tiny value; below that from the asymptotic series
/// `ln Φ(x) = −x²/2 − ln(−x) − ½ ln 2π + ln(1 − 1/x² + 3/x⁴ − 15/x⁶ + …)`.
pub fn log_ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > -1.0 {
        return (-0.5 * erfc(x * FRAC_1_SQRT_2)).ln_1p();
    }
    if x > -20.0 {
        return (0.5 * erfc(-x * FRAC_1_SQRT_2)).ln();
    }
    let x2 = x * x;
    let (mut term, mut sum) = (1.0, 1.0);
    for k in 1..12 {
        term *= -((2 * k - 1) as f64) / x2;
        sum += term;
    }
    -0.5 * x2 - (-x).ln() - 0.5 * (2.0 * PI).ln() + sum.ln()
}
// ANCHOR_END: log_ndtr

/// The Box-Cox transform `(x^λ − 1)/λ`, and `ln x` for `λ = 0` (SciPy's `special.boxcox`),
/// written with `exp_m1` so that it is continuous and accurate near `λ = 0`.
pub fn boxcox(x: f64, lambda: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if lambda.abs() < 1e-19 { x.ln() } else { (lambda * x.ln()).exp_m1() / lambda }
}

/// The Box-Cox transform of `1 + x`, accurate for small `x` (SciPy's `boxcox1p`).
pub fn boxcox1p(x: f64, lambda: f64) -> f64 {
    if x < -1.0 {
        return f64::NAN;
    }
    let l = x.ln_1p();
    if lambda.abs() < 1e-19 { l } else { (lambda * l).exp_m1() / lambda }
}

/// The inverse of [`boxcox`]: `(1 + λy)^{1/λ}`, and `e^y` for `λ = 0`.
pub fn inv_boxcox(y: f64, lambda: f64) -> f64 {
    if lambda.abs() < 1e-19 { y.exp() } else { ((lambda * y).ln_1p() / lambda).exp() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logsumexp_avoids_overflow() {
        assert!((logsumexp(&[1000.0, 1000.0]) - (1000.0 + 2f64.ln())).abs() < 1e-12);
        assert!((logsumexp(&[-1000.0, -1000.0 + 3f64.ln()]) - (-1000.0 + 4f64.ln())).abs() < 1e-12);
        assert_eq!(logsumexp(&[]), f64::NEG_INFINITY);
        let (l, s) = logsumexp_weighted(&[0.0, 1.0], &[1.0, -1.0]);
        assert!((l - (1f64.exp() - 1.0).ln()).abs() < 1e-14 && s == -1.0);
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn logistic_functions() {
        assert_eq!(expit(0.0), 0.5);
        assert!((logit(expit(3.7)) - 3.7).abs() < 1e-13);
        assert_eq!(expit(-1000.0), 0.0);
        assert!((log_expit(-1000.0) + 1000.0).abs() < 1e-12);
        assert!((log_expit(40.0) + (-40f64).exp()).abs() < 1e-30);
    }

    #[test]
    fn log_ndtr_in_both_tails() {
        // SciPy: special.log_ndtr.
        let cases = [(3.0, -0.0013508099647481925), (0.0, -std::f64::consts::LN_2), (-5.0, -15.064998393988727), (-30.0, -454.32124395634327), (-100.0, -5005.524208694205)];
        for (x, want) in cases {
            assert!((log_ndtr(x) - want).abs() <= 1e-13 * want.abs(), "{x}: {} vs {want}", log_ndtr(x));
        }
    }

    #[test]
    fn boxcox_round_trips() {
        for lambda in [-1.5, -1e-12, 0.0, 0.5, 2.0] {
            let y = boxcox(3.2, lambda);
            assert!((inv_boxcox(y, lambda) - 3.2).abs() < 1e-12);
        }
        assert!((boxcox1p(1e-10, 0.5) - 9.99999999975e-11).abs() < 1e-25);
    }
}
