//! Polygamma functions, the multivariate log-gamma function, and Bernoulli and Euler numbers
//! (SciPy's `polygamma`, `multigammaln`, `bernoulli`, `euler`).
//!
//! See the book entry [Log-domain functions and inverse incomplete integrals](https://jwkunz.github.io/lib_jsl/book/special_functions/log-domain.html#polygamma-and-friends).

use crate::factorial::factorial_f64;
use crate::gamma::{digamma, ln_gamma};
use crate::zeta::{hurwitz_zeta, zeta};
use std::f64::consts::PI;

// ANCHOR: polygamma
/// The polygamma function `ψ⁽ⁿ⁾(x)`, the `n`-th derivative of the digamma function (SciPy's
/// `polygamma`). For `n ≥ 1`, `ψ⁽ⁿ⁾(x) = (−1)ⁿ⁺¹ n! ζ(n + 1, x)` with the Hurwitz zeta function;
/// negative non-integer `x` is first shifted to positive values by
/// `ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x + 1) + (−1)ⁿ⁺¹ n! / xⁿ⁺¹`. `NaN` at the poles `x = 0, −1, −2, …`.
pub fn polygamma(n: u32, x: f64) -> f64 {
    if n == 0 {
        return digamma(x);
    }
    if x.is_nan() || (x <= 0.0 && x == x.floor()) {
        return f64::NAN;
    }
    let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
    let nf = factorial_f64(n as u64);
    let (mut x, mut shift) = (x, 0.0);
    while x <= 0.0 {
        shift += sign * nf / x.powi(n as i32 + 1);
        x += 1.0;
    }
    sign * nf * hurwitz_zeta(n as f64 + 1.0, x) + shift
}
// ANCHOR_END: polygamma

/// The logarithm of the multivariate gamma function,
/// `ln Γ_d(a) = d(d−1)/4 · ln π + Σ_{j=1}^{d} ln Γ(a + (1 − j)/2)`, for `a > (d − 1)/2`
/// (SciPy's `multigammaln`; it normalises the Wishart density).
pub fn multigammaln(a: f64, d: u32) -> f64 {
    if !(a > (d as f64 - 1.0) / 2.0) {
        return f64::NAN;
    }
    let d = d as f64;
    d * (d - 1.0) / 4.0 * PI.ln() + (1..=d as u32).map(|j| ln_gamma(a + (1.0 - j as f64) / 2.0)).sum::<f64>()
}

/// The Bernoulli numbers `B₀, …, B_n` with `B₁ = −1/2` (SciPy's `bernoulli`). The odd ones
/// beyond `B₁` are zero; the even ones come from `B_{2k} = (−1)^{k+1} 2 (2k)! ζ(2k) / (2π)^{2k}`,
/// accurate to rounding up to `n ≈ 260`, where they overflow.
pub fn bernoulli(n: usize) -> Vec<f64> {
    let mut b = vec![0.0; n + 1];
    b[0] = 1.0;
    if n >= 1 {
        b[1] = -0.5;
    }
    let mut ratio = 1.0; // (2k)! / (2π)^{2k}, updated incrementally.
    for k in 1..=n / 2 {
        let m = 2 * k;
        ratio *= ((m - 1) * m) as f64 / (2.0 * PI).powi(2);
        b[m] = if k % 2 == 1 { 2.0 } else { -2.0 } * ratio * zeta(m as f64);
    }
    b
}

/// The Euler numbers `E₀, …, E_n` (SciPy's `euler`): the coefficients of `sech t = Σ E_n tⁿ/n!`,
/// zero for odd `n`, from `E_{2k} = (−1)^k 2 (2k)! (2/π)^{2k+1} β(2k + 1)` with Dirichlet's
/// `β(s) = Σ (−1)^j/(2j + 1)^s = 4^{−s} (ζ(s, 1/4) − ζ(s, 3/4))`.
pub fn euler(n: usize) -> Vec<f64> {
    let mut e = vec![0.0; n + 1];
    e[0] = 1.0;
    let mut ratio = 2.0 / PI; // (2k)! (2/π)^{2k+1}
    for k in 1..=n / 2 {
        let m = 2 * k;
        ratio *= ((m - 1) * m) as f64 * (2.0 / PI).powi(2);
        let s = m as f64 + 1.0;
        let beta = 4f64.powf(-s) * (hurwitz_zeta(s, 0.25) - hurwitz_zeta(s, 0.75));
        e[m] = if k % 2 == 1 { -2.0 } else { 2.0 } * ratio * beta;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygamma_values() {
        // ψ'(1) = π²/6, ψ''(1) = −2ζ(3), ψ'(1/2) = π²/2.
        assert!((polygamma(1, 1.0) - PI * PI / 6.0).abs() < 1e-14);
        assert!((polygamma(2, 1.0) + 2.0 * 1.2020569031595942).abs() < 1e-13);
        assert!((polygamma(1, 0.5) - PI * PI / 2.0).abs() < 1e-13);
        // Recurrence across the poles: ψ'(x) = ψ'(x + 1) + 1/x².
        let x = -2.3;
        assert!((polygamma(1, x) - polygamma(1, x + 1.0) - 1.0 / (x * x)).abs() < 1e-10);
        assert!(polygamma(1, -2.0).is_nan());
    }

    #[test]
    fn bernoulli_and_euler_numbers() {
        let b = bernoulli(12);
        let want = [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0, -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0];
        for (g, w) in b.iter().zip(want) {
            assert!((g - w).abs() < 1e-15 * w.abs().max(1.0), "{g} vs {w}");
        }
        let e = euler(10);
        let want = [1.0, 0.0, -1.0, 0.0, 5.0, 0.0, -61.0, 0.0, 1385.0, 0.0, -50521.0];
        for (g, w) in e.iter().zip(want) {
            assert!((g - w).abs() < 1e-12 * w.abs().max(1.0), "{g} vs {w}");
        }
    }

    #[test]
    fn multigammaln_reduces_to_ln_gamma() {
        assert!((multigammaln(3.5, 1) - ln_gamma(3.5)).abs() < 1e-14);
        // d = 2: ln π / 2 + ln Γ(a) + ln Γ(a − 1/2).
        assert!((multigammaln(2.0, 2) - (0.5 * PI.ln() + ln_gamma(2.0) + ln_gamma(1.5))).abs() < 1e-14);
    }
}
