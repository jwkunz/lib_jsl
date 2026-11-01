//! Gamma function and related functions.
//!
//! See the book entry [The gamma function and related functions](https://jwkunz.github.io/lib_jsl/book/special_functions/gamma.html#the-gamma-function).

use std::f64::consts::PI;

const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Sum of the Lanczos series for argument `x - 1` (i.e. already shifted).
fn lanczos_sum(x: f64) -> f64 {
    let mut a = LANCZOS_COEFFS[0];
    for (i, &c) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    a
}

// ANCHOR: gamma
/// The gamma function Γ(x) for real `x`.
///
/// Uses a Lanczos approximation (relative error about 1e-15) with the reflection formula for
/// `x < 0.5`. Returns `NaN` at the poles (zero and the negative integers) and `+inf` on overflow
/// (`x > 171.62`). Positive integers are evaluated exactly as `(x-1)!`.
pub fn gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x == x.floor() && x <= 171.0 {
        let mut r = 1.0;
        let mut k = 2.0;
        while k < x {
            r *= k;
            k += 1.0;
        }
        return r;
    }
    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma(1.0 - x));
    }
    if x > 171.7 {
        return f64::INFINITY;
    }
    let x = x - 1.0;
    let t = x + LANCZOS_G + 0.5;
    // Split the power to avoid premature overflow of t^(x+0.5).
    let half = t.powf(0.5 * (x + 0.5));
    (2.0 * PI).sqrt() * half * (-t).exp() * half * lanczos_sum(x)
}
// ANCHOR_END: gamma

/// The natural logarithm of |Γ(x)|. Returns `+inf` at the poles.
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    if x < 0.5 {
        return (PI / (PI * x).sin().abs()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + lanczos_sum(x).ln()
}

/// The digamma function ψ(x) = Γ'(x)/Γ(x). Returns `NaN` at the poles.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || (x <= 0.0 && x == x.floor()) {
        return f64::NAN;
    }
    if x < 0.0 {
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut x = x;
    let mut acc = 0.0;
    while x < 10.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // Asymptotic series with Bernoulli numbers.
    let series = inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2
                        * (1.0 / 252.0
                            - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0 - inv2 * 691.0 / 32760.0)))));
    acc + x.ln() - 0.5 * inv - series
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn integers_and_half_integers() {
        assert_eq!(gamma(1.0), 1.0);
        assert_eq!(gamma(5.0), 24.0);
        assert!(close(gamma(0.5), PI.sqrt(), 1e-14));
        assert!(close(gamma(1.5), 0.5 * PI.sqrt(), 1e-14));
        assert!(close(gamma(-0.5), -2.0 * PI.sqrt(), 1e-14));
    }

    #[test]
    fn non_integer_values() {
        assert!(close(gamma(0.1), 9.513507698668732, 1e-14));
        assert!(close(gamma(3.7), 4.170651783796603, 1e-14));
    }

    #[test]
    fn poles_and_overflow() {
        assert!(gamma(0.0).is_nan());
        assert!(gamma(-3.0).is_nan());
        assert!(gamma(200.0).is_infinite());
    }

    #[test]
    fn recurrence() {
        for &x in &[0.3, 1.7, 5.2, 20.5, 60.1] {
            assert!(close(gamma(x + 1.0), x * gamma(x), 1e-13));
        }
    }

    #[test]
    fn log_gamma() {
        assert!(close(ln_gamma(0.5), 0.5 * PI.ln(), 1e-14));
        assert!(close(ln_gamma(10.0), 12.801827480081469, 1e-14));
        assert!(close(ln_gamma(1000.0), 5905.220423209181, 1e-14));
        assert!(close(ln_gamma(-0.5), (2.0 * PI.sqrt()).ln(), 1e-14));
    }

    #[test]
    fn digamma_values() {
        let euler = 0.5772156649015329;
        assert!(close(digamma(1.0), -euler, 1e-13));
        assert!(close(digamma(0.5), -euler - 2.0 * 2f64.ln(), 1e-13));
        assert!(close(digamma(10.0), 2.251752589066721, 1e-13));
        assert!(close(digamma(-0.5), 0.03648997397857652, 1e-12));
    }
}
