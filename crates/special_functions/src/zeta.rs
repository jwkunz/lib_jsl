//! The Riemann and Hurwitz zeta functions for real arguments.

use crate::gamma::gamma;
use std::f64::consts::{LN_2, PI};

/// Borwein's coefficients for `n` terms of the accelerated alternating series (error about
/// `3 / (3 + √8)ⁿ`, below 1e-17 for `n = 24`).
const BORWEIN_TERMS: usize = 24;

fn borwein_d() -> [f64; BORWEIN_TERMS + 1] {
    let n = BORWEIN_TERMS as f64;
    let mut d = [0.0; BORWEIN_TERMS + 1];
    let mut term = 1.0; // n (n+i−1)! 4^i / ((n−i)! (2i)!) at i = 0
    let mut sum = term;
    d[0] = sum;
    for (i, slot) in d.iter_mut().enumerate().skip(1) {
        let fi = i as f64;
        term *= 4.0 * (n + fi - 1.0) * (n - fi + 1.0) / ((2.0 * fi) * (2.0 * fi - 1.0));
        sum += term;
        *slot = sum;
    }
    d
}

/// The Dirichlet eta function `η(s) = Σ (−1)^{k−1} / k^s` for `s ≥ 0` by Borwein's algorithm.
fn eta(s: f64) -> f64 {
    let d = borwein_d();
    let dn = d[BORWEIN_TERMS];
    let sum: f64 = (0..BORWEIN_TERMS)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * (d[k] - dn) / ((k + 1) as f64).powf(s)
        })
        .sum();
    -sum / dn
}

/// The Riemann zeta function `ζ(s)` for real `s ≠ 1` (`+∞` at the pole).
///
/// For `s ≥ 0` it is `η(s) / (1 − 2^{1−s})` with Borwein's alternating-series acceleration
/// (relative error near 1e-15); for `s < 0` the functional equation
/// `ζ(s) = 2^s π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)` is used, with exact zeros at the negative even
/// integers. Overflows to ±∞ below about `s = −170`.
pub fn zeta(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY;
    }
    if s == f64::INFINITY {
        return 1.0;
    }
    if s < 0.0 {
        if s == f64::NEG_INFINITY {
            return f64::NAN;
        }
        if s.fract() == 0.0 && (s / 2.0).fract() == 0.0 {
            return 0.0;
        }
        return 2f64.powf(s) * PI.powf(s - 1.0) * (PI * s / 2.0).sin() * gamma(1.0 - s) * zeta(1.0 - s);
    }
    if s >= 60.0 {
        return 1.0 + 2f64.powf(-s) + 3f64.powf(-s);
    }
    // 1 − 2^{1−s} without cancellation near s = 1.
    let denominator = -((1.0 - s) * LN_2).exp_m1();
    eta(s) / denominator
}

/// The Hurwitz zeta function `ζ(s, a) = Σ_{k≥0} (k + a)^{−s}` for `s > 1` and `a > 0`, by
/// Euler-Maclaurin summation (relative error near 1e-15). `ζ(s, 1) = ζ(s)`.
pub fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    if !(s > 1.0) || !(a > 0.0) {
        return if s == 1.0 && a > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    // B_{2j} / (2j)! for j = 1..8.
    const B_OVER_FACTORIAL: [f64; 8] = [
        1.0 / 12.0,
        -1.0 / 720.0,
        1.0 / 30240.0,
        -1.0 / 1_209_600.0,
        1.0 / 47_900_160.0,
        -691.0 / 1_307_674_368_000.0,
        1.0 / 74_724_249_600.0,
        -3617.0 / 10_670_622_842_880_000.0,
    ];
    const N: usize = 16;
    let head: f64 = (0..N).map(|k| (k as f64 + a).powf(-s)).sum();
    let x = N as f64 + a;
    let mut tail = x.powf(1.0 - s) / (s - 1.0) + 0.5 * x.powf(-s);
    // Rising factorial s(s+1)…(s+2j−2) times x^{−s−2j+1}.
    let mut factor = s * x.powf(-s - 1.0);
    for (j, b) in B_OVER_FACTORIAL.iter().enumerate() {
        let term = b * factor;
        tail += term;
        if term.abs() < 1e-17 * (head + tail).abs() {
            break;
        }
        let jf = (j + 1) as f64;
        factor *= (s + 2.0 * jf - 1.0) * (s + 2.0 * jf) / (x * x);
    }
    head + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    #[test]
    fn known_values_of_the_riemann_zeta_function() {
        assert!(close(zeta(2.0), PI * PI / 6.0, 1e-15));
        assert!(close(zeta(4.0), PI.powi(4) / 90.0, 1e-15));
        assert!(close(zeta(3.0), 1.202_056_903_159_594_3, 1e-15));
        assert!(close(zeta(0.5), -1.460_354_508_809_586_8, 1e-14));
        assert!(close(zeta(1.5), 2.612_375_348_685_488, 1e-14));
        let eps = 2f64.powi(-30); // exactly representable offset from the pole
        assert!(close(zeta(1.0 + eps), 1.0 / eps + 0.577_215_664_901_532_9, 1e-13)); // Laurent expansion
        assert!(close(zeta(0.0), -0.5, 1e-15));
        assert!(close(zeta(-1.0), -1.0 / 12.0, 1e-14));
        assert!(close(zeta(-3.0), 1.0 / 120.0, 1e-13));
        assert!(close(zeta(-0.5), -0.207_886_224_977_354_57, 1e-13));
        assert_eq!((zeta(-2.0), zeta(-10.0)), (0.0, 0.0));
        assert!(close(zeta(80.0), 1.0, 1e-16) && zeta(1.0).is_infinite() && zeta(f64::NAN).is_nan());
    }

    #[test]
    fn functional_equation_and_hurwitz_relations() {
        // The functional equation holds across s = ½ for the non-reflected branch too.
        for &s in &[0.1, 0.3, 0.7, 0.9] {
            let reflected = 2f64.powf(s) * PI.powf(s - 1.0) * (PI * s / 2.0).sin() * gamma(1.0 - s) * zeta(1.0 - s);
            assert!(close(zeta(s), reflected, 1e-13), "s={s}");
        }
        for &s in &[1.1, 2.0, 3.5, 7.0, 20.0] {
            assert!(close(hurwitz_zeta(s, 1.0), zeta(s), 1e-14), "s={s}");
            // ζ(s, ½) = (2^s − 1) ζ(s), and ζ(s, a) = ζ(s, a + 1) + a^{−s}.
            assert!(close(hurwitz_zeta(s, 0.5), (2f64.powf(s) - 1.0) * zeta(s), 1e-14));
            assert!(close(hurwitz_zeta(s, 0.3), hurwitz_zeta(s, 1.3) + 0.3f64.powf(-s), 1e-14));
        }
        // Trigamma: ψ₁(1/4) = π² + 8G (Catalan's constant), and ζ(2, a) = ψ₁(a).
        assert!(close(hurwitz_zeta(2.0, 0.25), PI * PI + 8.0 * 0.915_965_594_177_219, 1e-14));
        assert!(hurwitz_zeta(0.5, 1.0).is_nan() && hurwitz_zeta(2.0, -1.0).is_nan());
    }
}
