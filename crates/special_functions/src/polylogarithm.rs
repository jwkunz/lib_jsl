//! The polylogarithm `Li_s(z) = Σ_{k≥1} z^k / k^s` for real order and real argument.

use crate::fermi_dirac_integrals::fermi_dirac_integral;
use crate::gamma::gamma;
use crate::zeta::zeta;

/// `Li_s(−1) = −η(s) = −(1 − 2^{1−s}) ζ(s)`, valid for every real `s` (with `Li_1(−1) = −ln 2`).
fn at_minus_one(s: f64) -> f64 {
    if s == 1.0 {
        return -std::f64::consts::LN_2;
    }
    ((1.0 - s) * std::f64::consts::LN_2).exp_m1() * zeta(s)
}

fn direct_series(s: f64, z: f64) -> f64 {
    let mut sum = 0.0;
    let mut power = 1.0;
    for k in 1..2000 {
        power *= z;
        let term = power / (k as f64).powf(s);
        sum += term;
        if term.abs() <= 1e-17 * sum.abs() {
            break;
        }
    }
    sum
}

/// The expansion in `μ = ln z` about `z = 1`, valid for `|μ| < 2π` (used for `½ < z < 1`).
fn log_series(s: f64, z: f64) -> f64 {
    let mu = z.ln();
    let integer = s.fract() == 0.0 && s >= 1.0;
    let n = s as i64;
    let mut sum = if integer {
        // μ^{n−1}/(n−1)! (H_{n−1} − ln(−μ)) replaces the singular k = n−1 term.
        let harmonic: f64 = (1..n).map(|k| 1.0 / k as f64).sum();
        let factorial: f64 = (1..n).map(|k| k as f64).product();
        mu.powi((n - 1) as i32) / factorial * (harmonic - (-mu).ln())
    } else {
        gamma(1.0 - s) * (-mu).powf(s - 1.0)
    };
    let mut power = 1.0; // μ^k / k!
    for k in 0..60 {
        if k > 0 {
            power *= mu / k as f64;
        }
        if integer && k == n - 1 {
            continue;
        }
        let term = zeta(s - k as f64) * power;
        sum += term;
        // ζ vanishes at the negative even integers, so only stop on a non-zero small term.
        if k > 2 && term != 0.0 && term.abs() <= 1e-17 * sum.abs() {
            break;
        }
    }
    sum
}

/// `Li_{−n}(z) = Σ_{k=0}^{n} k! S(n+1, k+1) w^{k+1}` with `w = z/(1−z)` and Stirling numbers of the
/// second kind.
fn negative_integer_order(n: usize, z: f64) -> f64 {
    let w = z / (1.0 - z);
    // Row n+1 of the Stirling numbers of the second kind, scaled by k!: a(m, k) = k! S(m, k)
    // satisfies a(m+1, k) = k (a(m, k) + a(m, k−1)).
    let mut row = vec![0.0; n + 2];
    row[1] = 1.0; // m = 1
    for _ in 1..=n {
        for k in (1..row.len()).rev() {
            row[k] = k as f64 * (row[k] + row[k - 1]);
        }
    }
    // row[k] = k! S(n+1, k); the sum needs (k−1)! S(n+1, k) = row[k] / k.
    (1..=n + 1).map(|k| row[k] / k as f64 * w.powi(k as i32)).sum()
}

/// The polylogarithm `Li_s(z)` for real `s` and real `z ≤ 1`.
///
/// * `|z| ≤ ½`: the defining series.
/// * `½ < z < 1`: the expansion in `ln z` with zeta coefficients.
/// * `−1 ≤ z < −½`: the duplication formula `Li_s(z) = 2^{1−s} Li_s(z²) − Li_s(−z)`.
/// * `z < −1`: the inversion formula for integer `s ≥ 1`, and `Li_s(−e^μ) = −F_{s−1}(μ)` (the
///   complete Fermi-Dirac integral) for other `s > 0`.
/// * Integer `s ≤ 1`: the closed forms `−ln(1−z)`, `z/(1−z)` and rational functions.
///
/// `Li_s(1) = ζ(s)` for `s > 1` and `+∞` otherwise. Returns NaN for `z > 1` (the value is complex)
/// and for `z < −1` with non-integer `s ≤ 0`. Orders within about 1e-6 of an integer lose
/// accuracy in the `ln z` expansion for `½ < z < 1`.
pub fn polylog(s: f64, z: f64) -> f64 {
    if s.is_nan() || z.is_nan() || z > 1.0 {
        return f64::NAN;
    }
    if z == 0.0 {
        return 0.0;
    }
    if z == 1.0 {
        return if s > 1.0 { zeta(s) } else { f64::INFINITY };
    }
    let integer = s.fract() == 0.0;
    if integer && s <= 1.0 {
        return match s as i64 {
            1 => -(-z).ln_1p(),
            0 => z / (1.0 - z),
            n => negative_integer_order((-n) as usize, z),
        };
    }
    if z == -1.0 {
        return at_minus_one(s);
    }
    if z.abs() <= 0.5 {
        return direct_series(s, z);
    }
    if z > 0.5 {
        return log_series(s, z);
    }
    if z >= -1.0 {
        return 2f64.powf(1.0 - s) * polylog(s, z * z) - polylog(s, -z);
    }
    // z < −1.
    let l = (-z).ln();
    if integer {
        let n = s as i64;
        let factorial = |m: i64| (1..=m).map(|k| k as f64).product::<f64>();
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let mut value = -sign * polylog(s, 1.0 / z) - l.powi(n as i32) / factorial(n);
        for k in 1..=n / 2 {
            value += 2.0 * l.powi((n - 2 * k) as i32) / factorial(n - 2 * k) * at_minus_one((2 * k) as f64);
        }
        return value;
    }
    if s > 0.0 {
        return -fermi_dirac_integral(s - 1.0, l);
    }
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{LN_2, PI};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    #[test]
    fn closed_forms_of_integer_orders() {
        let zeta3 = 1.202_056_903_159_594_3;
        assert!(close(polylog(2.0, 0.5), PI * PI / 12.0 - LN_2 * LN_2 / 2.0, 1e-15));
        assert!(close(polylog(2.0, -1.0), -PI * PI / 12.0, 1e-15));
        assert!(close(polylog(3.0, 0.5), 7.0 / 8.0 * zeta3 - PI * PI / 12.0 * LN_2 + LN_2.powi(3) / 6.0, 1e-14));
        assert!(close(polylog(2.0, 1.0), PI * PI / 6.0, 1e-15) && polylog(1.0, 1.0).is_infinite());
        for &z in &[-7.0, -0.9, -0.3, 0.2, 0.75, 0.99] {
            assert!(close(polylog(1.0, z), -(1.0 - z).ln(), 1e-14));
            assert!(close(polylog(0.0, z), z / (1.0 - z), 1e-14));
            assert!(close(polylog(-2.0, z), z * (1.0 + z) / (1.0 - z).powi(3), 1e-13));
            assert!(close(polylog(-3.0, z), z * (1.0 + 4.0 * z + z * z) / (1.0 - z).powi(4), 1e-13));
        }
        // Euler's reflection Li_2(z) + Li_2(1−z) = π²/6 − ln z ln(1−z) and Landen's identity
        // Li_2(z) + Li_2(z/(z−1)) = −½ ln²(1−z), which reaches z < −1.
        for &z in &[0.6, 0.9, 0.999] {
            assert!(close(polylog(2.0, z) + polylog(2.0, 1.0 - z), PI * PI / 6.0 - f64::ln(z) * f64::ln(1.0 - z), 1e-14), "z={z}");
        }
        for &z in &[-3.0, -20.0, -0.7] {
            let w: f64 = z / (z - 1.0);
            assert!(close(polylog(2.0, z), -0.5 * f64::ln(1.0 - z).powi(2) - polylog(2.0, w), 1e-13), "z={z}");
        }
        // The Li_4 inversion formula written out with its constants, Li_2(−1) = −π²/12 and
        // Li_4(−1) = −7π⁴/720, against the defining series at 1/z.
        let z = -4.0;
        let l = f64::ln(4.0);
        let want = -direct_series(4.0, 1.0 / z) - l.powi(4) / 24.0 - l * l * PI * PI / 12.0 - 7.0 * PI.powi(4) / 360.0;
        assert!(close(polylog(4.0, z), want, 1e-13));
    }

    #[test]
    fn non_integer_orders_match_the_series_and_integral() {
        // Direct summation is slow but converges for |z| < 1.
        let brute = |s: f64, z: f64| (1..20_000).map(|k| z.powi(k) / (k as f64).powf(s)).sum::<f64>();
        for &(s, z) in &[(1.5, 0.7), (0.5, 0.95), (2.5, -0.8), (-0.5, 0.6), (-1.5, -0.65), (3.3, 0.55)] {
            assert!(close(polylog(s, z), brute(s, z), 1e-12), "s={s} z={z}: {} {}", polylog(s, z), brute(s, z));
        }
        // Continuity across the switch between the series and the ln z expansion.
        assert!(close(polylog(1.5, 0.5 - 1e-12), polylog(1.5, 0.5 + 1e-12), 1e-10));
        // z < −1 from the integral representation Li_s(z) = z/Γ(s) ∫ t^{s−1}/(e^t − z) dt.
        let (s, z) = (2.5, -5.0);
        let m = 200_000;
        let h = 60.0 / m as f64;
        let integral: f64 = (0..=m)
            .map(|i| {
                let t = i as f64 * h;
                let w = if i == 0 || i == m { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
                w * t.powf(s - 1.0) / (t.exp() - z)
            })
            .sum::<f64>()
            * h
            / 3.0;
        assert!(close(polylog(s, z), z / gamma(s) * integral, 1e-9));
        assert!(polylog(2.0, 1.5).is_nan() && polylog(-0.5, -3.0).is_nan());
        assert!(close(polylog(0.5, -1.0), -(1.0 - 2f64.sqrt()) * zeta(0.5), 1e-14));
    }
}
