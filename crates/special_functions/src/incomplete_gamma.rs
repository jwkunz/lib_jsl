//! Incomplete gamma function.

use crate::gamma::ln_gamma;

const EPS: f64 = 1e-16;
const FPMIN: f64 = f64::MIN_POSITIVE / EPS;

fn max_iter(a: f64) -> usize {
    1000 + (20.0 * a.sqrt()) as usize
}

/// Series for the regularized lower function P(a, x); best for `x < a + 1`.
fn series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..max_iter(a) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued fraction (modified Lentz) for the regularized upper function Q(a, x); `x >= a + 1`.
fn continued_fraction(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..max_iter(a) {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// The regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a), for `a > 0`, `x >= 0`.
///
/// Returns `NaN` for invalid arguments.
pub fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if !(a > 0.0) || x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return 1.0;
    }
    if x < a + 1.0 {
        series(a, x)
    } else {
        1.0 - continued_fraction(a, x)
    }
}

/// The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x), computed without
/// cancellation in the tail.
pub fn incomplete_gamma_upper(a: f64, x: f64) -> f64 {
    if !(a > 0.0) || x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x < a + 1.0 {
        1.0 - series(a, x)
    } else {
        continued_fraction(a, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_equals_one_is_exponential() {
        for &x in &[0.1, 0.5, 1.0, 3.0, 10.0] {
            assert!((incomplete_gamma(1.0, x) - (1.0 - (-x).exp())).abs() < 1e-14);
            assert!((incomplete_gamma_upper(1.0, x) - (-x).exp()).abs() < 1e-14);
        }
    }

    #[test]
    fn known_values() {
        // P(2, x) = 1 - (1 + x) e^-x
        let x = 2.5f64;
        assert!((incomplete_gamma(2.0, x) - (1.0 - (1.0 + x) * (-x).exp())).abs() < 1e-14);
        // P(0.5, 1) = erf(1)
        assert!((incomplete_gamma(0.5, 1.0) - 0.8427007929497149).abs() < 1e-14);
        assert!((incomplete_gamma(5.0, 5.0) - 0.5595067149347875).abs() < 1e-13);
    }

    #[test]
    fn complementarity_and_tails() {
        for &(a, x) in &[(0.3, 0.2), (4.0, 9.0), (50.0, 40.0), (50.0, 70.0)] {
            assert!((incomplete_gamma(a, x) + incomplete_gamma_upper(a, x) - 1.0).abs() < 1e-13);
        }
        // Deep upper tail keeps relative accuracy: Q(1, 50) = e^-50.
        assert!((incomplete_gamma_upper(1.0, 50.0) / (-50.0f64).exp() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn invalid() {
        assert!(incomplete_gamma(-1.0, 1.0).is_nan());
        assert!(incomplete_gamma(1.0, -1.0).is_nan());
        assert_eq!(incomplete_gamma(2.0, 0.0), 0.0);
    }
}
