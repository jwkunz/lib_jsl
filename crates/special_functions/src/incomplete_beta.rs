//! Incomplete beta function.

use crate::gamma::ln_gamma;

const EPS: f64 = 1e-16;
const FPMIN: f64 = f64::MIN_POSITIVE / EPS;

/// Continued fraction for the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    let max_iter = 1000 + (10.0 * a.max(b).sqrt()) as usize * 10;
    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
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
    h
}

/// The regularized incomplete beta function I_x(a, b) for `a, b > 0` and `0 <= x <= 1`.
///
/// Returns `NaN` for invalid arguments.
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if !(a > 0.0) || !(b > 0.0) || x.is_nan() || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms() {
        // I_x(1, 1) = x; I_x(a, 1) = x^a; I_x(1, b) = 1 - (1-x)^b
        assert!((incomplete_beta(1.0, 1.0, 0.37) - 0.37).abs() < 1e-14);
        assert!((incomplete_beta(2.5, 1.0, 0.6) - 0.6f64.powf(2.5)).abs() < 1e-14);
        assert!((incomplete_beta(1.0, 3.0, 0.2) - (1.0 - 0.8f64.powi(3))).abs() < 1e-14);
        // I_x(2, 2) = 3x^2 - 2x^3
        let x = 0.3;
        assert!((incomplete_beta(2.0, 2.0, x) - (3.0 * x * x - 2.0 * x * x * x)).abs() < 1e-14);
    }

    #[test]
    fn symmetry_and_median() {
        for &(a, b, x) in &[(2.0, 5.0, 0.3), (0.5, 0.5, 0.9), (10.0, 3.0, 0.8), (40.0, 60.0, 0.45)] {
            let s = incomplete_beta(a, b, x) + incomplete_beta(b, a, 1.0 - x);
            assert!((s - 1.0).abs() < 1e-13);
        }
        assert!((incomplete_beta(3.0, 3.0, 0.5) - 0.5).abs() < 1e-14);
        // arcsine distribution: I_x(1/2, 1/2) = 2/pi asin(sqrt(x))
        let x: f64 = 0.7;
        let want = 2.0 / std::f64::consts::PI * x.sqrt().asin();
        assert!((incomplete_beta(0.5, 0.5, x) - want).abs() < 1e-14);
    }

    #[test]
    fn invalid() {
        assert!(incomplete_beta(-1.0, 1.0, 0.5).is_nan());
        assert!(incomplete_beta(1.0, 1.0, 1.5).is_nan());
        assert_eq!(incomplete_beta(1.0, 1.0, 0.0), 0.0);
        assert_eq!(incomplete_beta(1.0, 1.0, 1.0), 1.0);
    }
}
