//! Airy functions.
//!
//! A Maclaurin series is used for small |x|; otherwise the functions are expressed through
//! Bessel functions of order 1/3.

use crate::bessel_functions_fractional::{bessel_ik, bessel_jy};
use std::f64::consts::PI;

/// Ai(0)
const AI0: f64 = 0.355_028_053_887_817_2;
/// -Ai'(0)
const AIP0: f64 = 0.258_819_403_792_806_8;
const SERIES_LIMIT: f64 = 2.0;

/// Returns the Maclaurin series pair `(f(x), g(x))` such that Ai = AI0 f - AIP0 g and
/// Bi = √3 (AI0 f + AIP0 g).
fn series_fg(x: f64) -> (f64, f64) {
    let x3 = x * x * x;
    let mut f = 1.0;
    let mut g = x;
    let mut tf = 1.0;
    let mut tg = x;
    for k in 1..200 {
        let k3 = 3.0 * k as f64;
        tf *= x3 / ((k3 - 1.0) * k3);
        tg *= x3 / (k3 * (k3 + 1.0));
        f += tf;
        g += tg;
        if tf.abs() < 1e-17 * f.abs() && tg.abs() < 1e-17 * g.abs().max(1e-300) {
            break;
        }
    }
    (f, g)
}

/// The Airy function Ai(x).
pub fn airy_ai(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() <= SERIES_LIMIT {
        let (f, g) = series_fg(x);
        return AI0 * f - AIP0 * g;
    }
    if x > 0.0 {
        let z = 2.0 / 3.0 * x * x.sqrt();
        if z > 745.0 {
            return 0.0;
        }
        let k = bessel_ik(1.0 / 3.0, z).1;
        (x / 3.0).sqrt() / PI * k
    } else {
        let ax = -x;
        let z = 2.0 / 3.0 * ax * ax.sqrt();
        let (j, y) = bessel_jy(1.0 / 3.0, z);
        0.5 * ax.sqrt() * (j - y / 3f64.sqrt())
    }
}

/// The Airy function Bi(x).
pub fn airy_bi(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() <= SERIES_LIMIT {
        let (f, g) = series_fg(x);
        return 3f64.sqrt() * (AI0 * f + AIP0 * g);
    }
    if x > 0.0 {
        let z = 2.0 / 3.0 * x * x.sqrt();
        if z > 700.0 {
            return f64::INFINITY;
        }
        let (i, k) = bessel_ik(1.0 / 3.0, z);
        // I_{-1/3} = I_{1/3} + (2/π) sin(π/3) K_{1/3}
        let i_neg = i + 2.0 / PI * (PI / 3.0).sin() * k;
        (x / 3.0).sqrt() * (i_neg + i)
    } else {
        let ax = -x;
        let z = 2.0 / 3.0 * ax * ax.sqrt();
        let (j, y) = bessel_jy(1.0 / 3.0, z);
        -0.5 * ax.sqrt() * (y + j / 3f64.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn ai_values() {
        assert!(close(airy_ai(0.0), 0.3550280538878172, 1e-15));
        assert!(close(airy_ai(1.0), 0.13529241631288141, 1e-14));
        assert!(close(airy_ai(-1.0), 0.5355608832923521, 1e-14));
        assert!(close(airy_ai(3.0), 0.006591139357460719, 1e-13));
        assert!(close(airy_ai(-5.0), 0.3507610090241143, 1e-13));
        assert!(close(airy_ai(2.5), 0.01572592338047049, 1e-13));
        assert!(close(airy_ai(-2.5), -0.11232506769296609, 1e-13));
        assert!(close(airy_ai(10.0), 1.1047532552898686e-10, 1e-12));
        assert!(close(airy_ai(-10.0), 0.04024123848644319, 1e-12));
        assert_eq!(airy_ai(200.0), 0.0);
    }

    #[test]
    fn bi_values() {
        assert!(close(airy_bi(0.0), 0.6149266274460007, 1e-14));
        assert!(close(airy_bi(1.0), 1.2074235949528713, 1e-14));
        assert!(close(airy_bi(-1.0), 0.10399738949694461, 1e-13));
        assert!(close(airy_bi(3.0), 14.037328963730232, 1e-13));
        assert!(close(airy_bi(-5.0), -0.13836913490160058, 1e-12));
        assert!(close(airy_bi(10.0), 455641153.5482251, 1e-12));
        assert!(close(airy_bi(-10.0), -0.31467982964383864, 1e-12));
    }

    #[test]
    fn continuity_across_switch() {
        for &b in &[SERIES_LIMIT, -SERIES_LIMIT] {
            let e = 1e-9;
            assert!((airy_ai(b + e) - airy_ai(b - e)).abs() < 1e-8);
            assert!((airy_bi(b + e) - airy_bi(b - e)).abs() < 1e-8);
        }
    }
}
