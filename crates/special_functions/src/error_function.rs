//! Error function.

use crate::incomplete_gamma::{incomplete_gamma, incomplete_gamma_upper};
use std::f64::consts::PI;

/// The error function erf(x) = 2/√π ∫₀ˣ e^(-t²) dt.
pub fn error_function(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let r = incomplete_gamma(0.5, x * x);
    if x < 0.0 { -r } else { r }
}

/// The complementary error function erfc(x) = 1 - erf(x), accurate in the upper tail.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        incomplete_gamma_upper(0.5, x * x)
    } else {
        2.0 - incomplete_gamma_upper(0.5, x * x)
    }
}

/// The inverse error function: returns `x` such that `erf(x) = y`, for `y` in `(-1, 1)`.
///
/// Returns ±infinity at ±1 and `NaN` outside `[-1, 1]`.
pub fn erf_inv(y: f64) -> f64 {
    if y.is_nan() || y.abs() > 1.0 {
        return f64::NAN;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }
    if y == -1.0 {
        return f64::NEG_INFINITY;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y < 0.0 {
        return -erf_inv(-y);
    }
    // Winitzki initial guess, refined with Halley's method.
    let a = 0.147;
    let l = (1.0 - y * y).ln();
    let t = 2.0 / (PI * a) + 0.5 * l;
    let mut x = ((t * t - l / a).sqrt() - t).sqrt();
    for _ in 0..8 {
        // Solve in the better-conditioned form for the tail.
        let f = if y > 0.5 { (1.0 - y) - erfc(x) } else { error_function(x) - y };
        let d = 2.0 / PI.sqrt() * (-x * x).exp();
        if d == 0.0 {
            break;
        }
        let step = f / (d + x * f); // Halley: f / (f' - f f''/(2 f')), with f''/f' = -2x
        x -= step;
        if step.abs() < 1e-16 * x.abs() {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_values() {
        assert_eq!(error_function(0.0), 0.0);
        assert!((error_function(0.5) - 0.5204998778130465).abs() < 1e-15);
        assert!((error_function(1.0) - 0.8427007929497149).abs() < 1e-15);
        assert!((error_function(2.0) - 0.9953222650189527).abs() < 1e-15);
        assert!((error_function(-1.0) + 0.8427007929497149).abs() < 1e-15);
        assert_eq!(error_function(30.0), 1.0);
    }

    #[test]
    fn erfc_tail() {
        assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-15);
        assert!((erfc(5.0) / 1.537_459_794_428_035e-12 - 1.0).abs() < 1e-12);
        assert!((erfc(-1.0) - 1.8427007929497148).abs() < 1e-15);
    }

    #[test]
    fn inverse_round_trip() {
        for &y in &[-0.999, -0.7, -0.1, 1e-8, 0.3, 0.5, 0.9, 0.999999, 1.0 - 1e-12] {
            let x = erf_inv(y);
            assert!((error_function(x) - y).abs() < 1e-14, "y={y}");
        }
        assert!((erf_inv(0.5) - 0.4769362762044699).abs() < 1e-14);
        assert!(erf_inv(1.0).is_infinite());
        assert!(erf_inv(1.5).is_nan());
    }
}
