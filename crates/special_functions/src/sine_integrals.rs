//! Sine integrals.

use crate::cosine_integrals::si_ci_positive;
use std::f64::consts::FRAC_PI_2;

/// The sine integral Si(x) = ∫₀ˣ sin t / t dt. Odd in `x`, with limits ±π/2 at infinity.
pub fn sine_integral(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return FRAC_PI_2.copysign(x);
    }
    let s = si_ci_positive(x.abs()).0;
    if x < 0.0 { -s } else { s }
}

/// The auxiliary function si(x) = Si(x) − π/2 (useful for large `x` without cancellation issues
/// in the sum).
pub fn sine_integral_shifted(x: f64) -> f64 {
    sine_integral(x) - FRAC_PI_2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        let cases = [
            (0.5, 0.49310741804306669),
            (1.0, 0.94608307036718301),
            (2.0, 1.6054129768026948),
            (5.0, 1.5499312449446741),
            (10.0, 1.6583475942188740),
        ];
        for &(x, want) in &cases {
            assert!((sine_integral(x) - want).abs() < 1e-14, "Si({x})");
            assert!((sine_integral(-x) + want).abs() < 1e-14);
        }
    }

    #[test]
    fn edges() {
        assert_eq!(sine_integral(0.0), 0.0);
        assert!((sine_integral(f64::INFINITY) - FRAC_PI_2).abs() < 1e-15);
        assert!((sine_integral(1e4) - FRAC_PI_2).abs() < 1e-4);
        assert!((sine_integral(1e-8) - 1e-8).abs() < 1e-20);
        assert!((sine_integral_shifted(0.0) + FRAC_PI_2).abs() < 1e-15);
    }
}
