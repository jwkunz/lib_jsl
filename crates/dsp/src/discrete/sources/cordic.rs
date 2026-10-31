//! CORDIC sine/cosine.
//!
//! The COordinate Rotation DIgital Computer computes sine and cosine with only shifts and adds in
//! hardware: a vector is rotated toward the target angle by a fixed sequence of micro-rotations of
//! angle `atan(2^-i)`, each chosen to be clockwise or counter-clockwise. This implementation uses
//! floating-point arithmetic to show the algorithm (and to give a drop-in sine kernel for the NCO
//! sources); the result converges by about one bit per iteration.

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// `2π - fl(2π)`: the part of 2π lost when it is rounded to a double, for accurate reduction.
const TAU_LOW: f64 = 2.4492935982947064e-16;
use std::sync::OnceLock;

/// A CORDIC rotator with a fixed number of micro-rotations.
#[derive(Clone, Debug)]
pub struct Cordic {
    angles: Vec<f64>,
    inv_gain: f64,
}

impl Cordic {
    /// Iterations used by [`cordic_sin_cos`]; accurate to about 1e-15.
    pub const DEFAULT_ITERATIONS: u32 = 50;

    /// Creates a rotator with `iterations` micro-rotations (1 to 60). The error is roughly
    /// `2^-iterations`.
    pub fn new(iterations: u32) -> Result<Cordic, ErrorsJSL> {
        if !(1..=60).contains(&iterations) {
            return Err(ErrorsJSL::InvalidInputRange("CORDIC iterations must be between 1 and 60."));
        }
        let angles: Vec<f64> = (0..iterations).map(|i| 0.5f64.powi(i as i32).atan()).collect();
        let gain: f64 = (0..iterations).map(|i| (1.0 + 0.25f64.powi(i as i32)).sqrt()).product();
        Ok(Cordic { angles, inv_gain: 1.0 / gain })
    }

    pub fn iterations(&self) -> u32 {
        self.angles.len() as u32
    }

    /// The constant gain `K = Π sqrt(1 + 2^-2i)` that the micro-rotations add (about 1.6468).
    pub fn gain(&self) -> f64 {
        1.0 / self.inv_gain
    }

    /// Returns `(sin(angle), cos(angle))` for an angle in radians (any real value).
    pub fn sin_cos(&self, angle: f64) -> (f64, f64) {
        // Reduce to (-π/2, π/2], where the micro-rotations converge, tracking the cosine's sign.
        let k = (angle / TAU).round();
        let mut a = (angle - k * TAU) - k * TAU_LOW;
        let mut cos_sign = 1.0;
        if a > FRAC_PI_2 {
            a = PI - a;
            cos_sign = -1.0;
        } else if a < -FRAC_PI_2 {
            a = -PI - a;
            cos_sign = -1.0;
        }
        let (mut x, mut y, mut z) = (self.inv_gain, 0.0, a);
        let mut shift = 1.0;
        for &theta in &self.angles {
            let d = if z >= 0.0 { 1.0 } else { -1.0 };
            let (nx, ny) = (x - d * y * shift, y + d * x * shift);
            x = nx;
            y = ny;
            z -= d * theta;
            shift *= 0.5;
        }
        (y, cos_sign * x)
    }
}

/// Returns `(sin(angle), cos(angle))` using CORDIC with [`Cordic::DEFAULT_ITERATIONS`].
pub fn cordic_sin_cos(angle: f64) -> (f64, f64) {
    static DEFAULT: OnceLock<Cordic> = OnceLock::new();
    DEFAULT.get_or_init(|| Cordic::new(Cordic::DEFAULT_ITERATIONS).expect("valid iteration count")).sin_cos(angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_error(c: &Cordic) -> f64 {
        (-400..=400)
            .map(|k| k as f64 * 0.0173)
            .map(|a| {
                let (s, co) = c.sin_cos(a);
                (s - a.sin()).abs().max((co - a.cos()).abs())
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn default_accuracy_over_many_angles() {
        for k in -2000..=2000 {
            let a = k as f64 * 0.0517;
            let (s, c) = cordic_sin_cos(a);
            let tol = 2e-15 + 1e-16 * a.abs();
            assert!((s - a.sin()).abs() < tol, "sin({a}): {}", (s - a.sin()).abs());
            assert!((c - a.cos()).abs() < tol, "cos({a}): {}", (c - a.cos()).abs());
        }
        assert!(cordic_sin_cos(0.0).0.abs() < 1e-15);
        let (s, c) = cordic_sin_cos(std::f64::consts::FRAC_PI_2);
        assert!((s - 1.0).abs() < 1e-15 && c.abs() < 1e-15);
        let (s, c) = cordic_sin_cos(std::f64::consts::PI);
        assert!(s.abs() < 1e-15 && (c + 1.0).abs() < 1e-15);
        // Large arguments are reduced.
        let (s, _) = cordic_sin_cos(1000.0 * TAU + 0.3);
        assert!((s - 0.3f64.sin()).abs() < 1e-11);
    }

    #[test]
    fn error_shrinks_with_iterations() {
        let errs: Vec<f64> = [4u32, 8, 16, 24, 32].iter().map(|&n| max_error(&Cordic::new(n).unwrap())).collect();
        for w in errs.windows(2) {
            assert!(w[1] < w[0] * 0.5, "{errs:?}");
        }
        assert!(errs[0] < 0.2 && errs[2] < 1e-4 && errs[4] < 1e-8, "{errs:?}");
    }

    #[test]
    fn gain_and_identity() {
        let c = Cordic::new(50).unwrap();
        assert!((c.gain() - 1.646760258121066).abs() < 1e-12);
        assert_eq!(c.iterations(), 50);
        for k in 0..100 {
            let (s, co) = c.sin_cos(k as f64 * 0.31);
            assert!((s * s + co * co - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn invalid_iteration_counts() {
        assert!(Cordic::new(0).is_err());
        assert!(Cordic::new(61).is_err());
        assert!(Cordic::new(1).is_ok());
    }
}
