//! Fixed-point CORDIC sine/cosine.
//!
//! The COordinate Rotation DIgital Computer computes sine and cosine with only shifts and adds: a
//! vector is rotated toward the target angle by a fixed sequence of micro-rotations of angle
//! `atan(2^-i)`, each chosen clockwise or counter-clockwise. This implementation is entirely
//! integer arithmetic, as it would be in hardware:
//!
//! - **Angles** are `u32` binary angles: `2^32` is one full turn, so the value wraps naturally
//!   and the top two bits select the quadrant. Resolution is `2^-32` turn (about 1.5e-9 rad).
//! - **Results** are `i32` in Q1.30 two's complement: `1.0` is `1 << 30`. (They must be signed
//!   because sine and cosine are.)
//! - The micro-rotation angles and the cumulative gain compensation are precomputed integer
//!   tables, and the shifts use round-to-nearest to keep the bias down.
//!
//! The error is dominated by the micro-rotations for few iterations (roughly `2^-iterations`)
//! and bottoms out at about a dozen least-significant bits (1e-8) with the maximum of 30, limited
//! by rounding in the 32-bit angle table and in the shifts.

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::TAU;
use std::sync::OnceLock;

/// Number of fractional bits in the `i32` sine/cosine results.
pub const FRACTIONAL_BITS: u32 = 30;
/// The fixed-point representation of 1.0.
pub const ONE: i32 = 1 << FRACTIONAL_BITS;
/// The maximum number of micro-rotations.
pub const MAX_ITERATIONS: u32 = 30;

/// `atan(2^-i)` as a binary angle (`2^32` per turn).
const ATAN_BAM: [u32; 30] = [
    0x20000000, 0x12E4051E, 0x09FB385B, 0x051111D4, 0x028B0D43,
    0x0145D7E1, 0x00A2F61E, 0x00517C55, 0x0028BE53, 0x00145F2F,
    0x000A2F98, 0x000517CC, 0x00028BE6, 0x000145F3, 0x0000A2FA,
    0x0000517D, 0x000028BE, 0x0000145F, 0x00000A30, 0x00000518,
    0x0000028C, 0x00000146, 0x000000A3, 0x00000051, 0x00000029,
    0x00000014, 0x0000000A, 0x00000005, 0x00000003, 0x00000001,
];

/// `1 / K_n` in Q1.30, where `K_n = Π_{i<n} sqrt(1 + 2^-2i)` is the gain after `n` iterations.
const INV_GAIN_Q30: [i32; 31] = [
    1073741824, 759250125, 679093957, 658817909, 653730436,
    652457347, 652138997, 652059405, 652039507, 652034532,
    652033289, 652032978, 652032900, 652032881, 652032876,
    652032874, 652032874, 652032874, 652032874, 652032874,
    652032874, 652032874, 652032874, 652032874, 652032874,
    652032874, 652032874, 652032874, 652032874, 652032874,
    652032874,
];

/// Converts an angle in radians to a `u32` binary angle (modulo one turn).
pub fn angle_from_radians(radians: f64) -> u32 {
    angle_from_turns(radians / TAU)
}

/// Converts an angle in turns to a `u32` binary angle (modulo one turn).
pub fn angle_from_turns(turns: f64) -> u32 {
    ((turns.rem_euclid(1.0) * 4_294_967_296.0).round() as u64) as u32
}

/// Converts a `u32` binary angle to radians in `[0, 2π)`.
pub fn angle_to_radians(angle: u32) -> f64 {
    angle as f64 / 4_294_967_296.0 * TAU
}

/// Rounding arithmetic right shift.
fn shr_round(v: i32, shift: u32) -> i32 {
    if shift == 0 { v } else { (v + (1 << (shift - 1))) >> shift }
}

/// An integer CORDIC rotator with a fixed number of micro-rotations.
#[derive(Clone, Copy, Debug)]
pub struct Cordic {
    iterations: u32,
}

impl Cordic {
    /// Iterations used by [`cordic_sin_cos`].
    pub const DEFAULT_ITERATIONS: u32 = MAX_ITERATIONS;

    /// Creates a rotator with `iterations` micro-rotations (1 to [`MAX_ITERATIONS`]).
    pub fn new(iterations: u32) -> Result<Cordic, ErrorsJSL> {
        if !(1..=MAX_ITERATIONS).contains(&iterations) {
            return Err(ErrorsJSL::InvalidInputRange("CORDIC iterations must be between 1 and 30."));
        }
        Ok(Cordic { iterations })
    }

    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// The gain `K = Π sqrt(1 + 2^-2i)` that the micro-rotations add (about 1.6468), which the
    /// rotator compensates for by starting from `1/K`.
    pub fn gain(&self) -> f64 {
        (0..self.iterations).map(|i| (1.0 + 0.25f64.powi(i as i32)).sqrt()).product()
    }

    /// Returns `(sin, cos)` of a `u32` binary angle as Q1.30 `i32` values, using integer
    /// arithmetic only.
    pub fn sin_cos_fixed(&self, angle: u32) -> (i32, i32) {
        let inv_gain = INV_GAIN_Q30[self.iterations as usize];
        // The top two bits give the quadrant: start from the gain-compensated unit vector already
        // rotated by that many quarter turns, then rotate by the remaining angle in [0, π/2).
        let (mut x, mut y) = match angle >> 30 {
            0 => (inv_gain, 0),
            1 => (0, inv_gain),
            2 => (-inv_gain, 0),
            _ => (0, -inv_gain),
        };
        let mut z = (angle & 0x3FFF_FFFF) as i32;
        for i in 0..self.iterations {
            let (dx, dy) = (shr_round(y, i), shr_round(x, i));
            let theta = ATAN_BAM[i as usize] as i32;
            if z >= 0 {
                x -= dx;
                y += dy;
                z -= theta;
            } else {
                x += dx;
                y -= dy;
                z += theta;
            }
        }
        (y, x)
    }

    /// Floating-point convenience: `(sin, cos)` of an angle in radians (any real value). The angle
    /// is first quantized to a `u32` binary angle; the result is accurate to about 1e-8.
    pub fn sin_cos(&self, radians: f64) -> (f64, f64) {
        let (s, c) = self.sin_cos_fixed(angle_from_radians(radians));
        (s as f64 / ONE as f64, c as f64 / ONE as f64)
    }
}

/// Returns `(sin(angle), cos(angle))` for an angle in radians, using the integer CORDIC with
/// [`Cordic::DEFAULT_ITERATIONS`] (accurate to about 1e-8).
pub fn cordic_sin_cos(angle: f64) -> (f64, f64) {
    static DEFAULT: OnceLock<Cordic> = OnceLock::new();
    DEFAULT.get_or_init(|| Cordic::new(Cordic::DEFAULT_ITERATIONS).expect("valid iteration count")).sin_cos(angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LSB: f64 = 1.0 / ONE as f64;

    /// Worst error against libm over a dense sweep of binary angles.
    fn max_error(c: &Cordic, step: u32) -> f64 {
        let mut worst = 0.0f64;
        let mut a = 0u32;
        loop {
            let (s, co) = c.sin_cos_fixed(a);
            let r = angle_to_radians(a);
            worst = worst.max((s as f64 * LSB - r.sin()).abs()).max((co as f64 * LSB - r.cos()).abs());
            match a.checked_add(step) {
                Some(n) => a = n,
                None => break,
            }
        }
        worst
    }

    #[test]
    fn tables_match_their_definitions() {
        for (i, &t) in ATAN_BAM.iter().enumerate() {
            let want = (0.5f64.powi(i as i32)).atan() / TAU * 4_294_967_296.0;
            assert!((t as f64 - want).abs() <= 0.5 + 1e-6, "atan table entry {i}");
        }
        for n in 1..=30usize {
            let gain: f64 = (0..n).map(|i| (1.0 + 0.25f64.powi(i as i32)).sqrt()).product();
            assert!((INV_GAIN_Q30[n] as f64 - ONE as f64 / gain).abs() <= 0.5 + 1e-6, "gain table entry {n}");
        }
    }

    #[test]
    fn full_precision_accuracy_over_the_whole_circle() {
        let c = Cordic::new(30).unwrap();
        let err = max_error(&c, 104_729); // ~41k angles, covering all four quadrants
        assert!(err < 16.0 * LSB, "max error {err} ({} LSB)", err / LSB);
    }

    #[test]
    fn quadrant_boundaries_and_symmetry() {
        let c = Cordic::new(30).unwrap();
        let near = |got: i32, want: i32| (got - want).abs() <= 6;
        let (s, co) = c.sin_cos_fixed(0);
        assert!(near(s, 0) && near(co, ONE));
        let (s, co) = c.sin_cos_fixed(1 << 30); // 90 degrees
        assert!(near(s, ONE) && near(co, 0));
        let (s, co) = c.sin_cos_fixed(1 << 31); // 180 degrees
        assert!(near(s, 0) && near(co, -ONE));
        let (s, co) = c.sin_cos_fixed(3 << 30); // 270 degrees
        assert!(near(s, -ONE) && near(co, 0));
        // 45 degrees
        let (s, co) = c.sin_cos_fixed(1 << 29);
        let h = (ONE as f64 * std::f64::consts::FRAC_1_SQRT_2) as i32;
        assert!(near(s, h) && near(co, h));
        // Adding a quarter turn rotates (sin, cos) to (cos, -sin).
        for a in [123_456u32, 987_654_321, 2_000_000_000, 4_000_000_000] {
            let (s0, c0) = c.sin_cos_fixed(a);
            let (s1, c1) = c.sin_cos_fixed(a.wrapping_add(1 << 30));
            assert!((s1 - c0).abs() <= 12 && (c1 + s0).abs() <= 12, "angle {a}");
        }
    }

    #[test]
    fn error_shrinks_with_iterations_down_to_the_fixed_point_floor() {
        let errs: Vec<f64> = [4u32, 8, 12, 16, 20, 24, 30].iter().map(|&n| max_error(&Cordic::new(n).unwrap(), 1_000_003)).collect();
        for w in errs[..5].windows(2) {
            assert!(w[1] < w[0] * 0.2, "{errs:?}");
        }
        assert!(errs[0] < 0.2 && errs[3] < 1e-4 && errs[5] < 2e-6 && errs[6] < 2e-8, "{errs:?}");
    }

    #[test]
    fn unit_magnitude_and_gain() {
        let c = Cordic::new(30).unwrap();
        assert!((c.gain() - 1.646760258121066).abs() < 1e-12);
        assert_eq!(c.iterations(), 30);
        for k in 0..2000u32 {
            let (s, co) = c.sin_cos_fixed(k.wrapping_mul(2_147_483_659));
            let mag = ((s as f64).powi(2) + (co as f64).powi(2)).sqrt() * LSB;
            assert!((mag - 1.0).abs() < 1e-8, "angle index {k}: {mag}");
        }
    }

    #[test]
    fn float_wrappers_and_angle_conversion() {
        for k in -300..=300 {
            let a = k as f64 * 0.0271;
            let (s, c) = cordic_sin_cos(a);
            assert!((s - a.sin()).abs() < 1e-8 && (c - a.cos()).abs() < 1e-8, "angle {a}");
        }
        let (s, _) = cordic_sin_cos(1000.0 * TAU + 0.3);
        assert!((s - 0.3f64.sin()).abs() < 1e-8);
        assert_eq!(angle_from_turns(0.25), 1 << 30);
        assert_eq!(angle_from_turns(-0.25), 3 << 30);
        assert_eq!(angle_from_turns(1.0), 0);
        assert_eq!(angle_from_radians(std::f64::consts::PI), 1 << 31);
        assert!((angle_to_radians(1 << 30) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn invalid_iteration_counts() {
        assert!(Cordic::new(0).is_err());
        assert!(Cordic::new(31).is_err());
        assert!(Cordic::new(1).is_ok());
        assert!(Cordic::new(30).is_ok());
    }
}
