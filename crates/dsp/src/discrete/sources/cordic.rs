//! Fixed-point CORDIC sine/cosine, in 32-bit and 64-bit word sizes.
//!
//! The COordinate Rotation DIgital Computer computes sine and cosine with only shifts and adds: a
//! vector is rotated toward the target angle by a fixed sequence of micro-rotations of angle
//! `atan(2^-i)`, each chosen clockwise or counter-clockwise. This implementation is entirely
//! integer arithmetic, as it would be in hardware, and the single algorithm is instantiated for two
//! word sizes:
//!
//! | type | angle | results | max iterations | accuracy |
//! |------|-------|---------|----------------|----------|
//! | [`Cordic32`] (alias [`Cordic`]) | `u32` binary angle | `i32`, Q1.30 | 30 | about 1e-8 |
//! | [`Cordic64`] | `u64` binary angle | `i64`, Q1.62 | 62 | about 4e-16 at 52 iterations |
//!
//! A *binary angle* has `2^N` units per turn, so it wraps naturally and its top two bits select the
//! quadrant. Results are signed (sine and cosine go negative) with `1.0` represented as
//! `1 << FRACTIONAL_BITS`. The micro-rotation angles and the cumulative gain compensation are
//! precomputed integer tables, and the shifts use round-to-nearest to keep the bias down.
//!
//! Widening only the data would not help: the angle's resolution limits the accuracy, so both words
//! grow together. `Cordic32` is the hardware-style choice (1e-8 is about -160 dB, beyond what a
//! 24-bit converter chain can use); `Cordic64` matches the 64-bit phase of the NCO and reaches
//! double-precision accuracy, at the cost of about 1.7 times as many iterations.

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::TAU;
use std::sync::OnceLock;

/// `atan(2^-i)` as a binary angle (`2^32` per turn).
const ATAN_BAM_32: [u32; 30] = [
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


const ATAN_BAM_64: [u64; 62] = [
    0x2000000000000000, 0x12E4051D9DF30866, 0x09FB385B5EE39E8E,
    0x051111D41DDD9A1B, 0x028B0D430E589AED, 0x0145D7E159046278,
    0x00A2F61E5C28262A, 0x00517C5511D442AF, 0x0028BE5346D0C337,
    0x00145F2EBB30AB38, 0x000A2F980091BA7B, 0x000517CC14A80CB7,
    0x00028BE60CDFEC62, 0x000145F306C172F2, 0x0000A2F9836AE911,
    0x0000517CC1B6BA7C, 0x000028BE60DB85FC, 0x0000145F306DC816,
    0x00000A2F9836E4AE, 0x00000517CC1B726B, 0x0000028BE60DB938,
    0x00000145F306DC9C, 0x000000A2F9836E4E, 0x000000517CC1B727,
    0x00000028BE60DB94, 0x000000145F306DCA, 0x0000000A2F9836E5,
    0x0000000517CC1B72, 0x000000028BE60DB9, 0x0000000145F306DD,
    0x00000000A2F9836E, 0x00000000517CC1B7, 0x0000000028BE60DC,
    0x00000000145F306E, 0x000000000A2F9837, 0x000000000517CC1B,
    0x00000000028BE60E, 0x000000000145F307, 0x0000000000A2F983,
    0x0000000000517CC2, 0x000000000028BE61, 0x0000000000145F30,
    0x00000000000A2F98, 0x00000000000517CC, 0x0000000000028BE6,
    0x00000000000145F3, 0x000000000000A2FA, 0x000000000000517D,
    0x00000000000028BE, 0x000000000000145F, 0x0000000000000A30,
    0x0000000000000518, 0x000000000000028C, 0x0000000000000146,
    0x00000000000000A3, 0x0000000000000051, 0x0000000000000029,
    0x0000000000000014, 0x000000000000000A, 0x0000000000000005,
    0x0000000000000003, 0x0000000000000001,
];

const INV_GAIN_Q62: [i64; 63] = [
    4611686018427387904, 3260954456333195553, 2916686334356757942,
    2829601372552588592, 2807750841902562267, 2802282967498353433,
    2800915666627739259, 2800573820569637254, 2800488357751430639,
    2800466991965380887, 2800461650513774536, 2800460315150554575,
    2800459981309729686, 2800459897849522220, 2800459876984470276,
    2800459871768207285, 2800459870464141537, 2800459870138125100,
    2800459870056620990, 2800459870036244963, 2800459870031150956,
    2800459870029877455, 2800459870029559079, 2800459870029479485,
    2800459870029459587, 2800459870029454612, 2800459870029453369,
    2800459870029453058, 2800459870029452980, 2800459870029452960,
    2800459870029452956, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
    2800459870029452954, 2800459870029452954, 2800459870029452954,
];

/// Defines a CORDIC rotator type for one angle/word size. The algorithm is written once here.
macro_rules! define_cordic {
    (
        $(#[$meta:meta])*
        $name:ident,
        angle: $A:ty,
        word: $W:ty,
        angle_bits: $ABITS:expr,
        frac_bits: $FRAC:expr,
        max_iterations: $MAX:expr,
        default_iterations: $DEF:expr,
        atan: $ATAN:ident,
        inv_gain: $INV:ident
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            iterations: u32,
        }

        impl $name {
            /// Number of fractional bits in the signed results.
            pub const FRACTIONAL_BITS: u32 = $FRAC;
            /// The fixed-point representation of 1.0.
            pub const ONE: $W = 1 << $FRAC;
            /// The maximum number of micro-rotations.
            pub const MAX_ITERATIONS: u32 = $MAX;
            /// A good default iteration count (about the accuracy floor of the word size).
            pub const DEFAULT_ITERATIONS: u32 = $DEF;

            /// Creates a rotator with `iterations` micro-rotations (1 to [`Self::MAX_ITERATIONS`]).
            pub fn new(iterations: u32) -> Result<$name, ErrorsJSL> {
                if !(1..=$MAX).contains(&iterations) {
                    return Err(ErrorsJSL::InvalidInputRange("CORDIC iteration count is out of range."));
                }
                Ok($name { iterations })
            }

            pub fn iterations(&self) -> u32 {
                self.iterations
            }

            /// The gain `K = Π sqrt(1 + 2^-2i)` that the micro-rotations add (about 1.6468), which
            /// the rotator compensates for by starting from `1/K`.
            pub fn gain(&self) -> f64 {
                (0..self.iterations).map(|i| (1.0 + 0.25f64.powi(i as i32)).sqrt()).product()
            }

            /// Rounding arithmetic right shift.
            fn shr_round(v: $W, shift: u32) -> $W {
                if shift == 0 { v } else { (v + (1 << (shift - 1))) >> shift }
            }

            /// Returns `(sin, cos)` of a binary angle as signed fixed-point values with
            /// [`Self::FRACTIONAL_BITS`] fractional bits, using integer arithmetic only.
            pub fn sin_cos_fixed(&self, angle: $A) -> ($W, $W) {
                let inv_gain = $INV[self.iterations as usize];
                // The top two bits give the quadrant: start from the gain-compensated unit vector
                // already rotated by that many quarter turns, then rotate by the remaining angle
                // in [0, pi/2).
                let (mut x, mut y): ($W, $W) = match angle >> ($ABITS - 2) {
                    0 => (inv_gain, 0),
                    1 => (0, inv_gain),
                    2 => (-inv_gain, 0),
                    _ => (0, -inv_gain),
                };
                let mut z = (angle & (<$A>::MAX >> 2)) as $W;
                for i in 0..self.iterations {
                    let (dx, dy) = (Self::shr_round(y, i), Self::shr_round(x, i));
                    let theta = $ATAN[i as usize] as $W;
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

            /// Floating-point convenience: `(sin, cos)` of an angle in radians (any real value).
            /// The angle is first quantized to a binary angle.
            pub fn sin_cos(&self, radians: f64) -> (f64, f64) {
                let (s, c) = self.sin_cos_fixed(Self::angle_from_radians(radians));
                (s as f64 / Self::ONE as f64, c as f64 / Self::ONE as f64)
            }

            /// Converts an angle in turns to a binary angle (modulo one turn).
            pub fn angle_from_turns(turns: f64) -> $A {
                ((turns.rem_euclid(1.0) * 2f64.powi($ABITS)).round() as u128) as $A // a full turn wraps to 0
            }

            /// Converts an angle in radians to a binary angle (modulo one turn).
            pub fn angle_from_radians(radians: f64) -> $A {
                Self::angle_from_turns(radians / TAU)
            }

            /// Converts a binary angle to radians in `[0, 2*pi)`.
            pub fn angle_to_radians(angle: $A) -> f64 {
                angle as f64 / 2f64.powi($ABITS) * TAU
            }
        }
    };
}

define_cordic! {
    /// A 32-bit CORDIC rotator: `u32` binary angles (`2^32` per turn) and `i32` Q1.30 results.
    /// Accurate to about 1e-8 (a dozen least-significant bits) at 30 iterations.
    Cordic32,
    angle: u32,
    word: i32,
    angle_bits: 32,
    frac_bits: 30,
    max_iterations: 30,
    default_iterations: 30,
    atan: ATAN_BAM_32,
    inv_gain: INV_GAIN_Q30
}

define_cordic! {
    /// A 64-bit CORDIC rotator: `u64` binary angles (`2^64` per turn, the same width as the NCO
    /// phase) and `i64` Q1.62 results. Reaches double-precision accuracy (about 4e-16) at 52
    /// iterations and about 3e-18 at the maximum of 62.
    Cordic64,
    angle: u64,
    word: i64,
    angle_bits: 64,
    frac_bits: 62,
    max_iterations: 62,
    default_iterations: 52,
    atan: ATAN_BAM_64,
    inv_gain: INV_GAIN_Q62
}

/// The 32-bit rotator, kept under its original name.
pub type Cordic = Cordic32;

/// Number of fractional bits in [`Cordic32`] results.
pub const FRACTIONAL_BITS: u32 = Cordic32::FRACTIONAL_BITS;
/// The [`Cordic32`] fixed-point representation of 1.0.
pub const ONE: i32 = Cordic32::ONE;
/// The maximum number of [`Cordic32`] micro-rotations.
pub const MAX_ITERATIONS: u32 = Cordic32::MAX_ITERATIONS;

/// Converts an angle in radians to a `u32` binary angle (modulo one turn).
pub fn angle_from_radians(radians: f64) -> u32 {
    Cordic32::angle_from_radians(radians)
}

/// Converts an angle in turns to a `u32` binary angle (modulo one turn).
pub fn angle_from_turns(turns: f64) -> u32 {
    Cordic32::angle_from_turns(turns)
}

/// Converts a `u32` binary angle to radians in `[0, 2*pi)`.
pub fn angle_to_radians(angle: u32) -> f64 {
    Cordic32::angle_to_radians(angle)
}

/// Returns `(sin(angle), cos(angle))` for an angle in radians, using the 32-bit integer CORDIC with
/// [`Cordic32::DEFAULT_ITERATIONS`] (accurate to about 1e-8). Use [`Cordic64`] for full
/// double-precision accuracy.
pub fn cordic_sin_cos(angle: f64) -> (f64, f64) {
    static DEFAULT: OnceLock<Cordic32> = OnceLock::new();
    DEFAULT.get_or_init(|| Cordic32::new(Cordic32::DEFAULT_ITERATIONS).expect("valid iteration count")).sin_cos(angle)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LSB: f64 = 1.0 / ONE as f64;

    /// Worst error against libm over a dense sweep of binary angles.
    fn max_error(c: &Cordic32, step: u32) -> f64 {
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

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn tables_match_their_definitions() {
        for (i, &t) in ATAN_BAM_32.iter().enumerate() {
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

    // ------------------------------------------------------------ 64-bit

    const LSB64: f64 = 1.0 / (1u64 << 62) as f64;

    fn max_error64(c: &Cordic64, samples: u64) -> f64 {
        let step = u64::MAX / samples;
        let mut worst = 0.0f64;
        let mut a = 12_345u64;
        for _ in 0..samples {
            let (s, co) = c.sin_cos_fixed(a);
            let r = Cordic64::angle_to_radians(a);
            worst = worst.max((s as f64 * LSB64 - r.sin()).abs()).max((co as f64 * LSB64 - r.cos()).abs());
            a = a.wrapping_add(step);
        }
        worst
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn tables64_match_their_definitions_to_double_precision() {
        assert_eq!(ATAN_BAM_64[0], 1 << 61, "atan(1) is exactly an eighth of a turn");
        for (i, &t) in ATAN_BAM_64.iter().enumerate() {
            let want = (0.5f64.powi(i as i32)).atan() / TAU * 2f64.powi(64);
            assert!((t as f64 - want).abs() <= want * 4e-16 + 1.0, "atan table entry {i}");
        }
        assert_eq!(*ATAN_BAM_64.last().unwrap(), 1);
        for n in 1..=62usize {
            let gain: f64 = (0..n).map(|i| (1.0 + 0.25f64.powi(i as i32)).sqrt()).product();
            let want = 2f64.powi(62) / gain;
            assert!((INV_GAIN_Q62[n] as f64 - want).abs() <= want * 1e-15 + 1.0, "gain table entry {n}");
        }
        assert_eq!(INV_GAIN_Q62[0], 1 << 62);
    }

    #[test]
    fn double_precision_accuracy_at_52_iterations() {
        let c = Cordic64::new(Cordic64::DEFAULT_ITERATIONS).unwrap();
        assert_eq!(c.iterations(), 52);
        // The reference itself carries ~1e-15 of error: it converts the 64-bit angle to f64
        // radians. The exact-integer tests below resolve far smaller errors.
        let err = max_error64(&c, 40_000);
        assert!(err < 1.5e-15, "max error {err}");
        let full = max_error64(&Cordic64::new(62).unwrap(), 40_000);
        assert!(full < 1.5e-15, "max error {full}");
    }

    /// Exact integer checks that need no floating-point reference, so they can see errors far
    /// below double precision.
    #[test]
    fn identities_hold_beyond_double_precision() {
        let c = Cordic64::new(62).unwrap();
        let unit = 1i128 << 124;
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut worst_radial, mut worst_angle) = (0.0f64, 0.0f64);
        for _ in 0..4000 {
            let (a, b) = (next(), next());
            let (sa, ca) = c.sin_cos_fixed(a);
            let (sb, cb) = c.sin_cos_fixed(b);
            let (sab, _) = c.sin_cos_fixed(a.wrapping_add(b));
            // Radial error: |(s, c)| should be 1.
            let norm = sa as i128 * sa as i128 + ca as i128 * ca as i128;
            worst_radial = worst_radial.max((norm - unit).abs() as f64 / unit as f64 / 2.0);
            // Angle addition: sin(a+b) = sin a cos b + cos a sin b.
            let lhs = sa as i128 * cb as i128 + ca as i128 * sb as i128;
            let rhs = (sab as i128) << 62;
            worst_angle = worst_angle.max((lhs - rhs).abs() as f64 / unit as f64);
        }
        assert!(worst_radial < 2e-17, "radial error {worst_radial:e}");
        assert!(worst_angle < 5e-17, "angle-addition error {worst_angle:e}");
    }

    #[test]
    fn error_shrinks_geometrically_with_64_bit_words() {
        let errs: Vec<f64> = [8u32, 16, 24, 32, 40, 48].iter().map(|&n| max_error64(&Cordic64::new(n).unwrap(), 4001)).collect();
        for w in errs.windows(2) {
            assert!(w[1] < w[0] * 0.02, "{errs:?}");
        }
        assert!(errs[3] < 1e-9 && errs[5] < 2e-14, "{errs:?}");
    }

    #[test]
    fn wide_words_do_better_than_narrow_ones() {
        let narrow = max_error(&Cordic32::new(30).unwrap(), 1_000_003);
        let wide = max_error64(&Cordic64::new(52).unwrap(), 4001);
        assert!(wide * 1e5 < narrow, "{wide} vs {narrow}");
    }

    #[test]
    fn quadrants_symmetry_and_unit_magnitude_64() {
        let c = Cordic64::new(62).unwrap();
        let near = |got: i64, want: i64| (got - want).abs() <= 64;
        let one = Cordic64::ONE;
        let (s, co) = c.sin_cos_fixed(0);
        assert!(near(s, 0) && near(co, one));
        let (s, co) = c.sin_cos_fixed(1 << 62);
        assert!(near(s, one) && near(co, 0));
        let (s, co) = c.sin_cos_fixed(1 << 63);
        assert!(near(s, 0) && near(co, -one));
        let (s, co) = c.sin_cos_fixed(3 << 62);
        assert!(near(s, -one) && near(co, 0));
        for a in [123_456_789u64, 0x9E37_79B9_7F4A_7C15, 0xDEAD_BEEF_0000_0001, u64::MAX - 5] {
            let (s0, c0) = c.sin_cos_fixed(a);
            let (s1, c1) = c.sin_cos_fixed(a.wrapping_add(1 << 62));
            assert!((s1 - c0).abs() <= 128 && (c1 + s0).abs() <= 128, "angle {a}");
            let mag = ((s0 as f64).powi(2) + (c0 as f64).powi(2)).sqrt() * LSB64;
            assert!((mag - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn float_wrappers_and_angle_conversion_64() {
        let c = Cordic64::new(52).unwrap();
        for k in -300..=300 {
            let a = k as f64 * 0.0271;
            let (s, co) = c.sin_cos(a);
            assert!((s - a.sin()).abs() < 5e-15 && (co - a.cos()).abs() < 5e-15, "angle {a}");
        }
        assert_eq!(Cordic64::angle_from_turns(0.25), 1 << 62);
        assert_eq!(Cordic64::angle_from_turns(-0.25), 3 << 62);
        assert_eq!(Cordic64::angle_from_turns(1.0), 0);
        assert_eq!(Cordic64::angle_from_radians(std::f64::consts::PI), 1 << 63);
        assert!((Cordic64::angle_to_radians(1 << 62) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert!((c.gain() - 1.646760258121066).abs() < 1e-12);
    }

    #[test]
    fn invalid_iteration_counts_64() {
        assert!(Cordic64::new(0).is_err());
        assert!(Cordic64::new(63).is_err());
        assert!(Cordic64::new(1).is_ok() && Cordic64::new(62).is_ok());
        // The 32-bit alias is unchanged.
        let _: Cordic = Cordic::new(30).unwrap();
        assert_eq!(Cordic::MAX_ITERATIONS, 30);
    }
}
