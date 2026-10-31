//! Machine epsilon and unit-in-the-last-place helpers.

/// Machine epsilon of `f64`: the gap between 1.0 and the next representable number (2⁻⁵²).
pub fn machine_epsilon() -> f64 {
    f64::EPSILON
}

/// Machine epsilon of `f32` (2⁻²³).
pub fn machine_epsilon_f32() -> f32 {
    f32::EPSILON
}

/// Determines the `f64` machine epsilon empirically by halving until `1 + e/2 == 1`.
///
/// Useful to verify that the platform's floating-point behaviour matches the constant (it does not
/// on, e.g., x87 extended-precision code paths).
pub fn measure_machine_epsilon() -> f64 {
    let mut e = 1.0f64;
    // black_box prevents the optimiser folding the loop condition to a constant comparison.
    while std::hint::black_box(1.0 + e / 2.0) > 1.0 {
        e /= 2.0;
    }
    e
}

/// Spacing between `x` and the next larger-magnitude `f64` (the unit in the last place).
/// `NaN` for non-finite input.
pub fn ulp(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let x = x.abs();
    if x == f64::MAX {
        return x - f64::from_bits(x.to_bits() - 1);
    }
    f64::from_bits(x.to_bits() + 1) - x
}

/// Number of representable `f64` values between `a` and `b` (0 if equal; `None` if either is NaN).
pub fn ulps_between(a: f64, b: f64) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    // Map the bit patterns onto a monotonic integer line.
    let key = |v: f64| {
        let bits = v.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    Some(key(a).abs_diff(key(b)))
}

/// Whether `a` and `b` differ by at most `max_ulps` representable values.
pub fn approx_eq_ulps(a: f64, b: f64, max_ulps: u64) -> bool {
    ulps_between(a, b).is_some_and(|d| d <= max_ulps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_values_agree() {
        assert_eq!(machine_epsilon(), 2.0f64.powi(-52));
        assert_eq!(machine_epsilon_f32(), 2.0f32.powi(-23));
        assert_eq!(measure_machine_epsilon(), machine_epsilon());
    }

    #[test]
    fn ulp_properties() {
        assert_eq!(ulp(1.0), f64::EPSILON);
        assert_eq!(ulp(-1.0), f64::EPSILON);
        assert_eq!(ulp(2.0), 2.0 * f64::EPSILON);
        assert_eq!(ulp(0.0), f64::from_bits(1));
        assert!(ulp(f64::NAN).is_nan() && ulp(f64::INFINITY).is_nan());
        assert!(ulp(f64::MAX) > 0.0);
    }

    #[test]
    fn ulp_distance() {
        assert_eq!(ulps_between(1.0, 1.0), Some(0));
        assert_eq!(ulps_between(1.0, 1.0 + f64::EPSILON), Some(1));
        assert_eq!(ulps_between(-0.0, 0.0), Some(0));
        assert_eq!(ulps_between(-f64::from_bits(1), f64::from_bits(1)), Some(2));
        assert_eq!(ulps_between(f64::NAN, 1.0), None);
        assert!(approx_eq_ulps(0.1 + 0.2, 0.3, 1));
        assert!(!approx_eq_ulps(1.0, 1.001, 10));
    }
}
