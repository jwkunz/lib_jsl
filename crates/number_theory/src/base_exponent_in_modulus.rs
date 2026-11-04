use lib_jsl_core::ErrorsJSL;

/// Computes modular exponentiation using exponentiation by squaring.
///
/// This function calculates:
///
/// ```text
/// (base ^ exponent) mod modulus
/// ```
///
/// using 128-bit signed integers (`i128`). It is efficient for large exponents,
/// running in `O(log exponent)` time.
///
/// # Parameters
///
/// * `base` - The base value.
/// * `exponent` - The exponent. Must be non-negative.
/// * `modulus` - The modulus. Must be greater than zero.
///
/// # Returns
///
/// The result of `(base ^ exponent) % modulus`, normalized to the range
/// `[0, modulus - 1]`.
///
/// # Errors
///
/// This function will error if:
///
/// * `modulus <= 0`
/// * `exponent < 0`
///
/// # Notes
///
/// * The base is normalized modulo `modulus` before computation.
/// * Products are formed in 128-bit unsigned arithmetic when they fit and by
///   double-and-add otherwise, so every positive `i128` modulus is handled
///   without overflow.
pub fn base_exponent_in_modulus(base: i128, exponent: i128, modulus: i128) -> Result<i128, ErrorsJSL> {
    if modulus <= 0 {
        return Err(ErrorsJSL::InvalidInputRange("modulus must be greater than zero"));
    }
    if exponent < 0 {
        return Err(ErrorsJSL::InvalidInputRange("exponent must be nonnegative"));
    }

    // Work with residues in [0, modulus), as u128.
    let m = modulus as u128;
    let mut base = base.rem_euclid(modulus) as u128;
    let mut exp = exponent;
    let mut result: u128 = 1 % m;

    while exp > 0 {
        // If the current exponent bit is set, multiply result by base
        if exp & 1 == 1 {
            result = mul_mod_u128(result, base, m);
        }

        // Square the base and reduce modulo
        base = mul_mod_u128(base, base, m);

        // Shift exponent right by one bit
        exp >>= 1;
    }

    Ok(result as i128)
}

/// `a · b mod m` for `a, b < m < 2^127`, without overflow.
fn mul_mod_u128(a: u128, b: u128, m: u128) -> u128 {
    if let Some(product) = a.checked_mul(b) {
        return product % m;
    }
    // Double-and-add: every intermediate is below 2m < 2^128.
    let (mut result, mut a, mut b) = (0u128, a, b);
    while b > 0 {
        if b & 1 == 1 {
            result = (result + a) % m;
        }
        a = (a + a) % m;
        b >>= 1;
    }
    result
}

#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn invalid_arguments_are_errors_and_large_moduli_do_not_overflow() {
        // A zero modulus used to reach `% 0` and panic.
        assert!(base_exponent_in_modulus(3, 4, 0).is_err());
        assert!(base_exponent_in_modulus(3, 4, -7).is_err());
        assert!(base_exponent_in_modulus(3, -1, 7).is_err());
        assert_eq!(base_exponent_in_modulus(-2, 3, 7).unwrap(), 6); // (−8) mod 7
        // Fermat's little theorem for the Mersenne prime 2^89 − 1 and for 2^127 − 1, whose
        // squares overflow 128 bits.
        for p in [(1i128 << 89) - 1, i128::MAX] {
            assert_eq!(base_exponent_in_modulus(3, p - 1, p).unwrap(), 1);
            assert_eq!(base_exponent_in_modulus(12345, p, p).unwrap(), 12345);
        }
    }

    #[test]
    fn test_base_exponent_in_modulus(){
        let result = base_exponent_in_modulus(2, 10, 1_000).unwrap();
        assert_eq!(result, 24); // 2^10 = 1024, 1024 % 1000 = 24
        let result = base_exponent_in_modulus(-2, 3, 5).unwrap();
        assert_eq!(result, 2); // (-2)^3 = -8, -8 mod 5 = 2
    }
}
