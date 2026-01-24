use crate::prelude::ErrorsJSL;

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
/// * Intermediate results are reduced modulo `modulus` to prevent overflow.
/// * The base is normalized modulo `modulus` before computation.
/// * Although `i128` provides a wide range, extremely large values may still
///   overflow if the modulus itself is near `i128::MAX`.

pub fn base_exponent_in_modulus(base: i128, exponent: i128, modulus: i128) -> Result<i128,ErrorsJSL> {
    if ! modulus > 0{
        return Err(ErrorsJSL::InvalidInputRange("modulus must be greater than zero"))
    }
    if ! exponent >= 0{
        return Err(ErrorsJSL::InvalidInputRange("exponent must nonnegative"))
    }

    // Normalize base into the range [0, modulus - 1]
    let mut base = ((base % modulus) + modulus) % modulus;
    let mut exp = exponent;
    let mut result: i128 = 1 % modulus;

    while exp > 0 {
        // If the current exponent bit is set, multiply result by base
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }

        // Square the base and reduce modulo
        base = (base * base) % modulus;

        // Shift exponent right by one bit
        exp >>= 1;
    }

    Ok(result)
}


#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_base_exponent_in_modulus(){
        let result = base_exponent_in_modulus(2, 10, 1_000).unwrap();
        assert_eq!(result, 24); // 2^10 = 1024, 1024 % 1000 = 24
        let result = base_exponent_in_modulus(-2, 3, 5).unwrap();
        assert_eq!(result, 2); // (-2)^3 = -8, -8 mod 5 = 2
    }
}
