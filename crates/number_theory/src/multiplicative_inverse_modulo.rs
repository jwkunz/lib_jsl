use crate::greatest_common_divisor::extended_gcd;
use lib_jsl_core::ErrorsJSL;

/// Computes the multiplicative inverse of `a` modulo `m` using the
/// Extended Euclidean Algorithm.
///
/// # Mathematical Background
///
/// The multiplicative inverse of an integer `a` modulo `m` is an integer `a⁻¹`
/// such that:
///
/// ```text
/// (a * a⁻¹) ≡ 1 (mod m)
/// ```
///
/// Such an inverse exists if and only if `a` and `m` are coprime, i.e.,
///
/// ```text
/// gcd(a, m) = 1
/// ```
///
/// The Extended Euclidean Algorithm finds integers `x` and `y` satisfying:
///
/// ```text
/// a·x + m·y = gcd(a, m)
/// ```
///
/// When `gcd(a, m) = 1`, the coefficient `x` is the modular inverse of `a`
/// modulo `m`.
///
/// # Parameters
///
/// * `a` – The integer whose modular inverse is to be computed.
/// * `m` – The modulus. Must be nonzero.
///
/// # Returns
///
/// * `Ok(inv)` – The modular inverse of `a` modulo `m`, normalized to the
///   range `[0, |m| - 1]`.
/// * `Err(ErrorsJSL)` – If `m` is zero or the modular inverse does not exist.
///
/// # Errors
///
/// Returns an error if `gcd(a, m) ≠ 1`, meaning the inverse does not exist.
///
///
/// # Complexity
///
/// Runs in **O(log m)** time and **O(1)** space.
pub fn multiplicative_inverse_modulo(a: i128, m: i128) -> Result<i128, ErrorsJSL> {
    if m == 0 {
        return Err(ErrorsJSL::InvalidInputRange("m must be non-zero"));
    }

    let (g, x, _) = extended_gcd(a, m);

    if g != 1 {
        return Err(ErrorsJSL::RuntimeError("modular inverse does not exist (inputs are not coprime)"));
    }

    // Normalize the result into the range [0, |m| - 1]
    Ok(x.rem_euclid(m.abs()))
}

#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_multiplicative_inverse_modulo(){
        let a = 12;
        let m = 25;
        let b = multiplicative_inverse_modulo(a,m).unwrap();
        assert_eq!((a*b)%m,1)
    }

    #[test]
    fn negative_moduli_give_inverses_in_range(){
        let inv = multiplicative_inverse_modulo(3, -7).unwrap();
        assert!((0..7).contains(&inv) && (3 * inv).rem_euclid(7) == 1);
        assert!(multiplicative_inverse_modulo(4, 0).is_err() && multiplicative_inverse_modulo(4, 6).is_err());
    }
}
