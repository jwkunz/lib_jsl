use crate::{number_theory::greatest_common_divisor::extended_gcd, prelude::ErrorsJSL};

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
/// # Assumptions
///
/// This function assumes the existence of an `extended_gcd` function with
/// the following signature:
///
///
/// fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64);
///
///
/// which returns `(g, x, y)` such that:
///
/// ```text
/// a·x + b·y = g = gcd(a, b)
/// ```
///
/// # Parameters
///
/// * `a` – The integer whose modular inverse is to be computed.
/// * `m` – The modulus. Must be nonzero.
///
/// # Returns
///
/// * `Ok(inv)` – The modular inverse of `a` modulo `m`, normalized to the
///   range `[0, m - 1]`.
/// * `Err(&'static str)` – If the modular inverse does not exist.
///
/// # Errors
///
/// Returns an error if `gcd(a, m) ≠ 1`, meaning the inverse does not exist.
///
///
/// # Complexity
///
/// Runs in **O(log m)** time and **O(1)** space (excluding recursion inside
/// `extended_gcd`).
pub fn multiplicative_inverse_modulo(a: i128, m: i128) -> Result<i128, ErrorsJSL> {
    if m == 0 {
        return Err(ErrorsJSL::InvalidInputRange("m must be non-zero"));
    }

    let (g, x, _) = extended_gcd(a, m);

    if g != 1 {
        return Err(ErrorsJSL::RuntimeError("modular inverse does not exist (inputs are not coprime)"));
    }

    // Normalize the result into the range [0, m - 1]
    Ok((x % m + m) % m)
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
}