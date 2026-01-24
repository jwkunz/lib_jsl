/// Computes the greatest common divisor (GCD) of two unsigned 64-bit integers
/// using the classical Euclidean algorithm.
///
/// # Overview
/// The greatest common divisor of two integers is the largest positive integer
/// that divides both numbers without leaving a remainder. This implementation
/// uses the Euclidean algorithm, which is based on the mathematical principle
/// that:
///
/// ```text
/// gcd(a, b) = gcd(b, a mod b)
/// ```
///
/// This process is repeated until the second operand becomes zero. At that
/// point, the first operand contains the GCD.
///
/// # Algorithm
/// 1. Accept two `u64` values, `a` and `b`.
/// 2. While `b` is non-zero:
///    - Compute the remainder of `a` divided by `b`.
///    - Assign `b` to `a`.
///    - Assign the remainder to `b`.
/// 3. When `b` reaches zero, return `a`.
///
/// # Properties
/// - **Time Complexity:** O(log(min(a, b)))
/// - **Space Complexity:** O(1)
/// - **Deterministic:** Always produces the same result for the same inputs
/// - **Safe:** Uses only safe Rust constructs and does not allocate memory
///
/// # Edge Cases
/// - If one of the inputs is zero, the function returns the other input.
/// - If both inputs are zero, the function returns zero. While mathematically
///   the GCD of (0, 0) is undefined, returning zero is a common and practical
///   convention in software systems.
///
/// # Parameters
/// - `a`: The first unsigned integer.
/// - `b`: The second unsigned integer.
///
/// # Returns
/// The greatest common divisor of `a` and `b` .
pub fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Computes the greatest common divisor (GCD) of two unsigned 64-bit integers
/// and finds integer coefficients `x` and `y` such that:
///
/// ```text
/// a·x + b·y = gcd(a, b)
/// ```
///
/// # Overview
/// This function implements the **Extended Euclidean Algorithm**, which extends
/// the classical Euclidean algorithm by additionally computing the Bézout
/// coefficients. These coefficients are frequently used in number theory,
/// cryptography, and modular arithmetic (e.g., computing modular inverses).
///
/// While the inputs `a` and `b` are `u64`, the coefficients `x` and `y` may be
/// negative. Therefore, they are returned as signed 128-bit integers (`i128`)
/// to ensure sufficient range and avoid overflow during intermediate
/// calculations.
///
/// # Algorithm
/// The extended Euclidean algorithm is based on the recursive relationship:
///
/// ```text
/// gcd(a, b) = gcd(b, a mod b)
/// ```
///
/// Along with the invariant:
///
/// ```text
/// a·x + b·y = gcd(a, b)
/// ```
///
/// The algorithm proceeds iteratively, updating the coefficients until the
/// remainder becomes zero.
///
/// # Properties
/// - **Time Complexity:** O(log(min(a, b)))
/// - **Space Complexity:** O(1)
/// - **Deterministic:** Produces consistent results for identical inputs
/// - **Safe:** Uses only safe Rust constructs and does not allocate memory
///
/// # Edge Cases
/// - If `b == 0`, the function returns `(a, 1, 0)`, since `a·1 + 0·0 = a`.
/// - If `a == 0`, the function returns `(b, 0, 1)`, since `0·0 + b·1 = b`.
/// - If both inputs are zero, the function returns `(0, 0, 0)`. Although
///   mathematically undefined, this is a pragmatic convention.
///
/// # Parameters
/// - `a`: The first unsigned 64-bit integer.
/// - `b`: The second unsigned 64-bit integer.
///
/// # Returns
/// A tuple `(g, x, y)` where:
/// - `g` is the greatest common divisor of `a` and `b`
/// - `x` and `y` are integers satisfying `a·x + b·y = g`
pub fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    // Handle the degenerate case explicitly
    if a == 0 && b == 0 {
        return (0, 0, 0);
    }

    // Convert to signed integers for coefficient arithmetic
    let mut old_r: i128 = a as i128;
    let mut r: i128 = b as i128;

    let mut old_s: i128 = 1;
    let mut s: i128 = 0;

    let mut old_t: i128 = 0;
    let mut t: i128 = 1;

    while r != 0 {
        let quotient = old_r / r;

        let temp_r = old_r - quotient * r;
        old_r = r;
        r = temp_r;

        let temp_s = old_s - quotient * s;
        old_s = s;
        s = temp_s;

        let temp_t = old_t - quotient * t;
        old_t = t;
        t = temp_t;
    }

    // At this point:
    // old_r is the GCD
    // old_s and old_t are the Bézout coefficients
    (old_r as i128, old_s, old_t)
}

#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_gcd(){
        let a = 12;
        let b = 21;
        let g = gcd(a, b);
        assert_eq!(g,3);
        let (g,x,y) = extended_gcd(a,b);
        assert_eq!(g,3);
        assert_eq!(x,2);
        assert_eq!(y,-1);
    }
}

