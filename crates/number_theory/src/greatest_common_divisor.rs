/// Computes the greatest common divisor (GCD) of two integers using the
/// classical Euclidean algorithm.
///
/// # Overview
/// The greatest common divisor of two integers is the largest positive integer
/// that divides both numbers without leaving a remainder. This implementation
/// uses the Euclidean algorithm, based on
///
/// ```text
/// gcd(a, b) = gcd(b, a mod b)
/// ```
///
/// repeated until the second operand becomes zero; the first operand is then
/// the GCD up to sign, and its absolute value is returned.
///
/// # Properties
/// - **Time Complexity:** O(log(min(|a|, |b|)))
/// - **Space Complexity:** O(1)
///
/// # Edge Cases
/// - The result is never negative: `gcd(4, -6) = 2`.
/// - If one of the inputs is zero, the function returns the absolute value of the other.
/// - If both inputs are zero, the function returns zero (a common convention).
/// - `i128::MIN` with zero (or with itself) has no representable GCD and overflows.
pub fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.abs()
}

/// Computes the greatest common divisor (GCD) of two integers and finds
/// integer coefficients `x` and `y` such that:
///
/// ```text
/// a·x + b·y = gcd(a, b)
/// ```
///
/// # Overview
/// This function implements the **Extended Euclidean Algorithm**, which extends
/// the classical Euclidean algorithm by additionally computing the Bézout
/// coefficients, as used for modular inverses and the Chinese remainder theorem.
/// The algorithm keeps the invariant `a·s + b·t = r` for each remainder `r`
/// while running the Euclidean recurrence `gcd(a, b) = gcd(b, a mod b)`.
///
/// # Properties
/// - **Time Complexity:** O(log(min(|a|, |b|)))
/// - **Space Complexity:** O(1)
///
/// # Edge Cases
/// - The returned `g` is never negative (the coefficients' signs are adjusted to match).
/// - If `b == 0`, the function returns `(|a|, ±1, 0)`.
/// - If both inputs are zero, the function returns `(0, 0, 0)`.
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
    let mut old_r: i128 = a;
    let mut r: i128 = b;

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

    // At this point old_r is the GCD up to sign, and old_s and old_t are the Bézout
    // coefficients; make the GCD non-negative.
    if old_r < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn the_gcd_is_never_negative() {
        assert_eq!(gcd(4, -6), 2);
        assert_eq!(gcd(-4, -6), 2);
        assert_eq!(gcd(-5, 0), 5);
        for (a, b) in [(4i128, -6i128), (-4, 6), (-4, -6), (-35, 0), (0, -9), (240, -46)] {
            let (g, x, y) = extended_gcd(a, b);
            assert!(g >= 0 && a * x + b * y == g && g == gcd(a, b), "({a}, {b}) -> ({g}, {x}, {y})");
        }
    }

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
