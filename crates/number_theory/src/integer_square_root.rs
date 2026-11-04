/// Computes the integer square root of a `u128` using binary search (efficient, but slower)
///
/// The integer square root of `n` is defined as the largest integer `x`
/// such that:
///
/// ```text
/// x * x <= n
/// ```
///
/// This function:
/// - Works for the entire `u128` domain
/// - Does not use floating-point arithmetic
/// - Avoids overflow by using division-based comparisons
/// - Runs in `O(log n)` time using binary search
pub fn integer_square_root_binary_search(n: u128) -> u128 {
    // Handle trivial cases explicitly.
    if n < 2 {
        return n;
    }

    // Search space:
    // sqrt(n) is always <= n / 2 for n >= 2,
    // but we can tighten this slightly by using n.
    let mut low: u128 = 1;
    let mut high: u128 = n;
    let mut result: u128 = 0;

    while low <= high {
        let mid = low + (high - low) / 2;

        // Compare mid * mid <= n without overflow.
        if mid <= n / mid {
            result = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    result
}

/// Computes the integer square root of a `u128` using the
/// Newton–Raphson method (faster)
///
/// The integer square root of `n` is defined as the largest integer `x`
/// such that:
///
/// ```text
/// x * x <= n
/// ```
///
/// This function:
/// - Uses the Newton–Raphson iteration:
///   x_{k+1} = (x_k + n / x_k) / 2
/// - Starts above the root, where the iterates decrease monotonically to
///   ⌊√n⌋, and stops at the first iterate that does not decrease (starting
///   below the root, the iteration can alternate between ⌊√n⌋ and ⌊√n⌋ + 1)
/// - Avoids overflow by computing `n / x_k`
/// - Converges very quickly in `O(log log n)` iterations
pub fn integer_square_root_newton(n: u128) -> u128 {
    if n < 2 {
        return n;
    }

    // 2^⌈bits/2⌉ is at least √n.
    let bits = 128 - n.leading_zeros();
    let mut x = 1u128 << bits.div_ceil(2);

    loop {
        let next = (x + n / x) / 2;
        if next >= x {
            return x;
        }
        x = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;



    #[test]
    fn newton_terminates_and_agrees_with_binary_search() {
        // n = k² − 1 used to make the iteration alternate between k − 1 and k forever.
        for n in 0..20_000u128 {
            assert_eq!(integer_square_root_newton(n), integer_square_root_binary_search(n), "n = {n}");
        }
        for k in [3u128, 1 << 20, 1 << 40, (1 << 63) + 12345, u64::MAX as u128] {
            for n in [k * k - 1, k * k, k * k + 1] {
                let r = integer_square_root_newton(n);
                assert!(r * r <= n && (r + 1).checked_mul(r + 1).is_none_or(|v| v > n), "n = {n}");
            }
        }
        assert_eq!(integer_square_root_newton(u128::MAX), u64::MAX as u128);
    }

    #[test]
    fn test_integer_square_root_binary_search() {
        for i in 0u128..1000000 {
            let sq = i * i;
            assert_eq!(integer_square_root_binary_search(sq), i); // Slow
        }
    }


    #[test]
    fn test_integer_square_root_newton() {
        for i in 0u128..1000000 {
            let sq = i * i;
            assert_eq!(integer_square_root_newton(sq), i); // Fast
        }
    }

}
