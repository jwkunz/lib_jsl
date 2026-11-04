//! Deterministic primality testing for 64-bit integers.

/// `a · b mod m` without overflow.
pub fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// `base^exponent mod m` by binary exponentiation (`m > 0`).
pub fn pow_mod(mut base: u64, mut exponent: u64, m: u64) -> u64 {
    let mut result = 1 % m;
    base %= m;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, m);
        }
        base = mul_mod(base, base, m);
        exponent >>= 1;
    }
    result
}

/// Whether `n` passes the Miller-Rabin strong probable-prime test to base `a`.
pub fn is_strong_probable_prime(n: u64, a: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let a = a % n;
    if a == 0 {
        return true;
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    let mut x = pow_mod(a, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    for _ in 1..s {
        x = mul_mod(x, x, n);
        if x == n - 1 {
            return true;
        }
    }
    false
}

/// Deterministic primality for every `u64`: Miller-Rabin with the first twelve prime bases,
/// whose smallest common strong pseudoprime is 318 665 857 834 031 151 167 461 ≈ 3.2 · 10²³
/// (Sorenson and Webster, 2015), far above `u64::MAX` ≈ 1.8 · 10¹⁹, after trial division by
/// small primes.
pub fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for &p in &BASES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    if n < 41 * 41 {
        return true;
    }
    BASES.iter().all(|&a| is_strong_probable_prime(n, a))
}

/// The smallest prime `≥ n` (`None` past the largest 64-bit prime).
pub fn next_prime(n: u64) -> Option<u64> {
    (n.max(2)..=u64::MAX).find(|&k| is_prime(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prime_sieves::sieve_of_eratosthenes;

    #[test]
    fn agrees_with_the_sieve_and_rejects_pseudoprimes() {
        let primes: std::collections::HashSet<u64> = sieve_of_eratosthenes(100_000).into_iter().collect();
        assert!((0..100_000).all(|n| is_prime(n) == primes.contains(&n)));
        // Carmichael numbers and strong pseudoprimes to several small bases.
        // 3825123056546413051 is a strong pseudoprime to every prime base up to 23.
        for n in [561u64, 1105, 1729, 2047, 3_215_031_751, 3_825_123_056_546_413_051] {
            assert!(!is_prime(n), "{n}");
        }
        assert!([2, 3, 5, 7, 11, 13, 17, 19, 23].iter().all(|&a| is_strong_probable_prime(3_825_123_056_546_413_051, a)));
        assert!(is_strong_probable_prime(2047, 2) && !is_prime(2047)); // 2047 = 23 · 89 fools base 2
        assert!(is_prime(18_446_744_073_709_551_557) && !is_prime(u64::MAX));
        assert!(is_prime(1_000_000_000_039) && !is_prime(1_000_000_000_037));
        assert_eq!(next_prime(1_000_000_000_000), Some(1_000_000_000_039));
        assert_eq!(next_prime(u64::MAX - 1), None);
        assert_eq!(pow_mod(3, 200, 1_000_000_007), pow_mod(9, 100, 1_000_000_007));
        assert_eq!(pow_mod(2, 1_000_000_006, 1_000_000_007), 1); // Fermat
    }
}
