//! Prime sieves.

/// All primes `≤ n` by the sieve of Eratosthenes over odd numbers only (`n/16` bytes of
/// work space as a bit set), in `O(n log log n)` time.
pub fn sieve_of_eratosthenes(n: u64) -> Vec<u64> {
    if n < 2 {
        return Vec::new();
    }
    // Bit i represents the odd number 2i + 1.
    let count = n.div_ceil(2) as usize;
    let mut composite = vec![0u64; count.div_ceil(64)];
    let mut i = 1usize;
    while (2 * i + 1) * (2 * i + 1) <= n as usize {
        if composite[i / 64] >> (i % 64) & 1 == 0 {
            let p = 2 * i + 1;
            let mut j = (p * p) / 2;
            while j < count {
                composite[j / 64] |= 1 << (j % 64);
                j += p;
            }
        }
        i += 1;
    }
    let mut primes = vec![2];
    primes.extend((1..count).filter(|&i| composite[i / 64] >> (i % 64) & 1 == 0).map(|i| (2 * i + 1) as u64));
    primes
}

/// All primes in `[low, high]` by a segmented sieve: base primes up to `√high` cross off a
/// window of the requested range only, so the memory is `O(√high + (high − low))`. The base
/// sieve dominates the cost for very large `high` (it reaches 2³² near `u64::MAX`).
pub fn segmented_sieve(low: u64, high: u64) -> Vec<u64> {
    if high < 2 || low > high {
        return Vec::new();
    }
    let low = low.max(2);
    let root = crate::integer_square_root::integer_square_root_newton(high as u128) as u64;
    let base = sieve_of_eratosthenes(root);
    let width = (high - low + 1) as usize;
    let mut composite = vec![false; width];
    for &p in &base {
        // First multiple of p in the window, never p itself.
        let start = (p * p).max(low.div_ceil(p) * p);
        let mut m = start;
        while m <= high {
            composite[(m - low) as usize] = true;
            match m.checked_add(p) {
                Some(next) => m = next,
                None => break,
            }
        }
    }
    (0..width).filter(|&i| !composite[i]).map(|i| low + i as u64).collect()
}

/// The prime-counting function `π(n)`.
pub fn prime_count(n: u64) -> usize {
    sieve_of_eratosthenes(n).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(n: u64) -> bool {
        n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| !n.is_multiple_of(d))
    }

    #[test]
    fn sieve_matches_trial_division_and_known_counts() {
        let primes = sieve_of_eratosthenes(10_000);
        assert_eq!(primes, (0..=10_000).filter(|&n| trial(n)).collect::<Vec<_>>());
        assert_eq!(prime_count(1_000_000), 78_498);
        assert_eq!(prime_count(10_000_000), 664_579);
        assert!(sieve_of_eratosthenes(1).is_empty() && sieve_of_eratosthenes(2) == vec![2] && sieve_of_eratosthenes(9) == vec![2, 3, 5, 7]);
    }

    #[test]
    fn segmented_sieve_matches_and_reaches_large_ranges() {
        assert_eq!(segmented_sieve(0, 100), sieve_of_eratosthenes(100));
        let window = segmented_sieve(999_000, 1_000_100);
        assert_eq!(window, sieve_of_eratosthenes(1_000_100).into_iter().filter(|&p| p >= 999_000).collect::<Vec<_>>());
        // The primes just above 10¹² (known: 10¹² + 39 and 10¹² + 61).
        let near = segmented_sieve(1_000_000_000_000, 1_000_000_000_100);
        assert_eq!(&near[..2], &[1_000_000_000_039, 1_000_000_000_061]);
        assert!(segmented_sieve(20, 10).is_empty() && segmented_sieve(0, 1).is_empty());
        // Near 10¹⁵ the base primes only reach 3.2 · 10⁷; the first prime is 10¹⁵ + 37.
        let high = segmented_sieve(1_000_000_000_000_000, 1_000_000_000_001_000);
        assert_eq!(high[0], 1_000_000_000_000_037);
    }
}
