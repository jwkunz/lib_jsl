//! Factorization of 64-bit integers.

use crate::primality::{is_prime, mul_mod};

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A non-trivial factor of the odd composite `n` by Pollard's rho with Brent's cycle detection
/// and batched gcds, iterating `x ← x² + c` for successive `c`.
pub fn pollard_rho(n: u64) -> u64 {
    if n.is_multiple_of(2) {
        return 2;
    }
    for c in 1..n {
        let f = |x: u64| (mul_mod(x, x, n) + c) % n;
        let (mut y, mut r, mut q) = (2u64, 1u64, 1u64);
        let (mut x, mut ys, mut g) = (0, 0, 1);
        const BATCH: u64 = 128;
        while g == 1 {
            x = y;
            for _ in 0..r {
                y = f(y);
            }
            let mut k = 0;
            while k < r && g == 1 {
                ys = y;
                for _ in 0..BATCH.min(r - k) {
                    y = f(y);
                    q = mul_mod(q, x.abs_diff(y), n);
                }
                g = gcd(q, n);
                k += BATCH;
            }
            r *= 2;
        }
        if g == n {
            // The batch overshot: step back one at a time.
            loop {
                ys = f(ys);
                g = gcd(x.abs_diff(ys), n);
                if g > 1 {
                    break;
                }
            }
        }
        if g != n {
            return g;
        }
    }
    n
}

/// The prime factorization of `n ≥ 1` as `(prime, exponent)` pairs in increasing order (empty
/// for 1). Small factors are removed by trial division, the remainder is split recursively
/// with Pollard's rho and certified by deterministic Miller-Rabin. Any `u64` factors in
/// milliseconds.
pub fn factorize(n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    let mut n = n;
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
        while n.is_multiple_of(p) {
            factors.push(p);
            n /= p;
        }
    }
    let mut stack = vec![n];
    while let Some(m) = stack.pop() {
        if m == 1 {
            continue;
        }
        if is_prime(m) {
            factors.push(m);
            continue;
        }
        let d = pollard_rho(m);
        stack.push(d);
        stack.push(m / d);
    }
    factors.sort_unstable();
    let mut out: Vec<(u64, u32)> = Vec::new();
    for p in factors {
        match out.last_mut() {
            Some((q, e)) if *q == p => *e += 1,
            _ => out.push((p, 1)),
        }
    }
    out
}

/// Euler's totient `φ(n)`, from the factorization.
pub fn euler_totient(n: u64) -> u64 {
    factorize(n).iter().fold(n, |acc, &(p, _)| acc / p * (p - 1))
}

/// All positive divisors of `n ≥ 1`, sorted.
pub fn divisors(n: u64) -> Vec<u64> {
    let mut out = vec![1u64];
    for (p, e) in factorize(n) {
        let current = out.clone();
        let mut power = 1;
        for _ in 0..e {
            power *= p;
            out.extend(current.iter().map(|d| d * power));
        }
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn product(f: &[(u64, u32)]) -> u128 {
        f.iter().map(|&(p, e)| (p as u128).pow(e)).product()
    }

    #[test]
    fn factors_small_and_hard_64_bit_numbers() {
        for n in 1..5000u64 {
            let f = factorize(n);
            assert_eq!(product(&f), n as u128);
            assert!(f.iter().all(|&(p, _)| is_prime(p)));
        }
        assert_eq!(factorize(600_851_475_143), vec![(71, 1), (839, 1), (1471, 1), (6857, 1)]);
        // A semiprime of two 32-bit primes, and a large prime power.
        assert_eq!(factorize(4_294_967_291 * 4_294_967_279), vec![(4_294_967_279, 1), (4_294_967_291, 1)]);
        assert_eq!(factorize(3u64.pow(40)), vec![(3, 40)]);
        assert_eq!(factorize(18_446_744_073_709_551_557), vec![(18_446_744_073_709_551_557, 1)]);
        assert_eq!(factorize(u64::MAX), vec![(3, 1), (5, 1), (17, 1), (257, 1), (641, 1), (65_537, 1), (6_700_417, 1)]);
        assert!(factorize(1).is_empty());
    }

    #[test]
    fn totient_and_divisors() {
        assert_eq!(euler_totient(36), 12);
        assert_eq!(euler_totient(97), 96);
        assert_eq!(euler_totient(1), 1);
        assert_eq!(divisors(28), vec![1, 2, 4, 7, 14, 28]);
        assert_eq!(divisors(1), vec![1]);
        assert_eq!(divisors(720_720).len(), 240);
    }
}
