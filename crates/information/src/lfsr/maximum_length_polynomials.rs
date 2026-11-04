//! Maximum length (primitive) polynomials over GF(2).
//!
//! Polynomials are bit masks: bit `k` is the coefficient of `x^k`, so `x^3 + x + 1` is `0b1011`.
//! An LFSR with a primitive characteristic polynomial of degree `n` produces a maximum-length
//! sequence of period `2^n - 1`.
//!
//! See the book entry [LFSRs and Gold codes](https://jwkunz.github.io/lib_jsl/book/information/lfsr.html#testing-primitivity).

use std::sync::OnceLock;

/// Largest degree supported by [`is_primitive_polynomial`] (limited by trial-division factoring of
/// `2^n - 1`).
pub const MAX_CHECKED_DEGREE: u32 = 40;

/// Largest degree in the table returned by [`maximum_length_polynomials`].
pub const MAX_TABLE_DEGREE: u32 = 32;

fn degree(p: u64) -> u32 {
    63 - p.leading_zeros()
}

/// Carry-less multiplication of `a` and `b` modulo `p` (both reduced, degree of `p` at most 63).
fn mul_mod(mut a: u64, mut b: u64, p: u64) -> u64 {
    let n = degree(p);
    let top = 1u64 << n;
    let mut r = 0u64;
    while b != 0 {
        if b & 1 == 1 {
            r ^= a;
        }
        b >>= 1;
        a <<= 1;
        if a & top != 0 {
            a ^= p;
        }
    }
    r
}

fn pow_mod(base: u64, mut e: u64, p: u64) -> u64 {
    let mut result = 1u64;
    let mut b = base;
    while e > 0 {
        if e & 1 == 1 {
            result = mul_mod(result, b, p);
        }
        b = mul_mod(b, b, p);
        e >>= 1;
    }
    result
}

fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            out.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Tests whether `poly` (degree between 2 and [`MAX_CHECKED_DEGREE`], with nonzero constant term)
/// is primitive over GF(2), i.e. the multiplicative order of `x` modulo `poly` is `2^n - 1`.
pub fn is_primitive_polynomial(poly: u64) -> bool {
    if poly < 4 {
        return false;
    }
    let n = degree(poly);
    if n > MAX_CHECKED_DEGREE || poly & 1 == 0 {
        return false;
    }
    // ANCHOR: primitive
    let order = (1u64 << n) - 1;
    // x^order must be 1 and x^(order/q) != 1 for every prime q dividing order.
    if pow_mod(2, order, poly) != 1 {
        return false;
    }
    prime_factors(order).into_iter().all(|q| pow_mod(2, order / q, poly) != 1)
    // ANCHOR_END: primitive
}

/// Returns a minimal-weight primitive polynomial of the given degree (`2..=32`): a trinomial where
/// one exists, otherwise a pentanomial or heptanomial, taking the numerically smallest.
pub fn maximum_length_polynomial(degree: u32) -> Option<u64> {
    if !(2..=MAX_TABLE_DEGREE).contains(&degree) {
        return None;
    }
    maximum_length_polynomials().get(degree as usize - 2).copied()
}

/// Calls `f` with every mask built from `count` distinct bits chosen among bits `1..n`, in
/// increasing numeric order of the resulting masks; stops early when `f` returns `true`.
fn combos(n: u32, count: u32, f: &mut dyn FnMut(u64) -> bool) -> bool {
    fn rec(start: u32, n: u32, left: u32, acc: u64, f: &mut dyn FnMut(u64) -> bool) -> bool {
        if left == 0 {
            return f(acc);
        }
        for b in start..n {
            if n - b < left {
                break;
            }
            if rec(b + 1, n, left - 1, acc | (1 << b), f) {
                return true;
            }
        }
        false
    }
    rec(1, n, count, 0, f)
}

fn minimal_weight_primitive(n: u32) -> u64 {
    let base = (1u64 << n) | 1;
    for mid_terms in [1, 3, 5, 7, 9] {
        let mut best: Option<u64> = None;
        combos(n, mid_terms, &mut |m| {
            let p = base | m;
            if is_primitive_polynomial(p) {
                best = Some(best.map_or(p, |b| b.min(p)));
            }
            false
        });
        if let Some(p) = best {
            return p;
        }
    }
    unreachable!("a primitive polynomial of weight <= 11 exists for every degree up to 32")
}

/// Returns one primitive polynomial for each degree `2..=32`; element `i` has degree `i + 2`.
/// Each has minimal weight (number of nonzero terms).
pub fn maximum_length_polynomials() -> Vec<u64> {
    static TABLE: OnceLock<Vec<u64>> = OnceLock::new();
    TABLE
        .get_or_init(|| (2..=MAX_TABLE_DEGREE).map(minimal_weight_primitive).collect())
        .clone()
}

/// Enumerates every primitive polynomial of the given degree (`2..=24`).
pub fn all_maximum_length_polynomials(degree: u32) -> Vec<u64> {
    if !(2..=24).contains(&degree) {
        return Vec::new();
    }
    let base = (1u64 << degree) | 1;
    (0..(1u64 << (degree - 1)))
        .map(|k| base | (k << 1))
        .filter(|&p| is_primitive_polynomial(p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfsr::Lfsr;

    #[test]
    fn known_small_polynomials() {
        assert_eq!(maximum_length_polynomial(2), Some(0b111));
        assert_eq!(maximum_length_polynomial(3), Some(0b1011));
        assert_eq!(maximum_length_polynomial(4), Some(0b10011));
        assert_eq!(maximum_length_polynomial(5), Some(0b100101));
        assert!(is_primitive_polynomial(0b1011) && is_primitive_polynomial(0b1101));
        assert!(!is_primitive_polynomial(0b10101));
        assert!(!is_primitive_polynomial(0b11111)); // x^4+x^3+x^2+x+1 is irreducible, order 5
        assert_eq!(maximum_length_polynomial(1), None);
    }

    #[test]
    fn counts_match_euler_phi() {
        // Number of primitive polynomials of degree n is phi(2^n - 1) / n.
        assert_eq!(all_maximum_length_polynomials(2).len(), 1);
        assert_eq!(all_maximum_length_polynomials(3).len(), 2);
        assert_eq!(all_maximum_length_polynomials(4).len(), 2);
        assert_eq!(all_maximum_length_polynomials(5).len(), 6);
        assert_eq!(all_maximum_length_polynomials(6).len(), 6);
        assert_eq!(all_maximum_length_polynomials(8).len(), 16);
        assert_eq!(all_maximum_length_polynomials(10).len(), 60);
    }

    #[test]
    fn table_polynomials_generate_maximal_sequences() {
        let table = maximum_length_polynomials();
        assert_eq!(table.len(), 31);
        for (i, &p) in table.iter().enumerate() {
            let n = i as u32 + 2;
            assert_eq!(degree(p), n);
            if n <= 12 {
                assert_eq!(Lfsr::from_polynomial(p, 1).period(1 << 13), Some((1 << n) - 1), "degree {n}");
            }
        }
        assert!(table.iter().skip(2).all(|&p| p.count_ones() <= 5));
    }
}
