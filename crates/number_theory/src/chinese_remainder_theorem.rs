//! The Chinese remainder theorem.
//!
//! See the book entry [Divisibility and modular arithmetic](https://jwkunz.github.io/lib_jsl/book/number-theory/modular-arithmetic.html).

use crate::greatest_common_divisor::extended_gcd;
use lib_jsl_core::ErrorsJSL;

/// Solves the simultaneous congruences `x ≡ r_i (mod m_i)` and returns `(x, M)` with
/// `0 ≤ x < M`, where `M = lcm(m_i)`, so that the solution set is `x + kM`. The moduli need not
/// be pairwise coprime (Garner-style pairwise merging with `gcd`); returns an error when the
/// congruences are inconsistent or `M` overflows `i128`.
pub fn chinese_remainder(residues: &[i128], moduli: &[i128]) -> Result<(i128, i128), ErrorsJSL> {
    if residues.len() != moduli.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((residues.len(), moduli.len())));
    }
    if moduli.iter().any(|&m| m <= 0) {
        return Err(ErrorsJSL::InvalidInputRange("Moduli must be positive."));
    }
    let (mut x, mut m) = (0i128, 1i128);
    // ANCHOR: crt_merge
    for (&r, &mi) in residues.iter().zip(moduli) {
        let r = r.rem_euclid(mi);
        // Merge x (mod m) with r (mod mi): x + m t ≡ r (mod mi), i.e. m t ≡ r − x (mod mi).
        let (g, p, _) = extended_gcd(m, mi);
        let difference = r - x;
        if difference % g != 0 {
            return Err(ErrorsJSL::InvalidInputRange("The congruences are inconsistent."));
        }
        let step = mi / g;
        // t = (difference / g) · p (mod mi / g), with p the inverse of m/g modulo mi/g.
        let t = ((difference / g) % step * (p % step)).rem_euclid(step);
        let lcm = m.checked_mul(step).ok_or(ErrorsJSL::InvalidInputRange("The combined modulus overflows."))?;
        x = (x + m.checked_mul(t).ok_or(ErrorsJSL::InvalidInputRange("The combined modulus overflows."))?).rem_euclid(lcm);
        m = lcm;
    }
    // ANCHOR_END: crt_merge
    Ok((x, m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_coprime_and_non_coprime_systems() {
        // Sunzi's problem: x ≡ 2 (3), 3 (5), 2 (7) → 23 mod 105.
        assert_eq!(chinese_remainder(&[2, 3, 2], &[3, 5, 7]).unwrap(), (23, 105));
        // Non-coprime but consistent: x ≡ 3 (4), 5 (6) → 11 mod 12.
        assert_eq!(chinese_remainder(&[3, 5], &[4, 6]).unwrap(), (11, 12));
        // Negative residues are reduced first.
        assert_eq!(chinese_remainder(&[-1, -1], &[5, 7]).unwrap(), (34, 35));
        // Large moduli near 2⁶².
        let (m1, m2) = (4_611_686_018_427_387_847i128, 4_611_686_018_427_387_817i128);
        let (x, m) = chinese_remainder(&[123_456_789, 987_654_321], &[m1, m2]).unwrap();
        assert_eq!((x % m1, x % m2, m), (123_456_789, 987_654_321, m1 * m2));
        assert!(chinese_remainder(&[1, 2], &[4, 6]).is_err()); // inconsistent parity
        assert!(chinese_remainder(&[1], &[0]).is_err() && chinese_remainder(&[1], &[2, 3]).is_err());
        assert_eq!(chinese_remainder(&[], &[]).unwrap(), (0, 1));
    }

    #[test]
    fn brute_force_agreement() {
        for a in 0..8 {
            for b in 0..12 {
                let brute = (0..24).find(|x| x % 8 == a && x % 12 == b);
                match chinese_remainder(&[a, b], &[8, 12]) {
                    Ok((x, m)) => assert_eq!((Some(x), m), (brute, 24)),
                    Err(_) => assert!(brute.is_none()),
                }
            }
        }
    }
}
