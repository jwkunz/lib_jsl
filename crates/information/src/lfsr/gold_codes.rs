//! Gold codes.
//!
//! A Gold family for register length `n` consists of the two m-sequences of a preferred pair
//! together with the `2^n - 1` sequences obtained by adding the second sequence at every cyclic
//! shift to the first. Their periodic cross-correlations take only three values,
//! `-1`, `-t(n)` and `t(n) - 2`, where `t(n) = 1 + 2^((n+1)/2)` for odd `n` and
//! `1 + 2^((n+2)/2)` for `n ≡ 2 (mod 4)`.

use super::linear_feedback_shift_register::Lfsr;
use super::maximum_length_polynomials::maximum_length_polynomial;
use lib_jsl_core::ErrorsJSL;

/// The magnitude bound `t(n)` on the Gold-code correlation values, or `None` if no Gold family
/// exists for `n` (that is, when `n` is a multiple of 4).
pub fn gold_correlation_bound(n: u32) -> Option<u64> {
    match n % 4 {
        0 => None,
        2 => Some(1 + (1u64 << ((n + 2) / 2))),
        _ => Some(1 + (1u64 << n.div_ceil(2))),
    }
}

/// Generates the Gold family for register length `n` (`3..=20`, not a multiple of 4).
///
/// Returns `2^n + 1` sequences of `2^n - 1` chips each, as bits (0/1): index 0 is the first
/// m-sequence `u`, index 1 is the second m-sequence `v` (a decimation of `u`), and index `2 + τ`
/// is `u ⊕ (v cyclically shifted by τ)`. Map bits to ±1 chips with `1 - 2 * bit` for correlation.
pub fn gold_codes(n: u32) -> Result<Vec<Vec<u8>>, ErrorsJSL> {
    if !(3..=20).contains(&n) {
        return Err(ErrorsJSL::InvalidInputRange("Gold code register length must be between 3 and 20."));
    }
    if gold_correlation_bound(n).is_none() {
        return Err(ErrorsJSL::InvalidInputRange("Gold codes do not exist when the register length is a multiple of 4."));
    }
    let poly = maximum_length_polynomial(n).ok_or(ErrorsJSL::RuntimeError("No primitive polynomial available."))?;
    let period = (1usize << n) - 1;
    let u: Vec<u8> = Lfsr::from_polynomial(poly, 1).bits(period);
    // Preferred decimation q = 2^k + 1 with k = 1 (odd n) or k = 2 (n ≡ 2 mod 4).
    // ANCHOR: gold
    let q = if n % 2 == 1 { 3 } else { 5 };
    let v: Vec<u8> = (0..period).map(|t| u[(q * t) % period]).collect();
    let mut family = Vec::with_capacity(period + 2);
    family.push(u.clone());
    family.push(v.clone());
    for shift in 0..period {
        family.push((0..period).map(|t| u[t] ^ v[(t + shift) % period]).collect());
    }
    // ANCHOR_END: gold
    Ok(family)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chips(bits: &[u8]) -> Vec<i32> {
        bits.iter().map(|&b| 1 - 2 * b as i32).collect()
    }

    fn periodic_correlation(a: &[i32], b: &[i32], shift: usize) -> i32 {
        let n = a.len();
        (0..n).map(|t| a[t] * b[(t + shift) % n]).sum()
    }

    #[test]
    fn family_size_and_length() {
        let fam = gold_codes(5).unwrap();
        assert_eq!(fam.len(), 33);
        assert!(fam.iter().all(|s| s.len() == 31));
    }

    #[test]
    fn three_valued_cross_correlation() {
        for &n in &[5u32, 6] {
            let fam: Vec<Vec<i32>> = gold_codes(n).unwrap().iter().map(|s| chips(s)).collect();
            let t = gold_correlation_bound(n).unwrap() as i32;
            let allowed = [-1, -t, t - 2];
            let len = fam[0].len();
            for i in 0..fam.len() {
                for j in (i + 1)..fam.len() {
                    for shift in 0..len {
                        let c = periodic_correlation(&fam[i], &fam[j], shift);
                        assert!(allowed.contains(&c), "n={n} i={i} j={j} shift={shift} c={c}");
                    }
                }
            }
        }
    }

    #[test]
    fn m_sequences_have_two_valued_autocorrelation() {
        let fam = gold_codes(7).unwrap();
        let u = chips(&fam[0]);
        assert_eq!(periodic_correlation(&u, &u, 0), 127);
        for s in 1..127 {
            assert_eq!(periodic_correlation(&u, &u, s), -1);
        }
    }

    #[test]
    fn invalid_lengths() {
        assert!(gold_codes(4).is_err());
        assert!(gold_codes(8).is_err());
        assert!(gold_codes(2).is_err());
        assert!(gold_codes(21).is_err());
    }
}
