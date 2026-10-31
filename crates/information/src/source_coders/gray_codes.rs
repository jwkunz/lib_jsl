//! Gray codes (reflected binary codes), in which consecutive values differ in exactly one bit.

/// Returns the reflected binary Gray code of `n`.
pub fn gray_code(n: u64) -> u64 {
    n ^ (n >> 1)
}

/// Inverts [`gray_code`]: returns the `n` whose Gray code is `g`.
pub fn gray_decode(g: u64) -> u64 {
    let mut n = g;
    let mut shift = 1;
    while shift < 64 {
        n ^= n >> shift;
        shift <<= 1;
    }
    n
}

/// Returns the full `bits`-bit Gray code sequence of length `2^bits` (`bits <= 32`).
///
/// # Panics
/// Panics if `bits > 32`.
pub fn gray_code_sequence(bits: u32) -> Vec<u64> {
    assert!(bits <= 32, "gray_code_sequence supports at most 32 bits");
    (0..1u64 << bits).map(gray_code).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_values() {
        let seq: Vec<u64> = (0..8).map(gray_code).collect();
        assert_eq!(seq, vec![0, 1, 3, 2, 6, 7, 5, 4]);
    }

    #[test]
    fn round_trip() {
        for n in (0..5000u64).chain([u64::MAX, u64::MAX / 3, 1 << 63]) {
            assert_eq!(gray_decode(gray_code(n)), n);
        }
    }

    #[test]
    fn adjacent_codes_differ_by_one_bit() {
        let seq = gray_code_sequence(10);
        assert_eq!(seq.len(), 1024);
        for w in seq.windows(2) {
            assert_eq!((w[0] ^ w[1]).count_ones(), 1);
        }
        // The sequence is cyclic and a permutation.
        assert_eq!((seq[0] ^ seq[1023]).count_ones(), 1);
        let mut sorted = seq.clone();
        sorted.sort();
        assert!(sorted.iter().enumerate().all(|(i, &v)| v == i as u64));
    }
}
