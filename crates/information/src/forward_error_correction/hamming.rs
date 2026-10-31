//! Hamming codes, with an optional extended (SECDED) parity bit.
//!
//! A Hamming code with `r` parity bits has length `n = 2^r - 1` and dimension `k = n - r`, and
//! corrects any single bit error. Bits are represented as `u8` values 0 or 1. Codeword position
//! `p` (counting from 1) is a parity bit when `p` is a power of two; the syndrome of a received
//! word is the XOR of the positions of its set bits and equals the position of a single error.
//! The extended code appends one overall parity bit, giving single-error correction and
//! double-error detection.

use lib_jsl_core::ErrorsJSL;

/// What the decoder concluded about a received word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HammingStatus {
    /// The word was a valid codeword.
    NoError,
    /// A single bit error was corrected at this zero-based codeword index.
    Corrected(usize),
    /// An even number of errors (two) was detected and could not be corrected (extended code only).
    DoubleErrorDetected,
}

/// The result of decoding a Hamming codeword.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HammingDecoded {
    /// The decoded data bits (best effort if a double error was detected).
    pub data: Vec<u8>,
    /// What happened during decoding.
    pub status: HammingStatus,
}

/// A (possibly extended) Hamming code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HammingCode {
    r: u32,
    extended: bool,
}

impl HammingCode {
    /// A Hamming code with `r` parity bits (`2 <= r <= 20`), e.g. `r = 3` is the (7,4) code.
    pub fn new(r: u32) -> Result<Self, ErrorsJSL> {
        if !(2..=20).contains(&r) {
            return Err(ErrorsJSL::InvalidInputRange("Hamming parity bit count must be between 2 and 20."));
        }
        Ok(HammingCode { r, extended: false })
    }

    /// The extended (SECDED) code with `r` Hamming parity bits plus one overall parity bit,
    /// e.g. `r = 3` gives the (8,4) code.
    pub fn new_extended(r: u32) -> Result<Self, ErrorsJSL> {
        let mut c = Self::new(r)?;
        c.extended = true;
        Ok(c)
    }

    fn base_len(&self) -> usize {
        (1usize << self.r) - 1
    }

    /// Codeword length (including the overall parity bit for the extended code).
    pub fn n(&self) -> usize {
        self.base_len() + self.extended as usize
    }

    /// Number of data bits.
    pub fn k(&self) -> usize {
        self.base_len() - self.r as usize
    }

    /// Encodes `k` data bits (each 0 or 1) into an `n`-bit codeword.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        if data.len() != self.k() {
            return Err(ErrorsJSL::IncompatibleArraySizes((data.len(), self.k())));
        }
        let nb = self.base_len();
        let mut cw = vec![0u8; nb];
        let mut it = data.iter();
        let mut syndrome = 0usize;
        for pos in 1..=nb {
            if !pos.is_power_of_two() {
                let bit = *it.next().unwrap() & 1;
                cw[pos - 1] = bit;
                if bit == 1 {
                    syndrome ^= pos;
                }
            }
        }
        for i in 0..self.r {
            if syndrome >> i & 1 == 1 {
                cw[(1usize << i) - 1] = 1;
            }
        }
        if self.extended {
            let parity = cw.iter().fold(0u8, |a, &b| a ^ b);
            cw.push(parity);
        }
        Ok(cw)
    }

    /// Decodes an `n`-bit received word.
    pub fn decode(&self, received: &[u8]) -> Result<HammingDecoded, ErrorsJSL> {
        if received.len() != self.n() {
            return Err(ErrorsJSL::IncompatibleArraySizes((received.len(), self.n())));
        }
        let nb = self.base_len();
        let mut word: Vec<u8> = received.iter().map(|&b| b & 1).collect();
        let syndrome = (1..=nb).filter(|&p| word[p - 1] == 1).fold(0usize, |a, p| a ^ p);
        let overall_odd = self.extended && word.iter().fold(0u8, |a, &b| a ^ b) == 1;
        let status = if syndrome == 0 && !overall_odd {
            HammingStatus::NoError
        } else if self.extended && syndrome != 0 && !overall_odd {
            HammingStatus::DoubleErrorDetected
        } else if syndrome == 0 {
            // Only the overall parity bit is wrong.
            word[nb] ^= 1;
            HammingStatus::Corrected(nb)
        } else {
            word[syndrome - 1] ^= 1;
            HammingStatus::Corrected(syndrome - 1)
        };
        let data = (1..=nb).filter(|p| !p.is_power_of_two()).map(|p| word[p - 1]).collect();
        Ok(HammingDecoded { data, status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: usize, n: usize) -> Vec<u8> {
        (0..n).map(|i| (value >> i & 1) as u8).collect()
    }

    #[test]
    fn parameters() {
        let c = HammingCode::new(3).unwrap();
        assert_eq!((c.n(), c.k()), (7, 4));
        let e = HammingCode::new_extended(3).unwrap();
        assert_eq!((e.n(), e.k()), (8, 4));
        assert_eq!(HammingCode::new(4).unwrap().n(), 15);
        assert!(HammingCode::new(1).is_err());
    }

    #[test]
    fn known_codeword() {
        // Data 1,0,1,1 goes to positions 3,5,6,7; the syndrome 3^6^7 = 2 sets only parity p2,
        // giving positions 1..7 = 0 1 1 0 0 1 1.
        let c = HammingCode::new(3).unwrap();
        assert_eq!(c.encode(&[1, 0, 1, 1]).unwrap(), vec![0, 1, 1, 0, 0, 1, 1]);
    }

    #[test]
    fn corrects_every_single_error() {
        for r in [3u32, 4, 5] {
            let c = HammingCode::new(r).unwrap();
            for value in [0usize, 1, 0b1011, 0xA5A5 & ((1 << c.k()) - 1), (1 << c.k()) - 1] {
                let data = bits(value, c.k());
                let cw = c.encode(&data).unwrap();
                assert_eq!(c.decode(&cw).unwrap().status, HammingStatus::NoError);
                for i in 0..c.n() {
                    let mut bad = cw.clone();
                    bad[i] ^= 1;
                    let d = c.decode(&bad).unwrap();
                    assert_eq!(d.data, data, "r={r} bit {i}");
                    assert_eq!(d.status, HammingStatus::Corrected(i));
                }
            }
        }
    }

    #[test]
    fn minimum_distance_is_three() {
        let c = HammingCode::new(3).unwrap();
        let words: Vec<Vec<u8>> = (0..16).map(|v| c.encode(&bits(v, 4)).unwrap()).collect();
        let mut dmin = usize::MAX;
        for i in 0..16 {
            for j in (i + 1)..16 {
                dmin = dmin.min(words[i].iter().zip(&words[j]).filter(|(a, b)| a != b).count());
            }
        }
        assert_eq!(dmin, 3);
    }

    #[test]
    fn extended_code_detects_double_errors() {
        let c = HammingCode::new_extended(4).unwrap();
        let data = bits(0x5A5, c.k());
        let cw = c.encode(&data).unwrap();
        for i in 0..c.n() {
            let mut one = cw.clone();
            one[i] ^= 1;
            let d = c.decode(&one).unwrap();
            assert_eq!(d.data, data);
            assert_eq!(d.status, HammingStatus::Corrected(i));
            for j in (i + 1)..c.n() {
                let mut two = cw.clone();
                two[i] ^= 1;
                two[j] ^= 1;
                assert_eq!(c.decode(&two).unwrap().status, HammingStatus::DoubleErrorDetected, "{i},{j}");
            }
        }
    }

    #[test]
    fn length_mismatch() {
        let c = HammingCode::new(3).unwrap();
        assert!(c.encode(&[1, 0]).is_err());
        assert!(c.decode(&[0; 8]).is_err());
    }
}
