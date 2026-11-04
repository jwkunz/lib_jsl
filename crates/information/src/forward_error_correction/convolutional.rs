//! Convolutional codes with Viterbi decoding.
//!
//! A rate-`1/n` code is described by its constraint length `K` and `n` generator polynomials. The
//! generator for output `j` is a `K`-bit mask whose most significant bit multiplies the newest
//! input bit, so the classic `K = 3` code with generators `(7, 5)` octal is `[0b111, 0b101]`.
//! Bits are `u8` values 0 or 1; coded bits for each input bit are emitted in generator order.
//!
//! See the book entry [Convolutional codes and the Viterbi algorithm](https://jwkunz.github.io/lib_jsl/book/coding/convolutional.html).

use lib_jsl_core::ErrorsJSL;

/// A rate-`1/n` feed-forward convolutional code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvolutionalCode {
    constraint_length: u32,
    generators: Vec<u32>,
}

impl ConvolutionalCode {
    /// Creates a code with constraint length `2 <= K <= 16` and at least one generator, each
    /// generator being a nonzero `K`-bit mask.
    pub fn new(constraint_length: u32, generators: &[u32]) -> Result<Self, ErrorsJSL> {
        if !(2..=16).contains(&constraint_length) {
            return Err(ErrorsJSL::InvalidInputRange("Constraint length must be between 2 and 16."));
        }
        if generators.is_empty() || generators.iter().any(|&g| g == 0 || g >> constraint_length != 0) {
            return Err(ErrorsJSL::InvalidInputRange("Generators must be nonzero and fit in the constraint length."));
        }
        Ok(ConvolutionalCode { constraint_length, generators: generators.to_vec() })
    }

    /// The NASA / CCSDS standard `K = 7`, rate-1/2 code with generators `(171, 133)` octal.
    pub fn nasa_k7() -> Self {
        ConvolutionalCode { constraint_length: 7, generators: vec![0o171, 0o133] }
    }

    /// Number of coded bits per input bit.
    pub fn outputs(&self) -> usize {
        self.generators.len()
    }

    /// The number of tail (flush) bits appended by terminated encoding, `K - 1`.
    pub fn tail_len(&self) -> usize {
        self.constraint_length as usize - 1
    }

    fn outputs_for(&self, reg: u32) -> impl Iterator<Item = u8> + '_ {
        self.generators.iter().map(move |&g| ((reg & g).count_ones() & 1) as u8)
    }

    /// Encodes `bits`, appending `K - 1` zero tail bits when `terminate` is true so that the
    /// encoder returns to the all-zero state. Output length is `n * (len + tail)`.
    pub fn encode(&self, bits: &[u8], terminate: bool) -> Vec<u8> {
        let k = self.constraint_length;
        let tail = if terminate { self.tail_len() } else { 0 };
        let mut out = Vec::with_capacity(self.outputs() * (bits.len() + tail));
        let mut state = 0u32;
        for &b in bits.iter().chain(std::iter::repeat_n(&0u8, tail)) {
            let reg = ((b as u32 & 1) << (k - 1)) | state;
            out.extend(self.outputs_for(reg));
            state = reg >> 1;
        }
        out
    }

    /// Maximum-likelihood (Viterbi) decoding of soft values. `llr[i]` is the log-likelihood ratio
    /// `ln P(bit = 0) / P(bit = 1)` of coded bit `i` (positive favours 0). Returns the information
    /// bits (without tail bits when `terminated`).
    pub fn decode_soft(&self, llr: &[f64], terminated: bool) -> Result<Vec<u8>, ErrorsJSL> {
        let n = self.outputs();
        if !llr.len().is_multiple_of(n) {
            return Err(ErrorsJSL::IncompatibleArraySizes((llr.len(), n)));
        }
        let steps = llr.len() / n;
        let tail = if terminated { self.tail_len() } else { 0 };
        if steps < tail {
            return Err(ErrorsJSL::InvalidInputRange("Input is shorter than the termination tail."));
        }
        let k = self.constraint_length;
        let num_states = 1usize << (k - 1);
        let mut metric = vec![f64::INFINITY; num_states];
        metric[0] = 0.0;
        // survivor[t][state] = (previous state, input bit)
        let mut survivors: Vec<Vec<(u32, u8)>> = Vec::with_capacity(steps);
        for t in 0..steps {
            let obs = &llr[t * n..(t + 1) * n];
            let mut next = vec![f64::INFINITY; num_states];
            let mut surv = vec![(0u32, 0u8); num_states];
            // ANCHOR: viterbi
            for (state, &m) in metric.iter().enumerate() {
                if !m.is_finite() {
                    continue;
                }
                for input in 0..2u32 {
                    let reg = (input << (k - 1)) | state as u32;
                    // Branch cost: sum of llr over coded bits equal to 1.
                    let cost: f64 = self.outputs_for(reg).zip(obs).map(|(bit, &l)| if bit == 1 { l } else { 0.0 }).sum();
                    let ns = (reg >> 1) as usize;
                    if m + cost < next[ns] {
                        next[ns] = m + cost;
                        surv[ns] = (state as u32, input as u8);
                    }
                }
            }
            // ANCHOR_END: viterbi
            metric = next;
            survivors.push(surv);
        }
        let mut state = if terminated {
            0usize
        } else {
            metric.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
        };
        if !metric[state].is_finite() {
            return Err(ErrorsJSL::RuntimeError("No valid Viterbi path found."));
        }
        let mut bits = vec![0u8; steps];
        for t in (0..steps).rev() {
            let (prev, input) = survivors[t][state];
            bits[t] = input;
            state = prev as usize;
        }
        bits.truncate(steps - tail);
        Ok(bits)
    }

    /// Hard-decision Viterbi decoding of coded bits (each 0 or 1).
    pub fn decode_hard(&self, coded: &[u8], terminated: bool) -> Result<Vec<u8>, ErrorsJSL> {
        let llr: Vec<f64> = coded.iter().map(|&b| if b & 1 == 0 { 1.0 } else { -1.0 }).collect();
        self.decode_soft(&llr, terminated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn textbook_encoding() {
        let code = ConvolutionalCode::new(3, &[0b111, 0b101]).unwrap();
        let out = code.encode(&[1, 0, 1, 1], true);
        assert_eq!(out, vec![1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn noiseless_round_trip() {
        let mut seed = 7;
        for code in [ConvolutionalCode::new(3, &[7, 5]).unwrap(), ConvolutionalCode::nasa_k7()] {
            let bits: Vec<u8> = (0..200).map(|_| (lcg(&mut seed) & 1) as u8).collect();
            let coded = code.encode(&bits, true);
            assert_eq!(coded.len(), 2 * (200 + code.tail_len()));
            assert_eq!(code.decode_hard(&coded, true).unwrap(), bits);
            let open = code.encode(&bits, false);
            assert_eq!(code.decode_hard(&open, false).unwrap(), bits);
        }
    }

    #[test]
    fn corrects_scattered_errors() {
        // Free distance of the K=7 code is 10: any 4 well-separated errors are corrected.
        let code = ConvolutionalCode::nasa_k7();
        let mut seed = 99;
        let bits: Vec<u8> = (0..300).map(|_| (lcg(&mut seed) & 1) as u8).collect();
        let mut coded = code.encode(&bits, true);
        for pos in [10, 120, 300, 480] {
            coded[pos] ^= 1;
        }
        assert_eq!(code.decode_hard(&coded, true).unwrap(), bits);
    }

    #[test]
    fn soft_decision_beats_unreliable_flips() {
        // Flip many bits but with tiny confidence; soft decoding should still recover the data.
        let code = ConvolutionalCode::new(3, &[7, 5]).unwrap();
        let bits = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0];
        let coded = code.encode(&bits, true);
        let mut llr: Vec<f64> = coded.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
        for i in [2usize, 3, 8, 9, 14] {
            llr[i] = -llr[i].signum() * 0.1;
        }
        assert_eq!(code.decode_soft(&llr, true).unwrap(), bits);
    }

    #[test]
    fn invalid_inputs() {
        assert!(ConvolutionalCode::new(1, &[1]).is_err());
        assert!(ConvolutionalCode::new(3, &[0b1000]).is_err());
        assert!(ConvolutionalCode::new(3, &[]).is_err());
        let code = ConvolutionalCode::new(3, &[7, 5]).unwrap();
        assert!(code.decode_hard(&[0, 1, 1], true).is_err());
        assert!(code.decode_hard(&[0, 1], true).is_err());
    }
}
