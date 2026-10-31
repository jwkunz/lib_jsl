//! Polar codes with successive-cancellation (SC) decoding.
//!
//! A polar code of length `N = 2^m` and dimension `k` transmits `k` information bits on the most
//! reliable of the `N` synthesized bit channels and freezes the rest to zero. Encoding is
//! `x = u F^(⊗m)` with kernel `F = [[1, 0], [1, 1]]` (natural order, no bit reversal). Channel
//! reliabilities are ranked with the Bhattacharyya recursion for a binary erasure channel of a
//! chosen design erasure probability.
//!
//! Bits are `u8` values 0 or 1. Soft inputs are log-likelihood ratios `ln P(0)/P(1)`.

use lib_jsl_core::ErrorsJSL;

/// A polar code.
#[derive(Clone, Debug, PartialEq)]
pub struct PolarCode {
    log2_n: u32,
    frozen: Vec<bool>,
    info_positions: Vec<usize>,
}

impl PolarCode {
    /// Constructs a code of length `2^log2_n` (`1 <= log2_n <= 20`) carrying `k` information bits,
    /// designed for a binary erasure channel with the given erasure probability in `(0, 1)`.
    pub fn new(log2_n: u32, k: usize, design_erasure_probability: f64) -> Result<Self, ErrorsJSL> {
        if !(1..=20).contains(&log2_n) {
            return Err(ErrorsJSL::InvalidInputRange("Polar code length exponent must be between 1 and 20."));
        }
        let n = 1usize << log2_n;
        if k == 0 || k > n {
            return Err(ErrorsJSL::InvalidInputRange("Polar code dimension must satisfy 0 < k <= N."));
        }
        if !(design_erasure_probability > 0.0 && design_erasure_probability < 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("Design erasure probability must lie in (0, 1)."));
        }
        // Bhattacharyya parameters of the synthesized channels (smaller = more reliable).
        let mut z = vec![design_erasure_probability; 1];
        for _ in 0..log2_n {
            let half = z.len();
            let mut next = vec![0.0; 2 * half];
            for j in 0..half {
                let t = z[j];
                next[j] = 2.0 * t - t * t;
                next[j + half] = t * t;
            }
            z = next;
        }
        let mut order: Vec<usize> = (0..n).collect();
        // Most reliable first; ties broken toward higher index (the later, better channels).
        order.sort_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap().then(b.cmp(&a)));
        let mut info_positions: Vec<usize> = order[..k].to_vec();
        info_positions.sort_unstable();
        let mut frozen = vec![true; n];
        for &p in &info_positions {
            frozen[p] = false;
        }
        Ok(PolarCode { log2_n, frozen, info_positions })
    }

    /// Code length `N`.
    pub fn n(&self) -> usize {
        1 << self.log2_n
    }

    /// Number of information bits.
    pub fn k(&self) -> usize {
        self.info_positions.len()
    }

    /// Indices (in increasing order) of the information bit channels.
    pub fn info_positions(&self) -> &[usize] {
        &self.info_positions
    }

    /// Applies the polar transform `x = u F^(⊗m)` in place.
    fn transform(bits: &mut [u8]) {
        let n = bits.len();
        let mut len = 1;
        while len < n {
            for start in (0..n).step_by(2 * len) {
                for i in start..start + len {
                    bits[i] ^= bits[i + len];
                }
            }
            len *= 2;
        }
    }

    /// Encodes `k` information bits into an `N`-bit codeword.
    pub fn encode(&self, info: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        if info.len() != self.k() {
            return Err(ErrorsJSL::IncompatibleArraySizes((info.len(), self.k())));
        }
        let mut u = vec![0u8; self.n()];
        for (&p, &b) in self.info_positions.iter().zip(info) {
            u[p] = b & 1;
        }
        Self::transform(&mut u);
        Ok(u)
    }

    /// Successive-cancellation decoding (min-sum) of `N` channel LLRs, returning the `k`
    /// information bits.
    pub fn decode(&self, llr: &[f64]) -> Result<Vec<u8>, ErrorsJSL> {
        if llr.len() != self.n() {
            return Err(ErrorsJSL::IncompatibleArraySizes((llr.len(), self.n())));
        }
        let mut u = vec![0u8; self.n()];
        let mut pos = 0;
        self.sc(llr, &mut pos, &mut u);
        Ok(self.info_positions.iter().map(|&p| u[p]).collect())
    }

    /// Hard-decision convenience wrapper around [`decode`](Self::decode).
    pub fn decode_hard(&self, received: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        let llr: Vec<f64> = received.iter().map(|&b| if b & 1 == 0 { 1.0 } else { -1.0 }).collect();
        self.decode(&llr)
    }

    /// Recursive SC decoder. Decodes the next `llr.len()` input bits into `u[*pos..]` and returns
    /// the re-encoded partial sums for this sub-block.
    fn sc(&self, llr: &[f64], pos: &mut usize, u: &mut [u8]) -> Vec<u8> {
        let n = llr.len();
        if n == 1 {
            let bit = if self.frozen[*pos] { 0 } else if llr[0] < 0.0 { 1 } else { 0 };
            u[*pos] = bit;
            *pos += 1;
            return vec![bit];
        }
        let half = n / 2;
        let f: Vec<f64> = (0..half)
            .map(|i| {
                let (a, b) = (llr[i], llr[i + half]);
                a.signum() * b.signum() * a.abs().min(b.abs())
            })
            .collect();
        let left = self.sc(&f, pos, u);
        let g: Vec<f64> = (0..half)
            .map(|i| llr[i + half] + if left[i] == 0 { llr[i] } else { -llr[i] })
            .collect();
        let right = self.sc(&g, pos, u);
        let mut out = Vec::with_capacity(n);
        out.extend(left.iter().zip(&right).map(|(l, r)| l ^ r));
        out.extend(right);
        out
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
    fn classic_n8_k4_information_set() {
        let c = PolarCode::new(3, 4, 0.5).unwrap();
        assert_eq!(c.info_positions(), &[3, 5, 6, 7]);
        assert_eq!((c.n(), c.k()), (8, 4));
    }

    #[test]
    fn encoder_is_the_kernel_power() {
        // N = 2: x = [u0 ^ u1, u1]
        let c = PolarCode::new(1, 2, 0.5).unwrap();
        assert_eq!(c.encode(&[1, 0]).unwrap(), vec![1, 0]);
        assert_eq!(c.encode(&[0, 1]).unwrap(), vec![1, 1]);
        assert_eq!(c.encode(&[1, 1]).unwrap(), vec![0, 1]);
    }

    #[test]
    fn noiseless_round_trip() {
        let mut seed = 3;
        for (m, k) in [(3, 4), (5, 16), (8, 128), (10, 512)] {
            let c = PolarCode::new(m, k, 0.4).unwrap();
            let info: Vec<u8> = (0..k).map(|_| (lcg(&mut seed) & 1) as u8).collect();
            let cw = c.encode(&info).unwrap();
            assert_eq!(c.decode_hard(&cw).unwrap(), info, "N=2^{m}");
        }
    }

    #[test]
    fn corrects_channel_errors() {
        let c = PolarCode::new(8, 100, 0.4).unwrap();
        let mut seed = 21;
        let info: Vec<u8> = (0..100).map(|_| (lcg(&mut seed) & 1) as u8).collect();
        let mut cw = c.encode(&info).unwrap();
        for pos in [5, 77, 140, 200] {
            cw[pos] ^= 1;
        }
        assert_eq!(c.decode_hard(&cw).unwrap(), info);
    }

    #[test]
    fn soft_decoding_and_validation() {
        let c = PolarCode::new(6, 24, 0.4).unwrap();
        let mut seed = 5;
        let info: Vec<u8> = (0..24).map(|_| (lcg(&mut seed) & 1) as u8).collect();
        let cw = c.encode(&info).unwrap();
        let mut llr: Vec<f64> = cw.iter().map(|&b| if b == 0 { 3.0 } else { -3.0 }).collect();
        llr[10] = -llr[10].signum() * 0.2; // an unreliable wrong value
        assert_eq!(c.decode(&llr).unwrap(), info);
        assert!(c.encode(&[0; 3]).is_err());
        assert!(c.decode(&[0.0; 3]).is_err());
        assert!(PolarCode::new(0, 1, 0.5).is_err());
        assert!(PolarCode::new(3, 9, 0.5).is_err());
        assert!(PolarCode::new(3, 4, 1.5).is_err());
    }
}
