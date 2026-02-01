use crate::random::split_mix_64::*;
/// xoshiro256++
///
/// High-quality, fast pseudorandom number generator suitable for
/// Monte Carlo simulation and numerical work.
///
/// NOT cryptographically secure.
///
/// Reference:
///   David Blackman & Sebastiano Vigna
///   https://prng.di.unimi.it/xoshiro256plusplus.c
#[derive(Clone, Debug)]
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    /// Rotate left (safe wrapper)
    #[inline(always)]
    fn rotl(x: u64, k: u32) -> u64 {
        (x << k) | (x >> (64 - k))
    }

    /// Create a new generator from a non-zero seed.
    ///
    /// If you only have a single u64 seed, use `from_seed`.
    pub fn new(seed: [u64; 4]) -> Self {
        assert!(
            seed != [0, 0, 0, 0],
            "xoshiro256++ state must not be all zero"
        );
        Self { s: seed }
    }

    /// Seed using SplitMix64 (recommended).
    pub fn from_seed(seed: u64) -> Self {
        let mut sm64 = SplitMix64::new(seed);
        Self::new([
            sm64.next_u64(),
            sm64.next_u64(),
            sm64.next_u64(),
            sm64.next_u64(),
        ])
    }

    /// Generate next random u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = Self::rotl(self.s[0].wrapping_add(self.s[3]), 23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = Self::rotl(self.s[3], 45);

        result
    }

    /// Generate a floating-point value in [0, 1).
    ///
    /// Uses the high 53 bits for correct uniformity.
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Jump ahead 2^128 steps (for parallel streams).
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [
            0x180ec6d33cfd0aba,
            0xd5a61266f0c9392c,
            0xa9582618e03fc9aa,
            0x39abdc4529b1661c,
        ];

        let mut s0 = 0;
        let mut s1 = 0;
        let mut s2 = 0;
        let mut s3 = 0;

        for &jump in &JUMP {
            for b in 0..64 {
                if (jump & (1u64 << b)) != 0 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                self.next_u64();
            }
        }

        self.s = [s0, s1, s2, s3];
    }
}

#[cfg(test)]
mod test{
    use super::*;
    #[test]
    fn test_xoshiro256plusplus(){
            let mut rng = Xoshiro256PlusPlus::from_seed(0xdead_beef_cafe_f00d);
            let mut rng2 = rng.clone();
            rng.jump();
            let correlation_depth : usize = 1_000_000;
            let mut correlation = 0.0;
            for _ in 0..correlation_depth{
                let a = rng.next_f64();
                let b = rng2.next_f64();
                correlation += a*b;
            }   
            
            correlation = correlation/correlation_depth as f64 - 0.25;           
            assert!(correlation < 1E-4);        
    }
}