//! Low-discrepancy (quasi-random) sequences: Sobol and Halton.
//!
//! Quasi-random points fill the unit cube far more evenly than pseudo-random ones, so averages
//! over them converge like `(log N)^d / N` instead of `N^{−1/2}`. They are deterministic; for
//! error estimates, apply independent random shifts (randomized quasi-Monte Carlo).
//!
//! See the book entry [Quasi-random sequences](https://jwkunz.github.io/lib_jsl/book/random/quasi-random.html).

use crate::uniform_generator::UniformGenerator;
use lib_jsl_core::ErrorsJSL;

/// Primitive polynomials and initial direction numbers `(degree s, coefficients a, m_1..m_s)`
/// for Sobol dimensions 2 onwards, from Joe and Kuo's `new-joe-kuo-6.21201` table.
const JOE_KUO: [(u32, u32, &[u64]); 20] = [
    (1, 0, &[1]),
    (2, 1, &[1, 3]),
    (3, 1, &[1, 3, 1]),
    (3, 2, &[1, 1, 1]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 5, 5, 17]),
    (5, 4, &[1, 1, 5, 5, 5]),
    (5, 7, &[1, 1, 7, 11, 19]),
    (5, 11, &[1, 1, 5, 1, 1]),
    (5, 13, &[1, 1, 1, 3, 11]),
    (5, 14, &[1, 3, 5, 5, 31]),
    (6, 1, &[1, 3, 3, 9, 7, 49]),
    (6, 13, &[1, 1, 1, 15, 21, 21]),
    (6, 16, &[1, 3, 1, 13, 27, 49]),
    (6, 19, &[1, 1, 1, 15, 7, 5]),
    (6, 22, &[1, 3, 1, 15, 13, 25]),
    (6, 25, &[1, 1, 5, 5, 19, 61]),
    (7, 1, &[1, 3, 7, 11, 23, 15, 103]),
    (7, 4, &[1, 3, 7, 13, 13, 15, 69]),
];

const BITS: usize = 64;

/// The Sobol sequence in up to 21 dimensions (Joe-Kuo direction numbers), generated in Gray-code
/// order (Antonov and Saleev), one XOR per coordinate per point. The first point is the origin;
/// every block of `2^k` consecutive points starting at a multiple of `2^k` is a `(t, k, d)`-net.
#[derive(Debug, Clone)]
pub struct SobolSequence {
    directions: Vec<[u64; BITS]>,
    state: Vec<u64>,
    index: u64,
}

impl SobolSequence {
    /// The largest supported dimension.
    pub const MAX_DIMENSION: usize = JOE_KUO.len() + 1;

    pub fn new(dimension: usize) -> Result<Self, ErrorsJSL> {
        if dimension == 0 || dimension > Self::MAX_DIMENSION {
            return Err(ErrorsJSL::InvalidInputRange("The Sobol dimension must be between 1 and 21."));
        }
        let mut directions = Vec::with_capacity(dimension);
        let mut first = [0u64; BITS];
        for (i, v) in first.iter_mut().enumerate() {
            *v = 1u64 << (BITS - 1 - i);
        }
        directions.push(first);
        for &(s, a, m) in JOE_KUO.iter().take(dimension - 1) {
            let s = s as usize;
            let mut v = [0u64; BITS];
            for i in 0..s {
                v[i] = m[i] << (BITS - 1 - i);
            }
            for i in s..BITS {
                let mut value = v[i - s] ^ (v[i - s] >> s);
                for k in 1..s {
                    if (a >> (s - 1 - k)) & 1 == 1 {
                        value ^= v[i - k];
                    }
                }
                v[i] = value;
            }
            directions.push(v);
        }
        Ok(SobolSequence { directions, state: vec![0; dimension], index: 0 })
    }

    pub fn dimension(&self) -> usize {
        self.directions.len()
    }

    // ANCHOR: sobol_next
    /// The next point in `[0, 1)^d`.
    pub fn next_point(&mut self) -> Vec<f64> {
        let point = self.state.iter().map(|&x| (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)).collect();
        // Advance: flip the direction number at the lowest zero bit of the current index.
        let bit = self.index.trailing_ones() as usize;
        for (x, v) in self.state.iter_mut().zip(&self.directions) {
            *x ^= v[bit.min(BITS - 1)];
        }
        self.index += 1;
        point
    }
    // ANCHOR_END: sobol_next

    /// Skips `count` points.
    pub fn skip(&mut self, count: u64) {
        for _ in 0..count {
            self.next_point();
        }
    }
}

const PRIMES: [u64; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
    307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

// ANCHOR: radical_inverse
/// The van der Corput radical inverse of `n` in base `b`: the base-b digits of `n` mirrored about
/// the radix point.
pub fn radical_inverse(mut n: u64, base: u64) -> f64 {
    let inv = 1.0 / base as f64;
    let (mut value, mut scale) = (0.0, inv);
    while n > 0 {
        value += (n % base) as f64 * scale;
        n /= base;
        scale *= inv;
    }
    value
}
// ANCHOR_END: radical_inverse

/// The Halton sequence: coordinate `k` is the radical inverse in the k-th prime base. Simple and
/// extensible to any number of points; uniformity degrades in high dimensions (large bases), where
/// Sobol is preferable. Supports up to 100 dimensions. Starts at index 1 (skipping the origin).
#[derive(Debug, Clone)]
pub struct HaltonSequence {
    dimension: usize,
    index: u64,
}

impl HaltonSequence {
    pub fn new(dimension: usize) -> Result<Self, ErrorsJSL> {
        if dimension == 0 || dimension > PRIMES.len() {
            return Err(ErrorsJSL::InvalidInputRange("The Halton dimension must be between 1 and 100."));
        }
        Ok(HaltonSequence { dimension, index: 1 })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn next_point(&mut self) -> Vec<f64> {
        let point = PRIMES[..self.dimension].iter().map(|&p| radical_inverse(self.index, p)).collect();
        self.index += 1;
        point
    }

    pub fn skip(&mut self, count: u64) {
        self.index += count;
    }
}

/// Applies a Cranley-Patterson rotation `x ← (x + u) mod 1` with a uniform random shift `u` per
/// coordinate. Independent shifts give unbiased, independent replicates of a quasi-Monte Carlo
/// estimate, whose spread estimates its error.
pub fn random_shift<G: UniformGenerator>(points: &mut [Vec<f64>], rng: &mut G) {
    let Some(d) = points.first().map(|p| p.len()) else { return };
    let shift: Vec<f64> = (0..d).map(|_| rng.next_f64()).collect();
    for p in points.iter_mut() {
        for (x, u) in p.iter_mut().zip(&shift) {
            *x = (*x + u).fract();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    /// Whether x^s + a₁x^{s−1} + … + a_{s−1}x + 1 is primitive over GF(2): the order of x modulo
    /// it must be exactly 2^s − 1.
    fn is_primitive(s: u32, a: u32) -> bool {
        let poly = (1u64 << s) | ((a as u64) << 1) | 1;
        let period = (1u64 << s) - 1;
        let mut x = 1u64;
        for k in 1..=period {
            x <<= 1;
            if x >> s & 1 == 1 {
                x ^= poly;
            }
            if x == 1 {
                return k == period;
            }
        }
        false
    }

    #[test]
    fn direction_number_table_is_valid() {
        let mut seen = std::collections::HashSet::new();
        for &(s, a, m) in &JOE_KUO {
            assert!(is_primitive(s, a), "polynomial ({s}, {a}) is not primitive");
            assert!(seen.insert((s, a)), "duplicate polynomial");
            assert_eq!(m.len(), s as usize);
            for (i, &mi) in m.iter().enumerate() {
                assert!(mi % 2 == 1 && mi < 1 << (i + 1), "m_{} = {mi}", i + 1);
            }
        }
    }

    #[test]
    fn sobol_points_form_nets() {
        let mut sobol = SobolSequence::new(SobolSequence::MAX_DIMENSION).unwrap();
        let points: Vec<Vec<f64>> = (0..1024).map(|_| sobol.next_point()).collect();
        assert!(points[0].iter().all(|&x| x == 0.0) && points[1].iter().all(|&x| x == 0.5));
        // Every 1D projection of the first 2^k points is exactly the grid {i / 2^k}.
        for d in 0..SobolSequence::MAX_DIMENSION {
            let mut cells: Vec<u64> = points[..256].iter().map(|p| (p[d] * 256.0) as u64).collect();
            cells.sort_unstable();
            assert!(cells.iter().enumerate().all(|(i, &c)| c == i as u64), "dimension {d}");
        }
        // The first two dimensions form a (0, 10, 2)-net: each of the 2^10 elementary boxes of
        // any shape 2^{−i} × 2^{−(10−i)} holds exactly one point.
        for i in 0..=10 {
            let mut count = vec![0; 1024];
            for p in &points {
                let (bx, by) = ((p[0] * (1 << i) as f64) as usize, (p[1] * (1 << (10 - i)) as f64) as usize);
                count[bx * (1 << (10 - i)) + by] += 1;
            }
            assert!(count.iter().all(|&c| c == 1), "shape {i}");
        }
        assert!(SobolSequence::new(0).is_err() && SobolSequence::new(22).is_err());
    }

    #[test]
    fn halton_matches_radical_inverses_and_shifts_stay_in_the_cube() {
        assert_eq!(radical_inverse(6, 2), 0.375); // 110₂ → 0.011₂
        assert!((radical_inverse(7, 3) - (1.0 / 3.0 + 2.0 / 9.0)).abs() < 1e-15); // 21₃ → 0.12₃
        let mut h = HaltonSequence::new(3).unwrap();
        assert_eq!(h.next_point(), vec![0.5, 1.0 / 3.0, 0.2]);
        h.skip(2);
        assert_eq!(h.next_point()[0], 0.125);
        // Star discrepancy proxy: the mean of x·y·z over 4096 points is close to 1/8.
        let mut h = HaltonSequence::new(3).unwrap();
        let mut pts: Vec<Vec<f64>> = (0..4096).map(|_| h.next_point()).collect();
        let mean: f64 = pts.iter().map(|p| p[0] * p[1] * p[2]).sum::<f64>() / 4096.0;
        assert!((mean - 0.125).abs() < 1e-3);
        random_shift(&mut pts, &mut UniformRNG::from_seed(4));
        assert!(pts.iter().flatten().all(|&x| (0.0..1.0).contains(&x)));
        assert!(HaltonSequence::new(101).is_err());
    }
}
