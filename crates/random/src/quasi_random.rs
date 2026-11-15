//! Low-discrepancy (quasi-random) sequences: Sobol and Halton.
//!
//! Quasi-random points fill the unit cube far more evenly than pseudo-random ones, so averages
//! over them converge like `(log N)^d / N` instead of `N^{−1/2}`. They are deterministic; for
//! error estimates, apply independent random shifts (randomized quasi-Monte Carlo).
//!
//! See the book entry [Quasi-random sequences](https://jwkunz.github.io/lib_jsl/book/random/quasi-random.html).

use crate::uniform_generator::UniformGenerator;
use lib_jsl_core::ErrorsJSL;

use crate::sobol_directions::JOE_KUO;

const BITS: usize = 64;

/// The Sobol sequence in up to 1111 dimensions (Joe-Kuo direction numbers), generated in Gray-code
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
            return Err(ErrorsJSL::InvalidInputRange("The Sobol dimension must be between 1 and 1111."));
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

    // ANCHOR: sobol_scramble
    /// A scrambled Sobol sequence (SciPy's default, `Sobol(d, scramble=True)`): Matoušek's linear
    /// matrix scrambling of each coordinate's direction numbers by a random lower triangular
    /// binary matrix with unit diagonal, then a random digital shift (XOR). The points remain a
    /// `(t, m, d)`-net in every block of `2^m`, but are no longer biased towards the origin, and
    /// independent scramblings give independent, unbiased estimates.
    pub fn scrambled<G: UniformGenerator>(dimension: usize, rng: &mut G) -> Result<Self, ErrorsJSL> {
        let mut seq = Self::new(dimension)?;
        for (v, x) in seq.directions.iter_mut().zip(seq.state.iter_mut()) {
            // Row i of L acts on digit i (bit 63 − i); entries below the diagonal are random.
            let rows: Vec<u64> = (0..BITS).map(|i| {
                let below = if i == 0 { 0 } else { rng.next_u64() & !((1u64 << (BITS - i)) - 1) };
                below | (1u64 << (BITS - 1 - i))
            }).collect();
            for d in v.iter_mut() {
                *d = rows.iter().enumerate().fold(0u64, |acc, (i, r)| acc | (((r & *d).count_ones() as u64 & 1) << (BITS - 1 - i)));
            }
            *x = rng.next_u64();
        }
        Ok(seq)
    }
    // ANCHOR_END: sobol_scramble

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
    /// For a scrambled sequence, one random permutation of the digits per coordinate and digit
    /// position.
    permutations: Option<Vec<Vec<Vec<u64>>>>,
}

impl HaltonSequence {
    pub fn new(dimension: usize) -> Result<Self, ErrorsJSL> {
        if dimension == 0 || dimension > PRIMES.len() {
            return Err(ErrorsJSL::InvalidInputRange("The Halton dimension must be between 1 and 100."));
        }
        Ok(HaltonSequence { dimension, index: 1, permutations: None })
    }

    // ANCHOR: halton_scramble
    /// A scrambled Halton sequence (SciPy's `Halton(d, scramble=True)`): in coordinate `k`, digit
    /// position `j` of every index is mapped through one random permutation `π_{k,j}` of the
    /// digits `0..b`, over as many positions as a double resolves. Unlike the plain sequence it
    /// starts at index 0, and the permutations break up the correlations between large bases.
    pub fn scrambled<G: UniformGenerator>(dimension: usize, rng: &mut G) -> Result<Self, ErrorsJSL> {
        let mut seq = Self::new(dimension)?;
        seq.index = 0;
        let perms = PRIMES[..dimension]
            .iter()
            .map(|&b| {
                let digits = (54.0 / (b as f64).log2()).ceil() as usize;
                (0..digits)
                    .map(|_| {
                        let mut p: Vec<u64> = (0..b).collect();
                        for i in (1..b as usize).rev() {
                            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                            p.swap(i, j);
                        }
                        p
                    })
                    .collect()
            })
            .collect();
        seq.permutations = Some(perms);
        Ok(seq)
    }
    // ANCHOR_END: halton_scramble

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn next_point(&mut self) -> Vec<f64> {
        let point = match &self.permutations {
            None => PRIMES[..self.dimension].iter().map(|&p| radical_inverse(self.index, p)).collect(),
            Some(perms) => PRIMES[..self.dimension]
                .iter()
                .zip(perms)
                .map(|(&b, pk)| {
                    let (mut n, mut scale, mut x) = (self.index, 1.0 / b as f64, 0.0);
                    for pj in pk {
                        x += pj[(n % b) as usize] as f64 * scale;
                        n /= b;
                        scale /= b as f64;
                    }
                    x.min(1.0 - f64::EPSILON / 2.0)
                })
                .collect(),
        };
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
        assert!(SobolSequence::new(0).is_err() && SobolSequence::new(1112).is_err());
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
