//! Symbol constellations: BPSK, PSK, and square QAM with Gray mapping.
//!
//! A [`Constellation`] maps groups of bits to complex symbols and back. All constellations are
//! normalised to unit average symbol energy, and bits are mapped MSB first. PSK and QAM use Gray
//! labelling, so symbols that are nearest neighbours differ in exactly one bit; a single symbol
//! error from noise therefore usually costs one bit error.
//!
//! Demapping is available as hard decisions (nearest symbol) and as soft information: the
//! max-log log-likelihood ratio `ln P(bit = 0) / P(bit = 1)` per bit (positive favours 0), the
//! convention used by the LDPC and polar decoders in the `information` crate.

use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;
use std::f64::consts::TAU;

/// A symbol constellation with a bit labelling.
#[derive(Clone, Debug, PartialEq)]
pub struct Constellation {
    /// `points[label]` is the symbol sent for the bit pattern `label` (MSB first).
    points: Vec<Complex<f64>>,
    bits_per_symbol: usize,
}

fn gray(n: usize) -> usize {
    n ^ (n >> 1)
}

fn normalised(points: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let power = points.iter().map(|p| p.norm_sqr()).sum::<f64>() / points.len() as f64;
    let scale = 1.0 / power.sqrt();
    points.into_iter().map(|p| p * scale).collect()
}

impl Constellation {
    /// Binary phase-shift keying: bit 0 → +1, bit 1 → −1.
    pub fn bpsk() -> Constellation {
        Constellation { points: vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)], bits_per_symbol: 1 }
    }

    /// Quadrature PSK with Gray labelling (the same constellation as 4-QAM).
    pub fn qpsk() -> Constellation {
        Constellation::qam(4).expect("4 is a valid QAM order")
    }

    /// `m`-ary PSK (`m` a power of two, at least 2) on the unit circle with Gray labelling:
    /// symbol `k` sits at angle `2πk/m`.
    pub fn psk(m: usize) -> Result<Constellation, ErrorsJSL> {
        if m < 2 || !m.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange("PSK order must be a power of two, at least 2"));
        }
        let mut points = vec![Complex::new(0.0, 0.0); m];
        for k in 0..m {
            points[gray(k)] = Complex::from_polar(1.0, TAU * k as f64 / m as f64);
        }
        Ok(Constellation { points, bits_per_symbol: m.trailing_zeros() as usize })
    }

    /// Square `m`-QAM (`m` = 4, 16, 64, 256 or 1024) with independent Gray-coded PAM on the I and Q
    /// axes: the first half of the bits label I, the second half label Q.
    pub fn qam(m: usize) -> Result<Constellation, ErrorsJSL> {
        let side = (m as f64).sqrt().round() as usize;
        if side * side != m || !side.is_power_of_two() || !(2..=32).contains(&side) {
            return Err(ErrorsJSL::InvalidInputRange("QAM order must be 4, 16, 64, 256 or 1024"));
        }
        let half_bits = side.trailing_zeros() as usize;
        let mut points = vec![Complex::new(0.0, 0.0); m];
        for i in 0..side {
            for q in 0..side {
                // PAM level index i has the Gray label gray(i); levels are -(side-1), …, side-1.
                let label = (gray(i) << half_bits) | gray(q);
                points[label] = Complex::new(2.0 * i as f64 - (side - 1) as f64, 2.0 * q as f64 - (side - 1) as f64);
            }
        }
        Ok(Constellation { points: normalised(points), bits_per_symbol: 2 * half_bits })
    }

    /// Number of symbols.
    pub fn size(&self) -> usize {
        self.points.len()
    }

    pub fn bits_per_symbol(&self) -> usize {
        self.bits_per_symbol
    }

    /// The symbols, indexed by bit label.
    pub fn points(&self) -> &[Complex<f64>] {
        &self.points
    }

    /// Mean symbol energy (1 by construction).
    pub fn average_power(&self) -> f64 {
        self.points.iter().map(|p| p.norm_sqr()).sum::<f64>() / self.size() as f64
    }

    /// Smallest distance between two distinct symbols.
    pub fn min_distance(&self) -> f64 {
        let mut best = f64::INFINITY;
        for (i, a) in self.points.iter().enumerate() {
            for b in &self.points[i + 1..] {
                best = best.min((a - b).norm());
            }
        }
        best
    }

    /// The symbol for a bit label.
    pub fn symbol(&self, label: usize) -> Option<Complex<f64>> {
        self.points.get(label).copied()
    }

    /// Maps bits (each 0 or 1, MSB first within a symbol) to symbols; the bit count must be a
    /// multiple of [`bits_per_symbol`](Self::bits_per_symbol).
    pub fn map_bits(&self, bits: &[u8]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        let k = self.bits_per_symbol;
        if !bits.len().is_multiple_of(k) {
            return Err(ErrorsJSL::IncompatibleArraySizes((bits.len(), k)));
        }
        Ok(bits.chunks(k).map(|c| self.points[c.iter().fold(0usize, |acc, &b| (acc << 1) | (b & 1) as usize)]).collect())
    }

    /// The label of the symbol nearest to `sample`.
    pub fn nearest(&self, sample: Complex<f64>) -> usize {
        self.points
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - sample).norm_sqr().partial_cmp(&(b.1 - sample).norm_sqr()).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// The symbol nearest to `sample` (a slicer).
    pub fn slice(&self, sample: Complex<f64>) -> Complex<f64> {
        self.points[self.nearest(sample)]
    }

    /// Hard-decision demapping to bits (MSB first).
    pub fn demap_hard(&self, samples: &[Complex<f64>]) -> Vec<u8> {
        let k = self.bits_per_symbol;
        samples
            .iter()
            .flat_map(|&s| {
                let label = self.nearest(s);
                (0..k).map(move |b| ((label >> (k - 1 - b)) & 1) as u8)
            })
            .collect()
    }

    /// Soft demapping: the max-log LLR `ln P(bit=0)/P(bit=1)` of every bit, for complex noise of
    /// total variance `noise_variance` per sample (`E|n|²`).
    pub fn demap_soft(&self, samples: &[Complex<f64>], noise_variance: f64) -> Result<Vec<f64>, ErrorsJSL> {
        if !(noise_variance > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Noise variance must be positive"));
        }
        let k = self.bits_per_symbol;
        let mut llr = Vec::with_capacity(samples.len() * k);
        for &s in samples {
            for b in 0..k {
                let shift = k - 1 - b;
                let (mut d0, mut d1) = (f64::INFINITY, f64::INFINITY);
                for (label, p) in self.points.iter().enumerate() {
                    let d = (s - p).norm_sqr();
                    if (label >> shift) & 1 == 0 {
                        d0 = d0.min(d);
                    } else {
                        d1 = d1.min(d);
                    }
                }
                llr.push((d1 - d0) / noise_variance);
            }
        }
        Ok(llr)
    }
}

/// The 16-QAM constellation (4 bits per symbol, points `(±1, ±3)/√10`).
pub fn qam16() -> Constellation {
    Constellation::qam(16).expect("16 is a valid QAM order")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn gauss(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>()
    }

    fn all_constellations() -> Vec<(&'static str, Constellation)> {
        vec![
            ("bpsk", Constellation::bpsk()),
            ("qpsk", Constellation::qpsk()),
            ("8psk", Constellation::psk(8).unwrap()),
            ("16psk", Constellation::psk(16).unwrap()),
            ("16qam", qam16()),
            ("64qam", Constellation::qam(64).unwrap()),
            ("256qam", Constellation::qam(256).unwrap()),
        ]
    }

    #[test]
    fn unit_power_and_sizes() {
        for (name, c) in all_constellations() {
            assert!((c.average_power() - 1.0).abs() < 1e-12, "{name}");
            assert_eq!(c.size(), 1 << c.bits_per_symbol(), "{name}");
        }
        assert_eq!(qam16().bits_per_symbol(), 4);
        assert!(Constellation::qam(8).is_err() && Constellation::qam(2048).is_err());
        assert!(Constellation::psk(3).is_err() && Constellation::psk(1).is_err());
    }

    #[test]
    fn sixteen_qam_has_the_expected_geometry() {
        let c = qam16();
        let s = 1.0 / 10f64.sqrt();
        let mut levels: Vec<f64> = c.points().iter().map(|p| p.re / s).collect();
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        levels.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        assert_eq!(levels.iter().map(|v| v.round() as i32).collect::<Vec<_>>(), vec![-3, -1, 1, 3]);
        assert!((c.min_distance() - 2.0 * s).abs() < 1e-12);
        // Label 0b0000 is the corner (I and Q both at the Gray label 00 = level index 0 = -3).
        assert!((c.symbol(0).unwrap() - Complex::new(-3.0 * s, -3.0 * s)).norm() < 1e-12);
    }

    #[test]
    fn gray_labelling_nearest_neighbours_differ_by_one_bit() {
        for (name, c) in all_constellations() {
            let d = c.min_distance();
            for a in 0..c.size() {
                for b in (a + 1)..c.size() {
                    if (c.points()[a] - c.points()[b]).norm() < d * 1.0001 {
                        assert_eq!((a ^ b).count_ones(), 1, "{name}: labels {a} and {b} are neighbours");
                    }
                }
            }
        }
    }

    #[test]
    fn mapping_round_trips_through_hard_demapping() {
        let mut seed = 5;
        for (name, c) in all_constellations() {
            let bits: Vec<u8> = (0..c.bits_per_symbol() * 200).map(|_| (lcg(&mut seed) > 0.0) as u8).collect();
            let symbols = c.map_bits(&bits).unwrap();
            assert_eq!(symbols.len(), 200);
            assert_eq!(c.demap_hard(&symbols), bits, "{name}");
        }
        assert!(qam16().map_bits(&[0, 1, 0]).is_err());
    }

    #[test]
    fn soft_demapping_agrees_with_hard_decisions_and_scales_with_noise() {
        let mut seed = 9;
        for (name, c) in all_constellations() {
            let samples: Vec<Complex<f64>> = (0..100).map(|_| c.points()[(lcg(&mut seed) * 1e6).abs() as usize % c.size()] + Complex::new(gauss(&mut seed), gauss(&mut seed)) * 0.03).collect();
            let hard = c.demap_hard(&samples);
            let llr = c.demap_soft(&samples, 0.01).unwrap();
            assert_eq!(llr.len(), hard.len());
            for (l, h) in llr.iter().zip(&hard) {
                assert_eq!(*l < 0.0, *h == 1, "{name}: LLR sign must match the hard decision");
            }
            let llr2 = c.demap_soft(&samples, 0.02).unwrap();
            assert!(llr.iter().zip(&llr2).all(|(a, b)| (a / 2.0 - b).abs() < 1e-9), "LLRs scale as 1/variance");
        }
        assert!(Constellation::bpsk().demap_soft(&[Complex::new(1.0, 0.0)], 0.0).is_err());
        // BPSK max-log LLR is exactly 4 Re(y) / σ² (for E|n|² = σ²).
        let l = Constellation::bpsk().demap_soft(&[Complex::new(0.4, 0.2)], 0.5).unwrap()[0];
        assert!((l - 4.0 * 0.4 / 0.5).abs() < 1e-12);
    }

    #[test]
    fn qpsk_bit_error_rate_matches_theory() {
        // BER = Q(sqrt(2 Eb/N0)); at Eb/N0 = 4 dB this is about 1.25e-2.
        let c = Constellation::qpsk();
        let ebn0 = 10f64.powf(0.4);
        let sigma2 = 1.0 / (2.0 * ebn0); // Es = 1 = 2 Eb, N0 = sigma2 (total complex variance)
        let mut seed = 77;
        let n = 100_000;
        let bits: Vec<u8> = (0..2 * n).map(|_| (lcg(&mut seed) > 0.0) as u8).collect();
        let rx: Vec<Complex<f64>> = c.map_bits(&bits).unwrap().into_iter().map(|s| s + Complex::new(gauss(&mut seed), gauss(&mut seed)) * (sigma2 / 2.0).sqrt()).collect();
        let errors = c.demap_hard(&rx).iter().zip(&bits).filter(|(a, b)| a != b).count();
        let ber = errors as f64 / bits.len() as f64;
        let q = 0.5 * erfc_approx((ebn0).sqrt());
        assert!((ber / q - 1.0).abs() < 0.15, "measured {ber}, theory {q}");
    }

    /// Complementary error function (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7).
    fn erfc_approx(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        poly * (-x * x).exp()
    }
}
