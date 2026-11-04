//! Noise source.
//!
//! See the book entry [Signal sources: oscillators, CORDIC and waveforms](https://jwkunz.github.io/lib_jsl/book/comms/sources.html#waveforms-and-noise).

use super::nco::{Nco, NcoSource, Phase, Waveform};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::distributions::gaussian_distribution_box_muller;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;

/// The amplitude distribution of the noise before amplitude/offset scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseKind {
    /// Uniform on `[-1, 1)`: mean 0, variance 1/3.
    Uniform,
    /// Gaussian with mean 0 and variance 1.
    Gaussian,
}

/// A pseudo-random noise waveform.
///
/// In *white* mode a fresh value is drawn for every sample. In *sample-and-hold* mode a new value
/// is drawn each time the oscillator phase wraps and held in between, giving noise that changes at
/// the oscillator frequency (this requires a positive oscillator frequency). The generator is
/// seeded, so output is reproducible and [`reset`](Waveform::reset) replays the same sequence.
#[derive(Clone, Debug)]
pub struct NoiseWave {
    kind: NoiseKind,
    seed: u64,
    rng: Xoshiro256PlusPlus,
    hold: bool,
    last_phase: u64,
    held: f64,
    started: bool,
    spare: Option<f64>,
}

impl NoiseWave {
    fn new(kind: NoiseKind, seed: u64, hold: bool) -> NoiseWave {
        NoiseWave { kind, seed, rng: Xoshiro256PlusPlus::from_seed(seed), hold, last_phase: 0, held: 0.0, started: false, spare: None }
    }

    fn draw(&mut self) -> f64 {
        match self.kind {
            NoiseKind::Uniform => 2.0 * self.rng.next_f64() - 1.0,
            NoiseKind::Gaussian => {
                if let Some(v) = self.spare.take() {
                    return v;
                }
                loop {
                    let (a, b) = gaussian_distribution_box_muller(&mut self.rng, 0.0, 1.0);
                    if a.is_finite() && b.is_finite() {
                        self.spare = Some(b);
                        return a;
                    }
                }
            }
        }
    }
}

impl Waveform for NoiseWave {
    type Output = f64;

    fn sample(&mut self, phase: Phase) -> f64 {
        if !self.hold {
            return self.draw();
        }
        if !self.started || phase.0 < self.last_phase {
            self.held = self.draw();
            self.started = true;
        }
        self.last_phase = phase.0;
        self.held
    }

    fn reset(&mut self) {
        self.rng = Xoshiro256PlusPlus::from_seed(self.seed);
        self.last_phase = 0;
        self.held = 0.0;
        self.started = false;
        self.spare = None;
    }
}

impl NcoSource<NoiseWave> {
    /// White noise: an independent value every sample.
    pub fn white_noise(kind: NoiseKind, seed: u64) -> Self {
        let nco = Nco::new(0.25, 1.0).expect("valid placeholder oscillator");
        NcoSource::from_parts(nco, NoiseWave::new(kind, seed, false))
    }

    /// Sample-and-hold noise that draws a new value `rate_hz` times per second.
    pub fn sample_and_hold_noise(kind: NoiseKind, seed: u64, rate_hz: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        if rate_hz <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Sample-and-hold noise needs a positive update rate."));
        }
        Ok(NcoSource::from_parts(Nco::new(rate_hz, sample_rate)?, NoiseWave::new(kind, seed, true)))
    }
}

/// Convenience: `length` samples of unit-variance white Gaussian noise from the given `seed`.
pub fn noise(length: usize, seed: u64) -> Vec<f64> {
    NcoSource::white_noise(NoiseKind::Gaussian, seed).generate(length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};

    fn mean_var(x: &[f64]) -> (f64, f64) {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        (m, x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64)
    }

    #[test]
    fn gaussian_statistics() {
        let x = noise(200_000, 1);
        let (m, v) = mean_var(&x);
        assert!(m.abs() < 0.01 && (v - 1.0).abs() < 0.02, "mean {m} var {v}");
        // About 68% within one standard deviation.
        let within = x.iter().filter(|v| v.abs() < 1.0).count() as f64 / x.len() as f64;
        assert!((within - 0.6827).abs() < 0.01);
    }

    #[test]
    fn uniform_statistics_and_bounds() {
        let x = NcoSource::white_noise(NoiseKind::Uniform, 2).generate(200_000);
        assert!(x.iter().all(|v| (-1.0..1.0).contains(v)));
        let (m, v) = mean_var(&x);
        assert!(m.abs() < 0.01 && (v - 1.0 / 3.0).abs() < 0.01, "mean {m} var {v}");
    }

    #[test]
    fn white_noise_is_uncorrelated() {
        let x = noise(100_000, 3);
        for lag in 1..5 {
            let c: f64 = x.iter().zip(&x[lag..]).map(|(a, b)| a * b).sum::<f64>() / x.len() as f64;
            assert!(c.abs() < 0.02, "lag {lag}: {c}");
        }
    }

    #[test]
    fn seeded_and_resettable() {
        assert_eq!(noise(100, 5), noise(100, 5));
        assert_ne!(noise(100, 5), noise(100, 6));
        let mut s = NcoSource::white_noise(NoiseKind::Gaussian, 9);
        let first = s.generate(51); // odd count exercises the cached second Box-Muller value
        StreamOperatorManagement::reset(&mut s).unwrap();
        assert_eq!(s.generate(51), first);
    }

    #[test]
    fn amplitude_and_offset_scale_the_noise() {
        let mut s = NcoSource::white_noise(NoiseKind::Gaussian, 4).with_amplitude(3.0).with_offset(5.0);
        let (m, v) = mean_var(&s.generate(100_000));
        assert!((m - 5.0).abs() < 0.05 && (v - 9.0).abs() < 0.2, "mean {m} var {v}");
    }

    #[test]
    fn sample_and_hold_updates_at_the_oscillator_rate() {
        // 25 Hz at 100 Hz sampling: a new value every 4 samples.
        let mut s = NcoSource::sample_and_hold_noise(NoiseKind::Uniform, 7, 25.0, 100.0).unwrap();
        let x = s.generate(40);
        for block in x.chunks(4) {
            assert!(block.iter().all(|&v| v == block[0]), "{block:?}");
        }
        let distinct: std::collections::BTreeSet<u64> = x.iter().map(|v| v.to_bits()).collect();
        assert_eq!(distinct.len(), 10);
        assert!(NcoSource::sample_and_hold_noise(NoiseKind::Uniform, 1, 0.0, 100.0).is_err());
        assert!(NcoSource::sample_and_hold_noise(NoiseKind::Uniform, 1, 80.0, 100.0).is_err());
    }

    #[test]
    fn stream_operator_interface() {
        let mut s = NcoSource::white_noise(NoiseKind::Gaussian, 11);
        let a = StreamOperator::<(), f64>::process(&mut s, &[(); 30]).unwrap().unwrap();
        assert_eq!(a, noise(30, 11));
    }
}
