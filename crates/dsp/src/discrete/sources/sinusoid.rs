//! Sinusoid source.

use super::cordic::{Cordic, ONE};
use super::nco::{Nco, NcoSource, Phase, Waveform};
use lib_jsl_core::ErrorsJSL;

/// How the sine and cosine of the oscillator phase are evaluated.
#[derive(Clone, Debug)]
pub enum SineKernel {
    /// The platform's floating-point `sin` and `cos`.
    Direct,
    /// An integer [`Cordic`] rotator fed with the top 32 bits of the oscillator phase (so the
    /// output is accurate to about 1e-8).
    Cordic(Cordic),
}

impl SineKernel {
    /// Returns `(sin, cos)` of `2π · phase`.
    pub fn sin_cos(&self, phase: Phase) -> (f64, f64) {
        match self {
            SineKernel::Direct => phase.radians().sin_cos(),
            SineKernel::Cordic(c) => {
                // Round the 64-bit phase to the 32-bit binary angle the rotator takes.
                let angle = (phase.0.wrapping_add(1 << 31) >> 32) as u32;
                let (s, co) = c.sin_cos_fixed(angle);
                (s as f64 / ONE as f64, co as f64 / ONE as f64)
            }
        }
    }
}

/// The sine waveform, `sin(2π · phase)`.
#[derive(Clone, Debug)]
pub struct SineWave {
    kernel: SineKernel,
}

impl SineWave {
    pub fn new(kernel: SineKernel) -> SineWave {
        SineWave { kernel }
    }
}

impl Waveform for SineWave {
    type Output = f64;

    fn sample(&mut self, phase: Phase) -> f64 {
        self.kernel.sin_cos(phase).0
    }
}

impl NcoSource<SineWave> {
    /// A sine source at `frequency_hz` for the given sampling rate (starting at phase 0).
    pub fn sinusoid(frequency_hz: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, SineWave::new(SineKernel::Direct)))
    }

    /// Like [`sinusoid`](Self::sinusoid) but evaluating the sine with `iterations` (1 to 30)
    /// integer CORDIC steps.
    pub fn sinusoid_cordic(frequency_hz: f64, sample_rate: f64, iterations: u32) -> Result<Self, ErrorsJSL> {
        let kernel = SineKernel::Cordic(Cordic::new(iterations)?);
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, SineWave::new(kernel)))
    }
}

/// Convenience: `length` samples of a unit sinusoid at `frequency_hz`.
pub fn sinusoid(frequency_hz: f64, sample_rate: f64, length: usize) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(NcoSource::sinusoid(frequency_hz, sample_rate)?.generate(length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::stream_operator::StreamOperator;

    #[test]
    fn matches_the_analytic_sine() {
        let x = sinusoid(440.0, 8000.0, 2000).unwrap();
        for (n, v) in x.iter().enumerate() {
            let want = (std::f64::consts::TAU * 440.0 * n as f64 / 8000.0).sin();
            assert!((v - want).abs() < 1e-9, "n = {n}");
        }
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn no_phase_drift_over_long_runs() {
        // 3 Hz at 1 kHz: after 10 million samples the accumulated phase error stays negligible.
        let mut s = NcoSource::sinusoid(3.0, 1000.0).unwrap();
        let _ = s.generate(10_000_000 - 1);
        let last = s.next_sample();
        let want = (std::f64::consts::TAU * 3.0 * 9_999_999.0 / 1000.0).sin();
        assert!((last - want).abs() < 1e-6, "{last} vs {want}");
    }

    #[test]
    fn cordic_kernel_agrees_with_direct() {
        let mut direct = NcoSource::sinusoid(1234.5, 44_100.0).unwrap();
        let mut cordic = NcoSource::sinusoid_cordic(1234.5, 44_100.0, 30).unwrap();
        for _ in 0..3000 {
            assert!((direct.next_sample() - cordic.next_sample()).abs() < 3e-8);
        }
        assert!(NcoSource::sinusoid_cordic(10.0, 100.0, 0).is_err());
        assert!(NcoSource::sinusoid_cordic(10.0, 100.0, 31).is_err());
    }

    #[test]
    fn amplitude_offset_and_quadrature() {
        let mut cosine = NcoSource::sinusoid(100.0, 1000.0).unwrap().with_phase_turns(0.25).with_amplitude(2.0).with_offset(1.0);
        let x = cosine.generate(10);
        for (n, v) in x.iter().enumerate() {
            let want = 1.0 + 2.0 * (std::f64::consts::TAU * 0.1 * n as f64).cos();
            assert!((v - want).abs() < 1e-12);
        }
    }

    #[test]
    fn works_through_the_stream_operator_trait() {
        let mut s = NcoSource::sinusoid(50.0, 1000.0).unwrap();
        let a = StreamOperator::<(), f64>::process(&mut s, &[(); 100]).unwrap().unwrap();
        assert_eq!(a, sinusoid(50.0, 1000.0, 100).unwrap());
    }

    #[test]
    fn tone_has_unit_power_at_the_right_frequency() {
        let n = 4000;
        let x = sinusoid(250.0, 4000.0, n).unwrap();
        let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(r, i), (k, v)| {
            let a = std::f64::consts::TAU * 250.0 * k as f64 / 4000.0;
            (r + v * a.cos(), i - v * a.sin())
        });
        assert!(((re * re + im * im).sqrt() * 2.0 / n as f64 - 1.0).abs() < 1e-9);
        assert!(sinusoid(5000.0, 4000.0, 10).is_err());
    }
}
