//! Ramp (sawtooth) source.

use super::nco::{Nco, NcoSource, Phase, Waveform};
use lib_jsl_core::ErrorsJSL;

/// A rising sawtooth aligned with the sine: it crosses zero going up at phase 0, reaches `+1` just
/// before half a turn, jumps to `-1`, and rises again.
#[derive(Clone, Copy, Debug, Default)]
pub struct RampWave;

impl Waveform for RampWave {
    type Output = f64;

    fn sample(&mut self, phase: Phase) -> f64 {
        // Shifting by half a turn is an exact wrapping add on the fixed-point phase.
        let shifted = phase.0.wrapping_add(1 << 63);
        shifted as f64 / 18_446_744_073_709_551_616.0 * 2.0 - 1.0
    }
}

impl NcoSource<RampWave> {
    /// A rising sawtooth at `frequency_hz`. A negative frequency gives a falling sawtooth.
    pub fn ramp(frequency_hz: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, RampWave))
    }

    /// A rising sawtooth with an integer `period` in samples (>= 2).
    pub fn ramp_with_period(period: u64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::from_period(period)?, RampWave))
    }
}

/// Convenience: `length` samples of a unit sawtooth with an integer `period` (>= 2).
pub fn ramp(period: usize, length: usize) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(NcoSource::ramp_with_period(period as u64)?.generate(length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shape() {
        let x = ramp(8, 16).unwrap();
        let want = [0.0, 0.25, 0.5, 0.75, -1.0, -0.75, -0.5, -0.25];
        for (n, v) in x.iter().enumerate() {
            assert!((v - want[n % 8]).abs() < 1e-12, "n = {n}: {v}");
        }
        assert!(ramp(1, 4).is_err());
    }

    #[test]
    fn constant_slope_and_range() {
        let x = ramp(50, 500).unwrap();
        assert!(x.iter().all(|v| (-1.0..1.0).contains(v)));
        let mut jumps = 0;
        for w in x.windows(2) {
            let d = w[1] - w[0];
            if d < 0.0 {
                jumps += 1;
                assert!((d + 2.0 - 2.0 / 50.0).abs() < 1e-9);
            } else {
                assert!((d - 2.0 / 50.0).abs() < 1e-9);
            }
        }
        assert_eq!(jumps, 10, "one wrap every 50 samples, first at n = 25");
    }

    #[test]
    fn negative_frequency_falls() {
        let mut r = NcoSource::ramp(-1.0, 8.0).unwrap();
        let x = r.generate(4);
        assert!((x[0]).abs() < 1e-12 && (x[1] + 0.25).abs() < 1e-12 && (x[2] + 0.5).abs() < 1e-12);
    }

    #[test]
    fn zero_mean() {
        let x = ramp(100, 10_000).unwrap();
        assert!((x.iter().sum::<f64>() / x.len() as f64).abs() < 0.01);
    }
}
