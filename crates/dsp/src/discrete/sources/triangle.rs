//! Triangle wave source.
//!
//! See the book entry [Signal sources: oscillators, CORDIC and waveforms](https://jwkunz.github.io/lib_jsl/book/comms/sources.html#waveforms-and-noise).

use super::nco::{Nco, NcoSource, Phase, Waveform};
use lib_jsl_core::ErrorsJSL;

/// A symmetric triangle wave aligned with the sine: 0 at phase 0, `+1` at a quarter turn, 0 at
/// half a turn, `-1` at three quarters.
#[derive(Clone, Copy, Debug, Default)]
pub struct TriangleWave;

impl Waveform for TriangleWave {
    type Output = f64;

    fn sample(&mut self, phase: Phase) -> f64 {
        let t = phase.turns();
        if t < 0.25 {
            4.0 * t
        } else if t < 0.75 {
            2.0 - 4.0 * t
        } else {
            4.0 * t - 4.0
        }
    }
}

impl NcoSource<TriangleWave> {
    /// A triangle wave at `frequency_hz`.
    pub fn triangle(frequency_hz: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, TriangleWave))
    }

    /// A triangle wave with an integer `period` in samples (>= 2).
    pub fn triangle_with_period(period: u64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::from_period(period)?, TriangleWave))
    }
}

/// Convenience: `length` samples of a unit triangle wave with an integer `period` (>= 2).
pub fn triangle(period: usize, length: usize) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(NcoSource::triangle_with_period(period as u64)?.generate(length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shape() {
        let x = triangle(8, 8).unwrap();
        let want = [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        for (a, b) in x.iter().zip(want) {
            assert!((a - b).abs() < 1e-12, "{x:?}");
        }
        let x = triangle(4, 4).unwrap();
        assert!((x[0]).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12 && x[2].abs() < 1e-12 && (x[3] + 1.0).abs() < 1e-12);
        assert!(triangle(1, 4).is_err());
    }

    #[test]
    fn periodic_continuous_and_bounded() {
        let x = triangle(100, 1000).unwrap();
        for i in 0..900 {
            assert!((x[i] - x[i + 100]).abs() < 1e-9, "periodicity");
        }
        assert!(x.iter().all(|v| v.abs() <= 1.0 + 1e-12));
        // The slope magnitude is constant (4 / period) away from the corners.
        for w in x.windows(2) {
            assert!(((w[1] - w[0]).abs() - 0.04).abs() < 1e-9);
        }
    }

    #[test]
    fn triangle_is_the_arcsine_of_the_sine() {
        let mut tri = NcoSource::triangle(3.0, 100.0).unwrap();
        let mut sin = NcoSource::sinusoid(3.0, 100.0).unwrap();
        for _ in 0..200 {
            let (t, s) = (tri.next_sample(), sin.next_sample());
            assert!((t - 2.0 / std::f64::consts::PI * s.asin()).abs() < 1e-9);
        }
    }
}
