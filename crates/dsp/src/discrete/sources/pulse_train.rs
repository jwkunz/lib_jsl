//! Pulse train source.

use super::nco::{threshold_from_duty, threshold_from_ratio, Nco, NcoSource, Phase, Waveform};
use lib_jsl_core::ErrorsJSL;

/// A unipolar pulse train: `1` for the first part of every cycle (the duty fraction), `0` after.
#[derive(Clone, Debug)]
pub struct PulseTrainWave {
    threshold: u128,
}

impl PulseTrainWave {
    /// A pulse train with the given duty cycle in `[0, 1]`.
    pub fn new(duty: f64) -> Result<PulseTrainWave, ErrorsJSL> {
        Ok(PulseTrainWave { threshold: threshold_from_duty(duty)? })
    }
}

impl Waveform for PulseTrainWave {
    fn sample(&mut self, phase: Phase) -> f64 {
        if (phase.0 as u128) < self.threshold { 1.0 } else { 0.0 }
    }
}

impl NcoSource<PulseTrainWave> {
    /// Pulses at `frequency_hz` with the given duty cycle.
    pub fn pulse_train(frequency_hz: f64, sample_rate: f64, duty: f64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, PulseTrainWave::new(duty)?))
    }

    /// Pulses repeating every `period` samples, each `width` samples long (`width <= period`).
    pub fn pulse_train_with_period(period: u64, width: u64) -> Result<Self, ErrorsJSL> {
        let wave = PulseTrainWave { threshold: threshold_from_ratio(width, period)? };
        Ok(NcoSource::from_parts(Nco::from_period(period)?, wave))
    }
}

/// Convenience: `length` samples of a pulse train with an integer `period` (>= 2) and pulse
/// `width` in samples (0 to `period`).
pub fn pulse_train(period: usize, width: usize, length: usize) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(NcoSource::pulse_train_with_period(period as u64, width as u64)?.generate(length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_patterns() {
        assert_eq!(pulse_train(5, 2, 10).unwrap(), vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(pulse_train(4, 0, 4).unwrap(), vec![0.0; 4]);
        assert_eq!(pulse_train(4, 4, 4).unwrap(), vec![1.0; 4]);
        assert!(pulse_train(4, 5, 4).is_err());
        assert!(pulse_train(1, 1, 4).is_err());
    }

    #[test]
    fn widths_are_exact_for_every_period() {
        for period in 2..80usize {
            for width in 0..=period {
                let x = pulse_train(period, width, period * 12).unwrap();
                for (n, v) in x.iter().enumerate() {
                    assert_eq!(*v, if n % period < width { 1.0 } else { 0.0 }, "period {period} width {width} n {n}");
                }
            }
        }
    }

    #[test]
    fn frequency_form() {
        let mut p = NcoSource::pulse_train(10.0, 100.0, 0.3).unwrap();
        let x = p.generate(100);
        assert_eq!(x.iter().sum::<f64>(), 30.0);
        assert_eq!(&x[..10], &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(NcoSource::pulse_train(10.0, 100.0, -0.1).is_err());
    }

    #[test]
    fn amplitude_scaling_makes_a_clock_or_trigger() {
        let mut p = NcoSource::pulse_train_with_period(4, 1).unwrap().with_amplitude(3.3);
        assert_eq!(p.generate(5), vec![3.3, 0.0, 0.0, 0.0, 3.3]);
    }
}
