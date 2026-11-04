//! Square wave source.
//!
//! See the book entry [Signal sources: oscillators, CORDIC and waveforms](https://jwkunz.github.io/lib_jsl/book/comms/sources.html#waveforms-and-noise).

use super::nco::{threshold_from_duty, threshold_from_ratio, Nco, NcoSource, Phase, Waveform};
use lib_jsl_core::ErrorsJSL;

/// A bipolar rectangular wave: `+1` while the phase is within the duty fraction of the cycle
/// (starting at phase 0, like the positive half of a sine) and `-1` for the rest.
#[derive(Clone, Debug)]
pub struct SquareWave {
    threshold: u128,
}

impl SquareWave {
    /// A square wave with the given duty cycle in `[0, 1]` (0.5 for a symmetric wave).
    pub fn new(duty: f64) -> Result<SquareWave, ErrorsJSL> {
        Ok(SquareWave { threshold: threshold_from_duty(duty)? })
    }
}

impl Waveform for SquareWave {
    type Output = f64;

    fn sample(&mut self, phase: Phase) -> f64 {
        if (phase.0 as u128) < self.threshold { 1.0 } else { -1.0 }
    }
}

impl NcoSource<SquareWave> {
    /// A symmetric square wave at `frequency_hz`.
    pub fn square(frequency_hz: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, SquareWave::new(0.5)?))
    }

    /// A square wave at `frequency_hz` with an arbitrary duty cycle.
    pub fn square_with_duty(frequency_hz: f64, sample_rate: f64, duty: f64) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, SquareWave::new(duty)?))
    }

    /// A square wave with an integer `period` in samples and `high` samples of every period at `+1`.
    pub fn square_with_period(period: u64, high: u64) -> Result<Self, ErrorsJSL> {
        let wave = SquareWave { threshold: threshold_from_ratio(high, period)? };
        Ok(NcoSource::from_parts(Nco::from_period(period)?, wave))
    }
}

/// Convenience: `length` samples of a symmetric square wave with an integer `period` (>= 2).
/// For odd periods the positive half is the longer one.
pub fn square(period: usize, length: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let high = (period as u64).div_ceil(2);
    Ok(NcoSource::square_with_period(period as u64, high)?.generate(length))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_patterns() {
        assert_eq!(square(4, 8).unwrap(), vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]);
        assert_eq!(square(6, 12).unwrap(), [1.0, 1.0, 1.0, -1.0, -1.0, -1.0].repeat(2));
        assert_eq!(square(5, 10).unwrap(), [1.0, 1.0, 1.0, -1.0, -1.0].repeat(2));
        assert_eq!(square(2, 4).unwrap(), vec![1.0, -1.0, 1.0, -1.0]);
        assert!(square(1, 4).is_err());
    }

    #[test]
    fn every_period_and_high_count_is_exact_for_many_cycles() {
        for period in 2..60u64 {
            for high in 0..=period {
                let mut s = NcoSource::square_with_period(period, high).unwrap();
                for cycle in 0..20 {
                    for n in 0..period {
                        let want = if n < high { 1.0 } else { -1.0 };
                        assert_eq!(s.next_sample(), want, "period {period} high {high} cycle {cycle} n {n}");
                    }
                }
            }
        }
    }

    #[test]
    fn frequency_form_and_duty_cycle() {
        let mut s = NcoSource::square(1.0, 8.0).unwrap();
        assert_eq!(s.generate(8), vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        let mut d = NcoSource::square_with_duty(1.0, 10.0, 0.3).unwrap();
        let x = d.generate(10);
        assert_eq!(x.iter().filter(|&&v| v > 0.0).count(), 3);
        assert!(NcoSource::square_with_duty(1.0, 10.0, 1.2).is_err());
        assert!(NcoSource::square_with_period(4, 5).is_err());
    }

    #[test]
    fn mean_follows_the_duty_cycle() {
        let mut s = NcoSource::square_with_duty(7.0, 1000.0, 0.25).unwrap();
        let x = s.generate(10_000);
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        assert!((mean - (2.0 * 0.25 - 1.0)).abs() < 0.01, "mean {mean}");
    }
}
