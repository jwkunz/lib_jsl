//! A numerically controlled oscillator (NCO) and the machinery shared by every signal source.
//!
//! The oscillator is a fixed-point phase accumulator: the phase is a `u64` where `2^64` is one full
//! cycle, advanced each sample by a frequency tuning word. Wrap-around is free, the phase never
//! drifts, and frequency resolution is `sample_rate / 2^64`.
//!
//! A source is an [`NcoSource`], which pairs an [`Nco`] with a [`Waveform`] that maps phase to an
//! amplitude, plus an output amplitude and offset. Sine, square, triangle, ramp, pulse-train and
//! noise sources are all just different waveforms driven by the same oscillator, and they all
//! implement [`StreamOperator`] through the one implementation here.
//!
//! All periodic waveforms share the sine's phase reference: at phase 0 they are at their
//! rising zero crossing (or, for the 0/1 pulse train, at the start of the pulse).
//!
//! # Streaming
//! A source implements `StreamOperator<(), Output>` (free-running: the length of the input slice is
//! the number of samples to produce) and `StreamOperator<f64, Output>` (frequency modulation: each
//! input value is an instantaneous frequency offset in Hz added to the oscillator frequency), where
//! `Output` is `f64` for real sources and `Complex<f64>` for the quadrature source.

use crate::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};
use lib_jsl_core::ErrorsJSL;
use num::Complex;
use std::fmt::Debug;

/// `2^64` as a float: one full cycle of the phase accumulator.
const CYCLE: f64 = 18_446_744_073_709_551_616.0;

/// A point in the oscillator's cycle, as a fraction of a turn in `[0, 1)` scaled by `2^64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Phase(pub u64);

impl Phase {
    /// Builds a phase from a (possibly negative or greater than one) number of turns.
    pub fn from_turns(turns: f64) -> Phase {
        Phase(tuning_word(turns))
    }

    /// The phase as a fraction of a turn in `[0, 1)`.
    pub fn turns(self) -> f64 {
        self.0 as f64 / CYCLE
    }

    /// The phase in radians in `[0, 2π)`.
    pub fn radians(self) -> f64 {
        self.turns() * std::f64::consts::TAU
    }
}

/// Phase threshold (in `2^-64` turns, up to and including `2^64`) for a duty cycle in `[0, 1]`:
/// a waveform is "high" while its phase is below the threshold.
pub(crate) fn threshold_from_duty(duty: f64) -> Result<u128, ErrorsJSL> {
    if !(0.0..=1.0).contains(&duty) {
        return Err(ErrorsJSL::InvalidInputRange("Duty cycle must be between 0 and 1."));
    }
    Ok((duty * CYCLE).round() as u128)
}

/// Exact threshold for a duty cycle of `numerator / denominator`, rounded up so that on an
/// [`Nco::from_period`] oscillator exactly `numerator` of every `denominator` samples are high.
pub(crate) fn threshold_from_ratio(numerator: u64, denominator: u64) -> Result<u128, ErrorsJSL> {
    if denominator == 0 || numerator > denominator {
        return Err(ErrorsJSL::InvalidInputRange("Duty ratio must satisfy 0 <= numerator <= denominator, denominator > 0."));
    }
    Ok(((numerator as u128) << 64).div_ceil(denominator as u128))
}

/// Converts a number of turns (modulo 1) into a fixed-point tuning word.
fn tuning_word(turns: f64) -> u64 {
    let frac = turns.rem_euclid(1.0);
    ((frac * CYCLE).round() as u128) as u64 // 2^64 wraps to 0
}

/// A numerically controlled oscillator.
#[derive(Clone, Debug)]
pub struct Nco {
    phase: u64,
    increment: u64,
    initial_phase: u64,
    sample_rate: f64,
}

impl Nco {
    /// An oscillator at `frequency_hz` for the given sampling rate. The frequency may be negative
    /// (the phase runs backwards) and must not exceed Nyquist, `sample_rate / 2`, in magnitude.
    pub fn new(frequency_hz: f64, sample_rate: f64) -> Result<Nco, ErrorsJSL> {
        if !(sample_rate.is_finite() && sample_rate > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("NCO sample rate must be positive and finite."));
        }
        let mut nco = Nco { phase: 0, increment: 0, initial_phase: 0, sample_rate };
        nco.set_frequency(frequency_hz)?;
        Ok(nco)
    }

    /// An oscillator that completes exactly one cycle every `period` samples (`period >= 2`).
    ///
    /// The tuning word is rounded *up*, so samples that fall exactly on a waveform edge (for
    /// example the 4th sample of a 6-sample square wave) always land on the late side of it.
    pub fn from_period(period: u64) -> Result<Nco, ErrorsJSL> {
        if period < 2 {
            return Err(ErrorsJSL::InvalidInputRange("An NCO period must be at least 2 samples."));
        }
        let increment = ((1u128 << 64) + period as u128 - 1) / period as u128;
        Ok(Nco { phase: 0, increment: increment as u64, initial_phase: 0, sample_rate: 1.0 })
    }

    /// Sets the starting phase (and the phase that [`reset`](Self::reset) returns to), in turns.
    pub fn with_phase_turns(mut self, turns: f64) -> Nco {
        self.phase = tuning_word(turns);
        self.initial_phase = self.phase;
        self
    }

    /// Changes the frequency without disturbing the phase.
    pub fn set_frequency(&mut self, frequency_hz: f64) -> Result<(), ErrorsJSL> {
        if !frequency_hz.is_finite() || frequency_hz.abs() > 0.5 * self.sample_rate {
            return Err(ErrorsJSL::InvalidInputRange("NCO frequency must be finite and within Nyquist."));
        }
        self.increment = tuning_word(frequency_hz / self.sample_rate);
        Ok(())
    }

    /// The frequency actually realised by the tuning word, in Hz (negative if running backwards).
    pub fn frequency(&self) -> f64 {
        (self.increment as i64) as f64 / CYCLE * self.sample_rate
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// The phase that the next call to [`next_phase`](Self::next_phase) will return.
    pub fn phase(&self) -> Phase {
        Phase(self.phase)
    }

    /// Jumps to the given phase (in turns).
    pub fn set_phase_turns(&mut self, turns: f64) {
        self.phase = tuning_word(turns);
    }

    /// Returns the current phase and then advances by one sample.
    pub fn next_phase(&mut self) -> Phase {
        let current = Phase(self.phase);
        self.phase = self.phase.wrapping_add(self.increment);
        current
    }

    /// Like [`next_phase`](Self::next_phase), but this step advances at the oscillator frequency
    /// plus `offset_hz` (frequency modulation). The configured frequency is unchanged.
    pub fn next_phase_modulated(&mut self, offset_hz: f64) -> Phase {
        let current = Phase(self.phase);
        let delta = tuning_word(offset_hz / self.sample_rate);
        self.phase = self.phase.wrapping_add(self.increment).wrapping_add(delta);
        current
    }

    /// Returns to the starting phase.
    pub fn reset(&mut self) {
        self.phase = self.initial_phase;
    }
}

/// A type a source can emit: real (`f64`) or complex (`Complex<f64>`) samples.
pub trait Sample: Copy + Debug {
    /// The additive identity, used as the default offset.
    fn zero() -> Self;
    /// `amplitude * self + offset`.
    fn scaled(self, amplitude: f64, offset: Self) -> Self;
}

impl Sample for f64 {
    fn zero() -> f64 {
        0.0
    }
    fn scaled(self, amplitude: f64, offset: f64) -> f64 {
        amplitude * self + offset
    }
}

impl Sample for Complex<f64> {
    fn zero() -> Complex<f64> {
        Complex::new(0.0, 0.0)
    }
    fn scaled(self, amplitude: f64, offset: Complex<f64>) -> Complex<f64> {
        self * amplitude + offset
    }
}

/// Maps oscillator phase to an output value.
pub trait Waveform {
    /// The type of sample produced: `f64` for real waveforms, `Complex<f64>` for quadrature.
    type Output: Sample;

    /// The waveform value at `phase`. Called once per output sample, in order, so stateful
    /// waveforms (such as sample-and-hold noise) may keep history.
    fn sample(&mut self, phase: Phase) -> Self::Output;

    /// Returns any internal state to its starting condition.
    fn reset(&mut self) {}
}

/// A signal source: an [`Nco`] driving a [`Waveform`], scaled by `amplitude` and shifted by `offset`.
#[derive(Clone, Debug)]
pub struct NcoSource<W: Waveform> {
    nco: Nco,
    waveform: W,
    amplitude: f64,
    offset: W::Output,
}

impl<W: Waveform> NcoSource<W> {
    /// Builds a unit-amplitude, zero-offset source from an oscillator and a waveform.
    pub fn from_parts(nco: Nco, waveform: W) -> Self {
        NcoSource { nco, waveform, amplitude: 1.0, offset: W::Output::zero() }
    }

    /// Sets the output amplitude (a multiplier on the waveform).
    pub fn with_amplitude(mut self, amplitude: f64) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Sets a constant added to every output sample (a complex constant for complex sources).
    pub fn with_offset(mut self, offset: W::Output) -> Self {
        self.offset = offset;
        self
    }

    /// Sets the starting phase in turns (a quarter turn turns a sine into a cosine).
    pub fn with_phase_turns(mut self, turns: f64) -> Self {
        self.nco = self.nco.with_phase_turns(turns);
        self
    }

    /// Access to the oscillator, e.g. to retune it.
    pub fn nco(&self) -> &Nco {
        &self.nco
    }

    pub fn nco_mut(&mut self) -> &mut Nco {
        &mut self.nco
    }

    /// Access to the waveform.
    pub fn waveform(&self) -> &W {
        &self.waveform
    }

    pub fn waveform_mut(&mut self) -> &mut W {
        &mut self.waveform
    }

    fn shape(&self, w: W::Output) -> W::Output {
        w.scaled(self.amplitude, self.offset)
    }

    /// Produces the next output sample.
    pub fn next_sample(&mut self) -> W::Output {
        let phase = self.nco.next_phase();
        let w = self.waveform.sample(phase);
        self.shape(w)
    }

    /// Produces the next output sample with the frequency offset by `offset_hz` for this step.
    pub fn next_sample_modulated(&mut self, offset_hz: f64) -> W::Output {
        let phase = self.nco.next_phase_modulated(offset_hz);
        let w = self.waveform.sample(phase);
        self.shape(w)
    }

    /// Produces the next `count` samples.
    pub fn generate(&mut self, count: usize) -> Vec<W::Output> {
        (0..count).map(|_| self.next_sample()).collect()
    }
}

impl<W: Waveform> StreamOperatorManagement for NcoSource<W> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.nco.reset();
        self.waveform.reset();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

/// Free-running operation: the number of input items is the number of samples produced.
impl<W: Waveform> StreamOperator<(), W::Output> for NcoSource<W> {
    fn process(&mut self, data_in: &[()]) -> Result<Option<Vec<W::Output>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.generate(data_in.len())))
    }

    fn flush(&mut self) -> Result<Option<Vec<W::Output>>, ErrorsJSL> {
        Ok(None)
    }
}

/// Frequency-modulated operation: each input is a frequency offset in Hz for that sample.
impl<W: Waveform> StreamOperator<f64, W::Output> for NcoSource<W> {
    fn process(&mut self, data_in: &[f64]) -> Result<Option<Vec<W::Output>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(data_in.iter().map(|&df| self.next_sample_modulated(df)).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<W::Output>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Identity;
    impl Waveform for Identity {
        type Output = f64;
        fn sample(&mut self, phase: Phase) -> f64 {
            phase.turns()
        }
    }

    #[test]
    fn frequency_round_trip_and_phase_ramp() {
        let nco = Nco::new(1000.0, 8000.0).unwrap();
        assert!((nco.frequency() - 1000.0).abs() < 1e-9);
        let mut src = NcoSource::from_parts(nco, Identity);
        let out = src.generate(10);
        for (n, v) in out.iter().enumerate() {
            assert!((v - (n as f64 / 8.0) % 1.0).abs() < 1e-12, "n = {n}");
        }
        // Negative frequency runs the phase backwards.
        let mut back = NcoSource::from_parts(Nco::new(-1000.0, 8000.0).unwrap(), Identity);
        assert!((back.nco().frequency() + 1000.0).abs() < 1e-9);
        let out = back.generate(3);
        assert!((out[1] - 0.875).abs() < 1e-12 && (out[2] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn frequency_limits() {
        assert!(Nco::new(5000.0, 8000.0).is_err());
        assert!(Nco::new(f64::NAN, 8000.0).is_err());
        assert!(Nco::new(100.0, 0.0).is_err());
        assert!(Nco::new(4000.0, 8000.0).is_ok());
        let mut nco = Nco::new(10.0, 100.0).unwrap();
        assert!(nco.set_frequency(60.0).is_err());
        assert!((nco.frequency() - 10.0).abs() < 1e-9, "failed retune leaves the frequency alone");
    }

    #[test]
    fn exact_integer_periods() {
        for period in [2u64, 3, 5, 6, 7, 10, 97, 1000, 4096] {
            let mut nco = Nco::from_period(period).unwrap();
            let cycles = 50;
            for n in 0..period * cycles {
                let p = nco.next_phase();
                if n % period == 0 {
                    // Never negative-of-zero; the accumulated rounding is far below one sample.
                    assert!(p.0 < period * cycles, "period {period}, n {n}: phase {}", p.0);
                }
            }
        }
        assert!(Nco::from_period(1).is_err());
        assert!(Nco::from_period(0).is_err());
    }

    #[test]
    fn reset_and_phase_control() {
        let mut src = NcoSource::from_parts(Nco::new(1.0, 8.0).unwrap(), Identity).with_phase_turns(0.25);
        let first = src.generate(4);
        assert!((first[0] - 0.25).abs() < 1e-12);
        StreamOperatorManagement::reset(&mut src).unwrap();
        assert_eq!(src.generate(4), first, "reset returns to the configured starting phase");
        src.nco_mut().set_phase_turns(0.5);
        assert!((src.next_sample() - 0.5).abs() < 1e-12);
        assert!((Phase::from_turns(-0.25).turns() - 0.75).abs() < 1e-12);
        assert!((Phase::from_turns(1.5).turns() - 0.5).abs() < 1e-12);
        assert!((Phase(1 << 62).radians() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn amplitude_and_offset() {
        let mut src = NcoSource::from_parts(Nco::new(1.0, 4.0).unwrap(), Identity).with_amplitude(4.0).with_offset(10.0);
        let out = src.generate(4);
        assert_eq!(out.iter().map(|v| v.round()).collect::<Vec<_>>(), vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn stream_operator_contract() {
        let make = || NcoSource::from_parts(Nco::new(0.3, 8.0).unwrap(), Identity);
        let mut a = make();
        let whole = StreamOperator::<(), f64>::process(&mut a, &[(); 25]).unwrap().unwrap();
        let mut b = make();
        let mut chunked = Vec::new();
        for n in [1usize, 7, 3, 14] {
            chunked.extend(StreamOperator::<(), f64>::process(&mut b, &vec![(); n]).unwrap().unwrap());
        }
        assert_eq!(whole, chunked, "chunked processing is identical to one call");
        assert!(StreamOperator::<(), f64>::process(&mut b, &[]).unwrap().is_none());
        assert!(StreamOperator::<(), f64>::flush(&mut b).unwrap().is_none());
        StreamOperatorManagement::reset(&mut b).unwrap();
        assert_eq!(StreamOperator::<(), f64>::process(&mut b, &[(); 25]).unwrap().unwrap(), whole);
        StreamOperatorManagement::finalize(&mut b).unwrap();
    }

    #[test]
    fn frequency_modulation_input() {
        // A constant offset of +100 Hz equals running at the shifted frequency.
        let mut fm = NcoSource::from_parts(Nco::new(300.0, 8000.0).unwrap(), Identity);
        let mut direct = NcoSource::from_parts(Nco::new(400.0, 8000.0).unwrap(), Identity);
        let modulated = StreamOperator::<f64, f64>::process(&mut fm, &[100.0; 40]).unwrap().unwrap();
        let reference = direct.generate(40);
        for (m, r) in modulated.iter().zip(&reference) {
            // Phases are cyclic, so compare on the circle (rounding can straddle a wrap point).
            let d = (m - r).abs();
            assert!(d.min(1.0 - d) < 1e-9, "{m} vs {r}");
        }
        assert!((fm.nco().frequency() - 300.0).abs() < 1e-9, "modulation does not retune the NCO");
        assert!(StreamOperator::<f64, f64>::process(&mut fm, &[]).unwrap().is_none());
    }
}
