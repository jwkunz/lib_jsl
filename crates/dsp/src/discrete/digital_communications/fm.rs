//! Frequency modulation (FM) and demodulation of complex baseband signals.
//!
//! In FM the message `m[n]` sets the *instantaneous frequency* of a constant-envelope carrier:
//! `f[n] = k_f · m[n]` Hz, so the baseband signal is `y[n] = e^{jφ[n]}` with
//! `φ[n] = 2π Σ_{i≤n} f[i] / fs`. The peak deviation is `k_f · max|m|`, and Carson's rule
//! estimates the occupied bandwidth as `2 (Δf + f_m)`.
//!
//! The modulator accumulates phase in the same fixed-point [`Nco`] used by the signal sources, so
//! the phase never drifts, however long the stream runs. The demodulator is the classic
//! quadrature (phase-difference) discriminator `arg(y[n] · conj(y[n-1]))`, which needs no carrier
//! recovery and recovers the message exactly in the absence of noise, as long as the deviation
//! stays below `fs / 2`. An optional single-pole de-emphasis filter models the broadcast
//! treble-boost compensation.
//!
//! See the book entry [Modulation: constellations, OFDM and FM](https://jwkunz.github.io/lib_jsl/book/comms/modulation.html#frequency-modulation).

use crate::discrete::{
    sources::{
        nco::{Nco, Waveform},
        quadrature::{Polarity, QuadratureWave},
        sinusoid::SineKernel,
    },
    stream_operator::{StreamOperator, StreamOperatorManagement},
};
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;
use std::f64::consts::TAU;

fn check_params(sample_rate: f64, sensitivity: f64) -> Result<(), ErrorsJSL> {
    if !(sample_rate.is_finite() && sample_rate > 0.0) || !(sensitivity.is_finite() && sensitivity > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Sample rate and frequency sensitivity must be positive and finite"));
    }
    Ok(())
}

/// Carson's rule for the bandwidth (Hz) containing nearly all of an FM signal's power.
pub fn carson_bandwidth(peak_deviation_hz: f64, max_message_frequency_hz: f64) -> f64 {
    2.0 * (peak_deviation_hz + max_message_frequency_hz)
}

/// A streaming FM modulator: real message samples in, unit-amplitude complex baseband out.
#[derive(Clone, Debug)]
pub struct FmModulator {
    nco: Nco,
    wave: QuadratureWave,
    sensitivity: f64,
}

impl FmModulator {
    /// `sensitivity` is `k_f` in Hz per message unit; `sample_rate` in Hz.
    pub fn new(sensitivity_hz_per_unit: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        check_params(sample_rate, sensitivity_hz_per_unit)?;
        Ok(FmModulator {
            nco: Nco::new(0.0, sample_rate)?,
            wave: QuadratureWave::new(SineKernel::Direct, Polarity::Positive),
            sensitivity: sensitivity_hz_per_unit,
        })
    }

    /// Modulates one message sample.
    pub fn modulate_sample(&mut self, m: f64) -> Complex<f64> {
        self.nco.next_phase_modulated(self.sensitivity * m);
        self.wave.sample(self.nco.phase())
    }
}

impl StreamOperatorManagement for FmModulator {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.nco.reset();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<f64, Complex<f64>> for FmModulator {
    fn process(&mut self, data_in: &[f64]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(data_in.iter().map(|&m| self.modulate_sample(m)).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        Ok(None)
    }
}

/// Modulates a whole message: `k_f` in Hz per message unit, `sample_rate` in Hz.
pub fn fm_modulate(message: &[f64], sensitivity_hz_per_unit: f64, sample_rate: f64) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    let mut m = FmModulator::new(sensitivity_hz_per_unit, sample_rate)?;
    Ok(message.iter().map(|&x| m.modulate_sample(x)).collect())
}

/// A streaming quadrature FM demodulator: complex baseband in, recovered message out.
#[derive(Clone, Debug)]
pub struct FmDemodulator {
    scale: f64,
    previous: Complex<f64>,
    deemphasis_alpha: Option<f64>,
    deemphasis_state: f64,
}

impl FmDemodulator {
    /// `sensitivity` is the modulator's `k_f` in Hz per message unit.
    pub fn new(sensitivity_hz_per_unit: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        check_params(sample_rate, sensitivity_hz_per_unit)?;
        Ok(FmDemodulator { scale: sample_rate / (TAU * sensitivity_hz_per_unit), previous: Complex::new(1.0, 0.0), deemphasis_alpha: None, deemphasis_state: 0.0 })
    }

    /// Adds a single-pole de-emphasis low-pass with time constant `tau` seconds (50 or 75 µs for
    /// broadcast FM). `sample_rate` must match the one given to [`new`](Self::new).
    pub fn with_deemphasis(mut self, tau_seconds: f64, sample_rate: f64) -> Result<Self, ErrorsJSL> {
        if !(tau_seconds > 0.0) || !(sample_rate > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("De-emphasis time constant and sample rate must be positive"));
        }
        self.deemphasis_alpha = Some(1.0 - (-1.0 / (sample_rate * tau_seconds)).exp());
        Ok(self)
    }

    /// Demodulates one baseband sample. The first output compares against a unit-phase reference.
    // ANCHOR: fm_discriminator
    pub fn demodulate_sample(&mut self, y: Complex<f64>) -> f64 {
        let d = (y * self.previous.conj()).arg();
        self.previous = y;
        let m = d * self.scale;
        match self.deemphasis_alpha {
            Some(a) => {
                self.deemphasis_state += a * (m - self.deemphasis_state);
                self.deemphasis_state
            }
            None => m,
        }
    }
    // ANCHOR_END: fm_discriminator
}

impl StreamOperatorManagement for FmDemodulator {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.previous = Complex::new(1.0, 0.0);
        self.deemphasis_state = 0.0;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<Complex<f64>, f64> for FmDemodulator {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(data_in.iter().map(|&y| self.demodulate_sample(y)).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        Ok(None)
    }
}

/// Demodulates a whole baseband record (no de-emphasis). `k_f` is the modulator's sensitivity.
pub fn fm_demodulate(baseband: &[Complex<f64>], sensitivity_hz_per_unit: f64, sample_rate: f64) -> Result<Vec<f64>, ErrorsJSL> {
    let mut d = FmDemodulator::new(sensitivity_hz_per_unit, sample_rate)?;
    Ok(baseband.iter().map(|&y| d.demodulate_sample(y)).collect())
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

    fn tone(n: usize, f: f64, fs: f64, amp: f64) -> Vec<f64> {
        (0..n).map(|i| amp * (TAU * f * i as f64 / fs).sin()).collect()
    }

    #[test]
    fn noise_free_round_trip_is_exact() {
        let (fs, kf) = (48_000.0, 5_000.0);
        let m: Vec<f64> = tone(4800, 1000.0, fs, 0.8).iter().zip(tone(4800, 2300.0, fs, 0.15)).map(|(a, b)| a + b).collect();
        let y = fm_modulate(&m, kf, fs).unwrap();
        let back = fm_demodulate(&y, kf, fs).unwrap();
        for (a, b) in m.iter().zip(&back) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn the_envelope_is_constant_and_the_frequency_follows_the_message() {
        let (fs, kf) = (8000.0, 1000.0);
        let y = fm_modulate(&vec![0.5; 200], kf, fs).unwrap();
        assert!(y.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
        // A constant message of 0.5 is a pure tone at 500 Hz: 36 degrees per sample.
        let step = (y[100] * y[99].conj()).arg();
        assert!((step - TAU * 500.0 / fs).abs() < 1e-12);
        // The phase after n samples is 2 pi 500 n / fs.
        let want = Complex::from_polar(1.0, TAU * 500.0 * 51.0 / fs);
        assert!((y[50] - want).norm() < 1e-9);
    }

    #[test]
    fn no_phase_drift_over_a_long_stream() {
        let (fs, kf) = (1000.0, 1.0);
        let mut m = FmModulator::new(kf, fs).unwrap();
        let mut last = Complex::new(0.0, 0.0);
        for _ in 0..2_000_000 {
            last = m.modulate_sample(1.0);
        }
        // 2e6 samples of a constant 1 Hz offset at 1 kHz: exactly 2000 cycles, back to phase 0.
        assert!((last - Complex::new(1.0, 0.0)).norm() < 1e-9, "{last}");
    }

    #[test]
    fn streaming_in_chunks_equals_one_shot_and_reset_restarts() {
        let (fs, kf) = (16_000.0, 3_000.0);
        let msg = tone(1000, 440.0, fs, 1.0);
        let whole = fm_modulate(&msg, kf, fs).unwrap();
        let mut m = FmModulator::new(kf, fs).unwrap();
        let mut chunked = Vec::new();
        for c in msg.chunks(77) {
            chunked.extend(m.process(c).unwrap().unwrap());
        }
        assert_eq!(whole, chunked);
        StreamOperatorManagement::reset(&mut m).unwrap();
        assert_eq!(m.process(&msg).unwrap().unwrap(), whole);
        let mut d = FmDemodulator::new(kf, fs).unwrap();
        let mut got = Vec::new();
        for c in whole.chunks(123) {
            got.extend(d.process(c).unwrap().unwrap());
        }
        assert!(got.iter().zip(&msg).all(|(a, b)| (a - b).abs() < 1e-9));
        assert!(m.process(&[]).unwrap().is_none() && d.process(&[]).unwrap().is_none());
    }

    #[test]
    fn demodulation_error_grows_with_noise_but_stays_small_at_high_cnr() {
        let (fs, kf) = (48_000.0, 5_000.0);
        let msg = tone(20_000, 700.0, fs, 1.0);
        let clean = fm_modulate(&msg, kf, fs).unwrap();
        let mut seed = 3;
        let mut rms_err = Vec::new();
        for sigma in [0.01, 0.05] {
            let noisy: Vec<Complex<f64>> = clean.iter().map(|&c| c + Complex::new(gauss(&mut seed), gauss(&mut seed)) * sigma).collect();
            let out = fm_demodulate(&noisy, kf, fs).unwrap();
            rms_err.push((out.iter().zip(&msg).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / msg.len() as f64).sqrt());
        }
        assert!(rms_err[0] < rms_err[1], "{rms_err:?}");
        assert!(rms_err[0] < 0.1, "{rms_err:?}");
    }

    #[test]
    fn deemphasis_attenuates_high_frequencies() {
        let (fs, kf) = (48_000.0, 5_000.0);
        let tau = 75e-6; // corner at 2.12 kHz
        let amp = |f: f64| {
            let y = fm_modulate(&tone(9600, f, fs, 1.0), kf, fs).unwrap();
            let mut d = FmDemodulator::new(kf, fs).unwrap().with_deemphasis(tau, fs).unwrap();
            let out = d.process(&y).unwrap().unwrap();
            out[4800..].iter().fold(0.0f64, |a, v| a.max(v.abs()))
        };
        let (low, high) = (amp(200.0), amp(8000.0));
        assert!((low - 1.0).abs() < 0.03, "low-frequency gain {low}");
        let expect = 1.0 / (1.0 + (TAU * 8000.0 * tau).powi(2)).sqrt();
        assert!((high - expect).abs() < 0.03, "8 kHz gain {high}, single-pole prediction {expect}");
    }

    #[test]
    fn carson_rule_and_parameter_validation() {
        assert_eq!(carson_bandwidth(75_000.0, 15_000.0), 180_000.0);
        assert!(FmModulator::new(0.0, 8000.0).is_err());
        assert!(FmModulator::new(100.0, -1.0).is_err());
        assert!(FmDemodulator::new(100.0, 0.0).is_err());
        assert!(FmDemodulator::new(100.0, 8000.0).unwrap().with_deemphasis(0.0, 8000.0).is_err());
        assert!(fm_modulate(&[0.1], f64::NAN, 8000.0).is_err());
    }
}
