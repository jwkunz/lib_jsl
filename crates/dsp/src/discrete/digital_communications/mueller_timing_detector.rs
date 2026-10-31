//! Mueller & Müller timing error detector and symbol timing recovery.
//!
//! A receiver samples at a rate that is a multiple of the symbol rate, but with an unknown timing
//! phase and a slightly wrong clock. The Mueller & Müller detector measures the timing error from
//! just *one sample per symbol* and the receiver's own decisions:
//!
//! `e[n] = Re{ conj(d[n-1]) · y[n] − conj(d[n]) · y[n-1] }`
//!
//! where `y[n]` is the interpolated sample at symbol `n` and `d[n]` the symbol nearest to it. For
//! a Nyquist pulse the expected value of `e` is proportional to the timing error and zero at the
//! ideal instant, so it can drive a feedback loop. [`mueller_timing_detector`] is that detector;
//! [`MuellerTimingRecovery`] wraps it into a complete second-order loop:
//!
//! - a [`FarrowFilter`] interpolates the oversampled input at the current (fractional) instant;
//! - the error nudges the symbol-time estimate `μ` directly (proportional path) and the symbol
//!   period estimate `ω` (integral path), so the loop tracks both a timing offset and a
//!   clock-frequency error;
//! - `ω` is limited to a small band around its nominal value so noise cannot drag it away.
//!
//! The loop consumes the oversampled stream and emits one sample per symbol.

use super::{constellations::Constellation, farrow_filter::FarrowFilter};
use crate::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};
use lib_jsl_core::ErrorsJSL;
use num::Complex;

/// The Mueller & Müller timing error for consecutive symbol-rate samples `y_prev`, `y_curr` with
/// decisions `d_prev`, `d_curr`. The sign convention is that of the loop in this module: a positive
/// error moves the sampling instant later (it is added to the symbol-time estimate), a negative
/// error moves it earlier, and zero means the instants are ideal.
pub fn mueller_timing_detector(y_prev: Complex<f64>, y_curr: Complex<f64>, d_prev: Complex<f64>, d_curr: Complex<f64>) -> f64 {
    (d_prev.conj() * y_curr - d_curr.conj() * y_prev).re
}

/// Symbol timing recovery with a Mueller & Müller detector.
#[derive(Clone, Debug)]
pub struct MuellerTimingRecovery {
    constellation: Constellation,
    interpolator: FarrowFilter<Complex<f64>>,
    nominal_omega: f64,
    omega: f64,
    omega_limit: f64,
    gain_mu: f64,
    gain_omega: f64,
    /// Input history; `pos` indexes into it (integer part + fractional position).
    buffer: Vec<Complex<f64>>,
    pos: f64,
    prev_sample: Complex<f64>,
    prev_decision: Complex<f64>,
    last_error: f64,
}

impl MuellerTimingRecovery {
    /// A loop for a signal with `samples_per_symbol` (>= 2, may be fractional) samples per symbol
    /// and the given constellation for decisions, with the usual gains: `gain_mu = 0.175`,
    /// `gain_omega = 0.25 · gain_mu²` and `ω` limited to ±0.5% of nominal.
    pub fn new(samples_per_symbol: f64, constellation: Constellation) -> Result<Self, ErrorsJSL> {
        let gain_mu = 0.175;
        Self::with_gains(samples_per_symbol, constellation, gain_mu, 0.25 * gain_mu * gain_mu, 0.005)
    }

    /// Like [`new`](Self::new) with explicit loop gains and the relative limit on `ω`.
    pub fn with_gains(samples_per_symbol: f64, constellation: Constellation, gain_mu: f64, gain_omega: f64, omega_relative_limit: f64) -> Result<Self, ErrorsJSL> {
        if !(samples_per_symbol >= 2.0) || !samples_per_symbol.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("Samples per symbol must be at least 2"));
        }
        if !(gain_mu > 0.0) || !(gain_omega >= 0.0) || !(omega_relative_limit >= 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Loop gains must be non-negative (gain_mu positive)"));
        }
        let interpolator = FarrowFilter::new(3)?;
        let mut s = MuellerTimingRecovery {
            constellation,
            interpolator,
            nominal_omega: samples_per_symbol,
            omega: samples_per_symbol,
            omega_limit: omega_relative_limit * samples_per_symbol,
            gain_mu,
            gain_omega,
            buffer: Vec::new(),
            pos: 0.0,
            prev_sample: Complex::new(0.0, 0.0),
            prev_decision: Complex::new(0.0, 0.0),
            last_error: 0.0,
        };
        s.restart();
        Ok(s)
    }

    fn restart(&mut self) {
        self.omega = self.nominal_omega;
        self.buffer.clear();
        // Start with the first interpolation window centred on the first samples.
        self.pos = 1.0;
        self.prev_sample = Complex::new(0.0, 0.0);
        self.prev_decision = Complex::new(0.0, 0.0);
        self.last_error = 0.0;
    }

    /// The current estimate of the symbol period in samples.
    pub fn symbol_period(&self) -> f64 {
        self.omega
    }

    /// The most recent timing error from the detector.
    pub fn last_error(&self) -> f64 {
        self.last_error
    }
}

impl StreamOperatorManagement for MuellerTimingRecovery {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.restart();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<Complex<f64>, Complex<f64>> for MuellerTimingRecovery {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        self.buffer.extend_from_slice(data_in);
        let mut out = Vec::new();
        // The cubic window needs samples floor(pos)-1 ..= floor(pos)+2.
        while (self.pos.floor() as usize) + 2 < self.buffer.len() {
            let i = self.pos.floor() as usize;
            let mu = self.pos - i as f64;
            let y = self.interpolator.interpolate_window(&self.buffer[i - 1..=i + 2], mu)?;
            let d = self.constellation.slice(y);
            let err = mueller_timing_detector(self.prev_sample, y, self.prev_decision, d).clamp(-1.0, 1.0);
            self.omega = (self.omega + self.gain_omega * err).clamp(self.nominal_omega - self.omega_limit, self.nominal_omega + self.omega_limit);
            self.pos += self.omega + self.gain_mu * err;
            self.prev_sample = y;
            self.prev_decision = d;
            self.last_error = err;
            out.push(y);
        }
        // Drop consumed history, keeping the samples the next window will need.
        let keep_from = (self.pos.floor() as usize).saturating_sub(1).min(self.buffer.len());
        if keep_from > 0 {
            self.buffer.drain(..keep_from);
            self.pos -= keep_from as f64;
        }
        Ok(if out.is_empty() { None } else { Some(out) })
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn gauss(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>()
    }

    /// Raised-cosine pulse (roll-off `beta`, symbol period 1), a Nyquist pulse.
    fn raised_cosine(t: f64, beta: f64) -> f64 {
        let sinc = if t.abs() < 1e-12 { 1.0 } else { (PI * t).sin() / (PI * t) };
        let denom = 1.0 - (2.0 * beta * t).powi(2);
        if denom.abs() < 1e-9 {
            return PI / 4.0 * if beta > 0.0 { (PI / (2.0 * beta)).sin() / (PI / (2.0 * beta)) } else { 1.0 };
        }
        sinc * (PI * beta * t).cos() / denom
    }

    /// Samples the pulse-shaped signal at times `t_k = (k / sps)(1 + clock_error) + offset` symbol
    /// periods, evaluating the pulse analytically (no interpolation error).
    fn received(symbols: &[Complex<f64>], sps: f64, offset: f64, clock_error: f64, num_samples: usize) -> Vec<Complex<f64>> {
        (0..num_samples)
            .map(|k| {
                let t = k as f64 / sps * (1.0 + clock_error) + offset;
                let centre = t.round() as isize;
                (centre - 10..=centre + 10)
                    .filter(|&n| n >= 0 && (n as usize) < symbols.len())
                    .map(|n| symbols[n as usize] * raised_cosine(t - n as f64, 0.35))
                    .sum()
            })
            .collect()
    }

    fn random_symbols(c: &Constellation, n: usize, seed: &mut u64) -> Vec<Complex<f64>> {
        (0..n).map(|_| c.points()[((lcg(seed) + 0.5) * c.size() as f64) as usize % c.size()]).collect()
    }

    /// Symbol errors between recovered samples and the sent symbols, trying every alignment
    /// (recovered sample `i` corresponds to sent symbol `i + shift`, with a possible phase flip
    /// ambiguity ruled out because timing recovery does not rotate the signal).
    fn best_symbol_errors(c: &Constellation, recovered: &[Complex<f64>], sent: &[Complex<f64>], skip: usize) -> usize {
        (0..8usize)
            .map(|shift| {
                recovered
                    .iter()
                    .enumerate()
                    .skip(skip)
                    .filter(|(i, _)| i + shift < sent.len())
                    .filter(|(i, y)| c.slice(**y) != sent[i + shift])
                    .count()
            })
            .min()
            .unwrap()
    }

    #[test]
    fn detector_vanishes_at_the_ideal_sampling_instants() {
        let (a, b) = (Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
        assert_eq!(mueller_timing_detector(a, b, a, b), 0.0);
        let q = Complex::new(0.7, -0.7);
        assert!(mueller_timing_detector(q, q, q, q).abs() < 1e-15);
        // A sample that has leaked a bit of its neighbour produces a nonzero error.
        assert!(mueller_timing_detector(a, b * 0.9 + a * 0.1, a, b).abs() > 0.05);
    }

    #[test]
    fn recovers_symbols_despite_a_fractional_timing_offset() {
        let c = Constellation::qpsk();
        let mut seed = 4;
        let sent = random_symbols(&c, 3000, &mut seed);
        for offset in [0.0, 0.13, 0.37, 0.5, 0.81] {
            let rx = received(&sent, 4.0, offset, 0.0, 4 * 2900);
            let mut rec = MuellerTimingRecovery::new(4.0, c.clone()).unwrap();
            let out = rec.process(&rx).unwrap().unwrap();
            assert!(out.len() > 2800 && out.len() < 2950, "got {} symbols", out.len());
            assert_eq!(best_symbol_errors(&c, &out, &sent, 400), 0, "offset {offset}");
            assert!((rec.symbol_period() - 4.0).abs() < 0.01, "omega {}", rec.symbol_period());
        }
    }

    #[test]
    fn tracks_a_clock_frequency_error() {
        let c = Constellation::qpsk();
        let mut seed = 6;
        let sent = random_symbols(&c, 6000, &mut seed);
        for ppm in [-400.0, 250.0] {
            let eps = ppm * 1e-6;
            let rx = received(&sent, 4.0, 0.3, eps, 4 * 5800);
            let mut rec = MuellerTimingRecovery::new(4.0, c.clone()).unwrap();
            let out = rec.process(&rx).unwrap().unwrap();
            assert_eq!(best_symbol_errors(&c, &out, &sent, 800), 0, "{ppm} ppm");
            // The recovered symbol period follows the true one, 4/(1+eps).
            let true_period = 4.0 / (1.0 + eps);
            assert!((rec.symbol_period() - true_period).abs() < 0.002, "{ppm} ppm: omega {} vs {true_period}", rec.symbol_period());
        }
    }

    #[test]
    fn works_for_bpsk_and_with_noise_and_in_chunks() {
        let c = Constellation::bpsk();
        let mut seed = 8;
        let sent = random_symbols(&c, 4000, &mut seed);
        let mut rx = received(&sent, 8.0, 0.6, 100e-6, 8 * 3900);
        for v in rx.iter_mut() {
            *v += Complex::new(gauss(&mut seed), gauss(&mut seed)) * 0.05;
        }
        let mut rec = MuellerTimingRecovery::new(8.0, c.clone()).unwrap();
        let mut out = Vec::new();
        for chunk in rx.chunks(1000) {
            if let Some(o) = rec.process(chunk).unwrap() {
                out.extend(o);
            }
        }
        assert!(out.len() > 3800);
        assert_eq!(best_symbol_errors(&c, &out, &sent, 500), 0);
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let c = Constellation::qpsk();
        let mut seed = 10;
        let sent = random_symbols(&c, 500, &mut seed);
        let rx = received(&sent, 4.0, 0.2, 0.0, 4 * 480);
        let whole = MuellerTimingRecovery::new(4.0, c.clone()).unwrap().process(&rx).unwrap().unwrap();
        let mut rec = MuellerTimingRecovery::new(4.0, c).unwrap();
        let mut parts = Vec::new();
        for chunk in rx.chunks(37) {
            if let Some(o) = rec.process(chunk).unwrap() {
                parts.extend(o);
            }
        }
        assert_eq!(whole.len(), parts.len());
        assert!(whole.iter().zip(&parts).all(|(a, b)| (a - b).norm() < 1e-9));
        StreamOperatorManagement::reset(&mut rec).unwrap();
        assert_eq!(rec.symbol_period(), 4.0);
        assert!(rec.process(&[]).unwrap().is_none() && rec.flush().unwrap().is_none());
    }

    #[test]
    fn invalid_parameters() {
        let c = Constellation::bpsk();
        assert!(MuellerTimingRecovery::new(1.5, c.clone()).is_err());
        assert!(MuellerTimingRecovery::new(f64::NAN, c.clone()).is_err());
        assert!(MuellerTimingRecovery::with_gains(4.0, c.clone(), 0.0, 0.01, 0.01).is_err());
        assert!(MuellerTimingRecovery::with_gains(4.0, c, 0.1, -1.0, 0.01).is_err());
    }
}
