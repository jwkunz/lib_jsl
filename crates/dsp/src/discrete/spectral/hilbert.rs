//! The Hilbert transform and the analytic signal.
//!
//! See the book entry [The Hilbert transform and the analytic signal](https://jwkunz.github.io/lib_jsl/book/spectral/hilbert.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_ffts::{Fft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use num_complex::Complex;
use std::f64::consts::{PI, TAU};

/// The analytic signal `x + i H{x}` of a real signal, by the FFT: negative-frequency bins are
/// zeroed and positive ones doubled (DC, and Nyquist for even lengths, kept), as in SciPy's
/// `hilbert`. The transform treats `x` as periodic, so edges show some ringing for non-periodic
/// data.
pub fn analytic_signal(x: &[f64]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    let n = x.len();
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("The signal must not be empty."));
    }
    // ANCHOR: analytic_signal
    let input: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut spectrum = Fft::with_plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard)?.execute(&input)?;
    let half = n.div_ceil(2);
    for (k, s) in spectrum.iter_mut().enumerate() {
        if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
            continue;
        }
        *s *= if k < half { 2.0 } else { 0.0 };
    }
    Fft::with_plan(n, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard)?.execute(&spectrum)
    // ANCHOR_END: analytic_signal
}

/// The Hilbert transform `H{x}` (the imaginary part of the analytic signal): `cos → sin`,
/// `sin → −cos`.
pub fn hilbert_transform(x: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(analytic_signal(x)?.iter().map(|z| z.im).collect())
}

/// Instantaneous amplitude, unwrapped phase and frequency of a signal.
#[derive(Debug, Clone)]
pub struct InstantaneousAttributes {
    /// The envelope `|x + iH{x}|`.
    pub amplitude: Vec<f64>,
    /// Unwrapped phase in radians.
    pub phase: Vec<f64>,
    /// Frequency in Hz from phase differences (one shorter than the signal).
    pub frequency: Vec<f64>,
}

/// The instantaneous attributes of `x` sampled at `fs`, from its analytic signal.
pub fn instantaneous_attributes(x: &[f64], fs: f64) -> Result<InstantaneousAttributes, ErrorsJSL> {
    let z = analytic_signal(x)?;
    let amplitude = z.iter().map(|v| v.norm()).collect();
    let mut phase: Vec<f64> = z.iter().map(|v| v.arg()).collect();
    for i in 1..phase.len() {
        let jump = ((phase[i] - phase[i - 1] + PI).rem_euclid(TAU)) - PI;
        phase[i] = phase[i - 1] + jump;
    }
    let frequency = phase.windows(2).map(|w| (w[1] - w[0]) * fs / TAU).collect();
    Ok(InstantaneousAttributes { amplitude, phase, frequency })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_cosines_to_sines_for_even_and_odd_lengths() {
        for n in [64, 63, 100] {
            let k = 5.0;
            let x: Vec<f64> = (0..n).map(|i| (TAU * k * i as f64 / n as f64).cos()).collect();
            let h = hilbert_transform(&x).unwrap();
            assert!(h.iter().enumerate().all(|(i, v)| (v - (TAU * k * i as f64 / n as f64).sin()).abs() < 1e-12), "n={n}");
            // The real part of the analytic signal is the signal itself.
            assert!(analytic_signal(&x).unwrap().iter().zip(&x).all(|(z, v)| (z.re - v).abs() < 1e-12));
        }
        assert!(analytic_signal(&[]).is_err());
    }

    #[test]
    fn recovers_the_envelope_and_frequency_of_an_am_signal() {
        let (fs, n) = (1000.0, 1000);
        let carrier = 100.0;
        let envelope = |t: f64| 1.0 + 0.5 * (TAU * 3.0 * t).cos();
        let x: Vec<f64> = (0..n).map(|i| { let t = i as f64 / fs; envelope(t) * (TAU * carrier * t).cos() }).collect();
        let InstantaneousAttributes { amplitude, frequency, .. } = instantaneous_attributes(&x, fs).unwrap();
        for i in (100..900).step_by(50) {
            assert!((amplitude[i] - envelope(i as f64 / fs)).abs() < 1e-9, "i={i}");
            assert!((frequency[i] - carrier).abs() < 1e-6, "i={i}: {}", frequency[i]);
        }
    }
}
