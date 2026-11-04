//! Power spectral density estimation by Welch's method of averaged, windowed periodograms.
//!
//! See the book entry [Power spectra: periodogram, Welch, spectrogram and Lomb-Scargle](https://jwkunz.github.io/lib_jsl/book/spectral/power-spectra.html#variance-welch-averaging).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_ffts::RealFft;

/// How to scale a power spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralScaling {
    /// Power spectral density, units²/Hz: integrating it over frequency gives the variance.
    Density,
    /// Power spectrum, units²: a sinusoid of amplitude `A` centred on a bin reads `A²/2`.
    Spectrum,
}

/// Trend removed from each segment before windowing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detrend {
    None,
    /// Subtract the segment mean.
    Constant,
    /// Subtract the least-squares line.
    Linear,
}

/// A one-sided spectrum estimate of a real signal.
#[derive(Debug, Clone)]
pub struct PowerSpectrum {
    /// Bin frequencies `k · fs / segment_length`, from 0 to the Nyquist frequency.
    pub frequencies: Vec<f64>,
    pub power: Vec<f64>,
    /// Number of averaged segments.
    pub segments: usize,
}

pub(crate) fn detrend_in_place(x: &mut [f64], detrend: Detrend) {
    let n = x.len() as f64;
    match detrend {
        Detrend::None => {}
        Detrend::Constant => {
            let mean = x.iter().sum::<f64>() / n;
            x.iter_mut().for_each(|v| *v -= mean);
        }
        Detrend::Linear => {
            let tm = (n - 1.0) / 2.0;
            let ym = x.iter().sum::<f64>() / n;
            let sxy: f64 = x.iter().enumerate().map(|(i, v)| (i as f64 - tm) * (v - ym)).sum();
            let sxx: f64 = (0..x.len()).map(|i| (i as f64 - tm).powi(2)).sum();
            let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
            x.iter_mut().enumerate().for_each(|(i, v)| *v -= ym + slope * (i as f64 - tm));
        }
    }
}

/// Welch's estimate of the one-sided spectrum of the real signal `x` sampled at `fs`: the signal
/// is cut into segments of `window.len()` samples overlapping by `overlap` samples, each is
/// detrended, windowed and transformed, and the squared magnitudes are averaged. Averaging
/// trades frequency resolution (`fs / segment`) for a variance reduced roughly by the number of
/// segments. With a single segment spanning the signal this is the (modified) periodogram.
///
/// Scaling follows SciPy: `1 / (fs Σw²)` for density, `1 / (Σw)²` for spectrum, with all bins
/// except DC (and Nyquist for even lengths) doubled to fold in the negative frequencies.
pub fn welch(x: &[f64], fs: f64, window: &[f64], overlap: usize, detrend: Detrend, scaling: SpectralScaling) -> Result<PowerSpectrum, ErrorsJSL> {
    let m = window.len();
    if m == 0 || overlap >= m {
        return Err(ErrorsJSL::InvalidInputRange("The window must be non-empty and longer than the overlap."));
    }
    if x.len() < m {
        return Err(ErrorsJSL::InvalidInputRange("The signal is shorter than one segment."));
    }
    if fs.is_nan() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The sampling rate must be positive."));
    }
    // ANCHOR: welch
    let step = m - overlap;
    let segments = (x.len() - m) / step + 1;
    let mut fft = RealFft::new(m)?;
    let bins = fft.spectrum_len();
    let mut power = vec![0.0; bins];
    let mut segment = vec![0.0; m];
    for s in 0..segments {
        segment.copy_from_slice(&x[s * step..s * step + m]);
        detrend_in_place(&mut segment, detrend);
        segment.iter_mut().zip(window).for_each(|(v, w)| *v *= w);
        let spectrum = fft.forward(&segment)?;
        power.iter_mut().zip(&spectrum).for_each(|(p, c)| *p += c.norm_sqr());
    }
    let scale = match scaling {
        SpectralScaling::Density => 1.0 / (fs * window.iter().map(|w| w * w).sum::<f64>()),
        SpectralScaling::Spectrum => 1.0 / window.iter().sum::<f64>().powi(2),
    } / segments as f64;
    let last_doubled = if m.is_multiple_of(2) { bins - 1 } else { bins };
    for (k, p) in power.iter_mut().enumerate() {
        *p *= scale * if k > 0 && k < last_doubled { 2.0 } else { 1.0 };
    }
    // ANCHOR_END: welch
    let frequencies = (0..bins).map(|k| k as f64 * fs / m as f64).collect();
    Ok(PowerSpectrum { frequencies, power, segments })
}

/// The periodogram of `x` with the given window spanning the whole signal.
pub fn periodogram(x: &[f64], fs: f64, window: &[f64], detrend: Detrend, scaling: SpectralScaling) -> Result<PowerSpectrum, ErrorsJSL> {
    if window.len() != x.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((window.len(), x.len())));
    }
    welch(x, fs, window, 0, detrend, scaling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::{WindowType, generate_window};
    use lib_jsl_random::distributions::gaussian_distribution_box_muller_vec;
    use lib_jsl_random::uniform_generator::UniformRNG;
    use std::f64::consts::TAU;

    #[test]
    fn white_noise_density_integrates_to_the_variance() {
        let (fs, sigma) = (1000.0, 0.7);
        let mut x = gaussian_distribution_box_muller_vec(200_000, &mut UniformRNG::from_seed(1), 0.0, sigma);
        x.truncate(200_000);
        let w = generate_window(WindowType::Hann, 512, false);
        let psd = welch(&x, fs, &w, 256, Detrend::Constant, SpectralScaling::Density).unwrap();
        assert_eq!(psd.segments, (200_000 - 512) / 256 + 1);
        // Flat at 2σ²/fs (one-sided) and integrating to σ².
        let level = 2.0 * sigma * sigma / fs;
        let interior = &psd.power[5..250];
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;
        assert!((mean / level - 1.0).abs() < 0.02, "{mean} vs {level}");
        let df = fs / 512.0;
        let total: f64 = psd.power.iter().sum::<f64>() * df;
        assert!((total / (sigma * sigma) - 1.0).abs() < 0.02, "{total}");
        // Averaging 780 segments leaves a relative standard deviation near 1/√780 · (overlap factor).
        let sd = (interior.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / interior.len() as f64).sqrt();
        assert!(sd / mean < 0.06, "{}", sd / mean);
    }

    #[test]
    fn sinusoid_power_and_frequency() {
        let (fs, n, a, k) = (256.0, 1024, 3.0, 100);
        let f0 = k as f64 * fs / n as f64;
        let x: Vec<f64> = (0..n).map(|i| a * (TAU * f0 * i as f64 / fs).sin() + 5.0).collect();
        let w = generate_window(WindowType::Hann, n, false);
        let p = periodogram(&x, fs, &w, Detrend::Constant, SpectralScaling::Spectrum).unwrap();
        let peak = (0..p.power.len()).max_by(|&i, &j| p.power[i].total_cmp(&p.power[j])).unwrap();
        assert_eq!(peak, k);
        assert!((p.frequencies[peak] - f0).abs() < 1e-12);
        assert!((p.power[peak] - a * a / 2.0).abs() < 1e-9, "{}", p.power[peak]);
        assert!(p.power[0] < 1e-20); // the offset is removed
        // Linear detrending removes a ramp exactly.
        let ramp: Vec<f64> = (0..64).map(|i| 2.0 + 0.5 * i as f64).collect();
        let r = periodogram(&ramp, 1.0, &[1.0; 64], Detrend::Linear, SpectralScaling::Spectrum).unwrap();
        assert!(r.power.iter().all(|&v| v < 1e-20));
        assert!(welch(&x, fs, &w[..8], 8, Detrend::None, SpectralScaling::Density).is_err());
        assert!(welch(&x[..4], fs, &w[..8], 0, Detrend::None, SpectralScaling::Density).is_err());
    }
}
