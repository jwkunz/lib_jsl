//! Radar pulse compression: linear FM chirps and matched filtering.
//!
//! A long pulse carries energy for detection range, a wide bandwidth gives range resolution;
//! a frequency-modulated pulse has both. Correlating the echo with the transmitted pulse (the
//! matched filter, which maximises the output SNR in white noise) compresses each target's echo
//! into a peak about `1/B` wide, with a processing gain equal to the time-bandwidth product.

use crate::discrete::convolve::ConvolveMode;
use crate::discrete::fft_convolution::fft_convolve;
use lib_jsl_core::ErrorsJSL;
use num::Complex;
use std::f64::consts::PI;

type C64 = Complex<f64>;

/// A complex baseband linear-FM chirp sweeping `−B/2 … +B/2` over `duration` seconds at sample
/// rate `fs`: `s(t) = exp(jπ (B/T) (t − T/2)²)`.
pub fn linear_chirp(bandwidth: f64, duration: f64, fs: f64) -> Result<Vec<C64>, ErrorsJSL> {
    if !(bandwidth > 0.0 && duration > 0.0 && fs > 0.0) || bandwidth > fs {
        return Err(ErrorsJSL::InvalidInputRange("Need positive bandwidth, duration and rate, with bandwidth <= fs."));
    }
    let n = (duration * fs).round() as usize;
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("The pulse is shorter than one sample."));
    }
    let rate = bandwidth / duration;
    Ok((0..n).map(|i| { let t = i as f64 / fs - duration / 2.0; C64::from_polar(1.0, PI * rate * t * t) }).collect())
}

/// The matched filter of `pulse`, optionally amplitude-weighted (e.g. by a Hamming or Taylor
/// window) to lower range sidelobes at a small cost in SNR and resolution: the time-reversed,
/// conjugated pulse.
pub fn matched_filter(pulse: &[C64], weights: Option<&[f64]>) -> Result<Vec<C64>, ErrorsJSL> {
    if let Some(w) = weights
        && w.len() != pulse.len()
    {
        return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), pulse.len())));
    }
    Ok((0..pulse.len()).rev().map(|i| pulse[i].conj() * weights.map_or(1.0, |w| w[i])).collect())
}

/// Compresses `received` with the (optionally weighted) matched filter of `pulse`. Output index
/// `k` is the correlation at delay `k` samples, `Σ_n r[n + k] w[n] conj(p[n])`, so a target whose
/// echo starts at sample `k` peaks at index `k`. The output has `received.len()` samples.
pub fn pulse_compress(received: &[C64], pulse: &[C64], weights: Option<&[f64]>) -> Result<Vec<C64>, ErrorsJSL> {
    if pulse.is_empty() || received.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("The pulse and the received signal must not be empty."));
    }
    let filter = matched_filter(pulse, weights)?;
    let full = fft_convolve(received, &filter, ConvolveMode::Full)?;
    let offset = pulse.len() - 1;
    Ok((0..received.len()).map(|k| full[k + offset]).collect())
}

/// The peak sidelobe level (dB relative to the main peak) of a compressed pulse: the largest
/// magnitude outside the main lobe, which extends from the peak to the first local minima.
pub fn peak_sidelobe_level_db(compressed: &[C64]) -> f64 {
    let magnitude: Vec<f64> = compressed.iter().map(|c| c.norm()).collect();
    let Some(peak) = (0..magnitude.len()).max_by(|&i, &j| magnitude[i].total_cmp(&magnitude[j])) else { return f64::NEG_INFINITY };
    let mut right = peak;
    while right + 1 < magnitude.len() && magnitude[right + 1] < magnitude[right] {
        right += 1;
    }
    let mut left = peak;
    while left > 0 && magnitude[left - 1] < magnitude[left] {
        left -= 1;
    }
    let sidelobe = magnitude[..left].iter().chain(&magnitude[right + 1..]).fold(0.0f64, |m, &v| m.max(v));
    20.0 * (sidelobe / magnitude[peak]).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::{WindowType, generate_window};

    fn echo_scene(pulse: &[C64], delays: &[(usize, f64)], length: usize) -> Vec<C64> {
        let mut r = vec![C64::new(0.0, 0.0); length];
        for &(d, a) in delays {
            r[d..d + pulse.len()].iter_mut().zip(pulse).for_each(|(v, p)| *v += a * p);
        }
        r
    }

    #[test]
    fn resolves_close_targets_with_the_expected_gain() {
        // B = 10 MHz, T = 20 µs at 40 MHz: N = 800 samples, BT = 200.
        let (b, t, fs) = (10e6, 20e-6, 40e6);
        let pulse = linear_chirp(b, t, fs).unwrap();
        assert_eq!(pulse.len(), 800);
        // Targets 20 samples (half a microsecond, 5 resolution cells) apart overlap heavily in time.
        // A single echo compresses to a peak equal to the pulse energy (N for unit amplitude).
        let single = pulse_compress(&echo_scene(&pulse, &[(300, 1.0)], 2000), &pulse, None).unwrap();
        assert!((single[300].norm() - 800.0).abs() < 1e-6);
        let r = echo_scene(&pulse, &[(300, 1.0), (320, 0.5)], 2000);
        let y = pulse_compress(&r, &pulse, None).unwrap();
        let magnitude: Vec<f64> = y.iter().map(|v| v.norm()).collect();
        // Both peaks stand out (each sits on the other's sidelobes) with a deep gap between.
        assert!((magnitude[300] / 800.0 - 1.0).abs() < 0.03 && (magnitude[320] / 400.0 - 1.0).abs() < 0.06, "{} {}", magnitude[300], magnitude[320]);
        let gap = magnitude[302..319].iter().fold(f64::INFINITY, |m, &v| m.min(v));
        assert!(gap < 0.25 * magnitude[320], "{gap}");
        // The −3 dB main-lobe width is about 0.886 fs / B = 3.5 samples.
        let half_power = magnitude[300] / 2f64.sqrt();
        let width = magnitude[290..311].iter().filter(|&&m| m >= half_power).count();
        assert!((3..=5).contains(&width), "{width}");
    }

    #[test]
    fn weighting_trades_sidelobes_for_resolution_and_validates() {
        let pulse = linear_chirp(5e6, 40e-6, 20e6).unwrap();
        let r = echo_scene(&pulse, &[(1000, 1.0)], 3000);
        let plain = peak_sidelobe_level_db(&pulse_compress(&r, &pulse, None).unwrap());
        // An unweighted LFM shows the −13.3 dB sinc sidelobe.
        assert!((plain + 13.3).abs() < 0.8, "{plain}");
        let window = generate_window(WindowType::Hamming, pulse.len(), true);
        let weighted = peak_sidelobe_level_db(&pulse_compress(&r, &pulse, Some(&window)).unwrap());
        assert!(weighted < -35.0, "{weighted}");
        assert!(linear_chirp(5e6, 1e-6, 1e6).is_err() && matched_filter(&pulse, Some(&[1.0])).is_err());
        assert!(pulse_compress(&r, &[], None).is_err());
    }
}
