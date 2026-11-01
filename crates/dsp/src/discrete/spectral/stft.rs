//! The short-time Fourier transform, its inverse and the spectrogram.

use super::welch::SpectralScaling;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_ffts::RealFft;
use ndarray::Array2;
use num_complex::Complex;

/// The one-sided STFT of a real signal: row `t` holds the spectrum (`window.len()/2 + 1` bins)
/// of the frame starting at sample `t · hop`, windowed by `window`. Frames run while they fit
/// entirely inside the signal (pad the signal first to cover its edges).
pub fn stft(x: &[f64], window: &[f64], hop: usize) -> Result<Array2<Complex<f64>>, ErrorsJSL> {
    let m = window.len();
    if m == 0 || hop == 0 || hop > m {
        return Err(ErrorsJSL::InvalidInputRange("Need a non-empty window and 0 < hop <= window length."));
    }
    if x.len() < m {
        return Err(ErrorsJSL::InvalidInputRange("The signal is shorter than one frame."));
    }
    let frames = (x.len() - m) / hop + 1;
    let mut fft = RealFft::new(m)?;
    let mut out = Array2::zeros((frames, fft.spectrum_len()));
    let mut frame = vec![0.0; m];
    for t in 0..frames {
        frame.iter_mut().zip(&x[t * hop..t * hop + m]).zip(window).for_each(|((f, v), w)| *f = v * w);
        let spectrum = fft.forward(&frame)?;
        out.row_mut(t).iter_mut().zip(spectrum).for_each(|(o, s)| *o = s);
    }
    Ok(out)
}

/// Inverts [`stft`] by weighted overlap-add, `x[n] = Σ_t w[n − t·hop] y_t[n − t·hop] / Σ_t w²`,
/// the least-squares signal for possibly modified frames (Griffin and Lim). Exact for an
/// unmodified STFT wherever `Σ w² > 0`; samples not covered by any frame are zero.
pub fn istft(frames: &Array2<Complex<f64>>, window: &[f64], hop: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let m = window.len();
    if m == 0 || hop == 0 || frames.ncols() != m / 2 + 1 {
        return Err(ErrorsJSL::InvalidInputRange("The frames do not match the window."));
    }
    let count = frames.nrows();
    let length = if count == 0 { 0 } else { (count - 1) * hop + m };
    let mut fft = RealFft::new(m)?;
    let mut x = vec![0.0; length];
    let mut weight = vec![0.0; length];
    for t in 0..count {
        let frame = fft.inverse(&frames.row(t).to_vec())?;
        for i in 0..m {
            x[t * hop + i] += window[i] * frame[i];
            weight[t * hop + i] += window[i] * window[i];
        }
    }
    x.iter_mut().zip(&weight).for_each(|(v, w)| if *w > 1e-12 { *v /= w } else { *v = 0.0 });
    Ok(x)
}

/// A spectrogram: per-frame one-sided power with Welch's scaling, and the frame times and bin
/// frequencies.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    /// `power[[t, k]]`.
    pub power: Array2<f64>,
    /// Centre time of each frame (seconds).
    pub times: Vec<f64>,
    pub frequencies: Vec<f64>,
}

/// The spectrogram `|STFT|²` of `x` sampled at `fs`, scaled as a density or spectrum like
/// [`super::welch::welch`] (each row is a single-segment periodogram).
pub fn spectrogram(x: &[f64], fs: f64, window: &[f64], hop: usize, scaling: SpectralScaling) -> Result<Spectrogram, ErrorsJSL> {
    let s = stft(x, window, hop)?;
    let m = window.len();
    let scale = match scaling {
        SpectralScaling::Density => 1.0 / (fs * window.iter().map(|w| w * w).sum::<f64>()),
        SpectralScaling::Spectrum => 1.0 / window.iter().sum::<f64>().powi(2),
    };
    let bins = s.ncols();
    let last_doubled = if m.is_multiple_of(2) { bins - 1 } else { bins };
    let power = Array2::from_shape_fn(s.dim(), |(t, k)| s[[t, k]].norm_sqr() * scale * if k > 0 && k < last_doubled { 2.0 } else { 1.0 });
    let times = (0..s.nrows()).map(|t| (t * hop) as f64 / fs + (m as f64 / 2.0) / fs).collect();
    let frequencies = (0..bins).map(|k| k as f64 * fs / m as f64).collect();
    Ok(Spectrogram { power, times, frequencies })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::{WindowType, generate_window};
    use std::f64::consts::TAU;

    #[test]
    fn istft_reconstructs_the_signal() {
        let x: Vec<f64> = (0..2000).map(|i| (0.013 * i as f64).sin() + 0.3 * (0.21 * i as f64).cos() + ((i * 7) % 11) as f64 * 0.01).collect();
        // Hann windows overlapped, and a boxcar without overlap (Hann would leave the frame
        // starts, where w = 0, unrecoverable).
        for (m, hop, kind) in [(256, 64, WindowType::Hann), (255, 100, WindowType::Hann), (64, 64, WindowType::Boxcar)] {
            let w = generate_window(kind, m, false);
            let s = stft(&x, &w, hop).unwrap();
            let y = istft(&s, &w, hop).unwrap();
            // Every interior sample (with non-zero window weight) is recovered.
            let worst = (m..y.len() - m).map(|i| (y[i] - x[i]).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-12, "m={m} hop={hop}: {worst}");
        }
    }

    #[test]
    fn spectrogram_tracks_a_linear_chirp() {
        let fs = 8000.0;
        let n = 16000;
        // Instantaneous frequency 500 + 1000 t Hz over two seconds.
        let x: Vec<f64> = (0..n).map(|i| { let t = i as f64 / fs; (TAU * (500.0 * t + 500.0 * t * t)).cos() }).collect();
        let w = generate_window(WindowType::Hann, 512, false);
        let sg = spectrogram(&x, fs, &w, 128, SpectralScaling::Spectrum).unwrap();
        for (t, &time) in sg.times.iter().enumerate().step_by(10) {
            let row = sg.power.row(t);
            let peak = (0..row.len()).max_by(|&i, &j| row[i].total_cmp(&row[j])).unwrap();
            let expected = 500.0 + 1000.0 * time;
            assert!((sg.frequencies[peak] - expected).abs() < 2.0 * fs / 512.0, "t={time}: {} vs {expected}", sg.frequencies[peak]);
        }
        assert!(stft(&x, &w, 0).is_err() && stft(&x[..10], &w, 1).is_err());
    }
}
