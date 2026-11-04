//! Power spectra: periodogram, Welch averaging and the spectrogram.

use lib_jsl_dsp::discrete::spectral::stft::spectrogram;
use lib_jsl_dsp::discrete::spectral::welch::{Detrend, SpectralScaling, periodogram, welch};
use lib_jsl_dsp::discrete::windows::{WindowType, generate_window};
use std::f64::consts::PI;

fn main() {
    let fs = 1000.0;
    // A tone of amplitude 2 at 125 Hz plus a little deterministic "noise".
    let x: Vec<f64> = (0..8192).map(|n| 2.0 * (2.0 * PI * 125.0 * n as f64 / fs).sin() + 0.1 * ((n * 7919 % 1013) as f64 / 506.5 - 1.0)).collect();
    let p = periodogram(&x, fs, &generate_window(WindowType::Flattop, x.len(), false), Detrend::Constant, SpectralScaling::Spectrum).unwrap();
    let (k, peak) = p.power.iter().enumerate().fold((0, 0.0), |b, (k, &v)| if v > b.1 { (k, v) } else { b });
    println!("flat-top periodogram: peak {:.4} V² at {} Hz (A²/2 = 2)", peak, p.frequencies[k]);
    let w = welch(&x, fs, &generate_window(WindowType::Hann, 512, false), 256, Detrend::Constant, SpectralScaling::Density).unwrap();
    let total: f64 = w.power.iter().sum::<f64>() * fs / 512.0;
    println!("Welch: {} segments; integrated density {total:.3} (signal power {:.3})", w.segments, x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64);
    let s = spectrogram(&x, fs, &generate_window(WindowType::Hann, 256, false), 128, SpectralScaling::Density).unwrap();
    println!("spectrogram: {} frames × {} bins, first frame centred at {:.3} s", s.power.nrows(), s.power.ncols(), s.times[0]);
}
