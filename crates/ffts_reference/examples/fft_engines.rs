//! The FFT: textbook engines and the production engine, on the spectrum of a two-tone signal.
//!
//! Run with `cargo run -p lib_jsl_ffts_reference --example fft_engines`.

use lib_jsl_ffts::{Fft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use lib_jsl_ffts_reference::{optimized_radix2::OptimizedRadix2FFT, optimized_split_radix::OptimizedSplitRadixFFT, simple_cooley_tukey::SimpleCooleyTukeyFFT};
use num_complex::Complex;

fn main() {
    // One second of a 50 Hz and a 120 Hz tone sampled at 1024 Hz.
    let (n, rate) = (1024usize, 1024.0);
    let x: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let t = k as f64 / rate;
            Complex::new((2.0 * std::f64::consts::PI * 50.0 * t).sin() + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).cos(), 0.0)
        })
        .collect();

    let mut fft = Fft::new();
    fft.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
    let spectrum = fft.execute(&x).unwrap();
    // Bin k is the frequency k · rate / n = k Hz; a real tone of amplitude a gives |X_k| = a n / 2.
    let mut peaks: Vec<(usize, f64)> = (0..n / 2).map(|k| (k, spectrum[k].norm() * 2.0 / n as f64)).filter(|p| p.1 > 0.1).collect();
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("peaks (Hz, amplitude): {peaks:.3?}");

    // Every engine implements FftEngine1D and computes the same transform.
    let mut engines: Vec<(&str, Box<dyn FftEngine1D>)> = vec![
        ("textbook radix 2", Box::new(SimpleCooleyTukeyFFT::new())),
        ("radix 2, exact twiddles", Box::new(OptimizedRadix2FFT::new())),
        ("split radix", Box::new(OptimizedSplitRadixFFT::new())),
    ];
    for (name, engine) in engines.iter_mut() {
        engine.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
        let y = engine.execute(&x).unwrap();
        let diff = y.iter().zip(&spectrum).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
        println!("{name:>24}: max difference from Fft {diff:.1e}");
    }

    // The inverse transform with 1/n scaling recovers the signal.
    let mut inverse = Fft::new();
    inverse.plan(n, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard).unwrap();
    let back = inverse.execute(&spectrum).unwrap();
    println!("round trip error {:.1e}", back.iter().zip(&x).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max));
}
