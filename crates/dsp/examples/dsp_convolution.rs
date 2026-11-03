//! Convolution, streaming FFT filtering and time-delay estimation by cross-correlation.

use lib_jsl_dsp::discrete::convolve::{ConvolveMethod, ConvolveMode, convolve};
use lib_jsl_dsp::discrete::fft_cross_correlation::fft_correlation_peak;
use lib_jsl_dsp::discrete::filters::fft_fir::FftFir;
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;

fn main() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let h = [1.0, -1.0];
    for mode in [ConvolveMode::Full, ConvolveMode::Same, ConvolveMode::Valid] {
        println!("{mode:?}: {:?}", convolve(&x, &h, mode, ConvolveMethod::Direct).unwrap());
    }

    // Streaming: feed a long signal in uneven chunks; the output equals one big convolution.
    let signal: Vec<f64> = (0..1000).map(|n| (0.05 * n as f64).sin()).collect();
    let taps: Vec<f64> = (0..63).map(|k| 1.0 / (1.0 + (k as f64 - 31.0).powi(2))).collect();
    let mut fir = FftFir::new(&taps, 200).unwrap();
    let mut streamed = Vec::new();
    for chunk in signal.chunks(137) {
        if let Some(y) = fir.process(chunk).unwrap() {
            streamed.extend(y);
        }
    }
    streamed.extend(fir.flush().unwrap().unwrap_or_default());
    let reference = convolve(&signal, &taps, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
    let err = streamed.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("overlap-save: {} outputs (block {}, FFT {}), largest difference {err:.1e}", streamed.len(), fir.block_len(), fir.fft_len());

    // A pulse and its echo 23 samples later.
    let pulse: Vec<f64> = (0..64).map(|n| if n < 8 { 1.0 } else { 0.0 }).collect();
    let mut echo = vec![0.0; 64];
    for n in 0..64 - 23 {
        echo[n + 23] = 0.5 * pulse[n];
    }
    let (lag, value) = fft_correlation_peak(&echo, &pulse).unwrap();
    println!("echo delay: {lag} samples (correlation {value:.2})");
}
