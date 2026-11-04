//! Channelising a signal with a polyphase DFT analysis filter bank.

use lib_jsl_dsp::discrete::spectral::oversampled_polyphase_filter_bank::OversampledPolyphaseAnalysisFilterBank;
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
use num_complex::Complex;
use std::f64::consts::PI;

fn main() {
    // Two complex tones, in channels 2 and 5 of a 8-channel bank.
    let m = 8;
    let x: Vec<Complex<f64>> = (0..1024)
        .map(|n| Complex::from_polar(1.0, 2.0 * PI * 2.0 / m as f64 * n as f64) + Complex::from_polar(0.1, 2.0 * PI * 5.1 / m as f64 * n as f64))
        .collect();
    let mut bank = OversampledPolyphaseAnalysisFilterBank::new(m, 2, None).unwrap();
    let frames = bank.process(&x).unwrap().unwrap();
    println!("{} frames of {} channels (hop {})", frames.len(), m, m / 2);
    let tail = &frames[frames.len() / 2..];
    for c in 0..m {
        let p = tail.iter().map(|f| f[c].norm_sqr()).sum::<f64>() / tail.len() as f64;
        println!("channel {c}: {:7.1} dB", 10.0 * p.max(1e-30).log10());
    }
}
