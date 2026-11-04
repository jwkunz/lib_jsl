//! Parametric spectra: linear prediction, Burg's maximum entropy method and MUSIC.

use lib_jsl_dsp::discrete::spectral::linear_predictive_coding::{linear_predictive_coding, lpc_residual};
use lib_jsl_dsp::discrete::spectral::maximum_entropy::burg;
use lib_jsl_dsp::discrete::spectral::music::music_real;
use std::f64::consts::PI;

fn main() {
    // An AR(2) resonance: x[n] = 1.6 x[n−1] − 0.9 x[n−2] + e[n].
    let mut state = 7u64;
    let mut noise = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut x = vec![0.0; 2000];
    for n in 2..x.len() {
        x[n] = 1.6 * x[n - 1] - 0.9 * x[n - 2] + noise();
    }
    let lpc = linear_predictive_coding(&x, 2).unwrap();
    let b = burg(&x, 2).unwrap();
    println!("LPC   A(z) = {:.4?}, reflection {:.4?}", lpc.coefficients, lpc.reflection);
    println!("Burg  A(z) = {:.4?}, noise variance {:.5} (true {:.5})", b.coefficients, b.variance, 1.0 / 12.0);
    let e = lpc_residual(&x, &lpc.coefficients);
    println!("prediction gain {:.1} dB", 10.0 * (x.iter().map(|v| v * v).sum::<f64>() / e.iter().map(|v| v * v).sum::<f64>()).log10());
    // Three sinusoids in 100 samples.
    let y: Vec<f64> = (0..100).map(|n| [0.1, 0.13, 0.31].iter().map(|f| (2.0 * PI * f * n as f64).sin()).sum::<f64>() + 0.05 * noise()).collect();
    let m = music_real(&y, 3, 30, 4001, 1.0).unwrap();
    println!("MUSIC peaks {:.4?}", m.peak_frequencies(3));
    println!("largest covariance eigenvalues {:.2?}", &m.eigenvalues[..8]);
}
