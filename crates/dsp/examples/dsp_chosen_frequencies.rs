//! Evaluating spectra at chosen frequencies: Goertzel, the chirp-z zoom and Fourier integrals.

use lib_jsl_dsp::continuous::spectral::fourier_integrals::{FourierDirection, fourier_integral};
use lib_jsl_dsp::discrete::spectral::chirpz::chirpz;
use lib_jsl_dsp::discrete::spectral::goertzel::{goertzel_frequency, goertzel_power};
use num_complex::Complex;
use std::f64::consts::PI;

fn main() {
    // DTMF digit 5: 770 Hz and 1336 Hz, 205 samples at 8 kHz.
    let fs = 8000.0;
    let x: Vec<f64> = (0..205).map(|n| (2.0 * PI * 770.0 * n as f64 / fs).sin() + (2.0 * PI * 1336.0 * n as f64 / fs).sin()).collect();
    for f in [697.0, 770.0, 852.0, 941.0, 1209.0, 1336.0, 1477.0] {
        print!("{f}: {:6.1}   ", goertzel_frequency(&x, f, fs).unwrap().norm());
    }
    println!();
    println!("power in bin 20: {:.1}", goertzel_power(&x, 20.0).unwrap());
    // Zoom on 700-850 Hz with 16 points.
    let w = Complex::from_polar(1.0, -2.0 * PI * 10.0 / fs);
    let a = Complex::from_polar(1.0, 2.0 * PI * 700.0 / fs);
    let z = chirpz(&x, Some(16), Some(w), Some(a)).unwrap();
    let peak = (0..16).max_by(|&i, &j| z[i].norm().total_cmp(&z[j].norm())).unwrap();
    println!("chirp-z peak at {} Hz", 700 + 10 * peak);
    // The Fourier transform of a unit box on [-1, 1] is 2 sin(ω)/ω.
    let r = fourier_integral(|_| Complex::new(1.0, 0.0), -1.0, 1.0, 0.0, 6.0, 61, FourierDirection::Forward).unwrap();
    for (w, v) in r.omega.iter().zip(r.values.iter()).step_by(20) {
        let exact = if *w == 0.0 { 2.0 } else { 2.0 * w.sin() / w };
        println!("F({w:.1}) = {:.4} (exact {exact:.4})", v.re);
    }
}
