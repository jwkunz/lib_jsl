//! Discrete and continuous wavelet transforms.

use lib_jsl_dsp::discrete::spectral::wavelet_transform::{ContinuousWavelet, Wavelet, continuous_wavelet_transform, dwt, idwt, wavedec, waverec};

fn main() {
    let x: Vec<f64> = (0..16).map(|n| if n < 8 { 1.0 } else { 0.0 } + 0.1 * n as f64).collect();
    let (a, d) = dwt(&x, Wavelet::Haar).unwrap();
    println!("Haar approximation {a:.3?}");
    println!("Haar detail        {d:.3?}");
    let back = idwt(&a, &d, Wavelet::Haar).unwrap();
    println!("reconstruction error {:.1e}", back.iter().zip(&x).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max));
    // Daubechies 2 has two vanishing moments: a linear signal leaves no detail (away from the ends).
    let ramp: Vec<f64> = (0..64).map(|n| 3.0 + 0.5 * n as f64).collect();
    let dec = wavedec(&ramp, Wavelet::Daubechies(2), 3).unwrap();
    let interior = &dec.details[2][2..28];
    println!("db2 finest details of a ramp (interior): max |d| = {:.1e}", interior.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    println!("waverec error {:.1e}", waverec(&dec, Wavelet::Daubechies(2)).unwrap().iter().zip(&ramp).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max));
    // A Morlet transform locates a burst in time and frequency.
    let sig: Vec<f64> = (0..512).map(|n| if (200..260).contains(&n) { (2.0 * std::f64::consts::PI * 0.1 * n as f64).sin() } else { 0.0 }).collect();
    let morlet = ContinuousWavelet::Morlet { omega0: 6.0 };
    let freqs = [0.05, 0.1, 0.2];
    let scales: Vec<f64> = freqs.iter().map(|&f| morlet.scale_for_frequency(f)).collect();
    let w = continuous_wavelet_transform(&sig, &scales, morlet).unwrap();
    for (k, f) in freqs.iter().enumerate() {
        let (t, v) = (0..512).map(|t| (t, w[[k, t]].norm())).fold((0, 0.0), |b, c| if c.1 > b.1 { c } else { b });
        println!("|W| at {f} cycles/sample peaks at sample {t} with {v:.2}");
    }
}
