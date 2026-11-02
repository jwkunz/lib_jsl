//! Real-input and two-dimensional FFTs.
//!
//! Run with `cargo run -p lib_jsl_ffts --example fft_real_and_2d`.

use lib_jsl_ffts::{fft2, ifft2, irfft, rfft};
use ndarray::Array2;
use num_complex::Complex;

fn main() {
    // A real signal: only the n/2 + 1 non-negative frequencies are returned.
    let x: Vec<f64> = (0..16).map(|k| (k as f64 * 0.8).sin() + 0.2).collect();
    let spectrum = rfft(&x).unwrap();
    println!("{} bins for {} samples; DC = {:.4} (16 × mean)", spectrum.len(), x.len(), spectrum[0].re);
    let back = irfft(&spectrum, x.len()).unwrap();
    println!("round trip error {:.1e}", back.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));

    // A 2D transform: a plane wave cos(2π(3i/8 + 2j/16)) has two spectral peaks.
    let image = Array2::from_shape_fn((8, 16), |(i, j)| Complex::new((2.0 * std::f64::consts::PI * (3.0 * i as f64 / 8.0 + 2.0 * j as f64 / 16.0)).cos(), 0.0));
    let f = fft2(&image).unwrap();
    let peaks: Vec<(usize, usize)> = f.indexed_iter().filter(|(_, v)| v.norm() > 1.0).map(|(ij, _)| ij).collect();
    println!("2D peaks at {peaks:?} with magnitude {:.1}", f[[3, 2]].norm());
    let restored = ifft2(&f).unwrap();
    println!("2D round trip error {:.1e}", (&restored - &image).iter().map(|v| v.norm()).fold(0.0, f64::max));
}
