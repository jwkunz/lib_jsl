//! Histogram equalisation, adaptive Wiener denoising and Wiener deconvolution.

use lib_jsl_dsp::discrete::image::convolution::gaussian_kernel_1d;
use lib_jsl_dsp::discrete::image::filters::gaussian_blur::gaussian_blur;
use lib_jsl_dsp::discrete::image::filters::histogram_equalization::{equalization_lut, histogram, histogram_equalization};
use lib_jsl_dsp::discrete::image::filters::wiener_filter::{wiener_deconvolution, wiener_filter};
use ndarray::Array2;

fn mse(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64
}

fn main() {
    // Four gray levels crowded into 100..=130.
    let dull = Array2::from_shape_fn((8, 8), |(r, c)| [100u8, 110, 120, 130][(r + c) % 4]);
    let lut = equalization_lut(&histogram(&dull));
    println!("equalised levels: 100 -> {}, 110 -> {}, 120 -> {}, 130 -> {}", lut[100], lut[110], lut[120], lut[130]);
    println!("equalised image row 0: {:?}", histogram_equalization(&dull).unwrap().row(0).to_vec());
    // A smooth image with deterministic "noise".
    let clean = Array2::from_shape_fn((64, 64), |(r, c)| if (16..48).contains(&r) && (16..48).contains(&c) { 0.8 } else { 0.2 });
    let noise = Array2::from_shape_fn((64, 64), |(r, c)| 0.05 * (((r * 31 + c * 17) % 13) as f64 / 6.0 - 1.0));
    let noisy = &clean + &noise;
    let variance = noise.iter().map(|v| v * v).sum::<f64>() / noise.len() as f64;
    println!("MSE: noisy {:.2e}, Wiener 5x5 {:.2e}", mse(&noisy, &clean), mse(&wiener_filter(&noisy, 5, Some(variance)).unwrap(), &clean));
    // Deconvolve a known Gaussian blur.
    let k = gaussian_kernel_1d(1.5, 4.0).unwrap();
    let psf = Array2::from_shape_fn((k.len(), k.len()), |(i, j)| k[i] * k[j]);
    let blurred = gaussian_blur(&clean, 1.5).unwrap();
    for nsr in [1e-1, 1e-2, 1e-3] {
        println!("deconvolution with K = {nsr:.0e}: MSE {:.2e} (blurred {:.2e})", mse(&wiener_deconvolution(&blurred, &psf, nsr).unwrap(), &clean), mse(&blurred, &clean));
    }
}
