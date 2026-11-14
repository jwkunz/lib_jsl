//! Fourier-domain filters, following `scipy.ndimage`: `fourier_gaussian`, `fourier_uniform`,
//! `fourier_ellipsoid` and `fourier_shift`.
//!
//! By the convolution theorem, convolving an image with a kernel multiplies its discrete Fourier
//! transform by the kernel's transform. These functions take a spectrum (the output of a 2D FFT,
//! or of a real FFT along the rows) and multiply it by a transfer function given in closed form:
//! a Gaussian, a box (sinc), a disc or ellipse (the Airy pattern, `2 J₁(x)/x`), or a phase ramp
//! (a shift by any fraction of a pixel). The cost is that of the FFTs, whatever the kernel size,
//! and the convolution is circular: the image is treated as periodic.
//!
//! See the book entry [Fourier-domain filters](https://jwkunz.github.io/lib_jsl/book/image/fourier.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_ffts::multidimensional_fft::{fft2, ifft2};
use lib_jsl_special_functions::bessel_functions_integer::bessel_j_integer;
use ndarray::Array2;
use num_complex::Complex;

type C64 = Complex<f64>;

/// What kind of spectrum a Fourier filter receives (SciPy's `n` argument with `axis = -1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spectrum {
    /// The full 2D DFT (`n = -1`): frequencies `k/N` for `k < ⌈N/2⌉`, then `(k − N)/N`, on both axes.
    Full,
    /// A real FFT along the rows (the last axis) of an image with `columns` columns, so that each
    /// row holds `columns / 2 + 1` non-negative frequencies `k / columns`; the columns are a full DFT.
    RealRows { columns: usize },
}

// ANCHOR: frequencies
/// The frequencies (cycles per sample) of the rows and columns of a spectrum of shape `dim`.
pub fn frequencies(dim: (usize, usize), spectrum: Spectrum) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
    let full = |n: usize| -> Vec<f64> { (0..n).map(|k| if k < n.div_ceil(2) { k as f64 / n as f64 } else { (k as f64 - n as f64) / n as f64 }).collect() };
    let rows = full(dim.0);
    let cols = match spectrum {
        Spectrum::Full => full(dim.1),
        Spectrum::RealRows { columns } => {
            if columns / 2 + 1 != dim.1 {
                return Err(ErrorsJSL::IncompatibleArraySizes((dim.1, columns / 2 + 1)));
            }
            (0..dim.1).map(|k| k as f64 / columns as f64).collect()
        }
    };
    Ok((rows, cols))
}
// ANCHOR_END: frequencies

/// Multiplies a spectrum by a real or complex transfer function of the two frequencies.
fn apply(input: &Array2<C64>, spectrum: Spectrum, h: impl Fn(f64, f64) -> C64) -> Result<Array2<C64>, ErrorsJSL> {
    let (fr, fc) = frequencies(input.dim(), spectrum)?;
    Ok(Array2::from_shape_fn(input.dim(), |(i, j)| input[(i, j)] * h(fr[i], fc[j])))
}

// ANCHOR: transfer_functions
/// Gaussian smoothing with standard deviations `sigma = (rows, columns)` in pixels: multiplies by
/// `exp(−2π² (σ_r² f_r² + σ_c² f_c²))`, the Fourier transform of the continuous Gaussian.
pub fn fourier_gaussian(input: &Array2<C64>, sigma: (f64, f64), spectrum: Spectrum) -> Result<Array2<C64>, ErrorsJSL> {
    let k = -2.0 * std::f64::consts::PI.powi(2);
    apply(input, spectrum, |fr, fc| C64::new((k * (sigma.0 * sigma.0 * fr * fr + sigma.1 * sigma.1 * fc * fc)).exp(), 0.0))
}

/// A box (moving average) of `size = (rows, columns)` pixels: multiplies by
/// `sinc(s_r f_r) sinc(s_c f_c)`, with `sinc(x) = sin(πx)/(πx)`.
pub fn fourier_uniform(input: &Array2<C64>, size: (f64, f64), spectrum: Spectrum) -> Result<Array2<C64>, ErrorsJSL> {
    apply(input, spectrum, |fr, fc| C64::new(sinc(size.0 * fr) * sinc(size.1 * fc), 0.0))
}

/// A uniform ellipse with axes `size = (rows, columns)` pixels (a disc when they are equal):
/// multiplies by `2 J₁(x)/x` with `x = π √((s_r f_r)² + (s_c f_c)²)`, the Airy pattern.
pub fn fourier_ellipsoid(input: &Array2<C64>, size: (f64, f64), spectrum: Spectrum) -> Result<Array2<C64>, ErrorsJSL> {
    apply(input, spectrum, |fr, fc| {
        let x = std::f64::consts::PI * ((size.0 * fr).powi(2) + (size.1 * fc).powi(2)).sqrt();
        C64::new(if x == 0.0 { 1.0 } else { 2.0 * bessel_j_integer(1, x) / x }, 0.0)
    })
}

/// A shift by `shift = (rows, columns)` pixels, which need not be whole: multiplies by the phase
/// ramp `exp(−2πi (s_r f_r + s_c f_c))`. Content moves towards larger indices for positive shifts,
/// wrapping round the edges.
pub fn fourier_shift(input: &Array2<C64>, shift: (f64, f64), spectrum: Spectrum) -> Result<Array2<C64>, ErrorsJSL> {
    apply(input, spectrum, |fr, fc| C64::from_polar(1.0, -2.0 * std::f64::consts::PI * (shift.0 * fr + shift.1 * fc)))
}
// ANCHOR_END: transfer_functions

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Filters a real image in the Fourier domain: a 2D FFT, the given filter on the full spectrum,
/// and the real part of the inverse FFT. For example,
/// `fourier_filter(&image, |s| fourier_gaussian(s, (2.0, 2.0), Spectrum::Full))`.
pub fn fourier_filter(image: &Array2<f64>, filter: impl Fn(&Array2<C64>) -> Result<Array2<C64>, ErrorsJSL>) -> Result<Array2<f64>, ErrorsJSL> {
    let spectrum = fft2(&image.mapv(|v| C64::new(v, 0.0)))?;
    Ok(ifft2(&filter(&spectrum)?)?.mapv(|z| z.re))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequencies_of_odd_and_even_lengths() {
        let (r, c) = frequencies((4, 5), Spectrum::Full).unwrap();
        assert_eq!(r, vec![0.0, 0.25, -0.5, -0.25]);
        assert_eq!(c, vec![0.0, 0.2, 0.4, -0.4, -0.2]);
        let (_, c) = frequencies((4, 4), Spectrum::RealRows { columns: 7 }).unwrap();
        assert_eq!(c, vec![0.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0]);
    }

    #[test]
    fn a_whole_pixel_shift_rolls_the_image() {
        let img = Array2::from_shape_fn((6, 8), |(r, c)| (r * 8 + c) as f64);
        let out = fourier_filter(&img, |s| fourier_shift(s, (2.0, -3.0), Spectrum::Full)).unwrap();
        for ((r, c), v) in out.indexed_iter() {
            let want = img[((r + 6 - 2) % 6, (c + 3) % 8)];
            assert!((v - want).abs() < 1e-10, "({r},{c}) {v} vs {want}");
        }
    }

    #[test]
    fn smoothing_keeps_the_mean() {
        let img = Array2::from_shape_fn((16, 12), |(r, c)| ((r * 7 + c * 3) % 5) as f64);
        let mean = img.mean().unwrap();
        for f in [
            fourier_gaussian as fn(&Array2<C64>, (f64, f64), Spectrum) -> Result<Array2<C64>, ErrorsJSL>,
            fourier_uniform,
            fourier_ellipsoid,
        ] {
            let out = fourier_filter(&img, |s| f(s, (2.5, 1.5), Spectrum::Full)).unwrap();
            assert!((out.mean().unwrap() - mean).abs() < 1e-12);
        }
    }
}
