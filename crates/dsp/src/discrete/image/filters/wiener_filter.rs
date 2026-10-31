//! Wiener filtering: minimum-mean-square-error restoration of noisy and blurred images.
//!
//! Two classical forms are provided:
//!
//! - [`wiener_filter`]: **adaptive denoising**. At each pixel the local mean `μ` and variance `σ²`
//!   are estimated over a small window, and the pixel is pulled toward the mean in proportion to
//!   how much of the local variance is likely noise (variance `ν²`):
//!   `out = μ + max(σ² − ν², 0) / max(σ², ν²) · (x − μ)`.
//!   Flat regions (where `σ² ≈ ν²`) are smoothed heavily; edges and texture (where `σ² ≫ ν²`) are
//!   left almost alone. The noise variance may be supplied or is estimated as the mean local variance.
//! - [`wiener_deconvolution`]: **frequency-domain deblurring**. For an image blurred by a known
//!   point-spread function `h` (transfer function `H`) and corrupted by noise, the MMSE estimate is
//!   `F̂ = conj(H) / (|H|² + K) · G`, where `K` is the noise-to-signal power ratio. With `K → 0` this
//!   is plain inverse filtering (which explodes wherever `H` is small); `K > 0` rolls the gain off
//!   there, trading residual blur for noise suppression.

use crate::discrete::fft_convolution::fft2d;
use crate::discrete::image::convolution::{separable_convolve, Border};
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use num::Complex;

/// Adaptive (local) Wiener denoising with a `window × window` neighbourhood (odd, at least 3).
/// `noise_variance` is the variance of the additive noise; `None` estimates it as the mean of the
/// local variances over the image (robust when most of the image is smooth).
pub fn wiener_filter(image: &Array2<f64>, window: usize, noise_variance: Option<f64>) -> Result<Array2<f64>, ErrorsJSL> {
    if window < 3 || window.is_multiple_of(2) {
        return Err(ErrorsJSL::InvalidInputRange("Window size must be odd and at least 3"));
    }
    if let Some(v) = noise_variance
        && (!(v >= 0.0) || !v.is_finite()) {
            return Err(ErrorsJSL::InvalidInputRange("Noise variance must be non-negative and finite"));
        }
    let box_kernel = vec![1.0 / window as f64; window];
    let mean = separable_convolve(image, &box_kernel, &box_kernel, Border::Reflect)?;
    let squares = image.mapv(|v| v * v);
    let mean_sq = separable_convolve(&squares, &box_kernel, &box_kernel, Border::Reflect)?;
    let variance = Array2::from_shape_fn(image.dim(), |i| (mean_sq[i] - mean[i] * mean[i]).max(0.0));
    let noise = noise_variance.unwrap_or_else(|| variance.sum() / variance.len() as f64);
    Ok(Array2::from_shape_fn(image.dim(), |i| {
        if variance[i] >= noise && variance[i] > 0.0 {
            mean[i] + (variance[i] - noise) / variance[i] * (image[i] - mean[i])
        } else {
            mean[i]
        }
    }))
}

/// Wiener deconvolution of `image` blurred by `psf`, with noise-to-signal power ratio
/// `noise_to_signal` (`K >= 0`).
///
/// The PSF is taken to be centred on its middle element and is *not* renormalised (a unit-sum PSF
/// preserves the image mean). The image is padded to power-of-two dimensions by repeating its edge
/// pixels, which limits ringing at the borders; the result is cropped back to the input size. The
/// model is exact for a blur that is circular over the padded size, and approximate otherwise.
pub fn wiener_deconvolution(image: &Array2<f64>, psf: &Array2<f64>, noise_to_signal: f64) -> Result<Array2<f64>, ErrorsJSL> {
    let (rows, cols) = image.dim();
    let (pr_rows, pr_cols) = psf.dim();
    if rows == 0 || cols == 0 || pr_rows == 0 || pr_cols == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Image and PSF must be non-empty"));
    }
    if !(noise_to_signal >= 0.0) || !noise_to_signal.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("Noise-to-signal ratio must be non-negative and finite"));
    }
    if pr_rows > rows || pr_cols > cols {
        return Err(ErrorsJSL::InvalidInputRange("The PSF must not be larger than the image"));
    }
    let (n_rows, n_cols) = (rows.next_power_of_two(), cols.next_power_of_two());
    let padded = Array2::from_shape_fn((n_rows, n_cols), |(r, c)| Complex::new(image[(r.min(rows - 1), c.min(cols - 1))], 0.0));
    // Place the PSF with its centre at (0, 0): circularly shift by half its size.
    let (cr, cc) = (pr_rows / 2, pr_cols / 2);
    let mut h = Array2::<Complex<f64>>::zeros((n_rows, n_cols));
    for ((r, c), &v) in psf.indexed_iter() {
        let rr = (r + n_rows - cr) % n_rows;
        let cc2 = (c + n_cols - cc) % n_cols;
        h[(rr, cc2)] += Complex::new(v, 0.0);
    }
    let g = fft2d(&padded, false)?;
    let hf = fft2d(&h, false)?;
    let restored_f = Array2::from_shape_fn((n_rows, n_cols), |i| {
        let denom = hf[i].norm_sqr() + noise_to_signal;
        if denom < 1e-24 { Complex::new(0.0, 0.0) } else { hf[i].conj() / denom * g[i] }
    });
    let restored = fft2d(&restored_f, true)?;
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| restored[(r, c)].re))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::image::convolution::gaussian_kernel_1d;
    use crate::discrete::image::filters::gaussian_blur::gaussian_blur;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn gauss(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>()
    }

    fn mse(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64
    }

    /// Piecewise-constant test scene: dark background, a bright square and a mid-gray bar.
    fn scene(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(r, c)| {
            if (n / 4..n / 2).contains(&r) && (n / 4..n / 2).contains(&c) {
                1.0
            } else if (3 * n / 4..3 * n / 4 + 4).contains(&r) {
                0.5
            } else {
                0.0
            }
        })
    }

    #[test]
    fn adaptive_filter_denoises_flat_regions_and_improves_overall_error() {
        let mut seed = 5;
        let clean = scene(64);
        let noisy = &clean + &Array2::from_shape_fn((64, 64), |_| 0.1 * gauss(&mut seed));
        let out = wiener_filter(&noisy, 5, Some(0.01)).unwrap();
        assert!(mse(&out, &clean) < 0.4 * mse(&noisy, &clean), "{} vs {}", mse(&out, &clean), mse(&noisy, &clean));
        // A flat patch (the top-left corner, far from structure) is smoothed strongly.
        let patch = |a: &Array2<f64>| {
            let v: Vec<f64> = (0..8).flat_map(|r| (0..8).map(move |c| a[(r, c)])).collect();
            let m = v.iter().sum::<f64>() / 64.0;
            v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 64.0
        };
        assert!(patch(&out) < 0.1 * patch(&noisy));
    }

    #[test]
    fn edges_are_preserved_better_than_by_plain_averaging() {
        let mut seed = 6;
        let clean = scene(64);
        let noisy = &clean + &Array2::from_shape_fn((64, 64), |_| 0.05 * gauss(&mut seed));
        let wiener = wiener_filter(&noisy, 5, None).unwrap();
        let smooth = gaussian_blur(&noisy, 1.5).unwrap();
        // Contrast across the square's left edge (columns 15 -> 16, in the middle of its height).
        let contrast = |a: &Array2<f64>| a[(24, 17)] - a[(24, 14)];
        assert!(contrast(&wiener) > contrast(&smooth) + 0.05, "wiener {} vs blur {}", contrast(&wiener), contrast(&smooth));
        assert!(contrast(&wiener) > 0.85);
    }

    #[test]
    fn estimated_noise_variance_is_close_to_the_known_one_and_zero_noise_is_identity() {
        let mut seed = 7;
        let clean = scene(64);
        let noisy = &clean + &Array2::from_shape_fn((64, 64), |_| 0.08 * gauss(&mut seed));
        let est = wiener_filter(&noisy, 5, None).unwrap();
        let known = wiener_filter(&noisy, 5, Some(0.0064)).unwrap();
        assert!(mse(&est, &clean) < 0.7 * mse(&noisy, &clean));
        assert!(mse(&known, &clean) < 0.7 * mse(&noisy, &clean));
        // With no noise assumed, every pixel keeps its value.
        let identity = wiener_filter(&noisy, 5, Some(0.0)).unwrap();
        assert!(identity.iter().zip(noisy.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
    }

    fn circular_blur(image: &Array2<f64>, sigma: f64) -> (Array2<f64>, Array2<f64>) {
        let k = gaussian_kernel_1d(sigma, 3.0).unwrap();
        let n = k.len();
        let psf = Array2::from_shape_fn((n, n), |(i, j)| k[i] * k[j]);
        let (rows, cols) = image.dim();
        let blurred = Array2::from_shape_fn((rows, cols), |(r, c)| {
            psf.indexed_iter()
                .map(|((i, j), w)| w * image[((r + rows + i - n / 2) % rows, (c + cols + j - n / 2) % cols)])
                .sum()
        });
        (blurred, psf)
    }

    #[test]
    fn deconvolution_inverts_a_noise_free_circular_blur() {
        let clean = scene(64);
        let (blurred, psf) = circular_blur(&clean, 1.2);
        assert!(mse(&blurred, &clean) > 1e-3, "the blur must actually change the image");
        let restored = wiener_deconvolution(&blurred, &psf, 1e-20).unwrap();
        assert!(mse(&restored, &clean) < 1e-8, "restoration error {}", mse(&restored, &clean));
    }

    #[test]
    fn with_noise_the_regularised_filter_beats_both_the_blurred_image_and_naive_inversion() {
        let mut seed = 9;
        let clean = scene(64);
        let (blurred, psf) = circular_blur(&clean, 1.5);
        let noisy = &blurred + &Array2::from_shape_fn((64, 64), |_| 0.01 * gauss(&mut seed));
        // K is the noise-to-signal power ratio *per frequency*. A piecewise-constant scene has far
        // less high-frequency power than its mean power suggests, so the best K is well above the
        // white-signal estimate (1e-4 / 0.08 ~ 1e-3); here K ~ 1e-2 is near the optimum.
        let wiener = wiener_deconvolution(&noisy, &psf, 1e-2).unwrap();
        let naive = wiener_deconvolution(&noisy, &psf, 1e-12).unwrap();
        assert!(mse(&wiener, &clean) < 0.65 * mse(&noisy, &clean), "{} vs {}", mse(&wiener, &clean), mse(&noisy, &clean));
        assert!(mse(&wiener, &clean) < 0.1 * mse(&naive, &clean), "naive inversion amplifies the noise: {}", mse(&naive, &clean));
    }

    #[test]
    fn non_power_of_two_images_are_handled_and_improved() {
        let clean = scene(48);
        let (blurred, psf) = circular_blur(&clean, 1.2);
        let restored = wiener_deconvolution(&blurred, &psf, 1e-3).unwrap();
        assert_eq!(restored.dim(), (48, 48));
        assert!(mse(&restored, &clean) < mse(&blurred, &clean));
    }

    #[test]
    fn invalid_arguments() {
        let img = scene(16);
        assert!(wiener_filter(&img, 4, None).is_err() && wiener_filter(&img, 1, None).is_err());
        assert!(wiener_filter(&img, 3, Some(-1.0)).is_err());
        let psf = Array2::from_elem((3, 3), 1.0 / 9.0);
        assert!(wiener_deconvolution(&img, &psf, -1.0).is_err());
        assert!(wiener_deconvolution(&img, &Array2::from_elem((20, 3), 0.1), 0.1).is_err());
        assert!(wiener_deconvolution(&Array2::<f64>::zeros((0, 4)), &psf, 0.1).is_err());
    }
}
