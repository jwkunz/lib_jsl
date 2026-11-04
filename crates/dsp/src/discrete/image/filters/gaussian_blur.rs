//! Gaussian blur.
//!
//! Convolving an image with a Gaussian smooths it isotropically, suppressing noise and detail
//! smaller than about `sigma` pixels. The 2D Gaussian is *separable* (the product of two 1D
//! Gaussians), so the blur is applied as a horizontal then a vertical 1D pass, costing `O(r)` per
//! pixel instead of `O(r²)` for a radius-`r` kernel. Gaussian blurs also compose: blurring by
//! `σ₁` and then `σ₂` equals one blur of `√(σ₁² + σ₂²)`, which is what scale-space methods such
//! as SIFT rely on.
//!
//! See the book entry [Image filtering](https://jwkunz.github.io/lib_jsl/book/image/filtering.html#gaussian-blur).

use crate::discrete::image::convolution::{gaussian_kernel_1d, separable_convolve, Border};
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// Blurs `image` with a Gaussian of standard deviation `sigma` pixels (kernel truncated at 4σ),
/// repeating edge pixels at the borders.
pub fn gaussian_blur(image: &Array2<f64>, sigma: f64) -> Result<Array2<f64>, ErrorsJSL> {
    gaussian_blur_with_border(image, sigma, Border::Replicate)
}

/// Like [`gaussian_blur`] with an explicit border mode.
pub fn gaussian_blur_with_border(image: &Array2<f64>, sigma: f64, border: Border) -> Result<Array2<f64>, ErrorsJSL> {
    let k = gaussian_kernel_1d(sigma, 4.0)?;
    separable_convolve(image, &k, &k, border)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse(n: usize) -> Array2<f64> {
        let mut a = Array2::zeros((n, n));
        a[(n / 2, n / 2)] = 1.0;
        a
    }

    #[test]
    fn a_constant_image_is_unchanged_with_replicated_borders() {
        let img = Array2::from_elem((12, 9), 3.5);
        let out = gaussian_blur(&img, 2.0).unwrap();
        assert!(out.iter().all(|v| (v - 3.5).abs() < 1e-12));
    }

    #[test]
    fn an_impulse_becomes_the_gaussian_kernel_and_conserves_mass() {
        let out = gaussian_blur(&impulse(41), 2.5).unwrap();
        assert!((out.sum() - 1.0).abs() < 1e-12, "mass {}", out.sum());
        // Symmetric about the centre, peaked there, with variance sigma^2 along each axis.
        assert!((out[(20, 20)] - out.iter().cloned().fold(0.0, f64::max)).abs() < 1e-15);
        assert!((out[(20, 17)] - out[(20, 23)]).abs() < 1e-15 && (out[(17, 20)] - out[(23, 20)]).abs() < 1e-15);
        let var_x: f64 = out.indexed_iter().map(|((_, c), v)| (c as f64 - 20.0).powi(2) * v).sum();
        let var_y: f64 = out.indexed_iter().map(|((r, _), v)| (r as f64 - 20.0).powi(2) * v).sum();
        assert!((var_x - 6.25).abs() < 0.02 && (var_y - 6.25).abs() < 0.02, "{var_x} {var_y}");
        // The 2D kernel is the product of two 1D ones: out[r][c] * out[0][0]... check separability.
        assert!((out[(20, 22)] * out[(22, 20)] - out[(20, 20)] * out[(22, 22)]).abs() < 1e-15);
    }

    #[test]
    fn blurs_compose_by_adding_variances() {
        let img = impulse(61);
        let two_step = gaussian_blur(&gaussian_blur(&img, 2.0).unwrap(), 1.5).unwrap();
        let one_step = gaussian_blur(&img, (2.0f64 * 2.0 + 1.5 * 1.5).sqrt()).unwrap();
        let worst = two_step.iter().zip(one_step.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst < 5e-4, "max difference {worst}");
    }

    #[test]
    fn smoothing_reduces_noise_variance_by_the_expected_factor() {
        let mut seed = 3u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 12f64.sqrt() // unit variance
        };
        let img = Array2::from_shape_fn((128, 128), |_| next());
        let out = gaussian_blur(&img, 2.0).unwrap();
        let var = |a: &Array2<f64>| a.iter().map(|v| v * v).sum::<f64>() / a.len() as f64;
        // White noise variance falls by 1 / (4 pi sigma^2) for a 2D Gaussian.
        let expected = 1.0 / (4.0 * std::f64::consts::PI * 4.0);
        assert!((var(&out) / expected - 1.0).abs() < 0.15, "variance {} vs {expected}", var(&out));
    }

    #[test]
    fn a_step_edge_stays_monotone_and_other_borders_work() {
        let img = Array2::from_shape_fn((5, 30), |(_, c)| if c < 15 { 0.0 } else { 1.0 });
        let out = gaussian_blur(&img, 2.0).unwrap();
        let row: Vec<f64> = out.row(2).to_vec();
        assert!(row.windows(2).all(|w| w[1] >= w[0] - 1e-15));
        assert!((row[14] + row[15] - 1.0).abs() < 1e-12, "antisymmetric about the edge");
        // Zero borders darken the edges but leave the interior (further than the kernel radius
        // of 6 pixels from every edge) untouched.
        let zero = gaussian_blur_with_border(&Array2::from_elem((21, 21), 1.0), 1.5, Border::Zero).unwrap();
        assert!(zero[(0, 0)] < 0.5 && (zero[(10, 10)] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_sigma() {
        let img = Array2::<f64>::zeros((4, 4));
        assert!(gaussian_blur(&img, 0.0).is_err() && gaussian_blur(&img, -1.0).is_err() && gaussian_blur(&img, f64::NAN).is_err());
    }
}
