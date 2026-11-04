//! Image sharpening by unsharp masking.
//!
//! An image minus a blurred copy of itself keeps only the fine detail (the "mask"). Adding a
//! multiple of that detail back boosts local contrast along edges:
//!
//! `sharpened = image + amount · (image − blur(image, σ))`
//!
//! `sigma` sets the scale of detail that is boosted and `amount` the strength (0 does nothing, 1 is
//! a typical moderate setting). Edges gain overshoot ("halos") on either side, which is what makes
//! them look crisper. An optional threshold leaves low-contrast areas (mostly noise) untouched.
//!
//! See the book entry [Image filtering](https://jwkunz.github.io/lib_jsl/book/image/filtering.html#unsharp-masking).

use crate::discrete::image::filters::gaussian_blur::gaussian_blur;
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// Sharpens `image` with an unsharp mask of Gaussian scale `sigma` and strength `amount`.
pub fn sharpening(image: &Array2<f64>, sigma: f64, amount: f64) -> Result<Array2<f64>, ErrorsJSL> {
    sharpening_with_threshold(image, sigma, amount, 0.0)
}

/// Unsharp masking that only boosts pixels whose detail magnitude `|image − blur|` is at least
/// `threshold`, so flat noisy regions are not amplified.
pub fn sharpening_with_threshold(image: &Array2<f64>, sigma: f64, amount: f64, threshold: f64) -> Result<Array2<f64>, ErrorsJSL> {
    if !amount.is_finite() || !(threshold >= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("amount must be finite and threshold non-negative"));
    }
    // ANCHOR: unsharp
    let blurred = gaussian_blur(image, sigma)?;
    Ok(Array2::from_shape_fn(image.dim(), |i| {
        let detail = image[i] - blurred[i];
        if detail.abs() >= threshold { image[i] + amount * detail } else { image[i] }
    }))
    // ANCHOR_END: unsharp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(rows: usize, cols: usize, at: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(_, c)| if c < at { 0.2 } else { 0.8 })
    }

    #[test]
    fn amount_zero_and_flat_images_are_unchanged() {
        let img = step(8, 20, 10);
        assert_eq!(sharpening(&img, 1.5, 0.0).unwrap(), img);
        let flat = Array2::from_elem((10, 10), 0.4);
        assert!(sharpening(&flat, 2.0, 3.0).unwrap().iter().all(|v| (v - 0.4).abs() < 1e-12));
    }

    #[test]
    fn edges_gain_overshoot_and_steeper_slope() {
        let img = step(6, 30, 15);
        let out = sharpening(&img, 1.5, 1.5).unwrap();
        let row = out.row(3).to_vec();
        assert!(row[14] < 0.2 - 0.01, "undershoot on the dark side: {}", row[14]);
        assert!(row[15] > 0.8 + 0.01, "overshoot on the bright side: {}", row[15]);
        assert!((row[2] - 0.2).abs() < 1e-9 && (row[27] - 0.8).abs() < 1e-9, "far from the edge nothing changes");
        // Slope across the edge is steeper than the original's single-pixel step... compare to blur.
        let blurred = gaussian_blur(&img, 1.5).unwrap();
        assert!(row[15] - row[14] > blurred[(3, 15)] - blurred[(3, 14)]);
    }

    #[test]
    fn sharpening_restores_a_blurred_edge_toward_the_original() {
        let original = step(10, 40, 20);
        let blurred = gaussian_blur(&original, 1.5).unwrap();
        let restored = sharpening(&blurred, 1.5, 1.0).unwrap();
        let err = |a: &Array2<f64>| a.iter().zip(original.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
        // Unsharp masking is a contrast heuristic, not an exact inverse: it recovers a good part of
        // the lost edge (about 30% less squared error here), not all of it.
        assert!(err(&restored) < 0.8 * err(&blurred), "{} vs {}", err(&restored), err(&blurred));
    }

    #[test]
    fn the_threshold_protects_low_contrast_noise() {
        let mut img = Array2::from_elem((12, 12), 0.5);
        img[(6, 6)] = 0.51; // faint speck
        let boosted = sharpening(&img, 1.0, 5.0).unwrap();
        let guarded = sharpening_with_threshold(&img, 1.0, 5.0, 0.02).unwrap();
        assert!(boosted[(6, 6)] > 0.54, "unthresholded amplifies the speck: {}", boosted[(6, 6)]);
        assert_eq!(guarded, img);
    }

    #[test]
    fn invalid_arguments() {
        let img = Array2::<f64>::zeros((4, 4));
        assert!(sharpening(&img, 0.0, 1.0).is_err());
        assert!(sharpening(&img, 1.0, f64::NAN).is_err());
        assert!(sharpening_with_threshold(&img, 1.0, 1.0, -0.1).is_err());
    }
}
