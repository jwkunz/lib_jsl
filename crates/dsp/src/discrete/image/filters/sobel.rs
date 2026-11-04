//! Sobel edge/gradient filter.
//!
//! The Sobel operator estimates the image gradient with two 3×3 kernels that combine a central
//! difference along one axis with a `[1 2 1]` smoothing across the other, which makes it less
//! sensitive to noise than a plain difference:
//!
//! ```text
//!        [-1 0 1]          [-1 -2 -1]
//!   Gx = [-2 0 2]     Gy = [ 0  0  0]
//!        [-1 0 1]          [ 1  2  1]
//! ```
//!
//! `Gx` responds to intensity increasing to the right (along the column index) and `Gy` to
//! intensity increasing downward (along the row index). Their raw output is the gradient times 8;
//! pass `normalize = true` to get the gradient in intensity units per pixel.

use crate::discrete::image::convolution::{correlate2d, Border};
use lib_jsl_core::ErrorsJSL;
use ndarray::{array, Array2};

/// The gradient field of an image.
#[derive(Clone, Debug, PartialEq)]
pub struct SobelResult {
    /// Gradient along the columns (x).
    pub gx: Array2<f64>,
    /// Gradient along the rows (y, increasing downward).
    pub gy: Array2<f64>,
    /// Gradient magnitude `√(gx² + gy²)`.
    pub magnitude: Array2<f64>,
    /// Gradient direction `atan2(gy, gx)` in radians, in `(-π, π]`.
    pub direction: Array2<f64>,
}

/// Computes the Sobel gradient of `image` (borders replicated). With `normalize` the result is the
/// true gradient (the raw kernels' factor of 8 divided out).
pub fn sobel(image: &Array2<f64>, normalize: bool) -> Result<SobelResult, ErrorsJSL> {
    // ANCHOR: sobel
    let kx = array![[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let ky = array![[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    let mut gx = correlate2d(image, &kx, Border::Replicate)?;
    let mut gy = correlate2d(image, &ky, Border::Replicate)?;
    if normalize {
        gx.mapv_inplace(|v| v / 8.0);
        gy.mapv_inplace(|v| v / 8.0);
    }
    let magnitude = Array2::from_shape_fn(gx.dim(), |i| gx[i].hypot(gy[i]));
    let direction = Array2::from_shape_fn(gx.dim(), |i| gy[i].atan2(gx[i]));
    // ANCHOR_END: sobel
    Ok(SobelResult { gx, gy, magnitude, direction })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_ramps_give_their_exact_slope() {
        let ramp_x = Array2::from_shape_fn((9, 9), |(_, c)| 3.0 * c as f64 + 1.0);
        let s = sobel(&ramp_x, true).unwrap();
        for r in 0..9 {
            for c in 1..8 {
                assert!((s.gx[(r, c)] - 3.0).abs() < 1e-12 && s.gy[(r, c)].abs() < 1e-12);
            }
        }
        let ramp_y = Array2::from_shape_fn((9, 9), |(r, _)| -2.0 * r as f64);
        let s = sobel(&ramp_y, true).unwrap();
        assert!((s.gy[(4, 4)] + 2.0).abs() < 1e-12 && s.gx[(4, 4)].abs() < 1e-12);
        // Without normalisation the kernels' gain of 8 shows.
        assert!((sobel(&ramp_x, false).unwrap().gx[(4, 4)] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn a_vertical_edge_has_horizontal_gradient_and_direction_zero() {
        let img = Array2::from_shape_fn((8, 10), |(_, c)| if c < 5 { 0.0 } else { 10.0 });
        let s = sobel(&img, true).unwrap();
        for r in 1..7 {
            assert!((s.gx[(r, 4)] - 5.0).abs() < 1e-12 && (s.gx[(r, 5)] - 5.0).abs() < 1e-12, "row {r}");
            assert!(s.gy[(r, 4)].abs() < 1e-12);
            assert!(s.direction[(r, 4)].abs() < 1e-12, "gradient points toward increasing intensity (+x)");
            assert_eq!(s.gx[(r, 1)], 0.0, "flat regions have no gradient");
        }
    }

    #[test]
    fn direction_and_magnitude_follow_the_gradient_vector() {
        // I = 2x + 3y: gradient (2, 3).
        let img = Array2::from_shape_fn((9, 9), |(r, c)| 2.0 * c as f64 + 3.0 * r as f64);
        let s = sobel(&img, true).unwrap();
        assert!((s.magnitude[(4, 4)] - 13f64.sqrt()).abs() < 1e-12);
        assert!((s.direction[(4, 4)] - 3f64.atan2(2.0)).abs() < 1e-12);
        let falling = Array2::from_shape_fn((9, 9), |(_, c)| -(c as f64));
        assert!((sobel(&falling, true).unwrap().direction[(4, 4)].abs() - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn constant_images_and_empty_input() {
        let flat = Array2::from_elem((6, 6), 7.0);
        let s = sobel(&flat, false).unwrap();
        assert!(s.magnitude.iter().all(|v| *v == 0.0));
        assert!(sobel(&Array2::<f64>::zeros((0, 3)), true).is_err());
    }

    #[test]
    fn smoothing_makes_it_less_sensitive_to_a_single_noisy_pixel_than_a_plain_difference() {
        let mut img = Array2::<f64>::zeros((7, 7));
        img[(3, 3)] = 1.0;
        let s = sobel(&img, false).unwrap();
        // Plain central difference at (3,2) would be 1; Sobel weights the centre row by 2 of 4.
        assert!((s.gx[(3, 2)] - 2.0).abs() < 1e-12 && (s.gx[(2, 2)] - 1.0).abs() < 1e-12);
    }
}
