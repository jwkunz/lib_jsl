//! Blob detection with the scale-normalised Laplacian of Gaussian (LoG).
//!
//! A *blob* is a roughly round bright region on a darker background. Smoothing an image with a
//! Gaussian of scale `σ` and taking the Laplacian responds most strongly, at a blob's centre, when
//! `σ` matches the blob's size. Multiplying by `σ²` (scale normalisation) makes responses at
//! different scales comparable, so searching for local maxima over *position and scale* finds each
//! blob's centre and size at once. For a Gaussian blob of standard deviation `s` the response peaks
//! at `σ = s`; a disc of radius `R` peaks at `σ = R / √2`.
//!
//! Overlapping detections of the same blob at neighbouring positions or scales are pruned,
//! keeping the strongest.
//!
//! Only *bright* blobs are detected. A dark blob on a bright background gives a negative response
//! at its centre but, as with any LoG, weak positive sidelobes in a ring around it (about 13% of
//! the response of an equal-contrast bright blob), so use a `threshold` comfortably above that.
//!
//! See the book entry [Local features](https://jwkunz.github.io/lib_jsl/book/image/features.html#blobs-and-the-normalised-laplacian).

use crate::discrete::image::convolution::{correlate2d, Border};
use crate::discrete::image::filters::gaussian_blur::gaussian_blur;
use lib_jsl_core::ErrorsJSL;
use ndarray::{array, Array2};
use std::f64::consts::PI;

/// A detected blob.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blob {
    pub row: usize,
    pub col: usize,
    /// The detection scale; the blob's radius is about `sigma · √2`.
    pub sigma: f64,
    /// The scale-normalised LoG response (larger = stronger/higher-contrast).
    pub response: f64,
}

impl Blob {
    /// The blob's approximate radius, `σ √2`.
    pub fn radius(&self) -> f64 {
        self.sigma * std::f64::consts::SQRT_2
    }
}

/// Fraction of the smaller circle's area covered by the intersection of two circles.
fn overlap_fraction(a: &Blob, b: &Blob) -> f64 {
    let (ra, rb) = (a.radius(), b.radius());
    let d = ((a.row as f64 - b.row as f64).powi(2) + (a.col as f64 - b.col as f64).powi(2)).sqrt();
    let (small, large) = if ra < rb { (ra, rb) } else { (rb, ra) };
    if d >= ra + rb {
        return 0.0;
    }
    if d <= large - small {
        return 1.0; // the smaller circle lies inside the larger
    }
    let part = |r1: f64, r2: f64| {
        let x = ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).clamp(-1.0, 1.0);
        r1 * r1 * x.acos()
    };
    let kite = 0.5 * ((-d + ra + rb) * (d + ra - rb) * (d - ra + rb) * (d + ra + rb)).max(0.0).sqrt();
    (part(ra, rb) + part(rb, ra) - kite) / (PI * small * small)
}

/// Detects bright blobs of scale between `min_sigma` and `max_sigma` (`num_sigma` geometrically
/// spaced scales). A detection must be a local maximum over its 3×3×3 neighbourhood in position and
/// scale with a normalised response above `threshold`; of two detections overlapping by more than
/// `overlap` (a fraction of the smaller blob's area, in `[0, 1]`) only the stronger is kept.
/// The image intensity scale matters for `threshold` (about 0.1 is reasonable for images in `[0, 1]`).
pub fn blob_detection(image: &Array2<f64>, min_sigma: f64, max_sigma: f64, num_sigma: usize, threshold: f64, overlap: f64) -> Result<Vec<Blob>, ErrorsJSL> {
    if !(min_sigma > 0.0) || !(max_sigma >= min_sigma) || num_sigma < 3 {
        return Err(ErrorsJSL::InvalidInputRange("Need 0 < min_sigma <= max_sigma and at least 3 scales"));
    }
    if !(threshold >= 0.0) || !(0.0..=1.0).contains(&overlap) {
        return Err(ErrorsJSL::InvalidInputRange("threshold must be non-negative and overlap within [0, 1]"));
    }
    let (rows, cols) = image.dim();
    if rows < 3 || cols < 3 {
        return Err(ErrorsJSL::InvalidInputRange("Image must be at least 3x3"));
    }
    let ratio = if num_sigma > 1 { (max_sigma / min_sigma).powf(1.0 / (num_sigma - 1) as f64) } else { 1.0 };
    let sigmas: Vec<f64> = (0..num_sigma).map(|i| min_sigma * ratio.powi(i as i32)).collect();
    // ANCHOR: log_stack
    let laplacian = array![[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]];
    // Scale-normalised, sign-flipped so bright blobs are positive maxima.
    let stack: Vec<Array2<f64>> = sigmas
        .iter()
        .map(|&s| {
            let lap = correlate2d(&gaussian_blur(image, s)?, &laplacian, Border::Replicate)?;
            Ok(lap.mapv(|v| -s * s * v))
        })
        .collect::<Result<_, ErrorsJSL>>()?;
    // ANCHOR_END: log_stack

    let mut blobs = Vec::new();
    for i in 0..num_sigma {
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                let v = stack[i][(r, c)];
                if v < threshold {
                    continue;
                }
                let mut is_max = true;
                'nbr: for di in -1isize..=1 {
                    let ii = i as isize + di;
                    if ii < 0 || ii >= num_sigma as isize {
                        continue;
                    }
                    for dr in -1isize..=1 {
                        for dc in -1isize..=1 {
                            if (di, dr, dc) == (0, 0, 0) {
                                continue;
                            }
                            let n = stack[ii as usize][((r as isize + dr) as usize, (c as isize + dc) as usize)];
                            // Strict on the "earlier" half so a plateau yields one detection.
                            if n > v || (n == v && (di, dr, dc) < (0, 0, 0)) {
                                is_max = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if is_max {
                    blobs.push(Blob { row: r, col: c, sigma: sigmas[i], response: v });
                }
            }
        }
    }
    blobs.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
    let mut kept: Vec<Blob> = Vec::new();
    for b in blobs {
        if kept.iter().all(|k| overlap_fraction(k, &b) <= overlap) {
            kept.push(b);
        }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_blobs(n: usize, blobs: &[(f64, f64, f64, f64)]) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(r, c)| blobs.iter().map(|&(br, bc, s, a)| a * (-((r as f64 - br).powi(2) + (c as f64 - bc).powi(2)) / (2.0 * s * s)).exp()).sum())
    }

    #[test]
    fn finds_gaussian_blobs_at_their_centres_with_the_right_scale() {
        let img = gaussian_blobs(80, &[(24.0, 24.0, 4.0, 1.0), (52.0, 56.0, 7.0, 1.0)]);
        let blobs = blob_detection(&img, 2.5, 10.0, 11, 0.05, 0.5).unwrap();
        assert_eq!(blobs.len(), 2, "{blobs:?}");
        let small = blobs.iter().find(|b| b.row.abs_diff(24) <= 1).expect("small blob");
        let large = blobs.iter().find(|b| b.row.abs_diff(52) <= 1).expect("large blob");
        assert!(small.col.abs_diff(24) <= 1 && large.col.abs_diff(56) <= 1);
        // For a Gaussian blob of standard deviation s the normalised LoG peaks at sigma = s.
        assert!((small.sigma / 4.0 - 1.0).abs() < 0.15, "small sigma {}", small.sigma);
        assert!((large.sigma / 7.0 - 1.0).abs() < 0.15, "large sigma {}", large.sigma);
        assert!((large.radius() / (7.0 * std::f64::consts::SQRT_2) - 1.0).abs() < 0.15);
    }

    #[test]
    fn a_disc_peaks_at_its_radius_over_root_two() {
        let img = Array2::from_shape_fn((64, 64), |(r, c)| if ((r as f64 - 32.0).powi(2) + (c as f64 - 32.0).powi(2)).sqrt() < 10.0 { 1.0 } else { 0.0 });
        let blobs = blob_detection(&img, 3.0, 10.0, 10, 0.1, 0.5).unwrap();
        assert_eq!(blobs.len(), 1, "{blobs:?}");
        assert!(blobs[0].row.abs_diff(32) <= 1 && blobs[0].col.abs_diff(32) <= 1);
        assert!((blobs[0].radius() / 10.0 - 1.0).abs() < 0.2, "estimated radius {}", blobs[0].radius());
    }

    #[test]
    fn threshold_rejects_weak_blobs_and_dark_blobs_are_ignored() {
        let img = gaussian_blobs(80, &[(24.0, 24.0, 5.0, 1.0), (56.0, 56.0, 5.0, 0.15)]);
        let strong_only = blob_detection(&img, 3.0, 8.0, 8, 0.2, 0.5).unwrap();
        assert_eq!(strong_only.len(), 1);
        assert!(strong_only[0].row.abs_diff(24) <= 1);
        let both = blob_detection(&img, 3.0, 8.0, 8, 0.02, 0.5).unwrap();
        assert_eq!(both.len(), 2);
        // A dark blob on a bright background is a minimum of the response, not a detection.
        let dark = gaussian_blobs(60, &[(30.0, 30.0, 5.0, -1.0)]).mapv(|v| v + 1.0);
        // (A dark blob's LoG response has weak positive sidelobes in a ring around it, about 13%
        // of the response of an equal-contrast bright blob, so the threshold must exceed that.)
        assert!(blob_detection(&dark, 3.0, 8.0, 8, 0.15, 0.5).unwrap().is_empty());
        let bright = gaussian_blobs(60, &[(30.0, 30.0, 5.0, 1.0)]);
        assert_eq!(blob_detection(&bright, 3.0, 8.0, 8, 0.15, 0.5).unwrap().len(), 1, "the same contrast, bright, is found");
    }

    #[test]
    fn nearby_duplicates_are_pruned_by_the_overlap_rule() {
        let img = gaussian_blobs(80, &[(40.0, 36.0, 5.0, 1.0), (40.0, 44.0, 5.0, 0.9)]);
        let merged = blob_detection(&img, 3.0, 9.0, 10, 0.05, 0.2).unwrap();
        let separate = blob_detection(&img, 3.0, 9.0, 10, 0.05, 1.0).unwrap();
        assert!(merged.len() <= separate.len(), "stricter overlap keeps fewer blobs");
        // Overlap geometry sanity.
        let a = Blob { row: 10, col: 10, sigma: 5.0, response: 1.0 };
        assert_eq!(overlap_fraction(&a, &a), 1.0);
        assert_eq!(overlap_fraction(&a, &Blob { row: 10, col: 100, sigma: 5.0, response: 1.0 }), 0.0);
        let half = overlap_fraction(&a, &Blob { row: 10, col: 20, sigma: 5.0, response: 1.0 });
        assert!(half > 0.0 && half < 1.0, "{half}");
    }

    #[test]
    fn flat_and_invalid_inputs() {
        let flat = Array2::from_elem((30, 30), 0.5);
        assert!(blob_detection(&flat, 2.0, 6.0, 5, 0.01, 0.5).unwrap().is_empty());
        assert!(blob_detection(&flat, 0.0, 6.0, 5, 0.01, 0.5).is_err());
        assert!(blob_detection(&flat, 6.0, 2.0, 5, 0.01, 0.5).is_err());
        assert!(blob_detection(&flat, 2.0, 6.0, 2, 0.01, 0.5).is_err());
        assert!(blob_detection(&flat, 2.0, 6.0, 5, -1.0, 0.5).is_err());
        assert!(blob_detection(&flat, 2.0, 6.0, 5, 0.01, 1.5).is_err());
        assert!(blob_detection(&Array2::<f64>::zeros((2, 2)), 2.0, 6.0, 5, 0.01, 0.5).is_err());
    }
}
