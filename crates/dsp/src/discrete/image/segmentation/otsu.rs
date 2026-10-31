//! Otsu's method: automatic global thresholding.
//!
//! Given the gray-level histogram of an image, Otsu's method picks the threshold `t` that
//! maximises the *between-class variance* of the two classes it creates (pixels `<= t` and
//! pixels `> t`), equivalently minimises the within-class variance. For a bimodal histogram
//! (dark background, bright foreground) it lands in the valley between the modes without any
//! tuning. With class probabilities `ω₀(t)`, `ω₁(t)` and means `μ₀(t)`, `μ₁(t)`:
//!
//! `σ_b²(t) = ω₀ ω₁ (μ₀ − μ₁)²`
//!
//! When several thresholds tie (as for an image with only two gray levels), the middle of the
//! tied range is returned.

use crate::discrete::image::filters::histogram_equalization::histogram;
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// The result of Otsu thresholding.
#[derive(Clone, Debug, PartialEq)]
pub struct OtsuResult {
    /// The threshold: pixels `<= threshold` are background.
    pub threshold: u8,
    /// The thresholded image: 0 for background, 255 for foreground.
    pub binary: Array2<u8>,
    /// The maximised between-class variance.
    pub between_class_variance: f64,
    /// Separability `σ_b² / σ_total²` in `[0, 1]`: near 1 for cleanly bimodal images, 0 for a flat one.
    pub separability: f64,
}

/// Computes the Otsu threshold of an 8-bit image.
pub fn otsu_threshold(image: &Array2<u8>) -> Result<u8, ErrorsJSL> {
    Ok(otsu(image)?.threshold)
}

/// Thresholds an 8-bit image with Otsu's method.
pub fn otsu(image: &Array2<u8>) -> Result<OtsuResult, ErrorsJSL> {
    if image.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Image must be non-empty"));
    }
    let hist = histogram(image);
    let total = image.len() as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &h)| i as f64 * h as f64).sum();
    let mean_all = sum_all / total;
    let variance_all: f64 = hist.iter().enumerate().map(|(i, &h)| (i as f64 - mean_all).powi(2) * h as f64).sum::<f64>() / total;

    let (mut w0, mut sum0) = (0.0f64, 0.0f64);
    let mut best = 0.0f64;
    let mut best_range = (0usize, 0usize);
    for t in 0..256 {
        w0 += hist[t] as f64;
        sum0 += t as f64 * hist[t] as f64;
        let w1 = total - w0;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
        let sb = (w0 / total) * (w1 / total) * (m0 - m1) * (m0 - m1);
        if sb > best + 1e-12 * best.max(1.0) {
            best = sb;
            best_range = (t, t);
        } else if (sb - best).abs() <= 1e-12 * best.max(1.0) && best > 0.0 {
            best_range.1 = t;
        }
    }
    let threshold = if best > 0.0 {
        ((best_range.0 + best_range.1) / 2) as u8
    } else {
        // A single gray level: no meaningful split; everything is background.
        image.iter().copied().max().unwrap()
    };
    Ok(OtsuResult {
        threshold,
        binary: image.mapv(|v| if v > threshold { 255 } else { 0 }),
        between_class_variance: best,
        separability: if variance_all > 0.0 { best / variance_all } else { 0.0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn gauss(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>()
    }

    /// Dark background (mean 60) with a bright rectangle (mean 180), plus Gaussian noise.
    fn bimodal(sigma: f64, seed: &mut u64) -> (Array2<u8>, Array2<bool>) {
        let truth = Array2::from_shape_fn((80, 80), |(r, c)| (20..60).contains(&r) && (25..70).contains(&c));
        let img = Array2::from_shape_fn((80, 80), |i| {
            let base = if truth[i] { 180.0 } else { 60.0 };
            (base + sigma * gauss(seed)).round().clamp(0.0, 255.0) as u8
        });
        (img, truth)
    }

    #[test]
    fn finds_the_valley_between_two_modes_and_segments_correctly() {
        let mut seed = 3;
        let (img, truth) = bimodal(12.0, &mut seed);
        let r = otsu(&img).unwrap();
        assert!((r.threshold as i32 - 120).abs() <= 8, "threshold {}", r.threshold);
        let wrong = r.binary.indexed_iter().filter(|(i, v)| (**v == 255) != truth[*i]).count();
        assert!(wrong < 20, "{wrong} misclassified pixels out of {}", img.len());
        assert!(r.separability > 0.85, "{}", r.separability);
        assert!(r.binary.iter().all(|v| *v == 0 || *v == 255));
        assert_eq!(otsu_threshold(&img).unwrap(), r.threshold);
    }

    #[test]
    fn two_gray_levels_give_a_threshold_between_them_and_perfect_separability() {
        let img = Array2::from_shape_fn((10, 10), |(r, _)| if r < 3 { 50u8 } else { 200u8 });
        let r = otsu(&img).unwrap();
        assert!((50..200).contains(&r.threshold), "{}", r.threshold);
        assert!(r.binary.row(0).iter().all(|v| *v == 0) && r.binary.row(9).iter().all(|v| *v == 255));
        assert!((r.separability - 1.0).abs() < 1e-9);
    }

    #[test]
    fn robust_to_unequal_class_sizes_and_more_noise() {
        let mut seed = 4;
        // A small bright object on a large dark background: heavy class imbalance.
        let truth = Array2::from_shape_fn((100, 100), |(r, c)| (40..52).contains(&r) && (40..55).contains(&c));
        let img = Array2::from_shape_fn((100, 100), |i| ((if truth[i] { 200.0 } else { 70.0 }) + 15.0 * gauss(&mut seed)).round().clamp(0.0, 255.0) as u8);
        let r = otsu(&img).unwrap();
        assert!((90..180).contains(&(r.threshold as i32)), "threshold {}", r.threshold);
        let found = r.binary.iter().filter(|v| **v == 255).count();
        assert!((found as f64 / truth.iter().filter(|v| **v).count() as f64 - 1.0).abs() < 0.15, "found {found}");
    }

    #[test]
    fn a_constant_image_has_nothing_to_separate() {
        let img = Array2::from_elem((6, 6), 90u8);
        let r = otsu(&img).unwrap();
        assert_eq!(r.threshold, 90);
        assert!(r.binary.iter().all(|v| *v == 0));
        assert_eq!((r.between_class_variance, r.separability), (0.0, 0.0));
        assert!(otsu(&Array2::<u8>::zeros((0, 3))).is_err());
    }

    #[test]
    fn matches_a_brute_force_search_on_a_random_histogram_image() {
        let mut seed = 8;
        let img = Array2::from_shape_fn((40, 40), |_| ((lcg(&mut seed) + 0.5).powi(2) * 200.0) as u8);
        let r = otsu(&img).unwrap();
        // Brute force: recompute within-class variance directly from the pixels.
        let pix: Vec<f64> = img.iter().map(|&v| v as f64).collect();
        let within = |t: u8| {
            let (lo, hi): (Vec<f64>, Vec<f64>) = pix.iter().partition(|&&v| v <= t as f64);
            let var = |s: &Vec<f64>| if s.is_empty() { 0.0 } else { let m = s.iter().sum::<f64>() / s.len() as f64; s.iter().map(|x| (x - m).powi(2)).sum::<f64>() };
            var(&lo) + var(&hi)
        };
        let best = (0..255u8).min_by(|&a, &b| within(a).partial_cmp(&within(b)).unwrap()).unwrap();
        assert!((within(r.threshold) - within(best)).abs() < 1e-6 * within(best).max(1.0), "{} vs {}", r.threshold, best);
    }
}
