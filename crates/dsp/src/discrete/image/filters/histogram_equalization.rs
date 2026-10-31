//! Histogram equalization of 8-bit images.
//!
//! A low-contrast image uses only a narrow band of the available gray levels. Histogram
//! equalization remaps the levels through the image's own cumulative distribution function so that
//! they are spread (approximately) uniformly over `0..=255`, stretching dense regions of the
//! histogram apart. The mapping is monotonic, so the order of brightness is preserved.
//!
//! With `cdf(v)` the number of pixels at or below `v`, `cdf_min` the first nonzero `cdf` value and
//! `N` the pixel count, level `v` maps to `round(255 · (cdf(v) − cdf_min) / (N − cdf_min))`.

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// The 256-bin histogram of an 8-bit image.
pub fn histogram(image: &Array2<u8>) -> [usize; 256] {
    let mut h = [0usize; 256];
    for &v in image.iter() {
        h[v as usize] += 1;
    }
    h
}

/// The equalizing lookup table for a histogram (identity for a single-valued image).
pub fn equalization_lut(hist: &[usize; 256]) -> [u8; 256] {
    let total: usize = hist.iter().sum();
    let mut cdf = [0usize; 256];
    let mut running = 0;
    for (i, &h) in hist.iter().enumerate() {
        running += h;
        cdf[i] = running;
    }
    let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);
    let mut lut = [0u8; 256];
    if total == cdf_min {
        // Only one gray level is present: nothing to equalize.
        for (i, l) in lut.iter_mut().enumerate() {
            *l = i as u8;
        }
        return lut;
    }
    for i in 0..256 {
        let scaled = (cdf[i].saturating_sub(cdf_min)) as f64 / (total - cdf_min) as f64;
        lut[i] = (255.0 * scaled).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Equalizes the histogram of an 8-bit grayscale image.
pub fn histogram_equalization(image: &Array2<u8>) -> Result<Array2<u8>, ErrorsJSL> {
    if image.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Image must be non-empty"));
    }
    let lut = equalization_lut(&histogram(image));
    Ok(image.mapv(|v| lut[v as usize]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// A low-contrast image: values confined to 100..=140.
    fn low_contrast(seed: &mut u64) -> Array2<u8> {
        Array2::from_shape_fn((64, 64), |_| (100.0 + 40.0 * lcg(seed)) as u8)
    }

    #[test]
    fn stretches_a_narrow_range_to_the_full_range() {
        let mut seed = 5;
        let img = low_contrast(&mut seed);
        let out = histogram_equalization(&img).unwrap();
        assert_eq!((*out.iter().min().unwrap(), *out.iter().max().unwrap()), (0, 255));
        assert!(*img.iter().max().unwrap() < 141);
    }

    #[test]
    fn output_cdf_is_close_to_linear() {
        let mut seed = 6;
        let out = histogram_equalization(&low_contrast(&mut seed)).unwrap();
        let h = histogram(&out);
        let n = out.len() as f64;
        let mut running = 0usize;
        for v in 0..256 {
            running += h[v];
            // A 41-level image cannot be perfectly uniform; the CDF still hugs the diagonal.
            assert!((running as f64 / n - (v as f64 + 1.0) / 256.0).abs() < 0.04, "level {v}");
        }
    }

    #[test]
    fn the_mapping_is_monotonic_and_preserves_order() {
        let mut seed = 7;
        let img = Array2::from_shape_fn((50, 50), |_| (lcg(&mut seed).powi(3) * 255.0) as u8);
        let lut = equalization_lut(&histogram(&img));
        assert!(lut.windows(2).all(|w| w[1] >= w[0]));
        let out = histogram_equalization(&img).unwrap();
        for (a, b) in img.iter().zip(out.iter()).take(500) {
            assert_eq!(*b, lut[*a as usize]);
        }
    }

    #[test]
    fn a_hand_computed_example() {
        // Levels 0, 1, 2, 3 with counts 1, 1, 2, 4 (N = 8): cdf = 1, 2, 4, 8; cdf_min = 1.
        let img = Array2::from_shape_vec((2, 4), vec![0u8, 1, 2, 2, 3, 3, 3, 3]).unwrap();
        let out = histogram_equalization(&img).unwrap();
        // 255 * (cdf - 1) / 7 = 0, 36.4, 109.3, 255
        assert_eq!(out.iter().cloned().collect::<Vec<_>>(), vec![0, 36, 109, 109, 255, 255, 255, 255]);
    }

    #[test]
    fn single_valued_and_full_range_images_are_unchanged() {
        let flat = Array2::from_elem((5, 5), 77u8);
        assert_eq!(histogram_equalization(&flat).unwrap(), flat);
        let ramp = Array2::from_shape_fn((16, 16), |(r, c)| (r * 16 + c) as u8);
        assert_eq!(histogram_equalization(&ramp).unwrap(), ramp, "an already-uniform histogram is a fixed point");
        assert!(histogram_equalization(&Array2::<u8>::zeros((0, 4))).is_err());
    }
}
