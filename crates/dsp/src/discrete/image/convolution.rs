//! 2D convolution and correlation with border handling, the shared basis of the image filters.
//!
//! Images are `Array2<f64>` indexed `(row, column)`. Kernels have odd dimensions and are centred
//! on the output pixel. Pixels outside the image are supplied according to a [`Border`] mode.
//! Direct (spatial) evaluation is used; for large kernels see
//! [`fft_convolve_2d`](crate::discrete::fft_convolution::fft_convolve_2d).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// How to supply pixels beyond the image edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// Outside pixels are 0.
    Zero,
    /// The edge pixel is repeated (`a a a | a b c | c c c`).
    Replicate,
    /// The image is mirrored about its edge (`c b a | a b c | c b a`).
    Reflect,
}

/// Maps a possibly out-of-range index into `0..n` for a border mode (`None` means "zero").
pub fn border_index(i: isize, n: usize, border: Border) -> Option<usize> {
    let n_i = n as isize;
    if (0..n_i).contains(&i) {
        return Some(i as usize);
    }
    match border {
        Border::Zero => None,
        Border::Replicate => Some(i.clamp(0, n_i - 1) as usize),
        Border::Reflect => {
            let period = 2 * n_i;
            let m = i.rem_euclid(period);
            Some(if m < n_i { m as usize } else { (period - 1 - m) as usize })
        }
    }
}

fn check_kernel(kernel: &Array2<f64>) -> Result<(), ErrorsJSL> {
    let (kr, kc) = kernel.dim();
    if kr == 0 || kc == 0 || kr % 2 == 0 || kc % 2 == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Kernel dimensions must be odd and nonzero"));
    }
    Ok(())
}

/// 2D cross-correlation (the kernel is *not* flipped): `out[r, c] = Σ image[r+i, c+j] · kernel[i+hr, j+hc]`.
pub fn correlate2d(image: &Array2<f64>, kernel: &Array2<f64>, border: Border) -> Result<Array2<f64>, ErrorsJSL> {
    check_kernel(kernel)?;
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Image must be non-empty"));
    }
    let (kr, kc) = kernel.dim();
    let (hr, hc) = ((kr / 2) as isize, (kc / 2) as isize);
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| {
        let mut acc = 0.0;
        for i in 0..kr {
            let Some(rr) = border_index(r as isize + i as isize - hr, rows, border) else { continue };
            for j in 0..kc {
                if let Some(cc) = border_index(c as isize + j as isize - hc, cols, border) {
                    acc += image[(rr, cc)] * kernel[(i, j)];
                }
            }
        }
        acc
    }))
}

/// 2D convolution (the kernel is flipped in both axes), output the same size as the input.
pub fn convolve2d(image: &Array2<f64>, kernel: &Array2<f64>, border: Border) -> Result<Array2<f64>, ErrorsJSL> {
    check_kernel(kernel)?;
    let (kr, kc) = kernel.dim();
    let flipped = Array2::from_shape_fn((kr, kc), |(i, j)| kernel[(kr - 1 - i, kc - 1 - j)]);
    correlate2d(image, &flipped, border)
}

/// Separable filtering: convolves every row with `row_kernel` and then every column with
/// `col_kernel` (both odd length, symmetric or not; true convolution).
pub fn separable_convolve(image: &Array2<f64>, row_kernel: &[f64], col_kernel: &[f64], border: Border) -> Result<Array2<f64>, ErrorsJSL> {
    if row_kernel.len() % 2 == 0 || col_kernel.len() % 2 == 0 || row_kernel.is_empty() || col_kernel.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Kernel lengths must be odd and nonzero"));
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Image must be non-empty"));
    }
    let (hr, hc) = ((col_kernel.len() / 2) as isize, (row_kernel.len() / 2) as isize);
    // Horizontal pass: convolution flips the kernel, so tap j pairs with offset (hc - j).
    let horizontal = Array2::from_shape_fn((rows, cols), |(r, c)| {
        row_kernel
            .iter()
            .enumerate()
            .filter_map(|(j, k)| border_index(c as isize + hc - j as isize, cols, border).map(|cc| image[(r, cc)] * k))
            .sum::<f64>()
    });
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| {
        col_kernel
            .iter()
            .enumerate()
            .filter_map(|(i, k)| border_index(r as isize + hr - i as isize, rows, border).map(|rr| horizontal[(rr, c)] * k))
            .sum::<f64>()
    }))
}

/// A normalised 1D Gaussian kernel with standard deviation `sigma` (in pixels), truncated at
/// `truncate` standard deviations (4 is conventional) and always of odd length.
pub fn gaussian_kernel_1d(sigma: f64, truncate: f64) -> Result<Vec<f64>, ErrorsJSL> {
    if !(sigma > 0.0) || !sigma.is_finite() || !(truncate > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("sigma and truncate must be positive and finite"));
    }
    let radius = (truncate * sigma).ceil().max(1.0) as isize;
    let mut k: Vec<f64> = (-radius..=radius).map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp()).collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    #[test]
    fn border_modes_map_indices() {
        assert_eq!(border_index(-1, 5, Border::Replicate), Some(0));
        assert_eq!(border_index(7, 5, Border::Replicate), Some(4));
        assert_eq!(border_index(-1, 5, Border::Reflect), Some(0));
        assert_eq!(border_index(-2, 5, Border::Reflect), Some(1));
        assert_eq!(border_index(5, 5, Border::Reflect), Some(4));
        assert_eq!(border_index(6, 5, Border::Reflect), Some(3));
        assert_eq!(border_index(-11, 5, Border::Reflect), Some(0), "reflection is periodic with period 2n");
        assert_eq!(border_index(-6, 5, Border::Reflect), Some(4));
        assert_eq!(border_index(-1, 5, Border::Zero), None);
        assert_eq!(border_index(3, 5, Border::Zero), Some(3));
    }

    #[test]
    fn identity_and_shift_kernels() {
        let mut seed = 1;
        let img = Array2::from_shape_fn((6, 7), |_| lcg(&mut seed));
        let mut id = Array2::<f64>::zeros((3, 3));
        id[(1, 1)] = 1.0;
        assert_eq!(convolve2d(&img, &id, Border::Replicate).unwrap(), img);
        // A kernel with a one at the top-left: correlation reads the pixel up-left of each output.
        let mut tl = Array2::<f64>::zeros((3, 3));
        tl[(0, 0)] = 1.0;
        let c = correlate2d(&img, &tl, Border::Zero).unwrap();
        assert_eq!(c[(3, 4)], img[(2, 3)]);
        assert_eq!(c[(0, 4)], 0.0);
        // Convolution flips the kernel, so the same kernel reads the pixel down-right.
        let v = convolve2d(&img, &tl, Border::Zero).unwrap();
        assert_eq!(v[(3, 4)], img[(4, 5)]);
    }

    #[test]
    fn separable_matches_full_two_dimensional_convolution() {
        let mut seed = 2;
        let img = Array2::from_shape_fn((9, 11), |_| lcg(&mut seed));
        let (a, b) = ([0.2, 0.5, 0.3], [0.1, 0.2, 0.4, 0.2, 0.1]);
        let kernel = Array2::from_shape_fn((3, 5), |(i, j)| a[i] * b[j]);
        for border in [Border::Zero, Border::Replicate, Border::Reflect] {
            let direct = convolve2d(&img, &kernel, border).unwrap();
            let sep = separable_convolve(&img, &b, &a, border).unwrap();
            assert!(direct.iter().zip(sep.iter()).all(|(x, y)| (x - y).abs() < 1e-12), "{border:?}");
        }
    }

    #[test]
    fn gaussian_kernel_properties() {
        let k = gaussian_kernel_1d(2.0, 4.0).unwrap();
        assert_eq!(k.len(), 17);
        assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert!((0..8).all(|i| (k[i] - k[16 - i]).abs() < 1e-15), "symmetric");
        let var: f64 = k.iter().enumerate().map(|(i, v)| (i as f64 - 8.0).powi(2) * v).sum();
        assert!((var - 4.0).abs() < 0.01, "variance {var}");
        assert!(k[8] > k[7] && k[7] > k[6]);
        assert!(gaussian_kernel_1d(0.0, 4.0).is_err() && gaussian_kernel_1d(1.0, 0.0).is_err());
        assert_eq!(gaussian_kernel_1d(0.1, 4.0).unwrap().len(), 3, "tiny sigma still gives at least 3 taps");
    }

    #[test]
    fn rejects_bad_kernels_and_empty_images() {
        let img = Array2::<f64>::ones((4, 4));
        assert!(convolve2d(&img, &Array2::ones((2, 3)), Border::Zero).is_err());
        assert!(correlate2d(&img, &Array2::ones((0, 3)), Border::Zero).is_err());
        assert!(correlate2d(&Array2::<f64>::zeros((0, 4)), &Array2::ones((3, 3)), Border::Zero).is_err());
        assert!(separable_convolve(&img, &[1.0, 1.0], &[1.0], Border::Zero).is_err());
    }
}
