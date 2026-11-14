//! Gaussian smoothing and derivative filters, and the discrete Laplacian, following
//! `scipy.ndimage` (`gaussian_filter` with derivative orders, `gaussian_laplace`,
//! `gaussian_gradient_magnitude`, `laplace`).
//!
//! Differentiating a noisy image amplifies its noise; differentiating a Gaussian-smoothed image
//! does not, and since differentiation and convolution commute, both steps are one convolution
//! with a derivative of the Gaussian. The Laplacian of a Gaussian responds to blobs of size about
//! `σ√2`; the gradient magnitude to edges at scale `σ`.
//!
//! See the book entry [Rank filters, derivative filters and geometric transforms](https://jwkunz.github.io/lib_jsl/book/image/rank-geometric.html).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

use super::boundary::{Mode, sample};

// ANCHOR: gaussian_kernel
/// SciPy's one-dimensional Gaussian kernel of standard deviation `sigma`, radius
/// `⌊truncate σ + ½⌋`, for derivative order 0, 1 or 2, already reversed for correlation: weights
/// `φ(x)` normalised to sum 1, multiplied by `x/σ²` (first derivative) or `x²/σ⁴ − 1/σ²` (second).
pub fn gaussian_kernel(sigma: f64, order: usize, truncate: f64) -> Result<Vec<f64>, ErrorsJSL> {
    if !(sigma > 0.0) || !(truncate > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("sigma and truncate must be positive"));
    }
    if order > 2 {
        return Err(ErrorsJSL::InvalidInputRange("derivative order must be 0, 1 or 2"));
    }
    let radius = (truncate * sigma + 0.5) as isize;
    let s2 = sigma * sigma;
    let phi: Vec<f64> = (-radius..=radius).map(|x| (-0.5 * (x * x) as f64 / s2).exp()).collect();
    let total: f64 = phi.iter().sum();
    Ok((-radius..=radius)
        .zip(phi)
        .map(|(x, p)| {
            let x = x as f64;
            let q = match order {
                0 => 1.0,
                1 => x / s2,
                _ => x * x / (s2 * s2) - 1.0 / s2,
            };
            q * p / total
        })
        .collect())
}
// ANCHOR_END: gaussian_kernel

/// Correlates every line of `image` along `axis` (0: down the columns, 1: along the rows) with an
/// odd-length kernel centred on the output pixel.
pub fn correlate1d(image: &Array2<f64>, weights: &[f64], axis: usize, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    if weights.len().is_multiple_of(2) || image.is_empty() || axis > 1 {
        return Err(ErrorsJSL::InvalidInputRange("odd kernel, non-empty image and axis 0 or 1 required"));
    }
    let r = (weights.len() / 2) as isize;
    Ok(Array2::from_shape_fn(image.dim(), |(i, j)| {
        weights
            .iter()
            .enumerate()
            .map(|(k, w)| {
                let d = k as isize - r;
                let v = if axis == 0 { sample(image, i as isize + d, j as isize, mode) } else { sample(image, i as isize, j as isize + d, mode) };
                w * v
            })
            .sum()
    }))
}

// ANCHOR: gaussian_filter
/// Gaussian filtering with standard deviations `sigma = (rows, columns)` and derivative orders
/// `order = (rows, columns)` (SciPy's `gaussian_filter`, `truncate = 4` by default). Order
/// `(0, 0)` smooths; `(0, 1)` estimates `∂/∂column` of the smoothed image; `(2, 0)` the second
/// derivative down the columns.
pub fn gaussian_filter(image: &Array2<f64>, sigma: (f64, f64), order: (usize, usize), mode: Mode, truncate: f64) -> Result<Array2<f64>, ErrorsJSL> {
    let rows = correlate1d(image, &gaussian_kernel(sigma.0, order.0, truncate)?, 0, mode)?;
    correlate1d(&rows, &gaussian_kernel(sigma.1, order.1, truncate)?, 1, mode)
}
// ANCHOR_END: gaussian_filter

/// The Laplacian of the Gaussian-smoothed image, `∂²/∂r² + ∂²/∂c²` (SciPy's `gaussian_laplace`).
pub fn gaussian_laplace(image: &Array2<f64>, sigma: f64, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    Ok(gaussian_filter(image, (sigma, sigma), (2, 0), mode, 4.0)? + gaussian_filter(image, (sigma, sigma), (0, 2), mode, 4.0)?)
}

/// The gradient magnitude of the Gaussian-smoothed image (SciPy's `gaussian_gradient_magnitude`).
pub fn gaussian_gradient_magnitude(image: &Array2<f64>, sigma: f64, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    let gr = gaussian_filter(image, (sigma, sigma), (1, 0), mode, 4.0)?;
    let gc = gaussian_filter(image, (sigma, sigma), (0, 1), mode, 4.0)?;
    Ok(Array2::from_shape_fn(image.dim(), |p| gr[p].hypot(gc[p])))
}

/// The discrete Laplacian, the sum of the second differences `[1, −2, 1]` along each axis
/// (SciPy's `laplace`).
pub fn laplace(image: &Array2<f64>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    let d2 = [1.0, -2.0, 1.0];
    Ok(correlate1d(image, &d2, 0, mode)? + correlate1d(image, &d2, 1, mode)?)
}

/// The sum along `axis` of a filter's output, a helper for checking mass conservation.
#[cfg(test)]
fn total(a: &Array2<f64>) -> f64 {
    a.sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_of_polynomials() {
        // On a quadratic surface the Gaussian derivatives are exact away from the borders.
        let img = Array2::from_shape_fn((41, 41), |(r, c)| 0.5 * (r as f64).powi(2) + 3.0 * c as f64);
        let dc = gaussian_filter(&img, (2.0, 2.0), (0, 1), Mode::Nearest, 4.0).unwrap();
        let drr = gaussian_filter(&img, (2.0, 2.0), (2, 0), Mode::Nearest, 4.0).unwrap();
        // Not exact: the sampled, truncated kernel's second moment is not exactly σ² (as in SciPy).
        assert!((dc[(20, 20)] - 3.0).abs() < 2e-3, "{}", dc[(20, 20)]);
        assert!((drr[(20, 20)] - 1.0).abs() < 3e-2, "{}", drr[(20, 20)]);
        let lap = laplace(&img, Mode::Nearest).unwrap();
        assert!((lap[(20, 20)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn smoothing_conserves_mass_with_wrapping() {
        let mut img = Array2::zeros((21, 21));
        img[(3, 17)] = 1.0;
        let s = gaussian_filter(&img, (1.5, 2.5), (0, 0), Mode::Wrap, 4.0).unwrap();
        assert!((total(&s) - 1.0).abs() < 1e-12);
    }
}
