//! Trigonometric interpolation of doubly periodic gridded data.

use super::Interpolation2D;
use lib_jsl_core::*;
use num_complex::Complex;
use std::f64::consts::PI;

/// The band-limited doubly periodic trigonometric interpolant through `z[[i, j]] = f(x0 + i px/nx,
/// y0 + j py/ny)`. Coefficients come from a separable direct DFT; each evaluation costs O(nx·ny).
/// Nyquist terms (even sizes) are handled so the interpolant stays real.
#[derive(Debug, Clone)]
pub struct FourierInterpolator2D {
    x0: f64,
    y0: f64,
    px: f64,
    py: f64,
    coefficients: Vec<Vec<Complex<f64>>>,
}

fn dft_matrix(n: usize) -> Vec<Vec<Complex<f64>>> {
    (0..n).map(|k| (0..n).map(|j| Complex::from_polar(1.0, -2.0 * PI * ((k * j) % n) as f64 / n as f64)).collect()).collect()
}

/// Signed frequency of FFT bin `k` of `n`; the Nyquist bin keeps `+n/2`.
fn frequency(k: usize, n: usize) -> f64 {
    if k <= n / 2 { k as f64 } else { k as f64 - n as f64 }
}

impl FourierInterpolator2D {
    /// `px`, `py` are the periods along each axis, `(x0, y0)` the position of sample `[0, 0]`.
    pub fn new(z: VR2D, x0: f64, y0: f64, px: f64, py: f64) -> Result<Self, ErrorsJSL> {
        let (nx, ny) = z.dim();
        if nx == 0 || ny == 0 || px <= 0.0 || py <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("need samples and positive periods"));
        }
        let (fx, fy) = (dft_matrix(nx), dft_matrix(ny));
        // Rows (over j) first: t[i][l] = Σ_j z[i][j] e^{-2πi jl/ny}; then columns.
        let t: Vec<Vec<Complex<f64>>> =
            (0..nx).map(|i| (0..ny).map(|l| (0..ny).map(|j| fy[l][j] * z[[i, j]]).sum()).collect()).collect();
        let scale = 1.0 / (nx * ny) as f64;
        let coefficients = (0..nx)
            .map(|k| (0..ny).map(|l| (0..nx).map(|i| fx[k][i] * t[i][l]).sum::<Complex<f64>>() * scale).collect())
            .collect();
        Ok(FourierInterpolator2D { x0, y0, px, py, coefficients })
    }
}

impl Interpolation2D for FourierInterpolator2D {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        let (nx, ny) = (self.coefficients.len(), self.coefficients[0].len());
        let tx = 2.0 * PI * (x - self.x0) / self.px;
        let ty = 2.0 * PI * (y - self.y0) / self.py;
        // Pre-compute the 1D phasors, then Re Σ C[k][l] e^{i(f_k tx + f_l ty)}.
        let ex: Vec<Complex<f64>> = (0..nx).map(|k| Complex::from_polar(1.0, frequency(k, nx) * tx)).collect();
        let ey: Vec<Complex<f64>> = (0..ny).map(|l| Complex::from_polar(1.0, frequency(l, ny) * ty)).collect();
        let mut sum = Complex::new(0.0, 0.0);
        for (coef_row, phase) in self.coefficients.iter().zip(&ex) {
            let row: Complex<f64> = coef_row.iter().zip(&ey).map(|(c, e)| c * e).sum();
            sum += row * phase;
        }
        Ok(sum.re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_bandlimited_functions_off_grid() {
        let (nx, ny) = (12, 9);
        let (px, py) = (3.0, 2.0);
        let f = |a: f64, b: f64| {
            1.0 + (2.0 * PI * a / px).cos() * (2.0 * PI * b / py).sin() + 0.4 * (4.0 * PI * a / px + 2.0 * PI * b / py).cos()
        };
        let z = R2D::from_shape_fn((nx, ny), |(i, j)| f(0.2 + i as f64 * px / nx as f64, -0.1 + j as f64 * py / ny as f64));
        let fi = FourierInterpolator2D::new(z.view(), 0.2, -0.1, px, py).unwrap();
        for &(a, b) in &[(0.37, 0.52), (1.9, -0.4), (2.8, 1.7), (-1.0, 3.3)] {
            assert!((fi.interpolate_at(a, b).unwrap() - f(a, b)).abs() < 1e-11, "{a},{b}");
        }
    }

    #[test]
    fn exact_at_samples_for_arbitrary_data_even_sizes() {
        let z = R2D::from_shape_fn((8, 6), |(i, j)| ((i * 5 + j * 11) % 7) as f64 - 3.0);
        let fi = FourierInterpolator2D::new(z.view(), 0.0, 0.0, 8.0, 6.0).unwrap();
        for i in 0..8 {
            for j in 0..6 {
                assert!((fi.interpolate_at(i as f64, j as f64).unwrap() - z[[i, j]]).abs() < 1e-11);
            }
        }
        assert!(FourierInterpolator2D::new(z.view(), 0.0, 0.0, 0.0, 1.0).is_err());
    }
}
