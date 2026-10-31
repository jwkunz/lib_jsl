//! Lanczos resampling on a uniform grid.

use super::{Interpolation2D, SeparableResampler, sinc, uniform_axis, validate_grid};
use lib_jsl_core::*;

/// Separable Lanczos-`a` interpolation: kernel `sinc(d) sinc(d / a)` for `|d| < a`, a windowed sinc
/// that preserves constants (weights are renormalised) and passes through the grid samples.
#[derive(Debug, Clone)]
pub struct Lanczos2D {
    x0: f64,
    dx: f64,
    y0: f64,
    dy: f64,
    a: usize,
    z: R2D,
}

impl Lanczos2D {
    /// `a` is the lobe count (2 or 3 are typical); both axes must be uniformly spaced.
    pub fn new(x: VR1D, y: VR1D, z: VR2D, a: usize) -> Result<Self, ErrorsJSL> {
        validate_grid(x, y, z, 2)?;
        if a == 0 {
            return Err(ErrorsJSL::InvalidInputRange("lobe count must be positive"));
        }
        let (x0, dx) = uniform_axis(x)?;
        let (y0, dy) = uniform_axis(y)?;
        Ok(Lanczos2D { x0, dx, y0, dy, a, z: z.to_owned() })
    }
}

impl Interpolation2D for Lanczos2D {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        let a = self.a as f64;
        let kernel = |d: f64| if d.abs() < a { sinc(d) * sinc(d / a) } else { 0.0 };
        let r = SeparableResampler { x0: self.x0, dx: self.dx, y0: self.y0, dy: self.dy, support: self.a as isize, kernel: &kernel };
        Ok(r.eval(&self.z, x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_samples_and_preserves_constants() {
        let x = R1D::from_iter((0..12).map(|k| 1.0 + 0.5 * k as f64));
        let y = R1D::from_iter((0..10).map(|k| -2.0 + 0.25 * k as f64));
        let z = R2D::from_shape_fn((12, 10), |(i, j)| ((i * 7 + j * 3) % 5) as f64);
        let l = Lanczos2D::new(x.view(), y.view(), z.view(), 3).unwrap();
        for i in 0..12 {
            for j in 0..10 {
                assert!((l.interpolate_at(x[i], y[j]).unwrap() - z[[i, j]]).abs() < 1e-12);
            }
        }
        let c = Lanczos2D::new(x.view(), y.view(), R2D::from_elem((12, 10), 4.5).view(), 2).unwrap();
        assert!((c.interpolate_at(3.17, -1.1).unwrap() - 4.5).abs() < 1e-12);
    }

    #[test]
    fn accurate_on_a_smooth_oversampled_signal() {
        let ax = R1D::from_iter((0..40).map(|k| k as f64 * 0.1));
        let f = |a: f64, b: f64| (1.5 * a).sin() * (1.0 * b).cos();
        let z = R2D::from_shape_fn((40, 40), |(i, j)| f(ax[i], ax[j]));
        let l = Lanczos2D::new(ax.view(), ax.view(), z.view(), 3).unwrap();
        for &(px, py) in &[(1.23, 2.34), (2.05, 1.77), (3.01, 0.55)] {
            assert!((l.interpolate_at(px, py).unwrap() - f(px, py)).abs() < 2e-3, "{px},{py}");
        }
    }

    #[test]
    fn rejects_nonuniform_axes() {
        let x = R1D::from_vec(vec![0.0, 1.0, 3.0]);
        assert!(Lanczos2D::new(x.view(), x.view(), R2D::zeros((3, 3)).view(), 2).is_err());
    }
}
