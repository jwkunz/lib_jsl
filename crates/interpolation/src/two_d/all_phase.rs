//! All-phase interpolation on a uniform grid.

use super::{Interpolation2D, SeparableResampler, sinc, uniform_axis, validate_grid};
use lib_jsl_core::*;

/// All-phase interpolation of order `n`: the result is the average, over all `n` alignments of an
/// `n`-sample window that contains the query point, of the sinc interpolation computed inside that
/// window. A sample at distance `d` from the query lies in `n - |d|` of those windows, so the
/// averaged kernel is `sinc(d) (1 - |d|/n)` for `|d| < n` (a triangular taper, the convolution of
/// two rectangular windows). Compared with a hard truncation of the sinc this suppresses the
/// ringing caused by window edges. Applied separably; weights are renormalised along each axis.
#[derive(Debug, Clone)]
pub struct AllPhase2D {
    x0: f64,
    dx: f64,
    y0: f64,
    dy: f64,
    n: usize,
    z: R2D,
}

impl AllPhase2D {
    /// `n` is the window length per axis (e.g. 4 to 8); axes must be uniformly spaced.
    pub fn new(x: VR1D, y: VR1D, z: VR2D, n: usize) -> Result<Self, ErrorsJSL> {
        validate_grid(x, y, z, 2)?;
        if n < 2 {
            return Err(ErrorsJSL::InvalidInputRange("window length must be at least 2"));
        }
        let (x0, dx) = uniform_axis(x)?;
        let (y0, dy) = uniform_axis(y)?;
        Ok(AllPhase2D { x0, dx, y0, dy, n, z: z.to_owned() })
    }
}

impl Interpolation2D for AllPhase2D {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        let n = self.n as f64;
        let kernel = |d: f64| if d.abs() < n { sinc(d) * (1.0 - d.abs() / n) } else { 0.0 };
        let r = SeparableResampler { x0: self.x0, dx: self.dx, y0: self.y0, dy: self.dy, support: self.n as isize, kernel: &kernel };
        Ok(r.eval(&self.z, x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_through_samples_and_constants() {
        let ax = R1D::from_iter((0..16).map(|k| k as f64));
        let z = R2D::from_shape_fn((16, 16), |(i, j)| ((i * i + 3 * j) % 7) as f64);
        let a = AllPhase2D::new(ax.view(), ax.view(), z.view(), 6).unwrap();
        for i in [0, 5, 15] {
            for j in [0, 7, 15] {
                assert!((a.interpolate_at(i as f64, j as f64).unwrap() - z[[i, j]]).abs() < 1e-12);
            }
        }
        let c = AllPhase2D::new(ax.view(), ax.view(), R2D::from_elem((16, 16), -2.0).view(), 4).unwrap();
        assert!((c.interpolate_at(7.3, 4.9).unwrap() + 2.0).abs() < 1e-12);
    }

    #[test]
    fn reconstructs_a_lowpass_signal_between_samples() {
        let ax = R1D::from_iter((0..48).map(|k| k as f64 * 0.2));
        let f = |a: f64, b: f64| (1.2 * a).sin() + 0.5 * (0.8 * b).cos();
        let z = R2D::from_shape_fn((48, 48), |(i, j)| f(ax[i], ax[j]));
        let a = AllPhase2D::new(ax.view(), ax.view(), z.view(), 8).unwrap();
        for &(px, py) in &[(4.33, 5.01), (6.12, 3.47), (5.0, 4.5)] {
            assert!((a.interpolate_at(px, py).unwrap() - f(px, py)).abs() < 2e-2, "{px},{py}");
        }
    }
}
