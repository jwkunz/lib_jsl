//! Cardinal cubic basis interpolation on a rectilinear grid.

use super::{Interpolation2D, find_cell, hermite_basis, validate_grid};
use lib_jsl_core::*;

/// Tensor-product cardinal (Catmull-Rom family) cubic: the 1D cardinal interpolant of
/// [`crate::one_d::cubic_basis`] is applied along `y` in four neighbouring rows and then along `x`
/// to the four results. It touches a 4×4 neighbourhood, is C¹, exact for bilinear data, and
/// `tension` in `[0, 1]` scales node slopes by `1 - tension` (0 = Catmull-Rom).
#[derive(Debug, Clone)]
pub struct CubicBasis2D {
    x: R1D,
    y: R1D,
    z: R2D,
    tension: f64,
}

/// Hermite evaluation at `t` between two values with the given node slopes and cell width `h`.
fn hermite(t: f64, h: f64, v0: f64, m0: f64, v1: f64, m1: f64) -> f64 {
    let b = hermite_basis(t);
    b[0] * v0 + b[1] * h * m0 + b[2] * v1 + b[3] * h * m1
}

impl CubicBasis2D {
    pub fn new(x: VR1D, y: VR1D, z: VR2D, tension: f64) -> Result<Self, ErrorsJSL> {
        validate_grid(x, y, z, 2)?;
        Ok(CubicBasis2D { x: x.to_owned(), y: y.to_owned(), z: z.to_owned(), tension })
    }
}

impl Interpolation2D for CubicBasis2D {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        let (xs, ys) = (self.x.as_slice().expect("contiguous"), self.y.as_slice().expect("contiguous"));
        let (nx, ny) = self.z.dim();
        let (i, j) = (find_cell(xs, x), find_cell(ys, y));
        let c = 1.0 - self.tension;
        let (tx, ty) = ((x - xs[i]) / (xs[i + 1] - xs[i]), (y - ys[j]) / (ys[j + 1] - ys[j]));
        // Row interpolants along y, for the rows that feed the x slopes (clamped at the edges).
        let along_y = |r: usize| {
            let slope = |k: usize| {
                let (lo, hi) = (k.saturating_sub(1), (k + 1).min(ny - 1));
                c * (self.z[[r, hi]] - self.z[[r, lo]]) / (ys[hi] - ys[lo])
            };
            hermite(ty, ys[j + 1] - ys[j], self.z[[r, j]], slope(j), self.z[[r, j + 1]], slope(j + 1))
        };
        let slope_x = |k: usize| {
            let (lo, hi) = (k.saturating_sub(1), (k + 1).min(nx - 1));
            c * (along_y(hi) - along_y(lo)) / (xs[hi] - xs[lo])
        };
        Ok(hermite(tx, xs[i + 1] - xs[i], along_y(i), slope_x(i), along_y(i + 1), slope_x(i + 1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_bilinear_data_and_smooth_accuracy() {
        let x = R1D::from_vec(vec![0.0, 0.5, 1.5, 2.0, 3.0]);
        let y = R1D::from_vec(vec![0.0, 1.0, 1.5, 3.0]);
        let f = |a: f64, b: f64| 1.0 + a - 2.0 * b + 0.25 * a * b;
        let z = R2D::from_shape_fn((5, 4), |(i, j)| f(x[i], y[j]));
        let s = CubicBasis2D::new(x.view(), y.view(), z.view(), 0.0).unwrap();
        for &(px, py) in &[(0.2, 0.3), (1.7, 1.2), (2.9, 2.8), (0.0, 0.0), (3.0, 3.0)] {
            assert!((s.interpolate_at(px, py).unwrap() - f(px, py)).abs() < 1e-12, "{px},{py}");
        }
        let ax = R1D::from_iter((0..31).map(|k| k as f64 * 0.1));
        let g = |a: f64, b: f64| (a + 0.3).sin() * (0.7 * b).cos();
        let zz = R2D::from_shape_fn((31, 31), |(i, j)| g(ax[i], ax[j]));
        let t = CubicBasis2D::new(ax.view(), ax.view(), zz.view(), 0.0).unwrap();
        assert!((t.interpolate_at(1.234, 2.345).unwrap() - g(1.234, 2.345)).abs() < 1e-3);
        assert!((t.interpolate_at(1.0, 2.0).unwrap() - g(1.0, 2.0)).abs() < 1e-13);
    }

    #[test]
    fn full_tension_is_flat_at_nodes() {
        let x = R1D::from_vec(vec![0.0, 1.0, 2.0]);
        let z = R2D::from_shape_fn((3, 3), |(i, j)| (i + 2 * j) as f64);
        let s = CubicBasis2D::new(x.view(), x.view(), z.view(), 1.0).unwrap();
        // With zero slopes the surface is a smoothstep between nodes: midpoint is the average.
        assert!((s.interpolate_at(0.5, 0.0).unwrap() - 0.5).abs() < 1e-12);
    }
}
