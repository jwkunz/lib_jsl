//! Bicubic (C¹) interpolation on a rectilinear grid.

use super::{Interpolation2D, find_cell, hermite_basis, validate_grid};
use lib_jsl_core::*;

/// A piecewise bicubic Hermite surface. Each node carries `f`, `∂f/∂x`, `∂f/∂y` and `∂²f/∂x∂y`,
/// estimated by (non-uniform) central differences with one-sided differences at the boundary, and
/// the surface in each cell is the tensor-product cubic Hermite patch. Exact for bilinear data and
/// continuous in value and first derivatives across cells.
#[derive(Debug, Clone)]
pub struct Bicubic {
    x: R1D,
    y: R1D,
    z: R2D,
    zx: R2D,
    zy: R2D,
    zxy: R2D,
}

/// Derivative of `v` along an axis by clamped central differences.
fn diff(axis: &R1D, n: usize, i: usize, value: impl Fn(usize) -> f64) -> f64 {
    let (lo, hi) = (i.saturating_sub(1), (i + 1).min(n - 1));
    (value(hi) - value(lo)) / (axis[hi] - axis[lo])
}

impl Bicubic {
    pub fn new(x: VR1D, y: VR1D, z: VR2D) -> Result<Self, ErrorsJSL> {
        validate_grid(x, y, z, 2)?;
        let (x, y, z) = (x.to_owned(), y.to_owned(), z.to_owned());
        let (nx, ny) = z.dim();
        let zx = R2D::from_shape_fn((nx, ny), |(i, j)| diff(&x, nx, i, |k| z[[k, j]]));
        let zy = R2D::from_shape_fn((nx, ny), |(i, j)| diff(&y, ny, j, |k| z[[i, k]]));
        let zxy = R2D::from_shape_fn((nx, ny), |(i, j)| diff(&y, ny, j, |k| zx[[i, k]]));
        Ok(Bicubic { x, y, z, zx, zy, zxy })
    }
}

impl Interpolation2D for Bicubic {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        let (xs, ys) = (self.x.as_slice().expect("contiguous"), self.y.as_slice().expect("contiguous"));
        let (i, j) = (find_cell(xs, x), find_cell(ys, y));
        let (hx, hy) = (xs[i + 1] - xs[i], ys[j + 1] - ys[j]);
        let bx = hermite_basis((x - xs[i]) / hx);
        let by = hermite_basis((y - ys[j]) / hy);
        // Basis order is (value at 0, slope at 0, value at 1, slope at 1); slopes scale by cell size.
        let sx = [1.0, hx, 1.0, hx];
        let sy = [1.0, hy, 1.0, hy];
        let mut total = 0.0;
        for (a, (di, kind_x)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            for (b, (dj, kind_y)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let node = [i + di, j + dj];
                let value = match (kind_x, kind_y) {
                    (0, 0) => self.z[node],
                    (1, 0) => self.zx[node],
                    (0, 1) => self.zy[node],
                    _ => self.zxy[node],
                };
                total += value * bx[a] * sx[a] * by[b] * sy[b];
            }
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_at_nodes_and_for_bilinear_data() {
        let x = R1D::from_vec(vec![0.0, 0.7, 1.5, 3.0, 3.5]);
        let y = R1D::from_vec(vec![-1.0, 0.0, 2.0, 2.5]);
        let f = |a: f64, b: f64| 2.0 - a + 4.0 * b + 0.3 * a * b;
        let z = R2D::from_shape_fn((5, 4), |(i, j)| f(x[i], y[j]));
        let s = Bicubic::new(x.view(), y.view(), z.view()).unwrap();
        for &(px, py) in &[(0.3, -0.5), (1.1, 1.0), (2.9, 2.2), (3.4, 0.3)] {
            assert!((s.interpolate_at(px, py).unwrap() - f(px, py)).abs() < 1e-12, "{px},{py}");
        }
        let zz = R2D::from_shape_fn((5, 4), |(i, j)| (x[i] * y[j]).sin());
        let t = Bicubic::new(x.view(), y.view(), zz.view()).unwrap();
        for i in 0..5 {
            for j in 0..4 {
                assert!((t.interpolate_at(x[i], y[j]).unwrap() - zz[[i, j]]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn smooth_function_is_accurate_on_a_fine_grid() {
        let n = 41;
        let ax = R1D::from_iter((0..n).map(|k| k as f64 * 0.05));
        let f = |a: f64, b: f64| (a * 1.3).sin() * (b + 0.4).cos();
        let z = R2D::from_shape_fn((n, n), |(i, j)| f(ax[i], ax[j]));
        let s = Bicubic::new(ax.view(), ax.view(), z.view()).unwrap();
        let mut worst: f64 = 0.0;
        for p in 0..30 {
            for q in 0..30 {
                let (px, py) = (0.03 + p as f64 * 0.0657, 0.04 + q as f64 * 0.0651);
                worst = worst.max((s.interpolate_at(px, py).unwrap() - f(px, py)).abs());
            }
        }
        assert!(worst < 2e-4, "worst {worst}");
    }
}
