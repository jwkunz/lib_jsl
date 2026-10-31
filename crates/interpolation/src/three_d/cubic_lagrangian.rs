//! Tricubic Lagrange interpolation on a 3D rectilinear grid.

use super::{Interpolation3D, validate_grid_3d};
use lib_jsl_core::*;
use ndarray::Array3;

/// Separable 4-point (cubic) Lagrange interpolation: the 4×4×4 block of nodes nearest the query
/// (two on each side, shifted at the boundary) is interpolated exactly by tricubic polynomials, so
/// any polynomial of degree ≤ 3 in each coordinate is reproduced. Non-uniform axes are supported;
/// the result is continuous but not C¹ across cells.
#[derive(Debug, Clone)]
pub struct CubicLagrangian3D {
    axes: [R1D; 3],
    v: Array3<f64>,
}

/// First node of the 4-point stencil around `p` and its Lagrange weights.
fn stencil(axis: &R1D, p: f64) -> (usize, [f64; 4]) {
    let ax = axis.as_slice().expect("contiguous");
    let cell = crate::two_d::find_cell(ax, p);
    let start = cell.saturating_sub(1).min(ax.len() - 4);
    let mut w = [1.0; 4];
    for (a, wa) in w.iter_mut().enumerate() {
        for b in 0..4 {
            if a != b {
                *wa *= (p - ax[start + b]) / (ax[start + a] - ax[start + b]);
            }
        }
    }
    (start, w)
}

impl CubicLagrangian3D {
    pub fn new(x: VR1D, y: VR1D, z: VR1D, v: &Array3<f64>) -> Result<Self, ErrorsJSL> {
        validate_grid_3d([x, y, z], v, 4)?;
        Ok(CubicLagrangian3D { axes: [x.to_owned(), y.to_owned(), z.to_owned()], v: v.clone() })
    }
}

impl Interpolation3D for CubicLagrangian3D {
    fn interpolate_at(&self, x: f64, y: f64, z: f64) -> Result<f64, ErrorsJSL> {
        let (i0, wx) = stencil(&self.axes[0], x);
        let (j0, wy) = stencil(&self.axes[1], y);
        let (k0, wz) = stencil(&self.axes[2], z);
        let mut sum = 0.0;
        for (a, wa) in wx.iter().enumerate() {
            for (b, wb) in wy.iter().enumerate() {
                for (c, wc) in wz.iter().enumerate() {
                    sum += wa * wb * wc * self.v[[i0 + a, j0 + b, k0 + c]];
                }
            }
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_tricubic_polynomials_on_nonuniform_axes() {
        let x = R1D::from_vec(vec![0.0, 0.4, 1.1, 1.5, 2.6, 3.0]);
        let y = R1D::from_vec(vec![-1.0, 0.0, 0.7, 1.9, 2.2]);
        let z = R1D::from_vec(vec![0.0, 1.0, 1.4, 2.5, 4.0]);
        let f = |a: f64, b: f64, c: f64| a.powi(3) - 2.0 * a * b * b + b.powi(3) * c - c.powi(3) + 0.5 * a * b * c + 1.0;
        let v = Array3::from_shape_fn((6, 5, 5), |(i, j, k)| f(x[i], y[j], z[k]));
        let l = CubicLagrangian3D::new(x.view(), y.view(), z.view(), &v).unwrap();
        for &(a, b, c) in &[(0.2, -0.5, 0.3), (1.3, 1.0, 2.0), (2.9, 2.1, 3.9), (0.0, 2.2, 4.0)] {
            assert!((l.interpolate_at(a, b, c).unwrap() - f(a, b, c)).abs() < 1e-9, "{a},{b},{c}");
        }
        assert!(CubicLagrangian3D::new(x.view(), y.view(), R1D::zeros(3).view(), &Array3::zeros((6, 5, 3))).is_err());
    }
}
