//! Trilinear interpolation on a 3D rectilinear grid.

use super::{Interpolation3D, validate_grid_3d};
use crate::two_d::find_cell;
use lib_jsl_core::*;
use ndarray::Array3;

/// Linear in each coordinate within a cell; edge cells extend linearly outside the grid.
#[derive(Debug, Clone)]
pub struct Trilinear {
    axes: [R1D; 3],
    v: Array3<f64>,
}

impl Trilinear {
    pub fn new(x: VR1D, y: VR1D, z: VR1D, v: &Array3<f64>) -> Result<Self, ErrorsJSL> {
        validate_grid_3d([x, y, z], v, 2)?;
        Ok(Trilinear { axes: [x.to_owned(), y.to_owned(), z.to_owned()], v: v.clone() })
    }
}

impl Interpolation3D for Trilinear {
    fn interpolate_at(&self, x: f64, y: f64, z: f64) -> Result<f64, ErrorsJSL> {
        let p = [x, y, z];
        let mut idx = [0usize; 3];
        let mut t = [0.0; 3];
        for d in 0..3 {
            let ax = self.axes[d].as_slice().expect("contiguous");
            idx[d] = find_cell(ax, p[d]);
            t[d] = (p[d] - ax[idx[d]]) / (ax[idx[d] + 1] - ax[idx[d]]);
        }
        let mut sum = 0.0;
        for corner in 0..8usize {
            let (a, b, c) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let w = [if a == 1 { t[0] } else { 1.0 - t[0] }, if b == 1 { t[1] } else { 1.0 - t[1] }, if c == 1 { t[2] } else { 1.0 - t[2] }];
            sum += w[0] * w[1] * w[2] * self.v[[idx[0] + a, idx[1] + b, idx[2] + c]];
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_for_trilinear_functions() {
        let x = R1D::from_vec(vec![0.0, 1.0, 2.5]);
        let y = R1D::from_vec(vec![-1.0, 0.5, 2.0, 3.0]);
        let z = R1D::from_vec(vec![0.0, 2.0]);
        let f = |a: f64, b: f64, c: f64| 1.0 + a - 2.0 * b + 3.0 * c + a * b * c - 0.5 * b * c;
        let v = Array3::from_shape_fn((3, 4, 2), |(i, j, k)| f(x[i], y[j], z[k]));
        let t = Trilinear::new(x.view(), y.view(), z.view(), &v).unwrap();
        for &(a, b, c) in &[(0.3, 0.0, 1.0), (2.0, 2.6, 0.2), (3.0, 4.0, 3.0), (-1.0, -2.0, -1.0)] {
            assert!((t.interpolate_at(a, b, c).unwrap() - f(a, b, c)).abs() < 1e-11, "{a},{b},{c}");
        }
    }
}
