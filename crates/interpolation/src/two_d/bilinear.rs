//! Bilinear interpolation on a rectilinear grid.

use super::{Interpolation2D, find_cell, validate_grid};
use lib_jsl_core::*;

/// Linear in each coordinate within a cell; continuous across cells. Outside the grid the edge
/// cells are extended linearly.
#[derive(Debug, Clone)]
pub struct Bilinear {
    x: R1D,
    y: R1D,
    z: R2D,
}

impl Bilinear {
    pub fn new(x: VR1D, y: VR1D, z: VR2D) -> Result<Self, ErrorsJSL> {
        validate_grid(x, y, z, 2)?;
        Ok(Bilinear { x: x.to_owned(), y: y.to_owned(), z: z.to_owned() })
    }
}

impl Interpolation2D for Bilinear {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        let (xs, ys) = (self.x.as_slice().expect("contiguous"), self.y.as_slice().expect("contiguous"));
        let (i, j) = (find_cell(xs, x), find_cell(ys, y));
        let tx = (x - xs[i]) / (xs[i + 1] - xs[i]);
        let ty = (y - ys[j]) / (ys[j + 1] - ys[j]);
        let z = &self.z;
        Ok((1.0 - tx) * ((1.0 - ty) * z[[i, j]] + ty * z[[i, j + 1]]) + tx * ((1.0 - ty) * z[[i + 1, j]] + ty * z[[i + 1, j + 1]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_for_bilinear_functions_everywhere() {
        let x = R1D::from_vec(vec![-1.0, 0.0, 0.5, 2.0]);
        let y = R1D::from_vec(vec![0.0, 1.0, 4.0]);
        let f = |a: f64, b: f64| 1.0 + 2.0 * a - 3.0 * b + 0.5 * a * b;
        let z = R2D::from_shape_fn((4, 3), |(i, j)| f(x[i], y[j]));
        let b = Bilinear::new(x.view(), y.view(), z.view()).unwrap();
        for &(px, py) in &[(-1.0, 0.0), (0.25, 0.5), (1.3, 3.1), (3.0, 5.0), (-2.0, -1.0)] {
            assert!((b.interpolate_at(px, py).unwrap() - f(px, py)).abs() < 1e-12);
        }
    }

    #[test]
    fn validates_shapes_and_axes() {
        let x = R1D::from_vec(vec![0.0, 1.0]);
        let y = R1D::from_vec(vec![0.0, 0.0]);
        let z = R2D::zeros((2, 2));
        assert!(Bilinear::new(x.view(), y.view(), z.view()).is_err());
        assert!(Bilinear::new(x.view(), x.view(), R2D::zeros((3, 2)).view()).is_err());
    }
}
