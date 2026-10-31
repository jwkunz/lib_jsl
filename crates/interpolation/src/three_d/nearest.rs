//! Nearest-neighbour interpolation on a 3D rectilinear grid.

use super::{Interpolation3D, validate_grid_3d};
use lib_jsl_core::*;
use ndarray::Array3;

/// Returns the value of the closest grid node (ties go to the lower index).
#[derive(Debug, Clone)]
pub struct Nearest3D {
    axes: [R1D; 3],
    v: Array3<f64>,
}

fn nearest_index(axis: &R1D, p: f64) -> usize {
    let i = axis.as_slice().expect("contiguous").partition_point(|&a| a < p);
    if i == 0 {
        0
    } else if i == axis.len() || p - axis[i - 1] <= axis[i] - p {
        i - 1
    } else {
        i
    }
}

impl Nearest3D {
    pub fn new(x: VR1D, y: VR1D, z: VR1D, v: &Array3<f64>) -> Result<Self, ErrorsJSL> {
        validate_grid_3d([x, y, z], v, 1)?;
        Ok(Nearest3D { axes: [x.to_owned(), y.to_owned(), z.to_owned()], v: v.clone() })
    }
}

impl Interpolation3D for Nearest3D {
    fn interpolate_at(&self, x: f64, y: f64, z: f64) -> Result<f64, ErrorsJSL> {
        Ok(self.v[[nearest_index(&self.axes[0], x), nearest_index(&self.axes[1], y), nearest_index(&self.axes[2], z)]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_closest_node() {
        let a = R1D::from_vec(vec![0.0, 1.0, 2.0]);
        let v = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (100 * i + 10 * j + k) as f64);
        let n = Nearest3D::new(a.view(), a.view(), a.view(), &v).unwrap();
        assert_eq!(n.interpolate_at(0.4, 1.6, 1.4).unwrap(), 21.0);
        assert_eq!(n.interpolate_at(-9.0, 9.0, 0.5).unwrap(), 20.0); // tie -> lower
        assert!(Nearest3D::new(a.view(), a.view(), a.view(), &Array3::zeros((3, 3, 2))).is_err());
    }
}
