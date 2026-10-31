//! Nearest-neighbour interpolation on a rectilinear grid.

use super::{Interpolation2D, validate_grid};
use lib_jsl_core::*;

/// Returns the value of the closest grid node (ties go to the lower index).
#[derive(Debug, Clone)]
pub struct Nearest2D {
    x: R1D,
    y: R1D,
    z: R2D,
}

fn nearest_index(axis: &R1D, v: f64) -> usize {
    let i = axis.as_slice().expect("contiguous").partition_point(|&a| a < v);
    if i == 0 {
        0
    } else if i == axis.len() || v - axis[i - 1] <= axis[i] - v {
        i - 1
    } else {
        i
    }
}

impl Nearest2D {
    pub fn new(x: VR1D, y: VR1D, z: VR2D) -> Result<Self, ErrorsJSL> {
        validate_grid(x, y, z, 1)?;
        Ok(Nearest2D { x: x.to_owned(), y: y.to_owned(), z: z.to_owned() })
    }
}

impl Interpolation2D for Nearest2D {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        Ok(self.z[[nearest_index(&self.x, x), nearest_index(&self.y, y)]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_closest_node_with_clamping() {
        let x = R1D::from_vec(vec![0.0, 1.0, 3.0]);
        let y = R1D::from_vec(vec![0.0, 2.0]);
        let z = R2D::from_shape_fn((3, 2), |(i, j)| (10 * i + j) as f64);
        let n = Nearest2D::new(x.view(), y.view(), z.view()).unwrap();
        assert_eq!(n.interpolate_at(0.4, 0.9).unwrap(), 0.0);
        assert_eq!(n.interpolate_at(0.6, 1.1).unwrap(), 11.0);
        assert_eq!(n.interpolate_at(2.1, -5.0).unwrap(), 20.0);
        assert_eq!(n.interpolate_at(99.0, 99.0).unwrap(), 21.0);
        assert_eq!(n.interpolate_at(2.0, 0.0).unwrap(), 10.0); // tie -> lower
    }
}
