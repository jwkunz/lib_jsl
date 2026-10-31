//! Nearest-neighbour interpolation on an N-dimensional grid.

use super::{InterpolationND, check_point, validate_grid_nd};
use lib_jsl_core::*;
use ndarray::ArrayD;

/// Returns the value of the closest grid node (ties go to the lower index).
#[derive(Debug, Clone)]
pub struct NearestND {
    axes: Vec<Vec<f64>>,
    values: ArrayD<f64>,
}

impl NearestND {
    pub fn new(axes: &[&[f64]], values: &ArrayD<f64>) -> Result<Self, ErrorsJSL> {
        Ok(NearestND { axes: validate_grid_nd(axes, values, 1)?, values: values.clone() })
    }
}

impl InterpolationND for NearestND {
    fn interpolate_at(&self, point: &[f64]) -> Result<f64, ErrorsJSL> {
        check_point(self.axes.len(), point)?;
        let idx: Vec<usize> = self
            .axes
            .iter()
            .zip(point)
            .map(|(axis, &p)| {
                let i = axis.partition_point(|&a| a < p);
                if i == 0 {
                    0
                } else if i == axis.len() || p - axis[i - 1] <= axis[i] - p {
                    i - 1
                } else {
                    i
                }
            })
            .collect();
        Ok(self.values[&idx[..]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn four_dimensional_nearest() {
        let a = [0.0, 1.0, 2.0];
        let v = ArrayD::from_shape_fn(IxDyn(&[3, 3, 3, 3]), |i| (i[0] * 1000 + i[1] * 100 + i[2] * 10 + i[3]) as f64);
        let n = NearestND::new(&[&a, &a, &a, &a], &v).unwrap();
        assert_eq!(n.interpolate_at(&[0.4, 1.6, 1.4, 2.9]).unwrap(), 212.0);
        assert!(n.interpolate_at(&[0.0, 0.0]).is_err());
    }
}
