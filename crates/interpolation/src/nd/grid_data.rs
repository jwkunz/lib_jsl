//! Multilinear interpolation of gridded data in any number of dimensions.

use super::{InterpolationND, check_point, validate_grid_nd};
use crate::two_d::find_cell;
use lib_jsl_core::*;
use ndarray::ArrayD;

/// Multilinear ("grid data") interpolation: linear along each axis within the enclosing cell, a
/// weighted average of the cell's `2^N` corners. Exact for functions that are affine in each
/// coordinate; edge cells extend linearly outside the grid.
#[derive(Debug, Clone)]
pub struct GridData {
    axes: Vec<Vec<f64>>,
    values: ArrayD<f64>,
}

impl GridData {
    pub fn new(axes: &[&[f64]], values: &ArrayD<f64>) -> Result<Self, ErrorsJSL> {
        Ok(GridData { axes: validate_grid_nd(axes, values, 2)?, values: values.clone() })
    }
}

impl InterpolationND for GridData {
    fn interpolate_at(&self, point: &[f64]) -> Result<f64, ErrorsJSL> {
        let dims = self.axes.len();
        check_point(dims, point)?;
        let mut cell = Vec::with_capacity(dims);
        let mut t = Vec::with_capacity(dims);
        for (axis, &p) in self.axes.iter().zip(point) {
            let i = find_cell(axis, p);
            cell.push(i);
            t.push((p - axis[i]) / (axis[i + 1] - axis[i]));
        }
        // ANCHOR: multilinear
        let mut idx = vec![0usize; dims];
        let mut sum = 0.0;
        for corner in 0..(1usize << dims) {
            let mut weight = 1.0;
            for d in 0..dims {
                let up = (corner >> d) & 1;
                idx[d] = cell[d] + up;
                weight *= if up == 1 { t[d] } else { 1.0 - t[d] };
            }
            if weight != 0.0 {
                sum += weight * self.values[&idx[..]];
            }
        }
        // ANCHOR_END: multilinear
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn exact_for_multi_affine_functions_in_four_dimensions() {
        let ax: [Vec<f64>; 4] = [vec![0.0, 1.0, 2.5], vec![-1.0, 0.0, 3.0, 4.0], vec![0.0, 2.0], vec![1.0, 1.5, 2.0]];
        let f = |p: &[f64]| 1.0 + p[0] - 2.0 * p[1] + 0.5 * p[2] * p[3] + p[0] * p[1] * p[2] * p[3];
        let v = ArrayD::from_shape_fn(IxDyn(&[3, 4, 2, 3]), |i| f(&[ax[0][i[0]], ax[1][i[1]], ax[2][i[2]], ax[3][i[3]]]));
        let g = GridData::new(&[&ax[0], &ax[1], &ax[2], &ax[3]], &v).unwrap();
        for p in [[0.3, 0.2, 1.1, 1.2], [2.2, 3.5, 0.4, 1.9], [3.0, 5.0, 3.0, 2.5]] {
            assert!((g.interpolate_at(&p).unwrap() - f(&p)).abs() < 1e-10, "{p:?}");
        }
    }

    #[test]
    fn one_dimensional_and_validation() {
        let g = GridData::new(&[&[0.0, 2.0]], &ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 5.0]).unwrap()).unwrap();
        assert_eq!(g.interpolate_at(&[0.5]).unwrap(), 2.0);
        assert!(GridData::new(&[&[0.0, 2.0]], &ArrayD::zeros(IxDyn(&[3]))).is_err());
        assert!(g.interpolate_at(&[0.0, 1.0]).is_err());
    }
}
