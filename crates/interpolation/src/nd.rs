//! N-dimensional interpolation on rectilinear grids.
//!
//! See the book entry [Interpolation on grids](https://jwkunz.github.io/lib_jsl/book/interpolation/grids.html).
//!
//! A grid is given by one strictly increasing axis per dimension and a value array whose shape is
//! the axis lengths. Queries are slices with one coordinate per dimension.

use lib_jsl_core::*;
use ndarray::ArrayD;

/// Evaluation interface shared by the N-dimensional interpolators.
pub trait InterpolationND {
    /// Interpolated value at `point` (one coordinate per dimension).
    fn interpolate_at(&self, point: &[f64]) -> Result<f64, ErrorsJSL>;
}

/// Checks axes and shape consistency; returns the axes as owned vectors.
pub(crate) fn validate_grid_nd(axes: &[&[f64]], values: &ArrayD<f64>, min_points: usize) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    if axes.is_empty() || values.ndim() != axes.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((values.ndim(), axes.len())));
    }
    for (d, axis) in axes.iter().enumerate() {
        if axis.len() != values.shape()[d] {
            return Err(ErrorsJSL::IncompatibleArraySizes((values.shape()[d], axis.len())));
        }
        if axis.len() < min_points {
            return Err(ErrorsJSL::InvalidInputRange("not enough grid points along an axis"));
        }
        if axis.windows(2).any(|w| w[1] <= w[0]) {
            return Err(ErrorsJSL::InvalidInputRange("axes must be strictly increasing"));
        }
    }
    Ok(axes.iter().map(|a| a.to_vec()).collect())
}

pub(crate) fn check_point(dims: usize, point: &[f64]) -> Result<(), ErrorsJSL> {
    if point.len() != dims {
        return Err(ErrorsJSL::IncompatibleArraySizes((point.len(), dims)));
    }
    Ok(())
}

pub mod grid_data;
pub mod nearest;
pub mod cubic_basis;
