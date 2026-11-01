//! Three-dimensional interpolation.
//!
//! See the book entry [Interpolation on grids](https://jwkunz.github.io/lib_jsl/book/interpolation/grids.html).
//!
//! Gridded interpolators take axes `x`, `y`, `z` and a value array of shape `(nx, ny, nz)` with
//! `v[[i, j, k]] = f(x[i], y[j], z[k])`.

use lib_jsl_core::*;
use ndarray::Array3;

/// Evaluation interface shared by the 3D interpolators.
pub trait Interpolation3D {
    /// Interpolated value at `(x, y, z)`.
    fn interpolate_at(&self, x: f64, y: f64, z: f64) -> Result<f64, ErrorsJSL>;
}

/// Checks that axes are strictly increasing, long enough, and match the value array.
pub(crate) fn validate_grid_3d(axes: [VR1D; 3], v: &Array3<f64>, min_points: usize) -> Result<(), ErrorsJSL> {
    if v.dim() != (axes[0].len(), axes[1].len(), axes[2].len()) {
        return Err(ErrorsJSL::IncompatibleArraySizes((v.dim().0, axes[0].len())));
    }
    for axis in axes {
        if axis.len() < min_points {
            return Err(ErrorsJSL::InvalidInputRange("not enough grid points along an axis"));
        }
        if axis.iter().zip(axis.iter().skip(1)).any(|(a, b)| b <= a) {
            return Err(ErrorsJSL::InvalidInputRange("axes must be strictly increasing"));
        }
    }
    Ok(())
}

pub mod nearest;
pub mod trilinear;
pub mod b_spline;
pub mod cubic_lagrangian;
pub mod fourier_3d;
pub mod shepards;
