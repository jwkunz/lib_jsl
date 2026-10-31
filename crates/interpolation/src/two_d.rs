//! Two-dimensional interpolation.
//!
//! Gridded interpolators take axes `x` (length `nx`), `y` (length `ny`) and a value matrix `z` of
//! shape `(nx, ny)` with `z[[i, j]] = f(x[i], y[j])`.

use lib_jsl_core::*;

/// Evaluation interface shared by the 2D interpolators.
pub trait Interpolation2D {
    /// Interpolated value at `(x, y)`.
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL>;
}

/// Checks that axes are strictly increasing, long enough, and match the value matrix.
pub(crate) fn validate_grid(x: VR1D, y: VR1D, z: VR2D, min_points: usize) -> Result<(), ErrorsJSL> {
    if z.dim() != (x.len(), y.len()) {
        return Err(ErrorsJSL::IncompatibleArraySizes((z.nrows(), x.len())));
    }
    if x.len() < min_points || y.len() < min_points {
        return Err(ErrorsJSL::InvalidInputRange("not enough grid points along an axis"));
    }
    for axis in [x, y] {
        if axis.iter().zip(axis.iter().skip(1)).any(|(a, b)| b <= a) {
            return Err(ErrorsJSL::InvalidInputRange("axes must be strictly increasing"));
        }
    }
    Ok(())
}

/// Index `i` of the cell `[axis[i], axis[i+1]]` containing `v`, clamped to `0..=len-2`.
pub(crate) fn find_cell(axis: &[f64], v: f64) -> usize {
    axis.partition_point(|&a| a <= v).saturating_sub(1).min(axis.len() - 2)
}

/// Cubic Hermite basis functions `(h00, h10, h01, h11)` at `t` in `[0, 1]`.
pub(crate) fn hermite_basis(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, -2.0 * t3 + 3.0 * t2, t3 - t2]
}

pub mod nearest;
pub mod bilinear;
pub mod bicubic;
pub mod lanczos;
pub mod cubic_basis;
pub mod all_phase;
pub mod radial_basis_functions;
pub mod fourier_2d;
