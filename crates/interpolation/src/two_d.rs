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

/// Start and spacing of a uniformly spaced axis (relative tolerance 1e-9).
pub(crate) fn uniform_axis(axis: VR1D) -> Result<(f64, f64), ErrorsJSL> {
    let step = (axis[axis.len() - 1] - axis[0]) / (axis.len() - 1) as f64;
    if axis.iter().enumerate().any(|(i, &a)| (a - axis[0] - i as f64 * step).abs() > 1e-9 * step.abs()) {
        return Err(ErrorsJSL::InvalidInputRange("axis must be uniformly spaced"));
    }
    Ok((axis[0], step))
}

/// Separable kernel resampling on a uniform grid: the value at `(x, y)` is a weighted sum of the
/// `2 * support` nearest nodes per axis, with weights `kernel(distance in samples)` normalised to
/// sum to one along each axis (so constants are preserved). Indices outside the grid replicate the
/// edge samples.
pub(crate) struct SeparableResampler<'k> {
    pub x0: f64,
    pub dx: f64,
    pub y0: f64,
    pub dy: f64,
    pub support: isize,
    pub kernel: &'k dyn Fn(f64) -> f64,
}

impl SeparableResampler<'_> {
    fn axis_weights(&self, u: f64, n: usize) -> Vec<(usize, f64)> {
        let base = u.floor() as isize;
        let mut taps: Vec<(usize, f64)> = ((base - self.support + 1)..=(base + self.support))
            .map(|i| (i.clamp(0, n as isize - 1) as usize, (self.kernel)(u - i as f64)))
            .collect();
        let sum: f64 = taps.iter().map(|t| t.1).sum();
        taps.iter_mut().for_each(|t| t.1 /= sum);
        taps
    }

    pub fn eval(&self, z: &R2D, x: f64, y: f64) -> f64 {
        let (nx, ny) = z.dim();
        let wx = self.axis_weights((x - self.x0) / self.dx, nx);
        let wy = self.axis_weights((y - self.y0) / self.dy, ny);
        wx.iter().map(|&(i, a)| a * wy.iter().map(|&(j, b)| b * z[[i, j]]).sum::<f64>()).sum()
    }
}

/// Normalised sinc, `sin(πx)/(πx)`.
pub(crate) fn sinc(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) }
}
