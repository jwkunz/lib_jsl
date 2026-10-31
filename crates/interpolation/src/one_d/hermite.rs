//! Shared piecewise cubic Hermite machinery used by PCHIP, Akima and the cubic basis interpolators.

use crate::interpolation_trait::InterpolationTrait;
use lib_jsl_core::*;

/// A piecewise cubic Hermite interpolant: on each interval the cubic is fixed by the end values
/// and the end slopes. The slope-selection rule (PCHIP, Akima, Catmull-Rom, ...) lives in the
/// constructors of the respective modules.
///
/// Outside the data range the end cubic pieces are extended.
#[derive(Debug)]
pub struct PiecewiseHermite<'a> {
    x_values: VR1D<'a>,
    y_values: VR1D<'a>,
    slopes: R1D,
    location: Option<usize>,
}

impl<'a> PiecewiseHermite<'a> {
    /// Builds the interpolant from explicit node slopes `m[i] = f'(x[i])`.
    pub fn from_slopes(x_values: VR1D<'a>, y_values: VR1D<'a>, slopes: R1D) -> Result<Self, ErrorsJSL> {
        let mut result = PiecewiseHermite { x_values, y_values, slopes, location: None };
        result.verify_sizes(2)?;
        if result.slopes.len() != x_values.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x_values.len(), result.slopes.len())));
        }
        if x_values.iter().zip(x_values.iter().skip(1)).any(|(a, b)| b <= a) {
            return Err(ErrorsJSL::InvalidInputRange("x values must be strictly increasing"));
        }
        Ok(result)
    }

    /// The node slopes in use.
    pub fn slopes(&self) -> &R1D {
        &self.slopes
    }
}

impl<'a> InterpolationTrait<'a> for PiecewiseHermite<'a> {
    fn get_x_values(&self) -> VR1D<'a> {
        self.x_values
    }
    fn set_x_values(&mut self, x: VR1D<'a>) {
        self.x_values = x;
    }
    fn get_y_values(&self) -> VR1D<'a> {
        self.y_values
    }
    fn set_y_values(&mut self, y: VR1D<'a>) {
        self.y_values = y;
    }
    fn get_cached_location(&self) -> Option<usize> {
        self.location
    }
    fn set_cached_location(&mut self, x: usize) {
        self.location = Some(x)
    }

    fn raw_interpolate(&mut self, j: usize, x: f64) -> Result<f64, ErrorsJSL> {
        let (x0, x1) = (self.x_values[j], self.x_values[j + 1]);
        let (y0, y1) = (self.y_values[j], self.y_values[j + 1]);
        let h = x1 - x0;
        let t = (x - x0) / h;
        let (t2, t3) = (t * t, t * t * t);
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        Ok(h00 * y0 + h10 * h * self.slopes[j] + h01 * y1 + h11 * h * self.slopes[j + 1])
    }
}

/// Secant slopes `(y[i+1]-y[i])/(x[i+1]-x[i])`.
pub(crate) fn secants(x: VR1D, y: VR1D) -> Result<Vec<f64>, ErrorsJSL> {
    if x.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
    }
    if x.iter().zip(x.iter().skip(1)).any(|(a, b)| b <= a) {
        return Err(ErrorsJSL::InvalidInputRange("x values must be strictly increasing"));
    }
    Ok((0..x.len().saturating_sub(1)).map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i])).collect())
}
