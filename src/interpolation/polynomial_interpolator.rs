use crate::interpolation::interpolation_trait::InterpolationTrait;
use crate::prelude::*;

/// This stuct implements polynomial interpolation for y = f(x)
/// This uses Newtons divided differences for efficient computation

#[derive(Debug)]
pub struct PolynomialInterpolator<'a> {
    x_values: VR1D<'a>,
    y_values: VR1D<'a>,
    coeficients: Vec<RealNumber>,
}

impl<'a> InterpolationTrait<'a> for PolynomialInterpolator<'a> {
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
        None
    }
    fn set_cached_location(&mut self, _: usize) {
        // Do nothing
    }

    /// This is the logic of the linear interpolator
    fn raw_interpolate(&mut self, _: usize, _: RealNumber) -> Result<RealNumber, ErrorsJSL> {
        Err(ErrorsJSL::NotImplementedYet) // Not needed
    }

    /// Evaluate the Newton interpolating polynomial at a given x.
    ///
    /// Uses Horner’s method for efficiency.
    fn interpolate_at(&mut self, x: RealNumber) -> Result<RealNumber, ErrorsJSL> {
        let last_index = self.coeficients.len()-1;
        let mut result = self.coeficients[last_index];
        for i in (0..last_index).rev() {
            result = result * (x - self.x_values[i]) + self.coeficients[i];
        }
        Ok(result)
    }
}

impl<'a> PolynomialInterpolator<'a> {
    /// Compute Newton divided differences coefficients.
    ///
    /// # Arguments
    /// * `x_values` - View of x-coordinates (f64)
    /// * `y_values` - View of y-coordinates (f64)
    ///
    /// # Returns
    /// Coefficients `a` such that
    /// P(x) = a[0] + a[1](x - x0) + a[2](x - x0)(x - x1) + ...
    fn init_newton_divided_differences(&mut self){
        let n = self.x_values.len();
        self.coeficients.extend_from_slice(self.y_values.as_slice().expect("These were initialized before"));

        // Compute divided differences in-place
        for j in 1..n {
            for i in (j..n).rev() {
                self.coeficients[i] = (self.coeficients[i] - self.coeficients[i - 1]) / (self.x_values[i] - self.x_values[i - j]);
            }
        }
    }

    /// Initialize with y_values = f(x_values)
    pub fn new(x_values: VR1D<'a>, y_values: VR1D<'a>) -> Result<Self, ErrorsJSL> {
        let mut result = PolynomialInterpolator {
            x_values,
            y_values,
            coeficients: Vec::with_capacity(x_values.len()),
        };
        result.verify_sizes(2)?;
        result.init_newton_divided_differences();
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Helper for comparing floats
    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
    }

    #[test]
    fn tb_linear_interpolator() -> Result<(), ErrorsJSL> {
        let x = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = R1D::from_vec(vec![1.0, 2.0, 0.0, -1.0]);

        let mut dut = PolynomialInterpolator::new(x.view(), y.view())?;

        // Exact hits
        let y_interp = dut.interpolate_at(0.0)?;
        assert!(approx_eq(y_interp,1.0,1E-9));
        let y_interp = dut.interpolate_at(1.0)?;
        assert!(approx_eq(y_interp,2.0,1E-9));
        let y_interp = dut.interpolate_at(2.0)?;
        assert!(approx_eq(y_interp,0.0,1E-9));
        let y_interp = dut.interpolate_at(3.0)?;
        assert!(approx_eq(y_interp,-1.0,1E-9));

        // Interpolations
        let y_interp = dut.interpolate_at(0.5)?;
        assert!(approx_eq(y_interp,2.125,1E-9));
        let y_interp = dut.interpolate_at(1.5)?;
        assert!(approx_eq(y_interp,1.125,1E-9));
        let y_interp = dut.interpolate_at(2.5)?;
        assert!(approx_eq(y_interp,-0.875,1E-9));
        let y_interp = dut.interpolate_at(3.5)?;
        assert!(approx_eq(y_interp,0.125,1E-9));
        let y_interp = dut.interpolate_at(-1.0)?;
        assert!(approx_eq(y_interp,-7.0,1E-9));

        Ok(())
    }
}
