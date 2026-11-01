use crate::interpolation_trait::InterpolationTrait;
use lib_jsl_core::*;

/// The polynomial of degree `n − 1` through `n` points, in Newton's divided-difference form,
/// evaluated by Horner's scheme (`O(n)` per point after `O(n²)` set-up). The abscissas must be
/// distinct. For many points, or points that are not well spread, prefer the numerically stabler
/// barycentric form of [`LagrangeInterpolant`](super::lagrange_interpolant::LagrangeInterpolant).
///
/// See the book entry [Polynomial interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/polynomial.html).

#[derive(Debug)]
pub struct PolynomialInterpolator<'a> {
    x_values: VR1D<'a>,
    y_values: VR1D<'a>,
    coefficients: Vec<f64>,
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

    /// Not used: the polynomial is global, so [`interpolate_at`](InterpolationTrait::interpolate_at)
    /// evaluates it directly.
    fn raw_interpolate(&mut self, _: usize, _: f64) -> Result<f64, ErrorsJSL> {
        Err(ErrorsJSL::NotImplementedYet) // Not needed
    }

    /// Evaluate the Newton interpolating polynomial at a given x.
    ///
    /// Uses Horner’s method for efficiency.
    fn interpolate_at(&mut self, x: f64) -> Result<f64, ErrorsJSL> {
        let last_index = self.coefficients.len()-1;
        let mut result = self.coefficients[last_index];
        for i in (0..last_index).rev() {
            result = result * (x - self.x_values[i]) + self.coefficients[i];
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
        self.coefficients.extend(self.y_values.iter().copied());

        // Compute divided differences in-place
        for j in 1..n {
            for i in (j..n).rev() {
                self.coefficients[i] = (self.coefficients[i] - self.coefficients[i - 1]) / (self.x_values[i] - self.x_values[i - j]);
            }
        }
    }

    /// Initialize with y_values = f(x_values)
    pub fn new(x_values: VR1D<'a>, y_values: VR1D<'a>) -> Result<Self, ErrorsJSL> {
        let mut result = PolynomialInterpolator {
            x_values,
            y_values,
            coefficients: Vec::with_capacity(x_values.len()),
        };
        result.verify_sizes(2)?;
        for i in 0..x_values.len() {
            for j in i + 1..x_values.len() {
                if x_values[i] == x_values[j] {
                    return Err(ErrorsJSL::InvalidInputRange("x values must be distinct"));
                }
            }
        }
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
    fn cubic_through_four_points() -> Result<(), ErrorsJSL> {
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
        // Repeated abscissas are rejected instead of producing infinite coefficients.
        let xs = R1D::from_vec(vec![0.0, 1.0, 1.0]);
        let ys = R1D::from_vec(vec![0.0, 1.0, 2.0]);
        assert!(PolynomialInterpolator::new(xs.view(), ys.view()).is_err());
        Ok(())
    }
}
