use crate::interpolation_trait::InterpolationTrait;
use lib_jsl_core::*;

/// The natural cubic spline through `(x, y)`: piecewise cubic with continuous first and second
/// derivatives, and zero second derivative at both ends. `x` must be strictly increasing.
///
/// See the book entry [Piecewise and spline interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/splines.html#cubic-splines).

#[derive(Debug)]
pub struct NaturalCubicSplineInterpolator<'a> {
    x_values: VR1D<'a>,
    y_values: VR1D<'a>,
    location: Option<usize>,
    a: R1D,
    b: R1D,
    c: R1D, 
    d: R1D,
}

impl<'a> InterpolationTrait<'a> for NaturalCubicSplineInterpolator<'a> {
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

    /// Evaluates the cubic piece of interval `jlo`.
    fn raw_interpolate(&mut self, jlo: usize, x: f64) -> Result<f64, ErrorsJSL> {
        let dx = x - self.x_values[jlo];
        let dx2 = dx*dx;
        let dx3 = dx2*dx;
        Ok(self.a[jlo] + self.b[jlo] * dx + self.c[jlo] * dx2 + self.d[jlo] * dx3)
    }
}

impl<'a> NaturalCubicSplineInterpolator<'a> {
        /// Compute the coefficients of a natural cubic spline.
        /// 
        /// The spline is represented piecewise by:
        /// `S_i(x) = a[i] + b[i]*(x - x_i) + c[i]*(x - x_i)^2 + d[i]*(x - x_i)^3`
        /// for x in [x_i, x_{i+1}].
        pub fn init_natural_cubic_spline(&mut self){
        // ANCHOR: natural_spline
        let n = self.x_values.len();
        let a = self.y_values.to_owned();
        let mut b = R1D::zeros(n - 1);
        let mut c = R1D::zeros(n);
        let mut d = R1D::zeros(n - 1);

        // Step 1: compute h
        let h = R1D::from_iter(self.x_values.iter()
                .zip(self.x_values.iter().skip(1))
                .map(|(&x0, &x1)| x1 - x0)
        );

        // Step 2: set up tridiagonal system
        let mut alpha = R1D::zeros(n);
        for i in 1..n - 1 {
            alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i])
                - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
        }

        // Step 3: solve tridiagonal system for c (Thomas algorithm)
        let mut l = R1D::zeros(n);
        let mut mu = R1D::zeros(n);
        let mut z = R1D::zeros(n);

        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;

        for i in 1..n - 1 {
            l[i] = 2.0 * (self.x_values[i + 1] - self.x_values[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n - 1] = 1.0;
        z[n - 1] = 0.0;
        c[n - 1] = 0.0;

        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = ((a[j + 1] - a[j]) / h[j]) - (h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0);
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        // ANCHOR_END: natural_spline
        self.a = a;
        self.b = b;
        self.c = c;
        self.d = d;
    }

    /// Initialize with y_values = f(x_values)
    pub fn new(x_values: VR1D<'a>, y_values: VR1D<'a>) -> Result<Self, ErrorsJSL> {

        let mut result = NaturalCubicSplineInterpolator{
            x_values,
            y_values,
            location: None,
            a: R1D::default(0),
            b: R1D::default(0),
            c: R1D::default(0),
            d: R1D::default(0),
        };
        result.verify_sizes(3)?;
        if x_values.iter().zip(x_values.iter().skip(1)).any(|(a, b)| b <= a || a.is_nan() || b.is_nan()) {
            return Err(ErrorsJSL::InvalidInputRange("x values must be strictly increasing"));
        }
        result.init_natural_cubic_spline();
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
    fn natural_spline_values() -> Result<(), ErrorsJSL> {
        let x = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = R1D::from_vec(vec![1.0, 2.0, 0.0, -1.0]);

        let mut dut = NaturalCubicSplineInterpolator::new(x.view(), y.view())?;

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
        assert!(approx_eq(y_interp,1.825,1E-9));
        let y_interp = dut.interpolate_at(1.5)?;
        assert!(approx_eq(y_interp,1.15,1E-9));
        let y_interp = dut.interpolate_at(2.5)?;
        assert!(approx_eq(y_interp,-0.675,1E-9));
        let y_interp = dut.interpolate_at(3.5)?;
        assert!(approx_eq(y_interp,-1.325,1E-9));
        let y_interp = dut.interpolate_at(-1.0)?;
        assert!(approx_eq(y_interp,0.0,1E-9));
        Ok(())
    }

    #[test]
    fn rejects_unordered_abscissas() {
        let x = R1D::from_vec(vec![0.0, 2.0, 1.0]);
        let y = R1D::from_vec(vec![0.0, 1.0, 2.0]);
        assert!(NaturalCubicSplineInterpolator::new(x.view(), y.view()).is_err());
        assert!(crate::one_d::linear_interpolator::LinearInterpolator::new(x.view(), y.view()).is_err());
    }
}
