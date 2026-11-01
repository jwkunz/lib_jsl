//! Floater-Hormann rational interpolation (barycentric, no poles on the real line).
//!
//! See the book entry [Rational interpolation and approximation](https://jwkunz.github.io/lib_jsl/book/interpolation/rational.html#floater-hormann-interpolation).

use super::lagrange_interpolant::barycentric_eval;
use crate::interpolation_trait::InterpolationTrait;
use lib_jsl_core::*;

/// Rational interpolant of blending degree `d` (0 <= d <= n-1) through points with strictly
/// increasing abscissae. `d = 0` is Berrut's interpolant; `d = n-1` is the polynomial interpolant.
/// Convergence is O(h^(d+1)) for smooth functions on any node distribution.
#[derive(Debug)]
pub struct FloaterHormann<'a> {
    x_values: VR1D<'a>,
    y_values: VR1D<'a>,
    weights: R1D,
}

impl<'a> FloaterHormann<'a> {
    pub fn new(x_values: VR1D<'a>, y_values: VR1D<'a>, d: usize) -> Result<Self, ErrorsJSL> {
        let mut r = FloaterHormann { x_values, y_values, weights: R1D::zeros(0) };
        r.verify_sizes(1)?;
        let n = x_values.len();
        if d >= n {
            return Err(ErrorsJSL::InvalidInputRange("blending degree must be below the point count"));
        }
        if x_values.iter().zip(x_values.iter().skip(1)).any(|(a, b)| b <= a) {
            return Err(ErrorsJSL::InvalidInputRange("x values must be strictly increasing"));
        }
        // w_i = (-1)^(i-d) * sum over k in [i-d, i] (k in 0..n-d) of prod_{j=k..k+d, j != i} 1/|x_i - x_j|
        let mut w = R1D::zeros(n);
        for i in 0..n {
            let lo = i.saturating_sub(d);
            let hi = i.min(n - 1 - d);
            let mut sum = 0.0;
            for k in lo..=hi {
                let mut prod = 1.0;
                for j in k..=k + d {
                    if j != i {
                        prod /= (x_values[i] - x_values[j]).abs();
                    }
                }
                sum += prod;
            }
            w[i] = if (i + d).is_multiple_of(2) { sum } else { -sum };
        }
        r.weights = w;
        Ok(r)
    }
}

impl<'a> InterpolationTrait<'a> for FloaterHormann<'a> {
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
    fn set_cached_location(&mut self, _: usize) {}
    fn raw_interpolate(&mut self, _: usize, _: f64) -> Result<f64, ErrorsJSL> {
        Err(ErrorsJSL::NotImplementedYet)
    }
    fn interpolate_at(&mut self, x: f64) -> Result<f64, ErrorsJSL> {
        Ok(barycentric_eval(self.x_values, self.y_values, self.weights.view(), x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::one_d::lagrange_interpolant::LagrangeInterpolant;

    #[test]
    fn runge_function_equispaced_is_stable() -> Result<(), ErrorsJSL> {
        // Runge's function on 41 equispaced points: the polynomial interpolant explodes, FH does not.
        let n = 41;
        let x = R1D::from_iter((0..n).map(|k| -1.0 + 2.0 * k as f64 / (n - 1) as f64));
        let f = |v: f64| 1.0 / (1.0 + 25.0 * v * v);
        let y = x.mapv(f);
        let mut fh = FloaterHormann::new(x.view(), y.view(), 5)?;
        let mut poly = LagrangeInterpolant::new(x.view(), y.view())?;
        let (mut e_fh, mut e_poly): (f64, f64) = (0.0, 0.0);
        for k in 0..=400 {
            let v = -1.0 + 2.0 * (k as f64 + 0.37) / 401.0;
            e_fh = e_fh.max((fh.interpolate_at(v)? - f(v)).abs());
            e_poly = e_poly.max((poly.interpolate_at(v)? - f(v)).abs());
        }
        assert!(e_fh < 1e-4, "fh error {e_fh}");
        assert!(e_poly > 10.0 * e_fh, "poly error {e_poly}");
        Ok(())
    }

    #[test]
    fn full_degree_equals_polynomial_and_nodes_exact() -> Result<(), ErrorsJSL> {
        let x = R1D::from_vec(vec![0.0, 0.7, 1.5, 2.0, 3.1]);
        let y = R1D::from_vec(vec![1.0, -2.0, 0.5, 4.0, 3.0]);
        let mut fh = FloaterHormann::new(x.view(), y.view(), 4)?;
        let mut lg = LagrangeInterpolant::new(x.view(), y.view())?;
        for &v in &[0.3, 1.1, 2.6] {
            assert!((fh.interpolate_at(v)? - lg.interpolate_at(v)?).abs() < 1e-10);
        }
        assert_eq!(fh.interpolate_at(1.5)?, 0.5);
        Ok(())
    }
}
