//! Lagrange interpolation in barycentric form (second form, O(n) per evaluation).
//!
//! See the book entry [Polynomial interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/polynomial.html).

use crate::interpolation_trait::InterpolationTrait;
use lib_jsl_core::*;

/// The unique polynomial of degree `n - 1` through `n` points with distinct abscissae.
#[derive(Debug)]
pub struct LagrangeInterpolant<'a> {
    x_values: VR1D<'a>,
    y_values: VR1D<'a>,
    weights: R1D,
}

impl<'a> LagrangeInterpolant<'a> {
    /// Builds the interpolant; the `x` values must be distinct (any order). Weights cost O(n²).
    pub fn new(x_values: VR1D<'a>, y_values: VR1D<'a>) -> Result<Self, ErrorsJSL> {
        let mut r = LagrangeInterpolant { x_values, y_values, weights: R1D::zeros(0) };
        r.verify_sizes(1)?;
        let n = x_values.len();
        let mut w = R1D::ones(n);
        for j in 0..n {
            for k in 0..n {
                if k != j {
                    let d = x_values[j] - x_values[k];
                    if d == 0.0 {
                        return Err(ErrorsJSL::InvalidInputRange("x values must be distinct"));
                    }
                    w[j] /= d;
                }
            }
        }
        r.weights = w;
        Ok(r)
    }

    /// The barycentric weights.
    pub fn weights(&self) -> &R1D {
        &self.weights
    }
}

impl<'a> InterpolationTrait<'a> for LagrangeInterpolant<'a> {
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

// ANCHOR: barycentric
/// Second-form barycentric evaluation shared with Floater-Hormann and Chebyshev interpolation.
pub(crate) fn barycentric_eval(x_nodes: VR1D, y: VR1D, w: VR1D, x: f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..x_nodes.len() {
        let d = x - x_nodes[i];
        if d == 0.0 {
            return y[i];
        }
        let t = w[i] / d;
        num += t * y[i];
        den += t;
    }
    num / den
}
// ANCHOR_END: barycentric

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_polynomials_exactly() -> Result<(), ErrorsJSL> {
        let x = R1D::from_vec(vec![-2.0, -0.5, 0.0, 1.0, 3.5]);
        let f = |v: f64| 2.0 * v.powi(4) - v.powi(3) + 0.5 * v - 3.0;
        let y = x.mapv(f);
        let mut l = LagrangeInterpolant::new(x.view(), y.view())?;
        for &v in &[-2.0, -1.3, 0.2, 2.2, 3.5, 5.0] {
            assert!((l.interpolate_at(v)? - f(v)).abs() < 1e-9 * (1.0 + f(v).abs()), "v={v}");
        }
        Ok(())
    }

    #[test]
    fn rejects_duplicates() {
        let x = R1D::from_vec(vec![1.0, 2.0, 1.0]);
        let y = R1D::from_vec(vec![0.0, 1.0, 2.0]);
        assert!(LagrangeInterpolant::new(x.view(), y.view()).is_err());
    }
}
