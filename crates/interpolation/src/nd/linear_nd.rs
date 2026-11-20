//! Piecewise-linear interpolation of scattered data in any dimension (SciPy's
//! `LinearNDInterpolator` in 3-D and higher), on the Delaunay triangulation of the data.
//!
//! See the book entry [Geometry in any dimension](https://jwkunz.github.io/lib_jsl/book/geometry/hulls-nd.html#implementation).

use crate::nd::InterpolationND;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_geometry::algorithms::hull_nd::DelaunayNd;

// ANCHOR: linear_nd
/// Linear interpolation on the Delaunay triangulation of scattered points in `d ≥ 2`
/// dimensions: the value at `x` is the barycentric average of the values at the vertices of
/// the simplex containing `x`, and `fill_value` (NaN by default, as SciPy) outside the convex
/// hull.
#[derive(Clone, Debug)]
pub struct LinearNdInterpolator {
    tri: DelaunayNd,
    values: Vec<f64>,
    fill_value: f64,
}

impl LinearNdInterpolator {
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<LinearNdInterpolator, ErrorsJSL> {
        if points.len() != values.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((points.len(), values.len())));
        }
        Ok(LinearNdInterpolator { tri: DelaunayNd::new(points)?, values: values.to_vec(), fill_value: f64::NAN })
    }

    /// The value returned outside the convex hull of the data.
    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// The triangulation used.
    pub fn triangulation(&self) -> &DelaunayNd {
        &self.tri
    }

    pub fn evaluate(&self, x: &[f64]) -> f64 {
        match self.tri.find_simplex(x) {
            None => self.fill_value,
            Some(s) => self.tri.barycentric(s, x).iter().zip(&self.tri.simplices[s]).map(|(c, &i)| c * self.values[i]).sum(),
        }
    }
}
// ANCHOR_END: linear_nd

impl InterpolationND for LinearNdInterpolator {
    fn interpolate_at(&self, point: &[f64]) -> Result<f64, ErrorsJSL> {
        Ok(self.evaluate(point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_linear_functions_in_three_and_four_dimensions() {
        for d in [3usize, 4] {
            let mut s = 7u64 + d as u64;
            let mut next = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            };
            let pts: Vec<Vec<f64>> = (0..80).map(|_| (0..d).map(|_| next()).collect()).collect();
            let f = |p: &[f64]| 1.0 + p.iter().enumerate().map(|(k, v)| (k as f64 + 1.0) * v).sum::<f64>();
            let vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();
            let it = LinearNdInterpolator::new(&pts, &vals).unwrap();
            let x: Vec<f64> = vec![0.5; d];
            assert!((it.evaluate(&x) - f(&x)).abs() < 1e-12);
            assert!(it.evaluate(&vec![2.0; d]).is_nan());
            assert_eq!(it.clone().with_fill_value(-1.0).evaluate(&vec![2.0; d]), -1.0);
        }
    }
}
