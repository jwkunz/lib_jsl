//! Cardinal cubic basis interpolation on an N-dimensional grid.

use super::{InterpolationND, check_point, validate_grid_nd};
use crate::two_d::{find_cell, hermite_basis};
use lib_jsl_core::*;
use ndarray::ArrayD;

/// Tensor-product cardinal (Catmull-Rom family) cubic in N dimensions: along the last axis the 1D
/// cardinal interpolant is evaluated at the 4 neighbouring lines of the next axis, and so on
/// recursively, touching a `4^N` neighbourhood. C¹, exact for multi-affine data; `tension` in
/// `[0, 1]` scales node slopes by `1 - tension` (0 is Catmull-Rom).
#[derive(Debug, Clone)]
pub struct CubicBasisND {
    axes: Vec<Vec<f64>>,
    values: ArrayD<f64>,
    tension: f64,
}

impl CubicBasisND {
    pub fn new(axes: &[&[f64]], values: &ArrayD<f64>, tension: f64) -> Result<Self, ErrorsJSL> {
        Ok(CubicBasisND { axes: validate_grid_nd(axes, values, 2)?, values: values.clone(), tension })
    }

    /// Value of the interpolant restricted to the line/plane/... fixed by `prefix` (indices of the
    /// leading axes), evaluated over the remaining axes at `point`.
    fn eval(&self, prefix: &mut Vec<usize>, cells: &[(usize, f64)]) -> f64 {
        let d = prefix.len();
        if d == self.axes.len() {
            return self.values[&prefix[..]];
        }
        let axis = &self.axes[d];
        let (i, t) = cells[d];
        let c = 1.0 - self.tension;
        let mut at = |k: usize, this: &Self| {
            prefix.push(k);
            let v = this.eval(prefix, cells);
            prefix.pop();
            v
        };
        let slope = |lo_v: f64, hi_v: f64, lo: usize, hi: usize| c * (hi_v - lo_v) / (axis[hi] - axis[lo]);
        let n = axis.len();
        let (a, b) = (i, i + 1);
        let (va, vb) = (at(a, self), at(b, self));
        let (alo, bhi) = (a.saturating_sub(1), (b + 1).min(n - 1));
        let vlo = if alo == a { va } else { at(alo, self) };
        let vhi = if bhi == b { vb } else { at(bhi, self) };
        // Node a is bracketed by (alo, b); node b by (a, bhi).
        let ma = slope(vlo, vb, alo, b);
        let mb = slope(va, vhi, a, bhi);
        let h = axis[b] - axis[a];
        let w = hermite_basis(t);
        w[0] * va + w[1] * h * ma + w[2] * vb + w[3] * h * mb
    }
}

impl InterpolationND for CubicBasisND {
    fn interpolate_at(&self, point: &[f64]) -> Result<f64, ErrorsJSL> {
        check_point(self.axes.len(), point)?;
        let cells: Vec<(usize, f64)> = self
            .axes
            .iter()
            .zip(point)
            .map(|(axis, &p)| {
                let i = find_cell(axis, p);
                (i, (p - axis[i]) / (axis[i + 1] - axis[i]))
            })
            .collect();
        Ok(self.eval(&mut Vec::new(), &cells))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn matches_the_2d_implementation_and_is_exact_for_affine_data() {
        let x = [0.0, 0.5, 1.5, 2.0, 3.0];
        let y = [0.0, 1.0, 1.5, 3.0];
        let f = |a: f64, b: f64| 1.0 + a - 2.0 * b + 0.25 * a * b + (a * 0.7).sin() * b;
        let v = ArrayD::from_shape_fn(IxDyn(&[5, 4]), |i| f(x[i[0]], y[i[1]]));
        let nd = CubicBasisND::new(&[&x, &y], &v, 0.0).unwrap();
        let two = crate::two_d::cubic_basis::CubicBasis2D::new(
            ndarray::arr1(&x).view(),
            ndarray::arr1(&y).view(),
            v.view().into_dimensionality::<ndarray::Ix2>().unwrap(),
            0.0,
        )
        .unwrap();
        use crate::two_d::Interpolation2D;
        for &(a, b) in &[(0.2, 0.3), (1.7, 1.2), (2.9, 2.8), (0.0, 3.0)] {
            assert!((nd.interpolate_at(&[a, b]).unwrap() - two.interpolate_at(a, b).unwrap()).abs() < 1e-12);
        }
        let g = |p: &[f64]| 2.0 - p[0] + 3.0 * p[1] + 0.5 * p[2] + p[0] * p[2];
        let ax = [0.0, 1.0, 2.0, 3.5];
        let w = ArrayD::from_shape_fn(IxDyn(&[4, 4, 4]), |i| g(&[ax[i[0]], ax[i[1]], ax[i[2]]]));
        let c3 = CubicBasisND::new(&[&ax, &ax, &ax], &w, 0.0).unwrap();
        assert!((c3.interpolate_at(&[1.3, 2.2, 0.7]).unwrap() - g(&[1.3, 2.2, 0.7])).abs() < 1e-11);
    }
}
