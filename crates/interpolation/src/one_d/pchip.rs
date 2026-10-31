//! Piecewise cubic Hermite interpolating polynomial (PCHIP), after Fritsch and Carlson.
//!
//! The interpolant is monotone wherever the data are monotone and never overshoots the data, at
//! the price of only first-derivative continuity.

use super::hermite::{PiecewiseHermite, secants};
use lib_jsl_core::*;

/// End slope from the three-point formula, limited to preserve shape (as in MATLAB/SciPy).
fn end_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if m.signum() != d0.signum() {
        0.0
    } else if d0.signum() != d1.signum() && m.abs() > 3.0 * d0.abs() {
        3.0 * d0
    } else {
        m
    }
}

/// Builds a PCHIP interpolant through `(x, y)`; `x` must be strictly increasing, at least two points.
pub fn pchip<'a>(x: VR1D<'a>, y: VR1D<'a>) -> Result<PiecewiseHermite<'a>, ErrorsJSL> {
    let d = secants(x, y)?;
    let n = x.len();
    if n < 2 {
        return Err(ErrorsJSL::IncompatibleArraySizes((n, y.len())));
    }
    let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    let mut m = R1D::zeros(n);
    if n == 2 {
        m.fill(d[0]);
    } else {
        for i in 1..n - 1 {
            if d[i - 1] * d[i] > 0.0 {
                // Weighted harmonic mean (Fritsch-Butland).
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
            }
        }
        m[0] = end_slope(h[0], h[1], d[0], d[1]);
        m[n - 1] = end_slope(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
    }
    PiecewiseHermite::from_slopes(x, y, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation_trait::InterpolationTrait;

    #[test]
    fn exact_at_nodes_and_linear_data() -> Result<(), ErrorsJSL> {
        let x = R1D::from_vec(vec![0.0, 1.0, 2.5, 4.0, 5.0]);
        let y = x.mapv(|v| 2.0 * v + 1.0);
        let mut p = pchip(x.view(), y.view())?;
        for &v in &[0.0, 0.3, 1.0, 1.7, 3.3, 4.9, 5.0] {
            assert!((p.interpolate_at(v)? - (2.0 * v + 1.0)).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn preserves_monotonicity() -> Result<(), ErrorsJSL> {
        // A step-like monotone data set where a plain cubic spline would overshoot.
        let x = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = R1D::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let mut p = pchip(x.view(), y.view())?;
        let mut prev = -1.0;
        for k in 0..=600 {
            let v = p.interpolate_at(k as f64 * 0.01)?;
            assert!((-1e-12..=1.0 + 1e-12).contains(&v), "overshoot {v}");
            assert!(v >= prev - 1e-12, "not monotone");
            prev = v;
        }
        // Matches SciPy's PchipInterpolator at x = 2.5 (symmetric step => 0.5).
        assert!((p.interpolate_at(2.5)? - 0.5).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn rejects_bad_input() {
        let x = R1D::from_vec(vec![0.0, 1.0, 1.0]);
        let y = R1D::from_vec(vec![0.0, 1.0, 2.0]);
        assert!(pchip(x.view(), y.view()).is_err());
    }
}
