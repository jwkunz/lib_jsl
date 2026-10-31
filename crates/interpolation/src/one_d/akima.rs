//! Akima interpolation: a locally determined piecewise cubic that avoids the wiggles of a global spline.

use super::hermite::{PiecewiseHermite, secants};
use lib_jsl_core::*;

/// Builds an Akima (1970) interpolant through `(x, y)`; `x` strictly increasing, at least three points.
///
/// Slopes are weighted averages of neighbouring secants; the two secants beyond each end are
/// extrapolated linearly in the secant sequence, as in the original paper.
pub fn akima<'a>(x: VR1D<'a>, y: VR1D<'a>) -> Result<PiecewiseHermite<'a>, ErrorsJSL> {
    let d = secants(x, y)?;
    let n = x.len();
    if n < 3 {
        return Err(ErrorsJSL::IncompatibleArraySizes((n, 3)));
    }
    // Extended secants: m[k+2] = d[k], with two extrapolated values at each end.
    let mut m = vec![0.0; n + 3];
    m[2..n + 1].copy_from_slice(&d);
    m[1] = 2.0 * m[2] - m[3];
    m[0] = 2.0 * m[1] - m[2];
    m[n + 1] = 2.0 * m[n] - m[n - 1];
    m[n + 2] = 2.0 * m[n + 1] - m[n];
    let mut slopes = R1D::zeros(n);
    for i in 0..n {
        let w1 = (m[i + 3] - m[i + 2]).abs();
        let w2 = (m[i + 1] - m[i]).abs();
        slopes[i] = if w1 + w2 > 1e-14 * (m[i + 1].abs() + m[i + 2].abs()).max(f64::MIN_POSITIVE) {
            (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2)
        } else {
            0.5 * (m[i + 1] + m[i + 2])
        };
    }
    PiecewiseHermite::from_slopes(x, y, slopes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation_trait::InterpolationTrait;

    #[test]
    fn reproduces_lines_and_nodes() -> Result<(), ErrorsJSL> {
        let x = R1D::from_vec(vec![-1.0, 0.0, 0.5, 2.0, 3.0, 4.5]);
        let y = x.mapv(|v| 3.0 - 0.5 * v);
        let mut a = akima(x.view(), y.view())?;
        for &v in &[-1.0, -0.2, 0.5, 1.1, 2.9, 4.5] {
            assert!((a.interpolate_at(v)? - (3.0 - 0.5 * v)).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn matches_reference_values() -> Result<(), ErrorsJSL> {
        // Akima's classic data set; reference from the algorithm worked by hand at the knots' slopes.
        let x = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let y = R1D::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.5, 15.0, 50.0, 60.0, 85.0]);
        let mut a = akima(x.view(), y.view())?;
        // Flat region stays flat (no ringing, unlike a natural spline).
        for k in 0..=50 {
            assert!((a.interpolate_at(k as f64 * 0.1)? - 10.0).abs() < 1e-12);
        }
        // Slope at x=7 per Akima's weights: m = (|d8-d7| d6 + |d6-d5| d7)/(...) with d5=.5,d6=4.5,d7=35,d8=10.
        let w1 = (10.0f64 - 35.0).abs();
        let w2 = (4.5f64 - 0.5).abs();
        let want = (w1 * 4.5 + w2 * 35.0) / (w1 + w2);
        assert!((a.slopes()[7] - want).abs() < 1e-12);
        Ok(())
    }
}
