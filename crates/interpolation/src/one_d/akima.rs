//! Akima interpolation: a locally determined piecewise cubic that avoids the wiggles of a global spline.
//!
//! See the book entry [Piecewise and spline interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/splines.html).

use super::hermite::{PiecewiseHermite, secants};
use lib_jsl_core::*;

/// Builds an Akima (1970) interpolant through `(x, y)`; `x` strictly increasing, at least two
/// points (two give a straight line).
///
/// Slopes are weighted averages of neighbouring secants; the two secants beyond each end are
/// extrapolated linearly in the secant sequence, as in the original paper. The weights, the fill
/// value where they vanish and the cutoff for that follow SciPy's `Akima1DInterpolator`.
pub fn akima<'a>(x: VR1D<'a>, y: VR1D<'a>) -> Result<PiecewiseHermite<'a>, ErrorsJSL> {
    PiecewiseHermite::from_slopes(x, y, R1D::from_vec(akima_slopes(x, y, false)?))
}

/// The modified Akima interpolant (`makima`, Cosmin Ionita 2019; SciPy's
/// `Akima1DInterpolator(method="makima")`): the weights `|m₄ − m₃| + |m₄ + m₃|/2` and
/// `|m₂ − m₁| + |m₂ + m₁|/2` also count the secants' size, so flat stretches and repeated
/// secants no longer produce overshoot.
pub fn makima<'a>(x: VR1D<'a>, y: VR1D<'a>) -> Result<PiecewiseHermite<'a>, ErrorsJSL> {
    PiecewiseHermite::from_slopes(x, y, R1D::from_vec(akima_slopes(x, y, true)?))
}

// ANCHOR: akima_slopes
/// The node slopes of Akima's method (`modified = false`) or of `makima` (`modified = true`).
pub fn akima_slopes(x: VR1D, y: VR1D, modified: bool) -> Result<Vec<f64>, ErrorsJSL> {
    let d = secants(x, y)?;
    let n = x.len();
    if n < 2 {
        return Err(ErrorsJSL::IncompatibleArraySizes((n, 2)));
    }
    if n == 2 {
        return Ok(vec![d[0]; 2]);
    }
    // Extended secants: m[k+2] = d[k], with two extrapolated values at each end.
    let mut m = vec![0.0; n + 3];
    m[2..n + 1].copy_from_slice(&d);
    m[1] = 2.0 * m[2] - m[3];
    m[0] = 2.0 * m[1] - m[2];
    m[n + 1] = 2.0 * m[n] - m[n - 1];
    m[n + 2] = 2.0 * m[n + 1] - m[n];
    let weight = |a: f64, b: f64| (a - b).abs() + if modified { 0.5 * (a + b).abs() } else { 0.0 };
    let f1: Vec<f64> = (0..n).map(|i| weight(m[i + 3], m[i + 2])).collect();
    let f2: Vec<f64> = (0..n).map(|i| weight(m[i + 1], m[i])).collect();
    let f12: Vec<f64> = f1.iter().zip(&f2).map(|(a, b)| a + b).collect();
    let cutoff = 1e-9 * f12.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok((0..n)
        .map(|i| if f12[i] > cutoff { m[i + 1] + f2[i] / f12[i] * (m[i + 2] - m[i + 1]) } else { 0.5 * (m[i + 3] + m[i]) })
        .collect())
}
// ANCHOR_END: akima_slopes

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
