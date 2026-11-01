//! Cubic basis (cardinal / Catmull-Rom) interpolation.
//!
//! Each interval is a cubic Hermite segment whose node slopes come from a tension-scaled central
//! difference of the neighbouring points. With `tension = 0` this is the Catmull-Rom spline.
//!
//! See the book entry [Piecewise and spline interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/splines.html).

use super::hermite::PiecewiseHermite;
use lib_jsl_core::*;

/// Builds a cardinal cubic interpolant through `(x, y)`.
///
/// Interior slopes are `(1 - tension) (y[i+1]-y[i-1]) / (x[i+1]-x[i-1])`; end slopes use the
/// one-sided secant. `tension` is normally in `[0, 1]` (1 gives zero slopes, a smoothed staircase).
/// `x` must be strictly increasing with at least two points.
pub fn cubic_basis<'a>(x: VR1D<'a>, y: VR1D<'a>, tension: f64) -> Result<PiecewiseHermite<'a>, ErrorsJSL> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return Err(ErrorsJSL::IncompatibleArraySizes((n, y.len())));
    }
    if x.iter().zip(x.iter().skip(1)).any(|(a, b)| b <= a) {
        return Err(ErrorsJSL::InvalidInputRange("x values must be strictly increasing"));
    }
    let c = 1.0 - tension;
    let mut m = R1D::zeros(n);
    for i in 0..n {
        let (lo, hi) = (i.saturating_sub(1), (i + 1).min(n - 1));
        m[i] = c * (y[hi] - y[lo]) / (x[hi] - x[lo]);
    }
    PiecewiseHermite::from_slopes(x, y, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolation_trait::InterpolationTrait;

    #[test]
    fn nodes_lines_and_smooth_function() -> Result<(), ErrorsJSL> {
        let x = R1D::from_iter((0..21).map(|k| k as f64 * 0.1));
        let y = x.mapv(f64::sin);
        let mut c = cubic_basis(x.view(), y.view(), 0.0)?;
        for k in 0..21 {
            assert!((c.interpolate_at(x[k])? - y[k]).abs() < 1e-14);
        }
        for k in 0..199 {
            let v = 0.0105 + k as f64 * 0.01;
            if v < 2.0 {
                assert!((c.interpolate_at(v)? - v.sin()).abs() < 1e-3, "v={v}");
            }
        }
        let lx = R1D::from_vec(vec![0.0, 1.0, 3.0, 4.0]);
        let ly = lx.mapv(|v| 2.0 * v - 1.0);
        let mut l = cubic_basis(lx.view(), ly.view(), 0.0)?;
        assert!((l.interpolate_at(2.2)? - 3.4).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn full_tension_gives_flat_nodes() -> Result<(), ErrorsJSL> {
        let x = R1D::from_vec(vec![0.0, 1.0, 2.0]);
        let y = R1D::from_vec(vec![0.0, 2.0, 1.0]);
        let c = cubic_basis(x.view(), y.view(), 1.0)?;
        assert!(c.slopes().iter().all(|&s| s == 0.0));
        Ok(())
    }
}
