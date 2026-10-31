//! Bulirsch-Stoer rational function interpolation.

use crate::interpolation_trait::InterpolationTrait;
use lib_jsl_core::*;

/// Diagonal rational function interpolation (Stoer and Bulirsch) through points with distinct
/// abscissae. Unlike a polynomial, a rational function can follow poles and sharp resonances, and
/// it reproduces any rational function of matching degree exactly. Each evaluation is O(n²) and
/// also yields an error estimate (the size of the last correction).
///
/// A genuine pole of the interpolant at the query point produces an error. Iteration stops early
/// once corrections fall to rounding level, which happens when the data are exactly rational of
/// lower degree than the point count allows.
#[derive(Debug)]
pub struct BulirschStoer<'a> {
    x_values: VR1D<'a>,
    y_values: VR1D<'a>,
}

impl<'a> BulirschStoer<'a> {
    pub fn new(x_values: VR1D<'a>, y_values: VR1D<'a>) -> Result<Self, ErrorsJSL> {
        let mut r = BulirschStoer { x_values, y_values };
        r.verify_sizes(1)?;
        let n = x_values.len();
        for i in 0..n {
            if (0..i).any(|j| x_values[j] == x_values[i]) {
                return Err(ErrorsJSL::InvalidInputRange("x values must be distinct"));
            }
        }
        Ok(r)
    }

    /// Interpolated value at `x` together with an estimate of its error.
    pub fn interpolate_with_error(&self, x: f64) -> Result<(f64, f64), ErrorsJSL> {
        const TINY: f64 = 1e-25; // keeps a zero first denominator from producing 0/0
        let (xa, ya) = (&self.x_values, &self.y_values);
        let n = xa.len();
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        let mut ns = 0usize;
        let mut hh = (x - xa[0]).abs();
        for i in 0..n {
            let h = (x - xa[i]).abs();
            if h == 0.0 {
                return Ok((ya[i], 0.0));
            }
            if h < hh {
                ns = i;
                hh = h;
            }
            c[i] = ya[i];
            d[i] = ya[i] + TINY;
        }
        let mut y = ya[ns];
        let mut ns = ns as isize - 1;
        let mut dy = 0.0;
        for m in 1..n {
            for i in 0..n - m {
                let w = c[i + 1] - d[i];
                let h = xa[i + m] - x;
                let t = (xa[i] - x) * d[i] / h;
                let dd = t - c[i + 1];
                if dd == 0.0 {
                    return Err(ErrorsJSL::RuntimeError("interpolating function has a pole at the requested value"));
                }
                let dd = w / dd;
                d[i] = c[i + 1] * dd;
                c[i] = t * dd;
            }
            dy = if 2 * (ns + 1) < (n - m) as isize {
                c[(ns + 1) as usize]
            } else {
                let v = d[ns as usize];
                ns -= 1;
                v
            };
            y += dy;
            // Exactly representable data (a lower-degree rational function) makes later
            // denominators collapse to zero; once the corrections vanish the fit is complete.
            if m >= 2 && dy.abs() <= 4.0 * f64::EPSILON * y.abs() {
                break;
            }
        }
        Ok((y, dy.abs()))
    }
}

impl<'a> InterpolationTrait<'a> for BulirschStoer<'a> {
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
        self.interpolate_with_error(x).map(|(y, _)| y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_rational_function_exactly() {
        // A [2/2] rational function is determined by 5 points.
        let g = |x: f64| (2.0 * x * x - x + 3.0) / (x * x + 0.5 * x + 2.0);
        let x = R1D::from_vec(vec![-2.0, -0.7, 0.4, 1.3, 2.9]);
        let y = x.mapv(g);
        let mut b = BulirschStoer::new(x.view(), y.view()).unwrap();
        for &v in &[-3.5, -1.2, 0.0, 0.9, 2.0, 6.0] {
            assert!((b.interpolate_at(v).unwrap() - g(v)).abs() < 1e-9, "v={v}");
        }
    }

    #[test]
    fn follows_a_near_pole_far_better_than_a_polynomial() {
        let g = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
        let x = R1D::from_iter((0..9).map(|k| -0.93 + 0.25 * k as f64));
        let y = x.mapv(g);
        let mut b = BulirschStoer::new(x.view(), y.view()).unwrap();
        let mut p = crate::one_d::lagrange_interpolant::LagrangeInterpolant::new(x.view(), y.view()).unwrap();
        let (eb, ep) = ((b.interpolate_at(0.9).unwrap() - g(0.9)).abs(), (p.interpolate_at(0.9).unwrap() - g(0.9)).abs());
        assert!(eb < ep, "bs {eb} poly {ep}");
        // Exact at nodes with zero estimated error, and an error estimate elsewhere.
        assert_eq!(b.interpolate_with_error(x[4]).unwrap(), (y[4], 0.0));
        assert!(b.interpolate_with_error(0.37).unwrap().1 > 0.0);
    }

    #[test]
    fn rejects_duplicate_abscissae() {
        let x = R1D::from_vec(vec![1.0, 2.0, 1.0]);
        assert!(BulirschStoer::new(x.view(), x.view()).is_err());
    }
}
