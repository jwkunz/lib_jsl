//! Romberg integration.

use crate::can_integrate::CanIntegrate1D;
use lib_jsl_core::ErrorsJSL;

/// Romberg integration: trapezoid estimates with 1, 2, 4, … panels, combined by Richardson
/// extrapolation `R(k, j) = R(k, j−1) + (R(k, j−1) − R(k−1, j−1)) / (4^j − 1)` to cancel the even
/// powers of the step in the Euler-Maclaurin error expansion. Converges very fast for smooth
/// integrands (row `k` is exact for polynomials of degree `2k + 1`); for non-smooth or periodic
/// integrands use an adaptive or plain trapezoid rule instead.
#[derive(Debug, Clone)]
pub struct RombergEngine {
    lower_bound: f64,
    upper_bound: f64,
    /// Relative tolerance between the last two diagonal entries (default 1e-12).
    pub tolerance: f64,
    /// Maximum number of panel doublings (default 20, about a million evaluations).
    pub max_levels: usize,
}

impl RombergEngine {
    /// A Romberg integrator over `[lower_bound, upper_bound]` with the default tolerance and
    /// level limit.
    ///
    /// ```
    /// use lib_jsl_integration::romberg::RombergEngine;
    ///
    /// let engine = RombergEngine::new(0.0, std::f64::consts::PI);
    /// let (value, error_estimate, levels) = engine.integrate_with_error(f64::sin).unwrap();
    /// assert!((value - 2.0).abs() < 1e-12 && error_estimate < 1e-10);
    /// assert!(levels <= 8); // at most 2⁸ + 1 = 257 function evaluations
    /// ```
    pub fn new(lower_bound: f64, upper_bound: f64) -> Self {
        RombergEngine { lower_bound, upper_bound, tolerance: 1e-12, max_levels: 20 }
    }

    /// The full Romberg tableau with `levels + 1` rows: `table[k][0]` is the trapezoid rule with
    /// `2^k` panels and `table[k][j]` its `j`-th Richardson extrapolation, accurate to
    /// `O(h^{2j+2})`. The diagonal `table[k][k]` is what [`RombergEngine::integrate_with_error`]
    /// returns. Useful for studying how extrapolation removes the error terms.
    ///
    /// ```
    /// use lib_jsl_integration::romberg::RombergEngine;
    ///
    /// let t = RombergEngine::new(0.0, 1.0).tableau(|x| x.powi(3), 2).unwrap();
    /// assert_eq!(t[0][0], 0.5); // one-panel trapezoid: (f(0) + f(1)) / 2
    /// assert!((t[1][1] - 0.25).abs() < 1e-15); // the first extrapolation is Simpson's rule, exact for cubics
    /// ```
    pub fn tableau<F: Fn(f64) -> f64>(&self, f: F, levels: usize) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
        let (a, b) = (self.lower_bound, self.upper_bound);
        if !a.is_finite() || !b.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("Romberg integration needs finite bounds."));
        }
        let mut h = b - a;
        let mut table = vec![vec![0.5 * h * (f(a) + f(b))]];
        for k in 1..=levels {
            let panels = 1usize << (k - 1);
            let midpoints: f64 = (0..panels).map(|i| f(a + (i as f64 + 0.5) * h)).sum();
            h *= 0.5;
            let previous = &table[k - 1];
            let mut row = vec![0.5 * previous[0] + h * midpoints];
            let mut factor = 1.0;
            for j in 1..=k {
                factor *= 4.0;
                row.push(row[j - 1] + (row[j - 1] - previous[j - 1]) / (factor - 1.0));
            }
            table.push(row);
        }
        Ok(table)
    }

    /// The extrapolated integral, the difference between the last two diagonal entries as an
    /// error estimate, and the number of levels used.
    pub fn integrate_with_error<F: Fn(f64) -> f64>(&self, f: F) -> Result<(f64, f64, usize), ErrorsJSL> {
        let (a, b) = (self.lower_bound, self.upper_bound);
        if !a.is_finite() || !b.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("Romberg integration needs finite bounds."));
        }
        // ANCHOR: romberg_level
        let mut h = b - a;
        let mut previous = vec![0.5 * h * (f(a) + f(b))];
        for k in 1..=self.max_levels {
            // Refine the trapezoid rule with the new midpoints.
            let panels = 1usize << (k - 1);
            let midpoints: f64 = (0..panels).map(|i| f(a + (i as f64 + 0.5) * h)).sum();
            h *= 0.5;
            let mut row = Vec::with_capacity(k + 1);
            row.push(0.5 * previous[0] + h * midpoints);
            let mut factor = 1.0;
            for j in 1..=k {
                factor *= 4.0;
                let value = row[j - 1] + (row[j - 1] - previous[j - 1]) / (factor - 1.0);
                row.push(value);
            }
            let error = (row[k] - previous[k - 1]).abs();
            if k >= 3 && (error <= self.tolerance * row[k].abs() || error == 0.0) {
                return Ok((row[k], error, k));
            }
            previous = row;
        }
        // ANCHOR_END: romberg_level
        let last = previous.len() - 1;
        Ok((previous[last], (previous[last] - previous[last - 1]).abs(), self.max_levels))
    }
}

impl CanIntegrate1D for RombergEngine {
    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
    }
    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
    }
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        self.integrate_with_error(f).map(|(v, _, _)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI};

    #[test]
    fn converges_rapidly_for_smooth_integrands() {
        let (v, _, levels) = RombergEngine::new(0.0, PI).integrate_with_error(f64::sin).unwrap();
        assert!((v - 2.0).abs() < 1e-13 && levels <= 8, "{v} in {levels} levels");
        let (v, _, _) = RombergEngine::new(0.0, 1.0).integrate_with_error(f64::exp).unwrap();
        assert!((v - (E - 1.0)).abs() < 1e-13);
        let e = RombergEngine::new(-2.0, 3.0);
        assert!((e.integrate(|x| 1.0 / (1.0 + x * x)).unwrap() - (3f64.atan() + 2f64.atan())).abs() < 1e-12);
    }

    #[test]
    fn polynomials_are_exact_after_few_levels_and_bounds_are_checked() {
        // Degree 7 is exact at level 3 (Boole's rule and beyond).
        let mut e = RombergEngine::new(0.0, 2.0);
        e.tolerance = 0.0;
        e.max_levels = 3;
        let (v, _, _) = e.integrate_with_error(|x| x.powi(7) - 3.0 * x.powi(4) + x).unwrap();
        assert!((v - (32.0 - 96.0 / 5.0 + 2.0)).abs() < 1e-12, "{v}");
        let mut r = RombergEngine::new(1.0, 1.0);
        assert_eq!(r.integrate(|x| x).unwrap(), 0.0);
        r.set_upper_bound(f64::NAN);
        assert!(r.integrate(|x| x).is_err());
    }
}
