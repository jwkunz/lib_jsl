//! Brute-force minimisation over a grid (SciPy's `optimize.brute`).
//!
//! See the book entry [Brute force, isotonic regression and gradient checks](https://jwkunz.github.io/lib_jsl/book/optimization/brute-isotonic.html).

use crate::nelder_mead_method::NelderMeadMethod;
use crate::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction};
use lib_jsl_core::ErrorsJSL;

/// The result of [`brute`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct BruteResult {
    /// The minimiser: the best grid point, or the polished point when a finish was asked for.
    pub x: Vec<f64>,
    /// The objective at `x`.
    pub fun: f64,
    /// The grid's coordinates along each axis.
    pub grid: Vec<Vec<f64>>,
    /// The objective at every grid point, the last axis varying fastest.
    pub values: Vec<f64>,
}

struct Wrapped<'a>(&'a dyn Fn(&[f64]) -> f64);

impl ObjectiveFunction for Wrapped<'_> {
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        (self.0)(parameters)
    }
}

// ANCHOR: brute
/// Minimises `f` by evaluating it at every point of the grid with `ns` equally spaced points
/// from `lo` to `hi` (both included) on each axis (SciPy's `brute` with `(lo, hi)` ranges and
/// `Ns = ns`, SciPy's default 20). With `finish`, the best grid point is polished by Nelder-Mead,
/// as SciPy's default `finish = fmin`, and kept only if the polished point is better.
pub fn brute(f: impl Fn(&[f64]) -> f64, ranges: &[(f64, f64)], ns: usize, finish: bool) -> Result<BruteResult, ErrorsJSL> {
    if ranges.is_empty() || ns < 2 || ranges.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("brute: need at least one finite range and two points per axis"));
    }
    let d = ranges.len();
    let total = ns.checked_pow(d as u32).filter(|&t| t <= 100_000_000).ok_or(ErrorsJSL::InvalidInputRange("brute: grid too large"))?;
    let grid: Vec<Vec<f64>> = ranges.iter().map(|&(a, b)| (0..ns).map(|i| if i + 1 == ns { b } else { a + (b - a) * i as f64 / (ns - 1) as f64 }).collect()).collect();
    let mut values = Vec::with_capacity(total);
    let mut point = vec![0.0; d];
    let (mut best, mut best_x) = (f64::INFINITY, vec![f64::NAN; d]);
    for flat in 0..total {
        let mut r = flat;
        for k in (0..d).rev() {
            point[k] = grid[k][r % ns];
            r /= ns;
        }
        let v = f(&point);
        // The first of equal minima, as NumPy's argmin.
        if v < best || best_x[0].is_nan() {
            best = v;
            best_x.clone_from(&point);
        }
        values.push(v);
    }
    if finish {
        let mut nm = NelderMeadMethod::new();
        nm.set_tolerance(Some(1e-14));
        nm.set_max_iterations(Some(5000));
        if let Ok(r) = nm.gradient_free_minimize(&Wrapped(&f), best_x.clone())
            && r.objective_value < best
        {
            best = r.objective_value;
            best_x = r.parameters;
        }
    }
    Ok(BruteResult { x: best_x, fun: best, grid, values })
}
// ANCHOR_END: brute

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_then_polish() {
        // A shifted quadratic whose minimum (0.31, −0.77) is off the grid.
        let f = |x: &[f64]| (x[0] - 0.31).powi(2) + 2.0 * (x[1] + 0.77).powi(2);
        let r = brute(f, &[(-1.0, 1.0), (-1.0, 1.0)], 21, false).unwrap();
        assert_eq!(r.values.len(), 441);
        assert!((r.x[0] - 0.3).abs() < 1e-15 && (r.x[1] + 0.8).abs() < 1e-15);
        let r = brute(f, &[(-1.0, 1.0), (-1.0, 1.0)], 21, true).unwrap();
        assert!((r.x[0] - 0.31).abs() < 1e-6 && (r.x[1] + 0.77).abs() < 1e-6, "{:?}", r.x);
    }
}
