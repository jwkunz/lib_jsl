//! Powell's method: derivative-free minimisation by successive line searches along a set of
//! conjugate directions.
//!
//! Starting from the coordinate axes, each iteration minimises the objective along every direction
//! in turn, then tries the direction of the overall move. When that direction is a productive
//! replacement it takes the place of the direction that contributed the largest decrease, so after
//! `n` iterations the set is (for a quadratic) mutually conjugate and the minimum is reached
//! exactly. The line searches use bracketing followed by Brent's parabolic/golden-section method.
//!
//! Bounds, if set, are enforced by clamping every trial point into the box, so the returned
//! solution always lies inside it.

use crate::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction, OptimizationResult};
use lib_jsl_core::ErrorsJSL;

#[derive(Debug, Clone, Default)]
pub struct PowellsMethod {
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    bounds: Option<Vec<(f64, f64)>>,
}

impl PowellsMethod {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MinimizationControls for PowellsMethod {
    fn set_max_iterations(&mut self, max_iterations: Option<usize>) {
        self.max_iterations = max_iterations;
    }
    fn get_max_iterations(&self) -> Option<usize> {
        self.max_iterations
    }
    fn set_tolerance(&mut self, tolerance: Option<f64>) {
        self.tolerance = tolerance;
    }
    fn get_tolerance(&self) -> Option<f64> {
        self.tolerance
    }
    fn set_bounds(&mut self, bounds: Option<Vec<(f64, f64)>>) {
        self.bounds = bounds;
    }
    fn get_bounds(&self) -> Option<Vec<(f64, f64)>> {
        self.bounds.clone()
    }
}

fn clamp_into(point: &mut [f64], bounds: &Option<Vec<(f64, f64)>>) {
    if let Some(b) = bounds {
        for (x, &(lo, hi)) in point.iter_mut().zip(b) {
            *x = x.clamp(lo, hi);
        }
    }
}

/// Minimises `g(t)` along a line: brackets a minimum from `t = 0` and refines it with Brent's
/// method. Returns `(t, g(t))`.
fn line_search(g: &mut impl FnMut(f64) -> f64, tol: f64) -> (f64, f64) {
    const GOLD: f64 = 1.618_033_988_749_895;
    let (mut a, mut b) = (0.0, 1.0);
    let (mut fa, mut fb) = (g(a), g(b));
    if fb > fa {
        // Downhill is the other way (or the step overshot): swap so b is the lower point.
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    // Now f(b) <= f(a). Expand away from a until the function rises.
    let mut c = b + GOLD * (b - a);
    let mut fc = g(c);
    let mut guard = 0;
    while fc < fb && guard < 60 {
        let next = c + GOLD * (c - b);
        (a, b, c) = (b, c, next);
        (fa, fb, fc) = (fb, fc, g(c));
        guard += 1;
    }
    let _ = fa;
    // Brent on [min(a,c), max(a,c)] with b inside.
    let (mut lo, mut hi) = (a.min(c), a.max(c));
    let (mut x, mut w, mut v) = (b, b, b);
    let (mut fx, mut fw, mut fv) = (fb, fb, fb);
    let (mut d, mut e): (f64, f64) = (0.0, 0.0);
    for _ in 0..100 {
        let xm = 0.5 * (lo + hi);
        let tol1 = tol * x.abs() + 1e-12;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (hi - lo) {
            break;
        }
        let mut golden = true;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            if p.abs() < (0.5 * q * e).abs() && p > q * (lo - x) && p < q * (hi - x) {
                e = d;
                d = p / q;
                let u = x + d;
                if u - lo < tol2 || hi - u < tol2 {
                    d = tol1.copysign(xm - x);
                }
                golden = false;
            }
        }
        if golden {
            e = if x >= xm { lo - x } else { hi - x };
            d = 0.381_966_011_250_105_1 * e;
        }
        let u = if d.abs() >= tol1 { x + d } else { x + tol1.copysign(d) };
        let fu = g(u);
        if fu <= fx {
            if u >= x { lo = x } else { hi = x }
            (v, w, x) = (w, x, u);
            (fv, fw, fx) = (fw, fx, fu);
        } else {
            if u < x { lo = u } else { hi = u }
            if fu <= fw || w == x {
                (v, w) = (w, u);
                (fv, fw) = (fw, fu);
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    (x, fx)
}

impl<T> GradientFreeMinimizationEngine<T> for PowellsMethod
where
    T: ObjectiveFunction,
{
    fn gradient_free_minimize(&self, objective_function: &T, initial_parameters: Vec<f64>) -> Result<OptimizationResult, ErrorsJSL> {
        let n = initial_parameters.len();
        if n == 0 {
            return Err(ErrorsJSL::Misconfiguration("Initial parameters must be a non-empty vector."));
        }
        let bounds = self.get_bounds();
        if let Some(b) = &bounds {
            if b.len() != n {
                return Err(ErrorsJSL::Misconfiguration("Bounds length must match the number of parameters."));
            }
            if b.iter().any(|&(lo, hi)| lo > hi) {
                return Err(ErrorsJSL::Misconfiguration("Each bound must satisfy lower <= upper."));
            }
        }
        let max_iterations = self.get_max_iterations().unwrap_or(1000);
        let tol = self.get_tolerance().unwrap_or(1e-8);

        let mut p = initial_parameters;
        clamp_into(&mut p, &bounds);
        let mut fp = objective_function.evaluate(&p);
        if !fp.is_finite() {
            return Err(ErrorsJSL::Misconfiguration("Objective is not finite at the initial point."));
        }
        let mut dirs: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| f64::from(i == j)).collect()).collect();
        let mut iterations = 0;
        let mut converged = false;

        // Minimise along `dir` starting at `from`; returns the new point and value.
        let minimise_along = |from: &[f64], dir: &[f64]| -> (Vec<f64>, f64) {
            let trial = |t: f64| -> Vec<f64> {
                let mut x: Vec<f64> = from.iter().zip(dir).map(|(&a, &d)| a + t * d).collect();
                clamp_into(&mut x, &bounds);
                x
            };
            let mut g = |t: f64| objective_function.evaluate(&trial(t));
            let (t, ft) = line_search(&mut g, 1e-3_f64.min(tol.sqrt()).max(1e-10));
            (trial(t), ft)
        };

        while iterations < max_iterations {
            iterations += 1;
            let (p0, f0) = (p.clone(), fp);
            let (mut biggest, mut biggest_i) = (0.0, 0);
            for (i, dir) in dirs.iter().enumerate() {
                let before = fp;
                let (x, fx) = minimise_along(&p, dir);
                if fx < fp {
                    (p, fp) = (x, fx);
                }
                if before - fp > biggest {
                    biggest = before - fp;
                    biggest_i = i;
                }
            }
            if 2.0 * (f0 - fp) <= tol * (f0.abs() + fp.abs()) + 1e-300 {
                converged = true;
                break;
            }
            // Extrapolated point and the net displacement direction.
            let net: Vec<f64> = p.iter().zip(&p0).map(|(a, b)| a - b).collect();
            let mut extrapolated: Vec<f64> = p.iter().zip(&net).map(|(a, d)| a + d).collect();
            clamp_into(&mut extrapolated, &bounds);
            let fe = objective_function.evaluate(&extrapolated);
            if fe < f0 {
                let t = 2.0 * (f0 - 2.0 * fp + fe) * (f0 - fp - biggest).powi(2) - biggest * (f0 - fe).powi(2);
                if t < 0.0 && net.iter().any(|&d| d != 0.0) {
                    let (x, fx) = minimise_along(&p, &net);
                    if fx < fp {
                        (p, fp) = (x, fx);
                    }
                    let last = dirs.len() - 1;
                    dirs.swap(biggest_i, last);
                    dirs[last] = net;
                }
            }
        }
        Ok(OptimizationResult { parameters: p, objective_value: fp, iterations, converged })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fn1<F: Fn(&[f64]) -> f64>(F);
    impl<F: Fn(&[f64]) -> f64> ObjectiveFunction for Fn1<F> {
        fn evaluate(&self, p: &[f64]) -> f64 {
            (self.0)(p)
        }
    }

    #[test]
    fn quadratic_with_correlated_variables() {
        // A positive-definite skewed bowl: minimum at (1, -2, 3) with coupled coordinates.
        let f = Fn1(|p: &[f64]| {
            let (x, y, z) = (p[0] - 1.0, p[1] + 2.0, p[2] - 3.0);
            x * x + 10.0 * y * y + 5.0 * z * z + 4.0 * x * y + 3.0 * y * z + 7.0
        });
        let r = PowellsMethod::new().gradient_free_minimize(&f, vec![-4.0, 5.0, 0.0]).unwrap();
        assert!(r.converged);
        for (got, want) in r.parameters.iter().zip([1.0, -2.0, 3.0]) {
            assert!((got - want).abs() < 1e-4, "{:?}", r.parameters);
        }
        assert!((r.objective_value - 7.0).abs() < 1e-8);
    }

    #[test]
    fn rosenbrock_valley() {
        let f = Fn1(|p: &[f64]| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2));
        let mut opt = PowellsMethod::new();
        opt.set_tolerance(Some(1e-12));
        let r = opt.gradient_free_minimize(&f, vec![-1.2, 1.0]).unwrap();
        assert!((r.parameters[0] - 1.0).abs() < 1e-4 && (r.parameters[1] - 1.0).abs() < 1e-4, "{:?}", r.parameters);
    }

    #[test]
    fn bounds_keep_the_solution_inside_the_box() {
        let f = Fn1(|p: &[f64]| (p[0] - 3.0).powi(2) + (p[1] - 3.0).powi(2));
        let mut opt = PowellsMethod::new();
        opt.set_bounds(Some(vec![(-1.0, 2.0), (-1.0, 1.0)]));
        let r = opt.gradient_free_minimize(&f, vec![0.0, 0.0]).unwrap();
        assert!((r.parameters[0] - 2.0).abs() < 1e-6 && (r.parameters[1] - 1.0).abs() < 1e-6, "{:?}", r.parameters);
    }

    #[test]
    fn one_dimension_and_errors() {
        let f = Fn1(|p: &[f64]| (p[0] - 0.5).powi(4) + 1.0);
        let r = PowellsMethod::new().gradient_free_minimize(&f, vec![10.0]).unwrap();
        assert!((r.parameters[0] - 0.5).abs() < 1e-2);
        assert!(PowellsMethod::new().gradient_free_minimize(&f, vec![]).is_err());
        let mut bad = PowellsMethod::new();
        bad.set_bounds(Some(vec![(1.0, 0.0)]));
        assert!(bad.gradient_free_minimize(&f, vec![0.5]).is_err());
        bad.set_bounds(Some(vec![(0.0, 1.0), (0.0, 1.0)]));
        assert!(bad.gradient_free_minimize(&f, vec![0.5]).is_err());
        let nan = Fn1(|_: &[f64]| f64::NAN);
        assert!(PowellsMethod::new().gradient_free_minimize(&nan, vec![0.0]).is_err());
    }
}
