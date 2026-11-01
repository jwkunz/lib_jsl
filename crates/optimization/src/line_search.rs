//! Line searches for gradient-based minimizers.
//!
//! See the book entry [One-dimensional minimisation and line searches](https://jwkunz.github.io/lib_jsl/book/optimization/one-dimensional.html#line-searches-and-the-wolfe-conditions).

use crate::optimization_traits::{GradientFunction, ObjectiveFunction};

/// A point accepted by a line search: `x = x₀ + alpha·d`, with its value and gradient.
#[derive(Debug, Clone)]
pub struct LineSearchPoint {
    pub alpha: f64,
    pub x: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// Finds a step `alpha > 0` along the descent direction `d` from `x0` satisfying the strong Wolfe
/// conditions
///
/// `f(x₀ + αd) ≤ f(x₀) + c₁ α ∇f(x₀)·d` and `|∇f(x₀ + αd)·d| ≤ c₂ |∇f(x₀)·d|`
///
/// (Nocedal and Wright, algorithms 3.5 and 3.6: bracketing by doubling, then zooming with
/// safeguarded cubic interpolation). Use `c₂ ≈ 0.9` for quasi-Newton methods and `c₂ ≈ 0.1` for
/// nonlinear conjugate gradients. Returns `None` if `d` is not a descent direction or no
/// decreasing point is found; if the Wolfe conditions cannot be met to rounding precision, the
/// best decreasing point found is returned.
#[allow(clippy::too_many_arguments)]
pub fn strong_wolfe<T, U>(
    objective: &T,
    gradient: &U,
    x0: &[f64],
    f0: f64,
    g0: &[f64],
    d: &[f64],
    initial_step: f64,
    c1: f64,
    c2: f64,
) -> Option<LineSearchPoint>
where
    T: ObjectiveFunction + ?Sized,
    U: GradientFunction + ?Sized,
{
    let slope0 = dot(g0, d);
    if slope0.is_nan() || slope0 >= 0.0 {
        return None;
    }
    let eval = |alpha: f64| {
        let x: Vec<f64> = x0.iter().zip(d).map(|(a, b)| a + alpha * b).collect();
        let value = objective.evaluate(&x);
        let g = gradient.evaluate(&x);
        let slope = dot(&g, d);
        (LineSearchPoint { alpha, x, value, gradient: g }, slope)
    };
    let sufficient = |p: &LineSearchPoint| p.value <= f0 + c1 * p.alpha * slope0;
    let curvature = |slope: f64| slope.abs() <= -c2 * slope0;

    let zoom = |mut lo: (LineSearchPoint, f64), mut hi: (LineSearchPoint, f64)| -> Option<LineSearchPoint> {
        for _ in 0..60 {
            let (a, b) = (lo.0.alpha, hi.0.alpha);
            let width = (b - a).abs();
            // Cubic interpolation of φ on [a, b] from values and slopes, safeguarded to the interior.
            let d1 = lo.1 + hi.1 - 3.0 * (lo.0.value - hi.0.value) / (a - b);
            let disc = d1 * d1 - lo.1 * hi.1;
            let mut t = if disc >= 0.0 {
                let d2 = (b - a).signum() * disc.sqrt();
                b - (b - a) * (hi.1 + d2 - d1) / (hi.1 - lo.1 + 2.0 * d2)
            } else {
                f64::NAN
            };
            let (left, right) = (a.min(b), a.max(b));
            if !t.is_finite() || t < left + 0.1 * width || t > right - 0.1 * width {
                t = 0.5 * (a + b);
            }
            let (p, slope) = eval(t);
            if !sufficient(&p) || p.value >= lo.0.value {
                hi = (p, slope);
            } else {
                if curvature(slope) {
                    return Some(p);
                }
                if slope * (hi.0.alpha - lo.0.alpha) >= 0.0 {
                    hi = lo;
                }
                lo = (p, slope);
            }
            if width <= 1e-14 * right.max(1e-300) {
                break;
            }
        }
        (lo.0.value < f0).then_some(lo.0)
    };

    // ANCHOR: wolfe_bracket
    let start = LineSearchPoint { alpha: 0.0, x: x0.to_vec(), value: f0, gradient: g0.to_vec() };
    let mut prev = (start, slope0);
    let mut alpha = initial_step;
    for i in 0..60 {
        let (p, slope) = eval(alpha);
        if !p.value.is_finite() {
            // Stepped outside the domain: shrink towards the last good point.
            alpha = 0.5 * (prev.0.alpha + alpha);
            continue;
        }
        if !sufficient(&p) || (i > 0 && p.value >= prev.0.value) {
            return zoom(prev, (p, slope));
        }
        if curvature(slope) {
            return Some(p);
        }
        if slope >= 0.0 {
            return zoom((p, slope), prev);
        }
        alpha *= 2.0;
        prev = (p, slope);
    }
    (prev.0.value < f0).then_some(prev.0)
    // ANCHOR_END: wolfe_bracket
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Quartic;
    impl ObjectiveFunction for Quartic {
        fn evaluate(&self, p: &[f64]) -> f64 {
            (p[0] - 3.0).powi(4) + p[1] * p[1]
        }
    }
    impl GradientFunction for Quartic {
        fn evaluate(&self, p: &[f64]) -> Vec<f64> {
            vec![4.0 * (p[0] - 3.0).powi(3), 2.0 * p[1]]
        }
    }

    #[test]
    fn accepted_steps_satisfy_the_strong_wolfe_conditions() {
        let x0 = [0.0, 1.0];
        let f0 = ObjectiveFunction::evaluate(&Quartic, &x0);
        let g0 = GradientFunction::evaluate(&Quartic, &x0);
        let d: Vec<f64> = g0.iter().map(|v| -v).collect();
        let slope0 = dot(&g0, &d);
        for (step, c2) in [(1.0, 0.9), (1e-6, 0.9), (10.0, 0.1), (1e-3, 0.1)] {
            let p = strong_wolfe(&Quartic, &Quartic, &x0, f0, &g0, &d, step, 1e-4, c2).unwrap();
            assert!(p.value <= f0 + 1e-4 * p.alpha * slope0);
            assert!(dot(&p.gradient, &d).abs() <= -c2 * slope0, "step {step} c2 {c2}");
        }
        // An ascent direction is rejected.
        assert!(strong_wolfe(&Quartic, &Quartic, &x0, f0, &g0, &g0, 1.0, 1e-4, 0.9).is_none());
    }
}
