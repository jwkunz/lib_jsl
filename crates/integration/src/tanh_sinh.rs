//! Tanh-sinh (double exponential) quadrature.

use crate::can_integrate::CanIntegrate1D;
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::FRAC_PI_2;

/// Integrates over a finite interval with the substitution `x = tanh(π/2 · sinh t)` (Takahasi and
/// Mori), which makes the transformed integrand decay double exponentially and lets the
/// trapezoid rule in `t` converge exponentially fast, even for integrable endpoint singularities
/// such as `1/√x` or `ln x`. The step is halved each level (reusing earlier points) until two
/// successive estimates agree to the tolerance.
///
/// Abscissae are formed from the endpoint distance `(b − a)/(1 + e^{2u})`. Points whose absolute
/// position rounds onto an endpoint are skipped by [`CanIntegrate1D::integrate`]; for
/// singularities at endpoints away from zero (such as `1/√(1 − x²)`), use
/// [`TanhSinhEngine::integrate_with_complement`], which also passes that exact distance so the
/// integrand can avoid the cancellation in `1 − x`.
#[derive(Debug, Clone)]
pub struct TanhSinhEngine {
    lower_bound: f64,
    upper_bound: f64,
    /// Relative tolerance between successive levels (default 1e-12).
    pub tolerance: f64,
    /// Maximum number of step halvings (default 10, i.e. h = 2⁻¹⁰).
    pub max_levels: usize,
}

/// Half-width of the `t` interval. Because abscissae are formed from endpoint distances, `t` can
/// run until those distances (about `e^{−π e^t / 2}`) approach the smallest normal double, which
/// captures the tail mass of strong singularities such as `x^{−1/2}`.
const T_MAX: f64 = 6.0;

impl TanhSinhEngine {
    /// A tanh-sinh integrator over `[lower_bound, upper_bound]` with the default tolerance and
    /// level limit.
    ///
    /// ```
    /// use lib_jsl_integration::tanh_sinh::TanhSinhEngine;
    ///
    /// // ∫₀¹ ln(x) dx = −1, with a logarithmic singularity at x = 0.
    /// let (value, _) = TanhSinhEngine::new(0.0, 1.0).integrate_with_error(f64::ln).unwrap();
    /// assert!((value + 1.0).abs() < 1e-12);
    /// ```
    pub fn new(lower_bound: f64, upper_bound: f64) -> Self {
        TanhSinhEngine { lower_bound, upper_bound, tolerance: 1e-12, max_levels: 10 }
    }

    /// The integral together with the difference between the last two levels (an error
    /// estimate, typically very pessimistic since convergence is exponential).
    pub fn integrate_with_error<F: Fn(f64) -> f64>(&self, f: F) -> Result<(f64, f64), ErrorsJSL> {
        let (a, b) = (self.lower_bound, self.upper_bound);
        self.integrate_with_complement(|x, _| if x == a || x == b { 0.0 } else { f(x) })
    }

    /// Integrates `f(x, c)`, where `c` is the signed offset of `x` from its nearer endpoint
    /// (`x − a` in the lower half, `x − b` in the upper half), computed without rounding. Returns
    /// the integral and an error estimate.
    pub fn integrate_with_complement<F: Fn(f64, f64) -> f64>(&self, f: F) -> Result<(f64, f64), ErrorsJSL> {
        let (a, b) = (self.lower_bound, self.upper_bound);
        if !a.is_finite() || !b.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("Tanh-sinh quadrature needs finite bounds."));
        }
        if a == b {
            return Ok((0.0, 0.0));
        }
        let width = b - a;
        // ANCHOR: tanh_sinh_pair
        // Contribution of the symmetric pair ±t, scaled by the weight.
        let pair = |t: f64| -> f64 {
            let u = FRAC_PI_2 * t.sinh();
            let cosh_u = u.cosh();
            let weight = FRAC_PI_2 * t.cosh() / (cosh_u * cosh_u);
            if weight == 0.0 || !weight.is_finite() {
                return 0.0;
            }
            let offset = width / (1.0 + (2.0 * u).exp());
            if offset == 0.0 {
                return 0.0;
            }
            let mut s = f(a + offset, offset) + f(b - offset, -offset);
            if t == 0.0 {
                s *= 0.5;
            }
            weight * s
        };
        // ANCHOR_END: tanh_sinh_pair
        // ANCHOR: tanh_sinh_levels
        let mut h = 1.0;
        let mut sum: f64 = (0..=(T_MAX as usize)).map(|k| pair(k as f64)).sum();
        let mut estimate = 0.5 * width * h * sum;
        let mut error = f64::INFINITY;
        for _ in 0..self.max_levels {
            h *= 0.5;
            // New points are the odd multiples of the halved step.
            let count = (T_MAX / h) as usize;
            let new: f64 = (1..=count).step_by(2).map(|k| pair(k as f64 * h)).sum();
            sum += new;
            let next = 0.5 * width * h * sum;
            error = (next - estimate).abs();
            estimate = next;
            if error <= self.tolerance * estimate.abs() {
                break;
            }
        }
        // ANCHOR_END: tanh_sinh_levels
        Ok((estimate, error))
    }
}

impl CanIntegrate1D for TanhSinhEngine {
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
        self.integrate_with_error(f).map(|(v, _)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn handles_endpoint_singularities_to_near_machine_precision() {
        type Case<'a> = (&'a dyn Fn(f64) -> f64, f64, f64, f64);
        let cases: [Case; 5] = [
            (&|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 2.0),
            (&|x: f64| x.ln(), 0.0, 1.0, -1.0),
            (&|x: f64| (x * (1.0 - x)).ln(), 0.0, 1.0, -2.0),
            (&|x: f64| x.exp(), 0.0, 1.0, std::f64::consts::E - 1.0),
            (&|x: f64| 1.0 / (1.0 + x * x), -3.0, 5.0, 5f64.atan() + 3f64.atan()),
        ];
        for (i, (f, a, b, want)) in cases.iter().enumerate() {
            let (got, err) = TanhSinhEngine::new(*a, *b).integrate_with_error(f).unwrap();
            assert!((got - want).abs() < 1e-12 * want.abs().max(1.0), "case {i}: {got} vs {want} (est {err})");
        }
    }

    #[test]
    fn complement_form_resolves_singularities_at_nonzero_endpoints() {
        // 1/√(1 − x²) = 1/√((1 − x)(1 + x)); the offset c gives 1 ∓ x exactly near ±1.
        let engine = TanhSinhEngine::new(-1.0, 1.0);
        let (got, _) = engine
            .integrate_with_complement(|x, c| {
                let (one_minus, one_plus) = if c < 0.0 { (-c, 1.0 + x) } else { (1.0 - x, c) };
                1.0 / (one_minus * one_plus).sqrt()
            })
            .unwrap();
        assert!((got - PI).abs() < 1e-13, "{got}");
        // The plain form loses only the unresolvable tail next to the endpoints.
        let plain = engine.integrate(|x| 1.0 / (1.0 - x * x).sqrt()).unwrap();
        assert!((plain - PI).abs() < 1e-6, "{plain}");
    }

    #[test]
    fn reversed_and_degenerate_intervals() {
        let mut e = TanhSinhEngine::new(2.0, 0.0);
        assert!((e.integrate(|x| x * x).unwrap() + 8.0 / 3.0).abs() < 1e-13);
        e.set_upper_bound(2.0);
        assert_eq!(e.integrate(|x| x).unwrap(), 0.0);
        assert!(TanhSinhEngine::new(0.0, f64::INFINITY).integrate(|x| x).is_err());
    }
}
