//! Chebyshev approximation of a function on an interval.

use lib_jsl_core::*;
use std::f64::consts::PI;

/// A Chebyshev series `f(x) ≈ c0/2 + Σ c_k T_k(t)` with `t` the interval mapped onto `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct ChebyshevApproximation {
    a: f64,
    b: f64,
    coefficients: Vec<f64>,
}

impl ChebyshevApproximation {
    /// Fits `n` Chebyshev coefficients to `f` on `[a, b]` by sampling at the Chebyshev nodes.
    pub fn new(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> Result<Self, ErrorsJSL> {
        if n == 0 || a.partial_cmp(&b) != Some(std::cmp::Ordering::Less) {
            return Err(ErrorsJSL::InvalidInputRange("need n > 0 and a < b"));
        }
        let fv: Vec<f64> = (0..n)
            .map(|k| f(0.5 * (b - a) * (PI * (k as f64 + 0.5) / n as f64).cos() + 0.5 * (b + a)))
            .collect();
        let coefficients = (0..n)
            .map(|j| {
                let s: f64 = (0..n).map(|k| fv[k] * (PI * j as f64 * (k as f64 + 0.5) / n as f64).cos()).sum();
                2.0 * s / n as f64
            })
            .collect();
        Ok(ChebyshevApproximation { a, b, coefficients })
    }

    /// The series coefficients `c_0..c_{n-1}` (note the half weight on `c_0`).
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Drops trailing coefficients smaller than `tol` in magnitude (keeping at least one).
    pub fn truncate(&mut self, tol: f64) {
        while self.coefficients.len() > 1 && self.coefficients.last().is_some_and(|c| c.abs() < tol) {
            self.coefficients.pop();
        }
    }

    /// Evaluates the series with Clenshaw's recurrence.
    pub fn evaluate(&self, x: f64) -> f64 {
        let t = (2.0 * x - self.a - self.b) / (self.b - self.a);
        let (mut d, mut dd) = (0.0, 0.0);
        for &c in self.coefficients[1..].iter().rev() {
            let sv = d;
            d = 2.0 * t * d - dd + c;
            dd = sv;
        }
        t * d - dd + 0.5 * self.coefficients[0]
    }

    /// The Chebyshev series of the derivative.
    pub fn derivative(&self) -> ChebyshevApproximation {
        let n = self.coefficients.len();
        // d/dt T_j = 2 j Σ T_k over k = j-1, j-3, ... (k = 0 weighted 1/2), which gives
        // der[j-1] = der[j+1] + 2 j c[j].
        let mut der = vec![0.0; n + 1];
        for j in (1..n).rev() {
            der[j - 1] = der[j + 1] + 2.0 * j as f64 * self.coefficients[j];
        }
        der.truncate(n.saturating_sub(1).max(1));
        let scale = 2.0 / (self.b - self.a);
        for c in der.iter_mut() {
            *c *= scale;
        }
        ChebyshevApproximation { a: self.a, b: self.b, coefficients: der }
    }

    /// The Chebyshev series of the antiderivative that vanishes at `a`.
    pub fn integral(&self) -> ChebyshevApproximation {
        let n = self.coefficients.len();
        let con = 0.25 * (self.b - self.a);
        let c = |k: usize| self.coefficients.get(k).copied().unwrap_or(0.0);
        let mut out = vec![0.0; n + 1];
        let mut sum = 0.0;
        let mut fac = 1.0;
        for (j, o) in out.iter_mut().enumerate().skip(1) {
            *o = con * (c(j - 1) - c(j + 1)) / j as f64;
            sum += fac * *o;
            fac = -fac;
        }
        // Constant of integration so that the antiderivative vanishes at t = -1 (x = a).
        out[0] = 2.0 * sum;
        ChebyshevApproximation { a: self.a, b: self.b, coefficients: out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approximates_smooth_functions_to_machine_precision() {
        let c = ChebyshevApproximation::new(f64::exp, -1.0, 2.0, 24).unwrap();
        for k in 0..=30 {
            let x = -1.0 + 3.0 * k as f64 / 30.0;
            assert!((c.evaluate(x) - x.exp()).abs() < 1e-13, "x={x}");
        }
        let mut t = c.clone();
        t.truncate(1e-14);
        assert!(t.coefficients().len() < 22);
    }

    #[test]
    fn derivative_and_integral() {
        let c = ChebyshevApproximation::new(f64::sin, 0.0, 3.0, 30).unwrap();
        let d = c.derivative();
        let i = c.integral();
        for k in 1..10 {
            let x = k as f64 * 0.3;
            assert!((d.evaluate(x) - x.cos()).abs() < 1e-11, "d x={x}");
            assert!((i.evaluate(x) - (1.0 - x.cos())).abs() < 1e-11, "i x={x}");
        }
    }

    #[test]
    fn rejects_bad_interval() {
        assert!(ChebyshevApproximation::new(|x| x, 1.0, 1.0, 4).is_err());
    }
}
