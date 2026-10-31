//! Trigonometric (Fourier) interpolation of uniformly sampled periodic data.

use lib_jsl_core::*;
use num::Complex;
use std::f64::consts::PI;

/// Interpolates `n` equally spaced samples of a periodic function with the band-limited
/// trigonometric polynomial through them. For even `n` the Nyquist term is split evenly between
/// ±n/2 so the interpolant stays real.
///
/// Coefficients are computed with a direct DFT (O(n²), once); each evaluation costs O(n).
#[derive(Debug, Clone)]
pub struct FourierInterpolator1D {
    x0: f64,
    period: f64,
    coefficients: Vec<Complex<f64>>,
}

impl FourierInterpolator1D {
    /// `y[k]` is the function value at `x0 + k * period / n`.
    pub fn new(y: VR1D, x0: f64, period: f64) -> Result<Self, ErrorsJSL> {
        let n = y.len();
        if n == 0 || period <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("need samples and a positive period"));
        }
        let coefficients = (0..n)
            .map(|k| {
                (0..n)
                    .map(|j| Complex::from_polar(y[j], -2.0 * PI * ((k * j) % n) as f64 / n as f64))
                    .sum::<Complex<f64>>()
                    / n as f64
            })
            .collect();
        Ok(FourierInterpolator1D { x0, period, coefficients })
    }

    /// Evaluates the interpolant (periodically extended) at `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        let n = self.coefficients.len();
        let theta = 2.0 * PI * (x - self.x0) / self.period;
        let mut sum = self.coefficients[0].re;
        let half = (n - 1) / 2;
        for k in 1..=half {
            // Pair the +k and -k terms (conjugates for real data).
            let (a, b) = (self.coefficients[k], self.coefficients[n - k]);
            sum += (a * Complex::from_polar(1.0, k as f64 * theta)).re + (b * Complex::from_polar(1.0, -(k as f64) * theta)).re;
        }
        if n.is_multiple_of(2) && n > 1 {
            sum += self.coefficients[n / 2].re * (0.5 * n as f64 * theta).cos();
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_bandlimited_signals_off_grid() {
        let n = 16;
        let period = 2.0;
        let f = |x: f64| 1.0 + (2.0 * PI * x / period).sin() + 0.5 * (6.0 * PI * x / period).cos();
        let y = R1D::from_iter((0..n).map(|k| f(0.3 + k as f64 * period / n as f64)));
        let fi = FourierInterpolator1D::new(y.view(), 0.3, period).unwrap();
        for k in 0..40 {
            let x = -1.0 + k as f64 * 0.137;
            assert!((fi.evaluate(x) - f(x)).abs() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn odd_length_and_exact_at_samples() {
        let n = 9;
        let y = R1D::from_iter((0..n).map(|k| ((k * k) % 5) as f64));
        let fi = FourierInterpolator1D::new(y.view(), 0.0, 9.0).unwrap();
        for k in 0..n {
            assert!((fi.evaluate(k as f64) - y[k]).abs() < 1e-12);
        }
        assert!(FourierInterpolator1D::new(y.view(), 0.0, 0.0).is_err());
    }
}
