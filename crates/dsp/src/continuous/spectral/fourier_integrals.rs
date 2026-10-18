//! Numerical Fourier integrals of functions with finite support.
//!
//! The transform follows the angular-frequency convention
//! `F(omega) = integral f(x) exp(-i omega x) dx`, with inverse
//! `f(x) = (1 / 2 pi) integral F(omega) exp(i omega x) d omega`.
//! The integral is approximated by the trapezoidal rule on `M` evenly spaced
//! samples over the support. A chirp-z transform evaluates the requested
//! evenly spaced frequencies using the crate's `BestFft` implementation.

use num::Complex;
use std::f64::consts::TAU;

use crate::{
    discrete::spectral::chirpz::chirpz,
    prelude::{C1D, ErrorsJSL, R1D},
};

/// Polarity and normalization convention for a Fourier integral.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FourierDirection {
    /// `integral f(x) exp(-i omega x) dx`.
    Forward,
    /// `(1 / 2 pi) integral f(x) exp(i omega x) dx`.
    Inverse,
}

/// Samples of a Fourier integral on an angular-frequency grid.
#[derive(Clone, Debug, PartialEq)]
pub struct FourierIntegralResult {
    /// Angular-frequency samples in rad per unit of `x`.
    pub omega: R1D,
    /// Transform values corresponding to `omega`.
    pub values: C1D,
}

/// Approximate the Fourier integral of `f` over its finite support.
///
/// `M` is both the number of equally spaced samples used to approximate the
/// integral on `[x_start, x_stop]` and the number of equally spaced output
/// frequencies on `[omega_start, omega_stop]`. For `M == 1`, the sole
/// frequency is `omega_start` and the function is sampled at the midpoint of
/// its support. For larger `M`, the support endpoints are included and the
/// trapezoidal rule is used.
///
/// The frequency-domain evaluation uses the chirp-z transform, which is
/// implemented through FFT convolution with `BestFft`. This permits arbitrary
/// frequency ranges, not only the standard FFT bins.
pub fn fourier_integral(
    f: fn(f64) -> Complex<f64>,
    x_start: f64,
    x_stop: f64,
    omega_start: f64,
    omega_stop: f64,
    m: usize,
    direction: FourierDirection,
) -> Result<FourierIntegralResult, ErrorsJSL> {
    if !x_start.is_finite()
        || !x_stop.is_finite()
        || x_stop < x_start
        || !omega_start.is_finite()
        || !omega_stop.is_finite()
    {
        return Err(ErrorsJSL::InvalidInputRange(
            "integration and frequency bounds must be finite and ordered",
        ));
    }
    if m == 0 {
        return Err(ErrorsJSL::InvalidInputRange("M must be > 0"));
    }

    let omega = R1D::from_vec(if m == 1 {
        vec![omega_start]
    } else {
        (0..m)
            .map(|k| omega_start + (omega_stop - omega_start) * k as f64 / (m - 1) as f64)
            .collect()
    });

    if x_start == x_stop {
        return Ok(FourierIntegralResult {
            omega,
            values: C1D::from_elem(m, Complex::new(0.0, 0.0)),
        });
    }

    let support_width = x_stop - x_start;
    let (sample_spacing, samples) = if m == 1 {
        (
            support_width,
            vec![f((x_start + x_stop) * 0.5)],
        )
    } else {
        let dx = support_width / (m - 1) as f64;
        let samples = (0..m)
            .map(|n| {
                let x = x_start + n as f64 * dx;
                let weight = if n == 0 || n + 1 == m { 0.5 } else { 1.0 };
                f(x) * weight
            })
            .collect::<Vec<_>>();
        (dx, samples)
    };

    let polarity = match direction {
        FourierDirection::Forward => -1.0,
        FourierDirection::Inverse => 1.0,
    };
    let delta_omega = if m == 1 {
        0.0
    } else {
        (omega_stop - omega_start) / (m - 1) as f64
    };
    let chirp_angle = polarity * delta_omega * sample_spacing;
    let chirp_step = Complex::new(chirp_angle.cos(), chirp_angle.sin());
    let phased_samples = samples
        .iter()
        .enumerate()
        .map(|(n, sample)| {
            let x = if m == 1 {
                (x_start + x_stop) * 0.5
            } else {
                x_start + n as f64 * sample_spacing
            };
            let phase = polarity * omega_start * x;
            *sample * Complex::new(phase.cos(), phase.sin())
        })
        .collect::<Vec<_>>();
    let mut values = chirpz(&phased_samples, Some(m), Some(chirp_step), None)?;

    let normalization = sample_spacing
        * match direction {
            FourierDirection::Forward => 1.0,
            FourierDirection::Inverse => 1.0 / TAU,
        };
    for (k, value) in values.iter_mut().enumerate() {
        let x_origin_phase = polarity * delta_omega * k as f64 * x_start;
        *value *= Complex::new(x_origin_phase.cos(), x_origin_phase.sin()) * normalization;
    }

    Ok(FourierIntegralResult { omega, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(_: f64) -> Complex<f64> {
        Complex::new(1.0, 0.0)
    }

    fn linear(x: f64) -> Complex<f64> {
        Complex::new(x, 0.5 * x)
    }

    fn assert_close(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual - expected).norm() < tol,
            "actual={actual}, expected={expected}, tol={tol}"
        );
    }

    #[test]
    fn test_forward_integral_uses_requested_frequencies() {
        let result = fourier_integral(
            linear,
            -1.0,
            1.0,
            -2.0,
            2.0,
            5,
            FourierDirection::Forward,
        )
        .unwrap();

        assert_eq!(result.omega.to_vec(), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        let dx = 0.5;
        for (k, &omega) in result.omega.iter().enumerate() {
            let mut expected = Complex::new(0.0, 0.0);
            for n in 0..5 {
                let x = -1.0 + n as f64 * dx;
                let weight = if n == 0 || n == 4 { 0.5 } else { 1.0 };
                let phase = -omega * x;
                expected += linear(x) * Complex::new(phase.cos(), phase.sin()) * weight * dx;
            }
            assert_close(result.values[k], expected, 1e-10);
        }
    }

    #[test]
    fn test_inverse_integral_uses_positive_polarity_and_normalization() {
        let result = fourier_integral(
            constant,
            0.0,
            2.0,
            0.0,
            0.0,
            3,
            FourierDirection::Inverse,
        )
        .unwrap();

        for value in result.values {
            assert_close(value, Complex::new(2.0 / TAU, 0.0), 1e-12);
        }
    }

    #[test]
    fn test_single_frequency_uses_midpoint_rule() {
        let result = fourier_integral(
            constant,
            2.0,
            4.0,
            3.0,
            8.0,
            1,
            FourierDirection::Forward,
        )
        .unwrap();

        assert_eq!(result.omega.to_vec(), vec![3.0]);
        let phase = -9.0_f64;
        assert_close(
            result.values[0],
            Complex::new(2.0 * phase.cos(), 2.0 * phase.sin()),
            1e-12,
        );
    }

    #[test]
    fn test_zero_width_support_returns_zero_integral() {
        let result = fourier_integral(
            constant,
            1.0,
            1.0,
            -1.0,
            1.0,
            3,
            FourierDirection::Forward,
        )
        .unwrap();

        assert_eq!(result.values, C1D::zeros(3));
    }

    #[test]
    fn test_rejects_invalid_bounds_and_sample_count() {
        assert!(matches!(
            fourier_integral(
                constant,
                2.0,
                1.0,
                0.0,
                1.0,
                3,
                FourierDirection::Forward,
            ),
            Err(ErrorsJSL::InvalidInputRange(_))
        ));
        assert!(matches!(
            fourier_integral(
                constant,
                0.0,
                1.0,
                0.0,
                1.0,
                0,
                FourierDirection::Forward,
            ),
            Err(ErrorsJSL::InvalidInputRange(_))
        ));
    }
}