//! FIR differentiators: smoothed central differences of any order.
//!
//! See the book entry [FIR design: windows, frequency sampling and Parks-McClellan](https://jwkunz.github.io/lib_jsl/book/filters/fir-design.html#differentiators-and-integrators).

use crate::{
    dsp::discrete::{convolve::{ConvolveMethod, ConvolveMode, convolve}, sinc::generate_centered_sinc},
    prelude::ErrorsJSL,
};

/// Taps of a smoothed `derivative_order`-th derivative filter of length `numtaps`.
///
/// A smoothing kernel of `numtaps − 2·derivative_order` points (a sinc taper from
/// [`generate_centered_sinc`] with parameter `cutoff`, default 0.9, normalised to unit sum) is
/// convolved `derivative_order` times with the central difference `[1/2, 0, −1/2]`. The taps are
/// for convolution (`y = h ∗ x`); for a polynomial input of degree `derivative_order` the output is
/// its exact derivative, delayed by `(numtaps − 1)/2` samples, in units of the sample spacing.
pub fn generate_derivative_taps(numtaps: usize, derivative_order: usize, cutoff: Option<f64>) -> Result<Vec<f64>, ErrorsJSL> {
    if derivative_order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("derivative_order must be at least 1"));
    }
    if numtaps < 2 * derivative_order + 1 {
        return Err(ErrorsJSL::InvalidInputRange("numtaps must be at least 2 * derivative_order + 1"));
    }
    let smoothing = generate_centered_sinc(numtaps - 2 * derivative_order, cutoff.unwrap_or(0.9));
    let total: f64 = smoothing.iter().sum();
    let mut taps: Vec<f64> = smoothing.into_iter().map(|v| v / total).collect();
    let central_difference = [0.5, 0.0, -0.5];
    for _ in 0..derivative_order {
        taps = convolve(&taps, &central_difference, ConvolveMode::Full, ConvolveMethod::Direct)?;
    }
    Ok(taps)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The filter output at the end of a long polynomial input.
    fn apply(taps: &[f64], x: impl Fn(f64) -> f64) -> f64 {
        taps.iter().enumerate().map(|(k, &h)| h * x(100.0 - k as f64)).sum()
    }

    #[test]
    fn differentiates_polynomials_exactly() {
        for numtaps in [13, 31] {
            let first = generate_derivative_taps(numtaps, 1, None).unwrap();
            assert_eq!(first.len(), numtaps);
            assert!((apply(&first, |t| 3.0 * t + 2.0) - 3.0).abs() < 1e-12);
            assert!(apply(&first, |_| 5.0).abs() < 1e-12);
            let second = generate_derivative_taps(numtaps, 2, None).unwrap();
            assert_eq!(second.len(), numtaps);
            assert!((apply(&second, |t| 0.5 * t * t - t) - 1.0).abs() < 1e-9);
            assert!(apply(&second, |t| 2.0 * t).abs() < 1e-12);
        }
        assert!(generate_derivative_taps(13, 0, None).is_err());
        assert!(generate_derivative_taps(4, 2, None).is_err());
    }
}
