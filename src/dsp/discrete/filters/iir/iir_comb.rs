use std::f64::consts::PI;

use num::Complex;

use crate::{
    dsp::{
        continuous::filter::bilinear::BilinearResult,
        discrete::spectral::freqz::{freqz, FreqzWorN},
    },
    prelude::ErrorsJSL,
};

/// Comb-filter flavor for [`iircomb`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IirCombFilterType {
    Notch,
    Peak,
}

/// Design a digital IIR comb filter in a style similar to SciPy's `iircomb`.
///
/// A comb filter is built from a single repeated delay `N`, which creates
/// regularly spaced notches or peaks across the spectrum.
///
/// Parameters:
/// - `w0` is the fundamental spacing between comb features
/// - `q` is the quality factor, with `Q = w0 / bw`
/// - `ftype` chooses whether the repeated features are notches or peaks
/// - `pass_zero` toggles whether DC lies on a pass feature or a reject feature
///
/// Frequency convention:
/// - if `fs` is omitted, `fs = 2.0` is assumed
/// - under that convention, valid `w0` values lie in `(0, 1)`, where `1`
///   corresponds to Nyquist
///
/// Important constraint:
/// - `w0` must evenly divide `fs`, because the comb period is implemented with
///   an integer delay `N = fs / w0`
///
/// The returned coefficients are normalized so `alpha[0] = 1`.
pub fn iircomb(
    w0: f64,
    q: f64,
    ftype: Option<IirCombFilterType>,
    fs: Option<f64>,
    pass_zero: Option<bool>,
) -> Result<BilinearResult, ErrorsJSL> {
    let fs = fs.unwrap_or(2.0);
    let ftype = ftype.unwrap_or(IirCombFilterType::Notch);
    let pass_zero = pass_zero.unwrap_or(false);

    if !fs.is_finite() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be finite and > 0"));
    }
    if !w0.is_finite() || w0 <= 0.0 || w0 >= fs * 0.5 {
        return Err(ErrorsJSL::InvalidInputRange(
            "w0 must be finite and satisfy 0 < w0 < fs/2",
        ));
    }
    if !q.is_finite() || q <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("q must be finite and > 0"));
    }

    // The comb implementation is based on an integer delay length. That means
    // the requested spacing must divide the sampling rate exactly, within a
    // tiny floating-point tolerance.
    let n = (fs / w0).round();
    if n < 1.0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "fs / w0 must be >= 1 for a valid comb delay",
        ));
    }
    if ((w0 - fs / n) / fs).abs() > 1e-14 {
        return Err(ErrorsJSL::InvalidInputRange(
            "fs must be divisible by w0 for iircomb",
        ));
    }
    let n = n as usize;

    // Convert the user-facing spacing to radians/sample and derive the
    // requested 3 dB bandwidth from the quality factor.
    let w0_rad = 2.0 * PI * w0 / fs;
    let w_delta = w0_rad / q;

    // SciPy/Orfanidis parameterize both notch and peak combs with the same
    // intermediate `a`, `b`, `c` terms; only the endpoint gains differ.
    let (g0, g) = match ftype {
        IirCombFilterType::Notch => (1.0, 0.0),
        IirCombFilterType::Peak => (0.0, 1.0),
    };

    let beta = (n as f64 * w_delta * 0.25).tan();
    let ax = (1.0 - beta) / (1.0 + beta);
    let bx = (g0 + g * beta) / (1.0 + beta);
    let cx = (g0 - g * beta) / (1.0 + beta);

    // `pass_zero` controls the sign of the delayed term.
    // The sign choice is opposite for notch vs peak because the two transfer
    // functions exchange pass and reject locations.
    let negative_coef = match ftype {
        IirCombFilterType::Peak => pass_zero,
        IirCombFilterType::Notch => !pass_zero,
    };

    let sign = if negative_coef { -1.0 } else { 1.0 };

    // The numerator and denominator each contain only the present-time term
    // and the delayed term. This sparse structure is what makes the TF comb
    // numerically stable even for relatively large `N`.
    let mut beta_coeffs = vec![Complex::new(0.0, 0.0); n + 1];
    beta_coeffs[0] = Complex::new(bx, 0.0);
    beta_coeffs[n] = Complex::new(sign * cx, 0.0);

    let mut alpha_coeffs = vec![Complex::new(0.0, 0.0); n + 1];
    alpha_coeffs[0] = Complex::new(1.0, 0.0);
    alpha_coeffs[n] = Complex::new(sign * ax, 0.0);

    Ok(BilinearResult {
        beta: ndarray::Array1::from_vec(beta_coeffs),
        alpha: ndarray::Array1::from_vec(alpha_coeffs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_mag(result: &BilinearResult, w: f64, fs: f64) -> f64 {
        let fr = freqz(
            result.beta.as_slice().unwrap_or(&[]),
            Some(result.alpha.as_slice().unwrap_or(&[])),
            Some(FreqzWorN::Frequencies(vec![w])),
            None,
            Some(fs),
            None,
        )
        .unwrap();
        fr.h[0].norm()
    }

    #[test]
    fn test_iircomb_notch_returns_expected_length() {
        let result = iircomb(0.2, 30.0, None, None, None).unwrap();
        assert_eq!(result.beta.len(), 11);
        assert_eq!(result.alpha.len(), 11);
        assert!((result.alpha[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_iircomb_notch_has_null_at_spacing() {
        let result = iircomb(0.2, 30.0, Some(IirCombFilterType::Notch), None, Some(false)).unwrap();
        let mag = response_mag(&result, 0.2, 2.0);
        assert!(mag < 1e-6);
    }

    #[test]
    fn test_iircomb_notch_pass_zero_true_preserves_dc() {
        let result = iircomb(0.2, 30.0, Some(IirCombFilterType::Notch), None, Some(true)).unwrap();
        let mag = response_mag(&result, 1e-4, 2.0);
        assert!((mag - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_iircomb_peak_pass_zero_true_has_dc_peak() {
        let result = iircomb(0.25, 30.0, Some(IirCombFilterType::Peak), None, Some(true)).unwrap();
        let mag = response_mag(&result, 1e-4, 2.0);
        assert!((mag - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_iircomb_peak_pass_zero_false_has_peak_at_midpoint_series() {
        let result = iircomb(0.25, 30.0, Some(IirCombFilterType::Peak), None, Some(false)).unwrap();
        let mag = response_mag(&result, 0.125, 2.0);
        assert!((mag - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_iircomb_rejects_invalid_inputs() {
        assert!(iircomb(0.0, 30.0, None, None, None).is_err());
        assert!(iircomb(0.2, 0.0, None, None, None).is_err());
        assert!(iircomb(0.3, 30.0, None, Some(2.0), None).is_err());
        assert!(iircomb(1.0, 30.0, None, Some(2.0), None).is_err());
    }
}
