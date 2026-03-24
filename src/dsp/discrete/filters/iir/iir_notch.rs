use std::f64::consts::PI;

use num::Complex;

use crate::{
    dsp::continuous::filter::bilinear::BilinearResult,
    prelude::ErrorsJSL,
};

/// Design a second-order IIR notch filter in a style similar to SciPy's
/// `iirnotch`.
///
/// The filter is centered at `w0` and uses the standard quality-factor
/// relation `Q = w0 / bw`, where `bw` is the `-3 dB` bandwidth around the
/// notch frequency.
///
/// Frequency convention:
/// - if `fs` is omitted, `fs = 2.0` is assumed
/// - under that convention, valid `w0` values lie in `(0, 1)`, where `1`
///   corresponds to Nyquist
///
/// The returned coefficients are normalized so `alpha[0] = 1`, matching the
/// crate's other digital IIR helpers and SciPy's transfer-function output.
pub fn iirnotch(w0: f64, q: f64, fs: Option<f64>) -> Result<BilinearResult, ErrorsJSL> {
    let fs = fs.unwrap_or(2.0);
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

    // SciPy first converts the user-facing frequency into its normalized
    // half-cycle/sample form and derives the bandwidth from `Q = w0 / bw`.
    let normalized_w0 = 2.0 * w0 / fs;
    let normalized_bw = normalized_w0 / q;

    // The Orfanidis/SciPy implementation then works in radians per sample.
    let bw = normalized_bw * PI;
    let w0_rad = normalized_w0 * PI;

    // For the notch case, the -3 dB bandwidth simplification gives a compact
    // expression for `beta`. This is what SciPy uses internally.
    let beta = (bw * 0.5).tan();
    let gain = 1.0 / (1.0 + beta);

    // Numerator zeros sit exactly on the unit circle at the notch frequency.
    // The denominator poles pull slightly inward according to `Q`, which sets
    // the notch sharpness.
    let beta_coeffs = vec![
        Complex::new(gain, 0.0),
        Complex::new(-2.0 * gain * w0_rad.cos(), 0.0),
        Complex::new(gain, 0.0),
    ];
    let alpha_coeffs = vec![
        Complex::new(1.0, 0.0),
        Complex::new(-2.0 * gain * w0_rad.cos(), 0.0),
        Complex::new(2.0 * gain - 1.0, 0.0),
    ];

    Ok(BilinearResult {
        beta: ndarray::Array1::from_vec(beta_coeffs),
        alpha: ndarray::Array1::from_vec(alpha_coeffs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::discrete::spectral::freqz::{freqz, FreqzWorN};

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
    fn test_iirnotch_returns_second_order_coefficients() {
        let result = iirnotch(0.25, 30.0, None).unwrap();
        assert_eq!(result.beta.len(), 3);
        assert_eq!(result.alpha.len(), 3);
        assert!((result.alpha[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_iirnotch_has_deep_notch_at_center_frequency() {
        let result = iirnotch(0.25, 30.0, None).unwrap();
        let mag = response_mag(&result, 0.25, 2.0);
        assert!(mag < 1e-6);
    }

    #[test]
    fn test_iirnotch_preserves_dc_reasonably_well() {
        let result = iirnotch(0.25, 30.0, None).unwrap();
        let mag = response_mag(&result, 1e-4, 2.0);
        assert!((mag - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_iirnotch_rejects_invalid_inputs() {
        assert!(iirnotch(0.0, 30.0, None).is_err());
        assert!(iirnotch(0.25, 0.0, None).is_err());
        assert!(iirnotch(1.0, 30.0, Some(2.0)).is_err());
    }
}
