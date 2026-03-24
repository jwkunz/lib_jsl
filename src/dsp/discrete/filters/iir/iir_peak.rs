use std::f64::consts::PI;

use num::Complex;

use crate::{
    dsp::{
        continuous::filter::bilinear::BilinearResult,
        discrete::spectral::freqz::{freqz, FreqzWorN},
    },
    prelude::ErrorsJSL,
};

/// Design a second-order IIR peak (resonant) filter in a style similar to
/// SciPy's `iirpeak`.
///
/// The filter is centered at `w0` and uses the quality-factor definition
/// `Q = w0 / bw`, where `bw` is the `-3 dB` bandwidth around the resonant
/// frequency.
///
/// Frequency convention:
/// - if `fs` is omitted, `fs = 2.0` is assumed
/// - under that convention, valid `w0` values lie in `(0, 1)`, where `1`
///   corresponds to Nyquist
///
/// The returned coefficients are normalized so `alpha[0] = 1`, matching the
/// rest of the crate's digital IIR helpers and SciPy's transfer-function
/// output.
pub fn iirpeak(w0: f64, q: f64, fs: Option<f64>) -> Result<BilinearResult, ErrorsJSL> {
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

    // Match SciPy's convention by first converting to normalized half-cycles
    // per sample and deriving the -3 dB bandwidth from the requested quality
    // factor.
    let normalized_w0 = 2.0 * w0 / fs;
    let normalized_bw = normalized_w0 / q;

    // The closed-form Orfanidis design is stated in radians per sample.
    let bw = normalized_bw * PI;
    let w0_rad = normalized_w0 * PI;

    // As with the notch filter, the chosen -3 dB convention simplifies the
    // intermediate formulas considerably.
    let beta = (bw * 0.5).tan();
    let gain = 1.0 / (1.0 + beta);

    // In the peak case, the numerator no longer places zeros at the resonance.
    // Instead it creates a narrow band-pass / resonant shape around `w0`, while
    // the denominator poles set the sharpness through `Q`.
    let beta_coeffs = vec![
        Complex::new(1.0 - gain, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(gain - 1.0, 0.0),
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
    fn test_iirpeak_returns_second_order_coefficients() {
        let result = iirpeak(0.25, 30.0, None).unwrap();
        assert_eq!(result.beta.len(), 3);
        assert_eq!(result.alpha.len(), 3);
        assert!((result.alpha[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_iirpeak_has_unity_gain_at_center_frequency() {
        let result = iirpeak(0.25, 30.0, None).unwrap();
        let mag = response_mag(&result, 0.25, 2.0);
        assert!((mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iirpeak_rejects_dc_reasonably_well() {
        let result = iirpeak(0.25, 30.0, None).unwrap();
        let mag = response_mag(&result, 1e-4, 2.0);
        assert!(mag < 0.05);
    }

    #[test]
    fn test_iirpeak_rejects_invalid_inputs() {
        assert!(iirpeak(0.0, 30.0, None).is_err());
        assert!(iirpeak(0.25, 0.0, None).is_err());
        assert!(iirpeak(1.0, 30.0, Some(2.0)).is_err());
    }
}
