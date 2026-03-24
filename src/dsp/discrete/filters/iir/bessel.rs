use std::f64::consts::PI;

use crate::{
    dsp::{
        continuous::filter::{
            bilinear::BilinearResult,
            bilinear_zpk::{bilinear_zpk, BilinearZpkResult},
            design::{
                besselap, lp2bp_zpk, lp2bs_zpk, lp2hp_zpk, lp2lp_zpk, zpk_to_tf, BesselNorm,
                IirFilterBandType, Zpk,
            },
        },
        discrete::filters::iir::butterworth::{ButterOutput, ButterOutputType},
    },
    prelude::ErrorsJSL,
};

fn validate_fs(fs: f64) -> Result<(), ErrorsJSL> {
    if !fs.is_finite() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be finite and > 0"));
    }
    Ok(())
}

fn validate_critical_frequencies(
    wn: &[f64],
    filter_type: IirFilterBandType,
    fs: f64,
) -> Result<(), ErrorsJSL> {
    validate_fs(fs)?;
    let expected_len = match filter_type {
        IirFilterBandType::Lowpass | IirFilterBandType::Highpass => 1,
        IirFilterBandType::Bandpass | IirFilterBandType::Bandstop => 2,
    };
    if wn.len() != expected_len {
        return Err(ErrorsJSL::InvalidInputRange(
            "critical frequency count does not match filter type",
        ));
    }
    if wn.iter().any(|x| !x.is_finite() || *x <= 0.0 || *x >= fs * 0.5) {
        return Err(ErrorsJSL::InvalidInputRange(
            "critical frequencies must be finite and satisfy 0 < Wn < fs/2",
        ));
    }
    if wn.len() == 2 && wn[0] >= wn[1] {
        return Err(ErrorsJSL::InvalidInputRange(
            "band edges must satisfy Wn[0] < Wn[1]",
        ));
    }
    Ok(())
}

fn prewarp_frequency(w: f64, fs: f64) -> f64 {
    2.0 * fs * (PI * w / fs).tan()
}

/// Build the analog Bessel design in ZPK form before the bilinear transform.
///
/// The Bessel family starts from an all-pole prototype with a chosen
/// normalization (`phase`, `delay`, or `mag`). After that, the same lowpass /
/// highpass / bandpass / bandstop analog transforms used by the other IIR
/// families apply here as well.
fn bessel_analog_zpk(
    order: usize,
    norm: BesselNorm,
    warped: &[f64],
    filter_type: IirFilterBandType,
) -> Result<Zpk, ErrorsJSL> {
    let proto = besselap(order, Some(norm))?;
    match filter_type {
        IirFilterBandType::Lowpass => lp2lp_zpk(&proto, warped[0]),
        IirFilterBandType::Highpass => lp2hp_zpk(&proto, warped[0]),
        IirFilterBandType::Bandpass => {
            let wo = (warped[0] * warped[1]).sqrt();
            let bw = warped[1] - warped[0];
            lp2bp_zpk(&proto, wo, bw)
        }
        IirFilterBandType::Bandstop => {
            let wo = (warped[0] * warped[1]).sqrt();
            let bw = warped[1] - warped[0];
            lp2bs_zpk(&proto, wo, bw)
        }
    }
}

fn digital_zpk_to_ba(zpk: &BilinearZpkResult) -> Result<BilinearResult, ErrorsJSL> {
    let tf = zpk_to_tf(&Zpk {
        z: zpk.z.clone(),
        p: zpk.p.clone(),
        k: zpk.k,
    })?;
    // Keep the design in ZPK form until the end. That is especially natural for
    // Bessel filters because the prototype begins life as a pole set derived
    // from the reverse Bessel polynomial.
    let a0 = tf.a[0];
    if a0.norm_sqr() <= 1e-24 {
        return Err(ErrorsJSL::RuntimeError(
            "digital transfer function has zero leading denominator coefficient",
        ));
    }
    Ok(BilinearResult {
        beta: tf.b.mapv(|x| x / a0),
        alpha: tf.a.mapv(|x| x / a0),
    })
}

/// Design a digital Bessel/Thomson filter in a style similar to SciPy's
/// `bessel`.
///
/// This module follows the same user-facing conventions as the rest of the
/// crate's digital IIR designers:
/// - frequencies are normalized with `fs = 2.0` when `fs` is omitted
/// - `wn` is a single critical frequency for low/high-pass or a frequency pair
///   for band-pass / band-stop
/// - the result can be returned in either normalized `ba` form or `zpk` form
///
/// Unlike Butterworth/Chebyshev/Elliptic, the key extra knob is `norm`, which
/// determines how the analog Bessel prototype is scaled before the bilinear
/// transform. This matters because Bessel filters are primarily about delay and
/// phase shape rather than stopband sharpness.
pub fn bessel(
    order: usize,
    wn: &[f64],
    filter_type: IirFilterBandType,
    output: Option<ButterOutputType>,
    norm: Option<BesselNorm>,
    fs: Option<f64>,
) -> Result<ButterOutput, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    let fs = fs.unwrap_or(2.0);
    validate_critical_frequencies(wn, filter_type, fs)?;

    // As with the other digital IIR designers, prewarp before the analog
    // prototype work so the bilinear transform lands the cutoff in the expected
    // place.
    let warped = wn.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let analog = bessel_analog_zpk(
        order,
        norm.unwrap_or(BesselNorm::Phase),
        &warped,
        filter_type,
    )?;
    let digital_zpk = bilinear_zpk(
        analog.z.as_slice().unwrap_or(&[]),
        analog.p.as_slice().unwrap_or(&[]),
        analog.k,
        fs,
    )?;

    match output.unwrap_or(ButterOutputType::Ba) {
        ButterOutputType::Ba => Ok(ButterOutput::Ba(digital_zpk_to_ba(&digital_zpk)?)),
        ButterOutputType::Zpk => Ok(ButterOutput::Zpk(digital_zpk)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dsp::discrete::spectral::freqz::{freqz, FreqzWorN},
        prelude::C1D,
    };
    use num::Complex;

    fn response_mag_db(result: &BilinearResult, w: f64, fs: f64) -> f64 {
        let fr = freqz(
            result.beta.as_slice().unwrap_or(&[]),
            Some(result.alpha.as_slice().unwrap_or(&[])),
            Some(FreqzWorN::Frequencies(vec![w])),
            None,
            Some(fs),
            None,
        )
        .unwrap();
        20.0 * fr.h[0].norm().log10()
    }

    #[test]
    fn test_bessel_lowpass_ba_smoke() {
        let result = bessel(
            4,
            &[0.2],
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
            None,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = result else {
            panic!("expected ba output");
        };
        assert_eq!(ba.beta.len(), 5);
        assert_eq!(ba.alpha.len(), 5);
        assert!((ba.alpha[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_bessel_highpass_zpk_has_plus_one_zeros() {
        let result = bessel(
            4,
            &[0.3],
            IirFilterBandType::Highpass,
            Some(ButterOutputType::Zpk),
            None,
            None,
        )
        .unwrap();
        let ButterOutput::Zpk(zpk) = result else {
            panic!("expected zpk output");
        };
        assert_eq!(zpk.z.len(), 4);
        assert!(zpk.z.iter().all(|z| (*z - Complex::new(1.0, 0.0)).norm() < 1e-9));
    }

    #[test]
    fn test_bessel_mag_norm_is_near_minus_three_db_at_wn() {
        let result = bessel(
            5,
            &[0.2],
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
            Some(BesselNorm::Mag),
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = result else {
            panic!("expected ba output");
        };
        let db = response_mag_db(&ba, 0.2, 2.0);
        assert!((db + 3.0).abs() < 0.3);
    }

    #[test]
    fn test_bessel_bandpass_ba_smoke() {
        let result = bessel(
            3,
            &[0.25, 0.45],
            IirFilterBandType::Bandpass,
            Some(ButterOutputType::Ba),
            None,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = result else {
            panic!("expected ba output");
        };
        assert_eq!(ba.beta.len(), 7);
        assert_eq!(ba.alpha.len(), 7);
    }

    #[test]
    fn test_bessel_bandstop_ba_smoke() {
        let result = bessel(
            3,
            &[0.2, 0.5],
            IirFilterBandType::Bandstop,
            Some(ButterOutputType::Ba),
            None,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = result else {
            panic!("expected ba output");
        };
        assert_eq!(ba.beta.len(), 7);
        assert_eq!(ba.alpha.len(), 7);
    }

    #[test]
    fn test_bessel_phase_and_delay_norm_produce_different_coefficients() {
        let ButterOutput::Ba(phase) = bessel(
            4,
            &[0.2],
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
            Some(BesselNorm::Phase),
            None,
        )
        .unwrap()
        else {
            panic!("expected ba output");
        };
        let ButterOutput::Ba(delay) = bessel(
            4,
            &[0.2],
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
            Some(BesselNorm::Delay),
            None,
        )
        .unwrap()
        else {
            panic!("expected ba output");
        };

        let diff: C1D = &phase.alpha - &delay.alpha;
        assert!(diff.iter().any(|x| x.norm() > 1e-6));
    }
}
