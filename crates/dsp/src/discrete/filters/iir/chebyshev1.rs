use std::f64::consts::PI;

use crate::{
    dsp::{
        continuous::filter::{
            bilinear::BilinearResult,
            bilinear_zpk::{bilinear_zpk, BilinearZpkResult},
            design::{
                cheb1ap, lp2bp_zpk, lp2bs_zpk, lp2hp_zpk, lp2lp_zpk, zpk_to_tf,
                IirFilterBandType, Zpk,
            },
        },
        discrete::{
            filters::iir::butterworth::{ButterOutput, ButterOutputType},
        },
    },
    prelude::ErrorsJSL,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Cheb1ordResult {
    /// Minimum order predicted to satisfy the supplied passband and stopband
    /// constraints.
    pub order: usize,
    /// Critical frequency or frequency pair to feed back into [`cheby1`].
    ///
    /// For type I designs this is the passband edge after the usual digital
    /// prewarp / analog backsolve round-trip.
    pub wn: Vec<f64>,
}

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

/// Build the analog Chebyshev type I design in ZPK form before the bilinear
/// transform.
///
/// The digital designer mirrors SciPy's standard prototype flow:
/// prewarp digital edges -> create normalized analog prototype -> apply the
/// requested analog frequency transformation -> bilinear map into the z-plane.
fn cheby1_analog_zpk(
    order: usize,
    rp: f64,
    warped: &[f64],
    filter_type: IirFilterBandType,
) -> Result<Zpk, ErrorsJSL> {
    let proto = cheb1ap(order, rp)?;
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
    // Keep all of the design math in ZPK form until the very end, then
    // normalize to the common digital-transfer-function convention `a[0] = 1`.
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

/// Design a digital Chebyshev type I filter in a style similar to SciPy's
/// `cheby1`.
///
/// `wn` uses the same normalized-digital convention as the rest of this crate
/// when `fs` is omitted: `fs` defaults to `2.0`, so scalar edge frequencies
/// live on `(0, 1)` where `1` is Nyquist.
pub fn cheby1(
    order: usize,
    rp: f64,
    wn: &[f64],
    filter_type: IirFilterBandType,
    output: Option<ButterOutputType>,
    fs: Option<f64>,
) -> Result<ButterOutput, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("rp must be finite and > 0"));
    }
    let fs = fs.unwrap_or(2.0);
    validate_critical_frequencies(wn, filter_type, fs)?;

    // Prewarp the user-facing digital edges into the analog domain so the
    // subsequent bilinear transform lands the cutoff at the requested location.
    let warped = wn.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let analog = cheby1_analog_zpk(order, rp, &warped, filter_type)?;
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

fn cheb1ord_natural_frequency(
    wp: &[f64],
    ws: &[f64],
    filter_type: IirFilterBandType,
) -> Result<f64, ErrorsJSL> {
    // Reduce the full filter-spec geometry to the normalized lowpass-prototype
    // ratio used by the closed-form Chebyshev order equation.
    Ok(match filter_type {
        IirFilterBandType::Lowpass => ws[0] / wp[0],
        IirFilterBandType::Highpass => wp[0] / ws[0],
        IirFilterBandType::Bandpass => {
            let wo = (wp[0] * wp[1]).sqrt();
            let bw = wp[1] - wp[0];
            let nat1 = ((ws[0] * ws[0] - wo * wo) / (bw * ws[0])).abs();
            let nat2 = ((ws[1] * ws[1] - wo * wo) / (bw * ws[1])).abs();
            nat1.min(nat2)
        }
        IirFilterBandType::Bandstop => {
            let wo = (wp[0] * wp[1]).sqrt();
            let bw = wp[1] - wp[0];
            let nat1 = (bw * ws[0] / (wo * wo - ws[0] * ws[0])).abs();
            let nat2 = (bw * ws[1] / (wo * wo - ws[1] * ws[1])).abs();
            nat1.min(nat2)
        }
    })
}

/// Choose the minimum Chebyshev type I order meeting passband/stopband specs.
///
/// The returned `wn` is intended for direct use with [`cheby1`]. In keeping
/// with SciPy's `cheb1ord`, this comes back as the passband critical frequency,
/// not a stopband edge.
pub fn cheb1ord(
    wp: &[f64],
    ws: &[f64],
    gpass: f64,
    gstop: f64,
    filter_type: IirFilterBandType,
    fs: Option<f64>,
) -> Result<Cheb1ordResult, ErrorsJSL> {
    let fs = fs.unwrap_or(2.0);
    validate_critical_frequencies(wp, filter_type, fs)?;
    validate_critical_frequencies(ws, filter_type, fs)?;
    if !gpass.is_finite() || !gstop.is_finite() || gpass <= 0.0 || gstop <= gpass {
        return Err(ErrorsJSL::InvalidInputRange(
            "gpass and gstop must be finite and satisfy 0 < gpass < gstop",
        ));
    }

    match filter_type {
        IirFilterBandType::Lowpass => {
            if wp[0] >= ws[0] {
                return Err(ErrorsJSL::InvalidInputRange("lowpass requires wp < ws"));
            }
        }
        IirFilterBandType::Highpass => {
            if wp[0] <= ws[0] {
                return Err(ErrorsJSL::InvalidInputRange("highpass requires wp > ws"));
            }
        }
        IirFilterBandType::Bandpass => {
            if !(ws[0] < wp[0] && wp[1] < ws[1]) {
                return Err(ErrorsJSL::InvalidInputRange(
                    "bandpass requires ws[0] < wp[0] < wp[1] < ws[1]",
                ));
            }
        }
        IirFilterBandType::Bandstop => {
            if !(wp[0] < ws[0] && ws[1] < wp[1]) {
                return Err(ErrorsJSL::InvalidInputRange(
                    "bandstop requires wp[0] < ws[0] < ws[1] < wp[1]",
                ));
            }
        }
    }

    // Order selection is done in the analog domain because the Chebyshev
    // formulas are stated there. The digital frequencies are only a user-facing
    // interface layer around that calculation.
    let wp_warped = wp.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let ws_warped = ws.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let nat = cheb1ord_natural_frequency(&wp_warped, &ws_warped, filter_type)?;
    if !nat.is_finite() || nat <= 1.0 {
        return Err(ErrorsJSL::RuntimeError(
            "failed to determine a valid Chebyshev natural frequency ratio",
        ));
    }

    let gp = 10.0_f64.powf(0.1 * gpass) - 1.0;
    let gs = 10.0_f64.powf(0.1 * gstop) - 1.0;
    let order = ((gs / gp).sqrt().acosh() / nat.acosh()).ceil() as usize;

    Ok(Cheb1ordResult {
        order,
        wn: wp.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::spectral::freqz::{freqz, FreqzWorN};
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
    fn test_cheby1_lowpass_ba_smoke() {
        let result = cheby1(
            3,
            1.0,
            &[0.2],
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = result else {
            panic!("expected ba output");
        };
        assert_eq!(ba.beta.len(), 4);
        assert_eq!(ba.alpha.len(), 4);
        assert!((ba.alpha[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_cheby1_highpass_zpk_has_minus_one_zeros() {
        let result = cheby1(
            4,
            1.0,
            &[0.3],
            IirFilterBandType::Highpass,
            Some(ButterOutputType::Zpk),
            None,
        )
        .unwrap();
        let ButterOutput::Zpk(zpk) = result else {
            panic!("expected zpk output");
        };
        assert_eq!(zpk.z.len(), 4);
        for z in &zpk.z {
            assert!((*z - Complex::new(1.0, 0.0)).norm() < 1e-9);
        }
    }

    #[test]
    fn test_cheb1ord_lowpass_meets_specs() {
        let spec = cheb1ord(
            &[0.2],
            &[0.3],
            3.0,
            40.0,
            IirFilterBandType::Lowpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby1(
            spec.order,
            3.0,
            &spec.wn,
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
            None,
        )
        .unwrap()
        else {
            panic!("expected ba output");
        };

        let pass_db = response_mag_db(&ba, 0.2, 2.0);
        let stop_db = response_mag_db(&ba, 0.3, 2.0);
        assert!(pass_db >= -3.0 - 1e-6);
        assert!(stop_db <= -40.0 + 1e-6);
    }

    #[test]
    fn test_cheb1ord_highpass_meets_specs() {
        let spec = cheb1ord(
            &[0.35],
            &[0.2],
            3.0,
            30.0,
            IirFilterBandType::Highpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby1(
            spec.order,
            3.0,
            &spec.wn,
            IirFilterBandType::Highpass,
            Some(ButterOutputType::Ba),
            None,
        )
        .unwrap()
        else {
            panic!("expected ba output");
        };

        let pass_db = response_mag_db(&ba, 0.35, 2.0);
        let stop_db = response_mag_db(&ba, 0.2, 2.0);
        assert!(pass_db >= -3.0 - 1e-6);
        assert!(stop_db <= -30.0 + 1e-6);
    }

    #[test]
    fn test_cheb1ord_bandpass_meets_specs() {
        let spec = cheb1ord(
            &[0.25, 0.45],
            &[0.18, 0.55],
            3.0,
            30.0,
            IirFilterBandType::Bandpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby1(
            spec.order,
            3.0,
            &spec.wn,
            IirFilterBandType::Bandpass,
            Some(ButterOutputType::Ba),
            None,
        )
        .unwrap()
        else {
            panic!("expected ba output");
        };

        let pass_low_db = response_mag_db(&ba, 0.25, 2.0);
        let pass_high_db = response_mag_db(&ba, 0.45, 2.0);
        let stop_low_db = response_mag_db(&ba, 0.18, 2.0);
        let stop_high_db = response_mag_db(&ba, 0.55, 2.0);
        assert!(pass_low_db >= -3.0 - 1e-6);
        assert!(pass_high_db >= -3.0 - 1e-6);
        assert!(stop_low_db <= -30.0 + 1e-6);
        assert!(stop_high_db <= -30.0 + 1e-6);
    }

    #[test]
    fn test_cheb1ord_bandstop_meets_specs() {
        let spec = cheb1ord(
            &[0.2, 0.5],
            &[0.28, 0.4],
            3.0,
            30.0,
            IirFilterBandType::Bandstop,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby1(
            spec.order,
            3.0,
            &spec.wn,
            IirFilterBandType::Bandstop,
            Some(ButterOutputType::Ba),
            None,
        )
        .unwrap()
        else {
            panic!("expected ba output");
        };

        let pass_low_db = response_mag_db(&ba, 0.2, 2.0);
        let pass_high_db = response_mag_db(&ba, 0.5, 2.0);
        let stop_low_db = response_mag_db(&ba, 0.28, 2.0);
        let stop_high_db = response_mag_db(&ba, 0.4, 2.0);
        assert!(pass_low_db >= -3.0 - 1e-6);
        assert!(pass_high_db >= -3.0 - 1e-6);
        assert!(stop_low_db <= -30.0 + 1e-6);
        assert!(stop_high_db <= -30.0 + 1e-6);
    }
}
