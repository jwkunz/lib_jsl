use std::f64::consts::PI;

use num::Complex;

use crate::{
    dsp::{
        continuous::filter::{
            bilinear::BilinearResult,
            bilinear_zpk::{bilinear_zpk, BilinearZpkResult},
            design::{
                buttap, lp2bp_zpk, lp2bs_zpk, lp2hp_zpk, lp2lp_zpk, zpk_to_tf, IirFilterBandType,
                Zpk,
            },
        },
        discrete::spectral::freqz::{freqz, FreqzWorN},
    },
    prelude::ErrorsJSL,
};

/// Output container for `butter`.
///
/// This mirrors the common SciPy-style choice between polynomial (`ba`) output
/// and zero-pole-gain (`zpk`) output.
#[derive(Clone, Debug, PartialEq)]
pub enum ButterOutput {
    Ba(BilinearResult),
    Zpk(BilinearZpkResult),
}

/// Requested output representation for `butter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButterOutputType {
    Ba,
    Zpk,
}

/// Result of the Butterworth order-selection routine `buttord`.
///
/// `order` is the minimum filter order predicted to meet the supplied passband
/// and stopband constraints, and `wn` is the Butterworth critical frequency (or
/// pair of frequencies) that should be passed into `butter`.
#[derive(Clone, Debug, PartialEq)]
pub struct ButtordResult {
    pub order: usize,
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

fn unwarp_frequency(w: f64, fs: f64) -> f64 {
    fs * (w / (2.0 * fs)).atan() / PI
}

/// Build the analog Butterworth design in ZPK form before the bilinear step.
///
/// The discrete `butter` flow in this crate is:
/// 1. Prewarp digital edge frequencies into analog frequencies.
/// 2. Start from the normalized Butterworth prototype from `buttap`.
/// 3. Apply the requested analog lowpass/highpass/bandpass/bandstop mapping.
/// 4. Convert to digital with the bilinear transform.
fn butter_analog_zpk(
    order: usize,
    warped: &[f64],
    filter_type: IirFilterBandType,
) -> Result<Zpk, ErrorsJSL> {
    let proto = buttap(order)?;
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

/// Convert bilinear-transformed digital ZPK data back into normalized `ba`
/// form.
///
/// This keeps the design math in ZPK form until the end, which is easier for
/// analog prototype transforms, then returns coefficients with `a[0] = 1`.
fn digital_zpk_to_ba(zpk: &BilinearZpkResult) -> Result<BilinearResult, ErrorsJSL> {
    let tf = zpk_to_tf(&Zpk {
        z: zpk.z.clone(),
        p: zpk.p.clone(),
        k: zpk.k,
    })?;
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

/// Design a digital Butterworth filter.
///
/// This follows the standard prototype-design approach used by SciPy:
/// normalized analog Butterworth prototype -> analog frequency transform ->
/// bilinear transform -> requested digital output form.
pub fn butter(
    order: usize,
    wn: &[f64],
    filter_type: IirFilterBandType,
    output: Option<ButterOutputType>,
    fs: Option<f64>,
) -> Result<ButterOutput, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    let fs = fs.unwrap_or(2.0);
    validate_critical_frequencies(wn, filter_type, fs)?;

    let warped = wn.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let analog = butter_analog_zpk(order, &warped, filter_type)?;
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

/// Compute the Butterworth "natural frequency ratio" used in the closed-form
/// order calculation.
///
/// After prewarping and reducing the problem to a normalized lowpass prototype,
/// Butterworth order selection depends only on how far the stopband lies from
/// the passband in prototype coordinates.
fn buttord_natural_frequency(
    wp: &[f64],
    ws: &[f64],
    filter_type: IirFilterBandType,
) -> Result<f64, ErrorsJSL> {
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

/// Recover the Butterworth critical frequency from a chosen order and the
/// allowed passband ripple.
///
/// `buttord` first finds the minimum order, then backs out the critical
/// frequency that places the passband edge exactly on the requested loss
/// contour.
fn buttord_critical_wn(
    order: usize,
    wp: &[f64],
    filter_type: IirFilterBandType,
    gpass: f64,
) -> Result<Vec<f64>, ErrorsJSL> {
    let pass_factor = (10.0_f64.powf(0.1 * gpass) - 1.0).powf(1.0 / (2.0 * order as f64));

    Ok(match filter_type {
        IirFilterBandType::Lowpass => vec![wp[0] / pass_factor],
        IirFilterBandType::Highpass => vec![wp[0] * pass_factor],
        IirFilterBandType::Bandpass => {
            let wo = (wp[0] * wp[1]).sqrt();
            let bw = wp[1] - wp[0];
            let alpha = bw / pass_factor;
            let disc = (alpha * alpha + 4.0 * wo * wo).sqrt();
            vec![(-alpha + disc) * 0.5, (alpha + disc) * 0.5]
        }
        IirFilterBandType::Bandstop => {
            let wo = (wp[0] * wp[1]).sqrt();
            let bw = wp[1] - wp[0];
            let alpha = pass_factor * bw;
            let disc = (alpha * alpha + 4.0 * wo * wo).sqrt();
            vec![(-alpha + disc) * 0.5, (alpha + disc) * 0.5]
        }
    })
}

/// Choose the minimum Butterworth order that meets passband/stopband specs.
///
/// Inputs use the same normalized-digital convention as `butter` when `fs` is
/// omitted: frequencies are measured on `[0, fs/2]` with `fs = 2.0`.
///
/// Internally the routine prewarps all digital edge frequencies to the analog
/// domain, applies Butterworth closed-form order equations there, and then
/// maps the resulting critical frequency back to the digital domain.
pub fn buttord(
    wp: &[f64],
    ws: &[f64],
    gpass: f64,
    gstop: f64,
    filter_type: IirFilterBandType,
    fs: Option<f64>,
) -> Result<ButtordResult, ErrorsJSL> {
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
                return Err(ErrorsJSL::InvalidInputRange(
                    "lowpass requires wp < ws",
                ));
            }
        }
        IirFilterBandType::Highpass => {
            if wp[0] <= ws[0] {
                return Err(ErrorsJSL::InvalidInputRange(
                    "highpass requires wp > ws",
                ));
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

    let wp_warped = wp.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let ws_warped = ws.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();

    let nat = buttord_natural_frequency(&wp_warped, &ws_warped, filter_type)?;
    if !nat.is_finite() || nat <= 1.0 {
        return Err(ErrorsJSL::RuntimeError(
            "failed to determine a valid Butterworth natural frequency ratio",
        ));
    }

    let gpass_lin = 10.0_f64.powf(0.1 * gpass) - 1.0;
    let gstop_lin = 10.0_f64.powf(0.1 * gstop) - 1.0;
    let order = ((gstop_lin / gpass_lin).log10() / (2.0 * nat.log10())).ceil() as usize;
    let wn_analog = buttord_critical_wn(order, &wp_warped, filter_type, gpass)?;
    let wn_digital = wn_analog
        .into_iter()
        .map(|w| unwarp_frequency(w, fs))
        .collect::<Vec<_>>();

    Ok(ButtordResult {
        order,
        wn: wn_digital,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_butter_lowpass_ba_smoke() {
        let result = butter(
            2,
            &[0.2],
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = result else {
            panic!("expected ba output");
        };
        assert_eq!(ba.beta.len(), 3);
        assert_eq!(ba.alpha.len(), 3);
        assert!((ba.alpha[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_butter_highpass_zpk_has_minus_one_zeros() {
        let result = butter(
            3,
            &[0.3],
            IirFilterBandType::Highpass,
            Some(ButterOutputType::Zpk),
            None,
        )
        .unwrap();
        let ButterOutput::Zpk(zpk) = result else {
            panic!("expected zpk output");
        };
        assert_eq!(zpk.z.len(), 3);
        for z in &zpk.z {
            assert!((*z - Complex::new(1.0, 0.0)).norm() < 1e-9);
        }
    }

    #[test]
    fn test_buttord_lowpass_meets_specs() {
        let spec = buttord(
            &[0.2],
            &[0.3],
            3.0,
            40.0,
            IirFilterBandType::Lowpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = butter(
            spec.order,
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
    fn test_buttord_highpass_meets_specs() {
        let spec = buttord(
            &[0.35],
            &[0.2],
            3.0,
            30.0,
            IirFilterBandType::Highpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = butter(
            spec.order,
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
    fn test_buttord_bandpass_meets_specs() {
        let spec = buttord(
            &[0.25, 0.45],
            &[0.18, 0.55],
            3.0,
            30.0,
            IirFilterBandType::Bandpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = butter(
            spec.order,
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
    fn test_buttord_bandstop_meets_specs() {
        let spec = buttord(
            &[0.2, 0.5],
            &[0.28, 0.4],
            3.0,
            30.0,
            IirFilterBandType::Bandstop,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = butter(
            spec.order,
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
