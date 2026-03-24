use std::f64::consts::PI;

use num::Complex;

use crate::{
    dsp::{
        continuous::filter::{
            bilinear::BilinearResult,
            bilinear_zpk::{bilinear_zpk, BilinearZpkResult},
            design::{
                complete_elliptic_k, complete_elliptic_km1, ellipap, lp2bp_zpk, lp2bs_zpk,
                lp2hp_zpk, lp2lp_zpk, zpk_to_tf, IirFilterBandType, Zpk,
            },
        },
        discrete::{
            filters::iir::butterworth::{ButterOutput, ButterOutputType},
            spectral::freqz::{freqz, FreqzWorN},
        },
    },
    prelude::ErrorsJSL,
};

#[derive(Clone, Debug, PartialEq)]
pub struct EllipordResult {
    /// Minimum order predicted to satisfy the supplied passband and stopband
    /// constraints.
    pub order: usize,
    /// Passband critical frequency or frequency pair to feed back into
    /// [`ellip`].
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

/// Build the analog elliptic design in ZPK form before the bilinear transform.
///
/// This follows the same broad design pipeline as the Butterworth and
/// Chebyshev implementations, but the normalized prototype now comes from the
/// more selective elliptic family with ripple in both passband and stopband.
fn ellip_analog_zpk(
    order: usize,
    rp: f64,
    rs: f64,
    warped: &[f64],
    filter_type: IirFilterBandType,
) -> Result<Zpk, ErrorsJSL> {
    let proto = ellipap(order, rp, rs)?;
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
    // Hold the design in ZPK form until the end, then normalize the polynomial
    // form into the standard digital convention `a[0] = 1`.
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

fn ellipord_natural_frequency(
    passb: &[f64],
    stopb: &[f64],
    filter_type: IirFilterBandType,
) -> Result<f64, ErrorsJSL> {
    // Reduce each concrete spec geometry into the single normalized
    // lowpass-prototype spacing parameter that drives the elliptic order
    // equation.
    Ok(match filter_type {
        IirFilterBandType::Lowpass => stopb[0] / passb[0],
        IirFilterBandType::Highpass => passb[0] / stopb[0],
        IirFilterBandType::Bandpass => {
            let nat1 =
                ((stopb[0] * stopb[0] - passb[0] * passb[1]) / (stopb[0] * (passb[0] - passb[1])))
                    .abs();
            let nat2 =
                ((stopb[1] * stopb[1] - passb[0] * passb[1]) / (stopb[1] * (passb[0] - passb[1])))
                    .abs();
            nat1.min(nat2)
        }
        IirFilterBandType::Bandstop => {
            let nat1 =
                (stopb[0] * (passb[0] - passb[1]) / (stopb[0] * stopb[0] - passb[0] * passb[1]))
                    .abs();
            let nat2 =
                (stopb[1] * (passb[0] - passb[1]) / (stopb[1] * stopb[1] - passb[0] * passb[1]))
                    .abs();
            nat1.min(nat2)
        }
    })
}

fn ellip_order_from_nat(nat: f64, gpass: f64, gstop: f64) -> Result<f64, ErrorsJSL> {
    if !nat.is_finite() || nat <= 1.0 {
        return Err(ErrorsJSL::RuntimeError(
            "failed to determine a valid elliptic natural frequency ratio",
        ));
    }
    // Elliptic order selection depends on a ratio of complete elliptic
    // integrals rather than the simpler `acosh` expression used by the
    // Chebyshev families.
    let arg1_sq = (10.0_f64.powf(0.1 * gpass) - 1.0) / (10.0_f64.powf(0.1 * gstop) - 1.0);
    let arg0_sq = 1.0 / (nat * nat);
    let d0 = (complete_elliptic_k(arg0_sq)?, complete_elliptic_km1(arg0_sq)?);
    let d1 = (complete_elliptic_k(arg1_sq)?, complete_elliptic_km1(arg1_sq)?);
    Ok(d0.0 * d1.1 / (d0.1 * d1.0))
}

fn golden_section_search<F>(mut a: f64, mut b: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
    let resphi = 2.0 - phi;

    let mut c = b - resphi * (b - a);
    let mut d = a + resphi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..80 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - resphi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + resphi * (b - a);
            fd = f(d);
        }
    }

    if fc < fd { c } else { d }
}

fn optimize_bandstop_passband(
    passb: &[f64],
    stopb: &[f64],
    gpass: f64,
    gstop: f64,
) -> Result<Vec<f64>, ErrorsJSL> {
    // Mirror SciPy's bandstop behavior: nudge the passband edges inward to the
    // locations that minimize the required order before the final backsolve.
    let order_for_edge = |candidate: f64, index: usize| -> f64 {
        let mut varied = passb.to_vec();
        varied[index] = candidate;
        let nat = ellipord_natural_frequency(&varied, stopb, IirFilterBandType::Bandstop)
            .unwrap_or(1.0);
        ellip_order_from_nat(nat, gpass, gstop).unwrap_or(f64::INFINITY)
    };

    let low = golden_section_search(passb[0], stopb[0] - 1e-12, |w| order_for_edge(w, 0));
    let high = golden_section_search(stopb[1] + 1e-12, passb[1], |w| order_for_edge(w, 1));
    Ok(vec![low, high])
}

/// Design a digital elliptic filter in a style similar to SciPy's `ellip`.
///
/// `wn` is the passband critical frequency. When `fs` is omitted, the digital
/// convention is normalized with `fs = 2.0`, so Nyquist lives at `1.0`.
pub fn ellip(
    order: usize,
    rp: f64,
    rs: f64,
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
    if !rs.is_finite() || rs <= rp {
        return Err(ErrorsJSL::InvalidInputRange("rs must be finite and > rp"));
    }
    let fs = fs.unwrap_or(2.0);
    validate_critical_frequencies(wn, filter_type, fs)?;

    // Prewarp the user-facing digital edges into the analog domain, do the
    // classical prototype design there, then map the result back with the
    // bilinear transform.
    let warped = wn.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let analog = ellip_analog_zpk(order, rp, rs, &warped, filter_type)?;
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

/// Choose the minimum elliptic order meeting passband/stopband specs.
///
/// The returned `wn` is a passband critical frequency or pair intended for
/// direct use with [`ellip`]. As in SciPy, elliptic order selection returns a
/// passband critical frequency rather than a stopband edge.
pub fn ellipord(
    wp: &[f64],
    ws: &[f64],
    gpass: f64,
    gstop: f64,
    filter_type: IirFilterBandType,
    fs: Option<f64>,
) -> Result<EllipordResult, ErrorsJSL> {
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

    // The closed-form order equations live in the analog domain, so digital
    // edge frequencies are prewarped before any order computation is done.
    let mut passb = wp.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let stopb = ws.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();

    if matches!(filter_type, IirFilterBandType::Bandstop) {
        passb = optimize_bandstop_passband(&passb, &stopb, gpass, gstop)?;
    }

    let nat = ellipord_natural_frequency(&passb, &stopb, filter_type)?;
    let order = ellip_order_from_nat(nat, gpass, gstop)?.ceil() as usize;

    // Convert the passband critical frequencies back into the public digital
    // convention so the result can be passed directly into `ellip`.
    Ok(EllipordResult { order, wn: passb.into_iter().map(|w| fs * (w / (2.0 * fs)).atan() / PI).collect() })
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
    fn test_ellip_lowpass_ba_smoke() {
        let result = ellip(
            4,
            1.0,
            40.0,
            &[0.2],
            IirFilterBandType::Lowpass,
            Some(ButterOutputType::Ba),
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
    fn test_ellip_highpass_zpk_has_unit_circle_zeros() {
        let result = ellip(
            5,
            1.0,
            40.0,
            &[0.3],
            IirFilterBandType::Highpass,
            Some(ButterOutputType::Zpk),
            None,
        )
        .unwrap();
        let ButterOutput::Zpk(zpk) = result else {
            panic!("expected zpk output");
        };
        assert_eq!(zpk.z.len(), 5);
        assert!(zpk.z.iter().all(|z| (z.norm() - 1.0).abs() < 1e-9));
    }

    #[test]
    fn test_ellipord_lowpass_meets_specs() {
        let spec = ellipord(
            &[0.2],
            &[0.3],
            3.0,
            40.0,
            IirFilterBandType::Lowpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = ellip(
            spec.order,
            3.0,
            40.0,
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
        assert!(pass_db >= -3.0 - 1e-5);
        assert!(stop_db <= -40.0 + 1e-5);
    }

    #[test]
    fn test_ellipord_highpass_meets_specs() {
        let spec = ellipord(
            &[0.35],
            &[0.2],
            3.0,
            30.0,
            IirFilterBandType::Highpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = ellip(
            spec.order,
            3.0,
            30.0,
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
        assert!(pass_db >= -3.0 - 1e-5);
        assert!(stop_db <= -30.0 + 1e-5);
    }

    #[test]
    fn test_ellipord_bandpass_meets_specs() {
        let spec = ellipord(
            &[0.25, 0.45],
            &[0.18, 0.55],
            3.0,
            30.0,
            IirFilterBandType::Bandpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = ellip(
            spec.order,
            3.0,
            30.0,
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
        assert!(pass_low_db >= -3.0 - 1e-5);
        assert!(pass_high_db >= -3.0 - 1e-5);
        assert!(stop_low_db <= -30.0 + 1e-5);
        assert!(stop_high_db <= -30.0 + 1e-5);
    }

    #[test]
    fn test_ellipord_bandstop_meets_specs() {
        let spec = ellipord(
            &[0.2, 0.5],
            &[0.28, 0.4],
            3.0,
            30.0,
            IirFilterBandType::Bandstop,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = ellip(
            spec.order,
            3.0,
            30.0,
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
        assert!(pass_low_db >= -3.0 - 1e-5);
        assert!(pass_high_db >= -3.0 - 1e-5);
        assert!(stop_low_db <= -30.0 + 1e-5);
        assert!(stop_high_db <= -30.0 + 1e-5);
    }
}
