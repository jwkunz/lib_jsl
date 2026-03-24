use std::f64::consts::PI;

use crate::{
    dsp::{
        continuous::filter::{
            bilinear::BilinearResult,
            bilinear_zpk::{bilinear_zpk, BilinearZpkResult},
            design::{
                cheb2ap, lp2bp_zpk, lp2bs_zpk, lp2hp_zpk, lp2lp_zpk, zpk_to_tf,
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
pub struct Cheb2ordResult {
    /// Minimum order predicted to satisfy the supplied passband and stopband
    /// constraints.
    pub order: usize,
    /// Critical frequency or frequency pair to feed back into [`cheby2`].
    ///
    /// For type II designs this is the stopband critical frequency after the
    /// analog-domain backsolve.
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

/// Build the analog Chebyshev type II design in ZPK form before the bilinear
/// transform.
///
/// Type II differs from type I primarily in the normalized prototype:
/// attenuation is specified in the stopband, and the prototype contains finite
/// transmission zeros. The rest of the digital flow still follows the same
/// prototype-transform-bilinear structure.
fn cheby2_analog_zpk(
    order: usize,
    rs: f64,
    warped: &[f64],
    filter_type: IirFilterBandType,
) -> Result<Zpk, ErrorsJSL> {
    let proto = cheb2ap(order, rs)?;
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
    // As with the other IIR designers, hold the numerically nicer ZPK form as
    // long as possible and only normalize into polynomial coefficients at the
    // API boundary.
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

fn cheb2ord_natural_frequency(
    passb: &[f64],
    stopb: &[f64],
    filter_type: IirFilterBandType,
) -> Result<f64, ErrorsJSL> {
    // Convert the concrete filter specification into the normalized prototype
    // spacing used by the inverse-Chebyshev order equations.
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

fn cheby_order_from_nat(nat: f64, gpass: f64, gstop: f64) -> f64 {
    let gstop_lin = 10.0_f64.powf(0.1 * gstop);
    let gpass_lin = 10.0_f64.powf(0.1 * gpass);
    ((gstop_lin - 1.0) / (gpass_lin - 1.0)).sqrt().acosh() / nat.acosh()
}

// Small standalone 1-D minimizer used to emulate SciPy's passband-edge search
// for bandstop order selection without pulling in an optimization dependency.
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
    // For bandstop order selection, SciPy adjusts the passband edges before the
    // final order calculation so the worst-case prototype ratio is minimized.
    // That keeps the returned order closer to the true minimum than simply
    // freezing the user-supplied passband edges.
    let order_for_edge = |candidate: f64, index: usize| -> f64 {
        let mut varied = passb.to_vec();
        varied[index] = candidate;
        let nat = cheb2ord_natural_frequency(&varied, stopb, IirFilterBandType::Bandstop)
            .unwrap_or(1.0);
        if !nat.is_finite() || nat <= 1.0 {
            f64::INFINITY
        } else {
            cheby_order_from_nat(nat, gpass, gstop)
        }
    };

    let low = golden_section_search(passb[0], stopb[0] - 1e-12, |w| order_for_edge(w, 0));
    let high = golden_section_search(stopb[1] + 1e-12, passb[1], |w| order_for_edge(w, 1));
    Ok(vec![low, high])
}

/// Design a digital Chebyshev type II filter in a style similar to SciPy's
/// `cheby2`.
///
/// `wn` is interpreted as the stopband critical frequency because that is how
/// inverse-Chebyshev filters are parameterized. When `fs` is omitted, the
/// digital convention is normalized with `fs = 2.0`, so Nyquist lives at `1.0`.
pub fn cheby2(
    order: usize,
    rs: f64,
    wn: &[f64],
    filter_type: IirFilterBandType,
    output: Option<ButterOutputType>,
    fs: Option<f64>,
) -> Result<ButterOutput, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    if !rs.is_finite() || rs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("rs must be finite and > 0"));
    }
    let fs = fs.unwrap_or(2.0);
    validate_critical_frequencies(wn, filter_type, fs)?;

    // Prewarp into the analog domain, design there, and then rely on the
    // bilinear transform to return the final digital filter.
    let warped = wn.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let analog = cheby2_analog_zpk(order, rs, &warped, filter_type)?;
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

/// Choose the minimum Chebyshev type II order meeting passband/stopband specs.
///
/// The returned `wn` is the stopband critical frequency (or frequency pair)
/// intended for direct use with [`cheby2`]. This differs from `cheb1ord`,
/// which returns a passband critical frequency.
pub fn cheb2ord(
    wp: &[f64],
    ws: &[f64],
    gpass: f64,
    gstop: f64,
    filter_type: IirFilterBandType,
    fs: Option<f64>,
) -> Result<Cheb2ordResult, ErrorsJSL> {
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

    // As with the other order-selection helpers, all of the closed-form work is
    // done on prewarped analog frequencies.
    let mut passb = wp.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();
    let stopb = ws.iter().map(|&w| prewarp_frequency(w, fs)).collect::<Vec<_>>();

    if matches!(filter_type, IirFilterBandType::Bandstop) {
        passb = optimize_bandstop_passband(&passb, &stopb, gpass, gstop)?;
    }

    let nat = cheb2ord_natural_frequency(&passb, &stopb, filter_type)?;
    if !nat.is_finite() || nat <= 1.0 {
        return Err(ErrorsJSL::RuntimeError(
            "failed to determine a valid Chebyshev natural frequency ratio",
        ));
    }

    let order = cheby_order_from_nat(nat, gpass, gstop).ceil() as usize;
    // Once the order is fixed, backsolve the analog prototype frequency that
    // lands exactly on the requested passband-loss contour.
    let v_pass_stop = ((10.0_f64.powf(0.1 * gstop) - 1.0) / (10.0_f64.powf(0.1 * gpass) - 1.0))
        .sqrt()
        .acosh();
    let new_freq = 1.0 / (v_pass_stop / order as f64).cosh();

    let wn_analog = match filter_type {
        IirFilterBandType::Lowpass => vec![passb[0] / new_freq],
        IirFilterBandType::Highpass => vec![passb[0] * new_freq],
        IirFilterBandType::Bandpass => {
            let disc = (((passb[1] - passb[0]).powi(2) / (4.0 * new_freq * new_freq))
                + passb[0] * passb[1])
                .sqrt();
            let w0 = (passb[0] - passb[1]) / (2.0 * new_freq) + disc;
            let w1 = passb[0] * passb[1] / w0;
            vec![w0.abs().min(w1.abs()), w0.abs().max(w1.abs())]
        }
        IirFilterBandType::Bandstop => {
            let disc = (new_freq * new_freq * (passb[1] - passb[0]).powi(2) * 0.25
                + passb[0] * passb[1])
                .sqrt();
            let w0 = new_freq * (passb[0] - passb[1]) * 0.5 + disc;
            let w1 = passb[0] * passb[1] / w0;
            vec![w0.abs().min(w1.abs()), w0.abs().max(w1.abs())]
        }
    };

    let wn = wn_analog
        .into_iter()
        .map(|w| unwarp_frequency(w.abs(), fs))
        .collect::<Vec<_>>();

    Ok(Cheb2ordResult { order, wn })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::discrete::spectral::freqz::{freqz, FreqzWorN};
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
    fn test_cheby2_lowpass_ba_smoke() {
        let result = cheby2(
            4,
            40.0,
            &[0.3],
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
    fn test_cheby2_highpass_zpk_has_unit_circle_zeros() {
        let result = cheby2(
            4,
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
        assert_eq!(zpk.z.len(), 4);
        assert!(zpk.z.iter().all(|z| (z.norm() - 1.0).abs() < 1e-9));
    }

    #[test]
    fn test_cheb2ord_lowpass_meets_specs() {
        let spec = cheb2ord(
            &[0.2],
            &[0.3],
            3.0,
            40.0,
            IirFilterBandType::Lowpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby2(
            spec.order,
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
        assert!(pass_db >= -3.0 - 1e-6);
        assert!(stop_db <= -40.0 + 1e-6);
    }

    #[test]
    fn test_cheb2ord_highpass_meets_specs() {
        let spec = cheb2ord(
            &[0.35],
            &[0.2],
            3.0,
            30.0,
            IirFilterBandType::Highpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby2(
            spec.order,
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
        assert!(pass_db >= -3.0 - 1e-6);
        assert!(stop_db <= -30.0 + 1e-6);
    }

    #[test]
    fn test_cheb2ord_bandpass_meets_specs() {
        let spec = cheb2ord(
            &[0.25, 0.45],
            &[0.18, 0.55],
            3.0,
            30.0,
            IirFilterBandType::Bandpass,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby2(
            spec.order,
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
        assert!(pass_low_db >= -3.0 - 1e-6);
        assert!(pass_high_db >= -3.0 - 1e-6);
        assert!(stop_low_db <= -30.0 + 1e-6);
        assert!(stop_high_db <= -30.0 + 1e-6);
    }

    #[test]
    fn test_cheb2ord_bandstop_meets_specs() {
        let spec = cheb2ord(
            &[0.2, 0.5],
            &[0.28, 0.4],
            3.0,
            30.0,
            IirFilterBandType::Bandstop,
            None,
        )
        .unwrap();
        let ButterOutput::Ba(ba) = cheby2(
            spec.order,
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
        assert!(pass_low_db >= -3.0 - 1e-6);
        assert!(pass_high_db >= -3.0 - 1e-6);
        assert!(stop_low_db <= -30.0 + 1e-6);
        assert!(stop_high_db <= -30.0 + 1e-6);
    }
}
