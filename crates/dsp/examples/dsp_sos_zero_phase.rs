//! A narrow, high-order low-pass filter as second-order sections, applied once and with zero phase.

use lib_jsl_dsp::continuous::filter::design::IirFilterBandType;
use lib_jsl_dsp::discrete::filters::butterworth::{ButterOutput, ButterOutputType, butter};
use lib_jsl_dsp::discrete::filters::sos::{sosfilt, zpk2sos};
use lib_jsl_dsp::discrete::filters::zero_phase::{Padding, sosfiltfilt};
use std::f64::consts::PI;

fn main() {
    // A 10th-order Butterworth low-pass at 0.02 × Nyquist, designed as zeros, poles and gain.
    let ButterOutput::Zpk(zpk) = butter(10, &[0.02], IirFilterBandType::Lowpass, Some(ButterOutputType::Zpk), None).unwrap() else {
        unreachable!()
    };
    let sos = zpk2sos(&zpk.z.to_vec(), &zpk.p.to_vec(), zpk.k.re).unwrap();
    println!("{} sections; the last holds the poles nearest the unit circle:", sos.len());
    for s in &sos {
        println!("  b = [{:+.3e}, {:+.3e}, {:+.3e}]  a = [1, {:+.6}, {:+.6}]", s[0], s[1], s[2], s[4], s[5]);
    }
    // A slow sine (0.004 × Nyquist) plus a fast one (0.2 × Nyquist) to be removed.
    let x: Vec<f64> = (0..3000).map(|n| (PI * 0.004 * n as f64).sin() + 0.5 * (PI * 0.2 * n as f64).sin()).collect();
    let slow: Vec<f64> = (0..3000).map(|n| (PI * 0.004 * n as f64).sin()).collect();
    let (once, _) = sosfilt(&sos, &x, None).unwrap();
    let twice = sosfiltfilt(&sos, &x, Padding::Odd, None).unwrap();
    let rms = |y: &[f64]| (y[500..2500].iter().zip(&slow[500..2500]).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 2000.0).sqrt();
    // The delay of one pass: the lag that best aligns its output with the slow sine.
    let lag = (0..200).min_by(|&a, &b| {
        let e = |k: usize| (500..2500).map(|n| (once[n] - slow[n - k]).powi(2)).sum::<f64>();
        e(a).total_cmp(&e(b))
    }).unwrap();
    println!("RMS error against the slow sine: one pass {:.3}, zero phase {:.4}", rms(&once), rms(&twice));
    println!("one pass delays the slow sine by about {lag} samples; zero phase by none");
}
