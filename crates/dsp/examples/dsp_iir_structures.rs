//! Removing mains hum with a notch filter and its harmonics with a comb filter.

use lib_jsl_dsp::discrete::filters::discrete_linear_filter::DiscreteLinearFilter;
use lib_jsl_dsp::discrete::filters::iir_comb::{IirCombFilterType, iircomb};
use lib_jsl_dsp::discrete::filters::iir_notch::iirnotch;
use std::f64::consts::PI;

fn main() {
    let fs = 1000.0;
    // A 23 Hz "signal" plus 50 Hz hum and its third harmonic.
    let x: Vec<f64> = (0..4000)
        .map(|n| {
            let t = n as f64 / fs;
            (2.0 * PI * 23.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin() + 0.2 * (2.0 * PI * 150.0 * t).sin()
        })
        .collect();
    let clean: Vec<f64> = (0..4000).map(|n| (2.0 * PI * 23.0 * n as f64 / fs).sin()).collect();
    let error = |y: &[f64]| (y[2000..].iter().zip(&clean[2000..]).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 2000.0).sqrt();
    let run = |b: Vec<f64>, a: Vec<f64>| {
        let mut f = DiscreteLinearFilter::new(&b, &a);
        x.iter().map(|v| f.step(v)).collect::<Vec<f64>>()
    };
    let notch = iirnotch(50.0, 30.0, Some(fs)).unwrap();
    let y = run(notch.beta.iter().map(|c| c.re).collect(), notch.alpha.iter().map(|c| c.re).collect());
    println!("RMS error: input {:.3}, after a 50 Hz notch {:.3}", error(&x), error(&y));
    let comb = iircomb(50.0, 30.0, Some(IirCombFilterType::Notch), Some(fs), None).unwrap();
    println!("comb filter: {} numerator taps", comb.beta.len());
    let y = run(comb.beta.iter().map(|c| c.re).collect(), comb.alpha.iter().map(|c| c.re).collect());
    println!("after a 50 Hz comb (all harmonics): {:.3}", error(&y));
}
